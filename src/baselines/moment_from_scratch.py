"""
MOMENT-from-scratch baseline trained on ZTF light curves.

Why this exists
---------------
We have a pretrained-MOMENT baseline (110M params, 10^9 cross-domain
timesteps) at LogR=0.524.  The architectural-fairness comparison needs a
parameter-matched MOMENT trained on our 400K ZTF pool: does MOMENT collapse
the way Chronos-T5 does (LogR 0.65 → 0.32) or hold up?

Architecture / objective
------------------------
* Underlying model: ``momentfm.MOMENT`` with ``task_name="reconstruction"``.
  This is the masked-patch reconstruction objective the released MOMENT
  checkpoints were pretrained on.
* T5EncoderModel backbone with ``randomly_initialize_backbone=True`` so the
  encoder weights start from scratch (no Hugging Face initialization leak).
* d_model=192, num_layers=6, num_heads=4, d_ff=768 → ~5M params total
  (encoder + RevIN + PatchEmbedding + PretrainHead). Tuned at first
  smoke-test print to land near 5M; tweak via CLI if off.
* Univariate flux only (n_channels=1), seq_len=384 to match other
  baselines, patch_len=8 (MOMENT's published default).

Loss
----
MSE between reconstruction output and input flux, restricted to positions
that are (a) marked as MASKED by ``pretrain_mask`` (i.e. the model didn't
see them on the encoder side) and (b) NOT padding (input_mask=True).
Standard masked-recon target.

Data pipeline
-------------
Reuses ``ZTFMatchfileDataset(contrastive=False)`` so preprocessing matches
the rest of the benchmark exactly --- multiband, MAD-normalize, random
windowing.  Custom collate to produce (B, 1, L) flux tensors and (B, L)
input masks.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from momentfm import MOMENT
from src.data.matchfile_dataset import ZTFMatchfileDataset
from src.model.utils import (
    count_parameters, EarlyStopping, WarmupCosineSchedule,
)


# ---------------------------------------------------------------------------
# Dataset adapter: yield (flux, input_mask) per source
# ---------------------------------------------------------------------------

class MomentFluxDataset(ZTFMatchfileDataset):
    """Yields {'flux': (L,), 'input_mask': (L,)} per source.

    Uses parent's _read_source for multiband+MAD normalization; overrides
    _make_sample to produce MOMENT's expected shape rather than the
    contrastive paired views.
    """

    def __init__(self, *, seq_len: int, **kwargs):
        super().__init__(
            max_seq_len=seq_len,
            contrastive=False,
            augment=False,
            **kwargs,
        )
        self.seq_len = seq_len

    def _make_sample(self, lc: dict) -> Optional[dict]:
        flux = lc["flux"].astype(np.float32)
        n = len(flux)
        L = self.seq_len

        if n >= L:
            start = np.random.randint(0, n - L + 1)
            flux_w = flux[start:start + L]
            mask = np.ones(L, dtype=np.float32)
        else:
            flux_w = np.zeros(L, dtype=np.float32)
            mask = np.zeros(L, dtype=np.float32)
            flux_w[:n] = flux
            mask[:n] = 1.0

        return {
            "flux": torch.from_numpy(flux_w),
            "input_mask": torch.from_numpy(mask),
        }


def _flux_collate(batch):
    flux = torch.stack([b["flux"] for b in batch], dim=0)        # (B, L)
    mask = torch.stack([b["input_mask"] for b in batch], dim=0)  # (B, L)
    return flux, mask


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_moment(args) -> MOMENT:
    """Construct a from-scratch MOMENT model with a custom T5 backbone."""
    config = SimpleNamespace(
        task_name="reconstruction",
        seq_len=args.seq_len,
        patch_len=args.patch_len,
        patch_stride_len=args.patch_len,
        d_model=args.d_model,
        transformer_backbone="google/flan-t5-small",  # ignored when randomly_initialize_backbone=True
        transformer_type="encoder_only",
        randomly_initialize_backbone=True,
        n_channels=1,
        # Train everything (defaults freeze the backbone for fine-tuning use)
        freeze_embedder=False,
        freeze_encoder=False,
        freeze_head=False,
        # T5 architecture knobs
        t5_config={
            "d_model": args.d_model,
            "d_kv": args.d_model // args.num_heads,
            "d_ff": args.d_ff,
            "num_layers": args.num_layers,
            "num_decoder_layers": args.num_layers,
            "num_heads": args.num_heads,
            "relative_attention_num_buckets": 32,
            "relative_attention_max_distance": 128,
            "dropout_rate": args.dropout,
            "feed_forward_proj": "relu",
            "is_encoder_decoder": False,
            "use_cache": False,
            # vocab_size: T5 normally uses 32128, but we feed `inputs_embeds`
            # directly and never tokenize text, so the vocab embedding is dead
            # weight. Setting it to 2 (pad+eos placeholders) drops ~6M
            # unused parameters and lets us match Chronos-from-scratch's
            # ~5M-param target exactly.
            "vocab_size": 2,
            "pad_token_id": 0,
            "eos_token_id": 1,
            "decoder_start_token_id": 0,
        },
        mask_ratio=args.mask_ratio,
        revin_affine=False,
        head_dropout=args.dropout,
        enable_gradient_checkpointing=False,
    )
    return MOMENT(config)


# ---------------------------------------------------------------------------
# DDP setup
# ---------------------------------------------------------------------------

def setup_ddp():
    if "RANK" not in os.environ:
        return 0, 1, 0
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------

def masked_recon_loss(reconstruction, target, pretrain_mask, input_mask):
    """MSE on positions where pretrain_mask=0 (masked) AND input_mask=1 (real).

    reconstruction: (B, 1, L), target: (B, 1, L), pretrain_mask: (B, L), input_mask: (B, L)
    """
    masked = (1 - pretrain_mask) * input_mask  # (B, L), 1 where loss should be computed
    masked = masked.unsqueeze(1)  # (B, 1, L)
    diff = (reconstruction - target) ** 2
    n = masked.sum().clamp(min=1.0)
    return (diff * masked).sum() / n


def train_epoch(model, loader, optimizer, scheduler, device, rank, amp_dtype, use_amp):
    model.train()
    total, n_batches = 0.0, 0
    pbar = tqdm(loader, desc="Train", disable=(rank != 0))
    for flux, input_mask in pbar:
        flux = flux.to(device, non_blocking=True)              # (B, L)
        input_mask = input_mask.to(device, non_blocking=True)  # (B, L)
        x = flux.unsqueeze(1)                                   # (B, 1, L)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            inner = model.module if isinstance(model, DDP) else model
            out = inner.reconstruction(x_enc=x, input_mask=input_mask)
            loss = masked_recon_loss(
                out.reconstruction, x, out.pretrain_mask, input_mask
            )

        if not torch.isfinite(loss):
            if rank == 0:
                pbar.set_postfix({"loss": "NaN-skip"})
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total += loss.item()
        n_batches += 1
        if rank == 0:
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, device, rank):
    model.eval()
    total, n = 0.0, 0
    for flux, input_mask in tqdm(loader, desc="Val", disable=(rank != 0)):
        flux = flux.to(device, non_blocking=True)
        input_mask = input_mask.to(device, non_blocking=True)
        x = flux.unsqueeze(1)
        inner = model.module if isinstance(model, DDP) else model
        out = inner.reconstruction(x_enc=x, input_mask=input_mask)
        loss = masked_recon_loss(
            out.reconstruction, x, out.pretrain_mask, input_mask
        )
        if torch.isfinite(loss):
            total += loss.item()
            n += 1
    avg = torch.tensor(total / max(n, 1), device=device)
    if dist.is_initialized():
        dist.all_reduce(avg, op=dist.ReduceOp.AVG)
    return avg.item()


# ---------------------------------------------------------------------------
# Pretrain entrypoint
# ---------------------------------------------------------------------------

def pretrain(args):
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print(f"DDP={world_size>1} · world={world_size} · device={device} · "
              f"precision={args.precision}")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    train_ds = MomentFluxDataset(
        matchfile_dir=args.matchfile_dir,
        seq_len=args.seq_len,
        max_sources=args.max_sources,
        shuffle=True,
        multiband=args.multiband,
        holdout_path=args.holdout_path,
    )
    val_ds = MomentFluxDataset(
        matchfile_dir=args.matchfile_dir,
        seq_len=args.seq_len,
        max_sources=max(args.max_sources // 5, 1000) if args.max_sources else None,
        shuffle=False,
        multiband=args.multiband,
        holdout_path=args.holdout_path,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=True, drop_last=True, collate_fn=_flux_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        num_workers=max(args.num_workers // 2, 1),
        pin_memory=True, drop_last=True, collate_fn=_flux_collate,
    )

    model = build_moment(args).to(device)
    if rank == 0:
        n_params = count_parameters(model)
        print(f"MOMENT parameters: {n_params:,}  "
              f"(d_model={args.d_model}, layers={args.num_layers}, "
              f"d_ff={args.d_ff}, patch_len={args.patch_len})")

    if dist.is_initialized():
        dist.barrier()
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    estimated_batches = (args.max_sources or 400_000) // (args.batch_size * world_size)
    total_steps = estimated_batches * args.epochs
    warmup_steps = estimated_batches * args.warmup_epochs
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps, total_steps)

    use_amp = args.precision in ("fp16", "bf16")
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.precision]

    early = EarlyStopping(patience=args.patience, mode="min")
    best_val = float("inf")
    history = {"train": [], "val": []}

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            device, rank, amp_dtype, use_amp,
        )
        val_loss = validate(model, val_loader, device, rank)
        elapsed = time.time() - t0

        if rank == 0:
            print(f"Epoch {epoch:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | {elapsed:.0f}s",
                  flush=True)
            history["train"].append({"epoch": epoch, "loss": train_loss})
            history["val"].append({"epoch": epoch, "loss": val_loss})
            with open(Path(args.output_dir) / "training_log.json", "w") as f:
                json.dump(history, f, indent=2)
            if val_loss < best_val:
                best_val = val_loss
                inner = model.module if isinstance(model, DDP) else model
                torch.save({
                    "model_state_dict": inner.state_dict(),
                    "moment_config": vars(inner.config),
                    "epoch": epoch,
                    "val_loss": val_loss,
                }, str(Path(args.output_dir) / "best_model.pt"))

        if early(val_loss):
            if rank == 0:
                print(f"Early stopping at epoch {epoch}")
            break

    cleanup_ddp()
    if rank == 0:
        print(f"Training complete. Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="MOMENT-from-scratch baseline")
    sub = p.add_subparsers(dest="command", required=True)

    pt = sub.add_parser("pretrain")
    pt.add_argument("--matchfile_dir", type=str, required=True)
    pt.add_argument("--output_dir", type=str, default="checkpoints/moment_5M")
    pt.add_argument("--max_sources", type=int, default=400_000)
    pt.add_argument("--multiband", action="store_true")
    pt.add_argument("--holdout_path", type=str, default=None)

    # Architecture (~5M default target)
    pt.add_argument("--d_model", type=int, default=192)
    pt.add_argument("--num_layers", type=int, default=6)
    pt.add_argument("--num_heads", type=int, default=4)
    pt.add_argument("--d_ff", type=int, default=768)
    pt.add_argument("--dropout", type=float, default=0.1)
    pt.add_argument("--seq_len", type=int, default=384)
    pt.add_argument("--patch_len", type=int, default=8)
    pt.add_argument("--mask_ratio", type=float, default=0.3)

    # Optim
    pt.add_argument("--batch_size", type=int, default=128)
    pt.add_argument("--epochs", type=int, default=40)
    pt.add_argument("--lr", type=float, default=3e-4)
    pt.add_argument("--weight_decay", type=float, default=1e-4)
    pt.add_argument("--warmup_epochs", type=int, default=2)
    pt.add_argument("--patience", type=int, default=10)
    pt.add_argument("--precision", type=str, default="bf16",
                    choices=["fp32", "fp16", "bf16"])

    pt.add_argument("--num_workers", type=int, default=6)
    pt.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    if args.command == "pretrain":
        pretrain(args)
