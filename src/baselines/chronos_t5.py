"""
Chronos-T5 baseline trained from scratch on ZTF light curves.

Why this exists
---------------
The pretrained Chronos comparison in our paper is unfair: Amazon trained it on
~10^9 timesteps spanning many domains, while our model saw 400K ZTF sources.
This baseline trains a Chronos-shaped T5 from scratch on the same ZTF pool, so
the headline comparison becomes architectural rather than a pretraining-budget
artifact.

What's "Chronos-shaped"
-----------------------
* MeanScaleUniformBins tokenizer (4096 bins, ±15σ) — straight from the
  ``chronos`` package, identical to the released checkpoints.
* T5 encoder-decoder. Univariate flux only: time and flux_err are dropped, so
  the model has no continuous-time inductive bias by construction. That's the
  point — it isolates "does our B(Δt) bias help" from "is our pretraining
  objective stronger."
* ~5M params (param-matched to our ContinuousTimeLightCurveTransformer at
  4.8M). Tunable via --d_model / --num_layers / --d_ff.

Data pipeline
-------------
Reuses ZTFMatchfileDataset(contrastive=False) — same multiband + MAD-normalize
preprocessing as v5. We just override _make_sample to pull a random
context_length+prediction_length window (instead of the full max_seq_len pad)
and tokenize on the fly. NaN left-padding for short LCs; the tokenizer treats
NaN as missing.

Loss
----
Standard T5 seq2seq cross-entropy. Decoder label tokens come from the same
tokenizer applied to the prediction-window slice with the SAME per-source scale
(via label_input_transform). Padded label positions are set to -100 to be
ignored by CE.
"""

import sys
import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from chronos import ChronosConfig
from transformers import T5Config, T5ForConditionalGeneration

from src.data.matchfile_dataset import ZTFMatchfileDataset
from src.model.utils import (
    count_parameters,
    EarlyStopping,
    WarmupCosineSchedule,
)


# ---------------------------------------------------------------------------
# Dataset adapter: ZTF matchfiles -> raw flux windows for Chronos tokenizer
# ---------------------------------------------------------------------------

class ChronosFluxDataset(ZTFMatchfileDataset):
    """Yields a single (context_length + prediction_length,) flux tensor per
    source.

    Tokenization happens in the training loop (after batching) because the
    tokenizer needs the per-source scale to be shared between context and
    label, which is cleanest to do on the GPU after stacking.
    """

    def __init__(
        self,
        matchfile_dir: str,
        context_length: int,
        prediction_length: int,
        max_sources: Optional[int] = None,
        shuffle: bool = True,
        multiband: bool = True,
        min_epochs: int = 10,
        holdout_path: Optional[str] = None,
    ):
        super().__init__(
            matchfile_dir=matchfile_dir,
            max_seq_len=context_length + prediction_length,
            contrastive=False,
            augment=False,
            multiband=multiband,
            max_sources=max_sources,
            shuffle=shuffle,
            min_epochs=min_epochs,
            holdout_path=holdout_path,
        )
        self.context_length = context_length
        self.prediction_length = prediction_length

    def _make_sample(self, lc: dict) -> Optional[dict]:
        flux = lc["flux"].astype(np.float32)
        n = len(flux)
        total = self.context_length + self.prediction_length

        if n < total:
            # Left-pad with NaN — MeanScaleUniformBins treats NaN as missing
            # and excludes those positions from the scale computation.
            pad_len = total - n
            flux = np.concatenate(
                [np.full(pad_len, np.nan, dtype=np.float32), flux]
            )
        elif n > total:
            # Random window from longer sequences (data augmentation)
            start = np.random.randint(0, n - total + 1)
            flux = flux[start:start + total]

        return {"flux": torch.from_numpy(flux)}


def _flux_collate(batch):
    """Stack {flux: (T,)} dicts into a (B, T) tensor."""
    return torch.stack([b["flux"] for b in batch], dim=0)


# ---------------------------------------------------------------------------
# Chronos config + model construction
# ---------------------------------------------------------------------------

def build_chronos_config(context_length: int, prediction_length: int) -> ChronosConfig:
    """Match the released chronos-* tokenizer settings (vocab=4096, ±15σ)."""
    return ChronosConfig(
        tokenizer_class="MeanScaleUniformBins",
        tokenizer_kwargs={"low_limit": -15.0, "high_limit": 15.0},
        context_length=context_length,
        prediction_length=prediction_length,
        n_tokens=4096,
        n_special_tokens=2,
        pad_token_id=0,
        eos_token_id=1,
        use_eos_token=True,
        model_type="seq2seq",
        # Inference-only knobs; unused during training but required by dataclass.
        num_samples=20,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
    )


def build_t5_model(args, chronos_config: ChronosConfig) -> T5ForConditionalGeneration:
    """T5 sized to roughly --target_params (default 5M).

    We don't auto-search the hyperparameters; user picks d_model / layers / d_ff
    via CLI and we print the resulting count.
    """
    t5_config = T5Config(
        vocab_size=chronos_config.n_tokens,
        d_model=args.d_model,
        d_kv=args.d_model // args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        num_decoder_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout_rate=args.dropout,
        feed_forward_proj="relu",
        is_encoder_decoder=True,
        use_cache=False,
        tie_word_embeddings=True,
        pad_token_id=chronos_config.pad_token_id,
        eos_token_id=chronos_config.eos_token_id,
        decoder_start_token_id=chronos_config.pad_token_id,
    )
    return T5ForConditionalGeneration(t5_config)


# ---------------------------------------------------------------------------
# DDP plumbing
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
# Tokenize a batch of (B, T) flux into encoder/decoder inputs and labels.
# ---------------------------------------------------------------------------

def tokenize_batch(flux_cpu, tokenizer, ctx_len, pred_len, device):
    """flux_cpu: (B, ctx_len + pred_len) on CPU.

    Returns (input_ids, attention_mask, labels) on `device`, ready for
    T5.forward(). Padded label positions are -100 so they're ignored by CE.
    """
    context = flux_cpu[:, :ctx_len]
    label = flux_cpu[:, ctx_len:ctx_len + pred_len]

    input_ids, attention_mask, scale = tokenizer.context_input_transform(context)
    label_ids, label_mask = tokenizer.label_input_transform(label, scale)

    # T5 ignores -100 label positions in CE
    labels = label_ids.clone()
    labels[~label_mask.bool()] = -100

    return (
        input_ids.to(device, non_blocking=True),
        attention_mask.to(device, non_blocking=True),
        labels.to(device, non_blocking=True),
    )


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------

def train_epoch(model, loader, tokenizer, optimizer, scheduler, device, rank,
                ctx_len, pred_len, amp_dtype, use_amp):
    model.train()
    total, n_batches = 0.0, 0
    pbar = tqdm(loader, desc="Train", disable=(rank != 0))
    for flux in pbar:
        # flux stays on CPU; tokenize_batch handles device transfer of tokens
        input_ids, attn_mask, labels = tokenize_batch(
            flux, tokenizer, ctx_len, pred_len, device
        )

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            out = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                labels=labels,
            )
            loss = out.loss

        if not torch.isfinite(loss):
            # NaN guard: skip step, don't poison the running mean
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
def validate(model, loader, tokenizer, device, rank, ctx_len, pred_len):
    model.eval()
    total, n_batches = 0.0, 0
    for flux in tqdm(loader, desc="Val", disable=(rank != 0)):
        input_ids, attn_mask, labels = tokenize_batch(
            flux, tokenizer, ctx_len, pred_len, device
        )
        out = model(
            input_ids=input_ids, attention_mask=attn_mask, labels=labels
        )
        if torch.isfinite(out.loss):
            total += out.loss.item()
            n_batches += 1

    avg = torch.tensor(total / max(n_batches, 1), device=device)
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
        print(f"DDP={world_size>1} · world={world_size} · device={device} · precision={args.precision}")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    chronos_config = build_chronos_config(args.context_length, args.prediction_length)
    tokenizer = chronos_config.create_tokenizer()
    # Tokenization happens on CPU: chronos's MeanScaleUniformBins constructs
    # intermediate tensors (boundaries, EOS tokens, attention masks) without
    # a device argument, so they default to CPU. Rather than monkey-patching
    # every call site, we keep flux on CPU through tokenize_batch and only
    # transfer the resulting int tokens to GPU. Bonus: transfer volume drops
    # ~2× since int64 tokens are smaller than float flux even with EOS added.

    train_ds = ChronosFluxDataset(
        matchfile_dir=args.matchfile_dir,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        max_sources=args.max_sources,
        shuffle=True,
        multiband=args.multiband,
        holdout_path=args.holdout_path,
    )
    val_ds = ChronosFluxDataset(
        matchfile_dir=args.matchfile_dir,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
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
        val_ds, batch_size=args.batch_size, num_workers=max(args.num_workers // 2, 1),
        pin_memory=True, drop_last=True, collate_fn=_flux_collate,
    )

    model = build_t5_model(args, chronos_config).to(device)
    if rank == 0:
        n_params = count_parameters(model)
        print(f"T5 parameters: {n_params:,}  "
              f"(d_model={args.d_model}, layers={args.num_layers}+{args.num_layers}, "
              f"d_ff={args.d_ff}, vocab={chronos_config.n_tokens})")

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

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
            model, train_loader, tokenizer, optimizer, scheduler,
            device, rank, args.context_length, args.prediction_length,
            amp_dtype, use_amp,
        )
        val_loss = validate(
            model, val_loader, tokenizer, device, rank,
            args.context_length, args.prediction_length,
        )
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
                    "t5_config": inner.config.to_dict(),
                    "chronos_config": chronos_config.__dict__,
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Chronos-T5 from-scratch baseline")
    sub = p.add_subparsers(dest="command", required=True)

    pt = sub.add_parser("pretrain")
    pt.add_argument("--matchfile_dir", type=str, required=True)
    pt.add_argument("--output_dir", type=str, default="checkpoints/chronos_t5_5M")
    pt.add_argument("--max_sources", type=int, default=400_000)
    pt.add_argument("--multiband", action="store_true")
    pt.add_argument("--holdout_path", type=str, default=None)

    # Architecture (~5M default)
    pt.add_argument("--d_model", type=int, default=192)
    pt.add_argument("--num_layers", type=int, default=4,
                    help="Encoder and decoder layer count (T5 uses both).")
    pt.add_argument("--num_heads", type=int, default=4)
    pt.add_argument("--d_ff", type=int, default=768)
    pt.add_argument("--dropout", type=float, default=0.1)

    # Window
    pt.add_argument("--context_length", type=int, default=320)
    pt.add_argument("--prediction_length", type=int, default=64)

    # Optimizer
    pt.add_argument("--batch_size", type=int, default=64)
    pt.add_argument("--epochs", type=int, default=40)
    pt.add_argument("--lr", type=float, default=3e-4)
    pt.add_argument("--weight_decay", type=float, default=1e-4)
    pt.add_argument("--warmup_epochs", type=int, default=2)
    pt.add_argument("--patience", type=int, default=10)
    pt.add_argument("--precision", type=str, default="bf16",
                    choices=["fp32", "fp16", "bf16"])

    pt.add_argument("--num_workers", type=int, default=4)
    pt.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    if args.command == "pretrain":
        pretrain(args)
