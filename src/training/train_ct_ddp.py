"""
Multi-GPU DDP training for the continuous-time SSL transformer.

Mirrors src/training/train_ddp.py — same loss, same data pipeline, same
all_gather for InfoNCE — but instantiates ContinuousTimeLightCurveTransformer
and threads through n_time_bias_pairs.

Launch with torchrun:
    torchrun --nproc_per_node=4 src/training/train_ct_ddp.py \
        --matchfile_dir /path/to/ZTF/matchfiles ...

Notes on this file vs train_ddp.py:
  - bf16 path added (A100s prefer bf16 over fp16: same throughput, fp32
    dynamic range, no GradScaler needed).
  - The DDPDisentangledLoss / projection-head gradient pattern is copied
    verbatim from train_ddp.py so the A/B comparison vs the Fourier baseline
    isn't muddied by unrelated changes.
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.model.continuous_time_transformer import ContinuousTimeLightCurveTransformer
from src.model.losses import DisentangledLoss, ProjectionHead, InfoNCELoss
from src.model.utils import (
    count_parameters,
    EarlyStopping,
    WarmupCosineSchedule,
    save_checkpoint,
    load_checkpoint,
)
from src.data.matchfile_dataset import create_dataloader


def setup_ddp():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def gather_from_all(tensor: torch.Tensor) -> torch.Tensor:
    """All-gather tensors across GPUs preserving local autograd.

    dist.all_gather is non-differentiable: the gathered tensors are
    fresh allocations with no autograd connection to the input. If we
    just `torch.cat(gathered)` and pass that to a loss, backward stops
    at the gather and the InfoNCE gradient never reaches the local
    z_sig_head / projection_head — DDP's reducer then complains:
        "Parameter indices which did not receive grad: 94 95".
    The standard SSL-DDP pattern is to splice the local-rank slice back
    into the gathered list so its autograd graph is preserved. Other
    ranks' contributions remain detached (acting purely as negatives,
    which is what we want).
    """
    rank = dist.get_rank()
    gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    gathered[rank] = tensor  # restore autograd connection for the local slice
    return torch.cat(gathered, dim=0)


class DDPDisentangledLoss(nn.Module):
    """DDP-aware disentangled loss with all_gather for contrastive negatives.

    Takes z_qual from BOTH views: under DDP, every parameter that
    participates in forward must also participate in the loss, otherwise
    the reducer raises "Parameter indices which did not receive grad".
    The single-GPU DisentangledLoss could afford to use only z_qual_1
    because torch.autograd doesn't care about unused outputs; DDP does.
    Averaging the two MSE losses is also a stronger learning signal —
    both views' qual heads get gradient instead of one.
    """

    def __init__(self, d_sig=128, proj_hidden_dim=256, proj_output_dim=128,
                 temperature=0.07, lambda_qual=1.0):
        super().__init__()
        self.lambda_qual = lambda_qual
        self.projection_head = ProjectionHead(d_sig, proj_hidden_dim, proj_output_dim)
        self.infonce = InfoNCELoss(temperature=temperature)

    def forward(self, z_sig_1, z_sig_2, z_qual_1, z_qual_2, log_median_sigma):
        proj_1 = self.projection_head(z_sig_1)
        proj_2 = self.projection_head(z_sig_2)
        if dist.is_initialized() and dist.get_world_size() > 1:
            proj_1_all = gather_from_all(proj_1.contiguous())
            proj_2_all = gather_from_all(proj_2.contiguous())
        else:
            proj_1_all = proj_1
            proj_2_all = proj_2
        loss_contrastive = self.infonce(proj_1_all, proj_2_all)
        loss_qual_1 = nn.functional.mse_loss(z_qual_1.squeeze(-1), log_median_sigma)
        loss_qual_2 = nn.functional.mse_loss(z_qual_2.squeeze(-1), log_median_sigma)
        loss_qual = 0.5 * (loss_qual_1 + loss_qual_2)
        loss = loss_contrastive + self.lambda_qual * loss_qual
        return {
            "loss": loss,
            "loss_contrastive": loss_contrastive.item(),
            "loss_qual": loss_qual.item(),
        }


def _amp_dtype(precision: str):
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]


def train_epoch(model, dataloader, criterion, optimizer, scheduler, scaler,
                device, rank, amp_dtype, use_amp):
    model.train()
    criterion.train()
    total_loss, total_c, total_q, n_batches = 0.0, 0.0, 0.0, 0

    pbar = tqdm(dataloader, desc="Train", disable=(rank != 0))
    for batch in pbar:
        t1 = batch["time_1"].to(device); f1 = batch["flux_1"].to(device)
        e1 = batch["flux_err_1"].to(device); m1 = batch["mask_1"].to(device)
        t2 = batch["time_2"].to(device); f2 = batch["flux_2"].to(device)
        e2 = batch["flux_err_2"].to(device); m2 = batch["mask_2"].to(device)
        log_med_sigma = batch["log_median_sigma"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            out1 = model(t1, f1, e1, m1)
            out2 = model(t2, f2, e2, m2)
            losses = criterion(
                out1["z_sig"], out2["z_sig"],
                out1["z_qual"], out2["z_qual"], log_med_sigma,
            )

        if scaler.is_enabled():
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        total_loss += losses["loss"].item()
        total_c += losses["loss_contrastive"]
        total_q += losses["loss_qual"]
        n_batches += 1
        if rank == 0:
            pbar.set_postfix({"loss": f'{losses["loss"].item():.4f}'})

    n = max(n_batches, 1)
    return {"loss": total_loss / n, "loss_contrastive": total_c / n, "loss_qual": total_q / n}


@torch.no_grad()
def validate(model, dataloader, criterion, device, rank):
    model.eval()
    criterion.eval()
    total_loss, n_batches = 0.0, 0
    for batch in tqdm(dataloader, desc="Val", disable=(rank != 0)):
        t1 = batch["time_1"].to(device); f1 = batch["flux_1"].to(device)
        e1 = batch["flux_err_1"].to(device); m1 = batch["mask_1"].to(device)
        t2 = batch["time_2"].to(device); f2 = batch["flux_2"].to(device)
        e2 = batch["flux_err_2"].to(device); m2 = batch["mask_2"].to(device)
        log_med_sigma = batch["log_median_sigma"].to(device)
        out1 = model(t1, f1, e1, m1)
        out2 = model(t2, f2, e2, m2)
        losses = criterion(out1["z_sig"], out2["z_sig"],
                            out1["z_qual"], out2["z_qual"], log_med_sigma)
        total_loss += losses["loss"].item()
        n_batches += 1

    n = max(n_batches, 1)
    avg_loss = torch.tensor(total_loss / n, device=device)
    if dist.is_initialized():
        dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
    return {"loss": avg_loss.item()}


def train(args):
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"DDP training with {world_size} GPUs · precision={args.precision}")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    train_loader = create_dataloader(
        matchfile_dir=args.matchfile_dir, batch_size=args.batch_size,
        num_workers=args.num_workers, max_seq_len=args.max_seq_len,
        contrastive=True, max_sources=args.max_sources, multiband=args.multiband,
        holdout_path=args.holdout_path,
        use_disjoint_windows=args.use_disjoint_windows,
    )
    val_loader = create_dataloader(
        matchfile_dir=args.matchfile_dir, batch_size=args.batch_size,
        num_workers=max(args.num_workers // 2, 1), max_seq_len=args.max_seq_len,
        contrastive=True,
        max_sources=max(args.max_sources // 5, 1000) if args.max_sources else None,
        shuffle=False, multiband=args.multiband,
        holdout_path=args.holdout_path,
        use_disjoint_windows=args.use_disjoint_windows,
    )

    model = ContinuousTimeLightCurveTransformer(
        d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward, dropout=args.dropout,
        n_fourier_features=args.n_fourier_features,
        d_sig=args.d_sig, d_qual=args.d_qual,
        n_time_bias_pairs=args.n_time_bias_pairs,
    ).to(device)
    model = DDP(model, device_ids=[local_rank])

    if rank == 0:
        print(f"Model parameters: {count_parameters(model):,}")

    criterion = DDPDisentangledLoss(
        d_sig=args.d_sig, proj_hidden_dim=args.proj_hidden_dim,
        proj_output_dim=args.proj_output_dim,
        temperature=args.temperature, lambda_qual=args.lambda_qual,
    ).to(device)

    all_params = list(model.parameters()) + list(criterion.parameters())
    optimizer = optim.AdamW(all_params, lr=args.lr, weight_decay=args.weight_decay)

    estimated_batches = (args.max_sources or 1_000_000) // (args.batch_size * world_size)
    total_steps = estimated_batches * args.epochs
    warmup_steps = estimated_batches * args.warmup_epochs
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps, total_steps)

    use_amp = args.precision in ("fp16", "bf16")
    amp_dtype = _amp_dtype(args.precision)
    # GradScaler only meaningful for fp16; bf16 has fp32 dynamic range.
    scaler = GradScaler(enabled=(args.precision == "fp16"))
    early_stopping = EarlyStopping(patience=args.patience, mode="min")

    best_val_loss = float("inf")
    history = {"train": [], "val": []}
    start_epoch = 0

    # Optional resume — used by the SLURM script's auto-requeue path so a
    # job that exhausts its time limit picks up where it left off.
    if args.resume_from and Path(args.resume_from).exists():
        if rank == 0:
            print(f"Resuming from {args.resume_from}")
        ckpt = load_checkpoint(
            model.module, args.resume_from,
            optimizer=optimizer, criterion=criterion,
            scheduler=scheduler, device=device,
        )
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val_loss = float(ckpt.get("metrics", {}).get("loss", best_val_loss))
        log_path = Path(args.output_dir) / "training_log.json"
        if log_path.exists():
            try:
                history = json.loads(log_path.read_text())
            except Exception:
                pass
        if rank == 0:
            print(f"  resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler,
            device, rank, amp_dtype, use_amp,
        )
        val_metrics = validate(model, val_loader, criterion, device, rank)
        elapsed = time.time() - t0

        if rank == 0:
            print(f"Epoch {epoch:3d} | Train: {train_metrics['loss']:.4f} | "
                  f"Val: {val_metrics['loss']:.4f} | {elapsed:.0f}s")
            history["train"].append({**train_metrics, "epoch": epoch})
            history["val"].append({**val_metrics, "epoch": epoch})
            with open(Path(args.output_dir) / "training_log.json", "w") as f:
                json.dump(history, f, indent=2)
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                save_checkpoint(
                    model.module, optimizer, epoch, val_metrics,
                    str(Path(args.output_dir) / "best_model.pt"), criterion,
                    scheduler=scheduler,
                )

        if early_stopping(val_metrics["loss"]):
            if rank == 0:
                print(f"Early stopping at epoch {epoch}")
            break

    cleanup_ddp()
    if rank == 0:
        print(f"Training complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDP CT-SSL training")
    parser.add_argument("--matchfile_dir", type=str, required=True)
    parser.add_argument("--max_sources", type=int, default=None)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--multiband", action="store_true")

    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--dim_feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--n_fourier_features", type=int, default=64)
    parser.add_argument("--n_time_bias_pairs", type=int, default=16,
                        help="K in the continuous-time attention bias B(Δt)")
    parser.add_argument("--d_sig", type=int, default=128)
    parser.add_argument("--d_qual", type=int, default=32)

    parser.add_argument("--proj_hidden_dim", type=int, default=256)
    parser.add_argument("--proj_output_dim", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lambda_qual", type=float, default=1.0)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--precision", type=str, default="bf16",
                        choices=["fp32", "fp16", "bf16"])

    parser.add_argument("--output_dir", type=str, default="checkpoints/ct_ddp")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to a checkpoint .pt; if set, restore model, "
                             "optimizer, criterion state and continue training.")
    parser.add_argument("--holdout_path", type=str, default=None,
                        help="Path to a per-tile (ra, dec) holdout npz; "
                             "matched sources are skipped during iteration. "
                             "See scripts/build_chen_holdout.py.")
    parser.add_argument("--use_disjoint_windows", type=int, default=1,
                        help="1 (default): disjoint-window augmentation. 0: skip — "
                             "both contrastive views start from the full sequence. "
                             "Used for the augmentation ablation.")

    args = parser.parse_args()
    train(args)
