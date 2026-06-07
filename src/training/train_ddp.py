"""
Multi-GPU DDP training script for disentangled self-supervised learning.

Launch with torchrun:
    torchrun --nproc_per_node=2 src/training/train_ddp.py --matchfile_dir /path/to/ZTF/matchfiles

Key DDP consideration for contrastive learning:
  InfoNCE benefits from large negative sets. We use all_gather to collect
  z_sig embeddings across all GPUs before computing the similarity matrix.
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

from src.model.transformer import DisentangledLightCurveTransformer
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
    """Initialize DDP from torchrun environment variables."""
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def gather_from_all(tensor: torch.Tensor) -> torch.Tensor:
    """All-gather tensors across GPUs for full contrastive negative set."""
    gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)


class DDPDisentangledLoss(nn.Module):
    """
    DDP-aware disentangled loss with all_gather for contrastive negatives.

    Gathers z_sig projections across all GPUs before computing InfoNCE,
    giving a larger effective batch of negatives.
    """

    def __init__(
        self,
        d_sig: int = 128,
        proj_hidden_dim: int = 256,
        proj_output_dim: int = 128,
        temperature: float = 0.07,
        lambda_qual: float = 1.0,
    ):
        super().__init__()
        self.lambda_qual = lambda_qual
        self.projection_head = ProjectionHead(d_sig, proj_hidden_dim, proj_output_dim)
        self.infonce = InfoNCELoss(temperature=temperature)

    def forward(self, z_sig_1, z_sig_2, z_qual, log_median_sigma):
        # Project
        proj_1 = self.projection_head(z_sig_1)
        proj_2 = self.projection_head(z_sig_2)

        # All-gather across GPUs for larger negative set
        if dist.is_initialized() and dist.get_world_size() > 1:
            proj_1_all = gather_from_all(proj_1.contiguous())
            proj_2_all = gather_from_all(proj_2.contiguous())
        else:
            proj_1_all = proj_1
            proj_2_all = proj_2

        loss_contrastive = self.infonce(proj_1_all, proj_2_all)
        loss_qual = nn.functional.mse_loss(z_qual.squeeze(-1), log_median_sigma)
        loss = loss_contrastive + self.lambda_qual * loss_qual

        return {
            "loss": loss,
            "loss_contrastive": loss_contrastive.item(),
            "loss_qual": loss_qual.item(),
        }


def train_epoch(model, dataloader, criterion, optimizer, scheduler, scaler, device, rank):
    model.train()
    criterion.train()

    total_loss = 0.0
    total_contrastive = 0.0
    total_qual = 0.0
    n_batches = 0

    pbar = tqdm(dataloader, desc="Train", disable=(rank != 0))
    for batch in pbar:
        t1 = batch["time_1"].to(device)
        f1 = batch["flux_1"].to(device)
        e1 = batch["flux_err_1"].to(device)
        m1 = batch["mask_1"].to(device)
        t2 = batch["time_2"].to(device)
        f2 = batch["flux_2"].to(device)
        e2 = batch["flux_err_2"].to(device)
        m2 = batch["mask_2"].to(device)
        log_med_sigma = batch["log_median_sigma"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", enabled=scaler.is_enabled()):
            out1 = model(t1, f1, e1, m1)
            out2 = model(t2, f2, e2, m2)
            losses = criterion(
                out1["z_sig"], out2["z_sig"],
                out1["z_qual"], log_med_sigma,
            )

        scaler.scale(losses["loss"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += losses["loss"].item()
        total_contrastive += losses["loss_contrastive"]
        total_qual += losses["loss_qual"]
        n_batches += 1

        if rank == 0:
            pbar.set_postfix({"loss": f'{losses["loss"].item():.4f}'})

    n = max(n_batches, 1)
    return {
        "loss": total_loss / n,
        "loss_contrastive": total_contrastive / n,
        "loss_qual": total_qual / n,
    }


@torch.no_grad()
def validate(model, dataloader, criterion, device, rank):
    model.eval()
    criterion.eval()

    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Val", disable=(rank != 0)):
        t1 = batch["time_1"].to(device)
        f1 = batch["flux_1"].to(device)
        e1 = batch["flux_err_1"].to(device)
        m1 = batch["mask_1"].to(device)
        t2 = batch["time_2"].to(device)
        f2 = batch["flux_2"].to(device)
        e2 = batch["flux_err_2"].to(device)
        m2 = batch["mask_2"].to(device)
        log_med_sigma = batch["log_median_sigma"].to(device)

        out1 = model(t1, f1, e1, m1)
        out2 = model(t2, f2, e2, m2)
        losses = criterion(
            out1["z_sig"], out2["z_sig"],
            out1["z_qual"], log_med_sigma,
        )

        total_loss += losses["loss"].item()
        n_batches += 1

    n = max(n_batches, 1)

    # Average across GPUs
    avg_loss = torch.tensor(total_loss / n, device=device)
    if dist.is_initialized():
        dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)

    return {"loss": avg_loss.item()}


def train(args):
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"DDP training with {world_size} GPUs")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    # Data
    train_loader = create_dataloader(
        matchfile_dir=args.matchfile_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_seq_len=args.max_seq_len,
        contrastive=True,
        max_sources=args.max_sources,
        multiband=args.multiband,
    )

    val_loader = create_dataloader(
        matchfile_dir=args.matchfile_dir,
        batch_size=args.batch_size,
        num_workers=max(args.num_workers // 2, 1),
        max_seq_len=args.max_seq_len,
        contrastive=True,
        max_sources=max(args.max_sources // 5, 1000) if args.max_sources else None,
        shuffle=False,
        multiband=args.multiband,
    )

    # Model with DDP
    model = DisentangledLightCurveTransformer(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        n_fourier_features=args.n_fourier_features,
        d_sig=args.d_sig,
        d_qual=args.d_qual,
    ).to(device)

    model = DDP(model, device_ids=[local_rank])

    if rank == 0:
        print(f"Model parameters: {count_parameters(model):,}")

    # DDP-aware loss
    criterion = DDPDisentangledLoss(
        d_sig=args.d_sig,
        proj_hidden_dim=args.proj_hidden_dim,
        proj_output_dim=args.proj_output_dim,
        temperature=args.temperature,
        lambda_qual=args.lambda_qual,
    ).to(device)

    # Optimizer
    all_params = list(model.parameters()) + list(criterion.parameters())
    optimizer = optim.AdamW(all_params, lr=args.lr, weight_decay=args.weight_decay)

    estimated_batches = (args.max_sources or 1_000_000) // (args.batch_size * world_size)
    total_steps = estimated_batches * args.epochs
    warmup_steps = estimated_batches * args.warmup_epochs
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps, total_steps)

    scaler = GradScaler(enabled=(args.precision == "fp16"))
    early_stopping = EarlyStopping(patience=args.patience, mode="min")

    best_val_loss = float("inf")
    history = {"train": [], "val": []}

    for epoch in range(args.epochs):
        t0 = time.time()
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device, rank
        )
        val_metrics = validate(model, val_loader, criterion, device, rank)
        elapsed = time.time() - t0

        if rank == 0:
            print(
                f"Epoch {epoch:3d} | Train: {train_metrics['loss']:.4f} | "
                f"Val: {val_metrics['loss']:.4f} | {elapsed:.0f}s"
            )

            history["train"].append({**train_metrics, "epoch": epoch})
            history["val"].append({**val_metrics, "epoch": epoch})

            with open(Path(args.output_dir) / "training_log.json", "w") as f:
                json.dump(history, f, indent=2)

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                save_checkpoint(
                    model.module, optimizer, epoch, val_metrics,
                    str(Path(args.output_dir) / "best_model.pt"), criterion,
                )

        if early_stopping(val_metrics["loss"]):
            if rank == 0:
                print(f"Early stopping at epoch {epoch}")
            break

    cleanup_ddp()
    if rank == 0:
        print(f"Training complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDP disentangled SSL training")

    parser.add_argument("--matchfile_dir", type=str, required=True,
                        help="Root dir containing field subdirs with HDF5 matchfiles")
    parser.add_argument("--max_sources", type=int, default=None)
    parser.add_argument("--max_seq_len", type=int, default=768)
    parser.add_argument("--multiband", action="store_true",
                        help="Combine g/r/i bands per source (median-matched)")

    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--dim_feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--n_fourier_features", type=int, default=64)
    parser.add_argument("--d_sig", type=int, default=128)
    parser.add_argument("--d_qual", type=int, default=32)

    parser.add_argument("--proj_hidden_dim", type=int, default=256)
    parser.add_argument("--proj_output_dim", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lambda_qual", type=float, default=1.0)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--precision", type=str, default="fp16")

    parser.add_argument("--output_dir", type=str, default="checkpoints/")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    train(args)
