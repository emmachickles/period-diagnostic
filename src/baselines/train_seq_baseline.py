"""
DDP training for CNN / RNN sequence encoders on the ZTF SSL benchmark.

Identical pipeline to src/training/train_ct_ddp.py — same DDPDisentangledLoss,
same all_gather, same matchfile dataset, same checkpoint resume — but the
ContinuousTimeLightCurveTransformer is swapped for a 1D CNN or BiGRU.

Why a separate file: keeps train_ct_ddp.py untouched (the headline pipeline),
and isolates the baseline experiment so a future regression in either path
can't silently change the headline numbers.

Launch:
    torchrun --nproc_per_node=4 src/baselines/train_seq_baseline.py \
        --encoder cnn --matchfile_dir /path/to/ZTF/matchfiles \
        --output_dir checkpoints/cnn_v1 --max_sources 1000000 [...]
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

from src.training.train_ct_ddp import (
    setup_ddp, cleanup_ddp, DDPDisentangledLoss, train_epoch, validate, _amp_dtype,
)
from src.model.utils import (
    count_parameters, EarlyStopping, WarmupCosineSchedule,
    save_checkpoint, load_checkpoint,
)
from src.data.matchfile_dataset import create_dataloader
from src.baselines.seq_encoders import build_encoder


def train(args):
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"DDP training with {world_size} GPUs · "
              f"encoder={args.encoder} · precision={args.precision}")
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

    encoder_kwargs = dict(
        d_model=args.d_model,
        num_layers=args.num_layers,
        d_sig=args.d_sig,
        d_qual=args.d_qual,
        dropout=args.dropout,
    )
    if args.encoder == "cnn":
        encoder_kwargs["kernel_size"] = args.kernel_size
    elif args.encoder == "rnn":
        encoder_kwargs["bidirectional"] = bool(args.bidirectional)

    model = build_encoder(args.encoder, **encoder_kwargs).to(device)
    # Explicit barrier forces all ranks to finish model construction before DDP
    # introspects parameter shapes. Without it, _verify_params_across_processes
    # can race and report "Rank 0 has inconsistent 0 params" because rank 0
    # hasn't yet finished registering its modules when peers query.
    if dist.is_initialized():
        dist.barrier()
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

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
    scaler = GradScaler(enabled=(args.precision == "fp16"))
    early_stopping = EarlyStopping(patience=args.patience, mode="min")

    best_val_loss = float("inf")
    history = {"train": [], "val": []}
    start_epoch = 0

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
                  f"Val: {val_metrics['loss']:.4f} | {elapsed:.0f}s",
                  flush=True)
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
    p = argparse.ArgumentParser(description="DDP CNN/RNN baseline training")
    p.add_argument("--encoder", required=True, choices=["cnn", "rnn"])
    p.add_argument("--matchfile_dir", type=str, required=True)
    p.add_argument("--max_sources", type=int, default=None)
    p.add_argument("--max_seq_len", type=int, default=384)
    p.add_argument("--multiband", action="store_true")

    # Encoder hyperparams (defaults targeting ~4.5M params, near v5's 4.8M)
    p.add_argument("--d_model", type=int, default=352)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--kernel_size", type=int, default=7,
                   help="CNN only")
    p.add_argument("--bidirectional", type=int, default=1,
                   help="RNN only (1=BiGRU, 0=GRU)")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--d_sig", type=int, default=128)
    p.add_argument("--d_qual", type=int, default=32)

    # Loss head
    p.add_argument("--proj_hidden_dim", type=int, default=256)
    p.add_argument("--proj_output_dim", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--lambda_qual", type=float, default=1.0)

    # Optim
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int, default=2)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--precision", type=str, default="bf16",
                   choices=["fp32", "fp16", "bf16"])

    p.add_argument("--output_dir", type=str, default="checkpoints/seq_baseline")
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume_from", type=str, default=None)
    p.add_argument("--holdout_path", type=str, default=None)
    p.add_argument("--use_disjoint_windows", type=int, default=1)

    args = p.parse_args()
    train(args)
