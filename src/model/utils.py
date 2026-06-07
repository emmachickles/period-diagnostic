"""
Model utilities: checkpointing, scheduling, early stopping.
"""

import math
import torch
import torch.nn as nn
from typing import Dict, Optional
from pathlib import Path


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class EarlyStopping:
    """Early stopping based on validation metric."""

    def __init__(self, patience: int = 15, min_delta: float = 0.0, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "min":
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class WarmupCosineSchedule:
    """Linear warmup + cosine decay learning rate schedule."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if self.current_step < self.warmup_steps:
                lr = base_lr * self.current_step / self.warmup_steps
            else:
                progress = (self.current_step - self.warmup_steps) / max(
                    self.total_steps - self.warmup_steps, 1
                )
                lr = self.min_lr + (base_lr - self.min_lr) * 0.5 * (
                    1.0 + math.cos(math.pi * progress)
                )
            param_group["lr"] = lr

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def state_dict(self) -> Dict:
        # The optimizer is owned elsewhere; only persist scheduler-internal
        # state so a resumed run picks up at the same step on the curve.
        return {
            "current_step": self.current_step,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr": self.min_lr,
            "base_lrs": self.base_lrs,
        }

    def load_state_dict(self, state: Dict) -> None:
        self.current_step = int(state.get("current_step", 0))
        # Treat warmup/total/min/base as immutable across the run; restore
        # only if they happen to be present (defensive for old checkpoints).
        if "warmup_steps" in state: self.warmup_steps = int(state["warmup_steps"])
        if "total_steps" in state: self.total_steps = int(state["total_steps"])
        if "min_lr" in state: self.min_lr = float(state["min_lr"])
        if "base_lrs" in state: self.base_lrs = list(state["base_lrs"])


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    filepath: str,
    criterion: Optional[nn.Module] = None,
    scheduler: Optional[object] = None,
):
    """Save model checkpoint.

    The scheduler argument is optional but should be passed when the
    training loop uses a stateful LR schedule (e.g. WarmupCosineSchedule).
    Without it, a resumed run restarts the warmup at step 0 even though
    the optimizer is at step N — causing a transient train/val regression.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    if criterion is not None:
        checkpoint["criterion_state_dict"] = criterion.state_dict()
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, filepath)


def load_checkpoint(
    model: nn.Module,
    filepath: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    criterion: Optional[nn.Module] = None,
    scheduler: Optional[object] = None,
    device: Optional[torch.device] = None,
) -> Dict:
    """Load model checkpoint."""
    if device is None:
        device = get_device()

    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if criterion is not None and "criterion_state_dict" in checkpoint:
        criterion.load_state_dict(checkpoint["criterion_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint \
            and hasattr(scheduler, "load_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint
