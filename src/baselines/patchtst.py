"""
PatchTST baseline for comparison with our disentangled SSL transformer.

PatchTST (Nie et al. 2023) is a channel-independent patch-based transformer
for time series.  Rather than interpolating irregular ZTF light curves onto a
regular grid (which destroys period information), we feed the raw observations
directly as a 3-channel sequence: (time, flux, flux_err).  Each position
corresponds to one observation sorted by time, preserving the true cadence.
PatchTST treats the time channel like any other variate, allowing it to learn
irregular spacing from the data.

Self-supervised pretraining uses masked patch prediction (PatchTST's native
SSL objective). Evaluation uses the same linear probe protocol as our model.

Usage:
    # Pretrain on matchfiles
    python src/baselines/patchtst.py pretrain \
        --matchfile_dir /path/to/matchfiles \
        --output_dir checkpoints/patchtst_100k \
        --max_sources 100000

    # Evaluate with linear probe
    python src/baselines/patchtst.py evaluate \
        --checkpoint checkpoints/patchtst_100k/best_model.pt \
        --lightcurves_npz /path/to/raw_lightcurves.npz \
        --labels_csv /path/to/catalog_raw.csv
"""

import sys
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from src.data.preprocessing import calibrate_flux, filter_good, mad_normalize
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from tqdm import tqdm
from transformers import PatchTSTConfig, PatchTSTForPretraining, PatchTSTModel

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.model.utils import get_device


# ---------------------------------------------------------------------------
# Packing: irregular observations -> fixed-length (time, flux, flux_err)
# ---------------------------------------------------------------------------

def pack_observations(time, flux, flux_err, max_len=512):
    """
    Pack irregular observations into a fixed-length 3-channel tensor.

    If the light curve has more than max_len observations, we center-crop.
    If fewer, we zero-pad and return a mask.

    Time is normalised to [0, 1] over the observed baseline so PatchTST
    sees cadence structure at a consistent scale.

    Parameters
    ----------
    time, flux, flux_err : 1-d arrays
        Observations sorted by time, already MAD-normalised (flux/err).
    max_len : int
        Fixed output length.

    Returns
    -------
    packed : (max_len, 3)  float32 — channels are (time_norm, flux, flux_err)
    mask   : (max_len,)    bool    — True where real data exists
    """
    n = len(time)

    if n > max_len:
        # Center crop
        start = (n - max_len) // 2
        time = time[start : start + max_len]
        flux = flux[start : start + max_len]
        flux_err = flux_err[start : start + max_len]
        n = max_len

    # Normalise time to [0, 1]
    t_min, t_max = time[0], time[-1]
    baseline = t_max - t_min
    if baseline > 0:
        time_norm = (time - t_min) / baseline
    else:
        time_norm = np.zeros_like(time)

    # Pack into (max_len, 3) with zero-padding
    packed = np.zeros((max_len, 3), dtype=np.float32)
    packed[:n, 0] = time_norm.astype(np.float32)
    packed[:n, 1] = flux.astype(np.float32)
    packed[:n, 2] = flux_err.astype(np.float32)

    mask = np.zeros(max_len, dtype=bool)
    mask[:n] = True

    return packed, mask


# ---------------------------------------------------------------------------
# Streaming dataset for PatchTST pretraining (reads matchfiles)
# ---------------------------------------------------------------------------

class PatchTSTMatchfileDataset(IterableDataset):
    """
    Streaming dataset that reads ZTF matchfiles and returns raw observations
    as 3-channel sequences (time, flux, flux_err) for PatchTST.

    No interpolation — each position is a real observation.
    """

    def __init__(
        self,
        matchfile_dir: str,
        max_len: int = 512,
        max_sources: int = None,
        min_epochs: int = 50,
        shuffle: bool = True,
    ):
        super().__init__()
        import glob as _glob
        import os

        self.max_len = max_len
        self.max_sources = max_sources
        self.min_epochs = min_epochs
        self.shuffle = shuffle

        pattern = os.path.join(matchfile_dir, "*", "data_*_zr.h5")
        self._files = sorted(_glob.glob(pattern))
        if not self._files:
            raise FileNotFoundError(f"No matchfiles matching {pattern}")

        self._h5_cache = {}

    def _get_h5(self, path):
        import h5py
        if path not in self._h5_cache:
            self._h5_cache[path] = h5py.File(path, "r")
        return self._h5_cache[path]

    def _read_source(self, h5_path, src_idx):
        """Read, MAD-normalise, and pack a single source."""
        try:
            f = self._get_h5(h5_path)
            src = f["data"]["sources"][src_idx]
            exp = f["data"]["exposures"][:]
            n_exp = len(exp)

            rows = np.arange(n_exp, dtype=np.int64) + src_idx * n_exp
            sd = f["data"]["sourcedata"][rows]

            bjd = exp["bjd"].astype(np.float64)
            flag = sd["flag"]
            diff_flux = sd["flux"].astype(np.float64)
            diff_ferr = sd["flux_err"].astype(np.float64)

            mag_ref = float(src["mag_ref"])
            cal_flux, _ = calibrate_flux(diff_flux, mag_ref)
            good = filter_good(cal_flux, diff_flux, diff_ferr, flag)
            if good.sum() < self.min_epochs:
                return None

            t = bjd[good]
            f_cal = cal_flux[good]
            f_err = diff_ferr[good]

            flux_norm, err_norm, _, _ = mad_normalize(f_cal, f_err)
            if flux_norm is None:
                return None

            # Sort by time
            order = np.argsort(t)
            t = t[order]
            flux_norm = flux_norm[order]
            err_norm = err_norm[order]

            packed, mask = pack_observations(
                t, flux_norm, err_norm, max_len=self.max_len
            )
            if not mask.any():
                return None

            return torch.from_numpy(packed)

        except Exception:
            return None

    def __iter__(self):
        worker_info = get_worker_info()
        files = list(self._files)
        if self.shuffle:
            np.random.shuffle(files)

        if worker_info is not None:
            files = files[worker_info.id :: worker_info.num_workers]

        budget = self.max_sources
        if budget is not None and worker_info is not None:
            budget = budget // worker_info.num_workers

        n_yielded = 0
        for h5_path in files:
            if budget is not None and n_yielded >= budget:
                break
            try:
                f = self._get_h5(h5_path)
                n_src = len(f["data"]["sources"])
            except Exception:
                continue

            indices = np.arange(n_src)
            if self.shuffle:
                np.random.shuffle(indices)

            for idx in indices:
                if budget is not None and n_yielded >= budget:
                    break
                sample = self._read_source(h5_path, int(idx))
                if sample is not None:
                    yield sample
                    n_yielded += 1

    def __del__(self):
        import h5py
        for v in self._h5_cache.values():
            if isinstance(v, h5py.File):
                try:
                    v.close()
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# PatchTST wrapper with pooled embedding extraction
# ---------------------------------------------------------------------------

class PatchTSTWrapper(nn.Module):
    """
    Wraps HuggingFace PatchTSTModel to provide a pooled embedding.

    PatchTSTModel outputs (batch, n_channels, n_patches, d_model).
    We mean-pool over channels and patches to get (batch, d_model).
    """

    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.encoder = PatchTSTModel(config)
        self.d_model = config.d_model

    def forward(self, past_values: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        past_values : (batch, context_length, num_input_channels)

        Returns
        -------
        embedding : (batch, d_model)
        """
        out = self.encoder(past_values=past_values)
        # last_hidden_state: (batch, n_channels, n_patches, d_model)
        h = out.last_hidden_state
        # Mean pool over channels and patches
        return h.mean(dim=(1, 2))


# ---------------------------------------------------------------------------
# Pretraining
# ---------------------------------------------------------------------------

def pretrain(args):
    """Self-supervised pretraining with masked patch prediction."""
    device = get_device()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Creating dataset...")
    dataset = PatchTSTMatchfileDataset(
        matchfile_dir=args.matchfile_dir,
        max_len=args.context_length,
        max_sources=args.max_sources,
        min_epochs=args.min_epochs,
        shuffle=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    print("Creating PatchTST model...")
    config = PatchTSTConfig(
        num_input_channels=3,  # time, flux, flux_err
        context_length=args.context_length,
        patch_length=args.patch_length,
        patch_stride=args.patch_stride,
        d_model=args.d_model,
        num_attention_heads=args.nhead,
        num_hidden_layers=args.num_layers,
        ffn_dim=args.dim_feedforward,
        dropout=args.dropout,
        mask_type="random",
        random_mask_ratio=args.mask_ratio,
    )
    model = PatchTSTForPretraining(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Cosine schedule
    estimated_batches = (args.max_sources or 100_000) // args.batch_size
    total_steps = estimated_batches * args.epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-6
    )

    # Save config for loading later
    config.save_pretrained(str(output_dir))

    print(f"\nStarting pretraining for {args.epochs} epochs...")
    best_loss = float("inf")
    history = []

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            # batch: (B, context_length, 3)
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            out = model(past_values=batch)
            loss = out.loss

            if not math.isnan(loss.item()):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                n_batches += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | {elapsed:.0f}s")

        history.append({"epoch": epoch, "loss": avg_loss, "time_s": elapsed})

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config.to_dict(),
                "epoch": epoch,
                "loss": avg_loss,
            }, str(output_dir / "best_model.pt"))
            print(f"  -> New best model (loss={best_loss:.4f})")

    # Save history
    with open(output_dir / "training_log.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nPretraining complete. Best loss: {best_loss:.4f}")
    print(f"Saved to {output_dir}")


# ---------------------------------------------------------------------------
# Evaluation: linear probe on frozen PatchTST embeddings
# ---------------------------------------------------------------------------

def load_and_pack_labeled(npz_path, labels_csv, max_len=512):
    """
    Load labeled light curves, apply same MAD normalisation,
    then pack as raw 3-channel observations for PatchTST.

    Returns list of dicts with: packed (max_len, 3), label
    """
    data = np.load(npz_path, allow_pickle=True)
    jd_arr = data["jd"]
    mag_arr = data["mag"]
    err_arr = data["err"]
    found_arr = data["found"]
    var_type = data["label"]

    samples = []
    for i in range(len(jd_arr)):
        if not found_arr[i]:
            continue

        jd = np.array(jd_arr[i], dtype=np.float64)
        mag = np.array(mag_arr[i], dtype=np.float64)
        err = np.array(err_arr[i], dtype=np.float64)
        label = str(var_type[i])

        if len(jd) < 20:
            continue

        good = np.isfinite(jd) & np.isfinite(mag) & np.isfinite(err) & (err > 0)
        if good.sum() < 20:
            continue

        jd, mag, err = jd[good], mag[good], err[good]

        flux_norm, err_norm, _, _ = mad_normalize(mag, err)
        if flux_norm is None:
            continue

        order = np.argsort(jd)
        jd, flux_norm, err_norm = jd[order], flux_norm[order], err_norm[order]

        packed, mask = pack_observations(jd, flux_norm, err_norm, max_len=max_len)
        if not mask.any():
            continue

        samples.append({"packed": packed, "label": label})

    return samples


@torch.no_grad()
def extract_patchtst_embeddings(model, samples, device, batch_size=64):
    """Extract pooled embeddings from frozen PatchTST encoder."""
    model.eval()

    all_emb = []
    all_labels = []

    for i in range(0, len(samples), batch_size):
        batch_samples = samples[i : i + batch_size]
        data = np.stack([s["packed"] for s in batch_samples])
        x = torch.from_numpy(data).to(device)

        emb = model(x)  # (B, d_model)
        all_emb.append(emb.cpu().numpy())
        all_labels.extend(s["label"] for s in batch_samples)

    embeddings = np.concatenate(all_emb, axis=0)
    labels = np.array(all_labels)
    return embeddings, labels


def evaluate(args):
    """Linear probe evaluation on frozen PatchTST embeddings."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix

    device = get_device()

    # Load model
    print("Loading PatchTST model...")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = PatchTSTConfig(**ckpt["config"])
    pretrain_model = PatchTSTForPretraining(config)
    pretrain_model.load_state_dict(ckpt["model_state_dict"])

    # Extract encoder and wrap for pooled embeddings
    wrapper = PatchTSTWrapper(config).to(device)
    # Copy encoder weights from pretrained model
    wrapper.encoder.load_state_dict(pretrain_model.model.state_dict())
    del pretrain_model

    # Load and pack labeled data
    print("Loading and packing labeled light curves...")
    samples = load_and_pack_labeled(
        args.lightcurves_npz, args.labels_csv,
        max_len=config.context_length,
    )
    print(f"Loaded {len(samples)} light curves")

    # Extract embeddings
    print("Extracting embeddings...")
    embeddings, labels = extract_patchtst_embeddings(
        wrapper, samples, device, batch_size=args.batch_size
    )
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Classes: {np.unique(labels, return_counts=True)}")

    # Linear probe (identical protocol to our model)
    print("\n--- Linear Probe: PatchTST ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(embeddings)

    clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(clf, X_scaled, labels, cv=skf)

    bal_acc = balanced_accuracy_score(labels, y_pred)
    report = classification_report(labels, y_pred, output_dict=True)
    cm = confusion_matrix(labels, y_pred)

    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"Per-class report:\n{classification_report(labels, y_pred)}")
    print(f"Confusion Matrix:\n{cm}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "patchtst_balanced_accuracy": float(bal_acc),
        "patchtst_report": report,
        "patchtst_confusion_matrix": cm.tolist(),
        "n_samples": len(labels),
        "embedding_dim": int(embeddings.shape[1]),
        "pretrain_checkpoint": args.checkpoint,
    }
    results_path = output_dir / "patchtst_linear_probe_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    np.savez(
        output_dir / "patchtst_embeddings.npz",
        embeddings=embeddings, labels=labels,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PatchTST baseline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- pretrain ---
    p_train = subparsers.add_parser("pretrain")
    p_train.add_argument("--matchfile_dir", type=str, required=True)
    p_train.add_argument("--output_dir", type=str, default="checkpoints/patchtst/")
    p_train.add_argument("--max_sources", type=int, default=100_000)
    p_train.add_argument("--min_epochs", type=int, default=50)
    p_train.add_argument("--context_length", type=int, default=512)
    p_train.add_argument("--patch_length", type=int, default=16)
    p_train.add_argument("--patch_stride", type=int, default=16)
    p_train.add_argument("--d_model", type=int, default=128)
    p_train.add_argument("--nhead", type=int, default=4)
    p_train.add_argument("--num_layers", type=int, default=3)
    p_train.add_argument("--dim_feedforward", type=int, default=512)
    p_train.add_argument("--dropout", type=float, default=0.1)
    p_train.add_argument("--mask_ratio", type=float, default=0.4)
    p_train.add_argument("--batch_size", type=int, default=64)
    p_train.add_argument("--epochs", type=int, default=20)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--weight_decay", type=float, default=1e-4)
    p_train.add_argument("--num_workers", type=int, default=4)
    p_train.add_argument("--seed", type=int, default=42)

    # --- evaluate ---
    p_eval = subparsers.add_parser("evaluate")
    p_eval.add_argument("--checkpoint", type=str, required=True)
    p_eval.add_argument("--lightcurves_npz", type=str, required=True)
    p_eval.add_argument("--labels_csv", type=str, required=True)
    p_eval.add_argument("--output_dir", type=str, default="results/patchtst/")
    p_eval.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()

    if args.command == "pretrain":
        pretrain(args)
    elif args.command == "evaluate":
        evaluate(args)
