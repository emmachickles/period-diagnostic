#!/usr/bin/env python3
"""
subset_chen2020_25k.py — Pick a stratified ~25K subset of the existing
chen2020_100k/embeddings.npz so the demo bundle fits comfortably in
GitHub Pages' 1 GB limit.

Strategy: for each var_type that succeeded inference, keep min(n, 3000)
sources at random with a fixed seed. Rare classes (Mira, CEPII, CEP) are
kept in full.

Output: results/chen2020_25k/embeddings.npz with the same keys
(z_sig, z_qual, n_obs, success_mask, chen_indices) but ~25K rows.
"""

import os
import argparse
from pathlib import Path

import h5py
import numpy as np

REPO = Path(os.environ.get("PERIOD_DIAG_ROOT", str(Path(__file__).resolve().parents[1])))
CHEN_PATH_DEFAULT = os.environ.get("ZTF_CHEN_HDF5", str(REPO / "data" / "chen2020.hdf5"))
EMB_IN_DEFAULT = str(REPO / "results/chen2020_100k/embeddings.npz")
EMB_OUT_DEFAULT = str(REPO / "results/chen2020_25k/embeddings.npz")


def safe_str(x):
    if isinstance(x, bytes):
        x = x.decode("utf-8", errors="replace")
    s = str(x)
    if len(s) >= 3 and s.startswith("b'") and s.endswith("'"):
        s = s[2:-1]
    elif len(s) >= 3 and s.startswith('b"') and s.endswith('"'):
        s = s[2:-1]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chen_hdf5", default=CHEN_PATH_DEFAULT)
    ap.add_argument("--embeddings_in", default=EMB_IN_DEFAULT)
    ap.add_argument("--embeddings_out", default=EMB_OUT_DEFAULT)
    ap.add_argument("--target_per_class", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    emb = np.load(args.embeddings_in)
    z_sig = emb["z_sig"]
    z_qual = emb["z_qual"]
    n_obs = emb["n_obs"]
    success_mask = emb["success_mask"]
    chen_indices = emb["chen_indices"]
    print(f"Input: {len(chen_indices):,} sampled, {success_mask.sum():,} succeeded")

    keep_init = np.where(success_mask)[0]
    chen_idx_kept = chen_indices[keep_init]

    with h5py.File(args.chen_hdf5, "r") as f:
        var_type_raw = f["catalog/var_type"][:]
    var_type = np.array([safe_str(s) for s in var_type_raw])
    vt_kept = var_type[chen_idx_kept]

    classes = np.unique(vt_kept)
    chosen_local = []
    print(f"\nPer-class subsampling (cap = {args.target_per_class:,}):")
    for cls in sorted(classes):
        idxs = np.where(vt_kept == cls)[0]
        n_take = min(len(idxs), args.target_per_class)
        picked = rng.choice(idxs, size=n_take, replace=False) if n_take > 0 else np.array([], dtype=int)
        chosen_local.append(picked)
        print(f"  {cls:>8s}: {n_take:>6,} / {len(idxs):>6,}")

    chosen_local = np.sort(np.concatenate(chosen_local))
    chosen_global = keep_init[chosen_local]
    print(f"\nTotal kept: {len(chosen_global):,}")

    # Save in the same format as embeddings.npz so existing export scripts
    # work with --embeddings pointing here.
    out_path = Path(args.embeddings_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        z_sig=z_sig[chosen_global],
        z_qual=z_qual[chosen_global],
        n_obs=n_obs[chosen_global],
        success_mask=np.ones(len(chosen_global), dtype=bool),
        chen_indices=chen_indices[chosen_global],
    )
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
