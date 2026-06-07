#!/usr/bin/env python3
"""
embed_ct_burdge.py — Run the burdge CT checkpoint on the chen2020_25k
subset and produce a probe-comparable embeddings.npz.

Same pipeline as embed_ct_smoke.py, just with the production architecture
defaults that match orcd/train_ct_burdge_serious.slurm:
  d_model=256, num_layers=6, dim_feedforward=1024, max_seq_len=512.

Reads the matchfiles directly (not the chen2020 stored fluxes), which
matches the augmentation distribution the model trained on.
"""

import os
import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

REPO = Path(os.environ.get("PERIOD_DIAG_ROOT", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(REPO))
_ASTROTOOLS = os.environ.get("ASTROTOOLS_PATH")
if _ASTROTOOLS:  # local helper, only needed for raw matchfile ingestion
    sys.path.insert(0, _ASTROTOOLS)

from src.model.continuous_time_transformer import ContinuousTimeLightCurveTransformer
from src.model.utils import load_checkpoint, get_device
from src.data.preprocessing import mad_normalize, median_match
from astrotools.ztf import read_matchfile_lcs_batched

CHEN_PATH = os.environ.get("ZTF_CHEN_HDF5", str(REPO / "data" / "chen2020.hdf5"))
SUBSET_PATH = str(REPO / "results/chen2020_25k/subset_indices.npz")
CHECKPOINT = str(REPO / "checkpoints/ct_ddp_burdge_v1/best_model.pt")
OUTPUT = str(REPO / "results/chen2020_25k/embeddings_ct_burdge.npz")


def fetch_lcs(field, ccd, quad, ra, dec, row_keys, radius=3.0, min_epochs=10):
    raw = read_matchfile_lcs_batched(
        field=field, ccd=ccd, quad=quad,
        ra_targets=ra, dec_targets=dec,
        filters="gr", match_radius_arcsec=radius,
    )
    out = {}
    for ti, per_filt in raw.items():
        bands = list(per_filt.values())
        if not bands: continue
        if len(bands) == 1:
            t, f, e = bands[0]["time"], bands[0]["flux"], bands[0]["flux_err"]
        else:
            t, f, e = median_match(bands)
        if len(t) < min_epochs: continue
        flux_norm, err_norm, _, _ = mad_normalize(f, e)
        if flux_norm is None: continue
        out[int(row_keys[ti])] = {"time": t, "flux": flux_norm, "flux_err": err_norm}
    return out


@torch.no_grad()
def forward_batch(model, lcs, device, max_seq_len):
    B = len(lcs)
    t_b = torch.zeros(B, max_seq_len, dtype=torch.float32)
    f_b = torch.zeros(B, max_seq_len, dtype=torch.float32)
    e_b = torch.ones(B, max_seq_len, dtype=torch.float32)
    m_b = torch.zeros(B, max_seq_len, dtype=torch.bool)
    for j, lc in enumerate(lcs):
        t, f, e = lc["time"], lc["flux"], lc["flux_err"]
        L = len(t)
        if L > max_seq_len:
            s = (L - max_seq_len) // 2
            t = t[s:s + max_seq_len]; f = f[s:s + max_seq_len]; e = e[s:s + max_seq_len]
            sl = max_seq_len
        else: sl = L
        t_b[j, :sl] = torch.from_numpy(t[:sl].astype(np.float32))
        f_b[j, :sl] = torch.from_numpy(f[:sl].astype(np.float32))
        e_b[j, :sl] = torch.from_numpy(e[:sl].astype(np.float32))
        m_b[j, :sl] = True
    out = model(t_b.to(device), f_b.to(device), e_b.to(device), m_b.to(device))
    return out["z_sig"].cpu().numpy(), out["z_qual"].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default=SUBSET_PATH)
    ap.add_argument("--checkpoint", default=CHECKPOINT)
    ap.add_argument("--output", default=OUTPUT)
    # Production architecture (must match the SLURM config exactly).
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--num_layers", type=int, default=6)
    ap.add_argument("--dim_feedforward", type=int, default=1024)
    ap.add_argument("--n_fourier_features", type=int, default=64)
    ap.add_argument("--n_time_bias_pairs", type=int, default=16)
    ap.add_argument("--d_sig", type=int, default=128)
    ap.add_argument("--d_qual", type=int, default=32)
    args = ap.parse_args()

    sub = np.load(args.subset)
    chen_indices = sub["indices"]
    print(f"Subset: {len(chen_indices):,}")

    with h5py.File(CHEN_PATH, "r") as f:
        cat = f["catalog"]
        ls_field = cat["ls_field"][:][chen_indices].astype(int)
        ls_ccd = cat["ls_ccd"][:][chen_indices].astype(int)
        ls_quad = cat["ls_quad"][:][chen_indices].astype(int)
        ra = cat["ra"][:][chen_indices].astype(np.float64)
        dec = cat["dec"][:][chen_indices].astype(np.float64)

    keys = list(zip(ls_field, ls_ccd, ls_quad))
    unique_groups = sorted(set(keys))
    group_to_rows = {}
    for ri, k in enumerate(keys):
        group_to_rows.setdefault(k, []).append(ri)
    print(f"Groups: {len(unique_groups):,}")

    device = get_device()
    print(f"Device: {device}")
    print(f"Architecture: d_model={args.d_model}, layers={args.num_layers}, "
          f"K={args.n_time_bias_pairs} time-bias pairs, max_seq_len={args.max_seq_len}")
    model = ContinuousTimeLightCurveTransformer(
        d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward, dropout=0.1,
        n_fourier_features=args.n_fourier_features,
        d_sig=args.d_sig, d_qual=args.d_qual,
        n_time_bias_pairs=args.n_time_bias_pairs,
    ).to(device).eval()
    load_checkpoint(model, args.checkpoint, device=device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    N = len(chen_indices)
    z_sig = np.full((N, args.d_sig), np.nan, dtype=np.float32)
    z_qual = np.full((N, args.d_qual), np.nan, dtype=np.float32)
    success = np.zeros(N, dtype=bool)
    n_obs = np.zeros(N, dtype=np.int32)

    pending_lcs, pending_rows = [], []
    def flush():
        if not pending_lcs: return
        zs, zq = forward_batch(model, pending_lcs, device, args.max_seq_len)
        for ri, zsi, zqi in zip(pending_rows, zs, zq):
            z_sig[ri] = zsi; z_qual[ri] = zqi; success[ri] = True

    t0 = time.time(); n_done = 0
    for gi, (field, ccd, quad) in enumerate(unique_groups):
        rows = np.array(group_to_rows[(field, ccd, quad)])
        try:
            lcs_dict = fetch_lcs(int(field), int(ccd), int(quad), ra[rows], dec[rows], rows)
        except Exception:
            n_done += len(rows); continue
        for ri in rows:
            lc = lcs_dict.get(int(ri))
            if lc is None: continue
            n_obs[ri] = len(lc["time"])
            pending_lcs.append(lc); pending_rows.append(ri)
            if len(pending_lcs) >= args.batch_size:
                flush(); pending_lcs.clear(); pending_rows.clear()
        n_done += len(rows)
        if (gi + 1) % 1000 == 0 or gi == len(unique_groups) - 1:
            el = time.time() - t0
            r = n_done / max(el, 1e-9)
            eta = (N - n_done) / max(r, 1e-9)
            print(f"  group {gi+1:>5}/{len(unique_groups)} sources {n_done:,}/{N:,} ({r:.0f}/s, ~{eta/60:.1f} min)")
    flush()

    print(f"\n{success.sum():,}/{N:,} succeeded · {(time.time()-t0)/60:.1f} min")
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, z_sig=z_sig, z_qual=z_qual, n_obs=n_obs,
             success_mask=success, chen_indices=chen_indices)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
