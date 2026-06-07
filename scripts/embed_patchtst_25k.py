#!/usr/bin/env python3
"""
embed_patchtst_25k.py — Run PatchTST inference on the chen2020_25k
subset and save embeddings in the same npz schema as embed_ct_burdge.py
so probe_25k.py works directly.

Mirrors the embed_ct_burdge.py pipeline (group sources by tile, fetch
multi-band MAD-normalized matchfile lightcurves) but feeds them through
PatchTST's pack_observations + PatchTSTWrapper instead of our CT model.
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

from src.baselines.patchtst import (
    PatchTSTWrapper, pack_observations,
)
from transformers import PatchTSTConfig
from src.data.preprocessing import mad_normalize, median_match
from astrotools.ztf import read_matchfile_lcs_batched

CHEN_PATH = os.environ.get("ZTF_CHEN_HDF5", str(REPO / "data" / "chen2020.hdf5"))
SUBSET_PATH = str(REPO / "results/chen2020_25k/subset_indices.npz")
CHECKPOINT = str(REPO / "checkpoints/patchtst_v3scale/best_model.pt")
OUTPUT = str(REPO / "results/chen2020_25k/embeddings_patchtst_v3scale.npz")


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
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default=SUBSET_PATH)
    ap.add_argument("--checkpoint", default=CHECKPOINT)
    ap.add_argument("--output", default=OUTPUT)
    ap.add_argument("--context_length", type=int, default=384)
    ap.add_argument("--patch_length", type=int, default=16)
    ap.add_argument("--patch_stride", type=int, default=16)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--num_layers", type=int, default=3)
    ap.add_argument("--dim_feedforward", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=64)
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
    group_to_rows: dict = {}
    for ri, k in enumerate(keys):
        group_to_rows.setdefault(k, []).append(ri)
    print(f"Groups: {len(unique_groups):,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    config = PatchTSTConfig(
        num_input_channels=3,
        context_length=args.context_length,
        patch_length=args.patch_length,
        patch_stride=args.patch_stride,
        d_model=args.d_model,
        num_attention_heads=args.nhead,
        num_hidden_layers=args.num_layers,
        ffn_dim=args.dim_feedforward,
        do_mask_input=False,
    )
    wrapper = PatchTSTWrapper(config).to(device).eval()
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sd = state.get("model_state_dict", state)
    # Pretraining checkpoint has a different head; load only the encoder.
    encoder_sd = {k: v for k, v in sd.items()
                   if k in wrapper.state_dict() or k.startswith("model.")}
    missing, unexpected = wrapper.load_state_dict(encoder_sd, strict=False)
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"  missing keys: {len(missing)}, unexpected: {len(unexpected)}")

    N = len(chen_indices)
    z_sig = np.full((N, args.d_model), np.nan, dtype=np.float32)
    success = np.zeros(N, dtype=bool)
    n_obs = np.zeros(N, dtype=np.int32)

    pending_packs, pending_rows = [], []
    def flush():
        if not pending_packs: return
        x = torch.from_numpy(np.stack(pending_packs)).to(device)
        emb = wrapper(x).cpu().numpy()
        for ri, e in zip(pending_rows, emb):
            z_sig[ri] = e
            success[ri] = True

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
            packed, mask = pack_observations(
                lc["time"], lc["flux"], lc["flux_err"],
                max_len=args.context_length,
            )
            n_obs[ri] = int(mask.sum())
            pending_packs.append(packed)
            pending_rows.append(ri)
            if len(pending_packs) >= args.batch_size:
                flush(); pending_packs.clear(); pending_rows.clear()
        n_done += len(rows)
        if (gi + 1) % 1000 == 0 or gi == len(unique_groups) - 1:
            el = time.time() - t0
            r = n_done / max(el, 1e-9)
            eta = (N - n_done) / max(r, 1e-9)
            print(f"  group {gi+1:>5}/{len(unique_groups)} sources {n_done:,}/{N:,} "
                  f"({r:.0f}/s, ~{eta/60:.1f} min)")
    flush()

    print(f"\n{success.sum():,}/{N:,} succeeded · {(time.time()-t0)/60:.1f} min")
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    # z_qual is meaningless for PatchTST (no quality head) but probe_25k.py
    # only reads z_sig, so we can fill with zeros for schema parity.
    np.savez(out, z_sig=z_sig,
             z_qual=np.zeros((N, 32), dtype=np.float32),
             n_obs=n_obs, success_mask=success, chen_indices=chen_indices)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
