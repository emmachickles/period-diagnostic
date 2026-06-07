#!/usr/bin/env python3
"""
cache_25k_lcs.py — one-time cache of the 25K Chen subset's preprocessed
light curves to a local npz, so downstream FM-baseline embed scripts
don't each pay the SSHFS-read cost.

Output: results/chen2020_25k/lc_cache_25k.npz with:
  times       (N, max_len)  float32
  fluxes      (N, max_len)  float32
  flux_errs   (N, max_len)  float32
  masks       (N, max_len)  bool
  n_obs       (N,)          int32
  chen_indices(N,)          int64
  success     (N,)          bool
"""
import os
import sys, time
from pathlib import Path
import h5py
import numpy as np

REPO = Path(os.environ.get("PERIOD_DIAG_ROOT", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(REPO))
_ASTROTOOLS = os.environ.get("ASTROTOOLS_PATH")
if _ASTROTOOLS:  # local helper, only needed for raw matchfile ingestion
    sys.path.insert(0, _ASTROTOOLS)

from src.data.preprocessing import mad_normalize, median_match
from astrotools.ztf import read_matchfile_lcs_batched

CHEN_PATH = os.environ.get("ZTF_CHEN_HDF5", str(REPO / "data" / "chen2020.hdf5"))
SUBSET_PATH = REPO / "results/chen2020_25k/subset_indices.npz"
OUTPUT = REPO / "results/chen2020_25k/lc_cache_25k.npz"
MAX_LEN = 1024  # large enough that we can truncate later per FM


def fetch_lcs(field, ccd, quad, ra, dec, row_keys, radius=3.0, min_epochs=10):
    raw = read_matchfile_lcs_batched(
        field=field, ccd=ccd, quad=quad,
        ra_targets=ra, dec_targets=dec, filters="gr", match_radius_arcsec=radius,
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


def main():
    sub = np.load(SUBSET_PATH)
    chen_indices = sub["indices"]
    N = len(chen_indices)
    print(f"Subset: {N:,}")

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
    print(f"Unique tiles: {len(unique_groups):,}")

    times = np.zeros((N, MAX_LEN), dtype=np.float32)
    fluxes = np.zeros((N, MAX_LEN), dtype=np.float32)
    flux_errs = np.ones((N, MAX_LEN), dtype=np.float32)
    masks = np.zeros((N, MAX_LEN), dtype=bool)
    n_obs = np.zeros(N, dtype=np.int32)
    success = np.zeros(N, dtype=bool)

    t0 = time.time(); n_done = 0
    for gi, (field, ccd, quad) in enumerate(unique_groups):
        rows = np.array(group_to_rows[(field, ccd, quad)])
        try:
            lcs_dict = fetch_lcs(int(field), int(ccd), int(quad),
                                 ra[rows], dec[rows], rows)
        except Exception:
            n_done += len(rows); continue
        for ri in rows:
            lc = lcs_dict.get(int(ri))
            if lc is None: continue
            t, f, e = lc["time"], lc["flux"], lc["flux_err"]
            L = min(len(t), MAX_LEN)
            if len(t) > MAX_LEN:
                # Center-crop
                s = (len(t) - MAX_LEN) // 2
                t = t[s:s+MAX_LEN]; f = f[s:s+MAX_LEN]; e = e[s:s+MAX_LEN]
                L = MAX_LEN
            times[ri, :L] = t.astype(np.float32)
            fluxes[ri, :L] = f.astype(np.float32)
            flux_errs[ri, :L] = e.astype(np.float32)
            masks[ri, :L] = True
            n_obs[ri] = L
            success[ri] = True
        n_done += len(rows)
        if (gi + 1) % 1000 == 0 or gi == len(unique_groups) - 1:
            el = time.time() - t0
            r = n_done / max(el, 1e-9)
            eta = (N - n_done) / max(r, 1e-9)
            print(f"  group {gi+1:>5}/{len(unique_groups)} sources {n_done:,}/{N:,} "
                  f"({r:.0f}/s, ~{eta/60:.1f} min)", flush=True)

    print(f"\n{success.sum():,}/{N:,} succeeded · {(time.time()-t0)/60:.1f} min")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUTPUT,
             times=times, fluxes=fluxes, flux_errs=flux_errs, masks=masks,
             n_obs=n_obs, success=success, chen_indices=chen_indices)
    print(f"Saved {OUTPUT} ({OUTPUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
