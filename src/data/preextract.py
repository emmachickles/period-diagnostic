"""
Pre-extract light curves from matchfiles into a single NPZ for fast training.

Reads all sources from each HDF5 file at once (one read per file, not per source),
then saves to a compact NPZ that can be loaded entirely into RAM.

NOTE: This script currently extracts single-band data only.  For multi-band
median-matched training, use the streaming ZTFMatchfileDataset with
multiband=True (the default), which handles multi-band median matching
and MAD normalization automatically.

Usage:
    python src/data/preextract.py \
        --index_path index_catalog_dense500.npz \
        --matchfile_prefix /path/to/ZTF/matchfiles \
        --max_sources 100000 \
        --output preextracted_100k.npz
"""

import argparse
import sys
import time
import numpy as np
import h5py
from pathlib import Path
from collections import defaultdict

from src.data.preprocessing import calibrate_flux, filter_good, mad_normalize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", type=str, required=True)
    parser.add_argument("--matchfile_prefix", type=str, default=None)
    parser.add_argument("--max_sources", type=int, default=100000)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--output", type=str, default="preextracted.npz")
    args = parser.parse_args()

    # Load index
    print(f"Loading index from {args.index_path}...")
    idx = np.load(args.index_path, allow_pickle=True)
    unique_paths = idx["unique_paths"]
    path_idx = idx["path_idx"]
    src_indices = idx["src_idx"]
    n_total = len(src_indices)

    # Subsample if needed
    if args.max_sources and args.max_sources < n_total:
        rng = np.random.RandomState(42)
        sel = rng.choice(n_total, size=args.max_sources, replace=False)
        sel.sort()
    else:
        sel = np.arange(n_total)

    print(f"Will extract {len(sel):,} sources from {len(unique_paths)} files")

    # Group selected sources by file
    file_to_sources = defaultdict(list)
    for i in sel:
        file_to_sources[int(path_idx[i])].append((i, int(src_indices[i])))

    print(f"Sources spread across {len(file_to_sources)} files")

    # Remap paths if needed
    def remap(path):
        if args.matchfile_prefix:
            parts = path.split("/matchfiles/", 1)
            if len(parts) == 2:
                return args.matchfile_prefix.rstrip("/") + "/" + parts[1]
        return path

    # Extract file by file (one h5 open per file, read all sources at once)
    times_list = []
    flux_list = []
    err_list = []
    med_sigma_list = []

    t0 = time.time()
    count = 0
    failed_files = 0
    n_files = len(file_to_sources)

    for file_idx, (pidx, source_list) in enumerate(file_to_sources.items()):
        h5_path = remap(str(unique_paths[pidx]))

        try:
            with h5py.File(h5_path, "r") as f:
                sources = f["data"]["sources"][:]
                exposures = f["data"]["exposures"][:]
                n_exp = len(exposures)
                if n_exp == 0:
                    failed_files += 1
                    continue

                bjd = exposures["bjd"].astype(np.float64)

                # Read ALL sourcedata at once (single I/O op)
                all_sd = f["data"]["sourcedata"][:]

            n_src = len(sources)
            all_flags = all_sd["flag"].reshape(n_src, n_exp)
            all_flux = all_sd["flux"].reshape(n_src, n_exp).astype(np.float64)
            all_ferr = all_sd["flux_err"].reshape(n_src, n_exp).astype(np.float64)

            for _, src_idx in source_list:
                if src_idx >= n_src:
                    continue

                flag = all_flags[src_idx]
                diff_flux = all_flux[src_idx]
                diff_ferr = all_ferr[src_idx]

                mag_ref = float(sources[src_idx]["mag_ref"])
                cal_flux, _ = calibrate_flux(diff_flux, mag_ref)
                good = filter_good(cal_flux, diff_flux, diff_ferr, flag)
                if good.sum() < 10:
                    continue

                t = bjd[good]
                f_cal = cal_flux[good]
                f_err = diff_ferr[good]

                flux_norm, err_norm, _, _ = mad_normalize(f_cal, f_err)
                if flux_norm is None:
                    continue
                med_sigma = float(np.median(err_norm))

                # Truncate
                if len(t) > args.max_seq_len:
                    t = t[:args.max_seq_len]
                    flux_norm = flux_norm[:args.max_seq_len]
                    err_norm = err_norm[:args.max_seq_len]

                times_list.append(t.astype(np.float32))
                flux_list.append(flux_norm.astype(np.float32))
                err_list.append(err_norm.astype(np.float32))
                med_sigma_list.append(med_sigma)
                count += 1

        except Exception as e:
            failed_files += 1

        if (file_idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (file_idx + 1) / elapsed
            print(f"  Files: {file_idx+1:,}/{n_files:,} ({rate:.1f} files/sec), "
                  f"{count:,} sources extracted, {failed_files} failed files")

    elapsed = time.time() - t0
    print(f"\nExtracted {count:,} sources from {n_files} files in {elapsed:.0f}s "
          f"({failed_files} failed files)")

    # Save
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output,
        times=np.array(times_list, dtype=object),
        flux=np.array(flux_list, dtype=object),
        flux_err=np.array(err_list, dtype=object),
        median_sigma=np.array(med_sigma_list, dtype=np.float32),
    )

    fsize = output.stat().st_size / 1e6
    print(f"Saved to {output} ({fsize:.1f} MB)")


if __name__ == "__main__":
    main()
