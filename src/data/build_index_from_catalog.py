"""
Build a compact index from pre-existing ZTF Lomb-Scargle catalog files.

Much faster than scanning all HDF5 files directly: reads small text .result
files to get (field, ccd, quad, ps_id), then does a single HDF5 open per
file to map ps_id -> source index and extract median flux error.

Usage:
    python src/data/build_index_from_catalog.py \
        --catalog_dir /ztf/catalogs/lomb_scargle \
        --matchfile_dir /path/to/ZTF/matchfiles \
        --output index_catalog.npz
"""

import argparse
import glob
import os
import re
import time
import numpy as np
import h5py
from multiprocessing import Pool
from pathlib import Path


# Columns in the LS .result files
LS_COLUMNS = [
    'ra', 'dec', 'ps_id', 'period', 'significance',
    'significance_half', 'significance_twice',
    'mad', 'variance', 'skew', 'kurtosis',
    'ref_flux_flag', 'ref_flux',
]

# Regex to parse field/ccd/quad from filename: "0245_01_1.result"
_FNAME_RE = re.compile(r"(\d{4})_(\d{2})_(\d).result")


def process_one_file(args):
    """
    Process one .result file: read catalog, open corresponding HDF5,
    match ps_id -> source index, compute median flux error.

    Returns dict with numpy arrays or None.
    """
    result_path, matchfile_dir, filter_band, min_epochs = args

    # Parse field/ccd/quad from filename
    fname = os.path.basename(result_path)
    m = _FNAME_RE.match(fname)
    if m is None:
        return None

    field, ccd, quad = m.group(1), m.group(2), m.group(3)
    h5_path = os.path.join(
        matchfile_dir, field,
        f"data_{field}_{ccd}_{quad}_{filter_band}.h5"
    )

    if not os.path.exists(h5_path):
        return None

    try:
        # Read catalog: just need ps_id column (col 2), ra (col 0), dec (col 1)
        data = np.loadtxt(result_path, usecols=(0, 1, 2), dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        cat_ra = data[:, 0]
        cat_dec = data[:, 1]
        cat_psid = data[:, 2].astype(np.int64)

        # Open HDF5 and build gaia_id -> index mapping
        with h5py.File(h5_path, "r") as f:
            if "data" not in f or "sources" not in f["data"]:
                return None

            sources = f["data"]["sources"][:]
            h5_gaia_ids = sources["gaia_id"]

            exposures = f["data"]["exposures"][:]
            n_exp = len(exposures)
            if n_exp == 0:
                return None

            # Build lookup: gaia_id -> index in HDF5
            id_to_idx = {}
            for i, gid in enumerate(h5_gaia_ids):
                id_to_idx[int(gid)] = i

            # Match catalog sources to HDF5 indices
            matched_idx = []
            matched_cat_pos = []
            for cat_i, psid in enumerate(cat_psid):
                h5_idx = id_to_idx.get(int(psid))
                if h5_idx is not None:
                    matched_idx.append(h5_idx)
                    matched_cat_pos.append(cat_i)

            if len(matched_idx) == 0:
                return None

            matched_idx = np.array(matched_idx)
            matched_cat_pos = np.array(matched_cat_pos)

            # Read sourcedata to compute n_good and median_err
            all_sd = f["data"]["sourcedata"][:]

        n_src_total = len(sources)
        all_flags = all_sd["flag"].reshape(n_src_total, n_exp)
        all_ferr = all_sd["flux_err"].reshape(n_src_total, n_exp).astype(np.float64)

        # Vectorized for matched sources
        good_mask = all_flags[matched_idx] == 0  # (n_matched, n_exp)
        n_good_arr = good_mask.sum(axis=1)

        # Filter by min_epochs
        keep = n_good_arr >= min_epochs
        if keep.sum() == 0:
            return None

        matched_idx = matched_idx[keep]
        matched_cat_pos = matched_cat_pos[keep]
        good_mask = good_mask[keep]
        n_good_arr = n_good_arr[keep]

        # Compute median flux error for kept sources
        ferr_subset = all_ferr[matched_idx]  # (n_kept, n_exp)
        med_err = np.full(len(matched_idx), np.nan, dtype=np.float32)
        for i in range(len(matched_idx)):
            good_err = ferr_subset[i, good_mask[i]]
            finite = np.isfinite(good_err) & (good_err > 0)
            if finite.sum() > 0:
                med_err[i] = np.median(good_err[finite])

        valid = np.isfinite(med_err)
        if valid.sum() == 0:
            return None

        return {
            "h5_path": h5_path,
            "src_idx": matched_idx[valid].astype(np.int32),
            "n_good": n_good_arr[valid].astype(np.int32),
            "med_err": med_err[valid],
            "ra": cat_ra[matched_cat_pos[valid]],
            "dec": cat_dec[matched_cat_pos[valid]],
        }

    except Exception as e:
        print(f"  Warning: failed on {result_path}: {e}")
        return None


def build_index(catalog_dir, matchfile_dir, output_path,
                filter_band="zr", min_epochs=50, n_workers=8):
    """Build index from catalog .result files."""

    print(f"Scanning {catalog_dir} for .result files...")
    result_files = sorted(glob.glob(os.path.join(catalog_dir, "*.result")))
    print(f"Found {len(result_files):,} .result files")

    if len(result_files) == 0:
        print("No .result files found!")
        return

    t0 = time.time()
    args_list = [(f, matchfile_dir, filter_band, min_epochs) for f in result_files]

    # Accumulate numpy arrays
    chunk_path_idx = []
    chunk_src_idx = []
    chunk_n_good = []
    chunk_med_err = []
    chunk_ra = []
    chunk_dec = []
    total_sources = 0
    n_processed = 0

    # Build h5_path -> index mapping
    all_h5_paths = []
    path_to_idx = {}

    with Pool(n_workers) as pool:
        for result in pool.imap_unordered(process_one_file, args_list, chunksize=8):
            n_processed += 1

            if result is not None:
                h5_path = result["h5_path"]
                if h5_path not in path_to_idx:
                    path_to_idx[h5_path] = len(all_h5_paths)
                    all_h5_paths.append(h5_path)
                pidx = path_to_idx[h5_path]

                n = len(result["src_idx"])
                chunk_path_idx.append(np.full(n, pidx, dtype=np.int32))
                chunk_src_idx.append(result["src_idx"])
                chunk_n_good.append(result["n_good"])
                chunk_med_err.append(result["med_err"])
                chunk_ra.append(result["ra"])
                chunk_dec.append(result["dec"])
                total_sources += n

            if n_processed % 500 == 0:
                elapsed = time.time() - t0
                rate = n_processed / elapsed
                print(
                    f"  Processed {n_processed:,}/{len(result_files):,} files "
                    f"({rate:.1f} files/sec), "
                    f"{total_sources:,} sources so far"
                )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s. Total sources: {total_sources:,}")

    if total_sources == 0:
        print("No sources found!")
        return

    # Concatenate
    print("Concatenating arrays...")
    path_idx = np.concatenate(chunk_path_idx)
    src_idx = np.concatenate(chunk_src_idx)
    n_good = np.concatenate(chunk_n_good)
    med_err = np.concatenate(chunk_med_err)
    ra = np.concatenate(chunk_ra)
    dec = np.concatenate(chunk_dec)

    unique_paths = np.array(all_h5_paths, dtype="U256")

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Saving compressed index...")
    np.savez_compressed(
        output_path,
        unique_paths=unique_paths,
        path_idx=path_idx,
        src_idx=src_idx,
        n_good=n_good,
        med_err=med_err,
        ra=ra,
        dec=dec,
        min_epochs=np.array([min_epochs]),
        filter_band=np.array([filter_band]),
    )

    file_size = output_path.stat().st_size / 1e6
    print(f"Saved index to {output_path} ({file_size:.1f} MB)")
    print(f"  Sources: {total_sources:,}")
    print(f"  Unique h5 files: {len(unique_paths):,}")
    print(f"  Epoch range: {n_good.min()}-{n_good.max()} (median {np.median(n_good):.0f})")
    print(f"  Median error range: {med_err.min():.4f}-{med_err.max():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build ZTF index from Lomb-Scargle catalog files")
    parser.add_argument("--catalog_dir", type=str,
                        default="/ztf/catalogs/lomb_scargle")
    parser.add_argument("--matchfile_dir", type=str,
                        default="/ztf/matchfiles")
    parser.add_argument("--output", type=str, default="index_catalog.npz")
    parser.add_argument("--filter_band", type=str, default="zr",
                        choices=["zg", "zr", "zi"])
    parser.add_argument("--min_epochs", type=int, default=50)
    parser.add_argument("--n_workers", type=int, default=8)
    args = parser.parse_args()

    build_index(
        catalog_dir=args.catalog_dir,
        matchfile_dir=args.matchfile_dir,
        output_path=args.output,
        filter_band=args.filter_band,
        min_epochs=args.min_epochs,
        n_workers=args.n_workers,
    )
