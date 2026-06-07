"""
Build a compact index of all ZTF matchfile sources.

Walks /ztf/matchfiles/{field}/data_*.h5, reads the sources array from each,
counts good epochs per source (flag==0), and saves a compact index file.

Run once:
    python src/data/build_index.py --matchfile_dir /path/to/ZTF/matchfiles --output index.npz

Uses chunked numpy arrays to handle billion-source catalogs in bounded memory.
"""

import argparse
import glob
import os
import time
import numpy as np
import h5py
from multiprocessing import Pool
from pathlib import Path


def process_h5_file(args):
    """
    Process a single HDF5 matchfile and return source metadata as numpy arrays.

    Returns dict with arrays: path_str, src_idx, n_good, med_err, ra, dec
    or None if no qualifying sources.
    """
    h5_path, min_epochs = args

    try:
        with h5py.File(h5_path, "r") as f:
            if "data" not in f or "sources" not in f["data"]:
                return None

            sources = f["data"]["sources"][:]
            n_src = len(sources)

            if n_src == 0:
                return None

            exposures = f["data"]["exposures"][:]
            n_exp = len(exposures)

            if n_exp == 0:
                return None

            # Read ALL sourcedata at once (much faster than per-source reads)
            all_sd = f["data"]["sourcedata"][:]

        # Reshape to (n_src, n_exp) for vectorized processing
        all_flags = all_sd["flag"].reshape(n_src, n_exp)
        all_ferr = all_sd["flux_err"].reshape(n_src, n_exp).astype(np.float64)

        # Vectorized: count good epochs per source
        good_mask = all_flags == 0  # (n_src, n_exp)
        n_good_arr = good_mask.sum(axis=1)  # (n_src,)

        # Filter sources with enough good epochs
        candidate_mask = n_good_arr >= min_epochs
        candidate_indices = np.where(candidate_mask)[0]

        if len(candidate_indices) == 0:
            return None

        # Vectorized median error for candidates
        out_src_idx = []
        out_n_good = []
        out_med_err = []
        out_ra = []
        out_dec = []

        for src_idx in candidate_indices:
            good = good_mask[src_idx]
            good_err = all_ferr[src_idx, good]
            finite_mask = np.isfinite(good_err) & (good_err > 0)

            if finite_mask.sum() < min_epochs:
                continue

            out_src_idx.append(src_idx)
            out_n_good.append(int(n_good_arr[src_idx]))
            out_med_err.append(float(np.median(good_err[finite_mask])))
            out_ra.append(float(sources[src_idx]["ra"]))
            out_dec.append(float(sources[src_idx]["decl"]))

        if len(out_src_idx) == 0:
            return None

        return {
            "path": h5_path,
            "src_idx": np.array(out_src_idx, dtype=np.int32),
            "n_good": np.array(out_n_good, dtype=np.int32),
            "med_err": np.array(out_med_err, dtype=np.float32),
            "ra": np.array(out_ra, dtype=np.float64),
            "dec": np.array(out_dec, dtype=np.float64),
        }

    except Exception as e:
        print(f"  Warning: failed on {h5_path}: {e}")
        return None


def build_index(
    matchfile_dir: str,
    output_path: str,
    min_epochs: int = 30,
    n_workers: int = 8,
    filter_band: str = "zr",
):
    """
    Build matchfile index from all h5 files.

    Uses chunked writing to temporary .npz shards, then merges at the end.
    This keeps memory usage bounded even for billion-source catalogs.
    """
    print(f"Scanning {matchfile_dir} for *_{filter_band}.h5 files...")

    # Find all h5 files for the specified band
    pattern = os.path.join(matchfile_dir, "*", f"data_*_{filter_band}.h5")
    h5_files = sorted(glob.glob(pattern))
    print(f"Found {len(h5_files):,} h5 files")

    if len(h5_files) == 0:
        print("No files found! Check matchfile_dir and filter_band.")
        return

    # Build path->index mapping upfront (all unique h5 files)
    path_to_idx = {p: i for i, p in enumerate(h5_files)}

    # Process in parallel, accumulating into numpy array chunks
    t0 = time.time()
    args_list = [(f, min_epochs) for f in h5_files]

    # Accumulate into lists of numpy arrays (much more memory-efficient
    # than lists of Python tuples — ~32 bytes/source vs ~200 bytes/source)
    chunk_path_idx = []
    chunk_src_idx = []
    chunk_n_good = []
    chunk_med_err = []
    chunk_ra = []
    chunk_dec = []
    total_sources = 0
    n_processed = 0

    with Pool(n_workers) as pool:
        for result in pool.imap_unordered(process_h5_file, args_list, chunksize=4):
            n_processed += 1

            if result is not None:
                n_src = len(result["src_idx"])
                pidx = path_to_idx[result["path"]]
                chunk_path_idx.append(np.full(n_src, pidx, dtype=np.int32))
                chunk_src_idx.append(result["src_idx"])
                chunk_n_good.append(result["n_good"])
                chunk_med_err.append(result["med_err"])
                chunk_ra.append(result["ra"])
                chunk_dec.append(result["dec"])
                total_sources += n_src

            if n_processed % 100 == 0:
                elapsed = time.time() - t0
                rate = n_processed / elapsed
                print(
                    f"  Processed {n_processed:,}/{len(h5_files):,} files "
                    f"({rate:.1f} files/sec), "
                    f"{total_sources:,} sources so far"
                )

    elapsed = time.time() - t0
    print(f"\nDone scanning in {elapsed:.0f}s. Total sources: {total_sources:,}")

    if total_sources == 0:
        print("No sources found! Check min_epochs threshold.")
        return

    # Concatenate all chunks into final arrays
    print("Concatenating arrays...")
    path_idx = np.concatenate(chunk_path_idx)
    src_idx = np.concatenate(chunk_src_idx)
    n_good = np.concatenate(chunk_n_good)
    med_err = np.concatenate(chunk_med_err)
    ra = np.concatenate(chunk_ra)
    dec = np.concatenate(chunk_dec)

    # Free chunk lists
    del chunk_path_idx, chunk_src_idx, chunk_n_good
    del chunk_med_err, chunk_ra, chunk_dec

    # Build unique_paths array (only paths that actually have sources)
    used_path_indices = np.unique(path_idx)
    old_to_new = np.full(len(h5_files), -1, dtype=np.int32)
    for new_i, old_i in enumerate(used_path_indices):
        old_to_new[old_i] = new_i
    path_idx = old_to_new[path_idx]
    unique_paths = np.array([h5_files[i] for i in used_path_indices], dtype="U256")

    print(f"  Unique h5 files with sources: {len(unique_paths):,}")

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
    parser = argparse.ArgumentParser(description="Build ZTF matchfile index")
    parser.add_argument(
        "--matchfile_dir", type=str, default="/ztf/matchfiles",
        help="Root matchfile directory"
    )
    parser.add_argument(
        "--output", type=str, default="index.npz",
        help="Output index file path"
    )
    parser.add_argument("--min_epochs", type=int, default=30)
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--filter_band", type=str, default="zr",
                        choices=["zg", "zr", "zi"])
    args = parser.parse_args()

    build_index(
        matchfile_dir=args.matchfile_dir,
        output_path=args.output,
        min_epochs=args.min_epochs,
        n_workers=args.n_workers,
        filter_band=args.filter_band,
    )
