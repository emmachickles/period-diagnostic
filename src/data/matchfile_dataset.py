"""
PyTorch IterableDataset for streaming ZTF light curves from HDF5 matchfiles.

Index-free: takes a matchfile directory, globs all files, and iterates over
sources directly.  No pre-built index required — the filesystem structure
IS the index.

Each DataLoader worker opens its own h5 file handles (h5py is not thread-safe).

Supports multi-band loading: when multiband=True, reads g/r/i observations
for each source (deriving band paths from the primary r-band file),
MAD-normalizes each band independently, then concatenates and sorts by time.
"""

import glob
import os
import re

import numpy as np
import h5py
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from typing import Optional, Dict, Tuple

from .augmentations import augment_light_curve, create_contrastive_pair
from .preprocessing import calibrate_flux, filter_good, mad_normalize, median_match

# Regex to swap band suffix in matchfile paths: data_{field}_{ccd}_{quad}_{band}.h5
_BAND_RE = re.compile(r"(data_\d{4}_\d{2}_\d_)z[gri](\.h5)")
_ALL_BANDS = ["zg", "zr", "zi"]


class ZTFMatchfileDataset(IterableDataset):
    """
    Streaming dataset that reads ZTF light curves directly from matchfiles.

    No pre-built index needed — just point at the matchfile directory.

    Parameters
    ----------
    matchfile_dir : str
        Root directory containing field subdirectories with HDF5 matchfiles.
    max_seq_len : int
        Maximum sequence length (cap with random windowing for longer).
    contrastive : bool
        If True, return two augmented views per sample.
    augment : bool
        If True, apply augmentations (required for contrastive=True).
    noise_equalize_prob : float
        Probability of noise equalization in contrastive view 2.
    max_sources : int, optional
        If set, limit to this many sources total (for debugging / small runs).
    shuffle : bool
        If True, shuffle file and source order each epoch.
    multiband : bool
        If True, combine g/r/i observations for each source.
    primary_band : str
        Band used for file discovery and source enumeration.
    min_epochs : int
        Minimum good observations per source (per-band).
    """

    def __init__(
        self,
        matchfile_dir: str,
        max_seq_len: int = 768,
        contrastive: bool = True,
        augment: bool = True,
        noise_equalize_prob: float = 0.5,
        max_sources: Optional[int] = None,
        shuffle: bool = True,
        multiband: bool = True,
        primary_band: str = "zr",
        min_epochs: int = 10,
        holdout_path: Optional[str] = None,
        use_disjoint_windows: bool = True,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.contrastive = contrastive
        self.augment = augment
        self.noise_equalize_prob = noise_equalize_prob
        self.max_sources = max_sources
        self.shuffle = shuffle
        self.multiband = multiband
        self.primary_band = primary_band
        self.min_epochs = min_epochs
        self.use_disjoint_windows = use_disjoint_windows

        # Discover all primary-band matchfiles
        pattern = os.path.join(matchfile_dir, "*", f"data_*_{primary_band}.h5")
        self._files = sorted(glob.glob(pattern))
        if len(self._files) == 0:
            raise FileNotFoundError(
                f"No matchfiles found matching {pattern}"
            )

        # Per-worker h5 file handle cache
        self._h5_cache: Dict[str, h5py.File] = {}

        # Holdout: per-tile (ra, dec) of sources we must NOT train on
        # (e.g. the 25K Chen demo subset used for probe accuracy).
        self._holdout_per_tile: Dict[Tuple[int, int, int], np.ndarray] = {}
        self._holdout_match_radius_deg: float = 0.0
        if holdout_path is not None:
            self._load_holdout(holdout_path)

    def _load_holdout(self, path: str) -> None:
        """Group holdout (ra, dec) into a per-tile dict for O(1) lookup."""
        npz = np.load(path)
        match_arcsec = float(npz["match_arcsec"])
        self._holdout_match_radius_deg = match_arcsec / 3600.0
        for f, c, q, ra, dec in zip(npz["field"], npz["ccd"], npz["quad"],
                                    npz["ra"], npz["dec"]):
            key = (int(f), int(c), int(q))
            self._holdout_per_tile.setdefault(key, []).append((float(ra), float(dec)))
        # Convert lists to ndarray for vectorized per-tile distance check
        self._holdout_per_tile = {
            k: np.asarray(v, dtype=np.float64)
            for k, v in self._holdout_per_tile.items()
        }

    def _holdout_mask_for_file(self, h5_path: str) -> Optional[np.ndarray]:
        """Return a boolean mask of length n_sources marking held-out rows.

        None if no holdout configured. Match is RA/Dec within
        match_arcsec, computed once per file (cached).
        """
        if not self._holdout_per_tile:
            return None
        cache_key = ("_holdout_mask", h5_path)
        if cache_key in self._h5_cache:
            return self._h5_cache[cache_key]
        # Parse (field, ccd, quad) from the filename
        m = re.search(r"data_(\d{4})_(\d{2})_(\d)_z[gri]\.h5$", h5_path)
        if not m:
            self._h5_cache[cache_key] = None
            return None
        tile = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        chen_radec = self._holdout_per_tile.get(tile)
        if chen_radec is None:
            self._h5_cache[cache_key] = None
            return None
        try:
            f = self._get_h5(h5_path)
            ra = f["data"]["sources"]["ra"][:].astype(np.float64)
            dec = f["data"]["sources"]["decl"][:].astype(np.float64)
        except Exception:
            self._h5_cache[cache_key] = None
            return None
        # Small-angle distance with cos(dec) correction at the tile median
        cosdec = float(np.cos(np.deg2rad(np.median(dec))))
        mask = np.zeros(len(ra), dtype=bool)
        r_deg = self._holdout_match_radius_deg
        for chen_ra, chen_dec in chen_radec:
            d2 = ((ra - chen_ra) * cosdec) ** 2 + (dec - chen_dec) ** 2
            mask |= d2 < r_deg ** 2
        self._h5_cache[cache_key] = mask
        return mask

    def _get_h5(self, path: str) -> h5py.File:
        """Get or open an h5 file handle (cached per worker)."""
        if path not in self._h5_cache:
            self._h5_cache[path] = h5py.File(path, "r")
        return self._h5_cache[path]

    def _read_single_band_raw(self, h5_path: str, src_idx: int
                              ) -> Optional[Dict[str, np.ndarray]]:
        """Read calibrated flux for one band without normalization."""
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

            return {
                "time": bjd[good],
                "flux": cal_flux[good],
                "flux_err": diff_ferr[good],
            }

        except Exception:
            return None

    def _find_src_idx_by_gaia_id(self, h5_path: str, gaia_id: int
                                 ) -> Optional[int]:
        """Find a source's index in a matchfile by its gaia_id (= ps_id).

        Uses a per-file cached lookup dict for O(1) access after first load.
        """
        cache_key = ("_gid_map", h5_path)
        if cache_key not in self._h5_cache:
            try:
                f = self._get_h5(h5_path)
                gids = f["data"]["sources"]["gaia_id"][:]
                self._h5_cache[cache_key] = {
                    int(g): i for i, g in enumerate(gids)
                }
            except Exception:
                self._h5_cache[cache_key] = None
                return None
        lookup = self._h5_cache.get(cache_key)
        if lookup is None:
            return None
        return lookup.get(gaia_id)

    @staticmethod
    def _swap_band(h5_path: str, new_band: str) -> str:
        """Swap the band suffix in a matchfile path."""
        return _BAND_RE.sub(rf"\g<1>{new_band}\2", h5_path)

    def _read_source(self, h5_path: str, src_idx: int
                     ) -> Optional[Dict[str, np.ndarray]]:
        """Read a source's light curve, optionally combining all bands.

        Multi-band mode: reads calibrated flux from each band, median-matches
        to align the DC offsets, concatenates, then MAD-normalizes the
        combined light curve once.  This preserves the relative variability
        amplitude across bands and uses all available epochs.
        """
        if not self.multiband:
            lc = self._read_single_band_raw(h5_path, src_idx)
            if lc is None:
                return None
            flux_norm, err_norm, _, _ = mad_normalize(lc["flux"], lc["flux_err"])
            if flux_norm is None:
                return None
            return {
                "time": lc["time"],
                "flux": flux_norm,
                "flux_err": err_norm,
                "median_sigma": float(np.median(err_norm)),
                "n_obs": len(lc["time"]),
                "baseline": float(lc["time"].max() - lc["time"].min()),
            }

        # Multi-band: look up gaia_id, then find in each band's file
        try:
            f = self._get_h5(h5_path)
            gaia_id = int(f["data"]["sources"][src_idx]["gaia_id"])
        except Exception:
            return None

        band_data = []
        for band in _ALL_BANDS:
            band_path = self._swap_band(h5_path, band)
            if not os.path.exists(band_path):
                continue

            band_src_idx = self._find_src_idx_by_gaia_id(band_path, gaia_id)
            if band_src_idx is None:
                continue

            lc_band = self._read_single_band_raw(band_path, band_src_idx)
            if lc_band is not None:
                band_data.append(lc_band)

        if len(band_data) == 0:
            return None

        # Median-match across bands, then MAD-normalize combined light curve
        t_combined, f_combined, e_combined = median_match(band_data)

        flux_norm, err_norm, _, _ = mad_normalize(f_combined, e_combined)
        if flux_norm is None:
            return None

        return {
            "time": t_combined,
            "flux": flux_norm,
            "flux_err": err_norm,
            "median_sigma": float(np.median(err_norm)),
            "n_obs": len(t_combined),
            "baseline": float(t_combined.max() - t_combined.min()),
        }

    def _pack_sequence(
        self, time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Cap at max_seq_len with random windowing; pad short sequences."""
        seq_len = len(time)

        if seq_len > self.max_seq_len:
            start = np.random.randint(0, seq_len - self.max_seq_len + 1)
            end = start + self.max_seq_len
            time = time[start:end]
            flux = flux[start:end]
            flux_err = flux_err[start:end]
            mask = np.ones(self.max_seq_len, dtype=bool)
        else:
            mask = np.zeros(self.max_seq_len, dtype=bool)
            mask[:seq_len] = True

            t_pad = np.zeros(self.max_seq_len, dtype=np.float64)
            f_pad = np.zeros(self.max_seq_len, dtype=np.float64)
            e_pad = np.ones(self.max_seq_len, dtype=np.float64)

            t_pad[:seq_len] = time
            f_pad[:seq_len] = flux
            e_pad[:seq_len] = flux_err

            time, flux, flux_err = t_pad, f_pad, e_pad

        return time, flux, flux_err, mask

    def _make_sample(self, lc: dict) -> dict:
        """Convert raw light curve to a training sample dict of tensors."""
        time = lc["time"]
        flux = lc["flux"]
        flux_err = lc["flux_err"]

        if self.contrastive:
            view1, view2 = create_contrastive_pair(
                time, flux, flux_err,
                noise_equalize_prob=self.noise_equalize_prob,
                use_disjoint_windows=self.use_disjoint_windows,
            )

            t1, f1, e1, m1 = self._pack_sequence(
                view1["time"], view1["flux"], view1["flux_err"]
            )
            t2, f2, e2, m2 = self._pack_sequence(
                view2["time"], view2["flux"], view2["flux_err"]
            )

            return {
                "time_1": torch.from_numpy(t1.astype(np.float32)),
                "flux_1": torch.from_numpy(f1.astype(np.float32)),
                "flux_err_1": torch.from_numpy(e1.astype(np.float32)),
                "mask_1": torch.from_numpy(m1),
                "time_2": torch.from_numpy(t2.astype(np.float32)),
                "flux_2": torch.from_numpy(f2.astype(np.float32)),
                "flux_err_2": torch.from_numpy(e2.astype(np.float32)),
                "mask_2": torch.from_numpy(m2),
                "log_median_sigma": torch.tensor(
                    np.log10(max(lc["median_sigma"], 1e-8)), dtype=torch.float32
                ),
            }
        else:
            if self.augment:
                time, flux, flux_err = augment_light_curve(time, flux, flux_err)

            time, flux, flux_err, mask = self._pack_sequence(time, flux, flux_err)

            return {
                "time": torch.from_numpy(time.astype(np.float32)),
                "flux": torch.from_numpy(flux.astype(np.float32)),
                "flux_err": torch.from_numpy(flux_err.astype(np.float32)),
                "mask": torch.from_numpy(mask),
                "log_median_sigma": torch.tensor(
                    np.log10(max(lc["median_sigma"], 1e-8)), dtype=torch.float32
                ),
            }

    def __iter__(self):
        """Iterate over all sources in all matchfiles, sharding by file."""
        worker_info = get_worker_info()

        files = list(self._files)
        if self.shuffle:
            np.random.shuffle(files)

        # Shard files across workers (not sources — avoids duplicate file opens)
        if worker_info is not None:
            files = files[worker_info.id :: worker_info.num_workers]

        # Divide max_sources budget across workers
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

            holdout_mask = self._holdout_mask_for_file(h5_path)
            if holdout_mask is not None:
                src_indices = np.flatnonzero(~holdout_mask)
            else:
                src_indices = np.arange(n_src)
            if self.shuffle:
                np.random.shuffle(src_indices)

            for src_idx in src_indices:
                if budget is not None and n_yielded >= budget:
                    break

                lc = self._read_source(h5_path, int(src_idx))
                if lc is None:
                    continue

                sample = self._make_sample(lc)
                if sample is not None:
                    yield sample
                    n_yielded += 1

    def __del__(self):
        """Close cached h5 file handles."""
        for key, val in self._h5_cache.items():
            if isinstance(val, h5py.File):
                try:
                    val.close()
                except Exception:
                    pass


def create_dataloader(
    matchfile_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    max_seq_len: int = 768,
    contrastive: bool = True,
    max_sources: Optional[int] = None,
    **kwargs,
) -> DataLoader:
    """Create a DataLoader for ZTF matchfile streaming."""
    dataset = ZTFMatchfileDataset(
        matchfile_dir=matchfile_dir,
        max_seq_len=max_seq_len,
        contrastive=contrastive,
        max_sources=max_sources,
        **kwargs,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
