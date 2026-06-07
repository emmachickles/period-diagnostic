"""
Data augmentations for ZTF light curves.

Standard augmentations (preserve astrophysical signal):
  - Time shift
  - Amplitude scaling
  - Gaussian noise injection
  - Random point dropout

Noise-equalization augmentation (for disentangled z_sig training):
  - Rescale flux errors to common noise floor
  - Forces z_sig to be noise-invariant across contrastive views
"""

import numpy as np
from typing import Tuple


def augment_light_curve(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply random augmentations to a light curve.

    Returns augmented (time, flux, flux_err).
    """
    time = time.copy()
    flux = flux.copy()
    flux_err = flux_err.copy()

    # Time shift (doesn't affect period structure)
    if np.random.random() < 0.5:
        time = time + np.random.uniform(-100, 100)

    # Amplitude scaling around median
    if np.random.random() < 0.5:
        scale = np.random.uniform(0.8, 1.2)
        med = np.median(flux)
        flux = med + (flux - med) * scale
        flux_err = flux_err * scale

    # Additive Gaussian noise
    if np.random.random() < 0.3:
        noise_scale = np.random.uniform(0.01, 0.05) * np.std(flux)
        flux = flux + np.random.normal(0, noise_scale, size=len(flux))

    # Random point dropout
    if np.random.random() < 0.2:
        keep_frac = np.random.uniform(0.85, 0.95)
        n_keep = max(int(len(flux) * keep_frac), 10)
        idx = np.sort(np.random.choice(len(flux), size=n_keep, replace=False))
        time = time[idx]
        flux = flux[idx]
        flux_err = flux_err[idx]

    return time, flux, flux_err


def noise_equalize(
    flux: np.ndarray,
    flux_err: np.ndarray,
    target_noise_level: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-point noise equalization for contrastive augmentation.

    Adds Gaussian noise per-point so that each point's effective
    uncertainty reaches ``target_noise_level``.  Points whose error
    already exceeds the target are left unchanged (we can only add
    noise, not remove it).  This preserves the heteroscedastic
    structure of real ZTF data.

    Parameters
    ----------
    flux : (N,)
        Flux values.
    flux_err : (N,)
        Original per-point flux uncertainties.
    target_noise_level : float
        Target uncertainty level.

    Returns
    -------
    flux_aug : (N,)
        Flux with per-point noise injected.
    flux_err_aug : (N,)
        Updated per-point uncertainties.
    """
    if np.median(flux_err) < 1e-10:
        return flux.copy(), flux_err.copy()

    # Per-point extra variance needed to reach target; zero when already noisier
    extra_var = np.maximum(target_noise_level**2 - flux_err**2, 0.0)
    extra_sigma = np.sqrt(extra_var)

    noise = np.random.normal(0, 1, size=len(flux)) * extra_sigma
    flux_aug = flux + noise
    flux_err_aug = np.sqrt(flux_err**2 + extra_var)

    return flux_aug, flux_err_aug


def disjoint_window_pair(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    window_frac: float = 0.7,
    min_epochs: int = 12,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray],
           Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Pick two random contiguous-epoch windows of the same source.

    Each window covers ``window_frac × N`` consecutive epochs (in time-sorted
    order). The two starting offsets are drawn independently in
    ``[0, N - win_len]`` so the views share a guaranteed minimum overlap of
    ``(2·window_frac - 1)·N`` epochs (40 % when window_frac=0.7), but
    differ in the rest.

    Why: under SimCLR-style two-view augmentation, both contrastive views
    sharing identical timestamps lets the model identify the positive pair
    by cadence pattern alone — a shortcut that defeats the point of SSL
    when the architecture (e.g. continuous-time attention) has direct
    access to Δt at every layer. Disjoint windows break that shortcut
    without fabricating data: every flux value is a real observation.

    Sequences shorter than ``min_epochs`` fall through unchanged.
    """
    N = len(time)
    if N < min_epochs:
        full = (time, flux, flux_err)
        return full, full
    win_len = max(int(window_frac * N), 10)
    max_start = N - win_len
    if max_start <= 0:
        full = (time, flux, flux_err)
        return full, full
    s1 = np.random.randint(0, max_start + 1)
    s2 = np.random.randint(0, max_start + 1)
    return (
        (time[s1:s1 + win_len], flux[s1:s1 + win_len], flux_err[s1:s1 + win_len]),
        (time[s2:s2 + win_len], flux[s2:s2 + win_len], flux_err[s2:s2 + win_len]),
    )


def create_contrastive_pair(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    noise_equalize_prob: float = 0.5,
    use_disjoint_windows: bool = True,
    window_frac: float = 0.7,
) -> Tuple[dict, dict]:
    """
    Create two augmented views for contrastive learning.

    Pipeline (when ``use_disjoint_windows=True``):
        1. Pick two random contiguous epoch-windows from the source.
        2. Per-view ``augment_light_curve`` (time shift, amp scale, dropout).
        3. View 2 gets optional noise equalization.

    With ``use_disjoint_windows=False`` the windowing step is skipped and
    behavior matches the pre-2026 pipeline (both views start from the full
    sequence).
    """
    if use_disjoint_windows:
        (t_w1, f_w1, e_w1), (t_w2, f_w2, e_w2) = disjoint_window_pair(
            time, flux, flux_err, window_frac=window_frac,
        )
    else:
        t_w1, f_w1, e_w1 = time, flux, flux_err
        t_w2, f_w2, e_w2 = time, flux, flux_err

    t1, f1, e1 = augment_light_curve(t_w1, f_w1, e_w1)
    t2, f2, e2 = augment_light_curve(t_w2, f_w2, e_w2)

    if np.random.random() < noise_equalize_prob:
        log_min = np.log10(max(np.median(flux_err) * 0.1, 1e-6))
        log_max = np.log10(max(np.median(flux_err) * 10.0, 1e-4))
        target = 10 ** np.random.uniform(log_min, log_max)
        f2, e2 = noise_equalize(f2, e2, target)

    view1 = {"time": t1, "flux": f1, "flux_err": e1}
    view2 = {"time": t2, "flux": f2, "flux_err": e2}
    return view1, view2
