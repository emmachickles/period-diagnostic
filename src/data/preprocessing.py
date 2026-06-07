"""
Centralized preprocessing functions for ZTF light curves.

All code paths — training, evaluation, plotting — should import from
this module to guarantee identical preprocessing.
"""

import numpy as np


def calibrate_flux(diff_flux, mag_ref):
    """Convert differential flux to calibrated flux.

    Parameters
    ----------
    diff_flux : ndarray
        Differential flux from the matchfile sourcedata.
    mag_ref : float
        Reference magnitude from the matchfile sources table.

    Returns
    -------
    cal_flux : ndarray
        Calibrated flux (diff_flux + ref_flux).
    ref_flux : float
        Reference flux (10^(-0.4 * mag_ref)).
    """
    ref_flux = 10.0 ** (-0.4 * mag_ref)
    cal_flux = diff_flux + ref_flux
    return cal_flux, ref_flux


def filter_good(cal_flux, diff_flux, flux_err, flag):
    """Boolean mask for good observations.

    Keeps points with flag==0, positive calibrated flux, and finite
    values in both the differential flux and flux error.

    Parameters
    ----------
    cal_flux : ndarray
        Calibrated flux.
    diff_flux : ndarray
        Raw differential flux.
    flux_err : ndarray
        Flux error.
    flag : ndarray
        Quality flag (0 = good).

    Returns
    -------
    mask : ndarray of bool
        True for good observations.
    """
    return (
        (flag == 0)
        & (cal_flux > 0)
        & np.isfinite(diff_flux)
        & np.isfinite(flux_err)
    )


def clip_outliers(flux, threshold=10.0):
    """Boolean mask keeping points within threshold × MAD of the median.

    Uses median absolute deviation (MAD) as a robust scale estimator,
    consistent with the MAD normalization used throughout the pipeline.

    Parameters
    ----------
    flux : ndarray
        Flux values (calibrated or normalized).
    threshold : float
        Number of MADs from the median to clip at (default 10).

    Returns
    -------
    mask : ndarray of bool
        True for points to keep.
    """
    median = np.median(flux)
    mad = np.median(np.abs(flux - median))
    if mad < 1e-10:
        return np.ones(len(flux), dtype=bool)
    return np.abs(flux - median) < threshold * mad


def median_match(band_data):
    """Median-match multi-band light curves before combining.

    Shifts each band's calibrated flux so that all bands share a common
    median (zero), then concatenates and sorts by time.  This preserves
    the relative variability amplitude across bands while removing the
    DC offset from different reference magnitudes.

    Parameters
    ----------
    band_data : list of dict
        Each dict must have keys 'time', 'flux', 'flux_err' (ndarrays).

    Returns
    -------
    time : ndarray
        Combined timestamps, sorted.
    flux : ndarray
        Median-matched combined flux (median ≈ 0).
    flux_err : ndarray
        Combined flux errors.
    """
    all_t, all_f, all_e = [], [], []
    for bd in band_data:
        med = np.median(bd['flux'])
        all_t.append(bd['time'])
        all_f.append(bd['flux'] - med)
        all_e.append(bd['flux_err'])

    t_cat = np.concatenate(all_t)
    f_cat = np.concatenate(all_f)
    e_cat = np.concatenate(all_e)
    order = np.argsort(t_cat)
    return t_cat[order], f_cat[order], e_cat[order]


def mad_normalize(flux, flux_err):
    """MAD-normalize flux values.

    Subtracts the median and divides by the median absolute deviation.
    Falls back to standard deviation if MAD is near zero.

    Parameters
    ----------
    flux : ndarray
        Flux values to normalize.
    flux_err : ndarray
        Flux error values.

    Returns
    -------
    flux_norm : ndarray or None
        Normalized flux, or None if the scale is degenerate.
    flux_err_norm : ndarray or None
        Normalized flux error, or None if the scale is degenerate.
    median : float
        Median flux.
    mad : float
        MAD (or std fallback) used as the scale.
    """
    median = np.median(flux)
    mad = np.median(np.abs(flux - median))
    if mad < 1e-10:
        mad = np.std(flux)
    if mad < 1e-10:
        return None, None, median, mad
    return (flux - median) / mad, flux_err / mad, median, mad
