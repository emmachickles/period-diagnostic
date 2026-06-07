"""
Load ZTF light curves from HDF5 matchfiles by (ra, dec) or ps_id.

Adapted from /ztf/example/ztf_tools.py with corrected paths and
integrated into the ztf-ssl-transformer repository.

Usage
-----
    from src.data.ztf_loader import get_lightcurve, get_lightcurve_raw

    # Processed (BJD-converted, flag-filtered, clipped):
    lc = get_lightcurve(ra=100.903208, dec=3.307625, filters='gri')

    # Raw (everything from the matchfile, no filtering):
    raw = get_lightcurve_raw(ra=100.903208, dec=3.307625, filters='gri')
"""

import os
import glob

import numpy as np
import h5py

from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord
import astropy.units as u

from .preprocessing import calibrate_flux, filter_good, clip_outliers

# ── Paths ────────────────────────────────────────────────────────────

MATCHFILE_DIR = '/ztf/ztf_forced_photometry-main/test/matchfiles'

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

_FILTERS = {
    'g': {'suffix': 'zg', 'color': 'green'},
    'r': {'suffix': 'zr', 'color': 'red'},
    'i': {'suffix': 'zi', 'color': '#8B0000'},
}


# ── Field geometry ───────────────────────────────────────────────────

def _ang_dist(ra1, dec1, ra2, dec2):
    """Angular distance (degrees) between two points, all in radians."""
    adist = (np.sin(dec1) * np.sin(dec2)
             + np.cos(dec1) * np.cos(dec2) * np.cos(ra2 - ra1))
    return np.arccos(np.clip(adist, -1, 1)) * 180.0 / np.pi


def _fit_line(x, x0, y0, x1, y1):
    return (y1 - y0) * (x - x0) / (x1 - x0) + y0


def _ortographic_projection(ra, dec, ra0, dec0):
    x = -np.cos(dec) * np.sin(ra - ra0)
    y = np.cos(dec0) * np.sin(dec) - np.sin(dec0) * np.cos(dec) * np.cos(ra - ra0)
    return x * 180.0 / np.pi, y * 180.0 / np.pi


_CCD_X = [
    -3.646513, -3.647394, -1.920848, -1.920383, -1.790386, -1.790817,
    -0.064115, -0.064099, 0.062113, 0.062129, 1.788830, 1.788400,
    1.918441, 1.918905, 3.645452, 3.644571, -3.646416, -3.646708,
    -1.919998, -1.919844, -1.789454, -1.789597, -0.062733, -0.062727,
    0.061814, 0.061819, 1.788683, 1.788540, 1.918871, 1.919025,
    3.645736, 3.645443, -3.646562, -3.646270, -1.919698, -1.919852,
    -1.789413, -1.789270, -0.062544, -0.062549, 0.062876, 0.062871,
    1.789598, 1.789741, 1.919874, 1.919720, 3.646292, 3.646584,
    -3.645853, -3.644972, -1.918842, -1.919306, -1.789367, -1.788937,
    -0.062651, -0.062666, 0.063143, 0.063128, 1.789415, 1.789845,
    1.919878, 1.919413, 3.645543, 3.646424,
]
_CCD_Y = [
    -3.727898, -2.001758, -2.004785, -3.731333, -3.729368, -2.002803,
    -2.003812, -3.730512, -3.730976, -2.004276, -2.003269, -3.729834,
    -3.731505, -2.004957, -2.001932, -3.728073, -1.816060, -0.089749,
    -0.090622, -1.817335, -1.816611, -0.089881, -0.090172, -1.817035,
    -1.817472, -0.090609, -0.090319, -1.817048, -1.817584, -0.090871,
    -0.089998, -1.816309, 0.090679, 1.816989, 1.818266, 0.091552,
    0.091155, 1.817884, 1.818309, 0.091446, 0.090876, 1.817739,
    1.817315, 0.090586, 0.091290, 1.818003, 1.816728, 0.090417,
    2.002667, 3.728808, 3.732241, 2.005694, 2.003694, 3.730258,
    3.731401, 2.004701, 2.003834, 3.730533, 3.729391, 2.002826,
    2.004674, 3.731221, 3.727789, 2.001648,
]


def _inside_polygon(xp, yp):
    """Return (ccd, quad) if (xp, yp) falls inside a ZTF CCD, else (None, None)."""
    x, y = _CCD_X, _CCD_Y
    for i in range(16):
        idx = 4 * i
        y_test_1 = _fit_line(xp, x[idx], y[idx], x[idx + 3], y[idx + 3])
        y_test_2 = _fit_line(xp, x[idx + 1], y[idx + 1], x[idx + 2], y[idx + 2])
        if yp < y_test_1 or yp > y_test_2:
            continue
        x_test_1 = _fit_line(yp, y[idx], x[idx], y[idx + 1], x[idx + 1])
        x_test_2 = _fit_line(yp, y[idx + 3], x[idx + 3], y[idx + 2], x[idx + 2])
        if xp < x_test_1 or xp > x_test_2:
            continue
        ccd = i + 1
        y_test = _fit_line(
            xp, 0.5 * (x[idx + 2] + x[idx + 3]), 0.5 * (y[idx + 2] + y[idx + 3]),
            0.5 * (x[idx] + x[idx + 1]), 0.5 * (y[idx] + y[idx + 1]),
        )
        x_test = _fit_line(
            yp, 0.5 * (y[idx] + y[idx + 3]), 0.5 * (x[idx] + x[idx + 3]),
            0.5 * (y[idx + 1] + y[idx + 2]), 0.5 * (x[idx + 1] + x[idx + 2]),
        )
        if yp < y_test:
            quad = 4 if xp < x_test else 3
        else:
            quad = 1 if xp < x_test else 2
        return ccd, quad
    return None, None


def _read_fields():
    """Load ZTF field definitions from ZTF_Fields.txt."""
    path = os.path.join(_MODULE_DIR, 'ZTF_Fields.txt')
    fieldno, ra, dec = np.loadtxt(
        path, unpack=True, usecols=(0, 1, 2), dtype='int,float,float')
    return fieldno, ra, dec


_fieldno, _field_ra, _field_dec = _read_fields()


def get_field_id(ra_deg, dec_deg):
    """Find all (field, ccd, quad) tuples covering a given (ra, dec) in degrees."""
    deg = np.pi / 180.0
    ra = ra_deg * deg
    dec = dec_deg * deg
    ra_arr = _field_ra * deg
    dec_arr = _field_dec * deg

    ADIST_MAX = 5.66
    res = []
    for i in range(len(ra_arr)):
        adist = _ang_dist(ra, dec, ra_arr[i], dec_arr[i])
        if adist >= ADIST_MAX:
            continue
        x, y = _ortographic_projection(ra, dec, ra_arr[i], dec_arr[i])
        ccd, quad = _inside_polygon(x, y)
        if ccd is not None and quad is not None:
            res.append((int(_fieldno[i]), int(ccd), int(quad)))
    return res


# ── HDF5 matchfile reading ──────────────────────────────────────────

def _getobj(ps_id, fname):
    """Extract a single source's light curve from a ZTF HDF5 matchfile.

    Returns (jd, diff_flux, flux_err, flag, pid, flux_ref) or None.
    """
    with h5py.File(fname, 'r') as f:
        sources = f['data']['sources']['gaia_id'][:]
        idx = np.where(sources == ps_id)[0]
        if len(idx) == 0:
            return None

        idx = idx[0]
        mag_ref = float(f['data']['sources'][idx]['mag_ref'])
        _, flux_ref = calibrate_flux(np.array([0.0]), mag_ref)  # get ref_flux

        exposures = f['data']['exposures']
        jd = exposures['jd'][:]
        pid = exposures['pid'][:]
        n_exp = len(jd)

        rows = np.arange(n_exp, dtype=np.int64) + idx * n_exp
        sd = f['data']['sourcedata'][rows]

        return jd, sd['flux'], sd['flux_err'], sd['flag'], pid, flux_ref


def _find_ps_id(ra, dec, field_ccd_quads, matchfile_dir=MATCHFILE_DIR):
    """Look up the Pan-STARRS ID of the nearest source to (ra, dec)."""
    for suffix in ['zr', 'zg', 'zi']:
        for field, ccd, quad in field_ccd_quads:
            fname = (f'{matchfile_dir}/{field:04d}/'
                     f'data_{field:04d}_{ccd:02d}_{quad:1d}_{suffix}.h5')
            if not os.path.exists(fname):
                continue
            try:
                with h5py.File(fname, 'r') as f:
                    src_ra = f['data']['sources']['ra'][:]
                    src_dec = f['data']['sources']['decl'][:]
                    cos_dec = np.cos(np.deg2rad(dec))
                    dist = np.sqrt(((src_ra - ra) * cos_dec)**2
                                   + (src_dec - dec)**2)
                    nearest = np.argmin(dist)
                    ps_id = int(f['data']['sources'][nearest]['gaia_id'])
                    offset = float(dist[nearest]) * 3600.0
                    return ps_id, offset
            except Exception:
                continue
    raise ValueError(f'No matchfile found for (ra={ra}, dec={dec})')


# ── BJD conversion ──────────────────────────────────────────────────

def _jd_to_bjd(jd, ra, dec):
    """Convert JD timestamps to BJD_TCB."""
    mjd = jd - 2400000.5
    t = Time(mjd, format='mjd', scale='utc')
    t_tcb = t.tcb
    c = SkyCoord(ra, dec, unit='deg')
    observatory = EarthLocation.of_site('Palomar')
    delta = t_tcb.light_travel_time(c, kind='barycentric', location=observatory)
    return (t_tcb + delta).value


# ── Single-filter extraction ────────────────────────────────────────

def _extract_filter(ps_id, field_ccd_quads, filt_suffix, matchfile_dir):
    """Extract and combine light curve data for one filter across all fields.

    Calibrates each field's diff_flux with its own ref_flux before
    concatenation, then applies flag filtering and MAD-based outlier
    clipping (consistent with the training pipeline).
    """
    times_acc, cal_acc, ferr_acc, flag_acc = [], [], [], []

    for field, ccd, quad in field_ccd_quads:
        fname = (f'{matchfile_dir}/{field:04d}/'
                 f'data_{field:04d}_{ccd:02d}_{quad:1d}_{filt_suffix}.h5')
        try:
            result = _getobj(ps_id, fname)
            if result is None:
                continue
            jd, diff_flux, ferr, flag, pid, flux_ref = result
            # Calibrate per-field before concatenation
            cal_flux = diff_flux.astype(np.float64) + flux_ref
            finite = ~np.isnan(diff_flux)
            times_acc.append(jd[finite])
            cal_acc.append(cal_flux[finite])
            ferr_acc.append(ferr[finite].astype(np.float64))
            flag_acc.append(flag[finite])
        except Exception:
            pass

    if len(times_acc) == 0:
        return None

    t = np.concatenate(times_acc) - 2400000.5  # JD → MJD
    y = np.concatenate(cal_acc)
    dy = np.concatenate(ferr_acc)
    fl = np.concatenate(flag_acc)

    # Flag + finite filtering (cal_flux used in place of diff_flux since
    # calibration was already applied per-field before concatenation)
    good = filter_good(y, y, dy, fl)
    t, y, dy = t[good], y[good], dy[good]
    if len(t) < 3:
        return None

    # MAD-based outlier clipping (consistent with training pipeline)
    keep = clip_outliers(y, threshold=10.0)
    t, y, dy = t[keep], y[keep], dy[keep]
    if len(t) < 3:
        return None

    return t, y, dy


def _extract_filter_raw(ps_id, field_ccd_quads, filt_suffix, matchfile_dir):
    """Extract raw (unfiltered, unclipped) light curve data for one filter."""
    for field, ccd, quad in field_ccd_quads:
        fname = (f'{matchfile_dir}/{field:04d}/'
                 f'data_{field:04d}_{ccd:02d}_{quad:1d}_{filt_suffix}.h5')
        try:
            result = _getobj(ps_id, fname)
            if result is not None:
                jd, diff_flux, flux_err, flag, pid, flux_ref = result
                return {
                    'jd': jd,
                    'diff_flux': diff_flux.astype(np.float64),
                    'flux_err': flux_err.astype(np.float64),
                    'flag': flag,
                    'flux_ref': flux_ref,
                    'field': field,
                    'ccd': ccd,
                    'quad': quad,
                }
        except Exception:
            pass
    return None


# ── Public API ───────────────────────────────────────────────────────

def get_lightcurve(ra, dec, ps_id=None, filters='gri',
                   matchfile_dir=MATCHFILE_DIR):
    """Extract a processed ZTF light curve (BJD, flag-filtered, clipped).

    Parameters
    ----------
    ra, dec : float
        Source coordinates in degrees.
    ps_id : int, optional
        Pan-STARRS source ID. If omitted, nearest source is matched.
    filters : str
        Which ZTF filters to extract (subset of 'gri').
    matchfile_dir : str
        Root directory of matchfiles.

    Returns
    -------
    dict
        Keyed by filter letter. Each value has keys 'time' (BJD_TCB),
        'flux' (calibrated), 'flux_err'.
    """
    field_ccd_quads = get_field_id(ra, dec)
    if not field_ccd_quads:
        return {}

    if ps_id is None:
        ps_id, _ = _find_ps_id(ra, dec, field_ccd_quads, matchfile_dir)

    lcs = {}
    for filt_name in filters:
        if filt_name not in _FILTERS:
            continue
        result = _extract_filter(
            ps_id, field_ccd_quads, _FILTERS[filt_name]['suffix'], matchfile_dir)
        if result is None:
            continue
        t, y, dy = result
        bjd = _jd_to_bjd(t + 2400000.5, ra, dec)
        lcs[filt_name] = {'time': bjd, 'flux': y, 'flux_err': dy}
    return lcs


def get_lightcurve_raw(ra, dec, ps_id=None, filters='gri',
                       matchfile_dir=MATCHFILE_DIR):
    """Extract raw ZTF light curve data (no filtering, no clipping, no BJD).

    Returns everything from the matchfile so you can inspect the full
    preprocessing pipeline step by step.

    Parameters
    ----------
    ra, dec : float
        Source coordinates in degrees.
    ps_id : int, optional
        Pan-STARRS source ID. If omitted, nearest source is matched.
    filters : str
        Which ZTF filters to extract (subset of 'gri').
    matchfile_dir : str
        Root directory of matchfiles.

    Returns
    -------
    dict
        Keyed by filter letter. Each value is a dict with keys:
        'jd', 'diff_flux', 'flux_err', 'flag', 'flux_ref', 'field',
        'ccd', 'quad'.
    """
    field_ccd_quads = get_field_id(ra, dec)
    if not field_ccd_quads:
        return {}

    if ps_id is None:
        ps_id, _ = _find_ps_id(ra, dec, field_ccd_quads, matchfile_dir)

    raws = {}
    for filt_name in filters:
        if filt_name not in _FILTERS:
            continue
        result = _extract_filter_raw(
            ps_id, field_ccd_quads, _FILTERS[filt_name]['suffix'], matchfile_dir)
        if result is not None:
            raws[filt_name] = result
    return raws


def phase_fold(times, period):
    """Return phases in [0, 1) for the given period."""
    times = np.asarray(times)
    return ((times - times.min()) % period) / period
