#!/usr/bin/env python3
"""Encoder-seed confidence intervals for the BiGRU (camera-ready).

Runs the ridge period-diagnostic on each independently-trained BiGRU seed and
reports R^2, f_cls, rho_within as mean +/- sd and a 95% CI ACROSS ENCODER
SEEDS -- closing the "all results are single-seed" criticism at the level it
was raised (the probe-split CIs in Table 4 are a different axis of variation).

Expects results/chen2020_25k/embeddings_rnn_seed<seed>.npz for each seed
(produced by scripts/embed_rnn_seeds.sh). Env: ztfpe2.
  python scripts/diag_seed_ci.py --seeds 42 1 2 3
"""
import os
import argparse
from pathlib import Path

import numpy as np
import h5py
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, t as student_t

REPO = Path(os.environ.get("PERIOD_DIAG_ROOT", str(Path(__file__).resolve().parents[1])))
R = REPO / "results/chen2020_25k"
CHEN_HDF5 = os.environ.get("ZTF_CHEN_HDF5", str(REPO / "data" / "chen2020.hdf5"))


def load_labels():
    ci = np.load(R / "lc_cache_25k.npz")["chen_indices"]
    with h5py.File(CHEN_HDF5, "r") as f:
        vt = f["catalog/var_type"][:]
        cat = f["catalog"]
        pk = next(k for k in ("Per_g", "Per_r", "period_g", "period_r",
                              "period", "P") if k in cat.keys())
        per = cat[pk][:]

    def s(x):
        x = x.decode() if isinstance(x, bytes) else str(x)
        return x[2:-1] if x.startswith("b'") and x.endswith("'") else x
    return np.array([s(v) for v in vt])[ci], per[ci]


def diagnostic(z, y, logP, big):
    keep = np.all(np.isfinite(z), axis=1)
    z, yy, lp = z[keep], y[keep], logP[keep]
    Xs = StandardScaler().fit_transform(z)
    yp = cross_val_predict(RidgeCV(alphas=np.logspace(-2, 4, 13)), Xs, lp,
                           cv=KFold(5, shuffle=True, random_state=42), n_jobs=-1)
    r2 = r2_score(lp, yp)
    gm = yp.mean(); ssb = 0.0; n = 0; rh = []
    for c in big:
        m = yy == c
        if m.sum() < 30:
            continue
        ssb += m.sum() * (yp[m].mean() - gm) ** 2
        n += m.sum()
        rh.append(spearmanr(lp[m], yp[m])[0])
    return r2, ssb / (n * yp.var()), float(np.median(rh))


def ci95(vals):
    vals = np.asarray(vals, float)
    m, sd, k = vals.mean(), vals.std(ddof=1), len(vals)
    h = student_t.ppf(0.975, k - 1) * sd / np.sqrt(k)
    return m, sd, (m - h, m + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2, 3])
    args = ap.parse_args()

    y, P = load_labels()
    logP = np.log10(P)
    keep = np.isfinite(logP) & (P > 0)
    y, logP = y[keep], logP[keep]
    cls, cnt = np.unique(y, return_counts=True)
    big = cls[cnt >= 30]

    rows = {"R2": [], "f_cls": [], "rho": []}
    print(f"{'seed':<8}{'R2':>8}{'f_cls':>8}{'rho_within':>12}")
    for s in args.seeds:
        f = R / f"embeddings_rnn_seed{s}.npz"
        if not f.exists():
            print(f"{s:<8}  MISSING {f.name}"); continue
        z = np.load(f)["z_sig"]
        r2, fc, rho = diagnostic(z, y, logP, big)
        rows["R2"].append(r2); rows["f_cls"].append(fc); rows["rho"].append(rho)
        print(f"{s:<8}{r2:8.3f}{fc:8.3f}{rho:12.3f}")

    if len(rows["rho"]) >= 2:
        print(f"\nAcross {len(rows['rho'])} encoder seeds (mean +/- sd, 95% CI):")
        for k, lab in [("R2", "R^2"), ("f_cls", "f_cls"), ("rho", "rho_within")]:
            m, sd, (lo, hi) = ci95(rows[k])
            print(f"  {lab:<11} {m:.3f} +/- {sd:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
    else:
        print("\nNeed >= 2 seeds for a CI; train more via orcd/submit_rnn_seeds.sh")


if __name__ == "__main__":
    main()
