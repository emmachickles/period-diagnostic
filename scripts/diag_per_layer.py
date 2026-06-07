#!/usr/bin/env python3
"""Per-layer period-probe diagnostic for the CT-SSL transformer.

Rules out that mean-pooling the final layer discards period information held
in an intermediate layer (camera-ready, Appendix). Reads the saved per-layer
embeddings and runs the same ridge diagnostic as diag_period_template_match.py
on each layer. Env: ztfpe2.
"""
import os
from pathlib import Path
import numpy as np
import h5py
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

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


def main():
    y, P = load_labels()
    logP = np.log10(P)
    keep = np.isfinite(logP) & (P > 0)
    y, logP = y[keep], logP[keep]
    cls, cnt = np.unique(y, return_counts=True)
    big = cls[cnt >= 30]

    pl = np.load(R / "per_layer.npz")["per_layer"]  # (n_layers, N, d)
    print(f"{'layer':<14}{'R2':>8}{'f_cls':>8}{'rho_within':>12}")
    for L in range(pl.shape[0]):
        z = pl[L][keep]
        Xs = StandardScaler().fit_transform(z)
        yp = cross_val_predict(
            RidgeCV(alphas=np.logspace(-2, 4, 13)), Xs, logP,
            cv=KFold(5, shuffle=True, random_state=42), n_jobs=-1)
        r2 = r2_score(logP, yp)
        gm = yp.mean(); ssb = 0.0; n = 0; rh = []
        for c in big:
            m = y == c
            if m.sum() < 30:
                continue
            ssb += m.sum() * (yp[m].mean() - gm) ** 2
            n += m.sum()
            rh.append(spearmanr(logP[m], yp[m])[0])
        tag = f"{L}" + (" (pooled)" if L == pl.shape[0] - 1 else "")
        print(f"{tag:<14}{r2:8.3f}{ssb/(n*yp.var()):8.3f}{np.median(rh):12.3f}")


if __name__ == "__main__":
    main()
