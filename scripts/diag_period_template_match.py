#!/usr/bin/env python3
"""Quantify template-match vs absolute-time learning on period regression.

For each method, compute:
  R²            — overall ridge probe R² on log10(P)
  class_frac    — between-class share of predicted-log-P variance
                  (high = template-matching: model effectively predicts
                  the per-class mean period; low = uses source-level
                  information)
  withinR²      — within-class R²: average across classes (with N≥30) of
                  the R² between true and predicted log P after subtracting
                  per-class means. If withinR² ≈ 0 the encoder is doing
                  pure class -> mean-P template matching.
  rank_corr     — within-class Spearman correlation (median across classes)
                  as a non-parametric alternative.
"""
import os
import sys
from pathlib import Path
import numpy as np
import h5py
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

REPO = Path(os.environ.get("PERIOD_DIAG_ROOT", str(Path(__file__).resolve().parents[1])))
RESULTS = REPO / "results/chen2020_25k"
LC_CACHE = RESULTS / "lc_cache_25k.npz"
CHEN_HDF5 = os.environ.get("ZTF_CHEN_HDF5", str(REPO / "data" / "chen2020.hdf5"))

METHODS = [
    ("BiGRU (ours)",                 "embeddings_rnn_v1.npz"),
    ("Pretrained Chronos",           "embeddings_chronos.npz"),
    ("Chronos-T5 (from scratch)",    "embeddings_chronos_t5_e10.npz"),
    ("CT-SSL (ours)",                "embeddings_ct_v3_e3.npz"),
    ("1D CNN (ours)",                "embeddings_cnn_v1.npz"),
    ("Pretrained MOMENT",            "embeddings_moment.npz"),
    ("PatchTST (fs)",                "embeddings_patchtst_v3scale.npz"),
]


def load_labels_and_periods():
    # Self-contained fast path: use the bundled subset labels if present
    # (Zenodo release), so the 1.4 GB Chen+2020 catalog is not required.
    _labels = LC_CACHE.parent / "labels_chen2020_25k.npz"
    if _labels.exists():
        d = np.load(_labels, allow_pickle=True)
        return d["var_type"], d["period"]
    cache = np.load(LC_CACHE)
    chen_indices = cache["chen_indices"]
    with h5py.File(CHEN_HDF5, "r") as f:
        var_type_raw = f["catalog/var_type"][:]
        cat = f["catalog"]
        keys = list(cat.keys())
        period_key = None
        for k in ("Per_g", "Per_r", "period_g", "period_r", "period", "P"):
            if k in keys:
                period_key = k
                break
        period = cat[period_key][:]

    def safe_str(x):
        if isinstance(x, bytes):
            x = x.decode("utf-8", errors="replace")
        s = str(x)
        if len(s) >= 3 and s.startswith("b'") and s.endswith("'"):
            return s[2:-1]
        return s

    var_type = np.array([safe_str(s) for s in var_type_raw])
    y = var_type[chen_indices]
    P = period[chen_indices]
    return y, P


def main():
    y, P = load_labels_and_periods()
    log_P = np.log10(P)
    keep = np.isfinite(log_P) & (P > 0)
    y_keep = y[keep]
    log_P_keep = log_P[keep]
    classes, counts = np.unique(y_keep, return_counts=True)
    big_classes = classes[counts >= 30]

    print(f"\nN total = {keep.sum()},  classes (N>=30) = {list(big_classes)}\n")
    header = f"{'method':<28s}  {'R²':>6s}  {'class_frac':>10s}  {'withinR²':>9s}  {'med_rho':>7s}"
    print(header); print("-" * len(header))

    rows = []
    for name, npz in METHODS:
        path = RESULTS / npz
        if not path.exists():
            print(f"{name:<28s}  (missing: {npz})")
            continue
        emb = np.load(path)
        if "z_sig" not in emb.files:
            print(f"{name:<28s}  (no z_sig)")
            continue
        z = emb["z_sig"][keep]

        Xs = StandardScaler().fit_transform(z)
        reg = RidgeCV(alphas=np.logspace(-2, 4, 13))
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        y_pred = cross_val_predict(reg, Xs, log_P_keep, cv=kf, n_jobs=-1)

        r2 = r2_score(log_P_keep, y_pred)

        # class-driven fraction (between-class share of pred variance)
        gmp = y_pred.mean()
        ss_between = 0.0
        n_total = 0
        for c in big_classes:
            m = (y_keep == c)
            if m.sum() < 30: continue
            ss_between += m.sum() * (y_pred[m].mean() - gmp) ** 2
            n_total += m.sum()
        ss_total = n_total * y_pred.var()
        class_frac = ss_between / ss_total if ss_total > 0 else float("nan")

        # within-class R² and Spearman: subtract per-class means, average
        # weighted by class size. R² of (true_centered, pred_centered) per
        # class, then weighted-average.
        within_r2_acc, within_rho_list, weight = [], [], []
        for c in big_classes:
            m = (y_keep == c)
            if m.sum() < 30: continue
            y_t = log_P_keep[m] - log_P_keep[m].mean()
            y_p = y_pred[m] - y_pred[m].mean()
            ss_res = ((y_t - y_p) ** 2).sum()
            ss_tot = (y_t ** 2).sum()
            r2c = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            within_r2_acc.append(r2c)
            weight.append(m.sum())
            rho, _ = spearmanr(log_P_keep[m], y_pred[m])
            within_rho_list.append(rho)
        within_r2 = float(np.average(within_r2_acc, weights=weight))
        med_rho = float(np.median(within_rho_list))

        print(f"{name:<28s}  {r2:6.3f}  {class_frac:10.3f}  {within_r2:9.3f}  {med_rho:7.3f}")
        rows.append((name, r2, class_frac, within_r2, med_rho))

    print()
    print("Reading the metrics:")
    print("  class_frac near 1 → predictions are nearly constant within each class")
    print("                       (class-template matching, not learning absolute time)")
    print("  within_r²        → variance explained INSIDE each class (genuine time signal)")
    print("                       0 means class is mapped to its mean P only")
    print("  med_rho          → Spearman ρ within class (robust to alias issues)")


if __name__ == "__main__":
    main()
