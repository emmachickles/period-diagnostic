#!/usr/bin/env python3
"""
probe_period_recovery.py — Linear ridge probe on the SSL embedding to
predict log10(period). Tests whether the SSL transformer learned period
information directly from raw irregular photometry (vs. just learning
class-discriminative features).

Why this matters for the paper: classical RF on Chen+2020 features hits
88 % balanced accuracy because Per-g / Per-r are *given* as features.
The SSL transformer operates on raw photometry — no period given. If a
linear probe on z_sig recovers log(period) with R² > 0.5, we have direct
evidence the model learned period from raw light curves, not just
discriminated classes via correlated features.

Three metrics:
  R² on log10(period)        — what fraction of variance is linear in z_sig
  MAE in log10(period)       — typical fractional period error
  Alias-tolerant accuracy    — fraction within 5% of {P, P/2, 2P} of catalog
"""

import os
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

REPO = Path(os.environ.get("PERIOD_DIAG_ROOT", str(Path(__file__).resolve().parents[1])))
EMB_DEFAULT = str(REPO / "results/chen2020_25k/embeddings_ct_v3_e3.npz")
CHEN_DEFAULT = os.environ.get("ZTF_CHEN_HDF5", str(REPO / "data" / "chen2020.hdf5"))
OUT_DEFAULT = str(REPO / "results/chen2020_25k/probe_period_recovery_v3_e3.json")


def alias_tolerant_accuracy(p_true, p_pred, tol=0.05):
    """Fraction of sources whose predicted period is within tol of
    the true period or one of its half/double aliases."""
    p_true = np.asarray(p_true, dtype=np.float64)
    p_pred = np.asarray(p_pred, dtype=np.float64)
    valid = (p_true > 0) & (p_pred > 0)
    if valid.sum() == 0:
        return 0.0
    pt = p_true[valid]; pp = p_pred[valid]
    ratios = [pp / pt, pp / (pt * 0.5), pp / (pt * 2.0)]
    within = np.zeros_like(pt, dtype=bool)
    for r in ratios:
        within |= (np.abs(r - 1.0) <= tol)
    return float(within.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", default=EMB_DEFAULT,
                    help="npz with z_sig (N, D) and chen_indices (N,)")
    ap.add_argument("--chen_hdf5", default=CHEN_DEFAULT)
    ap.add_argument("--output", default=OUT_DEFAULT)
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    emb = np.load(args.embeddings)
    z_sig = emb["z_sig"]
    chen_indices = emb["chen_indices"]
    success = emb["success_mask"] if "success_mask" in emb.files else np.ones(len(z_sig), dtype=bool)
    print(f"Embeddings: {z_sig.shape}, {success.sum():,} successful")

    # Pull period from chen2020.hdf5
    with h5py.File(args.chen_hdf5, "r") as f:
        period = f["catalog/period"][:][chen_indices]

    # Drop sources with missing or non-positive period
    keep = success & np.isfinite(period) & (period > 0)
    X = z_sig[keep]
    p = period[keep]
    y = np.log10(p)
    print(f"  usable: {keep.sum():,} sources after dropping NaN/non-positive periods")
    print(f"  log10(period) range: [{y.min():.3f}, {y.max():.3f}] "
          f"= [{10**y.min():.4f} d, {10**y.max():.1f} d]")

    # Standardize z_sig
    Xs = StandardScaler().fit_transform(X)

    # 5-fold CV with RidgeCV (linear, regularized)
    print(f"\n→ RidgeCV linear probe · {args.n_splits}-fold CV...")
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    reg = RidgeCV(alphas=np.logspace(-2, 4, 20))
    y_pred = cross_val_predict(reg, Xs, y, cv=kf, n_jobs=-1)
    p_pred = 10.0 ** y_pred

    r2 = r2_score(y, y_pred)
    mae_log = mean_absolute_error(y, y_pred)
    alias_acc = alias_tolerant_accuracy(p, p_pred, tol=0.05)
    exact_acc = alias_tolerant_accuracy(p, p_pred, tol=0.05)  # actually defined inside; will recompute
    # Compute "exact" (no alias tolerance) separately
    ratio = p_pred / p
    exact_acc = float((np.abs(ratio - 1.0) <= 0.05).mean())

    print(f"  R²             {r2:.4f}")
    print(f"  MAE (log10 P)  {mae_log:.4f}  (≈ {(10**mae_log - 1)*100:.1f}% fractional period error)")
    print(f"  exact-match @5%   {100*exact_acc:.2f}%   (predicted P within 5% of catalog P)")
    print(f"  alias-tolerant @5% {100*alias_acc:.2f}%   (within 5% of P, P/2, or 2P)")

    # Per-class breakdown
    with h5py.File(args.chen_hdf5, "r") as f:
        var_type_raw = f["catalog/var_type"][:]
    def safe_str(x):
        s = str(x.decode() if isinstance(x, bytes) else x)
        if s.startswith("b'") and s.endswith("'"): return s[2:-1]
        return s
    classes_full = np.array([safe_str(s) for s in var_type_raw[chen_indices]])
    classes_kept = classes_full[keep]

    print("\nPer-class period recovery:")
    per_class = {}
    for c in sorted(np.unique(classes_kept)):
        m = classes_kept == c
        if m.sum() < 10: continue
        r2_c = r2_score(y[m], y_pred[m])
        mae_c = mean_absolute_error(y[m], y_pred[m])
        ratio_c = p_pred[m] / p[m]
        exact_c = float((np.abs(ratio_c - 1.0) <= 0.05).mean())
        alias_c = alias_tolerant_accuracy(p[m], p_pred[m], tol=0.05)
        per_class[c] = {
            "n": int(m.sum()),
            "r2": float(r2_c),
            "mae_log10": float(mae_c),
            "exact_5pct": exact_c,
            "alias_5pct": alias_c,
        }
        print(f"  {c:>8s}  N={m.sum():>5,}  R²={r2_c:+.3f}  "
              f"MAE={mae_c:.3f}  exact={100*exact_c:5.1f}%  alias={100*alias_c:5.1f}%")

    out = {
        "embeddings_path": args.embeddings,
        "n_sources": int(keep.sum()),
        "r2": float(r2),
        "mae_log10": float(mae_log),
        "exact_5pct": float(exact_acc),
        "alias_5pct": float(alias_acc),
        "per_class": per_class,
        "log_period_range": [float(y.min()), float(y.max())],
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
