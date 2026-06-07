#!/usr/bin/env python3
"""
probe_classical_rf_full.py — Random Forest baseline using EVERY hand-
crafted feature available to us: Chen+2020's catalog columns PLUS the
full LS and BLS catalog outputs (the latter only this group has at
scale, computed at substantial GPU cost over 2B+ ZTF sources).

This is the proper "hand-crafted-features ceiling" comparison for the
SSL-transformer numbers. The earlier scripts/probe_classical_rf.py used
only Chen+2020's features and hit 88.2 %; this version adds 19 BLS +
10 LS columns from the per-tile cross-match in
extract_full_bls_ls_features.py.

Output: results/chen2020_25k/probe_report_classical_rf_full.json
"""

import os
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix

REPO = Path(os.environ.get("PERIOD_DIAG_ROOT", str(Path(__file__).resolve().parents[1])))
EMB_DEFAULT = str(REPO / "results/chen2020_25k/embeddings.npz")
CHEN_DEFAULT = os.environ.get("ZTF_CHEN_HDF5", str(REPO / "data" / "chen2020.hdf5"))
FULL_FEATURES_DEFAULT = str(REPO / "results/chen2020_25k/features_full_bls_ls.npz")
OUT_DEFAULT = str(REPO / "results/chen2020_25k/probe_report_classical_rf_full.json")

# Same chen2020.hdf5 features as probe_classical_rf.py
CHEN_FEATURES = [
    "Per-g", "Per-r",
    "R2-g", "R2-r",
    "R21", "R21-g", "R21-r",
    "phi21", "phi21-g", "phi21-r",
    "gAmp", "rAmp",
    "ls_skew", "ls_kurtosis", "ls_mad", "ls_variance",
    "log(FAP-g)", "log(FAP-r)",
    "Ng", "Nr",
    "Dmin-g", "Dmin-r",
]


def safe_str(x):
    s = str(x.decode() if isinstance(x, bytes) else x)
    if s.startswith("b'") and s.endswith("'"): return s[2:-1]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", default=EMB_DEFAULT)
    ap.add_argument("--chen_hdf5", default=CHEN_DEFAULT)
    ap.add_argument("--full_features", default=FULL_FEATURES_DEFAULT)
    ap.add_argument("--output", default=OUT_DEFAULT)
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--n_estimators", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    emb = np.load(args.embeddings)
    chen_indices = emb["chen_indices"]
    print(f"chen_indices: {len(chen_indices):,}")

    # Chen features
    print(f"Loading {len(CHEN_FEATURES)} Chen features...")
    with h5py.File(args.chen_hdf5, "r") as f:
        cat = f["catalog"]
        chen_feat_cols = [cat[name][:][chen_indices].astype(np.float64)
                          for name in CHEN_FEATURES]
        var_type_raw = cat["var_type"][:]
    X_chen = np.column_stack(chen_feat_cols)
    y = np.array([safe_str(s) for s in var_type_raw[chen_indices]])

    # Extended LS + BLS features. Drop columns that are either survey-timing
    # artifacts or apparent-magnitude duplicates / quality flags.
    DROP_COLS = {
        "ls_full_ref_flux", "ls_full_ref_flux_flag",  # mag duplicate / flag
        # bls_full_mid_transit_time and bls_full_min_time were already
        # excluded at extraction time — keep this set defensively.
        "bls_full_mid_transit_time", "bls_full_min_time",
    }
    print(f"Loading extended features from {args.full_features}...")
    full = np.load(args.full_features)
    full_feature_names = sorted([k for k in full.files
                                  if k.startswith(("ls_full_", "bls_full_"))
                                  and k not in DROP_COLS])
    print(f"  {len(full_feature_names)} extended feature columns")
    X_full = np.column_stack([full[k].astype(np.float64) for k in full_feature_names])

    if len(full["chen_indices"]) != len(chen_indices):
        raise SystemExit("chen_indices mismatch between embeddings and full features")

    # Combine
    feature_names = CHEN_FEATURES + full_feature_names
    X = np.column_stack([X_chen, X_full])
    print(f"\nCombined feature matrix: {X.shape}")
    print(f"  Chen features:     {len(CHEN_FEATURES):>3}")
    print(f"  Extended LS:       {sum(1 for n in full_feature_names if n.startswith('ls_full_')):>3}")
    print(f"  Extended BLS:      {sum(1 for n in full_feature_names if n.startswith('bls_full_')):>3}")

    # NaN/inf handling. The BLS pipeline emits ±inf for some failure
    # modes (e.g., zero out_of_eclipse_scatter denominator); also use it
    # as a sentinel in some columns. Sklearn's tree fitter rejects inf
    # explicitly, so coerce to NaN first, then median-fill.
    X[~np.isfinite(X)] = np.nan
    n_nan = np.isnan(X).sum(axis=0)
    n_all_nan = (n_nan == len(X)).sum()
    print(f"  features with any non-finite: {(n_nan > 0).sum()}/{X.shape[1]}")
    print(f"  features all non-finite (will become constant): {n_all_nan}")
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.all(np.isnan(col)):
            X[:, j] = 0.0
        else:
            med = np.nanmedian(col)
            X[np.isnan(col), j] = med

    # 5-fold CV
    print(f"\n→ RandomForest (n_estimators={args.n_estimators}) · "
          f"{args.n_splits}-fold CV on {X.shape[1]} features...")
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        n_jobs=-1, random_state=args.seed,
        class_weight="balanced",
    )
    y_pred = cross_val_predict(clf, X, y, cv=skf, n_jobs=1)
    bacc = balanced_accuracy_score(y, y_pred)
    print(f"  balanced accuracy: {bacc:.4f}")

    classes = sorted(np.unique(y))
    cm = confusion_matrix(y, y_pred, labels=classes)
    rep = classification_report(y, y_pred, labels=classes, output_dict=True)
    print(f"\nPer-class report:")
    for c in classes:
        r = rep.get(c, {})
        print(f"  {c:>8s}  P={r.get('precision', 0):.3f}  R={r.get('recall', 0):.3f}  "
              f"F1={r.get('f1-score', 0):.3f}  N={r.get('support', 0):.0f}")

    # Feature importance over the full data
    clf_full = RandomForestClassifier(
        n_estimators=args.n_estimators, n_jobs=-1,
        random_state=args.seed, class_weight="balanced",
    ).fit(X, y)
    importances = sorted(
        zip(feature_names, clf_full.feature_importances_.tolist()),
        key=lambda kv: -kv[1],
    )
    print(f"\nTop 15 feature importances:")
    for name, imp in importances[:15]:
        print(f"  {name:>30s}: {imp:.4f}")

    out = {
        "method": "RandomForest on Chen+2020 + full LS + full BLS hand-crafted features",
        "n_features": X.shape[1],
        "feature_names": feature_names,
        "n_chen_features": len(CHEN_FEATURES),
        "n_extended_features": len(full_feature_names),
        "n_estimators": args.n_estimators,
        "logr_balanced_accuracy": float(bacc),
        "knn_balanced_accuracy": float(bacc),
        "classes": classes,
        "confusion_matrix": cm.tolist(),
        "per_class": {c: rep.get(c, {}) for c in classes},
        "feature_importance": importances,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
