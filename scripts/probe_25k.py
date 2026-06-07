#!/usr/bin/env python3
"""
probe_25k.py — Fresh logistic-regression probe + k-NN probe on the 25K
chen2020 subset's z_sig embeddings, including 11-class confusion matrix
and per-class breakdown. Saves probe_predictions.json keyed by demo idx.
"""

import os
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    balanced_accuracy_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler

REPO = Path(os.environ.get("PERIOD_DIAG_ROOT", str(Path(__file__).resolve().parents[1])))
EMB_DEFAULT = str(REPO / "results/chen2020_25k/embeddings.npz")
CHEN_DEFAULT = os.environ.get("ZTF_CHEN_HDF5", str(REPO / "data" / "chen2020.hdf5"))
OUT_PRED = os.environ.get("PROBE_PRED_OUT", str(REPO / "results" / "probe_predictions.json"))
OUT_REPORT = str(REPO / "results/chen2020_25k/probe_report.json")


def safe_str(x):
    if isinstance(x, bytes):
        x = x.decode("utf-8", errors="replace")
    s = str(x)
    if len(s) >= 3 and s.startswith("b'") and s.endswith("'"): return s[2:-1]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", default=EMB_DEFAULT)
    ap.add_argument("--chen_hdf5", default=CHEN_DEFAULT)
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--out_pred", default=OUT_PRED)
    ap.add_argument("--out_report", default=OUT_REPORT)
    args = ap.parse_args()

    emb = np.load(args.embeddings)
    z_sig = emb["z_sig"]
    chen_indices = emb["chen_indices"]
    print(f"z_sig: {z_sig.shape}")

    with h5py.File(args.chen_hdf5, "r") as f:
        var_type_raw = f["catalog/var_type"][:]
    var_type = np.array([safe_str(s) for s in var_type_raw])
    y = var_type[chen_indices]
    classes = sorted(np.unique(y))
    print(f"classes: {classes}")
    for c in classes:
        print(f"  {c}: {(y == c).sum():,}")

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(z_sig)

    # Logistic regression cross-val predict
    print("\n→ Logistic regression (5-fold CV)…")
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42, n_jobs=-1)
    y_pred_logr = cross_val_predict(clf, X, y, cv=skf, n_jobs=-1)
    bacc_logr = balanced_accuracy_score(y, y_pred_logr)
    print(f"  balanced accuracy: {bacc_logr:.4f}")

    # k-NN
    print("\n→ k-NN (k=10, cosine)…")
    knn = KNeighborsClassifier(n_neighbors=10, metric="cosine", n_jobs=-1)
    y_pred_knn = cross_val_predict(knn, X, y, cv=skf, n_jobs=-1)
    bacc_knn = balanced_accuracy_score(y, y_pred_knn)
    print(f"  balanced accuracy: {bacc_knn:.4f}")

    # Compare: what's the better of the two? Use that for predictions.
    use_pred = y_pred_knn if bacc_knn > bacc_logr else y_pred_logr
    cm = confusion_matrix(y, use_pred, labels=classes)
    rep = classification_report(y, use_pred, labels=classes, output_dict=True)

    print(f"\nConfusion matrix (rows=true, cols=pred), classes order = {classes}")
    print(cm)
    print("\nPer-class report:")
    for c in classes:
        r = rep.get(c, {})
        print(f"  {c:>8s}  P={r.get('precision', 0):.3f}  R={r.get('recall', 0):.3f}  "
              f"F1={r.get('f1-score', 0):.3f}  N={r.get('support', 0):.0f}")

    # Save predictions for demo. Use positional idx (= position in points.json).
    use_proba_clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42, n_jobs=-1).fit(X, y)
    proba = use_proba_clf.predict_proba(X)
    proba_idx_of = {c: i for i, c in enumerate(use_proba_clf.classes_.tolist())}

    # Map to demo idx (which equals positional index in points.json since
    # success_mask was all True in the 25K embeddings).
    preds_demo = []
    for i in range(len(y)):
        preds_demo.append({
            "idx": i,
            "true": str(y[i]),
            "pred_logr": str(y_pred_logr[i]),
            "pred_knn":  str(y_pred_knn[i]),
            "proba": [round(float(v), 3) for v in proba[i]],
        })
    out = {
        "classes": list(use_proba_clf.classes_),
        "logr_balanced_accuracy": float(bacc_logr),
        "knn_balanced_accuracy":  float(bacc_knn),
        "predictions": preds_demo,
    }
    Path(args.out_pred).write_text(json.dumps(out, separators=(",", ":")))
    print(f"\nWrote {args.out_pred} ({Path(args.out_pred).stat().st_size/1e6:.1f} MB)")

    # Report (smaller, no per-source predictions)
    report = {
        "logr_balanced_accuracy": float(bacc_logr),
        "knn_balanced_accuracy":  float(bacc_knn),
        "classes": classes,
        "confusion_matrix": cm.tolist(),
        "per_class": {c: rep.get(c, {}) for c in classes},
    }
    Path(args.out_report).write_text(json.dumps(report, indent=2))
    print(f"Wrote {args.out_report}")


if __name__ == "__main__":
    main()
