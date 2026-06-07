#!/usr/bin/env python3
"""Camera-ready additions for the ICML reviews:

  (1) Non-linear (MLP) period-regression probe alongside the linear ridge
      probe, to test whether period information is present-but-nonlinearly-
      decodable rather than absent (Reviewers JuTb & uEcd).
  (2) Multi-seed (data-split) point estimates + bootstrap 95% CIs on
      R^2, f_cls, and rho_within (Reviewers JuTb & uEcd: "3-5 seeds, 95% CIs").

The encoder training seed is fixed (42); these CIs quantify sensitivity to
the probe's CV split and to the finite labeled sample, which is what the
reviewers asked be reported. Encoder-retraining seeds remain N=1 (stated
as a limitation).

Outputs a markdown-ish table to stdout and a JSON blob for the manuscript.
"""
import os
import json
import sys
from pathlib import Path

import numpy as np
import h5py
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

REPO = Path(os.environ.get("PERIOD_DIAG_ROOT", str(Path(__file__).resolve().parents[1])))
RESULTS = REPO / "results/chen2020_25k"
LC_CACHE = RESULTS / "lc_cache_25k.npz"
CHEN_HDF5 = os.environ.get("ZTF_CHEN_HDF5", str(REPO / "data" / "chen2020.hdf5"))

METHODS = [
    ("Pretrained Chronos",        "embeddings_chronos.npz"),
    ("BiGRU (ours)",              "embeddings_rnn_v1.npz"),
    ("1D CNN (ours)",             "embeddings_cnn_v1.npz"),
    ("Pretrained MOMENT",         "embeddings_moment.npz"),
    ("CT-SSL (ours)",             "embeddings_ct_v3_e3.npz"),
    ("Chronos-T5 (from scratch)", "embeddings_chronos_t5_e10.npz"),
    ("PatchTST (fs)",             "embeddings_patchtst_v3scale.npz"),
]

N_SEEDS = 5          # CV-split seeds for split-robustness (reviewer: 3-5)
N_BOOT = 1000        # bootstrap resamples for the 95% CI
SEEDS = [42, 0, 1, 7, 2024][:N_SEEDS]


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
        period_key = next(k for k in ("Per_g", "Per_r", "period_g",
                                      "period_r", "period", "P") if k in keys)
        period = cat[period_key][:]

    def safe_str(x):
        if isinstance(x, bytes):
            x = x.decode("utf-8", errors="replace")
        s = str(x)
        if len(s) >= 3 and s.startswith("b'") and s.endswith("'"):
            return s[2:-1]
        return s

    var_type = np.array([safe_str(s) for s in var_type_raw])
    return var_type[chen_indices], period[chen_indices]


def metrics_from_pred(y_true, y_pred, y_cls, big_classes):
    """Return (R2, f_cls, rho_within) for a single prediction vector."""
    r2 = r2_score(y_true, y_pred)
    gmp = y_pred.mean()
    ss_between, n_total = 0.0, 0
    rho_list = []
    for c in big_classes:
        m = (y_cls == c)
        if m.sum() < 30:
            continue
        ss_between += m.sum() * (y_pred[m].mean() - gmp) ** 2
        n_total += m.sum()
        rho, _ = spearmanr(y_true[m], y_pred[m])
        rho_list.append(rho)
    ss_total = n_total * y_pred.var()
    f_cls = ss_between / ss_total if ss_total > 0 else np.nan
    rho_within = float(np.median(rho_list))
    return r2, f_cls, rho_within


def bootstrap_ci(y_true, y_pred, y_cls, big_classes, n_boot=N_BOOT, seed=42):
    """Bootstrap over sources -> 95% CI on each metric (fixed OOF preds)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    acc = {"R2": [], "f_cls": [], "rho": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        r2, f, rho = metrics_from_pred(y_true[idx], y_pred[idx],
                                       y_cls[idx], big_classes)
        acc["R2"].append(r2); acc["f_cls"].append(f); acc["rho"].append(rho)
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
            for k, v in acc.items()}


def make_probe(kind, seed):
    if kind == "ridge":
        return make_pipeline(StandardScaler(),
                             RidgeCV(alphas=np.logspace(-2, 4, 13)))
    # MLP: one hidden layer, early stopping, standardized input.
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(256,), activation="relu",
                     alpha=1e-3, learning_rate_init=1e-3, max_iter=300,
                     early_stopping=True, n_iter_no_change=12,
                     validation_fraction=0.1, random_state=seed))


def run():
    y, P = load_labels_and_periods()
    log_P = np.log10(P)
    keep = np.isfinite(log_P) & (P > 0)
    y, log_P = y[keep], log_P[keep]
    classes, counts = np.unique(y, return_counts=True)
    big = classes[counts >= 30]

    out = {}
    for kind in ("ridge", "mlp"):
        print(f"\n{'='*78}\nPROBE = {kind.upper()}\n{'='*78}")
        hdr = (f"{'method':<26s} {'R2 mean(sd)':>14s} {'R2 95%CI':>16s} "
               f"{'f_cls mean':>11s} {'rho mean(sd)':>14s} {'rho 95%CI':>16s}")
        print(hdr); print("-" * len(hdr))
        out[kind] = {}
        for name, npz in METHODS:
            path = RESULTS / npz
            if not path.exists():
                print(f"{name:<26s}  (missing)"); continue
            z = np.load(path)["z_sig"][keep]

            per_seed = {"R2": [], "f_cls": [], "rho": []}
            pred0 = None
            for si, seed in enumerate(SEEDS):
                kf = KFold(n_splits=5, shuffle=True, random_state=seed)
                probe = make_probe(kind, seed)
                yp = cross_val_predict(probe, z, log_P, cv=kf,
                                       n_jobs=-1 if kind == "ridge" else 4)
                if si == 0:
                    pred0 = yp
                r2, f, rho = metrics_from_pred(log_P, yp, y, big)
                per_seed["R2"].append(r2)
                per_seed["f_cls"].append(f)
                per_seed["rho"].append(rho)

            ci = bootstrap_ci(log_P, pred0, y, big)
            rec = {
                "R2_mean": float(np.mean(per_seed["R2"])),
                "R2_sd":   float(np.std(per_seed["R2"], ddof=1)),
                "R2_ci":   ci["R2"],
                "fcls_mean": float(np.mean(per_seed["f_cls"])),
                "fcls_sd":   float(np.std(per_seed["f_cls"], ddof=1)),
                "fcls_ci":   ci["f_cls"],
                "rho_mean": float(np.mean(per_seed["rho"])),
                "rho_sd":   float(np.std(per_seed["rho"], ddof=1)),
                "rho_ci":   ci["rho"],
            }
            out[kind][name] = rec
            print(f"{name:<26s} "
                  f"{rec['R2_mean']:.3f}({rec['R2_sd']:.3f})".rjust(14) + " "
                  f"[{ci['R2'][0]:.3f},{ci['R2'][1]:.3f}]".rjust(16) + " "
                  f"{rec['fcls_mean']:.3f}".rjust(11) + " "
                  f"{rec['rho_mean']:.3f}({rec['rho_sd']:.3f})".rjust(14) + " "
                  f"[{ci['rho'][0]:.3f},{ci['rho'][1]:.3f}]".rjust(16))

    with open(RESULTS / "mlp_and_ci_results.json", "w") as f:
        json.dump({"seeds": SEEDS, "n_boot": N_BOOT, "results": out}, f, indent=2)
    print(f"\nWrote {RESULTS/'mlp_and_ci_results.json'}")


if __name__ == "__main__":
    run()
