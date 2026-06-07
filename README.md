# Template Matching, Not Time Learning

**A diagnostic for self-supervised light-curve encoders.**

[![Paper (PDF)](https://img.shields.io/badge/paper-PDF-b5179e)](paper/template-matching-not-time-learning.pdf)
[![arXiv](https://img.shields.io/badge/arXiv-pending-7c3aed)](https://arxiv.org/abs/PENDING)
[![Venue](https://img.shields.io/badge/AI4Physics%20%40%20ICML-2026-2563eb)](https://ai4physicsicml.github.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-168a5a)](LICENSE)

> Accepted at the **AI4Physics workshop, ICML 2026**.
> 📄 [Read the paper](paper/template-matching-not-time-learning.pdf) · arXiv link coming soon.

---

## TL;DR

Period regression on a multi-class variable-star catalog *looks* like a strong
test of whether a self-supervised encoder has "learned time" from irregular
photometry — but the headline **R²** conflates two very different behaviors:

1. **Class-template matching** — the encoder identifies the variability *class*
   and the probe predicts that class's *mean* period (period "for free").
2. **Genuine time learning** — the encoder reads period off an *individual*
   source's time series.

This repo introduces a cheap, drop-in **two-axis diagnostic** that separates
them, and applies it to **eight encoders** (from a 4.4M cadence-as-channel
BiGRU to the 110M pretrained MOMENT). The verdict is unanimous:

- **60–70% of period R² is between-class** (class-template matching).
- **Within-class Spearman ρ ≤ 0.26** for *every* method — even pretrained
  Chronos (46M, ~10⁹ cross-domain steps) sits at 0.23.
- A direct **Lomb–Scargle extractor reaches ρ = 0.78** on the same labels, so
  the signal *is* recoverable — the encoders are **under-extracting**, not
  saturating a label-noise floor.

**No tested SSL encoder has learned absolute time on irregular ZTF photometry.**
We propose reporting the within-class diagnostic alongside R², and treating
**ρ_within as the metric of merit** for the time-encoding question.

## The diagnostic in one paragraph

Given a probe's predicted `log10(P)`, decompose its variance into a
between-class part and a within-class part:

- **f_cls** = between-class fraction of predicted-period variance → 1.0 means
  pure class-template matching.
- **ρ_within** = median (over classes) within-class Spearman correlation
  between true and predicted period → 1.0 means genuine source-level time
  learning.

Learned absolute time ⟺ high R² **and** low f_cls **and** ρ_within → 1.
Core implementation: [`scripts/diag_period_template_match.py`](scripts/diag_period_template_match.py).

## Repository layout

```
src/model/        Continuous-time SSL transformer, Fourier time embedding, losses
src/baselines/    From-scratch encoders: 1D CNN, BiGRU, Chronos-T5, PatchTST, MOMENT
src/data/         ZTF light-curve pipeline: loaders, preprocessing, augmentations
src/training/     DDP training entry points
scripts/          The contribution: diagnostic + probes + per-encoder embedding
paper/            Camera-ready PDF
```

Key scripts:

| Script | What it does |
|---|---|
| `scripts/diag_period_template_match.py` | The diagnostic: `f_cls`, `ρ_within` |
| `scripts/diag_mlp_and_ci.py` | Linear vs. nonlinear probe with bootstrap 95% CIs |
| `scripts/diag_seed_ci.py` | Encoder-training-seed robustness |
| `scripts/diag_per_layer.py` | Per-layer probe (rules out pooling artifacts) |
| `scripts/probe_classical_rf_full.py` | Classical RF + Lomb–Scargle reference ceiling |
| `scripts/embed_*_25k.py` | One embedding script per evaluated encoder |

## Install

```bash
pip install -r requirements.txt
```

Pretrained-FM baselines (Chronos, MOMENT) need extra packages — see the
commented lines in `requirements.txt`. They are **not** required to reproduce
the diagnostic itself.

## Reproducing the paper

The diagnostic runs on **frozen embeddings of a 25K-source, 11-class subset of
the [Chen et al. 2020](https://doi.org/10.3847/1538-4365/ab9cae) ZTF periodic
variable catalog**. The embeddings (~1.1 GB) and the cached light curves are
distributed via **Zenodo** ([record pending](#)), not git.

```bash
# 1. Download the data release from Zenodo into ./results/chen2020_25k/
#    (embeddings + lc_cache + labels_chen2020_25k.npz). The bundled labels file
#    means the full Chen+2020 catalog is NOT required to reproduce the paper.
# 2. Run the diagnostic on any encoder's embedding:
python scripts/diag_period_template_match.py
# 3. Linear-vs-MLP probe with confidence intervals:
python scripts/diag_mlp_and_ci.py
```

The `diag_*` scripts use the bundled `labels_chen2020_25k.npz` automatically and
fall back to `$ZTF_CHEN_HDF5` only if it is absent.

Paths resolve automatically from the repo location; override any of them with
environment variables if your data lives elsewhere:

| Variable | Default | Purpose |
|---|---|---|
| `PERIOD_DIAG_ROOT` | repo root (from `__file__`) | base for `results/`, `data/` |
| `ZTF_CHEN_HDF5` | `$PERIOD_DIAG_ROOT/data/chen2020.hdf5` | Chen+2020 catalog |
| `PROBE_PRED_OUT` | `$PERIOD_DIAG_ROOT/results/probe_predictions.json` | probe-prediction dump |
| `ASTROTOOLS_PATH` | unset | optional local helper, only for *raw* matchfile ingestion |

The diagnostic and probes depend only on the released embeddings — raw ZTF
matchfile ingestion (the `astrotools`-dependent scripts) is not needed to
reproduce the paper's results.

## Citation

```bibtex
@inproceedings{chickles2026template,
  title     = {Template Matching, Not Time Learning:
               A Diagnostic for Self-Supervised Light-Curve Encoders},
  author    = {Chickles, Emma and Audenaert, Jeroen},
  booktitle = {AI4Physics Workshop, ICML},
  year      = {2026}
}
```

## Acknowledgements

Thanks to Jeroen Audenaert for the dual-encoder structured-contrastive
framework underlying the disentangled loss, and to Kevin Burdge for guidance
and access to the ZTF data infrastructure. This work used the MIT ORCD/Engaging
cluster.
