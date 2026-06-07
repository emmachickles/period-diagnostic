#!/usr/bin/env python3
"""Embed the Chen-2020 25K subset with a PUBLISHED astronomy-SSL checkpoint
(ASTROMER) so the diagnostic can be run on a released astro-SSL model
(Reviewer uEcd's #1 con). Output schema mirrors the other embedders
(`z_sig` + `chen_indices`), so it drops straight into
scripts/diag_period_template_match.py and scripts/diag_mlp_and_ci.py.

Why this is feasible locally: results/chen2020_25k/lc_cache_25k.npz already
holds the raw light curves (times, fluxes, flux_errs, masks) for the exact
25,565 probe sources. We only need to (1) convert flux -> magnitude, (2)
format per-source (time, mag, magerr) sequences, (3) run ASTROMER's encoder,
(4) mean-pool the attention output over valid epochs.

ASTROMER is single-band. We use the ZTF r-band-equivalent stream already
median-matched in the cache (the cache is a single merged g+r+i flux series
per source after band median-matching; ASTROMER sees it as one band). If you
prefer strict single-band, regenerate the cache with --band r upstream.

IMPORTANT — two things to confirm before trusting the numbers:
  * Flux -> mag conversion below drops non-positive (forced-photometry) flux.
    Sanity-check that enough epochs survive per source (the dry-run prints this).
  * The ASTROMER API differs between the v1 `ASTROMER` pip package and the
    newer `astromer` repo. The encode() call is isolated in `run_astromer()`;
    adjust there for your installed version. Both common APIs are sketched.

Usage:
  # 1) validate data-prep with NO ASTROMER installed (recommended first):
  python scripts/embed_astromer_25k.py --dry_run
  # 2) real run (needs ASTROMER + weights + ideally a GPU):
  python scripts/embed_astromer_25k.py \
      --weights macho \
      --output results/chen2020_25k/embeddings_astromer.npz
"""
import argparse
import json
import os
from pathlib import Path

# ASTROMER 0.1.8 was written for Keras 2; route tf.keras to the legacy API so
# its custom Encoder layer builds under TF 2.21 / Keras 3. Must be set before
# any tensorflow/ASTROMER import.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "results/chen2020_25k/lc_cache_25k.npz"


def flux_to_mag(times, fluxes, flux_errs, masks, max_len=200, min_epochs=10):
    """Build ASTROMER-style (time, mag, magerr) sequences from cached flux.

    Returns a list of (L_i, 3) float32 arrays and the row index kept, plus a
    boolean `ok` array (True where >= min_epochs survived) aligned to all rows.
    """
    n = times.shape[0]
    seqs, ok = [], np.zeros(n, dtype=bool)
    survived = []
    for i in range(n):
        m = masks[i].astype(bool) & np.isfinite(fluxes[i]) & (fluxes[i] > 0)
        m &= np.isfinite(times[i]) & np.isfinite(flux_errs[i]) & (flux_errs[i] > 0)
        ne = int(m.sum())
        survived.append(ne)
        if ne < min_epochs:
            seqs.append(None)
            continue
        t = times[i][m].astype(np.float64)
        f = fluxes[i][m].astype(np.float64)
        fe = flux_errs[i][m].astype(np.float64)
        mag = -2.5 * np.log10(f)                       # relative magnitude
        magerr = (2.5 / np.log(10.0)) * (fe / f)        # error propagation
        t = t - t.min()                                 # zero-base time
        order = np.argsort(t)
        seq = np.stack([t[order], mag[order], magerr[order]], axis=1)
        if seq.shape[0] > max_len:                      # keep most recent max_len
            seq = seq[-max_len:]
        seqs.append(seq.astype(np.float32))
        ok[i] = True
    return seqs, ok, np.array(survived)


def run_astromer(seqs_ok, weights_dir, batch_size):
    """Return (n_ok, d_model) mean-pooled ASTROMER embeddings.

    Loads the pretrained ASTROMER from a LOCAL weights folder (conf.json +
    weights.*), replicating SingleBandEncoder.from_pretraining() without its
    hard-coded download URL (which is stale: the weights repo renamed
    macho.zip -> macho_a0.zip and serves via a path the package no longer
    knows). Fetch+unzip the folder once with --fetch_weights, or point
    --weights_dir at an existing one.

    encode() returns, per source, a (n_windows, max_obs, d_model) attention
    tensor (long curves are split into windows); we mean-pool over windows and
    timesteps to one d_model vector per source.
    """
    try:
        from ASTROMER.models import SingleBandEncoder
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            f"Could not import ASTROMER ({e}). Install it (pip install ASTROMER "
            "tf-keras tensorboard) in an env with TF. Use --dry_run to validate "
            "data-prep without ASTROMER.")

    conf = json.load(open(os.path.join(weights_dir, "conf.json")))
    model = SingleBandEncoder(
        num_layers=conf["layers"], d_model=conf["head_dim"],
        num_heads=conf["heads"], dff=conf["dff"], base=conf["base"],
        dropout=conf["dropout"], maxlen=conf["max_obs"])
    model.load_weights(weights_dir)
    print(f"Loaded ASTROMER from {weights_dir} "
          f"(d_model={conf['head_dim']}, max_obs={conf['max_obs']})")

    # encode() with concatenate=True groups each object's windows back into one
    # variable-length (n_obs, d_model) attention array, returned in
    # np.unique(ids) (lexicographic) order. Zero-pad the ids so lexicographic
    # order == source order, then mean-pool each source's per-obs embeddings.
    w = len(str(len(seqs_ok)))
    oids = [f"{i:0{w}d}" for i in range(len(seqs_ok))]
    print(f"  encoding {len(seqs_ok)} sources (concatenate by oid)...", flush=True)
    att = model.encode(seqs_ok, oids_list=oids, batch_size=batch_size,
                       concatenate=True)  # list of (n_obs_i, d_model), oid-sorted
    if len(att) != len(seqs_ok):
        raise RuntimeError(f"encode returned {len(att)} objects, expected "
                           f"{len(seqs_ok)} -- oid grouping mismatch")
    embs = []
    for a in att:
        a = np.asarray(a)
        embs.append(a.mean(axis=0) if a.size else np.full(a.shape[-1], np.nan))
    return np.stack(embs, axis=0)


def fetch_weights(name="macho_a0", dest="weights"):
    """Download + unzip the ASTROMER weights from the (current) repo layout.
    Returns the extracted folder path (the zip's top dir, e.g. weights/macho)."""
    import json as _json
    import urllib.request
    import zipfile
    os.makedirs(dest, exist_ok=True)
    api = (f"https://api.github.com/repos/astromer-science/weights/"
           f"contents/{name}.zip?ref=main")
    req = urllib.request.Request(api, headers={"User-Agent": "curl"})
    url = _json.load(urllib.request.urlopen(req, timeout=30))["download_url"]
    zpath = os.path.join(dest, f"{name}.zip")
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, zpath)
    with zipfile.ZipFile(zpath) as z:
        top = z.namelist()[0].split("/")[0]
        z.extractall(dest)
    folder = os.path.join(dest, top)
    print(f"Extracted to {folder}")
    return folder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(CACHE))
    ap.add_argument("--output", default=str(REPO / "results/chen2020_25k/embeddings_astromer.npz"))
    ap.add_argument("--weights_dir", default=str(REPO / "weights/macho"),
                    help="Local ASTROMER weights folder (conf.json + weights.*).")
    ap.add_argument("--fetch_weights", default=None,
                    help="If set (e.g. 'macho_a0'), download+unzip that weights "
                         "archive from the astromer-science/weights repo first.")
    ap.add_argument("--max_len", type=int, default=200)
    ap.add_argument("--min_epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--dry_run", action="store_true",
                    help="Build + validate sequences locally; skip ASTROMER.")
    args = ap.parse_args()

    d = np.load(args.cache)
    times, fluxes = d["times"], d["fluxes"]
    flux_errs, masks = d["flux_errs"], d["masks"]
    chen_indices = d["chen_indices"]
    print(f"Loaded cache: {times.shape[0]} sources, padded len {times.shape[1]}")

    seqs, ok, survived = flux_to_mag(
        times, fluxes, flux_errs, masks,
        max_len=args.max_len, min_epochs=args.min_epochs)
    print(f"Sources with >= {args.min_epochs} positive-flux epochs: "
          f"{ok.sum()} / {len(ok)}  ({100*ok.mean():.1f}%)")
    print(f"Median surviving epochs/source: {np.median(survived):.0f}  "
          f"(min {survived.min()}, max {survived.max()})")
    if ok.sum() < len(ok):
        print(f"  NOTE: {(~ok).sum()} sources dropped (too few positive-flux "
              "epochs); they will be NaN in z_sig and skipped by the probe.")

    if args.dry_run:
        ex = next(s for s in seqs if s is not None)
        print("\nDRY RUN ok. Example sequence shape:", ex.shape,
              "| mag range %.2f..%.2f" % (ex[:, 1].min(), ex[:, 1].max()),
              "| dt(days) median %.3f" % np.median(np.diff(np.sort(ex[:, 0]))))
        print("Re-run without --dry_run (with ASTROMER installed) to embed.")
        return

    seqs_ok = [s for s in seqs if s is not None]
    weights_dir = args.weights_dir
    if args.fetch_weights:
        weights_dir = fetch_weights(args.fetch_weights)
    print(f"Running ASTROMER on {len(seqs_ok)} sources from {weights_dir}")
    emb_ok = run_astromer(seqs_ok, weights_dir, args.batch_size)

    d_emb = emb_ok.shape[1]
    z_sig = np.full((len(ok), d_emb), np.nan, dtype=np.float32)
    z_sig[ok] = emb_ok
    np.savez(args.output, z_sig=z_sig, chen_indices=chen_indices,
             success_mask=ok)
    print(f"Wrote {args.output}  (z_sig {z_sig.shape}, {ok.sum()} valid rows)")
    print("Next: add ('Pretrained ASTROMER', 'embeddings_astromer.npz') to "
          "METHODS in scripts/diag_period_template_match.py and "
          "scripts/diag_mlp_and_ci.py, then re-run them.")


if __name__ == "__main__":
    main()
