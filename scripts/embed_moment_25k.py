#!/usr/bin/env python3
"""
embed_moment_25k.py — MOMENT (AutonLab) baseline. Pulls the pretrained
foundation model from HuggingFace and runs inference on the cached
25K Chen subset, saving embeddings in our standard schema.

MOMENT was pretrained on regularly-sampled univariate time series.
Our adaptation: feed the flux channel as the univariate signal,
center-crop or zero-pad to MOMENT's required context length (512),
use the mask to ignore padding. Time and σ channels are NOT given to
the model — that's the most honest interpretation of "this FM was
not designed for irregular sampling."

Output: results/chen2020_25k/embeddings_moment.npz
"""
import os
import argparse
from pathlib import Path

import numpy as np
import torch

from momentfm import MOMENTPipeline

REPO = Path(os.environ.get("PERIOD_DIAG_ROOT", str(Path(__file__).resolve().parents[1])))
RESULTS = REPO / "results/chen2020_25k"
CACHE = RESULTS / "lc_cache_25k.npz"
OUTPUT_DEFAULT = RESULTS / "embeddings_moment.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(CACHE))
    ap.add_argument("--model", default="AutonLab/MOMENT-1-large")
    ap.add_argument("--output", default=str(OUTPUT_DEFAULT))
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seq_len", type=int, default=512)
    args = ap.parse_args()

    print(f"Loading cache from {args.cache}")
    data = np.load(args.cache)
    fluxes = data["fluxes"]      # (N, max_len)
    masks = data["masks"]        # (N, max_len) bool
    success = data["success"]    # (N,) bool
    chen_indices = data["chen_indices"]
    n_obs = data["n_obs"]
    N, max_len = fluxes.shape
    print(f"  N = {N:,}, max_len = {max_len}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading MOMENT: {args.model}")
    model = MOMENTPipeline.from_pretrained(
        args.model,
        model_kwargs={"task_name": "embedding"},
    )
    model.init()
    model = model.to(device).eval()

    # MOMENT default context length is 512; we have max_len=1024 in the cache
    L = args.seq_len
    print(f"Using context length {L}; sources with longer LCs will be center-cropped.")

    # Prepare model input: (B, n_channels=1, seq_len) flux only
    x_all = np.zeros((N, 1, L), dtype=np.float32)
    m_all = np.zeros((N, L), dtype=np.float32)
    for i in range(N):
        if not success[i]:
            continue
        nobs = int(n_obs[i])
        if nobs >= L:
            # center crop
            s = (nobs - L) // 2
            x_all[i, 0, :L] = fluxes[i, s:s+L]
            m_all[i, :L] = masks[i, s:s+L].astype(np.float32)
        else:
            x_all[i, 0, :nobs] = fluxes[i, :nobs]
            m_all[i, :nobs] = masks[i, :nobs].astype(np.float32)

    # Inference in batches; save mean-pooled embeddings
    embeddings = None
    with torch.no_grad():
        for i0 in range(0, N, args.batch_size):
            i1 = min(i0 + args.batch_size, N)
            x = torch.from_numpy(x_all[i0:i1]).to(device)
            m = torch.from_numpy(m_all[i0:i1]).to(device)
            out = model(x_enc=x, input_mask=m)
            emb = out.embeddings.cpu().numpy()  # (B, D)
            if embeddings is None:
                D = emb.shape[1]
                embeddings = np.full((N, D), np.nan, dtype=np.float32)
                print(f"  embedding dim D = {D}")
            embeddings[i0:i1] = emb
            if (i0 // args.batch_size) % 20 == 0:
                print(f"  batch {i0:>6}/{N}", flush=True)

    print(f"\n{success.sum():,}/{N:,} sources have valid LC; embeddings shape {embeddings.shape}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output,
             z_sig=embeddings,
             z_qual=np.zeros((N, 32), dtype=np.float32),
             n_obs=n_obs,
             success_mask=success,
             chen_indices=chen_indices)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
