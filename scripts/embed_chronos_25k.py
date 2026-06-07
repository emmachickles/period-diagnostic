#!/usr/bin/env python3
"""
embed_chronos_25k.py — Chronos (Amazon) baseline. T5-based time-series
foundation model. Loaded via the chronos-forecasting package.

Adaptation: feed the variable-length flux sequence (no padding needed
— Chronos handles variable-length context natively). Time and σ are
ignored. Use the pipeline's built-in embed() endpoint to get a
representation per source.

Output: results/chen2020_25k/embeddings_chronos.npz
"""
import os
import argparse
from pathlib import Path

import numpy as np
import torch

from chronos import ChronosPipeline

REPO = Path(os.environ.get("PERIOD_DIAG_ROOT", str(Path(__file__).resolve().parents[1])))
RESULTS = REPO / "results/chen2020_25k"
CACHE = RESULTS / "lc_cache_25k.npz"
OUTPUT_DEFAULT = RESULTS / "embeddings_chronos.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(CACHE))
    ap.add_argument("--model", default="amazon/chronos-t5-small")
    ap.add_argument("--output", default=str(OUTPUT_DEFAULT))
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_context", type=int, default=512)
    args = ap.parse_args()

    print(f"Loading cache from {args.cache}")
    data = np.load(args.cache)
    fluxes = data["fluxes"]
    masks = data["masks"]
    success = data["success"]
    chen_indices = data["chen_indices"]
    n_obs = data["n_obs"]
    N = len(success)
    print(f"  N = {N:,}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Chronos: {args.model}")
    pipeline = ChronosPipeline.from_pretrained(
        args.model,
        device_map=device,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )

    # Per-source variable-length tensors; truncate to max_context
    contexts = []
    for i in range(N):
        if not success[i]:
            contexts.append(torch.zeros(1, dtype=torch.float32))
            continue
        nobs = int(n_obs[i])
        L = min(nobs, args.max_context)
        x = fluxes[i, :nobs]
        if nobs > L:
            s = (nobs - L) // 2
            x = fluxes[i, s:s+L]
        contexts.append(torch.from_numpy(x.astype(np.float32)))

    print(f"Computing Chronos embeddings (mean-pooled across context)...")
    embeddings = None
    for i0 in range(0, N, args.batch_size):
        i1 = min(i0 + args.batch_size, N)
        batch = contexts[i0:i1]
        # ChronosPipeline.embed returns (B, T, D) hidden states + scale
        try:
            emb_t, _ = pipeline.embed(batch)  # (B, T, D)
            # mean-pool over time, ignoring zeros
            emb = emb_t.float().mean(dim=1).cpu().numpy()
        except Exception as e:
            print(f"  batch {i0}: failed ({e}); filling NaN")
            emb = np.full((i1 - i0, embeddings.shape[1] if embeddings is not None else 1024), np.nan, dtype=np.float32)
        if embeddings is None:
            D = emb.shape[1]
            embeddings = np.full((N, D), np.nan, dtype=np.float32)
            print(f"  embedding dim D = {D}")
        embeddings[i0:i1] = emb
        if (i0 // args.batch_size) % 20 == 0:
            print(f"  batch {i0:>6}/{N}", flush=True)

    print(f"\nEmbeddings shape: {embeddings.shape}")
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
