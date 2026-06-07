#!/usr/bin/env python3
"""
embed_moment_from_scratch_25k.py — embed Chen 25K subset using a
MOMENT-from-scratch checkpoint trained by src/baselines/moment_from_scratch.py.

Mirrors embed_chronos_t5_25k.py's interface. Reconstructs the MOMENT model
from the saved moment_config + state_dict, runs the embed() call (mean-pool of
encoder hidden states across non-masked patches), saves z_sig in our standard
schema for probe_25k_from_pred.py.
"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from momentfm import MOMENT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache", required=True,
                    help="Path to lc_cache_25k.npz")
    ap.add_argument("--output", required=True)
    ap.add_argument("--seq_len", type=int, default=384)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading checkpoint {args.checkpoint} → device={device}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg_dict = dict(ckpt["moment_config"])
    # Keep training-time task_name ("reconstruction"); .embed() is a method,
    # not a task. The reconstruction head's weights are loaded but unused
    # during embedding, which is fine.
    config = SimpleNamespace(**cfg_dict)

    model = MOMENT(config).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  loaded MOMENT ({n_params:,} params), best epoch={ckpt.get('epoch', '?')}")

    print(f"Loading cache from {args.cache}")
    data = np.load(args.cache)
    fluxes = data["fluxes"]
    n_obs = data["n_obs"]
    chen_indices = data["chen_indices"]
    N, max_len_cache = fluxes.shape
    L = min(args.seq_len, max_len_cache)
    print(f"  N = {N:,}, seq_len = {L}")

    embeddings = None
    with torch.no_grad():
        for i0 in range(0, N, args.batch_size):
            i1 = min(i0 + args.batch_size, N)
            B = i1 - i0

            x = torch.from_numpy(fluxes[i0:i1, :L].astype(np.float32))
            mask = torch.zeros((B, L), dtype=torch.float32)
            for j, n in enumerate(n_obs[i0:i1]):
                mask[j, :min(int(n), L)] = 1.0

            x = x.unsqueeze(1).to(device)        # (B, 1, L)
            mask = mask.to(device)               # (B, L)

            out = model.embed(x_enc=x, input_mask=mask, reduction="mean")
            z = out.embeddings.float().cpu().numpy()  # (B, d_model)

            if embeddings is None:
                embeddings = np.full((N, z.shape[1]), np.nan, dtype=np.float32)
                print(f"  embedding dim D = {z.shape[1]}")
            embeddings[i0:i1] = z

            if (i0 // args.batch_size) % 20 == 0:
                print(f"  batch {i0:>6}/{N}", flush=True)

    print(f"\nEmbeddings shape: {embeddings.shape}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        z_sig=embeddings,
        z_qual=np.zeros((N, 32), dtype=np.float32),
        n_obs=n_obs,
        success_mask=np.ones(N, dtype=bool),
        chen_indices=chen_indices,
        checkpoint_path=str(args.checkpoint),
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
