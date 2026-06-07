#!/usr/bin/env python3
"""
embed_seq_baseline_25k.py — embed Chen 25K subset using CNN1DEncoder or
RNNEncoder trained by src/baselines/train_seq_baseline.py.

Mirrors embed_chronos_t5_25k.py's interface: takes --checkpoint, --cache,
--output, plus encoder-arch CLI args matching the slurm training script.
Output schema is z_sig + chen_indices (probe-compatible with probe_25k_from_pred.py).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.baselines.seq_encoders import build_encoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, choices=["cnn", "rnn"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache", required=True,
                    help="Path to lc_cache_25k.npz (times, fluxes, flux_errs, n_obs, chen_indices)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_seq_len", type=int, default=384)
    ap.add_argument("--batch_size", type=int, default=64)

    # Encoder hyperparams (must match training-time values)
    ap.add_argument("--d_model", type=int, default=352)
    ap.add_argument("--num_layers", type=int, default=6)
    ap.add_argument("--kernel_size", type=int, default=7, help="CNN only")
    ap.add_argument("--bidirectional", type=int, default=1, help="RNN only")
    ap.add_argument("--d_sig", type=int, default=128)
    ap.add_argument("--d_qual", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="Set 0 for inference; matters only because the modules' Dropout layers exist.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading checkpoint {args.checkpoint} → device={device}")

    encoder_kwargs = dict(
        d_model=args.d_model, num_layers=args.num_layers,
        d_sig=args.d_sig, d_qual=args.d_qual, dropout=args.dropout,
    )
    if args.encoder == "cnn":
        encoder_kwargs["kernel_size"] = args.kernel_size
    elif args.encoder == "rnn":
        encoder_kwargs["bidirectional"] = bool(args.bidirectional)

    model = build_encoder(args.encoder, **encoder_kwargs).to(device).eval()

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  loaded {args.encoder} ({n_params:,} params), best epoch={ckpt.get('epoch', '?')}")

    print(f"Loading cache from {args.cache}")
    data = np.load(args.cache)
    times = data["times"]
    fluxes = data["fluxes"]
    flux_errs = data["flux_errs"]
    n_obs = data["n_obs"]
    chen_indices = data["chen_indices"]
    N, max_len_cache = times.shape
    print(f"  N = {N:,}, cache max_len = {max_len_cache}")

    L = min(args.max_seq_len, max_len_cache)
    print(f"Embedding with seq_len={L}...")

    embeddings = None
    with torch.no_grad():
        for i0 in range(0, N, args.batch_size):
            i1 = min(i0 + args.batch_size, N)
            t = torch.from_numpy(times[i0:i1, :L].astype(np.float32)).to(device)
            f = torch.from_numpy(fluxes[i0:i1, :L].astype(np.float32)).to(device)
            e = torch.from_numpy(flux_errs[i0:i1, :L].astype(np.float32)).to(device)
            # Mask: True for real obs, False for padded
            n_b = n_obs[i0:i1]
            mask = torch.zeros((i1 - i0, L), dtype=torch.bool, device=device)
            for j, n in enumerate(n_b):
                mask[j, :min(int(n), L)] = True

            out = model(t, f, e, mask)
            z = out["z_sig"].float().cpu().numpy()  # (B, d_sig)

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
        z_qual=np.zeros((N, 32), dtype=np.float32),  # placeholder for schema parity
        n_obs=n_obs,
        success_mask=np.ones(N, dtype=bool),
        chen_indices=chen_indices,
        checkpoint_path=str(args.checkpoint),
        encoder=args.encoder,
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
