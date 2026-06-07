#!/usr/bin/env python3
"""
embed_chronos_t5_25k.py — embed Chen-2020 25K subset using a Chronos-T5
checkpoint trained from scratch by src/baselines/chronos_t5.py.

Mirrors embed_chronos_25k.py (same input cache, same output schema) but loads
our local checkpoint dict ({model_state_dict, t5_config, chronos_config})
instead of the pretrained ChronosPipeline. Pooled embedding is mean over
encoder hidden states across the (non-pad) context positions — same definition
as the pretrained-Chronos baseline so the fairness comparison stays clean.

Output: results/chen2020_25k/embeddings_chronos_t5_<tag>.npz
"""
import os
import argparse
from pathlib import Path

import numpy as np
import torch

from chronos import ChronosConfig
from transformers import T5Config, T5ForConditionalGeneration

REPO = Path(os.environ.get("PERIOD_DIAG_ROOT", str(Path(__file__).resolve().parents[1])))
RESULTS = REPO / "results/chen2020_25k"
CACHE = RESULTS / "lc_cache_25k.npz"


def load_checkpoint(ckpt_path: str, device: str):
    """Reconstruct T5 + tokenizer from our chronos_t5.py checkpoint format."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    t5_config = T5Config(**ckpt["t5_config"])
    model = T5ForConditionalGeneration(t5_config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    chronos_config = ChronosConfig(**ckpt["chronos_config"])
    tokenizer = chronos_config.create_tokenizer()

    return model, tokenizer, chronos_config


@torch.no_grad()
def embed_one_batch(model, tokenizer, contexts_cpu, chronos_config, device):
    """contexts_cpu: list of (T_i,) float tensors. Returns (B, d_model) numpy.

    Tokenize on CPU (matches training-side device handling), forward through
    encoder, mean-pool over non-padded positions.
    """
    # Left-pad with NaN to a common length so the tokenizer's per-row scale
    # is computed over real values only.
    max_len = max(c.shape[0] for c in contexts_cpu)
    padded = torch.full(
        (len(contexts_cpu), max_len), float("nan"), dtype=torch.float32
    )
    for i, c in enumerate(contexts_cpu):
        padded[i, -c.shape[0]:] = c

    input_ids, attn_mask, _ = tokenizer.context_input_transform(padded)
    input_ids = input_ids.to(device)
    attn_mask = attn_mask.to(device)

    enc = model.encoder(input_ids=input_ids, attention_mask=attn_mask)
    h = enc.last_hidden_state.float()  # (B, T, D)

    # Mean over real positions only
    mask = attn_mask.float().unsqueeze(-1)  # (B, T, 1)
    pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    return pooled.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="Path to chronos_t5.py best_model.pt")
    ap.add_argument("--cache", default=str(CACHE))
    ap.add_argument("--output", default=None,
                    help="Default: results/chen2020_25k/embeddings_chronos_t5_"
                         "<ckpt-parent-name>.npz")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_context", type=int, default=320,
                    help="Truncate longer contexts to this length. Defaults to "
                         "the chronos_t5 training context_length.")
    args = ap.parse_args()

    if args.output is None:
        tag = Path(args.checkpoint).parent.name
        args.output = str(RESULTS / f"embeddings_chronos_t5_{tag}.npz")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading checkpoint {args.checkpoint} → device={device}")
    model, tokenizer, chronos_config = load_checkpoint(args.checkpoint, device)
    print(f"  context_length (training)={chronos_config.context_length}, "
          f"vocab={chronos_config.n_tokens}")

    print(f"Loading cache from {args.cache}")
    data = np.load(args.cache)
    fluxes = data["fluxes"]
    success = data["success"]
    chen_indices = data["chen_indices"]
    n_obs = data["n_obs"]
    N = len(success)
    print(f"  N = {N:,}")

    # Build per-source CPU tensors, truncate centered to max_context
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
            x = fluxes[i, s:s + L]
        contexts.append(torch.from_numpy(x.astype(np.float32)))

    print("Computing embeddings (mean-pooled encoder hidden states)...")
    embeddings = None
    for i0 in range(0, N, args.batch_size):
        i1 = min(i0 + args.batch_size, N)
        batch = contexts[i0:i1]
        try:
            emb = embed_one_batch(model, tokenizer, batch, chronos_config, device)
        except Exception as e:
            print(f"  batch {i0}: failed ({e}); filling NaN")
            D = embeddings.shape[1] if embeddings is not None else model.config.d_model
            emb = np.full((i1 - i0, D), np.nan, dtype=np.float32)
        if embeddings is None:
            D = emb.shape[1]
            embeddings = np.full((N, D), np.nan, dtype=np.float32)
            print(f"  embedding dim D = {D}")
        embeddings[i0:i1] = emb
        if (i0 // args.batch_size) % 20 == 0:
            print(f"  batch {i0:>6}/{N}", flush=True)

    print(f"\nEmbeddings shape: {embeddings.shape}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        z_sig=embeddings,
        z_qual=np.zeros((N, 32), dtype=np.float32),  # placeholder for schema parity
        n_obs=n_obs,
        success_mask=success,
        chen_indices=chen_indices,
        checkpoint_path=str(args.checkpoint),
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
