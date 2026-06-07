#!/bin/bash
# ============================================================================
# Embed the Chen 25K subset with each BiGRU seed checkpoint, producing
# results/chen2020_25k/embeddings_rnn_seed<seed>.npz for the seed-CI diagnostic.
# Seed 42's embeddings already exist as embeddings_rnn_v1.npz; this script
# symlinks that and embeds the new seeds.
#   bash scripts/embed_rnn_seeds.sh "42 1 2 3"
# Run on a GPU node (env: pytorch on ORCD); CPU works but is slow.
# ============================================================================
set -eu
SEEDS=${1:-"42 1 2 3"}
REPO="$HOME/ztf-ssl-transformer"
PY="${PY:-python}"
CACHE="$REPO/results/chen2020_25k/lc_cache_25k.npz"
cd "$REPO"

for s in $SEEDS; do
    OUT="results/chen2020_25k/embeddings_rnn_seed${s}.npz"
    if [ "$s" = "42" ] && [ -f "results/chen2020_25k/embeddings_rnn_v1.npz" ]; then
        ln -sf embeddings_rnn_v1.npz "$OUT"
        echo "seed 42 -> linked existing embeddings_rnn_v1.npz"
        continue
    fi
    CKPT="checkpoints/rnn_seed${s}/best_model.pt"
    if [ ! -f "$CKPT" ]; then echo "MISSING $CKPT (skip seed $s)"; continue; fi
    echo "=== embedding seed $s from $CKPT ==="
    "$PY" -u scripts/embed_seq_baseline_25k.py \
        --encoder rnn --checkpoint "$CKPT" --cache "$CACHE" --output "$OUT" \
        --d_model 320 --num_layers 3 --bidirectional 1 \
        --d_sig 128 --d_qual 32 --max_seq_len 384
done
echo "done. aggregate: python scripts/diag_seed_ci.py --seeds $SEEDS"
