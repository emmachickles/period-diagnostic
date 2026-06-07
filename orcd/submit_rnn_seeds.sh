#!/bin/bash
# ============================================================================
# Submit the BiGRU encoder-seed sweep for camera-ready CIs.
# Seed 42 is already trained (checkpoints/rnn_v1 -> embeddings_rnn_v1.npz);
# this adds seeds 1, 2, 3. Edit SEEDS for more (3-5 total satisfies reviewers).
#   bash orcd/submit_rnn_seeds.sh
# ============================================================================
set -eu
SEEDS=${SEEDS:-"1 2 3"}
cd "$HOME/ztf-ssl-transformer"
for s in $SEEDS; do
    jid=$(sbatch --parsable --export=ALL,SEED=$s orcd/train_rnn_seed.slurm)
    echo "submitted seed $s -> job $jid  (checkpoints/rnn_seed$s)"
done
echo
echo "When all jobs finish, embed + aggregate with:"
echo "  bash scripts/embed_rnn_seeds.sh \"42 $SEEDS\""
echo "  python scripts/diag_seed_ci.py --seeds 42 $SEEDS"
