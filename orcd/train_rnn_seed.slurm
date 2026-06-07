#!/bin/bash
# ============================================================================
# BiGRU baseline, SEED-PARAMETERIZED, for camera-ready encoder-seed CIs.
# Identical recipe to train_rnn_v1.slurm (seed 42, already done) except the
# seed and output dir are driven by $SEED so we can report rho_within across
# independently trained encoders (Reviewers: "single seed is a meaningful gap").
#
# Submit one seed:   sbatch --export=ALL,SEED=1 orcd/train_rnn_seed.slurm
# Submit a sweep:    bash orcd/submit_rnn_seeds.sh
# ============================================================================

#SBATCH --job-name=rnn-seed
#SBATCH --partition=mit_preemptable
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=4
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/rnn_seed%a_%j.out
#SBATCH --error=logs/rnn_seed%a_%j.err

set -e
: "${SEED:?Set SEED, e.g. sbatch --export=ALL,SEED=1 orcd/train_rnn_seed.slurm}"
echo "Job $SLURM_JOB_ID (SEED=$SEED) started at $(date) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

PROJECT_DIR="$HOME/ztf-ssl-transformer"
MATCHFILE_DIR="/path/to/ZTF/matchfiles"
OUTPUT_DIR="$PROJECT_DIR/checkpoints/rnn_seed${SEED}"
mkdir -p "$OUTPUT_DIR" "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

RESUME_FLAG=""
if [ -f "$OUTPUT_DIR/best_model.pt" ]; then
    RESUME_FLAG="--resume_from $OUTPUT_DIR/best_model.pt"
    echo "Resuming from $OUTPUT_DIR/best_model.pt"
fi

source ~/miniforge3/etc/profile.d/conda.sh
conda activate pytorch
export NCCL_TIMEOUT=1800
export NCCL_DEBUG=WARN

# Unique rendezvous port per seed so concurrent jobs on one node don't clash.
RDZV_PORT=$((29503 + SEED))

torchrun --nnodes=1 --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint="localhost:${RDZV_PORT}" \
    src/baselines/train_seq_baseline.py \
        --encoder rnn \
        --matchfile_dir "$MATCHFILE_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --max_sources 400000 \
        --multiband \
        --holdout_path "$PROJECT_DIR/data/chen_holdout_per_tile.npz" \
        --d_model 320 --num_layers 3 --bidirectional 1 \
        --d_sig 128 --d_qual 32 \
        --max_seq_len 384 \
        --batch_size 96 --num_workers 8 \
        --epochs 40 --warmup_epochs 2 \
        --lr 1e-4 --weight_decay 1e-4 \
        --temperature 0.07 --lambda_qual 1.0 \
        --patience 10 --precision fp32 --seed ${SEED} \
        --use_disjoint_windows 1 \
        $RESUME_FLAG

echo "Job $SLURM_JOB_ID (SEED=$SEED) finished at $(date)"
