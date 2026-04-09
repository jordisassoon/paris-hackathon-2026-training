#!/bin/bash
#SBATCH --job-name=platypus_train
#SBATCH --partition=gpus
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --output=logs/%j_%N.out
#SBATCH --error=logs/%j_%N.err

set -euo pipefail

mkdir -p logs

# ── Rendezvous info derived from SLURM ──────────────────────────────────────
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=29500
NNODES=$SLURM_NNODES
NPROC_PER_NODE=$SLURM_GPUS_PER_NODE

echo "Master: $MASTER_ADDR:$MASTER_PORT  |  Nodes: $NNODES  |  GPUs/node: $NPROC_PER_NODE"

# ── Load env (wandb key, etc.) ──────────────────────────────────────────────
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# debugging flags (optional)
export NCCL_DEBUG=WARN
export PYTHONFAULTHANDLER=1
export LOGLEVEL=INFO

source .venv/bin/activate

# ── Training config ─────────────────────────────────────────────────────────
MODULE=${MODULE:-"qwen3"}
CONFIG=${CONFIG:-"hackathon_model"}

# Unique checkpoint folder per LR to avoid loading stale checkpoints
CHECKPOINT_FOLDER="platypus_checkpoints/final_run"

export WANDB_PROJECT=${WANDB_PROJECT:-"paris-hackathon-2026"}
export WANDB_TEAM=${WANDB_TEAM:-"aleph-alpha"}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-"final-run"}

# ── Launch one torchrun per node via srun ────────────────────────────────────
srun python -m torch.distributed.run \
    --nnodes="$NNODES" \
    --nproc_per_node="$NPROC_PER_NODE" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    --rdzv_id="$SLURM_JOB_ID" \
    -m torchtitan.train --module ${MODULE} --config ${CONFIG} \
    --checkpoint.folder ${CHECKPOINT_FOLDER}
