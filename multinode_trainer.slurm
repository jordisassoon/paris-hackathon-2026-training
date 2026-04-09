#!/bin/bash
#SBATCH --job-name=torchtitan_sweep
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

# debugging flags (optional)
export NCCL_DEBUG=WARN
export PYTHONFAULTHANDLER=1
export LOGLEVEL=INFO

source .venv/bin/activate

# ── Training config ─────────────────────────────────────────────────────────
MODULE=${MODULE:-"qwen3"}
CONFIG=${CONFIG:-"hackathon_model"}
STEPS=${STEPS:-"180"}

# Per-group learning rates (passed from run_sweep.sh via --export)
LR_EMB=${LR_EMB:-""}
LR_BACKBONE1D=${LR_BACKBONE1D:-""}
LR_MUON=${LR_MUON:-""}
LR_HEADS=${LR_HEADS:-""}
BASE_LR=${BASE_LR:-""}

# Build optimizer LR flags if set
OPT_FLAGS=""
if [ -n "$LR_EMB" ]; then
    OPT_FLAGS="${OPT_FLAGS} --optimizer.embedding.lr ${LR_EMB}"
fi
if [ -n "$LR_BACKBONE1D" ]; then
    OPT_FLAGS="${OPT_FLAGS} --optimizer.backbone_1d.lr ${LR_BACKBONE1D}"
fi
if [ -n "$LR_MUON" ]; then
    OPT_FLAGS="${OPT_FLAGS} --optimizer.backbone_2d.lr ${LR_MUON}"
fi
if [ -n "$LR_HEADS" ]; then
    OPT_FLAGS="${OPT_FLAGS} --optimizer.heads.lr ${LR_HEADS}"
fi

# Unique checkpoint folder per LR to avoid loading stale checkpoints
CHECKPOINT_FOLDER="checkpoint_lr_${BASE_LR:-default}"

# Wandb run name (passed from run_sweep.sh)
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-"sweep-baseLR${BASE_LR}"}

# ── Launch one torchrun per node via srun ────────────────────────────────────
srun python -m torch.distributed.run \
    --nnodes="$NNODES" \
    --nproc_per_node="$NPROC_PER_NODE" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    --rdzv_id="$SLURM_JOB_ID" \
    -m torchtitan.train --module ${MODULE} --config ${CONFIG} \
    --training.steps=${STEPS} \
    --checkpoint.folder ${CHECKPOINT_FOLDER} ${OPT_FLAGS}
