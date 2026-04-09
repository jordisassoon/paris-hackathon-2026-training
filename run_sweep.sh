#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# use envs as local overwrites for convenience
# e.g.
# LOG_RANK=0,1 NGPU=4 ./run_train.sh
#
# COMM_MODE options for debugging:
#
# 1. "fake_backend" - Dry-run mode for config validation without GPU execution
#    - Uses fake process groups (no actual communication)
#    - Runs on a single GPU without torchrun or NCCL initialization
#    - Useful for validating configuration and model setup
#    Example: NGPU=32 COMM_MODE="fake_backend" ./run_train.sh
#
# 2. "local_tensor" - Single-GPU debugging mode with simulated multi-GPU behavior
#    - All communication and computation execute on a single shared GPU
#    - Simulates the full training workflow without actual distributed communication
#    - Useful for debugging distributed training logic locally
#    Example: NGPU=32 COMM_MODE="local_tensor" ./run_train.sh

# Exit immediately if a command exits with a non-zero status, except in pipelines
set +e

# Create log dir if it doesn't exist
mkdir -p logs

NGPU=${NGPU:-"8"}
export LOG_RANK=${LOG_RANK:-0}
MODULE=${MODULE:-"qwen3"}
CONFIG=${CONFIG:-"hackathon_model"}
SUMMARY_PARSED_LOG_FILE=${SUMMARY_PARSED_LOG_FILE:-"summary_parsed_logs.csv"}

# Wandb configuration
# Load API key from .env file if present (not committed to repo)
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi
export WANDB_PROJECT=${WANDB_PROJECT:-"paris-hackathon-2026"}
export WANDB_TEAM=${WANDB_TEAM:-"aleph-alpha"}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-"lr-sweep"}

# Training parameters
STEPS=${STEPS:-"180"}
# Base LRs: 2^-3, 2^-4, 2^-5
BASE_LR_ARRAY=(0.125 0.0625 0.03125)

# Per-group multipliers
MULT_MUON=1.0        # backbone_2d
MULT_EMB=2.0         # embedding
MULT_BACKBONE1D=2.0  # backbone_1d
MULT_HEADS=0.25      # heads

TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}

for BASE_LR in "${BASE_LR_ARRAY[@]}"; do
    LR_EMB=$(awk "BEGIN {printf \"%.6g\", $BASE_LR * $MULT_EMB}")
    LR_BACKBONE1D=$(awk "BEGIN {printf \"%.6g\", $BASE_LR * $MULT_BACKBONE1D}")
    LR_MUON=$(awk "BEGIN {printf \"%.6g\", $BASE_LR * $MULT_MUON}")
    LR_HEADS=$(awk "BEGIN {printf \"%.6g\", $BASE_LR * $MULT_HEADS}")

    export BASE_LR LR_EMB LR_BACKBONE1D LR_MUON LR_HEADS
    echo "========================================"
    echo "Starting sweep with BASE_LR=${BASE_LR}"
    echo "  embedding.lr=${LR_EMB}  backbone_1d.lr=${LR_BACKBONE1D}  backbone_2d.lr=${LR_MUON}  heads.lr=${LR_HEADS}"
    echo "========================================"

    if [ "${USE_SLURM:-0}" -eq 1 ]; then
        WANDB_RUN_NAME="sweep-baseLR${BASE_LR}"
        CHECKPOINT_FOLDER="checkpoint_lr_${BASE_LR}"
        echo "Submitting Slurm job for BASE_LR=${BASE_LR} (wandb: ${WANDB_RUN_NAME})"

        job_output=$(sbatch \
            --export=ALL,MODULE="$MODULE",CONFIG="$CONFIG",STEPS="$STEPS",BASE_LR="$BASE_LR",LR_EMB="$LR_EMB",LR_BACKBONE1D="$LR_BACKBONE1D",LR_MUON="$LR_MUON",LR_HEADS="$LR_HEADS",WANDB_RUN_NAME="$WANDB_RUN_NAME" \
            multinode_trainer.slurm)
        job_id=$(echo "$job_output" | awk '{print $4}')
        echo "Submitted Job ID: $job_id"
    else    
        echo "Running local job on single node with ${NGPU} GPUs"
        TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
        export WANDB_RUN_NAME="sweep-baseLR${BASE_LR}"

        if [ -n "$COMM_MODE" ]; then
            # Communication mode specified: validate configuration or run in debug mode
            echo "Running with comm_mode=${COMM_MODE}"
            NGPU="${NGPU}" LOCAL_RANK=0 python3 -m torchtitan.train --module ${MODULE} --config ${CONFIG} "$@" --comm.mode=${COMM_MODE} --training.steps 1
        else
            # Normal training with torchrun
            PYTORCH_ALLOC_CONF="expandable_segments:True" \
            TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE} \
            torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
            --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
            -m torchtitan.train --module ${MODULE} --config ${CONFIG} \
            --optimizer.embedding.lr ${LR_EMB} --optimizer.backbone_1d.lr ${LR_BACKBONE1D} \
            --optimizer.backbone_2d.lr ${LR_MUON} --optimizer.heads.lr ${LR_HEADS} \
            --training.steps=${STEPS} \
            --checkpoint.folder "checkpoint_lr_${BASE_LR}" "$@"
        fi
    fi

    echo "Finished sweep with BASE_LR=${BASE_LR}"
done
