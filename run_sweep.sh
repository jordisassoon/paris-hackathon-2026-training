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

# Training parameters
STEPS=${STEPS:-"1000"}
LR_ARRAY=(0.01 0.001 0.003)

TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}

for LR in "${LR_ARRAY[@]}"; do
    export LR
    echo "========================================"
    echo "Starting sweep with LR=${LR}"
    echo "========================================"

    if [ "$NGPU" -eq 16 ]; then
        echo "Running 16-GPU job on Slurm cluster"

        MAX_RETRIES=3
        RETRY_COUNT=0

        while true; do
            echo "Submitting Slurm job (attempt $((RETRY_COUNT+1)))..."

            job_output=$(sbatch --export=ALL,CONFIG_FILE="$CONFIG_FILE",LBS="$LBS",STEPS="$STEPS",LR="$LR" multinode_trainer.slurm)
            job_id=$(echo "$job_output" | awk '{print $4}')

            LOG_PATH="/home/jordansassoon/torchtitanic/outputs/logs/multinode_titanic_${job_id}.out"

            echo "Submitted Job ID: $job_id (LR=${LR})"
            echo "Waiting for job to finish..."

            # Wait for job to leave queue
            while squeue -j "$job_id" 2>/dev/null | grep -q "$job_id"; do
                sleep 10
            done

            echo "Job finished. Checking log for stale file handle..."

            # Detect stale file handle error
            if grep -qi "stale file handle" "$LOG_PATH"; then
                echo "Detected 'Stale file handle' error in Slurm log."
                RETRY_COUNT=$((RETRY_COUNT+1))

                if [ "$RETRY_COUNT" -ge "$MAX_RETRIES" ]; then
                    echo "Max retries reached. Exiting."
                    break
                fi

                echo "Retrying job submission..."
                sleep 5
                continue
            fi

            # No stale file handle → exit retry loop
            break
        done
    else    
        echo "Running local job on single node with ${NGPU} GPUs"
        TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}

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
            -m torchtitan.train --module ${MODULE} --config ${CONFIG} --optimizer.embedding.lr ${LR} --optimizer.backbone_1d.lr ${LR} --optimizer.backbone_2d.lr ${LR} --optimizer.heads.lr ${LR} --training.steps=${STEPS} "$@"
        fi
    fi

    echo "Finished sweep with LR=${LR}"
done
