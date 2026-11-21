#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Train a fresh LIPM-plus-hands policy on two GPUs. Pass --dry_run to inspect
# the distributed launch command without starting training.
set -euo pipefail

task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$task_root"
motion_path="${MOTION_REFERENCE:-$task_root/../LIP/generated_trajectories/steps_010_020_030_hands.npz}"
if [[ ! -f "$motion_path" && ! -d "$motion_path" ]]; then
    echo "Reference not found: $motion_path" >&2
    echo "Generate it with LIP/LIPM/export_hand_ik_reference.py or set MOTION_REFERENCE." >&2
    exit 1
fi

export HEADLESS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
exec uv run --no-sync python scripts/reinforcement_learning/train_multigpu.py \
    --num_gpus 2 \
    --task Isaac-Velocity-Flat-G1 --rl_library rsl_rl \
    --num_envs "${ENVS_PER_GPU:-2048}" \
    --max_iterations "${MAX_ITERATIONS:-1500}" --logger tensorboard \
    presets=lipm_hands agent.init_at_random_ep_len=false \
    "agent.experiment_name=${EXPERIMENT_NAME:-g1_lipm_hands_2gpu}" \
    "env.commands.base_velocity.motion_directory=$motion_path" \
    "$@"
