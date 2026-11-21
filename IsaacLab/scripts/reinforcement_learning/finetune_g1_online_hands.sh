#!/usr/bin/env bash
# Initialize online hands from the original walking policy, then fine-tune.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
source_checkpoint="$root/logs/rsl_rl/g1_lipm_first050_2gpu/2026-10-29_22-36-50/model_1499.pt"
export HEADLESS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
num_gpus="${NUM_GPUS:-2}"
if [[ "${1:-}" == "--dry-run" ]]; then
    printf '%s\n' "Source: $source_checkpoint" "GPUs: $num_gpus; envs/GPU: ${ENVS_PER_GPU:-2048}" \
        "Transfer walking actor/critic from 115 to 143 inputs; fresh optimizer; unique output directory" \
        "Training: ${MAX_ITERATIONS:-500} iterations; gradually introduce reach over 10000 control steps; full IK arm control from start" \
        "Hand endpoints: ~40 cm forward, 20 cm high; world-axis offsets from current torso position; explicit heading feedback" \
        "RL arm corrections: ${ARM_RESIDUAL_LIMIT:-0.0} rad; joint-error penalty: ${ARM_JOINT_ERROR_WEIGHT:-0.5}" \
        "Reach: ${REACH_DURATION:-6.0}s after 1s hold; torso upright weight: ${TORSO_UPRIGHT_WEIGHT:-1.0}; free tilt: ${TORSO_TILT_TOLERANCE_DEG:-8.0} degrees"
    exit 0
fi
if [[ ! -f "$source_checkpoint" ]]; then
    printf 'Checkpoint not found: %s\n' "$source_checkpoint" >&2
    exit 1
fi
mkdir -p logs/rsl_rl/g1_lipm_online_hands_finetune
run_root=$(mktemp -d "$root/logs/rsl_rl/g1_lipm_online_hands_finetune/run_XXXXXXXX")
printf 'Fine-tuning source: %s\nOutput: %s\n' "$source_checkpoint" "$run_root/training"
uv run --no-sync python scripts/reinforcement_learning/transfer_g1_online_hands.py \
    --checkpoint "$source_checkpoint" --output "$run_root/initialization"
exec uv run --no-sync python -m torch.distributed.run --standalone --nproc_per_node="$num_gpus" \
    scripts/reinforcement_learning/finetune_g1_online_hands.py \
    --checkpoint "$run_root/initialization/model_0.pt" --output "$run_root/training" \
    --num-envs "${ENVS_PER_GPU:-2048}" --max-iterations "${MAX_ITERATIONS:-500}" \
    --learning-rate "${LEARNING_RATE:-0.0001}" \
    --arm-residual-limit "${ARM_RESIDUAL_LIMIT:-0.0}" \
    --arm-joint-error-weight "${ARM_JOINT_ERROR_WEIGHT:-0.5}" \
    --reach-duration "${REACH_DURATION:-6.0}" \
    --torso-upright-weight "${TORSO_UPRIGHT_WEIGHT:-1.0}" \
    --torso-tilt-tolerance-deg "${TORSO_TILT_TOLERANCE_DEG:-8.0}" \
    --hand-reward-weight "${HAND_REWARD_WEIGHT:-0.05}" "$@"
