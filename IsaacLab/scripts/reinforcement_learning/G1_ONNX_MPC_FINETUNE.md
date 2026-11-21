# Fine-tune the published G1 ONNX actor with MPC references

`finetune_g1_mpc.py` uses the existing `Isaac-Velocity-Flat-G1` reference preset
and Newton/MuJoCo-Warp simulation. It imports all actor weights from
`checkpoints/g1-29dof/policy.onnx`; it does not initialize the actor from scratch.
The original ONNX file, pretrained baseline, default task and reference preset
are unchanged. Configuration changes apply only to this script's environment.

## Preserving the pretrained behavior

The ONNX actor uses a 512/256/128 ELU network with 480 input values and 29 motor
outputs. Its inputs contain five frames of six observation terms, flattened
per term in the order recorded in `deploy.yaml`: angular velocity, gravity,
velocity command, relative joint positions, joint velocities and previous actions.
Joint observations/actions use the deployment's motor order. Angular velocity is
scaled by 0.2 and joint velocity by 0.05. Motor targets use the deployment's
0.25 scale and joint-specific offsets; the default pose and PD gains also follow
the deployment. Those are deliberately preserved from the ONNX baseline rather
than the earlier scratch-training configuration's 0.5 action scale.

The fine-tuning observation adds 66 current reference values after those 480:
stride, phase sin/cos, reference joint positions/velocities, root orientation and
height error. The first layer's corresponding columns initialize to zero;
all other actor weights/biases copy exactly from ONNX. Initial means therefore
match the published policy, and every actor layer, including the extra columns,
remains trainable. Extra joint reference channels use articulation order; the
480 original channels and 29 outputs use deployment order.

ONNX contains no critic, optimizer state, or exploration variance. These start
fresh. This is actor fine-tuning, not resuming the original full training state.
The script uses a starting learning rate of 1e-4 and initial action standard
deviation of 0.1, both configurable. PPO otherwise retains the existing task's
algorithm settings and inherited adaptive learning-rate schedule.

## Prepare and validate without training

```bash
cd /home/choon/isaaclab/IsaacLab
HEADLESS=1 uv run --no-sync python scripts/reinforcement_learning/finetune_g1_mpc.py \
  --num-envs 2 --validation-steps 100 \
  --output logs/g1_onnx_mpc_prepare
```

This imports weights, checks ONNX Runtime parity on actual and synthetic histories,
checks independence from new reference inputs at initialization, independently
reconstructs the deployment observation histories, verifies action processing,
and saves `model_initialized.pt`, configuration snapshots and `initialization.json`.
There are no PPO optimizer updates. Existing output directories are refused.

Validation with the local published actor passed 100 simulation steps with two
environments. The maximum initialized actor/ONNX action discrepancy was
3.815e-6. These checks establish interface/weight compatibility, not guaranteed
balance or convergence after learning. In particular, MPC contact dynamics and
Newton contact integration differ, and MPC resets can place the robot in new gait states.

## Start fine-tuning (not executed during implementation)

```bash
cd /home/choon/isaaclab/IsaacLab
HEADLESS=1 uv run --no-sync python scripts/reinforcement_learning/finetune_g1_mpc.py \
  --train --num-envs 128 --max-iterations 300
```

The default motion is `conditioned_50_fixed/conditioned_mpc_reference_0000.npz`.
Use `--motion /absolute/path/to/clip.npz` or a directory to select other data.
Other options include `--onnx`, `--deployment`, `--learning-rate`, `--initial-std`,
`--seed`, and `--output`. The default output is a fresh timestamped folder under
`logs/rsl_rl/g1_onnx_mpc_finetune`; it never overwrites the source policy.

`--train` starts from the ONNX actor every time. To continue an already fine-tuned
run, use its saved environment/agent settings and a normal RSL-RL checkpoint
resume workflow instead of importing ONNX again. Do not resume the earlier
165-input scratch-trained checkpoint as this 546-input policy.

## Inference

`play_g1_reference.py` detects the saved `params/deployment.yaml` and rebuilds the
matching deployment-compatible configuration before loading fine-tuned weights:

```bash
HEADLESS=1 uv run --no-sync python scripts/reinforcement_learning/play_g1_reference.py \
  --checkpoint /absolute/path/to/run/model_299.pt \
  --output /absolute/path/to/new/video_directory
```

The original 165-input reference checkpoint remains supported when there is no
saved deployment snapshot. No fine-tuning controller or torque action interface
is added: PPO still issues joint-position actions.

## MPC model parity

The reference preset now uses `checkpoints/g1-29dof/mpc-matched-assets`, built
from the exact motion-generator URDF. Body masses, COMs, inertias, joint frames
and source effort/velocity limits match the MPC robot. Rotor armature is zero
and contact friction is fixed at 0.4 with material randomization disabled.
The original pretrained USD/ONNX files are unchanged. The startup guard rejects
stale source/mesh/asset fingerprints. Rebuild instructions and measured dynamic
parity results are in
`source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/g1/MPC_MODEL_MATCH.md`.
Saved runs from before this change continue to name their original asset; new
fine-tuning runs use the matched model. This intentionally changes physical
parameters while preserving the ONNX observation and motor-target interfaces.
