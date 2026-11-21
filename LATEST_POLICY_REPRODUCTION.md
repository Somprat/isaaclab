# Reproduce the latest G1 walking and hand policy

Verified in this workspace on 2026-11-20. Run commands from `/home/choon/isaaclab`.

The latest rendered checkpoint is:

```text
IsaacLab/logs/rsl_rl/g1_lipm_online_hands_finetune/run_TY9vx7dE/training/model_499.pt
```

Its video is [closer5cm_TY9vx7dE_policy.mp4](LIP/previews/g1-closer5cm-TY9vx7dE-499/closer5cm_TY9vx7dE_policy.mp4).
This is live policy inference with body/leg physics and online IK controlling the arms through direct joint-state placement.

## 1. Files and environment needed

Use the existing `uv` environment in `IsaacLab/` and the current local controller changes.
The `--no-sync` commands below assume that environment is already installed. Training
uses two CUDA GPUs by default; the recorded inference uses one GPU.

Keep these files and directories together when copying the workspace:

| Required item | Purpose |
| --- | --- |
| `IsaacLab/` source and its Python environment | Newton simulation, RSL-RL policy, target and IK implementations |
| Latest `training/model_499.pt` **and `training/params/`** | Weights plus saved environment and agent configuration |
| `LIP/generated_trajectories/steps_010_020_030_first050.npz` | Walking reference used by this run |
| `IsaacLab/checkpoints/g1-29dof/mpc-matched-assets/` | Robot USD assets and dependencies |
| `wb_humanoid_mpc/robot_models/unitree_g1/g1_description/` | Matched URDF, MuJoCo XML and meshes |
| Original walking checkpoint and its adjacent `params/` | Needed to repeat fine-tuning from the walking policy |

Saved configurations contain absolute paths. Moving the workspace requires updating
those paths or restoring the same workspace layout. Copying the checkpoint alone is
insufficient for these scripts.

## 2. Render the existing latest checkpoint

**Inference and MP4 script:**
[`IsaacLab/scripts/reinforcement_learning/play_g1_reference.py`](IsaacLab/scripts/reinforcement_learning/play_g1_reference.py).
This is the script used to render the latest `run_TY9vx7dE/model_499.pt` policy.
It saves an MP4 automatically; no separate video-export script is required.

Choose an output directory that does not already exist:

```bash
cd /home/choon/isaaclab

HEADLESS=1 uv run --project IsaacLab --no-sync python \
  IsaacLab/scripts/reinforcement_learning/play_g1_reference.py \
  --checkpoint IsaacLab/logs/rsl_rl/g1_lipm_online_hands_finetune/run_TY9vx7dE/training/model_499.pt \
  --duration 20 \
  --record-full-state \
  --output LIP/previews/g1-closer5cm-TY9vx7dE-reproduced
```

The saved checkpoint configuration restores closer targets, world orientation, and
immediate arm placement. No playback override is needed. The output includes:

- `g1_reference_model_499_0000.mp4`
- `inference.json`, including checkpoint provenance and termination events
- `reference_tracking.csv` and `tracking_summary.json`
- `full_state_replay.npz`, for later MuJoCo diagnostics

The existing published preview was renamed to `closer5cm_TY9vx7dE_policy.mp4` after
rendering. The playback script itself uses the filename shown above.

The verified rollout completed 20 seconds / 1000 control steps. It had no fall or
tilt termination; its only termination was timeout/reference completion at 20 seconds.
This is one rollout. A rerun can differ slightly due to simulation nondeterminism.

## 3. Repeat the fine-tuning recipe

The existing shell script initializes from the original walking policy:

```text
IsaacLab/logs/rsl_rl/g1_lipm_first050_2gpu/2026-10-29_22-36-50/model_1499.pt
```

Run:

```bash
cd /home/choon/isaaclab

NUM_GPUS=2 ENVS_PER_GPU=2048 MAX_ITERATIONS=500 \
HAND_REWARD_WEIGHT=1.0 \
bash IsaacLab/scripts/reinforcement_learning/finetune_g1_online_hands.sh \
  --immediate-arm-placement \
  --world-identity-hand-orientation \
  --hand-target-frame world \
  --hand-forward-reduction 0.05 \
  --initial-hold 1.0 \
  --reach-duration 6.0
```

The script transfers the walking actor and critic from 115 to 143 observation inputs,
starts a fresh optimizer, and trains for 500 iterations with learning rate `0.0001`.
It creates a fresh `run_XXXXXXXX/` folder and prints its location. It does not resume
from `run_TY9vx7dE`.

The existing reach curriculum starts at fraction 0.1 and reaches 1.0 over 10000 global
control steps, sampled on episode reset/resampling. The additional 5 cm reduction
uses the episode-time ramp independently of that curriculum. Repeating this recipe
reproduces the training setup; it does not guarantee identical learned weights.

To render the newly trained checkpoint, use the inference command in section 2,
replacing the checkpoint run name and choosing another fresh output directory.

## 4. What the current controller does

| Setting | Latest policy behavior |
| --- | --- |
| Position X/Y | Offsets from the current torso position along fixed world axes |
| Position Z | Absolute world height, anchored using reset torso height |
| Closer target adjustment | Subtract 0.05 m from world X, ramped linearly from 1 to 7 seconds and held afterward |
| Reach trajectory | Original quintic trajectory, with the linear X adjustment added |
| Seven-second capture | Disabled; original configured targets continue |
| Hand orientation | Both hand links target world identity throughout; no initial torso yaw alignment |
| IK | Online pose DLS, damping 0.05, one solve per 20 ms |
| Joint correction cap | 0.05 rad per joint per solve, all 14 arm joints |
| Joint limits | 0.9 soft joint-limit factor |
| Arm ownership | Full IK control; policy arm residuals and persistent reference accumulation bypassed |
| Placement | Apply IK angles immediately, then hold them at every pre/post physics boundary |
| Arm velocity | Prescribed zero joint velocity between solves; pose jumps still occur at IK updates |
| Arm drives | Stiffness and damping zero |
| Body and legs | Policy commands with physics; physics remains enabled |
| Simulation timing | 0.0025 s physics steps, 8 steps per 0.02 s policy interval |

Hand orientation reward is `exp(-angular_error_rad / 0.5²)` at weight +1.0 per hand,
with an angular-error penalty at weight -0.5 per hand. The implementation compares
measured and desired orientations in torso coordinates; with world-fixed targets,
this yields the same angular error as comparison in world coordinates.

Hand position reward weight is +1.0 per hand. Hand velocity reward weight is +0.5
per hand with tolerance parameter 0.1 m/s. The forward reduction also adjusts target
velocity so the reward follows the same target trajectory.

## 5. LIP height and reference provenance

The walking clip is `LIP/generated_trajectories/steps_010_020_030_first050.npz`.
It contains a constant whole-robot CoM height of **0.714414 m**, derived from
`LIP/initial_states/g1_reset.json`. It does not specify a separate torso-link height.

The policy observes CoM position/velocity errors, including vertical errors, and is
rewarded for CoM tracking. Actual torso height and tilt can vary while tracking the
whole-body CoM. Hand Z targets are a separate reference.

The reference generator is `LIP/LIPM/demo_LIPM_3D_vt.py`, with dynamics in
`LIP/LIPM/trajectory.py`. Its `--initial-state` option overrides `--com-height`
using the measured reset CoM. Reuse the saved NPZ to reproduce this policy's reference;
generating a different gait is a separate experiment.

## 6. Matching MuJoCo comparison

The existing comparison with the same closer-target settings is:

[closer_5cm_QjzheWXO_body.mp4](LIP/previews/mujoco-QjzheWXO-online-ik-measured-body/closer_5cm_QjzheWXO_body.mp4)

It uses an older policy's recorded body trajectory. Its targets and arm IK settings
match, but its body motion differs from the latest policy.

To replay the **latest policy's recorded body motion** with fresh online arm IK:

```bash
cd /home/choon/isaaclab

uv run --project IsaacLab --no-sync python \
  LIP/scripts/replay_raw_hand_commands.py \
  --source LIP/previews/g1-closer5cm-TY9vx7dE-499 \
  --output LIP/previews/mujoco-TY9vx7dE-online-ik-reproduced \
  --mode matched-ik \
  --iterations 1
```

This outputs `replay.mp4`, `errors.csv`, and `metadata.json` in a fresh directory.
The source recording already contains closer hand targets, so do not subtract another
5 cm. To replay a new recording from section 2, change `--source` accordingly.

This MuJoCo diagnostic prescribes the measured pelvis, leg, and waist poses, solves
arm IK each frame, assigns arm joint positions exactly, and calls `mj_forward`.
It runs no policy inference or physics stepping, so it cannot establish balance.
The `all-commands` mode directly places leg/waist targets too and is a different
comparison: those targets can differ substantially from measured body angles.

## 7. Implementation map

| File | Role |
| --- | --- |
| [Fine-tuning shell launcher](IsaacLab/scripts/reinforcement_learning/finetune_g1_online_hands.sh) | Source checkpoint, transfer, GPU launch and output directory |
| [Fine-tuning Python entry point](IsaacLab/scripts/reinforcement_learning/finetune_g1_online_hands.py) | Target flags, rewards, direct-placement simulation configuration |
| [Checkpoint transfer](IsaacLab/scripts/reinforcement_learning/transfer_g1_online_hands.py) | Expand walking observation inputs |
| [Policy playback](IsaacLab/scripts/reinforcement_learning/play_g1_reference.py) | Load saved configuration, inference, MP4 and state recording |
| [Hand commands](IsaacLab/source/isaaclab_tasks/isaaclab_tasks/core/velocity/mdp/hand_commands.py) | Position/orientation targets, closer ramp and hand rewards |
| [Hand actions](IsaacLab/source/isaaclab_tasks/isaaclab_tasks/core/velocity/mdp/hand_actions.py) | Online IK and immediate state placement |
| [LIPM commands](IsaacLab/source/isaaclab_tasks/isaaclab_tasks/core/velocity/mdp/lipm_commands.py) | Reference clock, alignment and whole-body CoM |
| [MuJoCo diagnostic](LIP/scripts/replay_raw_hand_commands.py) | Recorded-body replay with fresh IK or recorded commands |

Older milestone notes and fine-tuning documents describe earlier configurations.
For this run, use the commands above and its saved `training/params/` configuration.
