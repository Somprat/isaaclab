# Reference-conditioned forward walking

This extends **Isaac-Velocity-Flat-G1** through `presets=reference`. It does not
register a new task ID. Without the preset, the original velocity task remains
the default. The existing inference-only `play_g1_29dof.py` is unchanged.

## Inputs and clock

The reference preset selects the MPC-URDF-derived 29-DOF G1 USD and locomotion actuator
gains, and samples the fixed MPC dataset. Override paths when needed:

- `env.commands.base_velocity.motion_directory`: directory containing NPZ clips, or a single NPZ file.
  Defaults to `wb_humanoid_mpc/generated_motions/conditioned_50_fixed` beside IsaacLab.
- `env.scene.robot.spawn.usd_path`: 29-DOF robot asset. Defaults to the cached
  `checkpoints/g1-29dof/mpc-matched-assets/g1_29dof/g1_29dof.usda`.

Each environment draws a clip and a random start time at reset, initializes its
root/joints/velocities from that frame, and retains a fixed origin/heading
transform. The existing `base_velocity` command now follows the clip's
instantaneous velocity condition, including transitions. Stride and all motion
tracking targets come from that same clip and time. MPC quaternions are converted from wxyz to this checkout's xyzw
asset/math convention at ingestion. Playback is interpolated at
`env.step_dt`; the source data remains at its original 30 Hz.

The observation target is one control step ahead, because the RL environment
computes rewards after physics and before updating commands. This ensures the
next action and its resulting reward use the same target frame. Clip exhaustion
is a timeout, not a wraparound jump. The full trajectory/NPZ data is not modified.

## Observations

The policy has 165 values for the 29-DOF asset:

- Base linear/angular velocity and projected gravity: 9.
- Velocity command `[vx, vy, yaw_rate]`: 3.
- Full-stride length condition: 1.
- Desired gait phase `[sin(phase), cos(phase)]`: 2.
- Current joint positions relative to default, joint velocities, previous actions: 87.
- Reference joint positions relative to default and reference joint velocities: 58.
- Reference root orientation relative to the current root (xyzw), and height error: 5.

Phase is reconstructed from consecutive **recorded left-only support onsets**,
with phase zero at an onset and one full turn per gait cycle. It is interpolated
between those onsets; incomplete edge cycles use the nearest observed period.
It does not reset at velocity transitions. This is a contact-derived phase
estimate, not an exact MPC oscillator state, which the existing NPZs do not store.
Reference joint targets are necessary because the randomized upper-body poses
cannot be inferred uniquely from velocity, stride and phase.

## Rewards

Tracking terms are positive bounded exponentials of squared error. Joint error
uses mean squared error over all 29 joints, so reward scale does not grow with DOF
count. Widths and weights are starting values, not validated training optima.

| Term | Weight | Error width |
| --- | ---: | ---: |
| Joint positions | 2.0 | 0.3 rad |
| Joint velocities | 0.5 | 3.0 rad/s |
| Root height | 0.5 | 0.1 m |
| Root orientation | 0.5 | 0.4 rad |
| Feet, wrists and torso positions relative to the root | 1.0 | 0.15 m |

These five reference terms are added to the existing flat G1 locomotion rewards,
including linear/angular velocity tracking, foot air time/sliding, orientation,
action rate, joint acceleration and torque penalties. Existing locomotion weights
are retained. Joint-deviation selectors are mapped to the 29-DOF waist/arm joints;
the finger term is omitted because this asset has no finger joints.

Reference positions are compared in each pose's local root frame. Absolute
world XY displacement is not a tracking reward. There is no separate delayed
stride-distance reward in this first version; stride is represented through the
reference joint/body trajectory, desired phase, and velocity condition.
The baseline pose and air-time shaping remains enabled; tune its weights if it
conflicts with a selected reference. Mass/push randomization is postponed;
root/joint resets are handled by the reference command. Falls and excessive tilt
terminate the episode.

## Training configuration

The task remains `Isaac-Velocity-Flat-G1` with `presets=reference` and the existing
`rsl_rl` PPO configuration. Set `env.commands.base_velocity.motion_directory` to
a single NPZ path to focus on one reference. The default task without this preset
is unchanged.

The 165-input preset trains a new policy. To initialize from the published
480-input ONNX walking actor, use `scripts/reinforcement_learning/finetune_g1_mpc.py`,
which preserves its deployment contract and appends the reference inputs.
The current dataset contains three correlated speed/stride pairs, so it does not
provide independent stride variation at a fixed speed.

## Integration and physical validation

The single-reference RSL-RL wrapper check produced two environments with 165
finite policy observations and 29 joint-position actions. Action processing is
unchanged (`default_joint_pos + 0.5 * action`), the baseline PPO configuration is
identical, and reference exhaustion resets correctly. This was inference/stepping
only: no optimizer updates, checkpoint loading or training occurred.

The integration audit verified name-based joint correspondence and exact initial
joint positions/velocities, root link pose and link-origin linear velocity. The
NPZ uses wxyz while this checkout uses xyzw; conversion happens once on ingestion.
The original cached asset had 9–10 mm waist/torso link-origin differences and
a 1.78 kg mass deficit. The current reference preset uses a separately converted
MPC URDF asset, eliminating those discrepancies. Its rigid-body inertias, COMs,
joint limits, kinematics, mass matrices and gravity torques were checked against
Pinocchio. See `MPC_MODEL_MATCH.md` for details and rebuild instructions.
At reset, both scheduled contacts are true, but measured foot forces start at
zero because of the configurable 2 cm clearance. Both feet loaded above 20 N by
0.08 seconds in the checked clip. Scheduled contacts are reference intent, not
measured contact labels.

Before switching to the matched asset, the audited feedback run fell after 4.335
seconds; plain 50 Hz PD tracking fell after
1.4 seconds. The feedback run had pre-fall position/velocity limit violations
and foot sliding, despite clipped motor requests and zero command/actuator-effort
estimate discrepancy. This is not a successful physical-tracking result or a
proof of reference infeasibility. Those measurements used the previous mismatched mass/inertia model. Estimated
actuator efforts are not independent measured solver torques. No
further controller tuning is required to begin the intended PPO learning test.

Example single-reference PPO command (not executed):

```bash
cd /home/choon/isaaclab/IsaacLab
HEADLESS=1 uv run --no-sync python scripts/reinforcement_learning/train.py \
  --task Isaac-Velocity-Flat-G1 --rl_library rsl_rl --num_envs 128 \
  presets=reference \
  env.commands.base_velocity.motion_directory=/home/choon/isaaclab/wb_humanoid_mpc/generated_motions/conditioned_50_fixed/conditioned_mpc_reference_0000.npz \
  agent.experiment_name=g1_reference_one
```

This selects a separate log namespace and starts a new policy, leaving the
pretrained baseline checkpoint untouched. RL learns stabilization and reference
tracking through the existing position-action interface; it does not receive
motor commands from the experimental controller.
