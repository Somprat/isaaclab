# G1 LIPM task-space reference adapter

Select `presets=lipm` with `Isaac-Velocity-Flat-G1`. It uses the matched 29-DOF G1
asset, 29 joint-position actions and existing PD gains. The `reference` and default
velocity presets are separate. No joint reference trajectory or IK is required.

The selected clip is `../LIP/generated_trajectories/step_020_g1_startup.npz` relative
to IsaacLab. Override `env.commands.base_velocity.motion_directory` with a single
NPZ or a directory of compatible clips. Each episode starts at a clip's first
sample; exhaustion is a timeout, not a wraparound jump.

## Startup and swing generation

`LIP/LIPM/trajectory.py` owns generation, called by `demo_LIPM_3D_vt.py`.
The default 0.8 s startup keeps both feet fixed. A quintic CoM trajectory starts
at the measured position/velocity with zero acceleration, then joins the
left-support LIPM with continuous position, velocity and acceleration. Its
terminal capture point is chosen to produce the requested first landing length
and width. The endpoint is optimized for low jerk; startup duration can increase
up to 2 s if needed. This is a startup boundary-value solve, not a walking MPC.

The implied ZMP must remain at least 3 mm inside the convex hull of the captured
foot contact points, and horizontal acceleration is limited to 0.35*g (the
preset friction coefficient is 0.4). Polynomial extrema check the full startup
interval. Infeasible initialization is rejected rather than silently exported.
These checks are for the reduced LIPM, not whole-body dynamics or joint reach.

Horizontal swings use a quintic blend; vertical lift uses a smooth sixth-order
arch. Both have zero velocity and acceleration at lift-off and landing. Phase
stays at zero during startup, then advances through the alternating steps.
The first single-support sample is at the startup endpoint; subsequent support
changes occur at touchdown. There is no extra double-support interval between
ordinary walking steps in this implementation.

## NPZ adapter and observations

`mdp/lipm_motion.py` accepts legacy schema 1 (single support, linear interpolation),
schema 2 (one or two support feet, foot velocities required), and schema 3 (scheduled
gait settings). Schemas 2 and 3 use
cubic Hermite interpolation for CoM/foot position and derives interpolated
velocities consistently. Support flags use sample-and-hold. Wrapped phase is
unwrapped to radians. No interpolation crosses clip boundaries.

`mdp/lipm_commands.py` aligns initial CoM XY and travel heading to the reset robot.
This transform stays fixed. Height is relative to ground. Targets are one control
step ahead so action observations and post-physics rewards use the same target.
The robot resets to a nominal joint pose; NPZ does not contain joint resets.

`mdp/lipm_mdp.py` exposes 19 target values: foot errors (6), actual mass-weighted
CoM position/velocity errors (6), support flags (2), phase sine/cosine (2), desired
step length/width/duration (3). Vector errors are in the current heading frame.
Together with robot state and last actions this gives 115 policy observations.

The LIPM preset replaces joint/root imitation rewards with horizontal sole-position tracking
(weight 2, width 0.08 m), swing-height tracking (1, 0.02 m), CoM position (1, 0.10 m), CoM velocity (0.5, 0.30 m/s),
and support agreement (0.5, contact-force threshold 10 N). During startup support
agreement encourages both feet to be loaded; it does not prescribe load fractions.
Existing stability, angular velocity, slip, effort, joint-limit and action-rate
terms remain. Weights are starting values, not validated tracking optima.

Both `lipm` and `lipm_hands` use the separate `lipm_swing_height` reward:
`exp(-(sole_z - reference_z)^2 / 0.02^2)`, averaged only over feet whose
`support_mask` is false. Double support earns zero. It follows the NPZ lift/lower
trajectory (5 cm peak in the current clip) in the aligned world frame, independently
of XY drift. `lipm_feet` now evaluates XY only. Observations and NPZs are unchanged;
the existing 3D reward function remains available for saved configurations.
New training uses the split rewards; an existing checkpoint's behavior does not
change without further training.

`lipm_loaded_swing` penalizes the fraction of planned swing feet with measured
net contact-force magnitude above 10 N (weight -1.0); double support gives zero.
`lipm_stance_slide` penalizes horizontal sole speed in m/s, gated by actual contact
and averaged over planned stance feet (weight -1.0). Sole velocity includes
`omega cross offset` from the ankle link origin. This replaces the generic
`feet_slide` penalty in both LIPM presets. These weights are initial tuning values;
there is no extra grace period at liftoff or touchdown.

`lipm_feet_com_relative` additionally tracks horizontal `(sole - CoM)` offsets
against `(reference_sole - reference_CoM)` for both feet (weight 1.0, width 0.12 m).
The actual CoM is mass-weighted over robot bodies, not the root position. Its
reward is `exp(-mean_foot_squared_XY_error / 0.12^2)`: common translation of the
robot leaves this term unchanged, but incorrect foot placement relative to the
body reduces it. World-position rewards retain path accuracy and the separate
swing-height term retains vertical tracking. The 12 cm width is fixed; automatic
tolerance tightening through a curriculum is not enabled.

Both LIPM presets also include `waist_velocity`, using the existing joint-velocity
squared penalty with weight -0.005 and selector `waist_.*`. This penalizes the sum
of squared yaw/roll/pitch waist velocities in rad/s, allowing slow balancing
motion while discouraging rapid oscillation. Arm rewards are unchanged. The
weight is an initial tuning value; no torso-angle limit is enabled.

Command metrics log `contact_agreement`, `loaded_swing_fraction`,
`stance_slide_speed_m_s`, and `sole_speed_m_s` separately. The last is mean XY
speed over both soles regardless of contact; stance slide is contact-gated.
Reward contributions also appear separately in training logs. Metrics use the
same command reference and default 10 N threshold as the penalties. Changing
the reward threshold alone does not change the metric threshold.

Arm-reference playback also records torso world XYZ Euler angles, torso-frame
angular velocity, waist joint positions/velocities, and hand world positions and
velocities. Hand torso-relative velocity subtracts torso translation and
`torso_angular_velocity cross hand_offset` before rotating into the torso frame.
`tracking_summary.json` includes `motion_metrics` with mean, RMS, min/max, and
95th-percentile absolute values. Torso-rotation-induced hand velocity is reported
separately; component speed magnitudes are not additive contribution percentages.

## Reset and geometry

Root height is 0.78420225 m with zero initial root/joint velocities. Measured CoM
height is 0.714414 m. Sole points and lowest collision surfaces meet ground within
3e-8 m. Left/right ankle sole offsets `(0,0,-0.035)` match the G1 collision spheres.
The capture script records their contact points and verifies initial loading.
Both feet initially load in nominal-PD simulation; this is not proof of settled
standing. Re-capture after changing the asset, reset posture or foot geometry.

## Training (user launched)

Keep `agent.init_at_random_ep_len=false`: randomized episode age would advance the
reference clock while leaving the robot at its standing reset. The command below
starts a fresh single-gait PPO run with local TensorBoard logging:

```bash
cd /home/choon/isaaclab/IsaacLab
HEADLESS=1 uv run --no-sync python scripts/reinforcement_learning/train.py \
  --task Isaac-Velocity-Flat-G1 --rl_library rsl_rl \
  --num_envs 512 --max_iterations 1500 --logger tensorboard \
  presets=lipm agent.init_at_random_ep_len=false \
  agent.experiment_name=g1_lipm_startup
```

No long training run was launched. Legacy velocity/full-body-reference checkpoints
do not automatically match this 115-input policy. Initial training is an experiment;
reduced-model startup checks do not guarantee learned walking or convergence.

## Walking with arm IK references

Select `presets=lipm_hands` for LIPM walking with arm joint targets generated
offline by IsaacLab's differential IK controller. Generate the default reference:

```bash
cd /home/choon/isaaclab/IsaacLab
uv run --no-sync python ../LIP/LIPM/export_hand_ik_reference.py \
  --motion ../LIP/generated_trajectories/steps_010_020_030_first050.npz \
  --output ../LIP/generated_trajectories/steps_010_020_030_hands.npz
```

Edit `LIP/hand_schedules/random_reach.json` to change hand selection, seed, target
ranges, or reach timing. The goals are in `torso_link` coordinates. A clip starts
at the nominal arm pose, reaches smoothly, then holds its goal. Joint targets and
hand positions are sampled on the same clock as the walking reference. Libraries
must have the same selected joints and hands in every clip; stored order may vary.

The policy adds reference joint positions relative to the robot's default pose,
followed by reference joint velocities, for the selected arms. This adds 28 values
for two 7-DOF arms (143 total policy observations), or 14 for one arm (129 total).
The hands preset additionally enables `lipm_left_hand_position` and
`lipm_right_hand_position`, each with weight 0.5 and width 0.10 m. Each independently
computes `exp(-||hand_torso - target_torso||^2 / 0.10^2)`; errors are not averaged
between hands before exponentiation. The reward uses the rubber-hand frame origin,
or the matched G1 wrist offsets `(0.0415, +/-0.003, 0)` when fixed hand bodies are
merged. An unreferenced hand earns zero for single-hand clips. Targets come from
the existing command clock; IK, observations, and NPZ contents are unchanged.
These are initial tuning values requiring training evaluation.

The position reward is `exp(-mean((q - q_ref)^2) / 0.3^2)` with weight 1.0;
velocity tracking uses width 1.0 rad/s and weight 0.1. These are initial tuning
values. Both rewards average only over referenced joints. Walking and stability
terms remain active. The nominal arm-posture penalty excludes referenced joints
and still regularizes an unreferenced arm. The command reports arm position RMSE.

The hands preset requires an augmented reference and fails on walking-only files.
Override `env.commands.base_velocity.motion_directory` to use another compatible
NPZ or directory. Start a fresh policy because the observation size differs from
the existing walking checkpoint. A training invocation is:

```bash
HEADLESS=1 uv run --no-sync python scripts/reinforcement_learning/train.py \
  --task Isaac-Velocity-Flat-G1 --rl_library rsl_rl \
  --num_envs 512 --max_iterations 1500 --logger tensorboard \
  presets=lipm_hands agent.init_at_random_ep_len=false \
  agent.experiment_name=g1_lipm_hands
```

The exporter checks sampled kinematics and joint limits/speeds. Whole-body balance
and collision-free execution still require evaluation in physics. This preset's
registration and term validation do not establish trained walking/reaching success.

### Two-GPU training

The ready-to-run launcher uses two GPUs, 2,048 environments **per GPU** (4,096
total), and 1,500 iterations. It trains a fresh policy with reference clocks
starting at zero and logs to `logs/rsl_rl/g1_lipm_hands_2gpu/<run>/`.

```bash
cd /home/choon/isaaclab/IsaacLab
bash scripts/reinforcement_learning/train_g1_lipm_hands.sh
```

Use `--dry_run` to inspect the command without launching workers. Set
`ENVS_PER_GPU`, `MAX_ITERATIONS`, `EXPERIMENT_NAME`, `CUDA_VISIBLE_DEVICES`, or
`MOTION_REFERENCE` to override the defaults. `MOTION_REFERENCE` can name one
augmented NPZ or a directory of compatible augmented clips. For example:

```bash
ENVS_PER_GPU=512 MAX_ITERATIONS=1500 \
  bash scripts/reinforcement_learning/train_g1_lipm_hands.sh
```

The default NPZ contains one seeded pair of hand goals. To train across different
goals, export multiple seeds into a separate directory, keeping the hand selection
and controlled joint sets consistent, and pass that directory as `MOTION_REFERENCE`.
Do not mix original walking-only NPZs into that directory. Training on one clip
does not demonstrate generalization to unseen hand positions.

### Playback and tracking reports

After training, substitute the actual run folder in this command. Keep
`params/env.yaml` and `params/agent.yaml` inside the checkpoint's run directory.

```bash
cd /home/choon/isaaclab/IsaacLab
uv run --no-sync python scripts/reinforcement_learning/play_g1_reference.py \
  --checkpoint logs/rsl_rl/g1_lipm_hands_2gpu/<run>/model_1499.pt \
  --preset lipm_hands --device cuda:0 \
  --motion ../LIP/generated_trajectories/steps_010_020_030_hands.npz \
  --duration 20 --evaluate-gait \
  --output ../LIP/previews/g1_lipm_hands_1499
```

Choose a new output directory each time. Preset detection defaults to `auto` from
the checkpoint's saved observation/command settings; an explicit `--preset` is
checked for compatibility. The same script supports `reference`, `lipm`, and
`lipm_hands` checkpoints. A 115-input walking checkpoint cannot be used as a
143-input hands policy. `--evaluate-gait` requires a single NPZ, including when
training used a library; select one held-out clip with `--motion`.

Playback writes the MP4 and `inference.json` with termination events. LIP presets
also write `reference_tracking.csv` and `tracking_summary.json` automatically:

- CoM position/velocity error and left/right sole-position error.
- For arm references, joint-position RMSE (rad), joint-velocity RMSE (rad/s), and
  each selected hand's position error (m), with mean, RMS, and maximum summaries.
- Actual and target hand positions in `torso_link` coordinates. The hand points
  are the exported URDF hand-frame origins, reconstructed from simulated bodies
  and fixed-joint offsets when the importer merged the rubber-hand links.

Errors compare **pre-action state to the reference at the current episode time**.
The training observations target one control step ahead; comparing those targets
to pre-action state would introduce an artificial tracking lag. Auto-reset states
are not substituted for terminal measurements: terminal frames are omitted from
the trace, and the corresponding terminations are listed separately.

`--evaluate-gait` additionally writes `velocity_tracking.csv`, gait summaries,
and plots. LIP gait labels use exported timestamps rather than an assumed FPS;
startup and blended schedule transitions are excluded from settled gait summaries.
The trace's velocity-command column remains the policy's lookahead command.
