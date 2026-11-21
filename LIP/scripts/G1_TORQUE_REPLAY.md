# Torque-driven horizontal-hand-hold replay

Run from `/home/choon/isaaclab`:

```bash
uv run --project IsaacLab --no-sync python \
  LIP/scripts/replay_hand_horizontal_hold_torque.py \
  --reach-end 14 \
  --output LIP/previews/my-torque-replay
```

Output directory must be fresh. Optional arguments: `--no-video`, `--duration SECONDS`,
`--physics-dt SECONDS`, `--reach-end SECONDS` (default 14), and `--source RECORDING_DIRECTORY`.

## What is simulated

The default source is `LIP/previews/perfect-arm-replay-source`, the SAME older body
recording used by `recomputed-horizontal-hand-hold/exact_arm_commands.mp4`.
It is not the latest policy checkpoint. No policy runs in this diagnostic.

Pelvis translation and lower-body joint reference trajectories use cubic splines;
pelvis orientation uses SciPy RotationSpline. A moving mocap weld drives the pelvis,
and joint equalities drive the legs and waist. References update every physics step.
These constraints provide external support with finite compliance. Their reactions
move the articulated robot and affect arm dynamics. Actual robot positions/velocities
are initialized once; no dynamic robot coordinates are overwritten thereafter.
A separate MjData is used to reconstruct recorded Cartesian targets and validate FK.

The arms use fresh measured-state pose DLS IK every 0.02 seconds, with damping 0.05,
shoulder-pitch increments capped at 0.30 rad (the user's current setting) and other arm increments capped at 0.05 rad, and target angles clipped to the 90% soft
range. There is no persistent target accumulation. Native affine actuators implement
`tau = clip(40 * (q_des - q) - 10 * qdot, -limit, limit)` inside MuJoCo, allowing
implicitfast integration to account for velocity feedback. Limits are 25 Nm for
shoulders, elbows and wrist roll; 5 Nm for wrist pitch/yaw. Physics timestep defaults
to 0.001 seconds (20 substeps per IK update). Arm targets stay fixed across substeps.

At the selected reach end (default 14 seconds), the replay captures its simulated hand XY offsets and WORLD
orientations, following torso translation thereafter. Original world Z goals remain.
The source 1–7 s reach is retimed to 1–14 s by default: recorded torso-frame targets are first converted to world-axis XY offsets and absolute world Z, then interpolated at source time. XY follows the current simulated torso; body motion stays at normal speed. This retains the HISTORICAL measured-pose capture behavior, not the newer
training behavior that preserves configured XY. The capture can accept a failed reach.

Contacts are disabled to isolate externally supported arm dynamics. Native joint-angle
limits remain enabled. This is not a balance, collision, motor-electronics, or exact
IsaacLab-backend validation. Constraint tracking errors are included in metadata.

MuJoCo reference: https://mujoco.readthedocs.io/en/latest/XMLreference.html
(equality weld/joint constraints and general affine actuators).

## Outputs and validation

- `torque_replay.mp4`: 20-second video by default, targets shown as spheres.
- `tracking.csv`: control-time errors, measured joints, targets and IK increments.
- `torques.csv`: per-physics-step velocity, requested torque and applied torque.
- `metadata.json`: configuration, source, body tracking, and hand-error summaries.

`uv run --project IsaacLab --no-sync python -m pytest LIP/tests/test_hand_torque_replay.py -q --confcutdir=LIP/tests`
checks that moving the body reference does not overwrite dynamic coordinates,
verifies native motor force against PD with saturation, checks the per-joint caps, and
verifies physics advances the arms without placing them exactly at the target.
The full replay also checks the actuator formula at every physics substep.

Validated 2026-11-16 output: `LIP/previews/horizontal-hold-torque005` (005 denotes
0.05-rad cap, not timestep). Mean post-seven-second hand errors: 18.85 cm left,
17.51 cm right; primarily low hand height. The original reach has already failed
before capture (6.98 s errors: 32.55/27.87 cm). Maximum body errors: pelvis 4.29 mm,
orientation 0.37 degrees, lower-joint 1.25 degrees. Halving timestep to 0.0005 s
changes hand-error traces by at most 0.276 mm left and 0.206 mm right.

The original replay script, robot XML, inference controller, and checkpoints are not
modified by this diagnostic. The user's inference-controller edits remain independent of this replay's caps.

## Retimed reach experiment (2026-11-16)

`LIP/previews/horizontal-torque-shoulder030-reach14/torque_replay.mp4` uses shoulder pitch 0.30 rad, other joints 0.05 rad, reach from 1–14 s and capture at 14 s. Body remains externally supported and runs at original speed. Before capture (13.98 s), hand errors are 9.33/7.78 cm. Mean post-capture errors are 8.98/12.01 cm; those use captured XY. Earlier baseline results above used different shoulder caps and must not be attributed to timing alone. Tests cover time mapping, world-frame reconstruction under a different torso pose, unchanged body reference speed, and delayed capture.

## Preserve original hand position after reaching

Add `--preserve-position` to retain the original Cartesian endpoint after the reach;
only world orientation is captured. X/Y are world-axis offsets from the current
simulated torso; Z remains absolute world height. Example:

```bash
uv run --project IsaacLab --no-sync python LIP/scripts/replay_hand_horizontal_hold_torque.py \
  --reach-end 14 --preserve-position \
  --output LIP/previews/horizontal-torque-shoulder030-reach14-original-position
```

The historical measured-XY capture mode is still the default. Three behavioral tests
now include preserving the original endpoint despite a failed reach and later
vertical/horizontal torso motion.
