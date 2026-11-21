# Explicit step-length / step-duration references

Use the random reference generator's repeatable `--step-condition LENGTH_M,DURATION_S` option. This is a separate mode from the older `--condition VELOCITY_MPS,FULL_STRIDE_M`; mixing them is rejected. No fixed set of pairs is built into the new mode.

- **Length** is displacement along the initial heading between consecutive alternating-foot touchdown targets. It is half a full stride in a symmetric steady gait.
- **Duration** is the interval between those touchdowns, including double support. It is half a full gait cycle.
- Nominal base forward velocity is derived as `length / duration`.
- The initial feet come from contact-frame forward kinematics at the initial robot state. Left supports the first right-foot swing. Time zero is a virtual preceding touchdown in double support.
- Each step latches the requested length and duration at its start. The next touchdown target is the other foot's anchor plus the commanded forward displacement, preserving the swinging foot's initial lateral offset. Targets stay fixed in world coordinates across MPC updates.
- A quintic interpolation supplies horizontal swing position/velocity with zero endpoint velocity/acceleration. Stance targets stay fixed. Existing vertical swing constraints and contact wrench constraints remain active.
- The foot cost reads these X/Y targets through the reference manager. `--foot-position-weight` (default 1000) sets the position penalty. Swing weight scales from 25% to 100% toward touchdown; stance uses 100%. These are soft costs, not exact touchdown constraints.
- `--double-support-fraction` (default `0.2/0.85`) determines the double-support part of each step. In this mode it replaces `--conditioned-gait` timing; the old gait option only applies to velocity/full-stride mode.

The existing plateau allocation and C2 transitions interpolate **length and duration independently**. The first transition ramps length from zero at finite duration. The requested plateau ratio is frame-based; applied commands are latched at step boundaries, so the applied plateau fraction can differ by up to a step around each boundary. No change can move an already planned touchdown midway through its step.

## Check only

From the host (arbitrary example pairs, no MPC solve or NPZ output):

```bash
docker exec wb-mpc-dev bash -lc '
  cd /wb_humanoid_mpc_ws/src/wb_humanoid_mpc &&
  make generate-g1-random-mpc-npz GENERATOR_ARGS="--step-condition 0.10,0.50 --step-condition 0.20,0.50 --step-condition 0.20,0.30 --duration 20 --fps 30 --plateau-ratio 0.8 --check-conditioning"
'
```

For a future single-motion generation, remove `--check-conditioning` and add `--num-motions 1 --max-attempts-per-motion 2 --output /wb_humanoid_mpc_ws/src/wb_humanoid_mpc/generated_motions/step_reference_smoke.npz`. The generator's existing output behavior can overwrite that chosen filename. Generating a file is not part of the implementation checks.

## NPZ fields

Existing motion arrays and legacy conditioning fields keep their meanings. New step-mode fields:

| Field | Shape | Meaning |
|---|---|---|
| `step_conditioning` | frames × 2 | Applied, step-latched `[length_m, duration_s]` at `command_time` |
| `step_conditioning_requested` | frames × 2 | Continuous requested pair at `command_time` |
| `step_conditioning_target` | frames × 2 | Current requested stage endpoint |
| `step_conditioning_names` | 2 | Explicit units/names |
| `foot_reference_pos_xy_w` | frames × 2 × 2 | Planned world X/Y contact-frame positions at `sample_time` |
| `foot_reference_vel_xy_w` | frames × 2 × 2 | Planned world X/Y contact-frame velocities at `sample_time` |
| `foot_reference_names` | 2 | left, right |
| `planned_steps` | steps × 8 | Step start, liftoff, touchdown, foot index, applied length/duration, touchdown X/Y |
| `planned_step_columns` | 8 | Column names/units |

The step table covers the MPC lookahead beyond the exported clip. Legacy `conditioning` holds derived `[L/T, 2L]`; `conditioning_target`, `condition_stage`, and `is_transition` describe the stage latched for that step. During transitions, `2L` is a nominal full stride; actual successive lengths can differ. The startup command has zero length/velocity. Existing `command_time` is the command applied before a rollout step; `sample_time` is the resulting state time. These are intentionally distinct.

The IsaacLab loader still uses the legacy velocity/full-stride channels. This change does not alter PPO observations, policies, action interfaces, or existing datasets. A later RL change must explicitly consume the new length/duration fields.

## Scope and validation

This implements straight, fixed-heading, flat-ground, alternating biped references, with no turning or terrain planner. No MPC state, control, dynamics, or solver change is required. Contact timing is prescribed, not optimized. The existing swing-height scaling for short swings still applies. Positive requested lengths and durations of at least 0.02 s are accepted as numerical inputs; this does not establish physical feasibility.

`test_step_reference_plan` checks independent geometry/timing, support mode alignment, rotated/translated anchors, zero stance velocity, analytical swing velocity against finite differences, command latching and invalid input rejection. `test_motion_conditioning` checks the existing conditioning behavior. Compilation and schedule tests do not demonstrate achieved step accuracy or physics stability; that needs a separate generated-motion and physics check before dataset production.

## Upper-body posture

Step-conditioned walking defaults to `--upper-body-mode neutral` through the
`auto` setting. Waist, arm, and wrist targets remain at `defaultJointState` from
the reference configuration. Actual joints can deviate under MPC dynamics and
tracking costs; this does not clamp exported poses. Broad random upper-body
motion requires `--upper-body-mode random`. Legacy velocity/full-stride and
unconditioned modes retain random upper-body behavior under `auto`.

## Pelvis-height bound in the generator

`task_random_reference.info` enables `pelvis_height_constraint` with a 0.70 m
world-frame pelvis floor, barrier strength `mu = 1.0`, and relaxation distance
`delta = 0.02` m. The centroidal base is the G1 pelvis. This assumes flat ground
at world Z = 0. Height tracking weights increase from 15 to 150 during the
horizon and from 20 to 200 at the terminal state; the nominal height remains
0.7925 m. The offline generator allows up to five SQP iterations per update
(previously one) so the optimizer can respond to the barrier. Other task files opt in with the same configuration block; absent or
`enabled false` leaves their behavior unchanged.

The lower bound is an explicit `StateSoftConstraint` with a relaxed log barrier
at intermediate and terminal states. It penalizes approaching the floor and
has quadratic continuation below the relaxation distance. This is a soft bound,
not an exact height clamp or a guarantee of feasibility. In this bundled SQP,
general inequality terms are collected but not forwarded to the QP solve, so
registering only a state inequality would not enforce the floor. The explicit
barrier contributes to the optimized cost and its derivatives.

`--height-min/max` control random **height commands**, not this constraint.
Step-conditioned mode still commands `defaultBaseHeight`. Choose a floor below
the nominal/initial height. Validate exported pelvis height and step accuracy
together after changing the floor or barrier settings. Torso-orientation
penalties and RL training are outside this change.

Validation: rebuilt the MPC library and statically linked ROS generator; both
`test_pelvis_height_constraint` tests pass (bound sign, finite-difference
Jacobian, cloning, invalid floors). A 20 s neutral-arm clip with step commands
0.10/0.15/0.20 m at 0.8 s, 30 Hz and plateau ratio 0.8 was generated as
`generated_motions/step_slow_20s/reference_pelvis_height_sqp5.npz`. Direct NPZ
measurements in `generated_motions/step_slow_20s/pelvis_height_sqp5_gait/` show
minimum pelvis height 0.758 m, maximum tilt 8.09 degrees, step lengths
0.101/0.150/0.197 m and durations 0.806/0.800/0.793 s. These are generated
trajectory measurements, not an independent physics stability test. Increasing
barrier strength alone worsened tracking; the final configuration uses mu 1.0
with five SQP iterations. No orientation weights were changed.

## Startup settling

The first MPC solve now occurs after the conditioned contact schedule and foot
anchors have been installed. A step-plan command with length <= 1e-9 m is a
double-support settling interval: both feet stay planted and the next real
swing retains the same foot. Its planned_steps row has liftOff == touchDown
and unchanged from/to positions; that row is not a physical touchdown event.
For the validated 20 s schedule, this gives 0.8 s of settling, followed by the
first right-foot swing, with length 0.0709 m. Positive lengths retain their
prescribed touchdown timing and swing height.

The startup regression test verifies contact modes, constant horizontal/vertical
foot targets through settling, and the first moving swing's foot. Geometry and
conditioning tests pass. In reference_startup_fixed.npz, peak finite-difference
knee speed in the first 0.8 s drops from 16.22 to 0.75 rad/s, and peak speed over
the first 2.4 s drops to 4.48 rad/s (later walking: 4.73 rad/s). Minimum knee
angle in the first 2.4 s is 0.71 degrees, versus -0.92 degrees before. Direct
measurements give minimum pelvis height 0.758 m, maximum tilt 9.82 degrees,
step lengths 0.099/0.150/0.197 m and durations 0.806/0.800/0.793 s. The saved
replay is IsaacLab/logs/mpc_reference_preview/step_slow_startup_fixed.mp4 in
the sibling IsaacLab checkout. No physics-policy evaluation or training is
involved.

## Standing-to-walking support transfer

The settling interval remains 0.8 s for the validated commands. During each
existing double-support interval, `StepReferencePlan::supportWeights` uses a
quintic blend to unload the upcoming swing foot. The initial load distribution
is 50/50; subsequent transfers start from the previous supporting leg. Weights
sum to one and reach zero on the swing foot at liftoff. The quadratic input cost
uses these continuous vertical-force references in step mode. Generator-only R
weights for vertical forces increase from 0.01 to 10 (before the existing 1e-3
scaling); otherwise the optimizer barely followed the unloading reference.
Contact event times, touchdown locations, and the independently supplied step
length and duration are unchanged. The first actual swing still lifts at
0.988235 s and touches down at 1.6 s for this clip.

The opt-in `knee_flexion_constraint` applies a relaxed barrier to both knee
angles at intermediate/terminal states: minimumFlexion 0.15 rad, mu 0.05,
delta 0.05. It is a soft safeguard against straight-leg singularities, not a
hard joint clamp. Joint reference angles remain constant through first liftoff.
`smoothSwing true` enables a C2 vertical arch with zero endpoint velocity and
acceleration; it requires zero configured liftoff/touchdown velocity. Other
configs retain the existing cubic spline unless enabled. Step-mode horizontal
tracking weights now vary continuously from 1 at both endpoints to 0.25 at
mid-swing, eliminating their previous jump at liftoff.

Validation: rebuilt common MPC, centroidal MPC and ROS generator. Tests cover
support-weight conservation/continuity/contact consistency, C2 endpoint and
finite-difference derivatives, knee-bound derivatives and existing independent
step geometry/timing. The new 20 s generated clip is
`generated_motions/step_slow_20s/reference_support_transfer.npz`; direct gait
measurements are in `support_transfer_gait/` beside it. At 1.000 -> 1.033 s,
the left support-knee change drops from 8.56 to 0.85 degrees, with the knee
remaining around 20 degrees bent. Right-foot vertical load before liftoff
(at 0.967 s) drops from about 342 N to 20 N; it is zero after liftoff. Minimum
pelvis height is 0.769 m and maximum tilt 6.32 degrees. Measured step lengths
are 0.098/0.149/0.198 m; durations are 0.806/0.800/0.793 s. These remain
saved-reference measurements, not independent physics verification. Video:
`IsaacLab/logs/mpc_reference_preview/step_slow_support_transfer.mp4` in the
sibling checkout.

## Disable plateau/ramp allocation for step commands

Pass `--no-step-transitions` with `--step-condition` to bypass plateau-ratio
allocation and all length/duration interpolation. Existing ramped behavior
remains the default, and legacy `--condition` mode is unchanged. The flag is
rejected without `--step-condition`.

Direct mode keeps one initial double-support settling interval equal to the
first requested step duration (0.8 s for the examples). After settling, the
first walking step uses the full requested length/duration. Multiple pairs
hold equal shares of the total clip duration (settling lies inside stage one) and switch without ramps;
commands still latch at step boundaries. `--plateau-ratio` is ignored in this
mode. With one pair, the command stays constant for all walking steps.

Example generator arguments:

```text
--step-condition 0.10,0.8 --no-step-transitions --duration 20 --fps 30 --upper-body-mode neutral
```

Validation: step-plan and legacy conditioning tests pass, including direct-mode
settling, first full step, multi-stage boundaries, long-horizon holds and invalid
inputs. The new clip `generated_motions/reference_no_step_transitions.npz`
measures 0.100 m steps and 0.800 s intervals. Minimum pelvis height is 0.777 m;
maximum tilt is 4.37 degrees. The first walking step starts at 0.8 s, lifts at
0.988235 s and touches down at 1.6 s. Pelvis forward displacement at 2 s is
0.127 m, versus 0.015 m in the prior single-condition ramped clip. Peak
finite-difference knee speed in the first 2.4 s is 3.82 rad/s versus 3.84 rad/s.
The sibling IsaacLab replay video is
`logs/mpc_reference_preview/no_step_transitions.mp4`. This is saved-trajectory
replay, not independent physics verification. No dynamics, cost, solver,
posture configuration or RL changes were made for this option.

Equal-third direct-mode example: for a 20 s clip with three conditions, requested
stage boundaries are 6.6667 s and 13.3333 s (settling belongs to the first
stage). With 0.8 s steps, the new commands latch at 7.2 s and 13.6 s; exact
third boundaries and uninterrupted 0.8 s intervals cannot both coincide.
`reference_no_ramp_thirds.npz` uses 0.10/0.20/0.30 m at 0.8 s without ramps,
with 0.8 s initial settling. Measured lengths are 0.099/0.199/0.298 m and
mean sampled intervals 0.806/0.800/0.793 s. Minimum pelvis height is 0.755 m,
maximum tilt 11.93 degrees. Replay is the sibling IsaacLab
`logs/mpc_reference_preview/no_ramp_step_thirds.mp4`. Scheduling and existing
conditioning tests pass. No MPC dynamics, costs, constraints or solver settings
changed for the equal-third allocation.

## Exact-command 50-clip dataset

`generate_step_conditioned_batch.py` runs inside the prepared ROS container and
uses the same no-ramp 20 s reference: 0.10/0.20/0.30 m, each at 0.8 s, 30 Hz,
neutral upper body, 0.8 s settling, equal-third requested stages with step-latched
changes at 7.2/13.6 s. It does not use the older velocity/full-stride wrapper.
It refuses existing output directories, snapshots task/reference configs,
records per-clip commands/seeds/logs and runs up to four independent workers.
Different seeds do not imply diversity in this exact-command neutral mode.
Repeated trajectories are intentionally permitted for the requested batch.

```bash
docker exec wb-mpc-dev bash -lc '
  source /bin/ros_setup.sh &&
  source /wb_humanoid_mpc_ws/install/setup.bash &&
  cd /wb_humanoid_mpc_ws/src/wb_humanoid_mpc &&
  python3 scripts/generate_step_conditioned_batch.py \
    --num-motions 50 --workers 4 --seed 1 \
    --output-dir generated_motions/step_no_ramp_50_exact
'
```

Use a fresh directory for a new batch. After generation, run
`python3 scripts/validate_step_conditioned_batch.py DIRECTORY` in the same
container/environment. Validation checks finite arrays, frame count/FPS,
sample times, exact applied commands and planned stage changes, then exports
per-clip gait measurements and checks measured mean length/duration errors.
Posture screening limits are 0.69 m minimum pelvis height and 30 degrees maximum
tilt. These broad screening thresholds are not a physics feasibility certificate.
The report includes exact pose hashes and duplicate groups within 1e-5 absolute
pose-array difference. No dynamics, solver, cost or training changes are involved
in these batch/validation scripts.
