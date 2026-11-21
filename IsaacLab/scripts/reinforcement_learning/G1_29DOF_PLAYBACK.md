# G1 29-DOF pretrained walking playback

From the IsaacLab repository root, on an NVIDIA RTX GPU:

```bash
HEADLESS=1 uv run --no-sync python scripts/reinforcement_learning/play_g1_29dof.py
```

On this workspace's Slurm login node, allocate the GPU and run in one command:

```bash
srun --account=tritondroids --partition=debug --gres=rtx2080ti:1 \
  --cpus-per-task=2 --mem=16G --time=00:30:00 \
  env HEADLESS=1 uv run --no-sync python scripts/reinforcement_learning/play_g1_29dof.py
```

The default run records 20 seconds, split evenly between forward speed targets of
0.2, 0.5, and 1.0 m/s. Lateral and yaw commands are zero. The camera follows the
robot. Outputs are under `/home/ssuratannon/unitree_rl_lab/videos/g1-29dof/` by default:

- `g1_29dof_walk_<index>.mp4`: 1280 x 720 video at 50 FPS, numbered from `0000`.
- `velocity_tracking.csv`: target speed, calibrated policy input, actual forward
  velocity, root height, and uprightness for every control step.

Videos receive a new index on each run; the CSV is replaced. Use a separate
output directory (`--output <directory>`) to retain each run's measurements.
`--duration 3` performs a short smoke test. A fall raises an error instead
of silently resetting the robot; a partial video and CSV may remain.

The first invocation downloads about 30 MB of official assets. To download
without starting the simulator, add `--download-only`. Subsequent invocations
reuse local files. `--assets <directory>` changes the cache location.

The published policy undershoots the raw commands in this setup, especially at
low speed. The three targets use fixed policy inputs of 0.27, 0.53, and 1.05 m/s,
respectively. This is a calibration for this flat-ground demonstration, not a
feedback controller or a change to the weights. The console reports measured
segment averages after excluding the first second of each transition. The CSV
retains every sample, including startup and transitions.

## Gait evaluation

Record the existing three target speeds for 15 seconds each, with foot positions
and normal contact forces in the CSV:

```bash
HEADLESS=1 uv run --extra isaacsim --extra video --with onnxruntime --with pyyaml \
  python scripts/reinforcement_learning/play_g1_29dof.py \
  --duration 45 --evaluate-gait --output checkpoints/g1-29dof/gait-evaluation
```

This retains the calibrated policy inputs above. Add `--raw-commands` to send
0.2, 0.5, and 1.0 m/s directly instead. Each output directory should be unique
to preserve its CSV files; videos are numbered automatically.

`velocity_tracking.csv` includes foot origin positions and vertical normal forces
sampled at 50 Hz. Touchdowns use 10 N on / 5 N off hysteresis, with 60 ms of
stable state required for both touchdown and liftoff. The summary excludes the
first two seconds of each segment (`--settle-time` changes this). Only complete
same-foot cycles inside the retained interval contribute to stride metrics.

- `gait_summary.csv` and `gait_summary.json`: actual speed, tracking error,
  mean stride length, stride period, period variability, and step frequency.
- `gait_strides.csv`: individual same-foot cycles and horizontal foot origin
  displacement between consecutive touchdowns (stride length).
- `gait_touchdowns.csv`: timestamp and position of each retained touchdown.
- `gait_trace.png`: velocity, foot height, and foot force traces.
- `gait_speed_summary.png`: measured speed, stride length, and cadence across targets.

Step frequency counts both feet: it is twice the reciprocal of mean stride
period. Cadence is step frequency times 60. Stride length is a full same-foot
cycle, rather than the distance between opposite feet. These are estimates from
50 Hz body origin and force samples, not pressure-center measurements. The force
and debounce thresholds are recorded here so contact detection can be audited.

Recompute summaries without simulation:

```bash
uv run python scripts/reinforcement_learning/evaluate_g1_29dof_gait.py \
  checkpoints/g1-29dof/gait-evaluation/velocity_tracking.csv
```

Compare cadence, stride length, and cycle variability across speeds. A fixed
cadence with shortening strides at low speed can explain a marching appearance,
but these metrics alone do not establish that a gait is unnatural; that requires
a reference appropriate to the robot's dimensions and a review of the video.

## Compatibility

This script runs Unitree's published ONNX velocity policy on its matching
29-DOF USD asset using Isaac Sim PhysX by default, or Newton/MuJoCo-Warp with `--backend mjwarp`. The deployment YAML determines joint
ordering, observation scaling, five-frame history, target offsets, and gains.
The controller runs at 50 Hz with four 5 ms physics steps per action. The ONNX
network executes on CPU; simulation and rendering use the GPU.

The existing `Isaac-Velocity-Flat-G1` task, `g1-flat-physx.pt` checkpoint, and
RSL-RL playback command are unchanged. This is a separate inference entrypoint,
not an RSL-RL training checkpoint or an MPC reference-tracking policy.

Dependencies are supplied by the existing workspace environment: IsaacLab,
Isaac Sim, ONNX Runtime, PyYAML, and the video extra. `--no-sync` deliberately
uses that environment without changing its installed packages.

## Provenance

- Policy and deployment YAML: `unitreerobotics/unitree_rl_lab`, revision
  `4960b84732b0c2ec593dccbfe963fda1bcd7b1e3`, under
  `deploy/robots/g1_29dof/config/policy/velocity/v0/`.
- USD robot: Hugging Face dataset `unitreerobotics/unitree_model`, revision
  `323e350252b9c3aee9c40acbfdad84f6ce46a5ac`, under
  `G1/29dof/usd/g1_29dof_rev_1_0/`.
- Motor limits and armature follow the official robot configuration at policy
  publication, commit `a7c9efabf6c029401de9bd3c65d4e567ca81c25f`.

The download cache includes `sources.json` with source URLs and SHA-256 hashes
of the cached files. Weights and robot assets retain their upstream licensing.

## Newton/MuJoCo-Warp comparison

```bash
HEADLESS=1 uv run --no-sync python scripts/reinforcement_learning/play_g1_29dof.py \
  --backend mjwarp --duration 45 --evaluate-gait \
  --output checkpoints/g1-29dof/baseline-mjwarp
```

Choose a fresh output directory for subsequent runs. The historical PhysX results
are in `checkpoints/g1-29dof/baseline-physx`. The Newton comparison uses the same
original robot asset and ONNX deployment interface, rather than the MPC-matched
asset. It preserves calibrated policy inputs 0.27, 0.53, and 1.05 m/s. Use
`--raw-commands` only for a separate uncalibrated evaluation.

Newton uses the locomotion task's MJWarp solver configuration, with 2 internal
substeps, implicitfast integration, and Newton contact generation. Both backends
use a 5 ms outer physics step and a 20 ms policy step. Solver/contact behavior
can differ despite matching assets and controls.

The run writes `evaluation_run.json`, `gait_report.md`, summary/stride/touchdown
CSVs, plots, a raw state trace, and an MP4. The raw trace now also records world
heading, world yaw rate, body lateral velocity, and world horizontal velocity.
The summary includes heading change and lateral displacement. A detected fall
stops inference and is recorded; any complete settled segments are summarized.
