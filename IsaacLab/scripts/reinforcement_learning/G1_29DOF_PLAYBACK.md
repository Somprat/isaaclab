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

## Compatibility

This script runs Unitree's published ONNX velocity policy on its matching
29-DOF USD asset using Isaac Sim PhysX. The deployment YAML determines joint
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
