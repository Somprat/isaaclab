# Exact-command MPC dataset: 50 clips

50 NPZ files, reference_0000.npz through reference_0049.npz.
Each clip: nominal 20 seconds, 601 saved frames, 30 FPS, neutral upper body.
Step conditions: 0.10/0.20/0.30 m, each with 0.8-second duration.
No plateau allocation or transition ramps. Initial settling: 0.8 seconds.
Requested stage boundaries: 6.6667 and 13.3333 seconds; commands latch at
step boundaries, at 7.2 and 13.6 seconds. Length and duration remain independent.

## Duplicates

Different seeds were used, but all 50 pose trajectories are exactly identical.
The user explicitly requested identical commands even if clips repeat.
This is 50 files containing one unique trajectory, not 50 distinct motions.

## Validation

All 50 passed exported-reference integrity, command schedule, measured gait
and posture screening. Total saved frames: 30,050. Nominal duration across
files: 1,000 seconds, including repeats. Minimum pelvis height: 0.7555 m;
maximum pelvis tilt: 11.9268 degrees. Measured stage-mean step lengths:
0.099/0.199/0.298 m. Mean sampled intervals: 0.806/0.800/0.793 seconds.
Timing is prescribed contact-schedule timing, not independently measured
physics contact timing. These checks do not certify independent physics
stability, zero stance slip, or full MPC convergence. No RL training ran.

## Contents

- reference_*.npz: motion arrays and independent step-conditioning metadata.
- generation_manifest.json: commands, seed per file, batch settings and status.
- task_snapshot.info, reference_snapshot.info: exact configuration inputs.
- validation_report.json: per-file measurements, errors and duplicate groups.
- quality_checks/reference_XXXX/: CSV/JSON gait measurements for each clip.
- logs/reference_XXXX.log: generator output for each clip.
- SHA256SUMS: complete-file SHA-256 checksums.

Batch runner: scripts/generate_step_conditioned_batch.py in wb_humanoid_mpc.
Validator: scripts/validate_step_conditioned_batch.py in wb_humanoid_mpc.
Use a fresh output directory for another batch; existing batches are preserved.
