# Controlled torso-yaw comparison

Only the torso task-space orientation_z weight changes: 0 -> 25 in
robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task_random_reference.info.
This modest setting is now the default. Other optimization settings, including
waist penalties, yaw-rate damping, solver iteration limit, startup settling,
step lengths and step durations, are unchanged. The matched experiments both
use logSize 4000 solely to retain every solver iteration; that logging setting
was not promoted to the default task file.

Both runs use seed 2, neutral upper body, 20 s at 30 Hz, plateau ratio 0.8,
step lengths 0.10/0.20/0.30 m and step duration 0.8 s. Desired world heading is
zero. World and pelvis-relative torso yaw use normalized quaternions, with
relative orientation computed as inverse(pelvis) * torso. Waist yaw comes
from the saved waist_yaw_joint angle. Whole-clip statistics include startup
and transitions. Per-stage statistics and all-frame traces are also saved.

| Metric | Original (0) | Modified (25) |
|---|---:|---:|
| Torso world-yaw RMS, degrees | 4.195 | 3.976 |
| Torso world-yaw max absolute, degrees | 9.853 | 9.977 |
| Torso/pelvis relative-yaw RMS, degrees | 0.620 | 0.656 |
| Torso/pelvis relative-yaw max absolute, degrees | 1.824 | 1.913 |
| Waist-yaw joint RMS, degrees | 0.644 | 0.664 |
| Waist-yaw joint max absolute, degrees | 2.192 | 1.891 |
| Minimum pelvis height, m | 0.75665 | 0.75657 |
| Maximum pelvis tilt, degrees | 11.746 | 11.797 |
| Long-step lateral swing error max, m | 0.01961 | 0.01950 |
| Long-step horizontal swing error max, m | 0.04708 | 0.04742 |

Measured step lengths round to 0.099/0.198/0.297 m in both runs; measured
intervals are 0.806/0.800/0.793 s. Foot/gait plateau statistics exclude
transitions and the first two seconds of each stage, matching the exporter.
Foot contact positions use the configured [0.035,0,-0.035] m ankle offset.

## What the current costs penalize

The original torso task tracks roll/pitch (orientation_x/y weights 100), but
its yaw-orientation weight was zero. Torso yaw-rate damping is 2, and waist-yaw
joint tracking is 2. Therefore yaw is not completely unpenalized, but there was
no direct torso-heading penalty. Adding weight 25 uses the existing torso
orientation residual and existing fixed desired heading; no new cost, state,
constraint, dynamics or solver implementation was added in this experiment.

Most visible world-frame torso rotation follows pelvis rotation rather than
large torso/pelvis twisting. The 25 setting reduces world-yaw RMS by 5.2%, but
peak world yaw does not improve, and relative/waist RMS increases slightly.
This is a modest improvement, not a stationary torso or a complete fix.

## Solver convergence and feasibility limits

Both logs cover t=0 through 20 s: 602 MPC solves, 3010 iterations, five
iterations per solve. ALL 602 solves in each run terminate with
"Maximum number of iterations reached". Neither run establishes convergence
to the configured tolerances; the generator's accepted-file status is not a
convergence or feasibility certificate.

| Final per-solve logged metric | Original | Modified |
|---|---:|---:|
| Dynamics violation SSE, p95 | 1.186e-10 | 1.230e-10 |
| Dynamics violation SSE, max | 1.621e-9 | 1.675e-9 |
| Equality constraint SSE, p95 | 5.883e-6 | 6.227e-6 |
| Equality constraint SSE, max | 1.019e-4 | 1.064e-4 |
| Aggregate constraint residual, p95 | 0.002426 | 0.002495 |
| Aggregate constraint residual, max | 0.01010 | 0.01032 |

These are the optimizer's horizon residuals, combining differently scaled
quantities; they are not distances in meters and do not cover every soft
inequality. Sampled exported states have no URDF joint-limit exceedance,
zero swing-foot force, positive stance vertical force, and no world-vertical
friction-cone exceedance at coefficient 0.4 in either run. These limited checks
and the similar posture/foot metrics show no material regression in this
experiment. They do not establish independent physics balance feasibility,
zero stance slip, or feasibility between exported samples. No RL training ran.

## Artifacts

Under generated_motions/step_slow_20s/:

- reference_torso_compare_baseline.npz
- reference_torso_compare_25.npz
- torso_controlled_comparison/summary.json (whole-gait and per-stage metrics)
- torso_controlled_comparison/{baseline,yaw25}_torso_trace.csv
- torso_controlled_comparison/{baseline,yaw25}_solver.csv
- torso_controlled_comparison/task_{baseline,yaw25}.info
- torso_controlled_comparison/{baseline,yaw25}_gait/

Sibling IsaacLab checkout:

- logs/mpc_reference_preview/torso_yaw_weight25.mp4
- logs/mpc_reference_preview/torso_yaw_controlled_comparison.png

The baseline-mjwarp directory is a separate trained-policy physics evaluation.
Its trace logs root heading, not separate torso orientation or waist-yaw angle,
so an exact torso/waist comparison with that policy cannot be recovered from
that CSV. Its visual appearance motivates the goal, but it is not the matched
MPC baseline in the table above.
