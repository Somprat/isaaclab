# G1 29-DOF gait evaluation

Backend: newton_mjwarp. Completed duration: 19.98 s.
Fall recorded: no. No training performed.
Minimum pelvis height: 0.759 m. Maximum tilt: 6.24 degrees.
Policy inputs (m/s): see trace. Seed: 42.

| Target m/s | Actual body-forward m/s | Full stride m | Step interval s (approx.) | Steps/min | Lateral displacement m |
|---:|---:|---:|---:|---:|---:|
| 0.1 | -0.721 | 0.258 | 0.176 | 340.3 | 0.560 |
| 0.2 | -0.721 | 0.252 | 0.174 | 344.1 | 0.715 |
| 0.2 | -0.685 | 0.243 | 0.177 | 339.9 | 0.249 |

The first 2 seconds of each speed segment are excluded. Stride means successive same-foot touchdowns; step length and duration are approximately half the stride values for symmetric alternating gait.

Forward speed is measured in the body frame. Lateral displacement is measured in world coordinates. Heading/lateral velocity fields in gait_summary.csv help distinguish turning from sideways motion.

Touchdowns use 10 N on / 5 N off force hysteresis and 60 ms debounce at 50 Hz. Foot positions are ankle body origins. This is one flat-ground run without perturbations; it does not establish robustness or contact slip.

See gait_summary.csv, gait_strides.csv, gait_touchdowns.csv, velocity_tracking.csv, gait_trace.png, gait_speed_summary.png, and evaluation_run.json.

## Reference command comparison

Measured step values below are half the same-foot stride means, assuming alternating symmetric steps.
Transitions are excluded; contact detection restarts after each episode reset.

| Segment | Commanded step m | Measured step m (approx.) | Commanded duration s | Measured duration s (approx.) |
|---:|---:|---:|---:|---:|
| 1 | 0.100 | 0.129 | 0.800 | 0.176 |
| 2 | 0.150 | 0.126 | 0.800 | 0.174 |
| 3 | 0.200 | 0.122 | 0.800 | 0.177 |

These are policy-inference measurements, not kinematic replay. The policy was trained earlier and does not receive step duration directly.

## Upper-body correction

The original generator sampled broad random waist/arm/wrist targets even during step-conditioned walking. Step conditioning now defaults to neutral upper-body targets from the reference configuration; explicit `--upper-body-mode random` restores random motion. This is a change to reference targets, not a clamp on exported joint states or policy actions.

Regenerating the slow clip reduced maximum absolute arm angles from 141.8 to 10.6 degrees and waist angles from 59.2 to 9.3 degrees. The reference still has large lower-body/posture excursions (minimum pelvis height 0.461 m, maximum tilt 27.8 degrees). Fixing the random arms does not establish reference feasibility.

The existing fine-tuned policy still walks backward and steps much faster than the requested 0.8 s. No retraining or policy-action modifications were made. Upper-body cleanup does not resolve this separate tracking failure.
