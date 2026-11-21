# G1 29-DOF gait evaluation

Backend: newton_mjwarp. Completed duration: 20.00 s.
Fall recorded: no. No training performed.
Minimum pelvis height: 0.758 m. Maximum tilt: 2.49 degrees.
Policy inputs (m/s): [0.27, 0.53, 1.05]. Seed: 42.

| Target m/s | Actual body-forward m/s | Full stride m | Step interval s (approx.) | Steps/min | Lateral displacement m |
|---:|---:|---:|---:|---:|---:|
| 0.2 | 0.243 | 0.127 | 0.261 | 229.7 | -0.020 |
| 0.5 | 0.517 | 0.294 | 0.285 | 210.8 | 0.045 |
| 1.0 | 1.007 | 0.591 | 0.294 | 204.4 | 0.667 |

The first 2 seconds of each speed segment are excluded. Stride means successive same-foot touchdowns; step length and duration are approximately half the stride values for symmetric alternating gait.

Forward speed is measured in the body frame. Lateral displacement is measured in world coordinates. Heading/lateral velocity fields in gait_summary.csv help distinguish turning from sideways motion.

Touchdowns use 10 N on / 5 N off force hysteresis and 60 ms debounce at 50 Hz. Foot positions are ankle body origins. This is one flat-ground run without perturbations; it does not establish robustness or contact slip.

See gait_summary.csv, gait_strides.csv, gait_touchdowns.csv, velocity_tracking.csv, gait_trace.png, gait_speed_summary.png, and evaluation_run.json.
