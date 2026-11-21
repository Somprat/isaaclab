# G1 29-DOF gait evaluation

Backend: newton_mjwarp. Completed duration: 19.98 s.
Fall recorded: no. No training performed.
Minimum pelvis height: 0.756 m. Maximum tilt: 7.37 degrees.
Policy inputs (m/s): see trace. Seed: 42.

| Target m/s | Actual body-forward m/s | Full stride m | Step interval s (approx.) | Steps/min | Lateral displacement m |
|---:|---:|---:|---:|---:|---:|
| 0.1 | -0.746 | 0.263 | 0.174 | 345.7 | 0.525 |
| 0.2 | -0.743 | 0.263 | 0.176 | 341.6 | 0.623 |
| 0.2 | -0.770 | 0.274 | 0.175 | 343.7 | 0.432 |

The first 2 seconds of each speed segment are excluded. Stride means successive same-foot touchdowns; step length and duration are approximately half the stride values for symmetric alternating gait.

Forward speed is measured in the body frame. Lateral displacement is measured in world coordinates. Heading/lateral velocity fields in gait_summary.csv help distinguish turning from sideways motion.

Touchdowns use 10 N on / 5 N off force hysteresis and 60 ms debounce at 50 Hz. Foot positions are ankle body origins. This is one flat-ground run without perturbations; it does not establish robustness or contact slip.

See gait_summary.csv, gait_strides.csv, gait_touchdowns.csv, velocity_tracking.csv, gait_trace.png, gait_speed_summary.png, and evaluation_run.json.

## Reference command comparison

Measured step values below are half the same-foot stride means, assuming alternating symmetric steps.
Transitions are excluded; contact detection restarts after each episode reset.

| Segment | Commanded step m | Measured step m (approx.) | Commanded duration s | Measured duration s (approx.) |
|---:|---:|---:|---:|---:|
| 1 | 0.100 | 0.131 | 0.800 | 0.174 |
| 2 | 0.150 | 0.132 | 0.800 | 0.176 |
| 3 | 0.200 | 0.137 | 0.800 | 0.175 |

These are policy-inference measurements, not kinematic replay. The policy was trained earlier and does not receive step duration directly.

## Tracking result

The policy stayed upright, but failed to track the slower reference. Settled actual body-forward speeds were approximately -0.746, -0.743, and -0.770 m/s against positive commands of 0.125, 0.1875, and 0.25 m/s. The recorded root moved -14.26 m along world X; the MPC reference itself moved +3.40 m. This confirms backward policy motion rather than a sign error in the commanded speed.

The requested touchdown interval is 0.8 s. Estimated measured step intervals are only 0.174–0.176 s (about 342–346 steps/min). The summary's stride length is an unsigned horizontal distance; use stride_forward_m in gait_strides.csv and actual velocity to check direction.

Minimum pelvis height was 0.756 m and maximum tilt 7.37 degrees. The sole reset occurred at 20.0 s for time_out/reference_finished, not a fall. These results show failed reference tracking despite remaining upright. No controller tuning or training was performed. The existing checkpoint was trained before this slower reference was generated and has no direct step-duration input.
