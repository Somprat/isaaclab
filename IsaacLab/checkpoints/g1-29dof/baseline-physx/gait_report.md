# G1 29-DOF gait evaluation

The existing ONNX policy completed one 45-second evaluation without falling. Each target ran for 15 seconds; the first 2 seconds after each command change were excluded. The existing calibrated inputs were retained, rather than feeding the targets directly.

| Target (m/s) | Policy input (m/s) | Actual (m/s) | Stride length (m) | Steps/s | Steps/min | Complete strides |
|---:|---:|---:|---:|---:|---:|---:|
| 0.2 | 0.27 | 0.241 | 0.126 | 3.80 | 228 | 47 |
| 0.5 | 0.53 | 0.506 | 0.289 | 3.49 | 210 | 44 |
| 1.0 | 1.05 | 0.988 | 0.589 | 3.35 | 201 | 42 |

Stride length means consecutive same-foot touchdowns. Average step length is approximately half the stride length in this symmetric, alternating gait.

## Interpretation

The low-speed behavior is rapid short stepping: about 6.3 cm per step at 228 steps/min. As actual speed increases from 0.241 to 0.988 m/s, stride length grows about 4.65 times, while cadence falls about 12%. The policy increases speed by lengthening its strides, with cadence changing in the opposite direction. This supports describing the low-speed pattern as rapid short stepping or shuffling; it does not establish a normative classification of unnatural gait. A suitable G1 reference gait and video review would be needed for that judgment.

All detected settled footfalls alternate left/right, with closely matched left/right stride lengths. Stride-period coefficients of variation are 1.7%, 1.7%, and 1.2%, respectively, indicating regular timing in this run. An independent FFT of left-foot height gives step-frequency estimates of 3.84, 3.53, and 3.38 Hz, consistent with contact-based cadence within the finite recording resolution.

## Measurement limits

These results cover a single seed on flat ground, with no perturbation evaluation. Foot positions and vertical normal forces were recorded at 50 Hz. Touchdown detection uses 10 N on / 5 N off hysteresis and 60 ms debounce in both directions. Only complete strides inside each settled segment are included. Stride lengths use horizontal ankle body-origin displacement, not center-of-pressure measurements. The calibrated 0.27 input overshoots the lowest target: mean speed is 0.241 m/s rather than 0.2 m/s.

## Files

- [Summary CSV](gait_summary.csv)
- [Individual strides](gait_strides.csv)
- [Touchdowns](gait_touchdowns.csv)
- [Raw state trace](velocity_tracking.csv)
- [Summary plot](gait_speed_summary.png)
- [Trace plot](gait_trace.png)
- [Video](g1_29dof_walk_0000.mp4)
- [Run configuration and policy hash](evaluation_run.json)
