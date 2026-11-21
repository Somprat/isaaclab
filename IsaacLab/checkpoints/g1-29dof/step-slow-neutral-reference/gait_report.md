# Generated MPC reference gait

MPC-generated trajectory; no policy inference or independent physics simulation

| Stage | Command speed | Measured body-forward speed | Command step length | Measured step length | Command duration | Measured duration |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.125 | 0.119 | 0.100 | 0.102 | 0.800 | 0.806 |
| 2 | 0.188 | 0.182 | 0.150 | 0.148 | 0.800 | 0.800 |
| 3 | 0.250 | 0.237 | 0.200 | 0.197 | 0.800 | 0.793 |

Units: meters, seconds, m/s. Startup/transition frames and the first two seconds of each stage are excluded.

Rising edges of planned contact_flags, quantized to the NPZ sampling rate. Recorded ankle body origin, as in the baseline CSV; not the sole contact frame.
Timing follows the prescribed MPC schedule; it is not independently verified contact timing in physics.
Baseline uses measured physics contact forces, so these are descriptive comparisons, not equivalent stability tests.
Minimum pelvis height: 0.461 m; maximum tilt: 27.77 degrees.
