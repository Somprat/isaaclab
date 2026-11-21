# Latest horizontal-hold policy: velocity and reward audit

Checkpoint: `g1_horizontal_hand_hold/run_pos1_vel05_20261115_170715/model_499.pt`.

Reran 1000 control steps (20 seconds, 50 Hz) with unchanged saved controller and reward configuration. Added read-only logging in a temporary copy of playback. Maximum forward-position difference from the original video rollout: 5.7 mm left, 2.8 mm right. No early reset. This report describes a fresh inference rollout, not telemetry recovered from the original MP4.

## Reward definitions

Per hand, weighted reward rates:

- Position: `exp(-||p_target - p_actual||² / 0.2²)`, maximum 1.0.
- Position penalty: `-||p_target - p_actual||²`.
- Velocity: `0.5 * exp(-||v_target_world - v_actual_world||² / 1.0²)`, maximum 0.5.

All three XYZ components enter each norm. Reward manager multiplies the summed rates by dt=0.02 for actual per-step reward. Reconstructed velocity rewards agreed with direct reward-function calls within 4.2e-8; the sum of logged actual manager components times dt agreed with returned reward within 3.8e-8. Pre-action velocities/positions and sampled rewards use time_s; actual manager components correspond to the following step ending at reward_time_s.

## Rollout phase means

Values below are left/right. Relative velocity is in the rotating torso frame, subtracting torso translation and rotation.

| Time | Relative forward velocity (m/s) | Velocity reward / 0.5 | Position reward / 1.0 | 3D position error (m) |
|---|---|---|---|---|
| 1–4 s | -0.012 / -0.014 | .495 / .494 | .752 / .787 | .103 / .091 |
| 4–7 s | -0.007 / -0.010 | .495 / .495 | .139 / .118 | .289 / .307 |
| 7–10 s | -0.058 / -0.056 | .494 / .492 | .107 / .078 | .301 / .324 |
| 10–15 s | +0.037 / +0.027 | .496 / .497 | .225 / .151 | .256 / .285 |
| 15–20 s | +0.001 / +0.010 | .492 / .494 | .599 / .604 | .144 / .143 |

During 7–10 seconds, backward relative motion occurs on 92%/87% of samples (threshold -0.001 m/s). Yet velocity reward remains 98%–99% of its maximum. Mean world X velocities are +.324/+.320 m/s while targets are +.381 m/s: the hands move forward in world coordinates while falling behind the torso. The 1 m/s tolerance barely penalizes that mismatch. There is no reversed sign in the formula.

At 6.98 seconds, measured forward positions are .165/.160 m against .414/.432 m targets. At 7.02 seconds, captured-hold targets in torso coordinates become .190/.186 m. Capture preserves measured world XY offsets; retained world Z and torso rotation explain why these torso-frame X values do not exactly equal the captured measured X. Position reward jumps upward at capture without a successful reach. Errors remain substantial because Z is not replaced.

Late position-reward recovery evaluates the captured target, not the original forward reach. Thus reducing velocity weight and increasing position weight did not resolve the failed reach or target replacement.

## Training curves

The last 50 iterations average .401/.396 position reward and .493/.492 velocity reward. Velocity reward is nearly saturated throughout most of training. Mean total episode return over those iterations is 236.86. Early zero episode reward logs precede completed episodes; early-to-late changes are not a clean measure of learning. Episode reward logs divide accumulated reward by configured maximum episode duration.

## Physical tracking observation

Original inference telemetry during 4–7 seconds shows per-joint command-tracking RMS errors approximately .02–.05 rad. Shoulder-pitch/roll/elbow peak applied torques are only 2.8–3.7 Nm versus 25 Nm actuator limits. This does not suggest saturation of those major joints. It does not establish adequate IK authority: repeatedly setting a small target lead from measured angles can keep command-tracking errors small while Cartesian reach error grows. A controlled actuator/IK transfer test is still needed before choosing a controller fix.

Artifacts: `velocities_and_rewards.png`, `training_rewards.png`, `reference_tracking.csv`, `velocity_reward_summary.json`.

No training rewards or controller behavior were changed.
