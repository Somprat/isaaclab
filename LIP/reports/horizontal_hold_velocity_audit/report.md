# Full-rollout velocity audit

Checkpoint: run_20261115_120037/model_499.pt. Re-ran 1000 steps at 50 Hz with unchanged controller configuration and additional read-only velocity diagnostics. No early reset. This is a fresh rollout; the maximum left-hand forward-position difference versus the previous video recording was 5.93 mm.

The walking COM reference has a small initial backward startup motion (mean X velocity -0.0244 m/s during 0–1 s), followed by forward motion. Average COM target X velocity is +0.1941 m/s during 1–7 s, +0.3713 during 7–10 s, +0.4648 during 10–15 s, and +0.6654 during 15–20 s. The torso follows those forward speeds.

During 1–7 s, left/right hands move backward relative to the rotating torso on 90.7%/73.3% of samples, using a -0.001 m/s threshold. Yet their mean weighted velocity rewards are 2.4803/2.4786 out of 2.5. During 7–10 s, they move backward relative to the torso on 90.7% of samples for both hands, averaging -0.0617/-0.0637 m/s, while rewards remain 2.4648/2.4608.

There is no subtraction/sign reversal in the reward. The exact reward function was compared against 2.5 * exp(-sum((v_target_world - v_actual_world)**2) / 1.0**2) at every sampled state; all comparisons passed. Squaring makes exchanging target and actual signs irrelevant. This positive exponential does not impose a no-reward rule on wrong-direction motion. A 0.06 m/s error in one axis alone would still earn about 2.491/2.5.

Targets use world velocity. Before seven seconds, horizontal target velocity equals torso world velocity plus the reach trajectory velocity; after seven seconds it equals torso world velocity. Actual hands can move forward in the world but lag behind the torso. A torso near +0.383 m/s and hand near +0.320 m/s means backward relative motion around -0.063 m/s, yet their world-velocity mismatch is small relative to std=1 m/s. This is a tolerance/objective mismatch, not a reversed reward.

Rewards are sampled from the actual reward function on pre-action diagnostic states, not the historical training scalar or exact previously integrated per-step reward. The manager multiplies weighted values by dt=0.02 for step integration. All three velocity-error components are included in the reward; the plot highlights X for forward/backward interpretation.

For reaching, penalizing negative progress toward the target or conditioning velocity reward on hand-position accuracy would address the mismatch. A strict ban on any backward velocity is inappropriate once the hand needs to correct overshoot. No rewards or controller behavior were modified.
