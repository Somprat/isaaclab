# G1 full-reach training and late-instability diagnosis

## Confirmed root cause: reach curriculum reset timing

`ManagerBasedRLEnv._reset_idx` resets commands before it clears `episode_length_buf`. `HandPositionCommand._resample_command` incorrectly used that buffer to detect an episode reset and to set the reach start time. After a 20-second episode it stored 20 seconds as the next trajectory start, preserved the old target, and then the environment reset the clock to 0. The next 20-second episode never reached the trajectory start. The sampled curriculum fraction still increased and was saved to checkpoint metadata.

A live two-environment reproduction reached 10% in episode1. Subsequent resets reported 19%, 28%, 37% while the actual commanded reach stayed 10%. This is direct evidence of the bug; historical actual reach fractions were not logged, and early falls can yield different carried targets. It is not valid to claim every historical environment stayed exactly10%.

The newest run's last 50 PPO iterations reported heading reward 1.997/2 and torso heading .995/1, with episode length 1000. A frozen replay using 2048 environments, stochastic policy actions, saved training settings and confirmed full reach completed only 399/2048 episodes (19.5%); the root heading reward was 1.424/2. No weight updates were made. Small inference batches and deterministic actions do not explain the discrepancy. The logs largely describe an easier realized reach than the full target used in inference. Stronger heading rewards could not teach balance under disturbances the curriculum failed to deliver.

## The saved MP4 was not a stable success

At 19.98 seconds the saved video trace has 42.7 cm CoM error, 0.866 m/s CoM velocity error, and 3.34 rad/s torso angular speed. Reconstructing torso orientation from the two heading-anchored hand target vectors gives 34.4 degrees total torso tilt; reconstructed yaw agrees with directly logged yaw to within 4e-6 degrees. The 20-second timeout censored developing instability. A separate headless deterministic replay actually triggered bad_orientation at 19.28 s. Small backend/render/batch differences shift the exact failure time, reinforcing the lack of stability margin.

The hardest 30 cm gait stage starts at 14.025 s. CoM tracking deteriorates after several seconds in this stage, from 2.8 cm mean over 10–16 s to 21.2 cm over 19–20 s. The earlier v2zonEaZ replay finished with 4.0 cm CoM error and 0.072 m/s velocity error, but used shorter torso-relative targets. That is not a matched-task comparison.

## Controller-only tests (no training)

| Intervention | Result |
|---|---|
| Original full reach, deterministic16-environment evaluation |4/16 completed20s|
| Full reach with stochastic policy sampling |6/16 completed20s|
| Quarter IK increment gain, deterministic16-environment evaluation |16/16 completed20s; torso heading RMS15.3degrees; mean hand error7.7cm|
| Half reach (single diagnostic environment) |Completed; late CoM error~1.7cm|
| Slow reach,6s (single diagnostic environment) |Failed at19.36s|
| Fixed waist yaw (single diagnostic environment) |Failed at5.24s|
| Low-pass executed arm targets (single diagnostic environment) |Failed at14.6s|
| Frozen arm references after4s (single diagnostic environment) |Completed; heading drift remained; world hand objective no longer enforced|
| Torso-relative targets (single diagnostic environment) |Completed; heading drift remained; changes requested target semantics|

Quarter gain improves survival but is not a complete solution. It is only an experimental inference override; production gain and checkpoint weights were not changed. See the separately labeled quarter-IK-test preview folder and intervention.json.

## Additional implementation findings

IK modifies only arm joints; no left-favoring sign bug was found in heading rewards or target transforms. Position-only persistent IK has no posture/nullspace objective or true reference acceleration limit. Its 0.05 rad per 20 ms correction permits 2.5 rad/s joint-reference motion. The existing action-rate reward penalizes raw policy outputs, not the executed IK references. Anchoring world hand offsets resists torso roll and pitch as well as yaw, increasing arm/body coupling.

The arm joint-error penalty measures lag behind a persistent servo reference. It also penalizes lead needed to hold arms against gravity; with IK-only arms, RL can influence it only through leg/waist motion and body loading. It should not be interpreted as the earlier gaitfix's independently planned arm-joint tracking reward. These are secondary design concerns; the confirmed reset bug should be fixed and verified before further reward tuning.

## Required next step

Correct reset detection using the command term's per-episode command counter, restart the trajectory at time0 on episode resets, preserve current-target continuity on mid-episode resampling, and log sampled versus realized reach. Then retrain from the walking checkpoint with real full-reach exposure. Reserve a sustained final-difficulty phase and evaluate stability in the 30 cm gait stage using roll/pitch, CoM/foot errors and actual hand tracking, rather than timeout completion alone. No retraining was started during this investigation.


## Fix implemented and validated

The production hand command now uses CommandTerm.command_counter == 0 to identify an episode reset. It starts the reach clock at zero, renews the heading anchor, and preserves the separate periodic-resampling behavior. New sampled_reach_fraction and realized_reach_fraction metrics expose declared versus actually commanded reach.

The actual-manager regression failed before the fix and passed afterward. Thirty-eight neighboring LIPM tests also passed; direct Ruff lint/format checks passed. Repository-wide scoped hooks could not resolve pre-existing parent-Git-root paths for license/changelog/LFS tooling.

A fresh GPU physics replay after the fix confirms: at the first 20-second reset, the new trajectory start is 0 instead of 20; four seconds into the next episode, actual target reach is 19%, matching the sampled 19%. No optimizer updates or retraining were performed. This fixes future curriculum execution; it does not teach the existing checkpoint the missing full-reach behavior.
