# G1 training-log audit

Latest cTgJrLdg last50 iterations: mean completed episode length1000, reference-finished statistic1.0, no recorded last-episode fallen/bad-orientation flags. Root-heading episode reward1.99739/2, torso-heading0.99517/1. Torso-yaw penalty−0.00326, waist-yaw−0.00223, normalized arm joint-error penalty−0.09270. Value loss0.00087. The logs do not show late optimizer divergence.

v2zonEaZ last50: episode length1000, root-heading0.99707/1, torso-heading0.48445/.5. yzUFUu5L and L12UzWeB also show full-duration episode statistics, but did not include the current heading objectives. Total returns are not directly comparable because objective weights differ.

Checkpoint metadata for cTgJrLdg: model400 common_step9624, scheduled/max sampled reach0.96616; model450 step10824 sampled1; model499 step12000 sampled1. These record sampled difficulty, not realized hand-trajectory progress.

A reset-order defect invalidates interpreting this as full-reach training exposure. ManagerBasedRLEnv._reset_idx calls command_manager.reset before clearing episode_length_buf. HandPositionCommand._resample_command reads the old episode length to set trajectory_start_time and resetting flag. After a20s timeout, start_time becomes20s and the old target is preserved. The episode counter then resets to0. Throughout the new20s episode, elapsed=episode_time−20s is nonpositive, and the reach phase stays0.

Confirmed with two live physics environments and the saved training curriculum: first episode reaches10%; atstep1000, episode_steps0 and trajectory_start20s while sampled difficulty19%; atstep1200(new episode4s), actual reach remains10%. Atstep2000 sampled28%, actual10%; step3000 sampled37%, actual10%. See reset_probe.json. Historical actual targets were not logged and early falls alter the carried target, so do not claim all historical environments stayed exactly10%.

Frozen-policy full-reach tests:32env(first16deterministic, second16stochastic):4/16deterministic complete20s;6/16stochastic complete. Root-heading rewards≈1.427 and1.421 rather than train1.997. Thus action sampling does not explain the mismatch. Exact training-size rollout2048env, saved configs, stochastic actor.train(), common_step_counter12000 before initial reset:1649failed,399completed(19.5%completion), earliestfailure5.02s, root heading1.424/2, torsoheading0.626/1. No gradient updates. See /tmp/g1_exact_train_rollout_summary.json.

Mean action std0.37684 hides a split:15leg/waist outputs std≈0.010–0.125,14ignored arm outputs≈0.527–0.905. Unused arm outputs still appear in entropy and last-action observations; this is a secondary design concern, not demonstrated cause. No need to increase heading reward blindly when training did not realize the intended target motion.

Reward logging computes integrated weighted reward divided by configured20s duration. RSL averages per-step episode summaries; completed episode-length logger uses its rolling completed-episode buffer. Termination logs track last completed termination per environment. These are not margin-to-fall metrics. Correct reset lifecycle, log actual trajectory progress, then validate repeated episodes and frozen policies at full reach before drawing new reward conclusions.
