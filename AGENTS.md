# Workspace guidance and project notes

This file covers `/home/choon/isaaclab`, alongside `IsaacLab/` and `LIP/`.
Read more-specific guidance when working inside a subproject.
Update these notes at meaningful milestones, changed assumptions, or handoffs; not after every edit.

## Working conventions

- Preserve the user's unrelated changes. Do not start training unless requested.
- Use `uv run --project IsaacLab --no-sync python` from this workspace for Python tools.
- Keep diagnostic replays separate from production controllers and checkpoint configuration.
- State whether a replay uses physics, policy inference, prescribed body motion, or exact joint placement.
- Do not interpret successful kinematic replay as proof of physical tracking or balance.

## Closer-target fine-tuned inference (2026-11-20)

- Inference and MP4 script: `IsaacLab/scripts/reinforcement_learning/play_g1_reference.py`. From workspace root: `HEADLESS=1 uv run --project IsaacLab --no-sync python IsaacLab/scripts/reinforcement_learning/play_g1_reference.py --checkpoint IsaacLab/logs/rsl_rl/g1_lipm_online_hands_finetune/run_TY9vx7dE/training/model_499.pt --duration 20 --record-full-state --output LIP/previews/g1-closer5cm-TY9vx7dE-reproduced`. Choose a fresh output directory each time. MP4 is saved automatically as `g1_reference_model_499_0000.mp4`.
- Reproduction guide: `LATEST_POLICY_REPRODUCTION.md` at workspace root. Covers the TY9vx7dE checkpoint, required assets/configs, playback and original-walking fine-tuning commands, current controller/target settings, LIP CoM height, and MuJoCo replay from the latest full-state recording. Local links, shell syntax, and required checkpoint/reference paths verified without starting training.
- Latest completed fine-tune and rendered checkpoint: `IsaacLab/logs/rsl_rl/g1_lipm_online_hands_finetune/run_TY9vx7dE/training/model_499.pt`. Saved config enables 0.05 m world X reach reduction (linear 1-7 s ramp), world-identity orientation, no horizontal capture, full online IK arm ownership, and immediate direct placement with zero prescribed arm velocity. Body and legs retain policy control and physics.
- Video: `LIP/previews/g1-closer5cm-TY9vx7dE-499/closer5cm_TY9vx7dE_policy.mp4`. Single 20 s / 1000-step live policy rollout completed with no fall/tilt termination; only timeout/reference completion at 20 s. Verified all 1000 video frames and final-frame decode. Full state recording, tracking CSV and metadata saved alongside for future MuJoCo comparisons. This is a single rollout, not a broad stability evaluation.

## Prior G1 hand-control status (2026-11-19)

- Latest completed fine-tune: `IsaacLab/logs/rsl_rl/g1_lipm_online_hands_finetune/run_jKZtMlgA/training/model_499.pt`. It was transferred from the original walking `g1_lipm_first050_2gpu/2026-10-29_22-36-50/model_1499.pt` using `IsaacLab/scripts/reinforcement_learning/finetune_g1_online_hands.sh`, then trained for 500 iterations.
- The saved run enables `direct_arm_placement=true`, `immediate_arm_placement=true`, `ik_only_arms=true`, `orientation_frame=world`, `target_frame=world`, and `horizontal_hold=false`. Configured position targets continue past 7 s: X/Y are offsets from the moving torso along world axes; Z is absolute world height. Both hand-link orientation targets are world identity from the start.
- Online pose DLS IK uses damping 0.05, a uniform 0.05 rad per-joint correction cap, 0.9 soft-limit factor, and one solve per 20 ms control interval. Policy arm residuals and reference accumulation are bypassed. The arm state is written immediately at each update and held at that angle with prescribed zero arm joint velocity at every physics substep. Arm actuator stiffness and damping are zero; legs and body retain policy control and physics. Zero reported joint velocity does not encode the pose jump at the next IK update.
- Added hand-orientation reward and penalty per hand following general-lowbody: `exp(-angular_quaternion_error_rad / 0.5²)` at weight +1.0, and angular error in radians at weight -0.5. Existing hand position, velocity, and body reward settings remain as configured.
- Latest restored comparison replay: `LIP/previews/horizontal-ikonly-world-identity-restored/exact_arm_commands.mp4`. It uses older `run_5ogXH686` recorded body motion, fresh IK per frame, and exact arm joint placement, with no physics stepping or policy inference. This replay cannot establish live balance. A different checkpoint and body trajectory prevent exact trace parity with live inference.
- Previous live inference used `run_QjzheWXO/model_499.pt` with immediate placement as a playback override, and completed a single 20 s rollout without a fall. The newest `run_jKZtMlgA/model_499.pt` has not yet been rendered or evaluated in this conversation. Its walking stability and hand-orientation benefit remain unverified.
- Relevant commands: inference via `IsaacLab/scripts/reinforcement_learning/play_g1_reference.py --checkpoint IsaacLab/logs/rsl_rl/g1_lipm_online_hands_finetune/run_jKZtMlgA/training/model_499.pt --duration 20 --output <fresh-output-dir>`; fine-tuning via `IsaacLab/scripts/reinforcement_learning/finetune_g1_online_hands.sh --immediate-arm-placement --world-identity-hand-orientation --hand-target-frame world`.
- Workspace notes preserve historical milestones below. Older phrases such as “latest” and earlier pending-work statements refer to their dated milestone only.

## Exact-command MuJoCo comparison (2026-11-20)

- IsaacLab closer-target configuration implemented: `HandPositionCommandCfg.world_x_reach_reduction_m`, set by fine-tuning `--hand-forward-reduction 0.05`, subtracts world X linearly from 1 to 7 s and retains the reduction thereafter. Matches the closer replay adjustment, independently of the existing reach curriculum. Target velocity includes the ramp derivative. Default zero preserves older checkpoints; saved command configuration restores the adjustment in inference. Use existing `finetune_g1_online_hands.sh` with `--immediate-arm-placement --world-identity-hand-orientation --hand-target-frame world --hand-forward-reduction 0.05 --initial-hold 1.0 --reach-duration 6.0`. Original walking weights remain the shell script's source. CPU trajectory/velocity/config round-trip validation, three existing orientation/hold tests, and fine-tuning dry-run with a temporary walking-policy transfer passed. No training launched.
- Closer-target comparison: moved both world X hand targets -0.05 m using a linear ramp from 1 to 7 s, keeping original Y, absolute Z, world-identity orientation, measured body motion, and capped online IK. Saved `horizontal-ikonly-world-identity-restored/closer_5cm_original_body.mp4` and `mujoco-QjzheWXO-online-ik-measured-body/closer_5cm_QjzheWXO_body.mp4` under `LIP/previews`, with matching CSV/metadata alongside. Both are 20 s / 1000 frames, exact arm placement without physics or policy inference. Mean post7 position errors: original-body 0.55/0.54 mm, Qjz-body 0.43/0.67 mm (previous Qjz-body 5.6/26.3 mm). Supports reach geometry as a contributor to the straighter-arm tracking deficit; live policy performance with closer targets is untested. Production controller/configuration unchanged.
- The older `g1-world-pose-QjzheWXO-499-immediate` preview did not record full root and joint states, so its exact rendered rollout cannot be reconstructed from that folder. Re-ran the same `run_QjzheWXO/model_499.pt` checkpoint with the immediate-arm-placement override and saved full state in `LIP/previews/g1-world-pose-QjzheWXO-499-immediate-state/full_state_replay.npz`. This is a new 20 s policy rollout, with small trajectory differences from the older preview.
- Replayed its recorded pelvis pose and all arm, leg, and waist joint targets in MuJoCo: `LIP/previews/mujoco-QjzheWXO-all-joint-commands/recorded_all_joint_commands_mujoco.mp4`. This uses exact joint placement and `mj_forward`, with no physics stepping or policy inference. All 14 logged IK arm targets equal the measured arm angles at every sampled control frame. MuJoCo forward kinematics of the measured poses agree with the IsaacLab hand log within 1.4 micrometers.
- `play_g1_reference.py --record-full-state` records root pose, all measured joint angles and targets, and world hand targets for this diagnostic. `replay_raw_hand_commands.py --mode all-commands` applies all recorded joint targets; `--mode commands` applies only recorded arm targets with measured leg and waist poses. Pelvis pose is measured in both because it has no joint-position target. The output is kinematic command visualization, not a live balance or actuator test.
- The `all-commands` video is not the comparison to `horizontal-ikonly-world-identity-restored`: placing leg and waist targets alongside the measured pelvis makes the body jump because those targets differ from measured joints by up to 0.83 rad. Corrected comparison: `LIP/previews/mujoco-QjzheWXO-online-ik-measured-body/online_ik_measured_body_mujoco.mp4`. This uses the measured pelvis/leg/waist trajectory and recomputes the arms from each current pose and original world hand targets, with one capped DLS IK solve per frame and previous displayed arm angles as the next seed. Its solved arms match the live run's next-frame IK arm state to mean 1.1e-6 rad and max 3.6e-5 rad. MuJoCo replay remains kinematic, with no physics or policy inference.

## G1 arm investigation (2026-11-16)

- Checkpoint discussed on 2026-11-16: `IsaacLab/logs/rsl_rl/g1_lipm_online_hands_finetune/run_LQ5yBA4s/training/model_499.pt`.
- Its source walking policy is `g1_lipm_first050_2gpu/2026-10-29_22-36-50/model_1499.pt`.
- Training now retains configured position targets past seven seconds; only orientation is captured. Historical horizontal-hold replays still capture measured XY and world orientation.
- The existing `LIP/scripts/replay_hand_horizontal_hold.py` is kinematic: it recomputes pose IK and assigns arm joint positions exactly; it does not simulate motors.
- Its source is `LIP/previews/perfect-arm-replay-source`, from an older checkpoint (`run_5ogXH686`). Keep that provenance explicit.
- At this historical stage, the inference controller `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/core/velocity/mdp/hand_actions.py` had a shoulder-pitch-only cap of 0.15 rad; other arm joints used 0.05 rad. The current direct-placement mode uses 0.05 rad for all arm joints.
- The torque replay requested here must use 0.05 rad for ALL arm joints, independently of that inference edit.
- Previously measured shoulder-pitch corrections hit the 0.05 rad step cap on every post-seven-second sample. This was not an angle-limit or torque-limit hit.
- Removed on request: unrestricted IK replay output and the 14-second reach output. Do not recreate them incidentally.

## Torque replay milestone (2026-11-16)

- Implemented `LIP/scripts/replay_hand_horizontal_hold_torque.py`; see `LIP/scripts/G1_TORQUE_REPLAY.md` for the command, model assumptions and validation.
- Output: `LIP/previews/horizontal-hold-torque005/torque_replay.mp4`.
- Body references drive mocap weld/joint constraints each physics substep; actual arm states are never overwritten after initialization. Native torque-limited PD actuators use Kp=40, Kd=10, implicitfast, timestep 0.001 s, IK interval 0.02 s.
- This diagnostic intentionally retains historical measured XY/orientation capture at 7 s. Contacts are disabled; body constraints provide external support. It is not a free-walking balance test.
- Arms still miss targets under torque dynamics: mean post7 errors 18.85/17.51 cm. Capture hides some horizontal error, so inspect pre-capture reach too.
- Half-timestep validation changed hand-error traces by <0.3 mm. Body tracking max: 4.29 mm pelvis, 0.37 deg root orientation, 1.25 deg lower joint.
- Behavioral regression: `LIP/tests/test_hand_torque_replay.py` (1 passing test covering state continuity, native PD force/saturation, IK cap and physics motion).

## Direct shoulder actuator experiment (2026-11-16)

- `LIP/scripts/replay_shoulder_actuator.py` bypasses IK entirely. Both shoulder-pitch commands follow initial+[0,0,.04,.13,.27,.36,.40,.40] rad at seconds 0..7, linearly interpolated; other arm joints PD-hold their initial angles.
- Same older recorded body, compliant support and PD physics as the torque replay. Output: `LIP/previews/shoulder-direct-positive040/shoulder_actuator.mp4` and `tracking.png`.
- At 7 s, actual shoulder offsets are +0.4003/+0.3971 rad for a +0.40 target. Mean absolute error during 7–10 s: .00477/.00392 rad. Initial hold drifts about .055 rad under load. No sampled shoulder torque saturation.
- This confirms direct positive shoulder-pitch command tracking in this supported setup; it does not establish ability to achieve the earlier negative-pitch Cartesian reach or isolate gravity/load effects.

## Retimed torque reach (2026-11-16)

- User increased the torque replay shoulder-pitch cap to 0.30 rad; other joints remain 0.05. This supersedes the original uniform-cap requirement for the new experiment only.
- Implemented `--reach-end` in `replay_hand_horizontal_hold_torque.py`, default 14 s: 1 s hold, original 1–7 s Cartesian trajectory retimed to 1–14 s, capture at 14 s. World XY offsets/absolute Z are reconstructed before interpolation; body motion runs at its original speed.
- Output: `LIP/previews/horizontal-torque-shoulder030-reach14/torque_replay.mp4`. Errors at 13.98 s before capture: 9.33/7.78 cm; post-hold mean: 8.98/12.01 cm. This is not a timing-only comparison against the original 0.05-cap baseline.
- Two tests pass, including frame-correct retiming and capture timing. Original training and checkpoint files unchanged.
- Added `--preserve-position` to the torque replay: retains original world-axis XY offsets and absolute world Z after reaching; only orientation is captured. Latest comparison: `LIP/previews/horizontal-torque-shoulder030-reach14-original-position/torque_replay.mp4`. Historical XY capture remains default; use the explicit flag for original-position experiments. Three regression tests pass.

## Preview cleanup (2026-11-18)

- Removed 27 requested preview folders and 13 checkpoint files referenced only by those previews. The two retimed shoulder030 output paths above are historical.
- Kept the protected and unlisted previews, including the standalone shoulder030 replay. Kept run_5ogXH686 and run_LQ5yBA4s model_499.pt because retained previews reference them.

## Live hand-target hold milestone (2026-11-19)

- Completed world-frame seven-second measured XY hold in hand_commands.py using the user's existing capture buffers/code. Captured XY offsets follow torso translation in world axes; world orientation is captured; requested absolute Z remains active.
- Held target XY velocity now matches torso XY velocity. Existing per-environment reset/capture masking retained.
- Enable through --horizontal-hand-hold with --hand-target-frame world; launcher timing defaults remain 1 s initial hold and 6 s reach. Configuration remains opt-in.
- Updated horizontal hold regression and launcher help. Hold and orientation tests pass. Two neighboring test_g1_hand_command_reset.py tests also fail with the new hold application removed (missing body_link_pos_w fixture, outdated world-Z expectation). Ruff is unavailable in the project environment.
- Only hand-target behavior implemented here. Live direct arm placement and physics-loop integration remain pending.

## IK arm ownership milestone (2026-11-19)

- Added HandIKJointPositionActionCfg.ik_only_arms, enabled by the horizontal-hand-hold launcher mode and retained in saved configuration for inference.
- This mode forces full IK influence, bypasses persistent references and policy residual/blend output, and holds measured angles on invalid solves. Existing action dimensions and non-arm command processing are retained.
- Preserved the user's uniform hardcoded 0.05 rad correction cap. Launcher explicitly enables pose IK.
- Configuration import, full-influence override, and syntax checks passed. Live physics evaluation remains pending; command application still uses actuators until the separate direct-state-placement integration is implemented.

## Direct arm placement integration (2026-11-19)

- Added --direct-arm-placement to finetune_g1_online_hands.py; at this milestone it required --horizontal-hand-hold. The requirement was removed in the later world-identity mode. Saved action configuration activates the same code through normal playback config loading.
- HandIKJointPositionAction now interpolates each online IK endpoint across the control interval, with matching joint velocities. It writes arm states before each physics step and restores them via Newton's post-step callback before sensor updates. Whole-body physics continues.
- Launcher preserves the original solver timestep and control interval by converting num_substeps into environment decimation (source .005 s / 2 becomes .0025 s x 8). Uses eager Newton stepping and disables backend-owned decimation for this mode. Arm drive stiffness/damping are zero; body gains remain configured.
- Source verified from LIP/previews/g1_lipm_first050_1499/inference.json: g1_lipm_first050_2gpu/2026-10-29_22-36-50/model_1499.pt. Existing finetune_g1_online_hands.sh transfers this walking checkpoint and launches fine-tuning.
- Validation used temporary transferred weights/config in /tmp/g1-direct-arm-validation, with no training. Saved config reload and live policy inference passed over 1050 and 800 control steps with two environments. Non-reset observation-time arm joint error versus IK endpoints was exactly zero; sampled arm actuator torque was zero. Falls/tilt terminations occurred with full reach and the original walking weights. Stable walking and replay-like Cartesian tracking remain unvalidated.
- Horizontal hold and orientation unit tests pass (2). Existing neighboring reset/Z tests have previously recorded failures. No training started.

## Fine-tuned direct-arm inference (2026-11-19)

- Rendered the run_ftpOc6km/training/model_499.pt checkpoint using saved direct-arm-placement configuration.
- Output: LIP/previews/g1-direct-arms-ftpOc6km-499/g1_reference_model_499_0000.mp4.
- Live policy inference with body physics and prescribed online-IK arm placement; 20 s / 1000 control steps. No falls or tilt terminations in this single rollout. Only reset was timeout/reference completion at 20 s.
- Video and inference/tracking metadata saved together. This is one successful rollout, not a broad stability evaluation.

## Original-position kinematic comparison (2026-11-19)

- Added --preserve-position to LIP/scripts/replay_hand_horizontal_hold.py. It retains the original recorded position targets after seven seconds and captures only world orientation. Historical measured XY capture remains the default.
- Output: LIP/previews/horizontal-ikonly-original-position/exact_arm_commands.mp4 (20 s, 1000 frames). Same older run_5ogXH686 body recording as horizontal-ikonly; exact IK joint placement, no physics stepping or policy inference.
- Pre-seven-second traces match horizontal-ikonly to floating-point precision. Mean post-seven-second 3D position error: left 1.58 mm, right 1.42 mm; maximum 20.40/6.08 mm. Video decoded through its final frame.
- Training and live controller configuration were not changed for this comparison.

## Original position and orientation replay (2026-11-19)

- Added --preserve-orientation to the kinematic replay. Combined with --preserve-position, it performs no seven-second capture and maintains the original torso-relative identity orientation throughout.
- Output: LIP/previews/horizontal-ikonly-original-pose/exact_arm_commands.mp4. Same older run_5ogXH686 recording, fresh IK each frame and exact joint placement, with no physics stepping or policy inference.
- Verified all position targets match the preceding original-position replay, capture metadata is empty, and the 20-second MP4 contains 1000 frames.

## Initial-yaw world orientation replay (2026-11-19)

- Added --initial-yaw-orientation to the kinematic replay. With --preserve-position, both hands keep a constant world quaternion derived once from initial torso yaw, with zero pitch/roll and no seven-second capture.
- Output: LIP/previews/horizontal-ikonly-initial-yaw/exact_arm_commands.mp4. Uses older run_5ogXH686 body recording, fresh IK each frame, exact joint placement, no physics stepping or policy inference.
- Verified constant orientation targets in all 1000 CSV frames, unchanged original position targets, and video decoding through the final frame. Live policy configuration remains unchanged.

## World-identity orientation replay (2026-11-19)

- Added --world-identity-orientation, fixing the hand-link world quaternion to [1,0,0,0] independently of initial torso yaw. Used with --preserve-position.
- Output: LIP/previews/horizontal-ikonly-world-identity/exact_arm_commands.mp4. All 1000 numerical frames exactly match the initial-yaw replay because this recording starts at zero yaw. Video decoded through final frame.
- Same older run_5ogXH686 prescribed body motion, fresh per-frame IK, exact joint placement, no physics stepping or policy inference.

## Torso-following XYZ replay (2026-11-19)

- Added --torso-following-z to the kinematic replay. New target Z = current torso Z + (original recorded target Z - initial torso Z). This retains the original reach trajectory as a vertical offset and follows torso translation along world Z.
- Output: LIP/previews/horizontal-ikonly-torso-xyz-world-orientation/exact_arm_commands.mp4, using --preserve-position --world-identity-orientation --torso-following-z.
- Verified original XY targets, torso-relative/world-axis Z offset at every frame, fixed world-identity orientation, and full 1000-frame video decode.
- Older run_5ogXH686 prescribed body motion with fresh IK and exact joint placement; no physics stepping or policy inference. Live training/controller configuration unchanged.

## Filtered world-X replay (2026-11-19)

- Added --x-velocity-filter-tau to the kinematic replay, requiring --preserve-position. A world X anchor starts at initial torso X and integrates exponentially filtered torso X velocity, estimated from causal backward differences; the original reach X offset is added. Filter starts at zero velocity. Zero tau reproduces torso-following X.
- Output: LIP/previews/horizontal-ikonly-filtered-world-x/exact_arm_commands.mp4, tau 0.25 s, original absolute world Z, existing torso-following world-axis Y, fixed world-identity hand orientation. No seven-second capture.
- Mean post-seven-second 3D hand error: 1.84 mm left / 0.60 mm right. Maximum X anchor lag relative to torso: 17.76 cm, so smoothing shortens the effective forward reach during forward motion. RMS frame-to-frame velocity changes reduced from .0418 to .00791 m/s.
- Verified zero-tau target parity, unchanged Y/Z and orientation, and 1000-frame video decoding. Same older run_5ogXH686 prescribed body motion, fresh IK, exact placement, no physics stepping or policy inference. Live training/controller configuration unchanged.

## Restored original position-frame replay (2026-11-19)

- User rejected filtered X and restored torso-following world-axis XY with original absolute world Z and fixed world-identity orientation, without seven-second target capture.
- Fresh output: LIP/previews/horizontal-ikonly-world-identity-restored/exact_arm_commands.mp4. All original CSV columns match horizontal-ikonly-world-identity exactly over 1000 frames; full video decoding verified.
- Same older recorded body motion with fresh IK and exact placement, no physics stepping or policy inference. Experimental filtering remains opt-in and was disabled for this render.

## Fixed-world hand targets in IsaacLab (2026-11-19)

- Added HandPositionCommandCfg.orientation_frame (default torso for older configs). World mode converts configured constant world quaternions into the current torso frame for online IK each update.
- Added --world-identity-hand-orientation to fine-tuning: identity xyzw quaternions from episode start with horizontal hold disabled. Original world-axis torso-following XY and absolute world Z targets persist beyond seven seconds.
- --direct-arm-placement now works independently of --horizontal-hand-hold and still enables full pose IK ownership and direct state writes. Saved config restores these settings for inference.
- Validated continuation from run_ftpOc6km/training/model_499.pt through dry-run and saved-config round trip. Two hand orientation/hold tests pass, including fixed world orientation under torso rotation and original target preservation across seven seconds. Live performance with this new configuration remains untested; no training launched.

## World-pose fine-tuned inference (2026-11-19)

- The run_QjzheWXO/training/model_499.pt checkpoint has direct_arm_placement=true, horizontal_hold=false, orientation_frame=world, target_frame=world.
- Rendered LIP/previews/g1-world-pose-QjzheWXO-499/g1_reference_model_499_0000.mp4 using saved configuration. Live policy inference with body physics and exact online-IK arm placement.
- Single 20-second / 1000-step rollout completed with no falls or tilt termination. Only timeout/reference completion at 20 s. Verified MP4 frame count and decoding of final frame.

## Immediate arm placement mode (2026-11-19)

- Added immediate_arm_placement action option (default false), --immediate-arm-placement in fine-tuning and playback. Training flag implies direct placement. Endpoints are applied immediately and held with zero arm joint velocity at every pre/post physics boundary; body physics continues.
- Existing interpolation remains available for older saved configs. Saved action flag restores immediate mode during inference. MuJoCo replay unchanged.
- Three focused tests pass and fine-tuning dry-run succeeds. Existing run_QjzheWXO/model_499 evaluated with immediate override for 20 s: LIP/previews/g1-world-pose-QjzheWXO-499-immediate/g1_reference_model_499_0000.mp4. Only timeout/reference completion at 20 s, no falls. Full video decode verified. No training launched.

## Hand orientation reward milestone (2026-11-19)

- Added only hand-orientation reward terms based on general-lowbody's reward_ee_reference_orientation_exp and penalty_ee_reference_orientation_l2: per hand, exp(-angular quaternion error / 0.5²) at weight +1.0 and angular error in radians at weight -0.5. Existing position, velocity, and body reward formulas/settings unchanged.
- Angular error compares measured hand-link orientation against the current configured target in torso coordinates; with orientation_frame=world this equals world target tracking.
- Added typed reward terms to G1 online-hands config and reattached them in finetune launcher when continuing from older checkpoint configs. Orientation reward test and config-load check pass. Fine-tuning dry-run succeeds. No training launched.
