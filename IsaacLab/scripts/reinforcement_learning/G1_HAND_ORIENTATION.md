# Hand orientation inputs

Add `--hand-orientation` to `finetune_g1_online_hands.py` to enable full-pose DLS IK for both arms. Without the flag, IK remains position-only.

The default hand-link quaternions are identity: fingers forward, with the mirrored G1 palms facing inward. Set custom targets with:

```bash
--hand-orientation \
--left-hand-orientation 0 0 0 1 \
--right-hand-orientation 0 0 0 1
```

Quaternions use `(x, y, z, w)` order and are normalized. Nonfinite or zero quaternions are rejected. Both orientations are relative to the current torso, independently of `--hand-target-frame`; they rotate with the torso even when positions use world axes.

Programmatically, set `cfg.actions.joint_pos.track_orientation = True` and `cfg.commands.hand_position.left_orientation` / `right_orientation`. At runtime, `command.orientation_targets` contains normalized torso-relative targets with shape `(num_envs, 2, 4)`; `command.orientations()` returns measured orientations in the same frame. Runtime writers must supply finite unit quaternions.

Targets remain constant from the start of the episode. There is no seven-second posture capture or null-space hold. Persistent reference accumulation and the existing joint/step limits remain in effect. Position and orientation share a six-dimensional DLS solve; position does not have strict priority over orientation.

The policy observation and action dimensions remain unchanged. These inputs drive the arm IK controller, not additional neural-network inputs. Configuration is saved in the normal environment YAML; the separate kinematic diagnostic replay is unchanged. No training is started by enabling the code alone.

## Configured position targets with orientation hold for fine-tuning

Pass `--horizontal-hand-hold` to `finetune_g1_online_hands.py`. It enables full-pose IK and captures only world hand orientation once per environment at seven seconds. Position targets always follow the configured reach trajectory and target frame, including the original endpoint after seven seconds. Measured hand positions never replace the IK or position-reward targets. The velocity reward follows the configured position trajectory as well. Each episode reset clears the orientation capture.

This mode sets IK influence to 1, arm policy residuals to 0, persistent reference accumulation off, damping to 0.05 and per-joint increments to 0.05 rad. All seven joints participate. Observation/action dimensions are unchanged. The environment YAML saves the mode for playback. The measured-XY diagnostic replay retains its original capture behavior; training now keeps the configured position targets. Actuator-driven physics does not place joints exactly as the kinematic replay does.
