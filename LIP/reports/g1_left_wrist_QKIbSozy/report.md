# Left wrist posture audit: run_QKIbSozy/model_999

The latest saved video shows left wrist curling. A fresh deterministic replay recorded measured joint angles, persistent IK references, actuator effort, and joint-limit margins through20seconds. No model weights or production controller settings changed. Small batch/render/backend differences can alter exact trajectories; all results below are from the instrumented replay, not recovered joint telemetry from the prior MP4.

## Held reach:7–20seconds

| Wrist joint | Left measured mean | Left IK mean | Right measured mean | Right IK mean |
|---|---:|---:|---:|---:|
| roll | -1.0 deg | -0.9 deg | -66.7 deg | -67.9 deg |
| pitch | 46.2 deg | 45.9 deg | -7.6 deg | -7.7 deg |
| yaw | -0.3 deg | -0.2 deg | -35.1 deg | -36.2 deg |

Left pitch spans28.3–58.8degrees during the hold. Its average reference nearly equals measured pitch, so the main bend is the commanded configuration rather than servo tracking lag. Pitch tracking RMSE is4.77degrees. Peak left pitch actuator effort is0.95Nm versus the model's5Nm rating; minimum soft-limit margin is24.5degrees. No evidence of left wrist effort saturation or a position-limit stop. The right wrist also twists substantially, despite a visually straighter pitch angle.

Both wrists start at zero joint angles, with matching pitch/yaw axis definitions and mirrored roll-link offsets. Position-only IK supplies no desired palm orientation or preferred wrist joint posture. Its persistent references continue changing during the hold as body motion and gravity loads produce Cartesian error. Redundant arm motion can therefore settle in different left/right wrist configurations while tracking similar hand positions. The exact mechanical contribution of each gait disturbance was not isolated, so this is not proof of a particular numerical null-space drift defect.

## Neutral-wrist reachability

A separate bounded kinematic solve fixed wrist roll/pitch/yaw at zero and solved only shoulder and elbow joints for the unchanged20cm-high hand targets. Both targets were solved to numerical precision with the existing0.02rad joint-limit margin. Thus wrist curl is not geometrically required for an upright robot. This is static reachability evidence, not a guarantee that fixing wrists during walking will preserve balance or all time-varying target reachability.

## Recommended next change

Add a soft neutral-wrist/posture objective or hand-orientation task to the online IK, preserving hand position as the primary objective. Apply it to both sides, with bounded correction rates and joint-limit checks. Test hand error, COM error, yaw and wrist posture on the current checkpoint before retraining. Avoid simply increasing the hand-position reward: it does not distinguish different wrist configurations reaching the same point. Hard-locking only the left wrist could shift the problem to another joint.

Artifacts:joint_trace.csv, summary.json, wrist_angles.png, neutral_wrist_reachability.json, events.json. No training was run.
