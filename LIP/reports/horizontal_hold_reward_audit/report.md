# Horizontal-hold fine-tuning reward audit

Run: run_20261115_120037, checkpoint model_499.pt.

Actual TensorBoard Episode_Reward contributions averaged over iterations 450–499. These already include reward weights; episode integrals are divided by the configured maximum episode duration (20 seconds). Late training mean episode length is 1000 steps. Early-window numbers in the CSV also reflect shorter episodes and must not be interpreted as pure tracking changes.

- Net contribution across terms: 14.99089.
- Both hand velocity terms: 4.93997 (maximum 5), 33.0% of net contribution.
- Both positive hand position terms: 0.03810 (maximum 0.1).
- Both squared position penalties: -0.22542.
- IK joint-reference error penalty: -0.00788.

Velocity reward is exp(-||v_target-v_hand||^2 / 1 m²/s²) with weight 2.5 per hand. It does not require position accuracy. It can approach its maximum while an incorrectly positioned hand follows body translation. Large reward magnitude alone does not prove it caused learning failure, but the measured near-maximum score confirms it does not diagnose this failed reach.

The position penalty has weight -2 per hand, and position exponential weight 0.05 with std 0.2 m. In the recorded inference, at 6.98 s left/right forward hand coordinates are 0.142/0.135 m while targets are 0.428/0.436 m. The hold then captures those failed measured offsets and replaces the original horizontal reach goal. Thus later position rewards no longer enforce the intended full reach.

Direct policy arm residuals are zero. The policy controls legs/waist, with only indirect influence on arm tracking through body motion. Larger hand rewards alone cannot provide a missing direct arm control channel.

The non-accumulating IK target is current measured joint angle plus a correction clipped to ±0.05 rad. Shoulder/elbow stiffness is 40 N m/rad, so its proportional component at the initial measured state is capped near 2 N m (not total torque: damping, substep motion and other dynamics also matter). This is a plausible actuator tracking limitation requiring a physics ablation. The kinematic replay placed joints exactly and could not reveal it.

The joint-reference error penalty checks the nearby IK joint command, not distance to the hand goal. It can stay small while Cartesian tracking is poor. It is not a direct action-magnitude reward. The 'realized_reach_fraction' metric measures the scheduled trajectory, not actual hand reach, so its value of 1 is not evidence of task success.

Recommended order: validate physical arm tracking first; retain the intended horizontal reach goal or gate measured-pose capture on reach accuracy; then reduce/gate velocity reward on position accuracy and reevaluate hand-position terms. Validate changes in a short physics rollout before more training. No rewards or checkpoint weights were modified by this audit.
