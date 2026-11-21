# G1 first-step inspection and training readiness

The selected NPZ already exists; no manual regeneration is needed.

The first scheduled support is left-only at t=0. There are no double-support
samples, although the reset places both feet on the plane. The current swing
interpolation immediately requests horizontal velocity and stops it abruptly at
landing; vertical clearance is smooth. A finite-difference estimate of the first
right-foot velocity is [0.4002822124965791, -0.8207485070404141, 0.06117935463105803] m/s.

The initial CoM is between the feet, approximately 0.118 m to the right of the
left support. With zero initial velocity and a left-foot LIPM pivot, the initial
lateral acceleration is -1.626 m/s²,
away from the left foot. At 0.4 s, lateral CoM velocity reaches
-0.916 m/s. The capture-style planner responds with a first landing
0.565 m laterally from the left foot, versus the requested nominal width of
0.237 m. This is a startup transient, not a hard step-width constraint violation.
Dynamic reachability was not established by this inspection.

Before a long tracking run, add a physically consistent two-foot support/load
transfer startup and smooth horizontal swing endpoint velocities. A mere pause
before the existing first step would leave the same CoM-state mismatch when
single support begins. The current loader requires exactly one support foot;
implementing double support needs coordinated exporter, loader and reward updates.

## PPO readiness check

A bounded run completed 20 PPO iterations with 64 environments (30,720 simulated
transitions). Checkpoint: `/home/choon/isaaclab/IsaacLab/logs/rsl_rl/g1_lipm_readiness/2026-10-29_00-14-18_grounded_startup_20iters/model_19.pt`. Checkpoint tensors are finite.
The final logged mean episode length was 28.92 steps (~0.58 seconds); this is an
initial learning result, not a converged gait. Falls/tilt still end episodes.

The run explicitly sets `agent.init_at_random_ep_len=false`. The default RSL-RL
setting otherwise randomizes `episode_length_buf` before learning, while this
command uses that buffer as its reference clock. That would jump the standing
robot ahead in the clip at startup. Keep the override for LIPM training.
