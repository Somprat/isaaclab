# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fine-tune transferred or previously trained online-hands weights with a fresh optimizer."""

import argparse
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
from play_g1_reference import load_playback_config

from isaaclab.app import launch_simulation
from isaaclab.utils import to_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_tasks.core.velocity.config.g1.reference_env_cfg import G1LipmOnlineHandsFlatEnvCfg
from isaaclab_tasks.core.velocity.mdp.hand_commands import HandPositionCommandCfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    initialization_mode = parser.add_mutually_exclusive_group()
    initialization_mode.add_argument(
        "--from-scratch",
        action="store_true",
        help="Reuse checkpoint configuration only; initialize actor, critic, and optimizer from scratch.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate settings without simulation or training.")
    initialization_mode.add_argument(
        "--from-finetuned",
        action="store_true",
        help="Load trained online-hands weights and hold their saved reach/IK stage.",
    )
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--ik-influence", type=float, default=1.0)
    parser.add_argument("--ik-ramp-steps", type=int, default=0)
    parser.add_argument("--arm-residual-limit", type=float, default=0.0)
    parser.add_argument("--arm-joint-error-weight", type=float, default=0.5)
    parser.add_argument("--hand-reward-weight", type=float, default=0.05)
    parser.add_argument("--reach-fraction", type=float, default=1.0)
    parser.add_argument("--initial-reach-fraction", type=float, default=0.1)
    parser.add_argument("--reach-curriculum-steps", type=int, default=10000)
    parser.add_argument("--reach-duration", type=float, default=6.0)
    parser.add_argument("--torso-upright-weight", type=float, default=1.0)
    parser.add_argument("--torso-tilt-tolerance-deg", type=float, default=8.0)
    parser.add_argument("--hand-target-frame", choices=("torso", "reset_heading", "world"), default="world")
    parser.add_argument(
        "--hand-forward-reduction", type=float, default=0.0,
        help="Reduce world X targets by this many meters using a linear initial-hold/reach ramp (0.05 matches closer replay).",
    )
    parser.add_argument(
        "--hand-orientation", action="store_true", help="Enable full-pose arm IK; default palms face inward."
    )
    parser.add_argument(
        "--left-hand-orientation",
        type=float,
        nargs=4,
        metavar=("X", "Y", "Z", "W"),
        help="Left hand-link quaternion relative to torso; requires --hand-orientation.",
    )
    parser.add_argument(
        "--right-hand-orientation",
        type=float,
        nargs=4,
        metavar=("X", "Y", "Z", "W"),
        help="Right hand-link quaternion relative to torso; requires --hand-orientation.",
    )
    parser.add_argument("--initial-hold", type=float, default=1.0)
    parser.add_argument(
        "--horizontal-hand-hold",
        action="store_true",
        help="Capture world orientation and, for world-frame targets, measured XY offsets at 7 s; full pose IK without accumulation.",
    )
    parser.add_argument(
        "--direct-arm-placement", action="store_true",
        help="Place online IK arm states directly while preserving body physics.",
    )
    parser.add_argument(
        "--world-identity-hand-orientation", action="store_true",
        help="Keep original position targets and fixed world-identity hand orientation throughout.",
    )
    parser.add_argument(
        "--immediate-arm-placement", action="store_true",
        help="Enable direct arm placement with immediate IK endpoints and zero arm joint velocities.",
    )
    args = parser.parse_args()
    args.direct_arm_placement = args.direct_arm_placement or args.immediate_arm_placement
    if not math.isfinite(args.hand_forward_reduction) or args.hand_forward_reduction < 0:
        parser.error("Hand forward reduction must be finite and nonnegative")
    if args.hand_forward_reduction and args.hand_target_frame != "world":
        parser.error("--hand-forward-reduction requires --hand-target-frame world")
    if args.world_identity_hand_orientation and args.horizontal_hand_hold:
        parser.error("--world-identity-hand-orientation cannot be combined with --horizontal-hand-hold")
    if args.world_identity_hand_orientation and (args.left_hand_orientation or args.right_hand_orientation):
        parser.error("World-identity mode uses identity quaternions for both hands")
    args.hand_orientation = (args.hand_orientation or args.horizontal_hand_hold
                             or args.direct_arm_placement or args.world_identity_hand_orientation)
    if not 0 <= args.initial_reach_fraction <= args.reach_fraction <= 1:
        parser.error("Require 0 <= initial reach fraction <= reach fraction <= 1")
    if args.reach_curriculum_steps < 0:
        parser.error("Reach curriculum steps must be nonnegative")
    if not math.isfinite(args.reach_duration) or args.reach_duration <= 0:
        parser.error("Reach duration must be finite and positive")
    if not math.isfinite(args.initial_hold) or args.initial_hold < 0:
        parser.error("Initial hold must be finite and nonnegative")
    for name in ("left_hand_orientation", "right_hand_orientation"):
        orientation = getattr(args, name)
        if orientation is not None:
            if not args.hand_orientation:
                parser.error("Custom hand quaternions require --hand-orientation")
            if not all(math.isfinite(v) for v in orientation) or math.hypot(*orientation) < 1.0e-8:
                parser.error("Hand orientations must be finite nonzero xyzw quaternions")
    initialization = {} if args.from_scratch else torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    provenance = initialization.get("infos") or {}
    if args.from_finetuned:
        for key in ("online_hands_control_steps", "online_hands_reach_fraction", "online_hands_ik_influence"):
            if key not in provenance:
                parser.error(f"Trained online-hands checkpoint is missing {key}")
        reach = provenance["online_hands_reach_fraction"]
        influence = provenance["online_hands_ik_influence"]
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in (reach, influence)):
            parser.error("Saved reach/IK stage must be finite and in [0, 1]")
        args.initial_reach_fraction = args.reach_fraction = reach
        args.reach_curriculum_steps = 0
        args.ik_influence = influence
        args.ik_ramp_steps = 0
    elif not args.from_scratch and (
        provenance.get("transfer") != "lipm_online_hands" or initialization.get("iter") != 0
    ):
        parser.error("Use --from-finetuned for a trained online-hands checkpoint")
    if int(os.environ.get("RANK", "0")) == 0 and args.output.exists():
        parser.error("Choose a fresh training output directory")
    if min(args.num_envs, args.max_iterations) < 1 or args.ik_ramp_steps < 0:
        parser.error("Counts must be positive; ramp steps can be zero")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Learning rate must be finite and positive")
    if not math.isfinite(args.ik_influence) or not 0 <= args.ik_influence <= 1:
        parser.error("IK influence must be in [0, 1]")
    if not math.isfinite(args.hand_reward_weight) or args.hand_reward_weight < 0:
        parser.error("Hand reward weight must be finite and nonnegative")
    for name in ("arm_residual_limit", "arm_joint_error_weight", "torso_upright_weight"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"{name} must be finite and nonnegative")
    if not math.isfinite(args.torso_tilt_tolerance_deg) or not 0 <= args.torso_tilt_tolerance_deg < 180:
        parser.error("Torso tilt tolerance must be finite and in [0, 180) degrees")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    device = f"cuda:{local_rank}"
    cfg, agent, _ = load_playback_config(args.checkpoint, preset="lipm_online_hands", device=device)
    if cfg.observations.policy.ik_joint_errors is None:
        parser.error(
            "Residual training requires a fresh transfer with IK reference observations from the walking policy"
        )
    cfg.scene.num_envs = args.num_envs
    cfg.actions.joint_pos.arm_residual_limit = args.arm_residual_limit
    cfg.rewards.online_arm_joint_error.weight = -args.arm_joint_error_weight
    cfg.actions.joint_pos.persistent_reference = True
    cfg.actions.joint_pos.track_orientation = args.hand_orientation
    cfg.rewards.joint_deviation_arms.weight = 0.0
    # Reapply current training objectives when loading older transferred configs.
    cfg.rewards.online_heading.weight = 2.0
    cfg.rewards.online_torso_heading.weight = 1.0
    cfg.rewards.online_torso_upright.weight = args.torso_upright_weight
    cfg.rewards.online_torso_upright.params["tolerance"] = math.radians(args.torso_tilt_tolerance_deg)
    cfg.rewards.online_torso_yaw_motion.weight = -0.05
    cfg.rewards.online_waist_yaw_motion.weight = -0.02
    cfg.seed = agent.seed = agent.seed + rank
    cfg.actions.joint_pos.ik_influence = args.ik_influence
    cfg.actions.joint_pos.ramp_steps = args.ik_ramp_steps
    cfg.actions.joint_pos.follow_reach_curriculum = False
    hand_cfg = cfg.commands.hand_position
    hand_cfg.target_frame = args.hand_target_frame
    hand_cfg.world_x_reach_reduction_m = args.hand_forward_reduction
    # Saved source configs contain the older, shorter reach; use current endpoints.
    hand_defaults = HandPositionCommandCfg()
    hand_cfg.left_orientation = tuple(args.left_hand_orientation or hand_defaults.left_orientation)
    hand_cfg.right_orientation = tuple(args.right_hand_orientation or hand_defaults.right_orientation)
    hand_cfg.left_reach_offset = hand_defaults.left_reach_offset
    hand_cfg.right_reach_offset = hand_defaults.right_reach_offset
    hand_cfg.initial_reach_fraction = args.initial_reach_fraction
    hand_cfg.reach_fraction = args.reach_fraction
    hand_cfg.curriculum_steps = args.reach_curriculum_steps
    hand_cfg.reach_duration_s = args.reach_duration
    hand_cfg.initial_hold_s = args.initial_hold
    hand_cfg.horizontal_hold = args.horizontal_hand_hold
    if args.world_identity_hand_orientation:
        hand_cfg.orientation_frame = "world"
        hand_cfg.left_orientation = hand_cfg.right_orientation = (0.0, 0.0, 0.0, 1.0)
    if args.horizontal_hand_hold or args.direct_arm_placement:
        cfg.actions.joint_pos.ik_only_arms = True
        cfg.actions.joint_pos.track_orientation = True
        # Replay parity: all seven arm joints, one measured-state DLS update per control step.
        cfg.actions.joint_pos.persistent_reference = False
        cfg.actions.joint_pos.ik_influence = 1.0
        cfg.actions.joint_pos.ramp_steps = 0
        cfg.actions.joint_pos.arm_residual_limit = 0.0
        cfg.actions.joint_pos.damping = 0.05
        cfg.actions.joint_pos.max_ik_step = 0.05
    if args.direct_arm_placement:
        cfg.actions.joint_pos.direct_arm_placement = True
        if args.immediate_arm_placement:
            cfg.actions.joint_pos.immediate_arm_placement = True
        substeps = cfg.sim.physics.num_substeps
        cfg.sim.dt /= substeps
        cfg.decimation *= substeps
        cfg.sim.render_interval *= substeps
        cfg.sim.physics.num_substeps = 1
        cfg.sim.physics.use_cuda_graph = False
        motors = cfg.scene.robot.actuators["motors"]
        for field in ("stiffness", "damping"):
            gains = getattr(motors, field).copy()
            for pattern in gains:
                if any(part in pattern for part in ("shoulder", "elbow", "wrist")):
                    gains[pattern] = 0.0
            setattr(motors, field, gains)
    cfg.rewards.online_left_hand_position.weight = args.hand_reward_weight
    cfg.rewards.online_right_hand_position.weight = args.hand_reward_weight
    # Add current penalty and velocity defaults even when the saved config predates them.
    reward_defaults = G1LipmOnlineHandsFlatEnvCfg().rewards
    cfg.rewards.online_left_hand_position_penalty = reward_defaults.online_left_hand_position_penalty
    cfg.rewards.online_right_hand_position_penalty = reward_defaults.online_right_hand_position_penalty
    cfg.rewards.online_left_hand_linear_velocity = reward_defaults.online_left_hand_linear_velocity
    cfg.rewards.online_right_hand_linear_velocity = reward_defaults.online_right_hand_linear_velocity
    for side in ("left", "right"):
        setattr(cfg.rewards, f"online_{side}_hand_orientation",
                getattr(reward_defaults, f"online_{side}_hand_orientation"))
        setattr(cfg.rewards, f"online_{side}_hand_orientation_penalty",
                getattr(reward_defaults, f"online_{side}_hand_orientation_penalty"))
    agent.algorithm.learning_rate = args.learning_rate
    agent.algorithm.schedule = "fixed"
    agent.max_iterations = args.max_iterations
    agent.init_at_random_ep_len = False
    agent.experiment_name = "g1_lipm_online_hands_finetune"
    agent.logger = "tensorboard"
    rank_output = args.output if rank == 0 else args.output / f"rank_{rank}"
    cfg.log_dir = str(rank_output)
    if args.dry_run:
        print("Initialization:", "random weights" if args.from_scratch else "checkpoint weights")
        print(f"Configuration: {args.checkpoint.parent / 'params'}")
        print(f"Output: {args.output}; iterations: {args.max_iterations}; envs/rank: {args.num_envs}")
        print(
            f"Hand frame: {hand_cfg.target_frame}; "
            f"reach: {hand_cfg.initial_reach_fraction} to {hand_cfg.reach_fraction}"
        )
        print(f"IK influence: {args.ik_influence}; arm residual: {args.arm_residual_limit}")
        print(
            "Hand position penalty weights (left/right): "
            f"{cfg.rewards.online_left_hand_position_penalty.weight}/"
            f"{cfg.rewards.online_right_hand_position_penalty.weight}"
        )
        print(
            f"Horizontal hand hold at 7 s: {hand_cfg.horizontal_hold}; "
            f"pose IK: {cfg.actions.joint_pos.track_orientation}; orientation frame: {hand_cfg.orientation_frame}; "
            f"persistent reference: {cfg.actions.joint_pos.persistent_reference}"
        )
        print(f"Reach: hold {args.initial_hold}s, move {args.reach_duration}s")
        print(f"World X reach reduction: {hand_cfg.world_x_reach_reduction_m} m (linear reach ramp)")
        for side in ("left", "right"):
            velocity_reward = getattr(cfg.rewards, f"online_{side}_hand_linear_velocity")
            print(
                f"{side.capitalize()} hand velocity: weight={velocity_reward.weight}; "
                f"std={velocity_reward.params['std']} m/s"
            )
        print(f"Torso upright weight: {args.torso_upright_weight}; free tilt: {args.torso_tilt_tolerance_deg} degrees")
        return
    with launch_simulation(cfg, {"headless": True, "distributed": distributed}):
        from rsl_rl.runners import OnPolicyRunner

        from isaaclab.envs import ManagerBasedRLEnv

        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

        env = ManagerBasedRLEnv(cfg=cfg)
        try:
            wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)

            class HandFineTuneRunner(OnPolicyRunner):
                def save(self, path, infos=None):
                    metadata = dict(infos or {})
                    metadata["online_hands_control_steps"] = env.common_step_counter
                    hands = env.command_manager.get_term("hand_position")
                    influence = env.action_manager.get_term("joint_pos").ik_influence
                    metadata["online_hands_reach_fraction"] = float(hands.episode_reach_fraction.max())
                    metadata["online_hands_scheduled_reach_fraction"] = hands.curriculum_fraction
                    metadata["online_hands_ik_influence"] = (
                        float(influence.max()) if torch.is_tensor(influence) else influence
                    )
                    metadata["from_scratch"] = args.from_scratch
                    metadata["from_finetuned"] = args.from_finetuned
                    metadata["configuration_checkpoint"] = str(args.checkpoint.resolve())
                    if not args.from_scratch:
                        metadata["source_checkpoint"] = provenance.get(
                            "source_checkpoint", str(args.checkpoint.resolve())
                        )
                        metadata["parent_checkpoint"] = str(args.checkpoint.resolve())
                        metadata["initialization_checkpoint"] = str(args.checkpoint.resolve())
                    super().save(path, metadata)

            runner = HandFineTuneRunner(wrapped, to_dict(agent), log_dir=str(rank_output), device=agent.device)
            # Fine-tuning always starts a new optimizer and iteration counter.
            if not args.from_scratch:
                runner.load(str(args.checkpoint), load_cfg={"actor": True, "critic": True}, map_location=device)
            dump_yaml(str(rank_output / "params/env.yaml"), cfg)
            dump_yaml(str(rank_output / "params/agent.yaml"), agent)
            runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=False)
        finally:
            env.close()
            if dist.is_initialized():
                dist.destroy_process_group()


if __name__ == "__main__":
    main()
