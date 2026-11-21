# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Prepare or fine-tune the published G1 ONNX actor using MPC reference rewards.

Default: import weights, verify deployment/ONNX parity, save initialization, exit.
Pass --train explicitly to run PPO. Neither the original ONNX nor baseline configs
are modified. The critic/optimizer are new; the entire actor remains trainable.
"""

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import torch
import warp as wp
import yaml

wp.config.enable_backward = False

from g1_onnx_actor import check_onnx_parity, import_onnx_actor

from isaaclab.app import launch_simulation
from isaaclab.paths import ISAACLAB_ROOT
from isaaclab.utils import to_dict
from isaaclab.utils.io import dump_yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.core.velocity.config.g1.onnx_finetune import configure_onnx_finetuning
from isaaclab_tasks.utils import resolve_task_config


def deployment_observation(env, names, deployment, histories, reset_mask=None):
    """Independent reconstruction of the published deployment history layout."""
    robot = env.scene["robot"]
    ids = [robot.joint_names.index(name) for name in names]
    default = torch.tensor(deployment["default_joint_pos"], device=env.device)
    values = {
        "base_ang_vel": robot.data.root_ang_vel_b.torch,
        "projected_gravity": robot.data.projected_gravity_b.torch,
        "velocity_commands": env.command_manager.get_command("base_velocity"),
        "joint_pos_rel": robot.data.joint_pos.torch[:, ids] - default,
        "joint_vel_rel": robot.data.joint_vel.torch[:, ids],
        "last_action": env.action_manager.action,
    }
    for name, spec in deployment["observations"].items():
        value = values[name] * torch.tensor(spec["scale"], device=env.device)
        if spec.get("clip") is not None:
            value = value.clamp(*spec["clip"])
        if name not in histories:
            histories[name] = value[:, None].repeat(1, spec["history_length"], 1)
        else:
            histories[name] = torch.cat([histories[name][:, 1:], value[:, None]], dim=1)
            if reset_mask is not None:
                histories[name][reset_mask] = value[reset_mask, None].repeat(1, spec["history_length"], 1)
    return torch.cat([histories[name].flatten(1) for name in deployment["observations"]], dim=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    assets = Path(ISAACLAB_ROOT) / "checkpoints/g1-29dof"
    parser.add_argument("--onnx", type=Path, default=assets / "policy.onnx")
    parser.add_argument("--deployment", type=Path, default=assets / "deploy.yaml")
    parser.add_argument(
        "--motion",
        type=Path,
        default=Path(ISAACLAB_ROOT).parent
        / "wb_humanoid_mpc/generated_motions/conditioned_50_fixed/conditioned_mpc_reference_0000.npz",
    )
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--initial-std", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-steps", type=int, default=10)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--step-length-observation", action="store_true", help="Append applied alternating-step length in meters."
    )
    parser.add_argument("--train", action="store_true", help="Run PPO after preparing and validating imported weights.")
    args = parser.parse_args()
    if min(args.num_envs, args.max_iterations, args.validation_steps) < 1:
        parser.error("Environment, iteration and validation step counts must be positive.")
    if not all(math.isfinite(v) and v > 0 for v in (args.learning_rate, args.initial_std)):
        parser.error("Learning rate and exploration std must be finite and positive.")
    for path in (args.onnx, args.deployment, args.motion):
        if not path.exists():
            parser.error(f"Missing input: {path}")
    output = args.output or Path(ISAACLAB_ROOT) / "logs/rsl_rl/g1_onnx_mpc_finetune" / datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )
    output = output.resolve()
    if output.exists():
        parser.error("Output directory already exists; choose a fresh path.")
    deployment = yaml.safe_load(args.deployment.read_text())
    if args.step_length_observation:
        deployment["mpc_step_length_observation"] = True
    step_length_enabled = bool(deployment.get("mpc_step_length_observation", False))
    cfg, agent = resolve_task_config(
        "Isaac-Velocity-Flat-G1", "rsl_rl_cfg_entry_point", overrides=["presets=reference"]
    )
    names = configure_onnx_finetuning(cfg, agent, deployment)
    cfg.scene.num_envs = args.num_envs
    cfg.commands.base_velocity.motion_directory = str(args.motion.resolve())
    cfg.commands.base_velocity.random_start = args.train
    cfg.seed = agent.seed = args.seed
    agent.experiment_name = "g1_onnx_mpc_finetune"
    agent.max_iterations = args.max_iterations
    agent.init_at_random_ep_len = False
    agent.algorithm.learning_rate = args.learning_rate
    agent.actor.distribution_cfg.init_std = args.initial_std
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    output.mkdir(parents=True)
    dump_yaml(str(output / "params/env.yaml"), cfg)
    dump_yaml(str(output / "params/agent.yaml"), agent)
    dump_yaml(str(output / "params/deployment.yaml"), deployment)
    with launch_simulation(cfg, {"headless": True}):
        from rsl_rl.runners import OnPolicyRunner

        from isaaclab.envs import ManagerBasedRLEnv

        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

        env = ManagerBasedRLEnv(cfg=cfg)
        try:
            wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
            runner = OnPolicyRunner(wrapped, to_dict(agent), log_dir=str(output), device=agent.device)
            imported = import_onnx_actor(runner.alg.actor, args.onnx)
            observations = wrapped.get_observations()
            expected_width = 546 + int(step_length_enabled)
            if observations["policy"].shape[1] != expected_width or wrapped.num_actions != 29:
                raise RuntimeError(f"Expected {expected_width} policy observations and 29 motor actions.")
            parity_error = check_onnx_parity(runner.alg.actor, observations, args.onnx)
            # Verify numerical invariance even when all appended channels change.
            changed = observations.clone()
            changed["policy"][:, 480:] = torch.randn_like(changed["policy"][:, 480:])
            parity_error = max(parity_error, check_onnx_parity(runner.alg.actor, changed, args.onnx))
            # Test a broad range of deployment histories independently of this reset.
            synthetic = observations.clone()
            synthetic["policy"] = torch.randn_like(synthetic["policy"])
            parity_error = max(parity_error, check_onnx_parity(runner.alg.actor, synthetic, args.onnx))
            histories = {}
            # For train mode, keep initialization exactly at the training reset;
            # use one observation comparison and let PPO collect the first rollout.
            torch.testing.assert_close(
                observations["policy"][:, :480],
                deployment_observation(env, names, deployment, histories),
                atol=1e-5,
                rtol=1e-5,
            )
            state = runner.alg.save()
            state.update(iter=0, infos={"source_onnx": str(args.onnx.resolve()), "actor_imported": True})
            torch.save(state, output / "model_initialized.pt")
            # Validation-only: deterministic inference, never an optimizer step.
            if not args.train:
                action_spec = deployment["actions"]["JointPositionAction"]
                scale = torch.tensor(action_spec["scale"], device=env.device)
                offset = torch.tensor(action_spec["offset"], device=env.device)
                with torch.inference_mode():
                    for _ in range(args.validation_steps):
                        action = runner.alg.actor(observations)
                        observations, rewards, done, _ = wrapped.step(action)
                        expected = deployment_observation(env, names, deployment, histories, reset_mask=done.bool())
                        torch.testing.assert_close(observations["policy"][:, :480], expected, atol=1e-5, rtol=1e-5)
                        if not done.any():
                            torch.testing.assert_close(
                                env.action_manager.get_term("joint_pos").processed_actions, action * scale + offset
                            )
                        if not torch.isfinite(observations["policy"]).all() or not torch.isfinite(rewards).all():
                            raise RuntimeError("Non-finite validation observations/rewards.")
            metadata = {
                "source_onnx": str(args.onnx.resolve()),
                "source_sha256": hashlib.sha256(args.onnx.read_bytes()).hexdigest(),
                "deployment": str(args.deployment.resolve()),
                "motion": str(args.motion.resolve()),
                "actor_initialization": "all ONNX weights copied; appended input columns zero",
                "critic_and_optimizer": "fresh; not contained in ONNX",
                "all_actor_layers_trainable": all(p.requires_grad for p in runner.alg.actor.parameters()),
                "policy_order": names,
                "onnx_max_action_error": parity_error,
                "deployment_history_check_passed": True,
                "deployment_action_processing_checked": not args.train,
                "training_requested": args.train,
                "step_length_observation": step_length_enabled,
                "validation_steps": 0 if args.train else args.validation_steps,
                **imported,
            }
            (output / "initialization.json").write_text(json.dumps(metadata, indent=2) + "\n")
            print(json.dumps(metadata, indent=2), flush=True)
            print(f"Output directory: {output}", flush=True)
            if args.train:
                runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=False)
            else:
                print("Preparation/validation complete. No PPO updates were run.", flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
