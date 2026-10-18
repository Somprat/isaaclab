# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL playback backend of the unified reinforcement learning entrypoint."""

from __future__ import annotations

import argparse
import contextlib
import csv
import logging
import math
import os
from datetime import datetime

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.utils import to_dict
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.seed import configure_seed
from isaaclab.utils.string import list_intersection

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, resolve_task_config, setup_preset_cli

from ...rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from ...utils.wandb import is_wandb_checkpoint, resolve_wandb_checkpoint
from ..common import (
    CHECKPOINT_SELECTORS,
    add_common_play_args,
    apply_env_overrides,
    apply_video_recording,
    close_env,
    create_isaaclab_env,
    enable_cameras_for_video,
    normalize_task_name,
    pre_launch_video_config,
    resolve_checkpoint_selector,
    resolve_published_checkpoint,
    run_playback,
    set_hydra_args,
    show_run_summary,
    startup_screen,
)
from . import cli_args_rsl_rl as cli_args

logger = logging.getLogger(__name__)

# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse RSL-RL playback arguments."""
    parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RSL-RL.")
    add_common_play_args(
        parser,
        agent_default="rsl_rl_cfg_entry_point",
        agent_help="Name of the RL agent configuration entry point.",
    )
    parser.add_argument(
        "--external_callback", default=None, help="Fully qualified path to an externally defined callback."
    )
    cli_args.add_rsl_rl_args(parser)
    add_launcher_args(parser)
    remaining_args_env_registration = cli_args.register_external_tasks(argv)
    args_cli, remaining_args = setup_preset_cli(parser, argv)
    enable_cameras_for_video(args_cli)
    set_hydra_args(list_intersection(remaining_args, remaining_args_env_registration))
    return args_cli


def _resolve_checkpoint(
    args_cli: argparse.Namespace, agent_cfg: RslRlBaseRunnerCfg, env_cfg: object, log_root_path: str
) -> str | None:
    """Resolve the checkpoint to play, or None when no published checkpoint exists."""
    if args_cli.checkpoint and is_wandb_checkpoint(args_cli.checkpoint):
        return resolve_wandb_checkpoint(args_cli.checkpoint)
    if args_cli.checkpoint == "pretrained":
        return resolve_published_checkpoint("rsl_rl", args_cli.task, env_cfg)
    if args_cli.checkpoint in CHECKPOINT_SELECTORS:
        return resolve_checkpoint_selector(
            log_root_path,
            args_cli.checkpoint,
            library="rsl_rl",
            task=normalize_task_name(args_cli.task),
            checkpoint_pattern=r"model_.*\.pt",
            metadata={"agent": args_cli.agent},
        )
    if args_cli.checkpoint and os.path.isdir(args_cli.checkpoint):
        return get_checkpoint_path(
            os.path.dirname(args_cli.checkpoint), os.path.basename(args_cli.checkpoint), agent_cfg.load_checkpoint
        )
    if args_cli.checkpoint:
        return retrieve_file_path(args_cli.checkpoint)
    return get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)


def run(argv: list[str]) -> None:
    """Play a checkpoint of an RSL-RL agent."""
    args_cli = _parse_args(argv)
    with startup_screen(args_cli, num_stages=3) as screen:
        env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, play_mode=not args_cli.train_env_cfg)
        pre_launch_video_config(env_cfg, args_cli)
        screen.stage("Launching simulation")
        with launch_simulation(env_cfg, args_cli), contextlib.ExitStack() as cleanup:
            show_run_summary(screen, args_cli, env_cfg, library="rsl_rl", action="play")
            agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
            apply_env_overrides(args_cli, env_cfg)
            # certain randomizations occur in the environment initialization so we set the seed here
            env_cfg.seed = agent_cfg.seed

            log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
            logger.info(f"Loading experiment from directory: {log_root_path}")
            resume_path = _resolve_checkpoint(args_cli, agent_cfg, env_cfg, log_root_path)
            if resume_path is None:
                return
            log_dir = os.path.dirname(resume_path)
            env_cfg.log_dir = log_dir
            playback_duration = 20.0
            playback_steps = math.ceil(playback_duration / (env_cfg.sim.dt * env_cfg.decimation))
            args_cli.video_length = playback_steps
            apply_video_recording(env_cfg, log_dir, args_cli, subdir="play", checkpoint_path=resume_path)
            for recorder_cfg in env_cfg.video_recorders or []:
                recorder_cfg.step_offset = 0

            screen.stage("Creating environment")
            env = create_isaaclab_env(
                args_cli.task,
                env_cfg,
                args_cli,
                convert_marl_to_single_agent=isinstance(env_cfg, DirectMARLEnvCfg),
            )
            cleanup.callback(lambda: close_env(env))

            screen.stage("Loading policy")
            env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
            logger.info(f"Loading model checkpoint from: {resume_path}")
            if agent_cfg.class_name == "OnPolicyRunner":
                runner = OnPolicyRunner(env, to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
            elif agent_cfg.class_name == "DistillationRunner":
                runner = DistillationRunner(env, to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
            else:
                raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
            # configure_seed must run after runner construction so torch determinism does not disturb its initialization
            if args_cli.deterministic:
                configure_seed(env_cfg.seed, torch_deterministic=True)
            runner.load(resume_path)
            policy = runner.get_inference_policy(device=env.unwrapped.device)

            export_model_dir = os.path.join(log_dir, "exported")
            runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
            runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")

            command = env.unwrapped.command_manager.get_term("base_velocity")
            obs = env.get_observations()
            step_count = 0
            tracking_dir = os.path.join(log_dir, "velocity_tracking")
            os.makedirs(tracking_dir, exist_ok=True)
            tracking_path = os.path.join(tracking_dir, f"velocity_{datetime.now():%Y%m%d_%H%M%S_%f}.csv")
            tracking_file = cleanup.enter_context(open(tracking_path, "w", newline=""))
            tracking_writer = csv.writer(tracking_file)
            tracking_writer.writerow(
                ["step", "time_s", "segment", "env_id", "command_vx_m_s", "actual_vx_m_s", "error_m_s", "reset"]
            )
            targets = (0.2, 0.5, 1.0)
            error_sums = [0.0] * len(targets)
            sample_counts = [0] * len(targets)
            reset_counts = [0] * len(targets)
            logger.info("Recording velocity tracking to: %s", tracking_path)

            # Translate with robot 0 while keeping the viewing direction fixed in world space.
            camera_eye_offset = (4.0, -4.0, 2.5)
            camera_target_offset = (0.0, 0.0, 0.3)

            def step() -> None:
                nonlocal obs, step_count
                if step_count >= playback_steps:
                    raise StopIteration
                if args_cli.video or args_cli.visualizer:
                    robot_position = env.unwrapped.scene["robot"].data.root_link_pos_w.torch[0].cpu().tolist()
                    env.unwrapped.sim.set_camera_view(
                        eye=tuple(position + offset for position, offset in zip(robot_position, camera_eye_offset)),
                        target=tuple(
                            position + offset for position, offset in zip(robot_position, camera_target_offset)
                        ),
                    )
                segment = min(step_count * len(targets) // playback_steps, len(targets) - 1)
                speed = targets[segment]
                # Set the forward target and keep lateral and angular targets at zero.
                command.vel_command_b[:, 0] = speed
                command.vel_command_b[:, 1:] = 0.0
                # Refresh the policy input without advancing observation history.
                env.unwrapped.obs_buf = env.unwrapped.observation_manager.compute(update_history=False)
                obs = env.get_observations()
                obs, _, dones, _ = env.step(policy(obs))
                # Read each robot's actual forward velocity in its body frame (m/s).
                actual_forward_velocity = env.unwrapped.scene["robot"].data.root_lin_vel_b.torch[:, 0]
                for env_id, (actual, reset) in enumerate(
                    zip(actual_forward_velocity.cpu().tolist(), dones.cpu().tolist())
                ):
                    error = actual - speed
                    tracking_writer.writerow(
                        [
                            step_count + 1,
                            (step_count + 1) * env.unwrapped.step_dt,
                            segment + 1,
                            env_id,
                            speed,
                            actual,
                            error,
                            bool(reset),
                        ]
                    )
                    reset_counts[segment] += int(reset)
                    if not reset:
                        error_sums[segment] += abs(error)
                        sample_counts[segment] += 1
                # reset recurrent states for episodes that have terminated
                policy.reset(dones)
                step_count += 1

            screen.close()

            with contextlib.suppress(StopIteration):
                run_playback(step, dt=env.unwrapped.step_dt, args_cli=args_cli, env_cfg=env_cfg)
            tracking_file.flush()
            for segment, target in enumerate(targets):
                if sample_counts[segment]:
                    logger.info(
                        "Velocity %.2f m/s: mean absolute error %.4f m/s (%d samples, %d resets).",
                        target,
                        error_sums[segment] / sample_counts[segment],
                        sample_counts[segment],
                        reset_counts[segment],
                    )
