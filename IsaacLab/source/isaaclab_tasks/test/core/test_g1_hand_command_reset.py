# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hand reaches must restart after environment resets without confusing command resampling."""

import math
from types import SimpleNamespace

import torch

from isaaclab.managers import CommandManager

from isaaclab_tasks.core.velocity.mdp.hand_commands import HandPositionCommandCfg


def _make_env():
    orientation = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1)
    body_orientation = orientation[:, None].repeat(1, 3, 1)
    robot = SimpleNamespace(
        body_names=["torso_link", "left_rubber_hand", "right_rubber_hand"],
        data=SimpleNamespace(
            root_quat_w=SimpleNamespace(torch=orientation),
            body_link_quat_w=SimpleNamespace(torch=body_orientation),
        ),
    )
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        sim=SimpleNamespace(is_playing=lambda: True),
        scene={"robot": robot},
        common_step_counter=0,
        episode_length_buf=torch.zeros(2, dtype=torch.long),
        step_dt=0.02,
    )
    return env


def test_hand_reach_restarts_after_timeout_and_preserves_periodic_resampling():
    """Exercise the real command-manager lifecycle while the old episode clock is still present."""
    env = _make_env()
    orientation = env.scene["robot"].data.root_quat_w.torch
    body_orientation = env.scene["robot"].data.body_link_quat_w.torch
    cfg = HandPositionCommandCfg(
        target_frame="reset_heading",
        left_position=(0.2, 0.1, 0.0),
        right_position=(0.2, -0.1, 0.0),
        left_reach_offset=(0.2, 0.1, 0.1),
        right_reach_offset=(0.2, -0.1, 0.1),
        initial_reach_fraction=0.1,
        reach_fraction=1.0,
        curriculum_steps=10000,
        reach_duration_s=6.0,
    )
    manager = CommandManager(SimpleNamespace(hand_position=cfg), env)
    command = manager.get_term("hand_position")
    manager.reset()
    env.episode_length_buf.fill_(350)
    manager.compute(env.step_dt)
    first_goal = torch.tensor([[0.22, 0.11, 0.01], [0.22, -0.11, 0.01]])
    torch.testing.assert_close(command.command.view(2, 2, 3), first_goal.expand(2, -1, -1))

    # ManagerBasedRLEnv resets commands before clearing the previous episode clock.
    env.common_step_counter = 10000
    env.episode_length_buf[:] = torch.tensor([1000, 350])
    orientation[0] = torch.tensor([0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)])
    body_orientation[0] = orientation[0]
    manager.reset(torch.tensor([0]))
    nominal = torch.tensor([[0.2, 0.1, 0.0], [0.2, -0.1, 0.0]])
    torch.testing.assert_close(command.command[0].view(2, 3), nominal)
    torch.testing.assert_close(command.command[1].view(2, 3), first_goal)
    env.episode_length_buf[0] = 0
    manager.compute(env.step_dt)
    torch.testing.assert_close(command.command[0].view(2, 3), nominal)
    env.episode_length_buf[0] = 350
    manager.compute(env.step_dt)
    full_goal = torch.tensor([[0.4, 0.2, 0.1], [0.4, -0.2, 0.1]])
    torch.testing.assert_close(command.command[0].view(2, 3), full_goal)
    torch.testing.assert_close(command.command[1].view(2, 3), first_goal)

    # Timer-driven resampling during an episode starts at the current target, not nominal.
    command.cfg.reach_fraction = 0.5
    command.cfg.curriculum_steps = 0
    env.episode_length_buf[0] = 400
    command.time_left[0] = 0.0
    manager.compute(env.step_dt)
    torch.testing.assert_close(command.command[0].view(2, 3), full_goal)
    env.episode_length_buf[0] = 600  # 1 s hold + halfway through the 6 s reach.
    manager.compute(env.step_dt)
    halfway = torch.tensor([[0.35, 0.175, 0.075], [0.35, -0.175, 0.075]])
    torch.testing.assert_close(command.command[0].view(2, 3), halfway)
    torch.testing.assert_close(command.command[1].view(2, 3), first_goal)


def test_world_offsets_follow_translation_but_ignore_initial_heading_and_torso_rotation():
    """World-axis goals stay relative to torso position through yaw and tilt."""
    env = _make_env()
    data = env.scene["robot"].data
    data.root_quat_w.torch[:] = torch.tensor([0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)])
    data.body_link_quat_w.torch[:, 0] = data.root_quat_w.torch
    data.body_link_pos_w = SimpleNamespace(torch=torch.zeros(2, 3, 3))
    cfg = HandPositionCommandCfg(target_frame="world", left_position=(0.2, 0.1, 0.0), right_position=(0.2, -0.1, 0.0))
    manager = CommandManager(SimpleNamespace(hand_position=cfg), env)
    command = manager.get_term("hand_position")
    manager.reset()
    expected_yaw = torch.tensor([[0.1, -0.2, 0.0], [-0.1, -0.2, 0.0]])
    torch.testing.assert_close(command.command.view(2, 2, 3), expected_yaw.expand(2, -1, -1))

    # Roll the torso 90 degrees and translate it; hands at the world offsets have zero error.
    data.body_link_quat_w.torch[:, 0] = torch.tensor([math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)])
    data.body_link_pos_w.torch[:, 0] = torch.tensor([[3.0, -2.0, 1.0], [-1.0, 4.0, 0.5]])
    data.body_link_pos_w.torch[:, 1:] = data.body_link_pos_w.torch[:, :1] + torch.tensor(
        [[0.2, 0.1, 0.0], [0.2, -0.1, 0.0]]
    )
    manager.compute(env.step_dt)
    expected_roll = torch.tensor([[0.2, 0.0, -0.1], [0.2, 0.0, 0.1]])
    torch.testing.assert_close(command.command.view(2, 2, 3), expected_roll.expand(2, -1, -1))
    torch.testing.assert_close(command.positions(), command.targets)
