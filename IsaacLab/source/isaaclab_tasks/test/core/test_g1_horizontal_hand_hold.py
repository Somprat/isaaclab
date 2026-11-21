# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""World-frame holds capture XY and orientation while preserving Z and independent episode lifecycles."""

import math
from types import SimpleNamespace as NS

import torch

from isaaclab.managers import CommandManager
from isaaclab.utils.math import quat_apply, quat_mul

from isaaclab_tasks.core.velocity.mdp.hand_commands import HandPositionCommandCfg


def test_hold():
    pos = torch.tensor([[[0.0, 0.0, 1.0], [0.4, 0.2, 1.2], [0.4, -0.2, 1.2]]]).repeat(2, 1, 1)
    quat = torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(2, 3, 1)
    data = NS(
        body_link_pos_w=NS(torch=pos),
        body_link_quat_w=NS(torch=quat),
        body_link_lin_vel_w=NS(torch=torch.zeros(2, 3, 3)),
        body_link_ang_vel_w=NS(torch=torch.zeros(2, 3, 3)),
    )
    env = NS(
        num_envs=2,
        device="cpu",
        scene={"robot": NS(body_names=["torso_link", "left_rubber_hand", "right_rubber_hand"], data=data)},
        sim=NS(is_playing=lambda: True),
        episode_length_buf=torch.tensor([349, 350]),
        step_dt=0.02,
        common_step_counter=0,
    )
    cfg = HandPositionCommandCfg(
        horizontal_hold=True,
        target_frame="world",
        left_position=(0.55, 0.25, 0.2),
        right_position=(0.6, -0.25, 0.2),
    )
    manager = CommandManager(NS(hand_position=cfg), env)
    cmd = manager.get_term("hand_position")
    manager.reset()
    cmd.positions()
    expected_offsets = torch.tensor([[0.55, 0.25, 0.2], [0.6, -0.25, 0.2]])
    torch.testing.assert_close(cmd.targets, expected_offsets.repeat(2, 1, 1))
    cmd.capture_horizontal_hold()
    cmd.positions()
    assert cmd.horizontal_hold_captured.tolist() == [False, True]
    torch.testing.assert_close(cmd.targets[0], expected_offsets)
    torch.testing.assert_close(cmd.targets[1], torch.tensor([[0.4, 0.2, 0.2], [0.4, -0.2, 0.2]]))
    old_z = quat_apply(quat[:, 0, None].expand(-1, 2, -1), cmd.targets)[1, :, 2] + pos[1, 0, 2]
    pos[1, 0] += torch.tensor([2.0, 3.0, 0.1])
    quat[1, 0] = torch.tensor([0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)])
    pos[1, 1:] += 10  # captured offsets must not follow subsequent measured hand motion
    cmd.capture_horizontal_hold()
    cmd.positions()
    torso = quat[:, 0, None].expand(-1, 2, -1)
    world = quat_apply(torso, cmd.targets) + pos[:, 0, None]
    torch.testing.assert_close(world[1, :, :2], torch.tensor([[2.4, 3.2], [2.4, 2.8]]))
    torch.testing.assert_close(world[1, :, 2], old_z)
    torch.testing.assert_close(
        quat_mul(torso, cmd.orientation_targets)[1], torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(2, 1)
    )
    cmd.trajectory_velocities[:, :, :2] = 0.3
    data.body_link_lin_vel_w.torch[1, 0] = torch.tensor([0.7, 0.8, 0.9])
    torch.testing.assert_close(cmd.target_velocities_world()[1, :, :2], torch.tensor([[0.7, 0.8], [0.7, 0.8]]))
    torch.testing.assert_close(cmd.target_velocities_world()[0, :, :2], torch.full((2, 2), 0.3))
    # Partial reset at an old clock must clear only that environment and not capture in reset.
    manager.reset(torch.tensor([1]))
    assert cmd.horizontal_hold_captured.tolist() == [False, False]
    env.episode_length_buf[1] = 0
    cmd.capture_horizontal_hold()
    assert not cmd.horizontal_hold_captured.any()
    env.episode_length_buf[0] = 350
    cmd.capture_horizontal_hold()
    assert cmd.horizontal_hold_captured.tolist() == [True, False]
    manager.reset(torch.tensor([1]))
    assert cmd.horizontal_hold_captured.tolist() == [True, False]
    env.episode_length_buf[1] = 350
    cmd.capture_horizontal_hold()
    assert cmd.horizontal_hold_captured.all()
    manager.reset()
    assert not cmd.horizontal_hold_captured.any()
