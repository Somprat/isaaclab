# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hand orientation inputs must use torso-relative xyzw poses without changing legacy IK."""

import math
from types import SimpleNamespace as NS

import torch

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

from isaaclab.utils.math import quat_apply, quat_mul

from isaaclab_tasks.core.velocity.mdp import hand_commands as module


def test_hand_orientation_frames_validation_and_pose_ik():
    yaw = torch.tensor([0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)])
    quats = yaw.repeat(2, 3, 1)
    data = NS(body_link_quat_w=NS(torch=quats), body_link_pos_w=NS(torch=torch.zeros(2, 3, 3)))
    robot = NS(body_names=["torso_link", "left_rubber_hand", "right_rubber_hand"], data=data)
    env = NS(num_envs=2, device="cpu", scene={"robot": robot})
    cmd = module.HandPositionCommand(module.HandPositionCommandCfg(left_orientation=(0.0, 0.0, 0.0, 2.0)), env)
    identity = torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(2, 2, 4)
    torch.testing.assert_close(cmd.orientations(), identity)
    torch.testing.assert_close(cmd.orientation_targets, identity)
    # A world-aligned left hand on a torso yawed +90 degrees reads -90 in torso coordinates.
    data.body_link_quat_w.torch[:, 1] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    expected = torch.tensor([0.0, 0.0, -math.sqrt(0.5), math.sqrt(0.5)]).expand(2, 4)
    torch.testing.assert_close(cmd.orientations()[:, 0], expected)
    for bad in ((0.0, 0.0, 0.0, 0.0), (float("nan"), 0.0, 0.0, 1.0)):
        try:
            module.HandPositionCommand(module.HandPositionCommandCfg(left_orientation=bad), env)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid quaternion accepted")
    # Fixed world targets survive torso rotation/translation and the seven-second boundary.
    world = module.HandPositionCommand(
        module.HandPositionCommandCfg(orientation_frame="world", target_frame="world"), env
    )
    env.episode_length_buf = torch.tensor([349, 351])
    env.step_dt = 0.02
    for rotation in (yaw, torch.tensor([math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)])):
        data.body_link_quat_w.torch[:, 0] = rotation
        data.body_link_pos_w.torch[:, 0] = torch.tensor([2.0, 3.0, 0.8])
        world.capture_horizontal_hold()
        world.positions()
        torso = rotation.expand(2, 2, 4)
        torch.testing.assert_close(quat_mul(torso, world.orientation_targets), identity)
        targets_world = quat_apply(torso, world.targets) + data.body_link_pos_w.torch[:, :1]
        expected_world = torch.tensor([
            [2.245238359, 3.151653753, 0.041230731],
            [2.245238359, 2.848356247, 0.041230731],
        ]).expand(2, -1, -1)
        torch.testing.assert_close(targets_world, expected_world)
        assert not world.horizontal_hold_captured.any()
    # Real pose DLS responds to a pure orientation error, while position-only DLS does not.
    jac = torch.zeros(1, 6, 7)
    jac[0, :, :6] = torch.eye(6)
    pos = torch.zeros(1, 3)
    q = torch.zeros(1, 7)
    cur = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    target = torch.tensor([[0.0, 0.0, math.sin(0.1), math.cos(0.1)]])
    pose = DifferentialIKController(
        DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"), 1, "cpu"
    )
    pose.set_command(torch.cat((pos, target), dim=-1), ee_quat=cur)
    solved = pose.compute(pos, cur, jac, q)
    assert solved[0, 5] > 0.19
    assert torch.count_nonzero(solved[:, :5]) == 0
    position = DifferentialIKController(
        DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls"), 1, "cpu"
    )
    position.set_command(pos, ee_quat=cur)
    torch.testing.assert_close(position.compute(pos, cur, jac, q), q)


def test_orientation_reward_and_penalty_use_angular_error():
    target = torch.tensor([0.0, 0.0, 0.0, 1.0])
    yaw = torch.tensor([0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)])
    command = NS(
        _refresh_targets=lambda: None,
        orientations=lambda: torch.stack((yaw, target)).view(1, 2, 4),
        orientation_targets=target.repeat(1, 2, 1),
    )
    env = NS(command_manager=NS(get_term=lambda _: command))
    error = module.hand_orientation_penalty(env, "left")
    torch.testing.assert_close(error, torch.tensor([math.pi / 2]))
    torch.testing.assert_close(module.hand_orientation_penalty(env, "right"), torch.zeros(1))
    torch.testing.assert_close(
        module.hand_orientation_tracking(env, "left", std=0.5),
        torch.exp(-error / 0.5**2),
    )
    torch.testing.assert_close(module.hand_orientation_tracking(env, "right", std=0.5), torch.ones(1))
