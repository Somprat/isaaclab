# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Torso upright tracking must detect waist lean independently of pelvis and heading."""

import math
from types import SimpleNamespace

import torch

from isaaclab.utils.math import quat_from_euler_xyz

from isaaclab_tasks.core.velocity.mdp.lipm_mdp import torso_upright_tracking


def test_torso_upright_reward_allows_small_lean_and_ignores_yaw():
    # Upright pelvis in every case: yaw-only, small pitch, both pitch signs, roll,
    # and the 33-degree backward torso lean seen in the scratch-policy replay.
    angles = torch.deg2rad(torch.tensor([[0, 0, 90], [0, 5, 0], [0, 20, 0], [0, -20, 0], [20, 0, 0], [0, -33, 0]]))
    torso = quat_from_euler_xyz(angles[:, 0], angles[:, 1], angles[:, 2])
    pelvis = torch.zeros_like(torso)
    pelvis[:, 3] = 1.0
    robot = SimpleNamespace(
        body_names=["pelvis", "torso_link"],
        data=SimpleNamespace(body_link_quat_w=SimpleNamespace(torch=torch.stack((pelvis, torso), dim=1))),
    )
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: SimpleNamespace(robot=robot)))
    reward = torso_upright_tracking(env)
    expected = torch.tensor([1.0, 1.0, math.exp(-0.64), math.exp(-0.64), math.exp(-0.64), math.exp(-((25 / 15) ** 2))])
    torch.testing.assert_close(reward, expected)
