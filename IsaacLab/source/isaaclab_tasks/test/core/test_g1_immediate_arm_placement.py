# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Immediate arm placement must prescribe the endpoint at every physics boundary."""

from types import SimpleNamespace

import torch

from isaaclab_tasks.core.velocity.mdp.hand_actions import HandIKJointPositionAction


def test_immediate_and_interpolated_arm_writes():
    writes = []

    def write(**kwargs):
        writes.append((kwargs["position"].clone(), kwargs["velocity"].clone(), kwargs["joint_ids"]))

    action = SimpleNamespace(
        cfg=SimpleNamespace(immediate_arm_placement=True),
        _arm_start=torch.tensor([[0.0, 0.0]]),
        _arm_goal=torch.tensor([[0.04, -0.02]]),
        _arm_velocity=torch.zeros(1, 2),
        _arm_ids=[3, 8],
        _asset=SimpleNamespace(write_joint_state_to_sim_index=write),
        _env=SimpleNamespace(cfg=SimpleNamespace(decimation=8), sim=SimpleNamespace(forward=lambda: None)),
    )
    for step in (0, 4, 8):
        action._placement_step = step
        HandIKJointPositionAction._write_arm_state(action)
        position, velocity, joint_ids = writes[-1]
        torch.testing.assert_close(position, torch.tensor([[0.04, -0.02]]))
        assert torch.count_nonzero(velocity) == 0
        assert joint_ids == [3, 8]
    action.cfg.immediate_arm_placement = False
    action._placement_step = 4
    action._arm_velocity[:] = torch.tensor([[2.0, -1.0]])
    HandIKJointPositionAction._write_arm_state(action)
    torch.testing.assert_close(writes[-1][0], torch.tensor([[0.02, -0.01]]))
    torch.testing.assert_close(writes[-1][1], torch.tensor([[2.0, -1.0]]))
