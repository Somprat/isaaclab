# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded imitation rewards; tracking uses the same clock as the observations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_position_tracking(env: ManagerBasedRLEnv, std: float, command_name: str = "base_velocity") -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    error = command.robot.data.joint_pos.torch - command.reference["joint_pos"]
    return torch.exp(-error.square().mean(-1) / std**2)


def joint_velocity_tracking(env: ManagerBasedRLEnv, std: float, command_name: str = "base_velocity") -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    error = command.robot.data.joint_vel.torch - command.reference["joint_vel"]
    return torch.exp(-error.square().mean(-1) / std**2)


def root_height_tracking(env: ManagerBasedRLEnv, std: float, command_name: str = "base_velocity") -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    error = command.robot.data.root_pos_w.torch[:, 2] - command.reference["body_pos_w"][:, command.root_index, 2]
    return torch.exp(-error.square() / std**2)


def root_orientation_tracking(env: ManagerBasedRLEnv, std: float, command_name: str = "base_velocity") -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    a = torch.nn.functional.normalize(command.robot.data.root_quat_w.torch, dim=-1)
    b = command.reference["body_quat_w"][:, command.root_index]
    angle = 2 * torch.acos((a * b).sum(-1).abs().clamp(0, 1))
    return torch.exp(-angle.square() / std**2)


def body_position_tracking(
    env: ManagerBasedRLEnv, std: float, body_names: list[str], command_name: str = "base_velocity"
) -> torch.Tensor:
    """Track key bodies relative to the pelvis in each pose's local root frame."""
    command = env.command_manager.get_term(command_name)
    indices = [command.robot.body_names.index(name) for name in body_names]
    robot = command.robot.data
    positions = robot.body_pos_w.torch[:, indices] - robot.root_pos_w.torch[:, None]
    reference = (
        command.reference["body_pos_w"][:, indices] - command.reference["body_pos_w"][:, command.root_index, None]
    )
    q = robot.root_quat_w.torch[:, None].expand(-1, len(indices), -1)
    qr = command.reference["body_quat_w"][:, command.root_index, None].expand_as(q)
    error = quat_apply_inverse(q, positions) - quat_apply_inverse(qr, reference)
    return torch.exp(-error.square().sum(-1).mean(-1) / std**2)


def reference_finished(env: ManagerBasedRLEnv, command_name: str = "base_velocity") -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    times = command.start_times + env.episode_length_buf * env.step_dt
    return times >= command.library.durations[command.clip_ids]
