# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Conditioning and reference-target observations for G1 imitation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def stride_length(env: ManagerBasedRLEnv, command_name: str = "base_velocity") -> torch.Tensor:
    return env.command_manager.get_term(command_name).conditioning[:, 1:2]


def step_length(env: ManagerBasedRLEnv, command_name: str = "base_velocity") -> torch.Tensor:
    """Nominal alternating-foot step length in meters (half the exported full stride).

    The step-conditioned MPC exporter stores legacy conditioning as [L/T, 2L],
    so this recovers the applied step command, including startup and stage changes.
    """
    return 0.5 * env.command_manager.get_term(command_name).conditioning[:, 1:2]


def gait_phase(env: ManagerBasedRLEnv, command_name: str = "base_velocity") -> torch.Tensor:
    phase = env.command_manager.get_term(command_name).phase
    return torch.stack((phase.sin(), phase.cos()), dim=-1)


def reference_joint_positions(env: ManagerBasedRLEnv, command_name: str = "base_velocity") -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    return command.reference["joint_pos"] - command.robot.data.default_joint_pos.torch


def reference_joint_velocities(env: ManagerBasedRLEnv, command_name: str = "base_velocity") -> torch.Tensor:
    return env.command_manager.get_term(command_name).reference["joint_vel"]


def reference_root_orientation(env: ManagerBasedRLEnv, command_name: str = "base_velocity") -> torch.Tensor:
    """Desired root orientation relative to the current root, in xyzw order."""
    from isaaclab.utils.math import quat_conjugate, quat_mul

    command = env.command_manager.get_term(command_name)
    relative = quat_mul(
        quat_conjugate(command.robot.data.root_quat_w.torch), command.reference["body_quat_w"][:, command.root_index]
    )
    return torch.where(relative[:, 3:4] < 0, -relative, relative)


def reference_root_height(env: ManagerBasedRLEnv, command_name: str = "base_velocity") -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    return (
        command.reference["body_pos_w"][:, command.root_index, 2] - command.robot.data.root_pos_w.torch[:, 2]
    ).unsqueeze(-1)
