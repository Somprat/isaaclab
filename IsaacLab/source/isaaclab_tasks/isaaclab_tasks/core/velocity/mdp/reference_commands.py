# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reference-backed velocity command for the existing G1 flat-walking task."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_mul, yaw_quat

from .reference_motion import ReferenceMotionLibrary

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class ReferenceMotionCommand(CommandTerm):
    """Drive commands/phase/targets from one clip and one fixed reset transform.

    A target is sampled one control step ahead: ManagerBasedRLEnv evaluates the
    reward before advancing its command manager. Thus the action and its reward
    refer to the same post-physics reference state, with no one-step lag.
    """

    def __init__(self, cfg: ReferenceMotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        self.library = ReferenceMotionLibrary(
            cfg.motion_directory, self.robot.joint_names, self.robot.body_names, self.device
        )
        self.root_index = self.robot.body_names.index(cfg.root_body)
        self.clip_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.start_times = torch.zeros(self.num_envs, device=self.device)
        self.anchor_xy = torch.zeros(self.num_envs, 3, device=self.device)
        self.rotation = torch.zeros(self.num_envs, 4, device=self.device)
        self.rotation[:, 3] = 1
        self.reference = self.library.sample(self.clip_ids, self.start_times)
        self._command = torch.zeros(self.num_envs, 3, device=self.device)
        self.conditioning = torch.zeros(self.num_envs, 2, device=self.device)
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.metrics["joint_position_rmse"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _aligned_sample(self, ids: torch.Tensor, times: torch.Tensor) -> dict[str, torch.Tensor]:
        data = self.library.sample(self.clip_ids[ids], times)
        rotation = self.rotation[ids, None, :].expand_as(data["body_quat_w"])
        data["body_pos_w"] = (
            quat_apply(rotation, data["body_pos_w"] - self.anchor_xy[ids, None, :])
            + self._env.scene.env_origins[ids, None, :]
        )
        data["body_quat_w"] = quat_mul(rotation, data["body_quat_w"])
        for name in ("body_lin_vel_w", "body_ang_vel_w"):
            data[name] = quat_apply(rotation, data[name])
        return data

    def _resample_command(self, env_ids):
        ids = (
            torch.arange(self.num_envs, device=self.device)[env_ids]
            if isinstance(env_ids, slice)
            else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        )
        self.clip_ids[ids] = torch.randint(len(self.library.files), (len(ids),), device=self.device)
        available = (self.library.durations[self.clip_ids[ids]] - self.cfg.minimum_remaining_s).clamp_min(0)
        self.start_times[ids] = torch.rand(len(ids), device=self.device) * available if self.cfg.random_start else 0
        raw = self.library.sample(self.clip_ids[ids], self.start_times[ids])
        self.anchor_xy[ids] = raw["body_pos_w"][:, self.root_index]
        self.anchor_xy[ids, 2] = 0
        initial_yaw = yaw_quat(raw["body_quat_w"][:, self.root_index])
        yaw = torch.empty(len(ids), device=self.device).uniform_(*self.cfg.reset_heading_range)
        zeros = torch.zeros_like(yaw)
        desired_yaw = quat_from_euler_xyz(zeros, zeros, yaw)
        initial_yaw[:, :3] *= -1
        self.rotation[ids] = quat_mul(desired_yaw, initial_yaw)
        reset = self._aligned_sample(ids, self.start_times[ids])
        root = self.root_index
        root_state = torch.cat(
            (
                reset["body_pos_w"][:, root],
                reset["body_quat_w"][:, root],
                reset["body_lin_vel_w"][:, root],
                reset["body_ang_vel_w"][:, root],
            ),
            dim=-1,
        )
        root_state[:, 2] += self.cfg.reset_height_offset
        # Body velocities in the export are link-origin velocities, not COM velocities.
        self.robot.write_root_link_pose_to_sim_index(root_pose=root_state[:, :7], env_ids=ids)
        self.robot.write_root_link_velocity_to_sim_index(root_velocity=root_state[:, 7:], env_ids=ids)
        self.robot.write_joint_state_to_sim(reset["joint_pos"], reset["joint_vel"], env_ids=ids)
        self._set_targets(ids, self.start_times[ids] + self._env.step_dt)

    def _set_targets(self, ids: torch.Tensor, times: torch.Tensor):
        data = self._aligned_sample(ids, times)
        for name, value in data.items():
            self.reference[name][ids] = value
        self.conditioning[ids] = data["conditioning"]
        self.phase[ids] = data["phase"]
        self._command[ids, 0] = data["conditioning"][:, 0]
        self._command[ids, 1:] = 0  # Straight forward clips; yaw change remains a tracked pose reference.

    def _update_command(self):
        ids = torch.arange(self.num_envs, device=self.device)
        times = self.start_times + (self._env.episode_length_buf + 1) * self._env.step_dt
        self._set_targets(ids, times)

    def _update_metrics(self):
        error = self.robot.data.joint_pos.torch - self.reference["joint_pos"]
        self.metrics["joint_position_rmse"][:] = error.square().mean(-1).sqrt()


@configclass
class ReferenceMotionCommandCfg(CommandTermCfg):
    class_type: type = ReferenceMotionCommand
    resampling_time_range: tuple[float, float] = (1e9, 1e9)
    asset_name: str = "robot"
    motion_directory: str = MISSING
    root_body: str = "pelvis"
    random_start: bool = True
    minimum_remaining_s: float = 3.0
    reset_heading_range: tuple[float, float] = (0.0, 0.0)
    reset_height_offset: float = 0.02
