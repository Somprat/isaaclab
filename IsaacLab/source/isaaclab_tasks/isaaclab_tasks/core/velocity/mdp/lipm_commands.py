# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared walking/arm reference clock and fixed reset alignment for LIPM clips."""

from __future__ import annotations

from dataclasses import MISSING

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_conjugate, quat_mul, yaw_quat

from .lipm_motion import LipmMotionLibrary


def robot_com(robot):
    """Actual mass-weighted articulated CoM position and velocity in world space."""
    mass = robot.data.body_mass.torch
    weights = (mass / mass.sum(-1, keepdim=True)).unsqueeze(-1)
    return ((weights * robot.data.body_com_pos_w.torch).sum(1), (weights * robot.data.body_com_lin_vel_w.torch).sum(1))


class LipmMotionCommand(CommandTerm):
    """Expose post-action targets one control step ahead, without setting robot joints.

    Reset events own the initial robot state. Only reference XY and yaw are
    aligned to that reset state; reference height is preserved above env ground.
    This is intentionally not a full-body reference-state initialization.

    For arm-augmented clips, ``reference`` also contains ``arm_joint_pos`` and
    ``arm_joint_vel`` in the robot order selected by ``arm_joint_ids``, plus
    ``hand_pos_torso`` in the library's hand order. All targets use the same
    clip and one-step-ahead clock, including after partial resets. Arm joint
    targets and torso-relative hand positions are independent of world alignment.
    The ``command`` tensor remains the three-value locomotion velocity command;
    arm observations/rewards consume the named targets from ``reference``.
    """

    def __init__(self, cfg: LipmMotionCommandCfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        self.library = LipmMotionLibrary(cfg.motion_directory, self.device, joint_names=self.robot.joint_names)
        self.foot_ids = [self.robot.body_names.index(name) for name in cfg.foot_body_names]
        if len(self.foot_ids) != 2 or len(cfg.foot_offsets) != 2:
            raise ValueError("LIPM needs two foot bodies and offsets ordered left, right")
        self.foot_offsets = torch.tensor(cfg.foot_offsets, device=self.device, dtype=torch.float32)
        self.clip_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.start_times = torch.zeros(self.num_envs, device=self.device)
        self.anchor = torch.zeros(self.num_envs, 3, device=self.device)
        self.origin = torch.zeros_like(self.anchor)
        self.rotation = torch.zeros(self.num_envs, 4, device=self.device)
        self.rotation[:, 3] = 1
        self.reference = self.library.sample(self.clip_ids, self.start_times)
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self._command = torch.zeros(self.num_envs, 3, device=self.device)
        self.metrics["com_position_error"] = torch.zeros(self.num_envs, device=self.device)
        for name in ("contact_agreement", "loaded_swing_fraction", "stance_slide_speed_m_s", "sole_speed_m_s"):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)
        if self.has_arm_reference:
            self.metrics["arm_joint_position_rmse"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self):
        return self._command

    @property
    def has_arm_reference(self) -> bool:
        """Whether the selected motion library supplies arm and hand targets."""
        return self.library.has_arm_reference

    @property
    def arm_joint_ids(self) -> torch.Tensor | None:
        """Robot joint indices matching arm target columns; None for walking-only clips."""
        return self.library.arm_joint_ids

    def foot_positions(self):
        """Sole points, not ankle origins; offsets are expressed in each foot frame."""
        q = self.robot.data.body_quat_w.torch[:, self.foot_ids]
        return self.robot.data.body_pos_w.torch[:, self.foot_ids] + quat_apply(
            q, self.foot_offsets.expand(self.num_envs, -1, -1)
        )

    def foot_velocities(self):
        """World velocity at the sole point: link-origin velocity plus omega cross offset."""
        data = self.robot.data
        offset = quat_apply(
            data.body_link_quat_w.torch[:, self.foot_ids], self.foot_offsets.expand(self.num_envs, -1, -1)
        )
        return data.body_link_lin_vel_w.torch[:, self.foot_ids] + torch.cross(
            data.body_link_ang_vel_w.torch[:, self.foot_ids], offset, dim=-1
        )

    def foot_contacts(self, force_threshold=10.0, sensor_name="contact_forces"):
        """Current net-force contact flags in reference left/right order."""
        sensor = self._env.scene[sensor_name]
        ids = [sensor.body_names.index(name) for name in self.cfg.foot_body_names]
        forces = sensor.data.net_forces_w
        forces = forces.torch if hasattr(forces, "torch") else forces
        return forces[:, ids].norm(dim=-1) > force_threshold

    def _resample_command(self, env_ids):
        ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        self.clip_ids[ids] = torch.randint(len(self.library.files), (len(ids),), device=self.device)
        self.start_times[ids] = 0
        data = self.library.sample(self.clip_ids[ids], self.start_times[ids])
        self.anchor[ids] = data["com_pos"]
        self.anchor[ids, 2] = 0
        com, _ = robot_com(self.robot)
        self.origin[ids] = com[ids]
        self.origin[ids, 2] = self._env.scene.env_origins[ids, 2]
        angle = data["heading"]
        source_yaw = torch.zeros(len(ids), 4, device=self.device)
        source_yaw[:, 2] = torch.sin(angle / 2)
        source_yaw[:, 3] = torch.cos(angle / 2)
        self.rotation[ids] = quat_mul(yaw_quat(self.robot.data.root_quat_w.torch[ids]), quat_conjugate(source_yaw))
        self._set_targets(ids, self.start_times[ids] + self._env.step_dt)

    def _set_targets(self, ids, times):
        data = self.library.sample(self.clip_ids[ids], times)
        q = self.rotation[ids]
        data["com_pos"] = quat_apply(q, data["com_pos"] - self.anchor[ids]) + self.origin[ids]
        data["foot_pos"] = (
            quat_apply(q[:, None].expand(-1, 2, -1), data["foot_pos"] - self.anchor[ids, None]) + self.origin[ids, None]
        )
        data["com_vel"] = quat_apply(q, data["com_vel"])
        # Joint angles/velocities and torso-frame hand positions need no world
        # transform. Copy them with the walking targets from this same sample.
        for key, value in data.items():
            self.reference[key][ids] = value
        self.phase[ids] = data["phase"]
        # Compatibility with locomotion terms expecting a heading-frame velocity command.
        speed = data["desired_step_length"] / data["step_duration"]
        nominal = torch.stack(
            (speed * data["heading"].cos(), speed * data["heading"].sin(), torch.zeros_like(speed)), -1
        )
        nominal = quat_apply(q, nominal)
        self._command[ids] = quat_apply_inverse(yaw_quat(self.robot.data.root_quat_w.torch[ids]), nominal)
        self._command[ids, 2] = 0

    def _update_command(self):
        ids = torch.arange(self.num_envs, device=self.device)
        self._set_targets(ids, self.start_times + (self._env.episode_length_buf + 1) * self._env.step_dt)

    def _update_metrics(self):
        com, _ = robot_com(self.robot)
        self.metrics["com_position_error"][:] = (com - self.reference["com_pos"]).norm(dim=-1)
        support = self.reference["support_mask"].bool()
        contact = self.foot_contacts()
        speed = self.foot_velocities()[..., :2].norm(dim=-1)
        self.metrics["contact_agreement"][:] = (contact == support).float().mean(-1)
        self.metrics["loaded_swing_fraction"][:] = (contact & ~support).sum(-1) / (~support).sum(-1).clamp_min(1)
        self.metrics["stance_slide_speed_m_s"][:] = (speed * (contact & support)).sum(-1) / support.sum(-1).clamp_min(1)
        self.metrics["sole_speed_m_s"][:] = speed.mean(-1)
        if self.has_arm_reference:
            error = self.robot.data.joint_pos.torch[:, self.arm_joint_ids] - self.reference["arm_joint_pos"]
            self.metrics["arm_joint_position_rmse"][:] = error.square().mean(dim=-1).sqrt()


@configclass
class LipmMotionCommandCfg(CommandTermCfg):
    class_type: type = LipmMotionCommand
    resampling_time_range: tuple[float, float] = (1e9, 1e9)
    asset_name: str = "robot"
    motion_directory: str = MISSING
    foot_body_names: tuple[str, str] = ("left_ankle_roll_link", "right_ankle_roll_link")
    # Matched G1 URDF collision spheres have centers at z=-0.03 and radius 0.005.
    # Use the sole point under the ankle; tune XY for another foot reference point.
    foot_offsets: tuple = ((0.0, 0.0, -0.035), (0.0, 0.0, -0.035))
