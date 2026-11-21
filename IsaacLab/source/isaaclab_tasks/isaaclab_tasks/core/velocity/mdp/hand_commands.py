# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hand goals with configurable offset frames with smooth reach-and-hold trajectories."""

import math

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_conjugate, quat_error_magnitude, quat_mul, yaw_quat


class HandPositionCommand(CommandTerm):
    """Move from nominal hand positions toward configured offsets.

    A quintic trajectory starts after a hold, then reaches and holds its goal.
    Reach distance is sampled from global curriculum progress on episode reset
    (or command resampling), so changing difficulty cannot jump an active target.
    Both endpoints have zero velocity and acceleration. No IK or joint control
    runs here. Defaults retain the original stationary targets.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        if cfg.orientation_frame not in ("torso", "world"):
            raise ValueError("orientation_frame must be torso or world")
        if cfg.target_frame not in ("torso", "reset_heading", "world"):
            raise ValueError("target_frame must be torso, reset_heading, or world")
        if not math.isfinite(cfg.world_x_reach_reduction_m) or cfg.world_x_reach_reduction_m < 0:
            raise ValueError("world_x_reach_reduction_m must be finite and nonnegative")
        if cfg.world_x_reach_reduction_m and cfg.target_frame != "world":
            raise ValueError("World X reach reduction requires target_frame=world")
        self.anchor_orientation = torch.zeros(self.num_envs, 4, device=self.device)
        self.anchor_orientation[:, 3] = 1.0
        self.torso_id = self.robot.body_names.index("torso_link")
        # get initial z position of torso.
        self.anchor_z = self.robot.data.body_link_pos_w.torch[:, self.torso_id, 2].clone()
        self.body_ids = []
        offsets = []
        for side in ("left", "right"):
            frame = f"{side}_rubber_hand"
            direct = frame in self.robot.body_names
            self.body_ids.append(self.robot.body_names.index(frame if direct else f"{side}_wrist_yaw_link"))
            offsets.append((0.0, 0.0, 0.0) if direct else (0.0415, 0.003 if side == "left" else -0.003, 0.0))
        self.offsets = torch.tensor(offsets, device=self.device)
        targets = torch.tensor((cfg.left_position, cfg.right_position), device=self.device)
        if targets.shape != (2, 3) or not torch.isfinite(targets).all():
            raise ValueError("Hand targets must be two finite three-dimensional positions")
        orientations = targets.new_tensor((cfg.left_orientation, cfg.right_orientation))
        if orientations.shape != (2, 4) or not torch.isfinite(orientations).all():
            raise ValueError("Hand orientations must be two finite xyzw quaternions")
        norms = torch.linalg.vector_norm(orientations, dim=-1, keepdim=True)
        if (norms < 1.0e-8).any():
            raise ValueError("Hand orientation quaternions must be nonzero")
        self.orientation_targets = (orientations / norms).unsqueeze(0).repeat(self.num_envs, 1, 1)
        self._nominal_orientations = self.orientation_targets.clone()
        self.horizontal_hold_captured = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._held_xy_world_offset = torch.zeros(self.num_envs, 2, 2, device=self.device)
        self._held_orientation_world = self.orientation_targets.clone()
        self.start_positions = targets.unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.targets = self.start_positions.clone()
        self.trajectory_positions = self.start_positions.clone()
        self.trajectory_velocities = torch.zeros_like(self.start_positions)
        self.reach_offsets = targets.new_tensor((cfg.left_reach_offset, cfg.right_reach_offset))
        if self.reach_offsets.shape != (2, 3) or not torch.isfinite(self.reach_offsets).all():
            raise ValueError("Reach offsets must be two finite three-dimensional vectors")
        if not math.isfinite(cfg.reach_duration_s) or cfg.reach_duration_s <= 0:
            raise ValueError("reach_duration_s must be finite and positive")
        if not math.isfinite(cfg.initial_hold_s) or cfg.initial_hold_s < 0:
            raise ValueError("initial_hold_s must be finite and nonnegative")
        if not all(math.isfinite(v) for v in (cfg.initial_reach_fraction, cfg.reach_fraction)):
            raise ValueError("Reach fractions must be finite")
        if not 0 <= cfg.initial_reach_fraction <= cfg.reach_fraction <= 1:
            raise ValueError("Require 0 <= initial_reach_fraction <= reach_fraction <= 1")
        if not isinstance(cfg.curriculum_steps, int) or cfg.curriculum_steps < 0:
            raise ValueError("curriculum_steps must be a nonnegative control-step count")
        self.episode_reach_fraction = torch.zeros(self.num_envs, device=self.device)
        self.trajectory_start_time = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampled_reach_fraction"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["realized_reach_fraction"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def curriculum_fraction(self):
        """Reach fraction scheduled by global control steps; episode resets do not rewind it."""
        progress = (
            1.0
            if self.cfg.curriculum_steps == 0
            else min(1.0, max(0.0, self._env.common_step_counter / self.cfg.curriculum_steps))
        )
        return self.cfg.initial_reach_fraction + progress * (self.cfg.reach_fraction - self.cfg.initial_reach_fraction)

    @property
    def command(self):
        return self.targets.flatten(1)

    def positions(self):
        """Measured rubber-hand origins in the current torso frame."""
        self._refresh_targets()
        data = self.robot.data
        world = data.body_link_pos_w.torch[:, self.body_ids] + quat_apply(
            data.body_link_quat_w.torch[:, self.body_ids], self.offsets.expand(self.num_envs, -1, -1)
        )
        return quat_apply_inverse(
            data.body_link_quat_w.torch[:, self.torso_id, None].expand(-1, 2, -1),
            world - data.body_link_pos_w.torch[:, self.torso_id, None],
        )

    def capture_horizontal_hold(self):
        """Capture world orientations and world-axis XY offsets once at seven seconds."""
        if not self.cfg.horizontal_hold:
            return
        capture = (~self.horizontal_hold_captured) & (self._env.episode_length_buf * self._env.step_dt >= 7.0)
        data = self.robot.data
        hand_world = data.body_link_pos_w.torch[:, self.body_ids] + quat_apply(
            data.body_link_quat_w.torch[:, self.body_ids],
            self.offsets.expand(self.num_envs, -1, -1),
        )
        xy_offset = (
            hand_world[:, :, :2]
            - data.body_link_pos_w.torch[:, self.torso_id, None, :2]
        )
        self._held_xy_world_offset[:] = torch.where(
            capture[:, None, None],
            xy_offset,
            self._held_xy_world_offset,
        )
        self._held_orientation_world[:] = torch.where(
            capture[:, None, None], data.body_link_quat_w.torch[:, self.body_ids], self._held_orientation_world
        )

        self.horizontal_hold_captured |= capture

    def _apply_horizontal_hold(self):
        """Express fixed or captured world orientations in the current torso frame."""
        if self.cfg.orientation_frame == "world":
            torso = self.robot.data.body_link_quat_w.torch[:, self.torso_id, None].expand(-1, 2, -1)
            self.orientation_targets[:] = quat_mul(quat_conjugate(torso), self._nominal_orientations)
            return
        if not self.cfg.horizontal_hold:
            return
        torso = self.robot.data.body_link_quat_w.torch[:, self.torso_id, None].expand(-1, 2, -1)
        self.orientation_targets[:] = torch.where(
            self.horizontal_hold_captured[:, None, None],
            quat_mul(quat_conjugate(torso), self._held_orientation_world),
            self._nominal_orientations,
        )

    def orientations(self):
        """Measured hand-link orientations relative to the torso, in xyzw order.

        The fallback wrist-yaw frame has the same orientation as the rubber hand:
        the G1 palm fixed joint adds only a translation.
        """
        data = self.robot.data
        torso = data.body_link_quat_w.torch[:, self.torso_id, None].expand(-1, 2, -1)
        return quat_mul(quat_conjugate(torso), data.body_link_quat_w.torch[:, self.body_ids])

    def velocities_world(self):
        """Measured velocities of the position-reward hand points, in world m/s."""
        data = self.robot.data
        world_offsets = quat_apply(
            data.body_link_quat_w.torch[:, self.body_ids], self.offsets.expand(self.num_envs, -1, -1)
        )
        return data.body_link_lin_vel_w.torch[:, self.body_ids] + torch.cross(
            data.body_link_ang_vel_w.torch[:, self.body_ids], world_offsets, dim=-1
        )

    def target_velocities_world(self):
        """World velocity of the torso-following target, including offset motion."""
        data = self.robot.data
        offset_velocity = self.trajectory_velocities
        if self.cfg.target_frame == "reset_heading":
            offset_velocity = quat_apply(self.anchor_orientation[:, None].expand(-1, 2, -1), offset_velocity)
        elif self.cfg.target_frame == "torso":
            rotation = data.body_link_quat_w.torch[:, self.torso_id, None].expand(-1, 2, -1)
            world_offsets = quat_apply(rotation, self.trajectory_positions)
            offset_velocity = quat_apply(rotation, offset_velocity) + torch.cross(
                data.body_link_ang_vel_w.torch[:, self.torso_id, None].expand(-1, 2, -1),
                world_offsets,
                dim=-1,
            )
        target_velocity = data.body_link_lin_vel_w.torch[:, self.torso_id, None] + offset_velocity
        if self.cfg.target_frame == "world":
            target_velocity = target_velocity.clone()
            target_velocity[:, :, 2] = self.trajectory_velocities[:, :, 2]
            if self.cfg.world_x_reach_reduction_m:
                _, reduction_velocity = self._world_x_reach_reduction()
                target_velocity[:, :, 0] -= reduction_velocity[:, None]
        if self.cfg.horizontal_hold and self.cfg.target_frame == "world":
            target_velocity[:, :, :2] = torch.where(
                self.horizontal_hold_captured[:, None, None],
                data.body_link_lin_vel_w.torch[:, self.torso_id, None, :2],
                target_velocity[:, :, :2],
            )
        return target_velocity

    def _world_x_reach_reduction(self):
        """Linear episode-time adjustment matching the closer-target MuJoCo replay."""
        elapsed = self._env.episode_length_buf * self._env.step_dt - self.cfg.initial_hold_s
        phase = (elapsed / self.cfg.reach_duration_s).clamp(0.0, 1.0)
        reduction = self.cfg.world_x_reach_reduction_m * phase
        velocity = ((elapsed > 0.0) & (elapsed < self.cfg.reach_duration_s)).to(phase.dtype)
        velocity *= self.cfg.world_x_reach_reduction_m / self.cfg.reach_duration_s
        return reduction, velocity

    def _resample_command(self, env_ids):
        self.episode_reach_fraction[env_ids] = self.curriculum_fraction
        # CommandTerm.reset clears this counter before resampling. The environment
        # clears episode_length_buf only after command reset, so its old clock
        # cannot distinguish a new episode from periodic command resampling.
        resetting = self.command_counter[env_ids] == 0
        self.horizontal_hold_captured[env_ids] &= ~resetting
        # Do not capture here: command reset precedes clearing the old episode clock.

        # reset z position only when episode ends, not every 0.02 seconds
        self.anchor_z[env_ids] = torch.where(
            resetting, self.robot.data.body_link_pos_w.torch[env_ids, self.torso_id, 2], self.anchor_z[env_ids]
        )
        self.trajectory_start_time[env_ids] = torch.where(
            resetting, 0.0, self._env.episode_length_buf[env_ids] * self._env.step_dt
        )
        # Resampling during an episode starts from the current target, preventing
        # a jump back to nominal. Episode resets always restart at nominal.
        if self.cfg.target_frame == "reset_heading":
            orientation = yaw_quat(self.robot.data.root_quat_w.torch[env_ids])
            self.anchor_orientation[env_ids] = torch.where(
                resetting[:, None], orientation, self.anchor_orientation[env_ids]
            )
        self.start_positions[env_ids] = torch.where(
            resetting[:, None, None],
            self.start_positions.new_tensor((self.cfg.left_position, self.cfg.right_position)),
            self.trajectory_positions[env_ids],
        )
        self.trajectory_positions[env_ids] = self.start_positions[env_ids]
        self.trajectory_velocities[env_ids] = 0.0
        self._refresh_targets()

    def _update_command(self):
        elapsed = self._env.episode_length_buf * self._env.step_dt - self.trajectory_start_time
        phase = ((elapsed - self.cfg.initial_hold_s) / self.cfg.reach_duration_s).clamp(0.0, 1.0)
        blend = phase**3 * (10.0 + phase * (-15.0 + 6.0 * phase))
        nominal = self.targets.new_tensor((self.cfg.left_position, self.cfg.right_position))
        goal = nominal + self.episode_reach_fraction[:, None, None] * self.reach_offsets
        self.trajectory_positions[:] = torch.lerp(self.start_positions, goal, blend[:, None, None])
        blend_velocity = 30.0 * phase.square() * (1.0 - phase).square() / self.cfg.reach_duration_s
        self.trajectory_velocities[:] = (goal - self.start_positions) * blend_velocity[:, None, None]
        self._refresh_targets()

    def _refresh_targets(self):
        """Convert configured offsets to torso coordinates for IK and observations.

        World offsets use fixed world axes; reset-heading offsets use the initial
        heading. Both translate with the torso without following its rotation.
        """
        if self.cfg.target_frame == "torso":
            self.targets[:] = self.trajectory_positions
            self._apply_horizontal_hold()
            return
        world_offsets = self.trajectory_positions
        if self.cfg.target_frame == "reset_heading":
            world_offsets = quat_apply(self.anchor_orientation[:, None].expand(-1, 2, -1), world_offsets)

        if self.cfg.target_frame == "world":
            world_offsets = world_offsets.clone()
            if self.cfg.world_x_reach_reduction_m:
                reduction, _ = self._world_x_reach_reduction()
                world_offsets[:, :, 0] -= reduction[:, None]
            if self.cfg.horizontal_hold:
                world_offsets[:, :, :2] = torch.where(
                    self.horizontal_hold_captured[:, None, None],
                    self._held_xy_world_offset,
                    world_offsets[:, :, :2],
                )
            current_torso_z = self.robot.data.body_link_pos_w.torch[:, self.torso_id, 2]
            world_offsets[:, :, 2] += (self.anchor_z - current_torso_z)[:, None]

        self.targets[:] = quat_apply_inverse(
            self.robot.data.body_link_quat_w.torch[:, self.torso_id, None].expand(-1, 2, -1),
            world_offsets,
        )

        self._apply_horizontal_hold()

    def _update_metrics(self):
        """Report scheduled difficulty separately from the realized command trajectory.

        The realized fraction measures target displacement, not measured hand
        tracking. Both values use the trajectory frame, independent of torso yaw.
        """
        self.metrics["sampled_reach_fraction"].copy_(self.episode_reach_fraction)
        nominal = self.trajectory_positions.new_tensor((self.cfg.left_position, self.cfg.right_position))
        displacement = self.trajectory_positions - nominal
        offset_squared = self.reach_offsets.square().sum(-1)
        projection = (displacement * self.reach_offsets).sum(-1)
        fraction = torch.where(offset_squared > 1.0e-12, projection / offset_squared.clamp_min(1.0e-12), 0.0)
        self.metrics["realized_reach_fraction"].copy_(fraction.mean(-1))


@configclass
class HandPositionCommandCfg(CommandTermCfg):
    class_type: type = HandPositionCommand
    asset_name: str = "robot"
    horizontal_hold: bool = False
    """Hold measured world orientation after seven seconds; world-frame targets also hold measured XY offsets."""
    orientation_frame: str = "torso"
    """Frame of configured hand quaternions; world keeps their orientation fixed from episode start."""
    target_frame: str = "torso"
    """Offset axes: torso, initial heading, or world. The origin always follows the torso position."""
    world_x_reach_reduction_m: float = 0.0
    """Subtract this world X distance with a linear initial-hold/reach ramp, independently of reach curriculum."""
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    # FK of LIP/initial_states/g1_reset.json using the matched G1 URDF.
    left_position: tuple[float, float, float] = (0.245238359, 0.151653753, 0.041230731)
    right_position: tuple[float, float, float] = (0.245238359, -0.151643753, 0.041230731)
    # Keep gaitfix forward/lateral endpoints, with both goals 0.20 m above the
    # torso origin (3.778 cm below the shoulder pivots when upright).
    # Orientation targets use orientation_frame independently of the position target_frame.
    # Identity aligns the mirrored G1 hand links with torso axes (fingers +X,
    # left palm inward toward -Y, right palm inward toward +Y).
    left_orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    right_orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    """Desired hand-link xyzw quaternions in orientation_frame; used when orientation tracking is enabled."""
    left_reach_offset: tuple[float, float, float] = (0.15739560465512467, 0.026332706505700465, 0.158769269)
    right_reach_offset: tuple[float, float, float] = (0.15935026853875714, -0.014757519561061638, 0.158769269)
    initial_hold_s: float = 1.0
    reach_duration_s: float = 6.0
    initial_reach_fraction: float = 0.0
    reach_fraction: float = 0.0
    """Maximum fraction of the configured offsets; zero preserves stationary goals."""
    curriculum_steps: int = 10000
    """Global control steps to reach maximum distance; zero disables the ramp."""


def hand_targets_and_errors(env, command_name: str = "hand_position"):
    """12 appended values: left/right targets, then target-minus-actual errors."""
    command = env.command_manager.get_term(command_name)
    return torch.cat((command.command, (command.targets - command.positions()).flatten(1)), dim=-1)


def hand_position_tracking(env, side: str, std: float = 0.10, command_name: str = "hand_position"):
    """Independent bounded position reward for a hand; std is in meters."""
    if side not in ("left", "right"):
        raise ValueError("Hand side must be left or right")
    if not math.isfinite(std) or std <= 0:
        raise ValueError("Hand tracking std must be finite and positive")
    command = env.command_manager.get_term(command_name)
    index = 0 if side == "left" else 1
    error = command.targets[:, index] - command.positions()[:, index]
    return torch.exp(-error.square().sum(-1) / std**2)


def hand_position_penalty(env, side: str, command_name: str = "hand_position"):
    """Squared hand-position error in meters squared, one value per environment.

    Targets and measured positions use the same torso frame as hand tracking.
    The squared distance is invariant under conversion to world coordinates.
    Use a negative reward weight (for example, -2.0) to penalize this error.
    """
    if side not in ("left", "right"):
        raise ValueError("Hand side must be left or right")
    command = env.command_manager.get_term(command_name)
    index = 0 if side == "left" else 1
    error = command.targets[:, index] - command.positions()[:, index]
    return error.square().sum(-1)


def hand_orientation_tracking(env, side: str, std: float = 0.5, command_name: str = "hand_position"):
    """General-lowbody end-effector orientation reward, using angular error in radians."""
    if side not in ("left", "right"):
        raise ValueError("side must be left or right")
    if not math.isfinite(std) or std <= 0:
        raise ValueError("Hand orientation tracking std must be finite and positive")
    command = env.command_manager.get_term(command_name)
    command._refresh_targets()
    index = 0 if side == "left" else 1
    error = quat_error_magnitude(command.orientations()[:, index], command.orientation_targets[:, index])
    return torch.exp(-error / std**2)


def hand_orientation_penalty(env, side: str, command_name: str = "hand_position"):
    """General-lowbody end-effector orientation penalty in radians."""
    if side not in ("left", "right"):
        raise ValueError("side must be left or right")
    command = env.command_manager.get_term(command_name)
    command._refresh_targets()
    index = 0 if side == "left" else 1
    return quat_error_magnitude(command.orientations()[:, index], command.orientation_targets[:, index])


def hand_linear_velocity_tracking(env, side: str, std: float = 1.0, command_name: str = "hand_position"):
    """Bounded hand velocity reward; std is in m/s, with weight configured separately."""
    if side not in ("left", "right"):
        raise ValueError("Hand side must be left or right")
    if not math.isfinite(std) or std <= 0:
        raise ValueError("Hand velocity tracking std must be finite and positive")
    command = env.command_manager.get_term(command_name)
    index = 0 if side == "left" else 1
    error = command.target_velocities_world()[:, index] - command.velocities_world()[:, index]
    return torch.exp(-error.square().sum(-1) / std**2)


def ik_joint_errors(env, action_name: str = "joint_pos"):
    """Previous control step's measured-minus-IK references, observable before the next action."""
    return env.action_manager.get_term(action_name).joint_reference_errors()


def ik_joint_error_penalty(env, action_name: str = "joint_pos", std: float = 0.3):
    """Mean squared tracking error normalized by std radians over all 14 arm joints.

    References are generated online before physics; rewards compare them with the
    measured joints after physics. Invalid and reset references contribute zero.
    """
    if not math.isfinite(std) or std <= 0:
        raise ValueError("Joint error std must be finite and positive")
    return (ik_joint_errors(env, action_name) / std).square().mean(dim=-1)


def heading_feedback(env):
    """Signed root and torso heading errors; optional for freshly transferred walking policies."""
    from .lipm_mdp import heading_error

    return torch.stack((heading_error(env), heading_error(env, body_name="torso_link")), dim=-1)
