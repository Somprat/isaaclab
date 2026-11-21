# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LIPM walking and optional arm-reference observations and tracking rewards."""

import math

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat

from .lipm_commands import robot_com


def tracking_errors(env, command_name: str = "base_velocity"):
    """Foot and true CoM errors in the robot heading frame (meters, m/s)."""
    command = env.command_manager.get_term(command_name)
    com, vel = robot_com(command.robot)
    q = yaw_quat(command.robot.data.root_quat_w.torch)
    feet = quat_apply_inverse(q[:, None].expand(-1, 2, -1), command.reference["foot_pos"] - command.foot_positions())
    return (
        feet,
        quat_apply_inverse(q, command.reference["com_pos"] - com),
        quat_apply_inverse(q, command.reference["com_vel"] - vel),
    )


def lipm_targets(env, command_name: str = "base_velocity"):
    """19 values: foot errors(6), CoM errors(6), support(2), phase(2), gait(3)."""
    command = env.command_manager.get_term(command_name)
    feet, com, vel = tracking_errors(env, command_name)
    ref = command.reference
    return torch.cat(
        (
            feet.flatten(1),
            com,
            vel,
            ref["support_mask"].float(),
            command.phase.sin()[:, None],
            command.phase.cos()[:, None],
            ref["desired_step_length"][:, None],
            ref["desired_step_width"][:, None],
            ref["step_duration"][:, None],
        ),
        -1,
    )


def heading_error(env, body_name: str | None = None, command_name: str = "base_velocity"):
    """Signed target-minus-actual heading error in radians, wrapped to [-pi, pi].

    Reference heading is transformed by the episode's fixed reset alignment;
    it does not follow subsequent robot turns. Project local +X into world XY
    so pelvis and torso can be measured independently of their roll/pitch.
    """
    command = env.command_manager.get_term(command_name)
    robot = command.robot
    if body_name is None:
        orientation = robot.data.root_quat_w.torch
    else:
        orientation = robot.data.body_link_quat_w.torch[:, robot.body_names.index(body_name)]
    heading = command.reference["heading"]
    desired = torch.stack((heading.cos(), heading.sin(), torch.zeros_like(heading)), dim=-1)
    desired = quat_apply(command.rotation, desired)
    forward = torch.zeros_like(desired)
    forward[:, 0] = 1.0
    actual = quat_apply(orientation, forward)
    cross = actual[:, 0] * desired[:, 1] - actual[:, 1] * desired[:, 0]
    dot = (actual[:, :2] * desired[:, :2]).sum(-1)
    return torch.atan2(cross, dot)


def heading_tracking(
    env,
    std: float = math.radians(15),
    tolerance: float = math.radians(3),
    body_name: str | None = None,
    command_name: str = "base_velocity",
):
    """Allow small heading excursions, then reward returning to the reference."""
    if not math.isfinite(std) or std <= 0:
        raise ValueError("Heading std must be finite and positive")
    if not math.isfinite(tolerance) or not 0 <= tolerance < math.pi:
        raise ValueError("Heading tolerance must be finite and in [0, pi)")
    error = (heading_error(env, body_name, command_name).abs() - tolerance).clamp_min(0.0)
    return torch.exp(-error.square() / std**2)


def torso_upright_tracking(
    env,
    tolerance: float = math.radians(8),
    std: float = math.radians(15),
    body_name: str = "torso_link",
    command_name: str = "base_velocity",
):
    """Reward world-upright torso orientation, allowing a cone of natural lean.

    Tilt is measured from the torso's local +Z axis to world +Z. It constrains
    roll and pitch independently of pelvis posture and leaves heading unrestricted.
    """
    if not math.isfinite(tolerance) or not 0 <= tolerance < math.pi:
        raise ValueError("Torso tilt tolerance must be finite and in [0, pi)")
    if not math.isfinite(std) or std <= 0:
        raise ValueError("Torso tilt std must be finite and positive")
    robot = env.command_manager.get_term(command_name).robot
    orientation = robot.data.body_link_quat_w.torch[:, robot.body_names.index(body_name)]
    up = torch.zeros_like(orientation[:, :3])
    up[:, 2] = 1.0
    up_world = quat_apply(orientation, up)
    tilt = torch.atan2(up_world[:, :2].norm(dim=-1), up_world[:, 2])
    excess = (tilt - tolerance).clamp_min(0.0)
    return torch.exp(-excess.square() / std**2)


def torso_angular_velocity(env, command_name: str = "base_velocity"):
    """Torso-local angular velocity (rad/s), from the physical link state."""
    robot = env.command_manager.get_term(command_name).robot
    index = robot.body_names.index("torso_link")
    return quat_apply_inverse(
        robot.data.body_link_quat_w.torch[:, index], robot.data.body_link_ang_vel_w.torch[:, index]
    )


def torso_yaw_motion_penalty(env, tolerance: float = 0.1, command_name: str = "base_velocity"):
    """Squared excess torso rotation speed about world vertical, allowing gentle correction."""
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Yaw speed tolerance must be finite and nonnegative")
    robot = env.command_manager.get_term(command_name).robot
    index = robot.body_names.index("torso_link")
    speed = robot.data.body_link_ang_vel_w.torch[:, index, 2]
    return (speed.abs() - tolerance).clamp_min(0.0).square()


def waist_yaw_motion_penalty(env, tolerance: float = 0.1, command_name: str = "base_velocity"):
    """Squared excess waist-yaw joint speed; does not penalize waist roll/pitch."""
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Yaw speed tolerance must be finite and nonnegative")
    robot = env.command_manager.get_term(command_name).robot
    index = robot.joint_names.index("waist_yaw_joint")
    speed = robot.data.joint_vel.torch[:, index]
    return (speed.abs() - tolerance).clamp_min(0.0).square()


def _arm_command(env, command_name: str):
    command = env.command_manager.get_term(command_name)
    if not command.has_arm_reference or command.arm_joint_ids is None:
        raise ValueError(
            "Arm tracking requires an arm-augmented LIP NPZ. Generate one with "
            "LIP/LIPM/export_hand_ik_reference.py and set commands.base_velocity.motion_directory."
        )
    return command


def arm_targets(env, command_name: str = "base_velocity"):
    """2J values: reference arm positions relative to default pose, then velocities.

    J is the number of referenced arm joints (typically 7 or 14). Columns follow
    command.arm_joint_ids, and velocities are absolute joint velocities in rad/s.
    The existing policy observations already supply actual joint positions and
    velocities, so the policy can infer tracking errors from these targets.
    """
    command = _arm_command(env, command_name)
    nominal = command.robot.data.default_joint_pos.torch[:, command.arm_joint_ids]
    return torch.cat((command.reference["arm_joint_pos"] - nominal, command.reference["arm_joint_vel"]), dim=-1)


def arm_joint_position_tracking(env, std: float = 0.3, command_name: str = "base_velocity"):
    """Bounded tracking reward over referenced arm joints only; std is in radians."""
    if not math.isfinite(std) or std <= 0:
        raise ValueError("Arm position tracking std must be finite and positive")
    command = _arm_command(env, command_name)
    error = command.robot.data.joint_pos.torch[:, command.arm_joint_ids] - command.reference["arm_joint_pos"]
    return torch.exp(-error.square().mean(dim=-1) / std**2)


def hand_position_tracking(env, side: str, std: float = 0.10, command_name: str = "base_velocity"):
    """Track one G1 rubber-hand origin in the reference torso frame.

    Use the hand body directly when present, otherwise the matched G1 URDF's
    fixed wrist-to-hand translation. Each hand has an independent exponential;
    an unreferenced hand contributes zero for single-hand clips.
    """
    if side not in ("left", "right"):
        raise ValueError("Hand side must be left or right")
    if not math.isfinite(std) or std <= 0:
        raise ValueError("Hand position tracking std must be finite and positive")
    command = _arm_command(env, command_name)
    if side not in command.library.hand_names:
        return torch.zeros(env.num_envs, device=command.device)
    index = command.library.hand_names.index(side)
    frame = command.library.hand_frame_names[index]
    if frame != f"{side}_rubber_hand" or command.library.hand_target_frame != "torso_link":
        raise ValueError("G1 hand reward requires rubber-hand targets in torso_link")
    robot = command.robot
    data = robot.data
    torso_id = robot.body_names.index("torso_link")
    if frame in robot.body_names:
        hand_world = data.body_link_pos_w.torch[:, robot.body_names.index(frame)]
    else:
        wrist_id = robot.body_names.index(f"{side}_wrist_yaw_link")
        offset = data.body_link_pos_w.torch.new_tensor([0.0415, 0.003 if side == "left" else -0.003, 0.0])
        hand_world = data.body_link_pos_w.torch[:, wrist_id] + quat_apply(
            data.body_link_quat_w.torch[:, wrist_id], offset.expand(env.num_envs, -1)
        )
    actual = quat_apply_inverse(
        data.body_link_quat_w.torch[:, torso_id], hand_world - data.body_link_pos_w.torch[:, torso_id]
    )
    error = actual - command.reference["hand_pos_torso"][:, index]
    return torch.exp(-error.square().sum(-1) / std**2)


def arm_joint_velocity_tracking(env, std: float = 1.0, command_name: str = "base_velocity"):
    """Bounded tracking reward over referenced arm joints only; std is in rad/s."""
    if not math.isfinite(std) or std <= 0:
        raise ValueError("Arm velocity tracking std must be finite and positive")
    command = _arm_command(env, command_name)
    error = command.robot.data.joint_vel.torch[:, command.arm_joint_ids] - command.reference["arm_joint_vel"]
    return torch.exp(-error.square().mean(dim=-1) / std**2)


def unreferenced_joint_deviation_l1(env, asset_cfg: SceneEntityCfg, command_name: str = "base_velocity"):
    """Keep the selected joints near nominal, excluding joints with IK targets.

    Used for the inherited arm posture penalty: a one-hand reference leaves the
    opposite arm's posture regularization active, while a two-hand reference
    excludes both arms. This does not modify robot state or reference tensors.
    """
    command = _arm_command(env, command_name)
    robot = env.scene[asset_cfg.name]
    if robot is not command.robot:
        raise ValueError("Posture penalty asset must match the arm reference command asset")
    error = (robot.data.joint_pos.torch - robot.data.default_joint_pos.torch).abs()
    error[:, command.arm_joint_ids] = 0
    return error[:, asset_cfg.joint_ids].sum(dim=-1)


def foot_position_tracking(env, std: float = 0.08, command_name: str = "base_velocity"):
    feet, _, _ = tracking_errors(env, command_name)
    return torch.exp(-feet.square().sum(-1).mean(-1) / std**2)


def foot_horizontal_position_tracking(env, std: float = 0.08, command_name: str = "base_velocity"):
    """Track sole placement in XY independently of vertical swing error."""
    feet, _, _ = tracking_errors(env, command_name)
    return torch.exp(-feet[..., :2].square().sum(-1).mean(-1) / std**2)


def foot_com_relative_tracking(env, std: float = 0.12, command_name: str = "base_velocity"):
    """Track horizontal sole offsets from actual/reference mass-weighted CoM.

    Subtracting the CoM error from each foot error cancels common translation.
    Both errors share the current heading frame; heading/path errors remain
    constrained by the separate world-position and stability rewards.
    """
    if not math.isfinite(std) or std <= 0:
        raise ValueError("CoM-relative foot tracking std must be finite and positive")
    feet, com, _ = tracking_errors(env, command_name)
    error = feet[..., :2] - com[:, None, :2]
    return torch.exp(-error.square().sum(-1).mean(-1) / std**2)


def swing_height_tracking(env, std: float = 0.02, command_name: str = "base_velocity"):
    """Track reference sole Z for scheduled swing feet, with zero during double support.

    Targets and measured sole points share the aligned world frame. The NPZ
    supplies the complete lift/lower trajectory; no fixed peak height is imposed.
    Average only over swing feet so stance errors and XY drift cannot dilute it.
    """
    if not math.isfinite(std) or std <= 0:
        raise ValueError("Swing height tracking std must be finite and positive")
    command = env.command_manager.get_term(command_name)
    error = command.reference["foot_pos"][..., 2] - command.foot_positions()[..., 2]
    swing = ~command.reference["support_mask"].bool()
    reward = torch.exp(-error.square() / std**2) * swing
    return reward.sum(-1) / swing.sum(-1).clamp_min(1)


def loaded_swing_penalty(env, force_threshold: float = 10.0, command_name: str = "base_velocity"):
    """Fraction of scheduled swing feet with contact; zero in double support."""
    command = env.command_manager.get_term(command_name)
    swing = ~command.reference["support_mask"].bool()
    return (command.foot_contacts(force_threshold) & swing).sum(-1) / swing.sum(-1).clamp_min(1)


def stance_slide_penalty(env, force_threshold: float = 10.0, command_name: str = "base_velocity"):
    """Mean horizontal sole speed over planned stance feet, gated by actual contact."""
    command = env.command_manager.get_term(command_name)
    support = command.reference["support_mask"].bool()
    speed = command.foot_velocities()[..., :2].norm(dim=-1)
    return (speed * (support & command.foot_contacts(force_threshold))).sum(-1) / support.sum(-1).clamp_min(1)


def com_position_tracking(env, std: float = 0.10, command_name: str = "base_velocity"):
    _, com, _ = tracking_errors(env, command_name)
    return torch.exp(-com.square().sum(-1) / std**2)


def com_velocity_tracking(env, std: float = 0.30, command_name: str = "base_velocity"):
    _, _, vel = tracking_errors(env, command_name)
    return torch.exp(-vel.square().sum(-1) / std**2)


def support_tracking(
    env, force_threshold: float = 10.0, command_name: str = "base_velocity", sensor_name: str = "contact_forces"
):
    """Compare measured loading with planned one- or two-foot support.

    Both flags are true during startup; the reward encourages both feet to stay
    loaded while the CoM/sole rewards guide the weight transfer. This is a soft
    agreement term, not a prescribed contact-force distribution.
    """
    command = env.command_manager.get_term(command_name)
    sensor = env.scene[sensor_name]
    ids = [sensor.body_names.index(name) for name in command.cfg.foot_body_names]
    forces = sensor.data.net_forces_w
    forces = forces.torch if hasattr(forces, "torch") else forces
    contact = forces[:, ids].norm(dim=-1) > force_threshold
    return (contact == command.reference["support_mask"]).float().mean(-1)


def reference_finished(env, command_name: str = "base_velocity"):
    command = env.command_manager.get_term(command_name)
    return command.start_times + env.episode_length_buf * env.step_dt >= command.library.durations[command.clip_ids]
