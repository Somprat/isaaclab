from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_error_magnitude, yaw_quat
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster, FrameTransformer
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from typing import Optional, Sequence

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from general_low_body.mdp import ImpedanceCommand


"""
Velocity tracking
"""
def body_lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.body_lin_vel_w[:, asset_cfg.body_ids[0], 2])

def body_ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.body_ang_vel_w[:, asset_cfg.body_ids[0], :2]), dim=1)

def body_flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]
    body_projected_gravity = quat_apply_inverse(body_quat_w, asset.data.GRAVITY_VEC_W)
    return torch.sum(torch.square(body_projected_gravity[:, :2]), dim=1)

def track_lin_vel_xy_body_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]
    body_lin_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids[0], :]
    vel_yaw = quat_apply_inverse(yaw_quat(body_quat_w), body_lin_vel_w)
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_lin_vel_xy_yaw_frame_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of planar linear velocity in a gravity-aligned yaw-only frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_body_frame_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]
    body_ang_vel_w = asset.data.body_ang_vel_w[:, asset_cfg.body_ids[0], :]
    body_ang_vel_b = quat_apply_inverse(body_quat_w, body_ang_vel_w)
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - body_ang_vel_b[:, 2])
    return torch.exp(-ang_vel_error / std**2)

def track_ang_vel_z_body_world_frame_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_ang_vel_w = asset.data.body_ang_vel_w[:, asset_cfg.body_ids[0], :]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - body_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of yaw rate around the world vertical axis."""
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)

"""
Joint penalties.
"""


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def stand_still(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    reward = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return reward * (cmd_norm < 0.1)


"""
Robot.
"""


def orientation_l2(
    env: ManagerBasedRLEnv, desired_gravity: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward the agent for aligning its gravity with the desired gravity vector using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    desired_gravity = torch.tensor(desired_gravity, device=env.device)
    cos_dist = torch.sum(asset.data.projected_gravity_b * desired_gravity, dim=-1)  # cosine distance
    normalized = 0.5 * cos_dist + 0.5  # map from [-1, 1] to [0, 1]
    return torch.square(normalized)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)

def body_force(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 500, max_reward: float = 400
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    reward = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].norm(dim=-1)
    reward[reward < threshold] = 0
    reward[reward > threshold] -= threshold
    reward = reward.clamp(min=0, max=max_reward)
    return reward



"""
Feet rewards.
"""


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward

def feet_force_z(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 500, max_excess_force: float = 400) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    excess_force = torch.clamp(forces_z - threshold, min=0.0)
    clamped_excess = torch.clamp(excess_force, max=max_excess_force)
    return torch.sum(clamped_excess, dim=1)


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        )
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def feet_too_near(
    env: ManagerBasedRLEnv, threshold: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity"
) -> torch.Tensor:
    """
    Reward for feet contact when the command is zero.
    """
    # asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    reward = torch.sum(is_contact, dim=-1).float()
    return reward * (command_norm < 0.1)


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


def feet_sliding(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # NOTE: original code uses sensor_cfg.body_ids, but it should be asset_cfg.body_ids
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward

"""
Feet Gait rewards.
"""


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward


"""
Other rewards.
"""


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        reward += torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    return reward



def upper_body_action_rate(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    upper_action_dim = env.action_manager.get_term("UpperBodyAction").action_dim
    return torch.sum(torch.square(env.action_manager.action[:, :upper_action_dim] - env.action_manager.prev_action[:, :upper_action_dim]), dim=1)

def lower_body_action_rate(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    lower_action_dim = env.action_manager.get_term("LowerBodyAction").action_dim
    return torch.sum(torch.square(env.action_manager.action[:, -lower_action_dim:] - env.action_manager.prev_action[:, -lower_action_dim:]), dim=1)


def tracking_joint_pos_exp(
        env: ManagerBasedRLEnv, 
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
        sigma: float = 0.1
    ) -> torch.Tensor:
    """Reward tracking of joint positions using exponential kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_error = torch.sum(torch.square(asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1)
    return torch.exp(-joint_pos_error / sigma)

def tracking_joint_pos_l1_exp(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
    lambda_exp: float = 0.1
) -> torch.Tensor:
    """Reward tracking of joint positions using exponential kernel on L1 error."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_error = torch.sum(torch.abs(asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1)
    return torch.exp(-joint_pos_error * lambda_exp)

def torso_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_target_height = target_height + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    body_heights = asset.data.body_pos_w[:, asset_cfg.body_ids[0], 2]
    return torch.square(body_heights - adjusted_target_height)


"""
Plate Rewards
"""

def track_plate_pos_cmd_exp(
    env: ManagerBasedRLEnv,
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("plate_contact_sensor"),
    command_name: str = "plate_pose",
    sigma: float = 0.05,
    contact_threshold: float = 0.01
) -> torch.Tensor:
    """Reward tracking of plate positions using exponential kernel."""
    plate: RigidObject = env.scene[plate_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    pelvis_quat_w = robot.data.body_quat_w[:, robot_cfg.body_ids[0], :]
    pelvis_pos_w = robot.data.body_pos_w[:, robot_cfg.body_ids[0], :]
    plate_pos_pelvis = quat_apply_inverse(pelvis_quat_w, plate.data.root_pos_w - pelvis_pos_w)
    target_plate_pos_pelvis = env.command_manager.get_command(command_name)
    plate_pos_error = torch.sum(torch.square(plate_pos_pelvis - target_plate_pos_pelvis), dim=1)

    # contact mask
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > contact_threshold
    )
    zero_contact = (~contacts).all(dim=1)
    reward = torch.exp(-plate_pos_error / sigma) * ~zero_contact

    return reward



def penalty_frame_position_deviation_l2(
    env: ManagerBasedRLEnv,
    frame_cfg: SceneEntityCfg,
    target_name: str,
)-> torch.Tensor:
    """Penalize poistion deviation of the frame from the target frame."""
    transform: FrameTransformer = env.scene[frame_cfg.name]
    current_pos = torch.zeros(env.num_envs, 3, device=env.device)
    target_pos = torch.zeros(env.num_envs, 3, device=env.device)
    if frame_cfg.name == "left_tray_holder_transform":
        if target_name == "plate":
            current_pos[:, :] = transform.data.target_pos_source[:, env.plate2left_ee_idx, :]
            target_pos[:, :] = env.default_plate2left_ee_pos
        elif target_name == "right_tray_holder":
            current_pos[:, :] = transform.data.target_pos_source[:, env.right_ee2left_ee_idx, :]
            target_pos[:, :] = env.default_right_ee2left_ee_pos
    elif frame_cfg.name == "right_tray_holder_transform":
        if target_name == "plate":
            current_pos[:, :] = transform.data.target_pos_source[:, env.plate2right_ee_idx, :]
            target_pos[:, :] = env.default_plate2right_ee_pos
        elif target_name == "left_tray_holder":
            current_pos[:, :] = transform.data.target_pos_source[:, env.left_ee2right_ee_idx, :]
            target_pos[:, :] = env.default_left_ee2right_ee_pos
    elif frame_cfg.name == "plate_transform":
        if target_name == "left_tray_holder":
            current_pos[:, :] = transform.data.target_pos_source[:, env.left_ee2plate_idx, :]
            target_pos[:, :] = env.default_left_ee2plate_pos
        elif target_name == "right_tray_holder":
            current_pos[:, :] = transform.data.target_pos_source[:, env.right_ee2plate_idx, :]
            target_pos[:, :] = env.default_right_ee2plate_pos
        elif target_name == "object":
            current_pos[:, :] = transform.data.target_pos_source[:, env.object2plate_idx, :]
            target_pos[:, :] = env.default_object2plate_pos
    return torch.sum(torch.square(current_pos - target_pos), dim=1)

def track_plate_pos_exp(
    env: ManagerBasedRLEnv,
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link"),
    frame_cfg: SceneEntityCfg = SceneEntityCfg("plate_transform"),
    std: float = 0.05
)-> torch.Tensor:
    """Reward tracking of plate positions using exponential kernel."""
    transform: FrameTransformer = env.scene[frame_cfg.name]
    plate: RigidObject = env.scene[plate_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    # left ee to plate
    left2plate_current_pos = transform.data.target_pos_source[:, env.left_ee2plate_idx, :]
    left2plate_target_pos = env.default_left_ee2plate_pos
    left2plate_pos_error = torch.sum(torch.square(left2plate_current_pos - left2plate_target_pos), dim=1)
    left2plate_reward = torch.exp(-left2plate_pos_error / (std**2))

    # right ee to plate
    right2plate_current_pos = transform.data.target_pos_source[:, env.right_ee2plate_idx, :]
    right2plate_target_pos = env.default_right_ee2plate_pos
    right2plate_pos_error = torch.sum(torch.square(right2plate_current_pos - right2plate_target_pos), dim=1)
    right2plate_reward = torch.exp(-right2plate_pos_error / (std**2))

    # height gate
    gate = plate.data.root_pos_w[:, 2] >= robot.data.body_pos_w[:, robot_cfg.body_ids[0], 2]
    return (left2plate_reward + right2plate_reward) * gate


def track_plate_orientation_exp(
    env: ManagerBasedRLEnv,
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link"),
    frame_cfg: SceneEntityCfg = SceneEntityCfg("plate_transform"),
    std: float = 0.05
)-> torch.Tensor:
    """Reward tracking of plate orientation using exponential kernel."""
    transform: FrameTransformer = env.scene[frame_cfg.name]
    plate: RigidObject = env.scene[plate_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    # left ee to plate
    left2plate_current_ori = transform.data.target_quat_source[:, env.left_ee2plate_idx, :]
    left2plate_target_ori = env.default_left_ee2plate_quat
    left2plate_ori_error = quat_error_magnitude(left2plate_current_ori, left2plate_target_ori)
    left2plate_reward = torch.exp(-left2plate_ori_error / (std**2))

    # right ee to plate
    right2plate_current_ori = transform.data.target_quat_source[:, env.right_ee2plate_idx, :]
    right2plate_target_ori = env.default_right_ee2plate_quat
    right2plate_ori_error = quat_error_magnitude(right2plate_current_ori, right2plate_target_ori)
    right2plate_reward = torch.exp(-right2plate_ori_error / (std**2))

    # height gate
    gate = plate.data.root_pos_w[:, 2] >= robot.data.body_pos_w[:, robot_cfg.body_ids[0], 2]
    return (left2plate_reward + right2plate_reward) * gate



def track_object_pos_tanh(
    env: ManagerBasedRLEnv,
    frame_cfg: SceneEntityCfg = SceneEntityCfg('plate_transform'),
    std: float = 0.05
) -> torch.Tensor:
    """Reward tracking of commanded position using tanh kernel, gated by contact presence."""
    transform: FrameTransformer = env.scene[frame_cfg.name]
    current_pos = transform.data.target_pos_source[:, env.object2plate_idx, :]
    target_pos = env.default_object2plate_pos
    distance = torch.norm(current_pos - target_pos, dim=1) 
    return (1 - torch.tanh(distance / std))


def object_upright_tanh(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    std: float = 0.05
) -> torch.Tensor:
    asset: RigidObject = env.scene[object_cfg.name]
    tilt_magnitude = torch.norm(asset.data.projected_gravity_b[:, :2], dim=1)
    return (1 - torch.tanh(tilt_magnitude / std))


def penalty_frame_quat_deviation(
    env: ManagerBasedRLEnv,
    frame_cfg: SceneEntityCfg,
    target_name: str,
)-> torch.Tensor:
    """Penalize quaternion deviation of the frame from the target frame."""
    transform: FrameTransformer = env.scene[frame_cfg.name]
    current_quat = torch.zeros(env.num_envs, 4, device=env.device)
    target_quat = torch.zeros(env.num_envs, 4, device=env.device)
    if frame_cfg.name == "left_tray_holder_transform":
        if target_name == "plate":
            current_quat[:, :] = transform.data.target_quat_source[:, env.plate2left_ee_idx, :]
            target_quat[:, :] = env.default_plate2left_ee_quat
        elif target_name == "right_tray_holder":
            current_quat[:, :] = transform.data.target_quat_source[:, env.right_ee2left_ee_idx, :]
            target_quat[:, :] = env.default_right_ee2left_ee_quat
    elif frame_cfg.name == "right_tray_holder_transform":
        if target_name == "plate":
            current_quat[:, :] = transform.data.target_quat_source[:, env.plate2right_ee_idx, :]
            target_quat[:, :] = env.default_plate2right_ee_quat
        elif target_name == "left_tray_holder":
            current_quat[:, :] = transform.data.target_quat_source[:, env.left_ee2right_ee_idx, :]
            target_quat[:, :] = env.default_left_ee2right_ee_quat
    elif frame_cfg.name == "plate_transform":
        if target_name == "left_tray_holder":
            current_quat[:, :] = transform.data.target_quat_source[:, env.left_ee2plate_idx, :]
            target_quat[:, :] = env.default_left_ee2plate_quat
        elif target_name == "right_tray_holder":
            current_quat[:, :] = transform.data.target_quat_source[:, env.right_ee2plate_idx, :]
            target_quat[:, :] = env.default_right_ee2plate_quat
    return quat_error_magnitude(current_quat, target_quat)


def desired_contacts(env, sensor_cfg: SceneEntityCfg, contact_threshold: float = 0.01) -> torch.Tensor:
    """Penalize if none of the desired contacts are present."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > contact_threshold
    )
    zero_contact = (~contacts).all(dim=1)
    return ~zero_contact

def penalty_asset_projected_gravity(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("plate"), 
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("plate_contact_sensor"), 
    contact_threshold: float = 0.01
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > contact_threshold
    )
    zero_contact = (~contacts).all(dim=1)
    gate = ~zero_contact
    if sensor_cfg.name == "object_contact_sensor":
        plate: RigidObject = env.scene['plate']
        gate = torch.logical_and(gate, asset.data.root_pos_w[:, 2] > plate.data.root_pos_w[:,2])
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1) * gate

def alive_termination_term(env: ManagerBasedRLEnv, term_name: str) -> torch.Tensor:
    """Penalize the robot for falling."""
    return (~env.termination_manager.get_term(term_name)).float()



def residual_upper_body_action_rate(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    upper_action_dim = env.action_manager.get_term("UpperBodyAction").action_dim
    return torch.sum(torch.square(env.action_manager.residual_action[:, :upper_action_dim] - env.action_manager.prev_residual_action[:, :upper_action_dim]), dim=1)

def residual_lower_body_action_rate(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    upper_action_dim = env.action_manager.get_term("UpperBodyAction").action_dim
    return torch.sum(torch.square(env.action_manager.residual_action[:, upper_action_dim:] - env.action_manager.prev_residual_action[:, upper_action_dim:]), dim=1)

def residual_whole_body_action_rate(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.sum(torch.square(env.action_manager.residual_action - env.action_manager.prev_residual_action), dim=1)


def reward_residual_upper_body_action(env: ManagerBasedRLEnv, std: float=0.1) -> torch.Tensor:
    upper_action_dim = env.action_manager.get_term("UpperBodyAction").action_dim
    residual_upper_action = env.action_manager.residual_action[:, :upper_action_dim]
    total_action = torch.sum(torch.square(residual_upper_action), dim=-1)
    return torch.exp(- total_action / (std ** 2))

def reward_residual_lower_body_action(env: ManagerBasedRLEnv, std: float=0.1) -> torch.Tensor:
    upper_action_dim = env.action_manager.get_term("UpperBodyAction").action_dim
    residual_lower_action = env.action_manager.residual_action[:, upper_action_dim:]
    total_action = torch.sum(torch.square(residual_lower_action), dim=-1)
    return torch.exp(- total_action / (std ** 2))


def asset_lin_acc_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("object_contact_sensor"),
    contact_threshold: float = 0.01,
) -> torch.Tensor:
    """Penalize the asset for having linear acceleration."""
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > contact_threshold
    )
    zero_contact = (~contacts).all(dim=1)
    gate = ~zero_contact
    return torch.sum(torch.square(asset.data.body_lin_acc_w[:, 0, :]), dim=1) * gate


def asset_ang_acc_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("object_contact_sensor"),
    contact_threshold: float = 0.01,
) -> torch.Tensor:
    """Penalize the asset for having angular acceleration."""
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > contact_threshold
    )
    zero_contact = (~contacts).all(dim=1)
    gate = ~zero_contact
    return torch.sum(torch.square(asset.data.body_ang_acc_w[:, 0, :]), dim=1) * gate

def track_object_pos_exp(
    env: ManagerBasedRLEnv,
    frame_cfg: SceneEntityCfg = SceneEntityCfg("plate_transform"),
    std: float = 0.1,
) -> torch.Tensor:
    """Reward tracking of commanded position using exponential kernel."""
    transform: FrameTransformer = env.scene[frame_cfg.name]
    current_pos = transform.data.target_pos_source[:, env.object2plate_idx, :]
    target_pos = env.default_object2plate_pos
    distance = torch.sum(torch.square(current_pos - target_pos), dim=1)
    return torch.exp(-distance / (std**2))

def asset_upright_exp(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    std: float = 0.1
) -> torch.Tensor:
    """Penalize the asset for being not upright."""
    asset: RigidObject = env.scene[asset_cfg.name]
    tilt_magnitude = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    return torch.exp(-tilt_magnitude / (std**2))

def asset_upright_exp_gate(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link"),
    limit_angle: float | None = None,
    std: float = 0.1
) -> torch.Tensor:
    """Penalize the asset for being not upright."""
    asset: RigidObject = env.scene[asset_cfg.name]
    tilt_magnitude = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    reward = torch.exp(-tilt_magnitude / (std**2))

    # add gate
    robot: Articulation = env.scene[robot_cfg.name]
    if asset_cfg.name == "plate":
        plate_dropped = asset.data.root_pos_w[:, 2] < robot.data.body_pos_w[:, robot_cfg.body_ids[0], 2]
        gate = ~plate_dropped
    else:
        plate: RigidObject = env.scene["plate"]
        plate_dropped = plate.data.root_pos_w[:, 2] < robot.data.body_pos_w[:, robot_cfg.body_ids[0], 2]
        object_dropped = asset.data.root_pos_w[:, 2] < plate.data.root_pos_w[:, 2]
        object_bad_orientation = torch.acos(-asset.data.projected_gravity_b[:, 2]).abs() > limit_angle
        gate = ~plate_dropped & ~object_dropped & ~object_bad_orientation
    
    return reward * gate


def asset_lin_acc_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    std: float = 0.1,
) -> torch.Tensor:
    """Penalize the asset for having linear acceleration."""
    asset: RigidObject = env.scene[asset_cfg.name]
    acc_magnitude = torch.sum(torch.square(asset.data.body_lin_acc_w[:, 0, :]), dim=1)
    return torch.exp(-acc_magnitude / (std**2))

def asset_ang_acc_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    std: float = 0.1,
) -> torch.Tensor:
    """Penalize the asset for having angular acceleration."""
    asset: RigidObject = env.scene[asset_cfg.name]
    acc_magnitude = torch.sum(torch.square(asset.data.body_ang_acc_w[:, 0, :]), dim=1)
    return torch.exp(-acc_magnitude / (std**2))



def track_ee_position_exp(
    env: ManagerBasedRLEnv,
    frame_cfg: SceneEntityCfg = SceneEntityCfg('robot_base_transform'),
    target_name: str = 'left_ee',
    std: float = 0.1
) -> torch.Tensor:
    """Reward for tracking end-effector position."""
    transform: FrameTransformer = env.scene[frame_cfg.name]
    
    if target_name == "left_ee":
        current_pos = transform.data.target_pos_source[:, env.robot2left_ee_idx, :]
        target_pos = env.default_robot2left_ee_pos
    elif target_name == "right_ee":
        current_pos = transform.data.target_pos_source[:, env.robot2right_ee_idx, :]
        target_pos = env.default_robot2right_ee_pos
    else:
        raise ValueError(f"Unknown target_name: {target_name}")

    error = torch.sum(torch.square(current_pos - target_pos), dim=1)
    return torch.exp(-error / (std**2))


def track_ee_quat_exp(
    env: ManagerBasedRLEnv,
    frame_cfg: SceneEntityCfg = SceneEntityCfg('robot_base_transform'),
    target_name: str = 'left_ee',
    std: float = 0.5
) -> torch.Tensor:
    """Reward for tracking end-effector orientation."""
    transform: FrameTransformer = env.scene[frame_cfg.name]
    
    if target_name == "left_ee":
        current_quat = transform.data.target_quat_source[:, env.robot2left_ee_idx, :]
        target_quat = env.default_robot2left_ee_quat
    elif target_name == "right_ee":
        current_quat = transform.data.target_quat_source[:, env.robot2right_ee_idx, :]
        target_quat = env.default_robot2right_ee_quat
    else:
        raise ValueError(f"Unknown target_name: {target_name}")
    
    error = quat_error_magnitude(current_quat, target_quat)**2
    return torch.exp(-error / (std**2)) 

def penalty_ee_quat_l2(
    env: ManagerBasedRLEnv,
    frame_cfg: SceneEntityCfg = SceneEntityCfg('robot_base_transform'),
    target_name: str = 'left_ee',
) -> torch.Tensor:
    """Reward for tracking end-effector orientation."""
    transform: FrameTransformer = env.scene[frame_cfg.name]
    
    if target_name == "left_ee":
        current_quat = transform.data.target_quat_source[:, env.robot2left_ee_idx, :]
        target_quat = env.default_robot2left_ee_quat
    elif target_name == "right_ee":
        current_quat = transform.data.target_quat_source[:, env.robot2right_ee_idx, :]
        target_quat = env.default_robot2right_ee_quat
    else:
        raise ValueError(f"Unknown target_name: {target_name}")
    
    return quat_error_magnitude(current_quat, target_quat)


def penlaty_body_lin_acc_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
)-> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name] 
    return torch.sum(torch.square(asset.data.body_lin_acc_w[:, asset_cfg.body_ids[0], :]), dim=1)


def penlaty_body_ang_acc_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
)-> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name] 
    return torch.sum(torch.square(asset.data.body_ang_acc_w[:, asset_cfg.body_ids[0], :]), dim=1)

def reward_body_lin_acc_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 3.0
)-> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name] 
    error = torch.sum(torch.square(asset.data.body_lin_acc_w[:, asset_cfg.body_ids[0], :]), dim=1)
    return torch.exp(-error / (std**2))

def reward_body_ang_acc_exp (
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 10.0
)-> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name] 
    error = torch.sum(torch.square(asset.data.body_ang_acc_w[:, asset_cfg.body_ids[0], :]), dim=1)
    return torch.exp(-error / (std**2))


def reward_object_lin_vel_in_plate_frame_exp(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
    std: float = 0.1,
) -> torch.Tensor:
    """Penalize the object for having linear velocity in the plate frame."""
    object: RigidObject = env.scene[object_cfg.name]
    plate: RigidObject = env.scene[plate_cfg.name]
    lin_vel_in_plate = quat_apply_inverse(plate.data.root_quat_w, object.data.root_lin_vel_w - plate.data.root_lin_vel_w)
    error = torch.sum(torch.square(lin_vel_in_plate), dim=1)
    return torch.exp(-error / (std**2))



def reward_object_lin_vel_in_plate_frame_exp_gate(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link"),
    limit_angle: float = 0.9,
    std: float = 0.1,
) -> torch.Tensor:
    """Penalize the object for having linear velocity in the plate frame."""
    object: RigidObject = env.scene[object_cfg.name]
    plate: RigidObject = env.scene[plate_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    lin_vel_in_plate = quat_apply_inverse(plate.data.root_quat_w, object.data.root_lin_vel_w - plate.data.root_lin_vel_w)
    error = torch.sum(torch.square(lin_vel_in_plate), dim=1)

    # gate
    plate_dropped = plate.data.root_pos_w[:, 2] < robot.data.body_pos_w[:, robot_cfg.body_ids[0], 2]
    object_dropped = object.data.root_pos_w[:, 2] < plate.data.root_pos_w[:, 2]
    object_bad_orientation = torch.acos(-object.data.projected_gravity_b[:, 2]).abs() > limit_angle
    gate = ~plate_dropped & ~object_dropped & ~object_bad_orientation

    return torch.exp(-error / (std**2)) * gate


def reward_object_ang_vel_in_plate_frame_exp(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
    std: float = 0.1,
) -> torch.Tensor:
    """Penalize the object for having angular velocity in the plate frame."""
    object: RigidObject = env.scene[object_cfg.name]
    plate: RigidObject = env.scene[plate_cfg.name]
    ang_vel_in_plate_frame = quat_apply_inverse(plate.data.root_quat_w, object.data.root_ang_vel_w - plate.data.root_ang_vel_w)
    error = torch.sum(torch.square(ang_vel_in_plate_frame), dim=1)
    return torch.exp(-error / (std**2))

def reward_object_ang_vel_in_plate_frame_exp_gate(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link"),
    limit_angle: float = 0.9,
    std: float = 0.1,
) -> torch.Tensor:
    """Penalize the object for having angular velocity in the plate frame."""
    object: RigidObject = env.scene[object_cfg.name]
    plate: RigidObject = env.scene[plate_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    ang_vel_in_plate_frame = quat_apply_inverse(plate.data.root_quat_w, object.data.root_ang_vel_w - plate.data.root_ang_vel_w)
    error = torch.sum(torch.square(ang_vel_in_plate_frame), dim=1)

    # gate
    plate_dropped = plate.data.root_pos_w[:, 2] < robot.data.body_pos_w[:, robot_cfg.body_ids[0], 2]
    object_dropped = object.data.root_pos_w[:, 2] < plate.data.root_pos_w[:, 2]
    object_bad_orientation = torch.acos(-object.data.projected_gravity_b[:, 2]).abs() > limit_angle
    gate = ~plate_dropped & ~object_dropped & ~object_bad_orientation

    return torch.exp(-error / (std**2)) * gate



def reward_quat_diff_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lambda_exp: float = 4.0,
) -> torch.Tensor:
    """Reward for the difference in orientation between the object and the robot."""
    object: RigidObject = env.scene[asset_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    object_quat = object.data.body_quat_w[:, 0, :]  # Shape: [num_envs, 4] - (w, x, y, z)
    robot_quat = robot.data.body_quat_w[:, robot_cfg.body_ids[0], :]  # Shape: [num_envs, 4] - (w, x, y, z)
    quat_dot = torch.sum(object_quat * robot_quat, dim=-1)  # Shape: [num_envs]
    quat_dot = torch.clamp(torch.abs(quat_dot), max=1.0)  # Clamp to avoid numerical issues
    orientation_difference = 2.0 * torch.acos(quat_dot)  # Angular distance in radians
    
    return torch.exp(-lambda_exp * orientation_difference)

def reward_quat_error_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.5,
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    asset_quat = asset.data.root_quat_w
    robot_quat = robot.data.body_quat_w[:, robot_cfg.body_ids[0], :]
    error = torch.square(quat_error_magnitude(asset_quat, robot_quat))
    return torch.exp(-error/ (std ** 2))

def reward_frame_position_exp(
    env: ManagerBasedRLEnv,
    frame_cfg: SceneEntityCfg,
    target_name: str,
    std: float = 0.1
)-> torch.Tensor:
    """Penalize poistion deviation of the frame from the target frame."""
    transform: FrameTransformer = env.scene[frame_cfg.name]
    current_pos = torch.zeros(env.num_envs, 3, device=env.device)
    target_pos = torch.zeros(env.num_envs, 3, device=env.device)
    if frame_cfg.name == "left_tray_holder_transform":
        if target_name == "plate":
            current_pos[:, :] = transform.data.target_pos_source[:, env.plate2left_ee_idx, :]
            target_pos[:, :] = env.default_plate2left_ee_pos
        elif target_name == "right_tray_holder":
            current_pos[:, :] = transform.data.target_pos_source[:, env.right_ee2left_ee_idx, :]
            target_pos[:, :] = env.default_right_ee2left_ee_pos
    elif frame_cfg.name == "right_tray_holder_transform":
        if target_name == "plate":
            current_pos[:, :] = transform.data.target_pos_source[:, env.plate2right_ee_idx, :]
            target_pos[:, :] = env.default_plate2right_ee_pos
        elif target_name == "left_tray_holder":
            current_pos[:, :] = transform.data.target_pos_source[:, env.left_ee2right_ee_idx, :]
            target_pos[:, :] = env.default_left_ee2right_ee_pos
    elif frame_cfg.name == "plate_transform":
        if target_name == "left_tray_holder":
            current_pos[:, :] = transform.data.target_pos_source[:, env.left_ee2plate_idx, :]
            target_pos[:, :] = env.default_left_ee2plate_pos
        elif target_name == "right_tray_holder":
            current_pos[:, :] = transform.data.target_pos_source[:, env.right_ee2plate_idx, :]
            target_pos[:, :] = env.default_right_ee2plate_pos
        elif target_name == "object":
            current_pos[:, :] = transform.data.target_pos_source[:, env.object2plate_idx, :]
            target_pos[:, :] = env.default_object2plate_pos
    error = torch.sum(torch.square(current_pos - target_pos), dim=1)
    return torch.exp(-error / (std**2))

def reward_frame_quat_exp(
    env: ManagerBasedRLEnv,
    frame_cfg: SceneEntityCfg,
    target_name: str,
    sigma: float = 0.1
)-> torch.Tensor:
    """Penalize quaternion deviation of the frame from the target frame."""
    transform: FrameTransformer = env.scene[frame_cfg.name]
    current_quat = torch.zeros(env.num_envs, 4, device=env.device)
    target_quat = torch.zeros(env.num_envs, 4, device=env.device)
    if frame_cfg.name == "left_tray_holder_transform":
        if target_name == "plate":
            current_quat[:, :] = transform.data.target_quat_source[:, env.plate2left_ee_idx, :]
            target_quat[:, :] = env.default_plate2left_ee_quat
        elif target_name == "right_tray_holder":
            current_quat[:, :] = transform.data.target_quat_source[:, env.right_ee2left_ee_idx, :]
            target_quat[:, :] = env.default_right_ee2left_ee_quat
    elif frame_cfg.name == "right_tray_holder_transform":
        if target_name == "plate":
            current_quat[:, :] = transform.data.target_quat_source[:, env.plate2right_ee_idx, :]
            target_quat[:, :] = env.default_plate2right_ee_quat
        elif target_name == "left_tray_holder":
            current_quat[:, :] = transform.data.target_quat_source[:, env.left_ee2right_ee_idx, :]
            target_quat[:, :] = env.default_left_ee2right_ee_quat
    elif frame_cfg.name == "plate_transform":
        if target_name == "left_tray_holder":
            current_quat[:, :] = transform.data.target_quat_source[:, env.left_ee2plate_idx, :]
            target_quat[:, :] = env.default_left_ee2plate_quat
        elif target_name == "right_tray_holder":
            current_quat[:, :] = transform.data.target_quat_source[:, env.right_ee2plate_idx, :]
            target_quat[:, :] = env.default_right_ee2plate_quat
    quat_error = quat_error_magnitude(current_quat, target_quat)
    return torch.exp(-quat_error / sigma)


class FootImpactReward(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot_cfg = cfg.params.get("robot_cfg", SceneEntityCfg("robot"))
        self.prev_foot_vel_z = torch.zeros(self.num_envs, len(self.robot_cfg.body_ids), device=self.device)
        self.lambda_exp = cfg.params.get("lambda_exp", 2.0)
        self.robot: Articulation = env.scene[self.robot_cfg.name]

    def __call__(
        self, 
        env: ManagerBasedRLEnv, 
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        lambda_exp: float = 2.0
    ) -> torch.Tensor:
       foot_vel_z = self.robot.data.body_lin_vel_w[:, self.robot_cfg.body_ids, 2]
       delta_vel_z = torch.abs(foot_vel_z - self.prev_foot_vel_z)
       total_dv = torch.sum(delta_vel_z, dim=1)
       self.prev_foot_vel_z = foot_vel_z.clone()
       return torch.exp(-total_dv * self.lambda_exp)

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            self.prev_foot_vel_z[:] = 0.0
        else:
            self.prev_foot_vel_z[env_ids] = 0.0


class FootImpactL2Reward(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot_cfg = cfg.params.get("robot_cfg", SceneEntityCfg("robot"))
        self.prev_foot_vel_z = torch.zeros(self.num_envs, len(self.robot_cfg.body_ids), device=self.device)
        self.std = cfg.params.get("std", 0.5)
        self.robot: Articulation = env.scene[self.robot_cfg.name]

    def __call__(
        self, 
        env: ManagerBasedRLEnv, 
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        std: float = 0.5
    ) -> torch.Tensor:
       foot_vel_z = self.robot.data.body_lin_vel_w[:, self.robot_cfg.body_ids, 2]
       total_dv = torch.sum(torch.square(foot_vel_z - self.prev_foot_vel_z), dim=1)
       self.prev_foot_vel_z = foot_vel_z.clone()
       return torch.exp(-total_dv / (self.std**2))

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            self.prev_foot_vel_z[:] = 0.0
        else:
            self.prev_foot_vel_z[env_ids] = 0.0

class AssetDeltaVelReward(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("object"))
        self.vel_type = cfg.params.get("vel_type", "lin") # vel_type can be "lin" or "ang"
        self.prev_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.std = cfg.params.get("std", 1.0)
        self.asset: RigidObject = env.scene[self.asset_cfg.name]

    def __call__(
        self, 
        env: ManagerBasedRLEnv, 
        asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        vel_type: str = "lin",
        std: float = 1.0
    ) -> torch.Tensor:
        if self.vel_type == "lin":
            vel = self.asset.data.root_lin_vel_w
        else:
            vel = self.asset.data.root_ang_vel_w
        total_dv = torch.sum(torch.square(vel - self.prev_vel), dim=1)
        self.prev_vel = vel.clone()
        return torch.exp(-total_dv / (self.std**2))
    
    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            self.prev_vel[:, :] = 0.0
        else:
            self.prev_vel[env_ids, :] = 0.0

class AssetDeltaVelRewardGate(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("object"))
        self.plate_cfg = cfg.params.get("plate_cfg", SceneEntityCfg("plate"))
        self.robot_cfg = cfg.params.get("robot_cfg", SceneEntityCfg("robot", body_names="torso_link"))
        self.vel_type = cfg.params.get("vel_type", "lin") # vel_type can be "lin" or "ang"
        self.prev_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.limit_angle = cfg.params.get("limit_angle", 0.9)
        self.std = cfg.params.get("std", 1.0)
        self.asset: RigidObject = env.scene[self.asset_cfg.name]
        self.plate: RigidObject = env.scene[self.plate_cfg.name]
        self.robot: Articulation = env.scene[self.robot_cfg.name]

    def __call__(
        self, 
        env: ManagerBasedRLEnv, 
        asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link"),
        vel_type: str = "lin",
        limit_angle: float = 0.9,
        std: float = 1.0
    ) -> torch.Tensor:
        if self.vel_type == "lin":
            vel = self.asset.data.root_lin_vel_w
        else:
            vel = self.asset.data.root_ang_vel_w
        total_dv = torch.sum(torch.square(vel - self.prev_vel), dim=1)
        self.prev_vel = vel.clone()

        # gate
        plate_dropped = self.plate.data.root_pos_w[:, 2] < self.robot.data.body_pos_w[:, self.robot_cfg.body_ids[0], 2]
        object_dropped = self.asset.data.root_pos_w[:, 2] < self.plate.data.root_pos_w[:, 2]
        object_bad_orientation = torch.acos(-self.asset.data.projected_gravity_b[:, 2]).abs() > self.limit_angle
        gate = ~plate_dropped & ~object_dropped & ~object_bad_orientation

        return torch.exp(-total_dv / (self.std**2)) * gate
    
    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            self.prev_vel[:, :] = 0.0
        else:
            self.prev_vel[env_ids, :] = 0.0


def reward_plate_alive(
    env: ManagerBasedRLEnv,
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link"),
) -> torch.Tensor:
    """Reward for keeping the plate alive (not fallen)."""
    plate: RigidObject = env.scene[plate_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    return (plate.data.root_pos_w[:, 2] >= robot.data.body_pos_w[:, robot_cfg.body_ids[0], 2]).float()

def reward_plate_in_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("plate_contact_sensor"),
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
    contact_threshold: float = 0.01,
    force_multiplier: float = 1.0,
    force_excessive_lambda: float = 1.0,
    force_diff_lambda: float = 1.0,
) -> torch.Tensor:
    """Reward for keeping the plate in contact with the robot."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    plate: RigidObject = env.scene[plate_cfg.name]
    left_ee_vec = contact_sensor.data.force_matrix_w[:, 0, 0, :] # (N, 3)
    right_ee_vec = contact_sensor.data.force_matrix_w[:, 0, 1, :] # (N, 3)
    device = left_ee_vec.device
    left_force = torch.sqrt(torch.sum(torch.square(left_ee_vec), dim=-1) + 1e-6) # (N,)
    right_force = torch.sqrt(torch.sum(torch.square(right_ee_vec), dim=-1) + 1e-6) # (N,)

    # in contact
    reward_in_contact = torch.tanh(left_force / contact_threshold) * torch.tanh(right_force / contact_threshold)

    # excessive force penalty
    plate_mass = plate.data._root_physx_view.get_masses().squeeze().to(device) # (N,)
    max_force = plate_mass * 9.81 * force_multiplier
    total_force = left_force + right_force
    reward_not_excessive = torch.sigmoid((max_force - total_force) / force_excessive_lambda)

    # balance contact force
    diff_force = torch.square(left_force - right_force)
    reward_force_diff = torch.exp(-diff_force / (force_diff_lambda ** 2))

    return (reward_in_contact * reward_not_excessive * reward_force_diff)


def reward_object_alive(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
    limit_angle: float = 0.9
)-> torch.Tensor:
    """Reward for keeping the object on the plate."""
    object: RigidObject = env.scene[object_cfg.name]
    plate: RigidObject = env.scene[plate_cfg.name]
    object_on_plate = object.data.root_pos_w[:, 2] >= plate.data.root_pos_w[:, 2]
    object_upright = torch.acos(-object.data.projected_gravity_b[:, 2]).abs() < limit_angle
    return (object_on_plate & object_upright).float()

def penalty_object_bad_state(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
    limit_angle: float = 0.9
) -> torch.Tensor:
    """Penalty for the object being in a bad state (fallen or dropped)."""
    object: RigidObject = env.scene[object_cfg.name]
    plate: RigidObject = env.scene[plate_cfg.name]
    object_on_plate = object.data.root_pos_w[:, 2] >= plate.data.root_pos_w[:, 2]
    object_upright = torch.acos(-object.data.projected_gravity_b[:, 2]).abs() < limit_angle
    return (~(object_on_plate & object_upright)).float()

def penalty_plate_dropped(
    env: ManagerBasedRLEnv,
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link"),
) -> torch.Tensor:
    """Penalty for the plate being dropped."""
    plate: RigidObject = env.scene[plate_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    return (plate.data.root_pos_w[:, 2] < robot.data.body_pos_w[:, robot_cfg.body_ids[0], 2]).float()

def asset_ang_vel_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    std: float = 0.5
)-> torch.Tensor:
    """Penalize the asset for having angular velocity."""
    asset: RigidObject = env.scene[asset_cfg.name]
    ang_vel_magnitude = torch.sum(torch.square(asset.data.root_ang_vel_w), dim=1)
    reward = torch.exp(-ang_vel_magnitude / (std**2))
    return reward


def reward_plate_desire_contact(
    env: ManagerBasedRLEnv,
    plate_sensor_cfg: SceneEntityCfg = SceneEntityCfg("plate_contact_sensor", body_names=".*tray_holder.*"),
    contact_threshold: float = 0.01,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene[plate_sensor_cfg.name]
    # force_matrix_w_history: (N, T, B, M, 3)
    left_contact_normal = contact_sensor.data.force_matrix_w_history[:, :, 0, 0, :]
    right_contact_normal = contact_sensor.data.force_matrix_w_history[:, :, 0, 1, :]
    # Both: (N, T, 3)
    left_force_mag = left_contact_normal.norm(dim=-1)
    right_force_mag = right_contact_normal.norm(dim=-1)
    left_max_force = left_force_mag.max(dim=1)[0] 
    right_max_force = right_force_mag.max(dim=1)[0]
    left_contact = left_max_force > contact_threshold
    right_contact = right_max_force > contact_threshold
    
    in_contact = left_contact & right_contact
    return in_contact


def contact_force_exp(
    env: ManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg, 
    lambda_exp: float = 0.01
) -> torch.Tensor:
    """Exponential reward for minimizing contact forces (encourages gentle contact).
    
    This function rewards low contact forces using an exponential kernel. Higher contact
    forces result in lower rewards, encouraging the agent to maintain gentle contact.
    The reward approaches 1.0 when contact forces are minimal and decreases exponentially
    as forces increase.
    
    Args:
        env: The environment instance.
        sensor_cfg: Configuration for the contact sensor. The body_ids should specify which bodies to check.
                   Requires filter_prim_paths_expr to be configured in the sensor.
        lambda_exp: Lambda parameter for the exponential kernel. Higher values make the 
                   reward more sensitive to contact forces. Default is 0.01.
                   Typical range: 0.001 - 0.1 depending on expected force magnitudes.

    Returns:
        Reward tensor of shape (num_envs,) with values in range (0, 1], where 1 is 
        minimal contact force and lower values indicate higher forces.
        For example, with lambda_exp=0.01:
        - Force of 0 N -> reward = 1.0
        - Force of 10 N -> reward ≈ 0.90
        - Force of 50 N -> reward ≈ 0.61
        - Force of 100 N -> reward ≈ 0.37
    
    Note:
        This uses the most recent filtered contact forces (not history).
        Uses force_matrix_w which contains forces between the sensor body and 
        specified filter bodies (e.g., tray and tray holders).
        The sensor must be configured with filter_prim_paths_expr.
    """
    # Get contact sensor data
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    
    # Use filtered force matrix (requires filter_prim_paths_expr configured in sensor)
    # Get current filtered contact forces for specified bodies
    # Shape: [num_envs, num_bodies, num_filtered_bodies, 3]
    contact_forces = contact_sensor.data.force_matrix_w[:, sensor_cfg.body_ids, :, :]
    
    # Compute the L2 norm of forces for each (body, filtered_body) pair
    # Shape: [num_envs, num_bodies, num_filtered_bodies]
    force_magnitudes = torch.norm(contact_forces, dim=-1)
    
    # Sum forces across all bodies and filtered bodies to get total contact force magnitude
    # Shape: [num_envs]
    total_force = force_magnitudes.sum(dim=(1, 2))
    
    # Apply exponential kernel: lower forces -> higher reward
    return torch.exp(-lambda_exp * total_force)


def feet_soft_landing(
    env: ManagerBasedRLEnv,
    sensor_cfg:  SceneEntityCfg = SceneEntityCfg("ropbot_contact_sensor", body_names=".*ankle_roll.*"),
    command_name: str = "base_velocity",
    command_threshold: float = 0.1,
) -> torch.Tensor:
    # penalize high impact forces on the feet when landing
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)
    force_magnitude = torch.norm(contact_sensor.data.net_forces_w, dim=-1)
    landing_impact = force_magnitude * first_contact.float()
    cost = torch.sum(landing_impact, dim=1)

    # condition on command
    command = env.command_manager.get_command(command_name)
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()
    cost = cost * active
    return cost


def penalty_ee_position_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="left_wrist_yaw_link"),
    command_name: str = "left_ee_position"
) -> torch.Tensor:
    """Penalize the end-effector for deviating from the commanded position."""
    robot: Articulation = env.scene[asset_cfg.name]
    ee_cur_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids[0], :]
    ee_target_pos_b = env.command_manager.get_command(command_name)
    ee_target_pos_w = quat_apply(robot.data.root_quat_w, ee_target_pos_b) + robot.data.root_pos_w
    return torch.sum(torch.square(ee_cur_pos_w - ee_target_pos_w), dim=1)

def reward_ee_position_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="left_wrist_yaw_link"),
    command_name: str = "left_ee_position",
    std: float = 0.1
) -> torch.Tensor:
    """Reward the end-effector for tracking the commanded position."""
    robot: Articulation = env.scene[asset_cfg.name]
    ee_cur_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids[0], :]
    ee_target_pos_b = env.command_manager.get_command(command_name)
    ee_target_pos_w = quat_apply(robot.data.root_quat_w, ee_target_pos_b) + robot.data.root_pos_w
    error = torch.sum(torch.square(ee_cur_pos_w - ee_target_pos_w), dim=1)
    return torch.exp(-error / (std**2))

    
def penalty_ee_reference_position_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="left_wrist_yaw_link"),
    command_name: str = "left_ee_position",
) -> torch.Tensor:
    """Penalize the end-effector for deviating from the commanded position."""
    command: ImpedanceCommand = env.command_manager.get_term(command_name)
    mode = command.integrate_method
    robot: Articulation = env.scene[asset_cfg.name]
    ee_cur_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids[0], :]
    if mode == "closed_loop":
        ee_ref_pos_w = command.reference_position_closedloop_w
    elif mode == "open_loop":
        ee_ref_pos_w = command.reference_position_openloop_w
    else:
        ee_ref_pos_w = command.reference_position_temporal_w

    # if mode == "closed_loop" or mode == "open_loop":
    #     pos_error = ee_ref_pos_w - ee_cur_pos_w
    #     error = torch.sum(torch.square(pos_error), dim=-1)
    # else:
    #     pos_error = ee_ref_pos_w - ee_cur_pos_w.unsqueeze(1)  # (N, 1, 3)
    #     error = torch.sum(torch.square(pos_error), dim=-1).mean(dim=1)
    pos_error = ee_ref_pos_w - ee_cur_pos_w
    error = torch.sum(torch.square(pos_error), dim=-1)
    return error


def reward_ee_reference_position_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="left_wrist_yaw_link"),
    command_name: str = "left_ee_position",
    std: float = 0.1
) -> torch.Tensor:
    """Reward the end-effector for tracking the commanded position."""
    command: ImpedanceCommand = env.command_manager.get_term(command_name)
    mode = command.integrate_method
    robot: Articulation = env.scene[asset_cfg.name]
    ee_cur_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids[0], :]
    if mode == "closed_loop":
        ee_ref_pos_w = command.reference_position_closedloop_w
    elif mode == "open_loop":
        ee_ref_pos_w = command.reference_position_openloop_w
    else:
        ee_ref_pos_w = command.reference_position_temporal_w

    # if mode == "closed_loop" or mode == "open_loop":
    #     pos_error = ee_ref_pos_w - ee_cur_pos_w
    #     error = torch.sum(torch.square(pos_error), dim=-1)
    # else:
    #     pos_error = ee_ref_pos_w - ee_cur_pos_w.unsqueeze(1)  # (N, 1, 3)
    #     error = torch.sum(torch.square(pos_error), dim=-1).mean(dim=1)
    pos_error = ee_ref_pos_w - ee_cur_pos_w
    error = torch.sum(torch.square(pos_error), dim=-1)
    return torch.exp(-error / (std**2))


def penalty_ee_reference_velocity_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="left_wrist_yaw_link"),
    command_name: str = "left_ee_position",
) -> torch.Tensor:
    """Penalize the end-effector for deviating from the commanded position."""
    command: ImpedanceCommand = env.command_manager.get_term(command_name)
    mode = command.integrate_method
    robot: Articulation = env.scene[asset_cfg.name]
    ee_cur_vel_w = robot.data.body_lin_vel_w[:, asset_cfg.body_ids[0], :]
    if mode == "closed_loop":
        ee_ref_vel_w = command.reference_velocity_closedloop_w
    elif mode == "open_loop":
        ee_ref_vel_w = command.reference_velocity_openloop_w
    else:
        ee_ref_vel_w = command.reference_velocity_temporal_w

    # if mode == "closed_loop" or mode == "open_loop":
    #     vel_error = ee_ref_vel_w - ee_cur_vel_w
    #     error = torch.sum(torch.square(vel_error), dim=-1)
    # else:
    #     vel_error = ee_ref_vel_w - ee_cur_vel_w.unsqueeze(1)  # (N, 1, 3)
    #     error = torch.sum(torch.square(vel_error), dim=-1).mean(dim=1)
    vel_error = ee_ref_vel_w - ee_cur_vel_w
    error = torch.sum(torch.square(vel_error), dim=-1)
    return error

def reward_ee_reference_velocity_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="left_wrist_yaw_link"),
    command_name: str = "left_ee_position",
    std: float = 0.5
) -> torch.Tensor:
    """Reward the end-effector for tracking the commanded velocity."""
    command: ImpedanceCommand = env.command_manager.get_term(command_name)
    mode = command.integrate_method
    command: ImpedanceCommand = env.command_manager.get_term(command_name)
    robot: Articulation = env.scene[asset_cfg.name]
    ee_cur_vel_w = robot.data.body_lin_vel_w[:, asset_cfg.body_ids[0], :]
    if mode == "closed_loop":
        ee_ref_vel_w = command.reference_velocity_closedloop_w
    elif mode == "open_loop":
        ee_ref_vel_w = command.reference_velocity_openloop_w
    else:
        ee_ref_vel_w = command.reference_velocity_temporal_w

    # if mode == "closed_loop" or mode == "open_loop":
    #     vel_error = ee_ref_vel_w - ee_cur_vel_w
    #     error = torch.sum(torch.square(vel_error), dim=-1)
    # else:
    #     vel_error = ee_ref_vel_w - ee_cur_vel_w.unsqueeze(1)  # (N, 1, 3)
    #     error = torch.sum(torch.square(vel_error), dim=-1).mean(dim=1)
    vel_error = ee_ref_vel_w - ee_cur_vel_w
    error = torch.sum(torch.square(vel_error), dim=-1)
    return torch.exp(-error / (std**2))


def penalty_ee_reference_orientation_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="left_wrist_yaw_link"),
    command_name: str = "left_ee_position",
) -> torch.Tensor:
    """Penalize the end-effector for deviating from the commanded orientation."""
    command: ImpedanceCommand = env.command_manager.get_term(command_name)
    mode = command.integrate_method
    robot: Articulation = env.scene[asset_cfg.name]
    ee_cur_quat_w = robot.data.body_quat_w[:, asset_cfg.body_ids[0], :]
    if mode == "closed_loop":
        ee_ref_quat_w = command.reference_orientation_quat_closedloop_w
    elif mode == "open_loop":
        ee_ref_quat_w = command.reference_orientation_quat_openloop_w
    else:
        ee_ref_quat_w = command.reference_orientation_quat_temporal_w
    # if mode == "closed_loop" or mode == "open_loop":
    #     error = quat_error_magnitude(ee_cur_quat_w, ee_ref_quat_w)
    # else:
    #     error = quat_error_magnitude(ee_cur_quat_w.unsqueeze(1), ee_ref_quat_w).mean(dim=1)
    error = quat_error_magnitude(ee_cur_quat_w, ee_ref_quat_w)
    return error


def reward_ee_reference_orientation_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="left_wrist_yaw_link"),
    command_name: str = "left_ee_position",
    std: float = 0.5
) -> torch.Tensor:
    """Reward the end-effector for tracking the commanded orientation."""
    command: ImpedanceCommand = env.command_manager.get_term(command_name)
    mode = command.integrate_method
    robot: Articulation = env.scene[asset_cfg.name]
    ee_cur_quat_w = robot.data.body_quat_w[:, asset_cfg.body_ids[0], :]
    if mode == "closed_loop":
        ee_ref_quat_w = command.reference_orientation_quat_closedloop_w
    elif mode == "open_loop":
        ee_ref_quat_w = command.reference_orientation_quat_openloop_w
    else:
        ee_ref_quat_w = command.reference_orientation_quat_temporal_w
    # if mode == "closed_loop" or mode == "open_loop":
    #     error = quat_error_magnitude(ee_cur_quat_w, ee_ref_quat_w)
    # else:
    #     error = quat_error_magnitude(ee_cur_quat_w.unsqueeze(1), ee_ref_quat_w).mean(dim=1)
    error = quat_error_magnitude(ee_cur_quat_w, ee_ref_quat_w)
    return torch.exp(-error / (std**2))

def penalty_ee_reference_angular_velocity_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="left_wrist_yaw_link"),
    command_name: str = "left_ee_position",
) -> torch.Tensor:
    """Penalize the end-effector for deviating from the commanded angular velocity."""
    command: ImpedanceCommand = env.command_manager.get_term(command_name)
    mode = command.integrate_method
    robot: Articulation = env.scene[asset_cfg.name]
    ee_cur_ang_vel_w = robot.data.body_ang_vel_w[:, asset_cfg.body_ids[0], :]
    if mode == "closed_loop":
        ee_ref_ang_vel_w = command.reference_angular_velocity_closedloop_w
    elif mode == "open_loop":
        ee_ref_ang_vel_w = command.reference_angular_velocity_openloop_w
    else:
        ee_ref_ang_vel_w = command.reference_angular_velocity_temporal_w

    # if mode == "closed_loop" or mode == "open_loop":
    #     ang_vel_error = ee_ref_ang_vel_w - ee_cur_ang_vel_w
    #     error = torch.sum(torch.square(ang_vel_error), dim=-1)
    # else:
    #     ang_vel_error = ee_ref_ang_vel_w - ee_cur_ang_vel_w.unsqueeze(1)  # (N, 1, 3)
    #     error = torch.sum(torch.square(ang_vel_error), dim=-1).mean(dim=1)
    ang_vel_error = ee_ref_ang_vel_w - ee_cur_ang_vel_w
    error = torch.sum(torch.square(ang_vel_error), dim=-1)
    return error

def reward_ee_reference_angular_velocity_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="left_wrist_yaw_link"),
    command_name: str = "left_ee_position",
    std: float = 0.5
) -> torch.Tensor:
    """Reward the end-effector for tracking the commanded angular velocity."""
    command: ImpedanceCommand = env.command_manager.get_term(command_name)
    mode = command.integrate_method
    robot: Articulation = env.scene[asset_cfg.name]
    ee_cur_ang_vel_w = robot.data.body_ang_vel_w[:, asset_cfg.body_ids[0], :]
    if mode == "closed_loop":
        ee_ref_ang_vel_w = command.reference_angular_velocity_closedloop_w
    elif mode == "open_loop":
        ee_ref_ang_vel_w = command.reference_angular_velocity_openloop_w
    else:
        ee_ref_ang_vel_w = command.reference_angular_velocity_temporal_w

    # if mode == "closed_loop" or mode == "open_loop":
    #     ang_vel_error = ee_ref_ang_vel_w - ee_cur_ang_vel_w
    #     error = torch.sum(torch.square(ang_vel_error), dim=-1)
    # else:
    #     ang_vel_error = ee_ref_ang_vel_w - ee_cur_ang_vel_w.unsqueeze(1)  # (N, 1, 3)
    #     error = torch.sum(torch.square(ang_vel_error), dim=-1).mean(dim=1)
    ang_vel_error = ee_ref_ang_vel_w - ee_cur_ang_vel_w
    error = torch.sum(torch.square(ang_vel_error), dim=-1)
    return torch.exp(-error / (std**2))