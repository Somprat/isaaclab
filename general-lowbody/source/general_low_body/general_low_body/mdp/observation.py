from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal, Sequence

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera, FrameTransformer, ContactSensor
from isaaclab.utils.math import quat_apply_inverse, subtract_frame_transforms, quat_mul, quat_inv, euler_xyz_from_quat, quat_apply, yaw_quat
from isaaclab.utils.math import matrix_from_quat
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
import isaaclab.sim as sim_utils
import numpy as np
from isaaclab.utils.noise import NoiseCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from .command import ImpedanceCommand


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase


def base_lin_vel_yaw_frame(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Root linear velocity expressed in a gravity-aligned yaw-only frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])


def base_ang_vel_yaw_frame(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Root angular velocity expressed in a gravity-aligned yaw-only frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_ang_vel_w[:, :3])


def base_yaw_rate(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Root yaw rate around the world vertical axis."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_w[:, 2:3]

def impedance_external_force_w(
    env: ManagerBasedRLEnv, 
    command_name: str = "left_ee_position", 
) -> torch.Tensor:
    """The external force of the impedance command in world frame."""
    command: ImpedanceCommand = env.command_manager.get_term(command_name)
    return command.external_force_w

def impedance_external_force_b(
    env: ManagerBasedRLEnv, 
    command_name: str = "left_ee_position", 
) -> torch.Tensor:
    """The external force of the impedance command in robot root frame."""
    command: ImpedanceCommand = env.command_manager.get_term(command_name)
    return command.external_force_b

def impedance_parameters(
    env: ManagerBasedRLEnv, 
    command_name: str = "left_ee_position", 
) -> torch.Tensor:
    """The impedance parameters (virtual mass, stiffness, damping) of the impedance command."""
    command: ImpedanceCommand = env.command_manager.get_term(command_name)
    virtual_mass = command.mass
    virtual_inertia = command.inertia
    linear_kp = command.linear_kp
    linear_kd = command.linear_kd
    angular_kp = command.angular_kp
    angular_kd = command.angular_kd
    return torch.cat([linear_kp, linear_kd, angular_kp, angular_kd, virtual_mass, virtual_inertia], dim=-1)

def impedance_ref_pos_b(
    env: ManagerBasedRLEnv,
    command_name: str = "left_ee_position",
) -> torch.Tensor:
    """The reference position of the impedance command in robot root frame."""
    command: ImpedanceCommand = env.command_manager.get_term(command_name)
    mode = command.integrate_method
    if mode == "open_loop":
        ref_pos_b = command.reference_position_openloop_b
    elif mode == "closed_loop":
        ref_pos_b = command.reference_position_closedloop_b
    else:
        ref_pos_b = command.reference_position_temporal_b
    return ref_pos_b

def impedance_ref_quat_b(
    env: ManagerBasedRLEnv,
    command_name: str = "left_ee_position",
) -> torch.Tensor:
    """The reference orientation of the impedance command in robot root frame, represented as a quaternion."""
    command: ImpedanceCommand = env.command_manager.get_term(command_name)
    mode = command.integrate_method
    if mode == "open_loop":
        ref_quat_b = command.reference_orientation_quat_openloop_b
    elif mode == "closed_loop":
        ref_quat_b = command.reference_orientation_quat_closedloop_b
    else:
        ref_quat_b = command.reference_orientation_quat_temporal_b
    return ref_quat_b   


def impedance_ref_vel_b(
    env: ManagerBasedRLEnv,
    command_name: str = "left_ee_position",
) -> torch.Tensor:
    """The reference velocity of the impedance command in robot root frame."""
    command: ImpedanceCommand = env.command_manager.get_term(command_name)
    mode = command.integrate_method
    if mode == "open_loop":
        ref_vel_b = command.reference_velocity_openloop_b
    elif mode == "closed_loop":
        ref_vel_b = command.reference_velocity_closedloop_b
    else:
        ref_vel_b = command.reference_velocity_temporal_b
    return ref_vel_b

def impedance_ref_ang_vel_b(
    env: ManagerBasedRLEnv,
    command_name: str = "left_ee_position",
) -> torch.Tensor:
    """The reference angular velocity of the impedance command in robot root frame."""
    command: ImpedanceCommand = env.command_manager.get_term(command_name)
    mode = command.integrate_method
    if mode == "open_loop":
        ref_ang_vel_b = command.reference_angular_velocity_openloop_b
    elif mode == "closed_loop":
        ref_ang_vel_b = command.reference_angular_velocity_closedloop_b
    else:
        ref_ang_vel_b = command.reference_angular_velocity_temporal_b
    return ref_ang_vel_b

