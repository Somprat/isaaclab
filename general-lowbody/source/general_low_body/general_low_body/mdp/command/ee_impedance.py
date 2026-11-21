"""Command manager for generating and updating commands."""

from __future__ import annotations

import torch
from pathlib import Path
import numpy as np
from collections.abc import Sequence
from typing import TYPE_CHECKING
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG
from dataclasses import MISSING
from typing import Literal
from isaaclab.assets import Articulation
from ..utils.force import ConstantForce, ImpulseForce, saturate, clamp_norm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import (
    quat_apply,
    quat_apply_inverse,
    yaw_quat,
    quat_from_angle_axis,
    quat_from_euler_xyz,
    euler_xyz_from_quat,
    quat_mul,
    quat_conjugate,
    quat_box_plus,
    quat_box_minus,
    quat_error_magnitude,
    wrap_to_pi,
)
from general_low_body.managers import DenseCommandTerm, DenseCommandTermCfg
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

class ImpedanceCommand(DenseCommandTerm):
    cfg: ImpedanceCommandCfg
    _env: ManagerBasedRLEnv

    def __init__(self, cfg: ImpedanceCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        super().__init__(cfg, env)
        # initialize robot and link information
        assert cfg.asset_name is not None, f"Asset name must be specified in the configuration for {self.__class__.__name__}."
        self.robot: Articulation = env.scene[cfg.asset_name]
        assert cfg.link_name is not None, f"Link name must be specified in the configuration for {self.__class__.__name__}."
        assert cfg.link_name in ["left_wrist_yaw_link", "right_wrist_yaw_link"], "End-effector name must be either 'left_wrist_yaw_link' or 'right_wrist_yaw_link'."
        self.link_indx = self.robot.find_bodies(cfg.link_name)[0][0]

        # initialize the forces
        self._init_forces(cfg)

        # initialize position target
        self._init_target(cfg)
        self.pose_command_yaw = torch.zeros(env.num_envs, 3, device=env.device)
        self.pose_command_b = torch.zeros(env.num_envs, 3, device=env.device)
        self.pose_command_w = torch.zeros(env.num_envs, 3, device=env.device)
        self.external_target_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        # initialize orientation target (RPY-based, mirrors position target storage)
        self.orient_command_rpy_w = torch.zeros(env.num_envs, 3, device=env.device)
        self.orient_command_quat_w = torch.zeros(env.num_envs, 4, device=env.device)
        self.orient_command_quat_w[:, 0] = 1.0  # identity quat (w=1)
        self.orient_command_quat_b = torch.zeros(env.num_envs, 4, device=env.device)
        self.orient_command_quat_b[:, 0] = 1.0

        # initialize impedance parameters
        self._init_compliant_parameters(cfg)


        # initialize reference
        self.history_length = cfg.history_length + 1 # NOTE: 1 more history for open-loop
        assert cfg.ref_time_step < self.history_length, \
            f"ref_time_step={cfg.ref_time_step} must be < history_length={self.history_length}"
        self.ref_time_step = cfg.ref_time_step - 1
        # NOTE: valid index is [0, history_length+1],
        # The First index stores the CLOSED-LOOP reference
        # The last index stores the OPEN-LOOP reference
        self.ref_pos_w = torch.zeros(env.num_envs, self.history_length, 3, device=env.device)
        self.ref_vel_w = torch.zeros(env.num_envs, self.history_length, 3, device=env.device)
        # Orientation reference: quaternion (world frame) integrated via tangent-space MSD,
        # and angular velocity (world frame).
        # Same buffer convention as position: [:, 0] is closed-loop, [:, -1] is open-loop.
        self.ref_quat_w = torch.zeros(env.num_envs, self.history_length, 4, device=env.device)
        self.ref_quat_w[..., 0] = 1.0  # identity quat (w=1)
        self.ref_omega_w = torch.zeros(env.num_envs, self.history_length, 3, device=env.device)

        # create buffers to store the command
        # -- metrics that can be used for logging
        self.metrics = dict()
        self.integrate_method = cfg.compliant_parameters.integrate_method
        if self.integrate_method not in ["closed_loop", "open_loop", "temporal_smoothing"]:
            raise ValueError(f"Invalid integrate_method {self.integrate_method}. Must be one of 'closed_loop', 'open_loop', or 'temporal_smoothing'.")
        if self.integrate_method == "temporal_smoothing":
            self.metric_suffix = "Temporal Smoothing"
        elif self.integrate_method == "open_loop":
            self.metric_suffix = "Open-loop"
        else:
            self.metric_suffix = "Closed-loop"
        self.metrics[f"Target Position"] = torch.zeros(env.num_envs, device=env.device)
        self.metrics[f"Reference Position {self.metric_suffix}"] = torch.zeros(env.num_envs, device=env.device)
        self.metrics[f"Reference Velocity {self.metric_suffix}"] = torch.zeros(env.num_envs, device=env.device)
        self.metrics[f"Target Orientation"] = torch.zeros(env.num_envs, device=env.device)
        self.metrics[f"Reference Orientation {self.metric_suffix}"] = torch.zeros(env.num_envs, device=env.device)
        self.metrics[f"Reference AngularVelocity {self.metric_suffix}"] = torch.zeros(env.num_envs, device=env.device)
        # -- time left before resampling
        self.time_left = torch.zeros(self.num_envs, device=self.device)
        # -- counter for the number of times the command has been resampled within the current episode
        self.command_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.metric_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self._debug_vis_handle = None
        # set initial state of debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

    def _init_forces(self, cfg: ImpedanceCommandCfg):
        """Initialize the forces for the impedance command term."""
        self.force_ext_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.force_saturate = cfg.force_saturate

        # forces
        self.use_constant_force = cfg.constant_force is not None and not getattr(cfg.constant_force, "disable", False)
        self.use_impulse_force  = cfg.impulse_force  is not None and not getattr(cfg.impulse_force,  "disable", False)

        self.constant_force = ConstantForce.sample(size=self.num_envs, device=self.device)
        self.constant_force.duration.zero_()
        
        self.impulse_force = ImpulseForce.sample(size=self.num_envs, device=self.device)
        self.impulse_force.duration.zero_()

    def _init_target(self, cfg: ImpedanceCommandCfg):
        if cfg.link_name == "left_wrist_yaw_link":
            boundary_file = cfg.target.left_boundary_file
        else:
            boundary_file = cfg.target.right_boundary_file
        self.boundary_dir = Path(__file__).parent / cfg.target.boundary_dir
        self.boundary_path = self.boundary_dir / boundary_file
        data = np.load(self.boundary_path)
        self.boundaries = {
            "radius": float(data["radius"]),
            "z_lower": float(data["z_lower"]),
            "z_upper": float(data["z_upper"]),
        }


    def _init_compliant_parameters(self, cfg: ImpedanceCommandCfg):
        """Initialize the compliant control parameters for the impedance command term."""
        self.lin_kp_range = cfg.compliant_parameters.linear_kp_range
        self.ang_kp_range = cfg.compliant_parameters.angular_kp_range

        self.lin_kp = torch.zeros(self.num_envs, 1, device=self.device)
        self.lin_kd = torch.zeros(self.num_envs, 1, device=self.device)
        self.ang_kp = torch.zeros(self.num_envs, 1, device=self.device)
        self.ang_kd = torch.zeros(self.num_envs, 1, device=self.device)

        if cfg.compliant_parameters.virtual_mass_range is not None:
            self.virtual_mass_range = torch.tensor(cfg.compliant_parameters.virtual_mass_range, device=self.device).float().reshape(-1, 1)
        else:
            self.virtual_mass_range = None
        self.virtual_mass = torch.zeros(self.num_envs, 1, device=self.device)

        if cfg.compliant_parameters.virtual_inertia_range is not None:
            self.virtual_inertia_range = torch.tensor(cfg.compliant_parameters.virtual_inertia_range, device=self.device).float().reshape(-1, 1)
        else:
            self.virtual_inertia_range = None
        self.virtual_inertia = torch.zeros(self.num_envs, 1, device=self.device)

        self.vel_norm_clamp = cfg.compliant_parameters.velocity_norm_clamp
        self.omega_norm_clamp = cfg.compliant_parameters.omega_norm_clamp

        

        

    """
    Properties
    """
    # ============================================
    # Target
    # ============================================
    @property
    def command(self) -> torch.Tensor:
        """The desired end-effector position command. Shape is (num_envs, 7).
        """
        return torch.cat([self.pose_command_b, self.orient_command_quat_b], dim=-1)
    
    @property
    def command_world(self) -> torch.Tensor:
        """The desired end-effector position command in world frame. Shape is (num_envs, 7).
        """
        return torch.cat([self.pose_command_w, self.orient_command_quat_w], dim=-1)

    # Target orientation surfaces:
    #   - world quat: self.orient_command_quat_w  (stored)
    #   - world RPY:  self.orient_command_rpy_w   (stored)
    #   - base quat:  self.orient_command_quat_b  (computed in _update_command)
    #   - base RPY:   property below (derived)
    @property
    def orient_command_rpy_b(self) -> torch.Tensor:
        """Desired EE orientation in base frame as RPY. Shape (N, 3)."""
        return self._quat_buffer_to_rpy(self.orient_command_quat_b)
    
    # ============================================
    # Reference Position
    # ============================================
    @property
    def reference_position_buffer_w(self) -> torch.Tensor:
        """The reference end-effector position computed by integrating the impedance control. Shape is (num_envs,T, 3)."""
        return self.ref_pos_w
    
    @property
    def reference_position_buffer_b(self) -> torch.Tensor:
        """Reference position buffer in root frame. Shape is (num_envs, history_length, 3)."""
        return self._world_to_base_points(self.ref_pos_w)
    
    def reference_position_w(self, t: int) -> torch.Tensor:
        """The reference end-effector position at time step t computed by integrating the impedance control. Shape is (num_envs, 3)."""
        assert t <= self.history_length, f"Time step t must be less than or equal to history_length {self.history_length}."
        return self.ref_pos_w[:, t-1]
    
    def reference_position_b(self, t: int) -> torch.Tensor:
        """The reference end-effector position at time step t computed by integrating the impedance control and transformed to the robot root frame. Shape is (num_envs, 3)."""
        ref_pos_w = self.reference_position_w(t)
        ref_pos_b = self._world_to_base_points(ref_pos_w)
        return ref_pos_b
    
    @property
    def reference_position_temporal_w(self) -> torch.Tensor:
        """Reference EE position at the configured time step. Shape (num_envs, 3)."""
        return self.ref_pos_w[:, self.ref_time_step, :]

    @property
    def reference_position_temporal_b(self) -> torch.Tensor:
        """Reference EE position at the configured time step in robot root frame. Shape (num_envs, 3)."""
        return self._world_to_base_points(self.reference_position_temporal_w)
    
    @property
    def reference_position_closedloop_w(self) -> torch.Tensor:
        """The reference end-effector position computed by integrating the impedance control in a closed-loop manner. Shape is (num_envs, 3)."""
        return self.ref_pos_w[:, 0, :]
    
    @property
    def reference_position_closedloop_b(self) -> torch.Tensor:
        """The reference end-effector position computed by integrating the impedance control in a closed-loop manner and transformed to the robot root frame. Shape is (num_envs, 3)."""
        ref_pos_w = self.reference_position_closedloop_w
        ref_pos_b = self._world_to_base_points(ref_pos_w)
        return ref_pos_b
    
    @property
    def reference_position_openloop_w(self) -> torch.Tensor:
        """The reference end-effector position computed by integrating the impedance control in an open-loop manner. Shape is (num_envs, 3)."""
        return self.ref_pos_w[:, -1]
    
    @property
    def reference_position_openloop_b(self) -> torch.Tensor:
        """The reference end-effector position computed by integrating the impedance control in an open-loop manner and transformed to the robot root frame. Shape is (num_envs, 3)."""
        ref_pos_w = self.reference_position_openloop_w
        ref_pos_b = self._world_to_base_points(ref_pos_w)
        return ref_pos_b
    
    # ============================================
    # Reference Velocity
    # ============================================
    @property
    def reference_velocity_buffer_w(self) -> torch.Tensor:
        """The reference end-effector velocity computed by integrating the impedance control. Shape is (num_envs,T, 3)."""
        return self.ref_vel_w
    
    @property
    def reference_velocity_buffer_b(self) -> torch.Tensor:
        """Reference velocity buffer in root frame. Shape is (num_envs, history_length, 3)."""
        return self._world_to_base_vectors(self.ref_vel_w, subtract_base_lin_vel=True)
    
    def reference_velocity_w(self, t: int) -> torch.Tensor:
        """The reference end-effector velocity at time step t computed by integrating the impedance control. Shape is (num_envs, 3)."""
        assert t <= self.history_length, f"Time step t must be less than or equal to history_length {self.history_length}."
        return self.ref_vel_w[:, t-1]
    
    def reference_velocity_b(self, t: int) -> torch.Tensor:
        """The reference end-effector velocity at time step t computed by integrating the impedance control and transformed to the robot root frame. Shape is (num_envs, 3)."""
        ref_vel_w = self.reference_velocity_w(t)
        ref_vel_b = self._world_to_base_vectors(ref_vel_w, subtract_base_lin_vel=True)
        return ref_vel_b
    
    @property
    def reference_velocity_temporal_w(self) -> torch.Tensor:
        """Reference EE velocity at the configured time step. Shape (num_envs, 3)."""
        return self.ref_vel_w[:, self.ref_time_step, :]

    @property
    def reference_velocity_temporal_b(self) -> torch.Tensor:
        """Reference EE velocity at the configured time step in robot root frame. Shape (num_envs, 3)."""
        return self._world_to_base_vectors(self.reference_velocity_temporal_w, subtract_base_lin_vel=True)
    
    @property
    def reference_velocity_closedloop_w(self) -> torch.Tensor:
        """The reference end-effector velocity computed by integrating the impedance control in a closed-loop manner. Shape is (num_envs, 3)."""
        return self.ref_vel_w[:, 0]
    
    @property
    def reference_velocity_closedloop_b(self) -> torch.Tensor:
        """The reference end-effector velocity computed by integrating the impedance control in a closed-loop manner and transformed to the robot root frame. Shape is (num_envs, 3)."""
        ref_vel_w = self.reference_velocity_closedloop_w
        ref_vel_b = self._world_to_base_vectors(ref_vel_w, subtract_base_lin_vel=True)
        return ref_vel_b
    
    @property
    def reference_velocity_openloop_w(self) -> torch.Tensor:
        """The reference end-effector velocity computed by integrating the impedance control in an open-loop manner. Shape is (num_envs, 3)."""
        return self.ref_vel_w[:, -1]
    
    @property
    def reference_velocity_openloop_b(self) -> torch.Tensor:
        """The reference end-effector velocity computed by integrating the impedance control in an open-loop manner and transformed to the robot root frame. Shape is (num_envs, 3)."""
        ref_vel_w = self.reference_velocity_openloop_w
        ref_vel_b = self._world_to_base_vectors(ref_vel_w, subtract_base_lin_vel=True)
        return ref_vel_b

    # ============================================
    # Reference Orientation
    # ============================================
    # Internal storage is quat (world frame). RPY accessors convert on demand.
    @property
    def reference_orientation_quat_buffer_w(self) -> torch.Tensor:
        """Reference EE orientation as quaternion in world frame. Shape (N, history_length, 4)."""
        return self.ref_quat_w

    @property
    def reference_orientation_quat_buffer_b(self) -> torch.Tensor:
        """Reference EE orientation as quaternion in base frame. Shape (N, history_length, 4)."""
        return self._quat_buffer_world_to_base(self.ref_quat_w)

    @property
    def reference_orientation_rpy_buffer_w(self) -> torch.Tensor:
        """Reference EE orientation as RPY in world frame (derived from quat). Shape (N, history_length, 3)."""
        return self._quat_buffer_to_rpy(self.ref_quat_w)

    @property
    def reference_orientation_rpy_buffer_b(self) -> torch.Tensor:
        """Reference EE orientation as RPY in base frame. Shape (N, history_length, 3)."""
        return self._quat_buffer_to_rpy(self.reference_orientation_quat_buffer_b)

    @property
    def reference_orientation_quat_temporal_w(self) -> torch.Tensor:
        """Reference EE orientation quat at the configured time step in world frame. Shape (N, 4)."""
        return self.ref_quat_w[:, self.ref_time_step, :]

    @property
    def reference_orientation_quat_temporal_b(self) -> torch.Tensor:
        """Reference EE orientation quat at the configured time step in base frame. Shape (N, 4)."""
        return self._quat_buffer_world_to_base(self.reference_orientation_quat_temporal_w)

    @property
    def reference_orientation_quat_closedloop_w(self) -> torch.Tensor:
        return self.ref_quat_w[:, 0, :]

    @property
    def reference_orientation_quat_closedloop_b(self) -> torch.Tensor:
        return quat_mul(quat_conjugate(self.robot.data.root_quat_w), self.ref_quat_w[:, 0, :])

    @property
    def reference_orientation_rpy_closedloop_w(self) -> torch.Tensor:
        return self._quat_buffer_to_rpy(self.ref_quat_w[:, 0, :])

    @property
    def reference_orientation_rpy_closedloop_b(self) -> torch.Tensor:
        return self._quat_buffer_to_rpy(self.reference_orientation_quat_closedloop_b)

    @property
    def reference_orientation_quat_openloop_w(self) -> torch.Tensor:
        return self.ref_quat_w[:, -1, :]

    @property
    def reference_orientation_quat_openloop_b(self) -> torch.Tensor:
        return quat_mul(quat_conjugate(self.robot.data.root_quat_w), self.ref_quat_w[:, -1, :])

    @property
    def reference_orientation_rpy_openloop_w(self) -> torch.Tensor:
        return self._quat_buffer_to_rpy(self.ref_quat_w[:, -1, :])

    @property
    def reference_orientation_rpy_openloop_b(self) -> torch.Tensor:
        return self._quat_buffer_to_rpy(self.reference_orientation_quat_openloop_b)

    # ============================================
    # Reference Angular Velocity
    # ============================================
    @property
    def reference_angular_velocity_buffer_w(self) -> torch.Tensor:
        return self.ref_omega_w

    @property
    def reference_angular_velocity_buffer_b(self) -> torch.Tensor:
        return self._world_to_base_vectors(self.ref_omega_w, subtract_base_lin_vel=False)

    @property
    def reference_angular_velocity_temporal_w(self) -> torch.Tensor:
        """Reference EE angular velocity at the configured time step. Shape (N, 3)."""
        return self.ref_omega_w[:, self.ref_time_step, :]

    @property
    def reference_angular_velocity_temporal_b(self) -> torch.Tensor:
        """Reference EE angular velocity at the configured time step in base frame. Shape (N, 3)."""
        return self._world_to_base_vectors(self.reference_angular_velocity_temporal_w, subtract_base_lin_vel=False)

    @property
    def reference_angular_velocity_closedloop_w(self) -> torch.Tensor:
        return self.ref_omega_w[:, 0]

    @property
    def reference_angular_velocity_closedloop_b(self) -> torch.Tensor:
        return self._world_to_base_vectors(self.reference_angular_velocity_closedloop_w, subtract_base_lin_vel=False)

    @property
    def reference_angular_velocity_openloop_w(self) -> torch.Tensor:
        return self.ref_omega_w[:, -1]

    @property
    def reference_angular_velocity_openloop_b(self) -> torch.Tensor:
        return self._world_to_base_vectors(self.reference_angular_velocity_openloop_w, subtract_base_lin_vel=False)

    # ============================================
    # Compliance Parameter
    # ============================================
    @property
    def linear_kp(self) -> torch.Tensor:
        """The linear stiffness coefficient for the impedance control. Shape is (num_envs, 1)."""
        return self.lin_kp
    
    @property
    def linear_kd(self) -> torch.Tensor:
        """The linear damping coefficient for the impedance control. Shape is (num_envs, 1)."""
        return self.lin_kd
    
    @property
    def angular_kp(self) -> torch.Tensor:
        """The angular stiffness coefficient for the impedance control. Shape is (num_envs, 1)."""
        return self.ang_kp
    
    @property
    def angular_kd(self) -> torch.Tensor:
        """The angular damping coefficient for the impedance control. Shape is (num_envs, 1)."""
        return self.ang_kd
    
    @property
    def mass(self) -> torch.Tensor:
        """The virtual mass for the impedance control. Shape is (num_envs, 1)."""
        return self.virtual_mass
    
    @property
    def inertia(self) -> torch.Tensor:
        """The virtual inertia for the impedance control. Shape is (num_envs, 1)."""
        return self.virtual_inertia
    
    # ============================================
    # External Force
    # ============================================
    @property
    def external_force_w(self) -> torch.Tensor:
        """The external force applied to the end-effector. Shape is (num_envs, 3)."""
        return self.force_ext_w
    
    @property
    def external_force_b(self) -> torch.Tensor:
        """The external force applied to the end-effector in the robot root frame. Shape is (num_envs, 3)."""
        body_quat_w = self.robot.data.body_quat_w[:, self.link_indx, :]
        return quat_apply_inverse(body_quat_w, self.force_ext_w)
    
    

    """
    Operations.
    """
    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Reset the command generator and torch.log metrics.

        This function resets the command counter and resamples the command. It should be called
        at the beginning of each episode.

        Args:
            env_ids: The list of environment IDs to reset. Defaults to None.

        Returns:
            A dictionary containing the information to log under the "{name}" key.
        """
        # resolve the environment IDs
        if env_ids is None:
            env_ids = slice(None)

        # add logging metrics
        extras = {}
        for metric_name, metric_value in self.metrics.items():
            count = self.metric_counter[env_ids].clamp_min(1)
            extras[metric_name] = (metric_value[env_ids] / count).mean().item()
            metric_value[env_ids] = 0.0
        self.metric_counter[env_ids] = 0

        # reset references
        self.ref_pos_w[env_ids] = self.robot.data.body_link_pos_w[env_ids, self.link_indx, :].unsqueeze(1)
        self.ref_vel_w[env_ids] = 0.0
        # reset orientation references to current EE orientation
        ee_quat_w = self.robot.data.body_quat_w[env_ids, self.link_indx, :]
        self.ref_quat_w[env_ids] = ee_quat_w.unsqueeze(1)
        self.ref_omega_w[env_ids] = 0.0

        # set force duration to zero
        self.force_ext_w[env_ids] = 0.0
        self.constant_force.duration[env_ids] = 0.0
        self.impulse_force.duration[env_ids] = 0.0
        # set the command counter to zero
        self.command_counter[env_ids] = 0
        # resample the command
        self._resample(env_ids)
        self._update_command()
        return extras
    
    def _resample(self, env_ids: Sequence[int]):
        """Resample the command.

        This function resamples the command and time for which the command is applied for the
        specified environment indices.

        Args:
            env_ids: The list of environment IDs to resample.
        """
        if len(env_ids) != 0:
            # resample the time left before resampling
            self.time_left[env_ids] = self.time_left[env_ids].uniform_(*self.cfg.resampling_time_range)
            # resample the command
            self._resample_command(env_ids)
            # resample the compliant control parameters
            self._resample_parameters(env_ids)
            # increment the command counter
            self.command_counter[env_ids] += 1


    def compute_substep(self, dt: float) -> None:
        """Apply the forces to the robot."""
        body_quat_w = self.robot.data.body_quat_w[:, self.link_indx, :]
        force_ext_b = quat_apply_inverse(body_quat_w, self.force_ext_w)
        constant_force_b = quat_apply_inverse(body_quat_w, self.constant_force.get_force())
        torque_ext_b = self.constant_force.offset.cross(constant_force_b, dim=-1)

        self.robot.set_external_force_and_torque(
            forces=force_ext_b.unsqueeze(1),
            torques=torque_ext_b.unsqueeze(1),
            body_ids=[self.link_indx],
            is_global=False,
        )

    def compute(self, dt:float):
        """Compute the command and update the metrics."""
        # update command
        self._update_command()
        # update reference
        self._update_reference(dt)
        # update metrics
        self._update_metrics()
        # update time left for resampling
        self.time_left -= dt
        resample_env_ids = (self.time_left <= 0.0).nonzero().flatten()
        if len(resample_env_ids) > 0:
            self._resample(resample_env_ids)
            self._update_command()
        self._update_forces(dt)


    """
    Implementation specific functions.
    """
    def _world_to_base_points(self, points_w: torch.Tensor) -> torch.Tensor:
        """Transform points from world to base frame. points_w: (N, ..., 3)."""
        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w
        # expand to match
        leading_shape = points_w.shape[:-1]
        while root_pos_w.dim() < points_w.dim():
            root_pos_w = root_pos_w.unsqueeze(1)
            root_quat_w = root_quat_w.unsqueeze(1)
        root_quat_w = root_quat_w.expand(*leading_shape, 4)
        return quat_apply_inverse(root_quat_w, points_w - root_pos_w)

    def _quat_buffer_to_rpy(self, quat: torch.Tensor) -> torch.Tensor:
        """Convert (..., 4) quaternion buffer to (..., 3) RPY (roll, pitch, yaw)."""
        leading_shape = quat.shape[:-1]
        flat = quat.reshape(-1, 4)
        roll, pitch, yaw = euler_xyz_from_quat(flat)
        return torch.stack([roll, pitch, yaw], dim=-1).reshape(*leading_shape, 3)

    def _quat_buffer_world_to_base(self, quat_w: torch.Tensor) -> torch.Tensor:
        """Convert (..., 4) world-frame quat to base-frame quat. Broadcasts root_quat over leading dims."""
        root_quat_w = self.robot.data.root_quat_w
        leading_shape = quat_w.shape[:-1]
        while root_quat_w.dim() < quat_w.dim():
            root_quat_w = root_quat_w.unsqueeze(1)
        root_quat_w = root_quat_w.expand(*leading_shape, 4)
        flat_root = root_quat_w.reshape(-1, 4)
        flat_w = quat_w.reshape(-1, 4)
        flat_b = quat_mul(quat_conjugate(flat_root), flat_w)
        return flat_b.reshape(*leading_shape, 4)

    def _world_to_base_vectors(self, vecs_w: torch.Tensor, subtract_base_lin_vel: bool = False) -> torch.Tensor:
        """Transform vectors from world to base frame (rotation only, optionally subtract base linear vel)."""
        root_quat_w = self.robot.data.root_quat_w
        leading_shape = vecs_w.shape[:-1]
        
        if subtract_base_lin_vel:
            root_lin_vel_w = self.robot.data.root_lin_vel_w
            while root_lin_vel_w.dim() < vecs_w.dim():
                root_lin_vel_w = root_lin_vel_w.unsqueeze(1)
            vecs_w = vecs_w - root_lin_vel_w
        
        while root_quat_w.dim() < vecs_w.dim():
            root_quat_w = root_quat_w.unsqueeze(1)
        root_quat_w = root_quat_w.expand(*leading_shape, 4)
        return quat_apply_inverse(root_quat_w, vecs_w)

    def _resample_parameters(self, env_ids: Sequence[int]):
        """Sample the parameters for the impedance control."""
        lin_kp = torch.empty(len(env_ids), 1, device=self.device).uniform_(*self.lin_kp_range)
        ang_kp = torch.empty(len(env_ids), 1, device=self.device).uniform_(*self.ang_kp_range)
        lin_kd = 1.8 * lin_kp.sqrt()
        ang_kd = 1.8 * ang_kp.sqrt()

        self.lin_kp[env_ids] = lin_kp
        self.lin_kd[env_ids] = lin_kd
        self.ang_kp[env_ids] = ang_kp
        self.ang_kd[env_ids] = ang_kd

        if self.virtual_mass_range is not None:
            i = torch.randint(0, self.virtual_mass_range.shape[0], (len(env_ids),), device=self.device)
            virtual_mass = self.virtual_mass_range[i]
        else:
            virtual_mass = 1.0
        self.virtual_mass[env_ids] = virtual_mass

        if self.virtual_inertia_range is not None:
            i = torch.randint(0, self.virtual_inertia_range.shape[0], (len(env_ids),), device=self.device)
            virtual_inertia = self.virtual_inertia_range[i]
        else:
            virtual_inertia = 1.0
        self.virtual_inertia[env_ids] = virtual_inertia

    def _integrate(self, dt:float):
        """Integrate the reference position and velocity based on the current command and forces."""
        # compute acceleration
        command_w = self.pose_command_w.view(self.num_envs, 1, 3).repeat(1, self.history_length, 1)
        ref_acc_w = (
            self.lin_kp.reshape(self.num_envs, 1, 1) * (command_w - self.ref_pos_w) # stiffness
            + self.lin_kd.reshape(self.num_envs, 1, 1) * (0.0 - self.ref_vel_w) # damping
            + saturate(self.force_ext_w, self.force_saturate).reshape(self.num_envs, 1, 3) # external force
        ) / self.virtual_mass.unsqueeze(1)
        # compute velocity
        if self.vel_norm_clamp > 0:
            ref_vel_w = self.ref_vel_w + ref_acc_w * dt
            self.ref_vel_w = clamp_norm(ref_vel_w, 0, self.vel_norm_clamp)
        else:
            self.ref_vel_w.add_(ref_acc_w * dt)
        # compute position
        self.ref_pos_w.add_(self.ref_vel_w * dt)

        # ----- orientation MSD in SO(3) tangent space (quat-based, no gimbal lock) -----
        # Error: e = log(q_cmd ⊗ q_ref^*) ∈ ℝ³ (world-frame axis-angle vector)
        command_quat_w = self.orient_command_quat_w.unsqueeze(1).expand(-1, self.history_length, -1)
        error_rotvec = quat_box_minus(command_quat_w, self.ref_quat_w)  # (N, T, 3)
        # MSD in tangent space
        ref_alpha_w = (
            self.ang_kp.reshape(self.num_envs, 1, 1) * error_rotvec
            + self.ang_kd.reshape(self.num_envs, 1, 1) * (0.0 - self.ref_omega_w)
        ) / self.virtual_inertia.unsqueeze(1)
        if self.omega_norm_clamp > 0:
            ref_omega_w = self.ref_omega_w + ref_alpha_w * dt
            self.ref_omega_w = clamp_norm(ref_omega_w, 0, self.omega_norm_clamp)
        else:
            self.ref_omega_w.add_(ref_alpha_w * dt)
        # World-frame integration: q_new = exp(ω·dt) ⊗ q_ref
        self.ref_quat_w = quat_box_plus(self.ref_quat_w, self.ref_omega_w * dt)
        # renormalize to combat numerical drift over long horizons (quat_box_plus does not renormalize)
        self.ref_quat_w = self.ref_quat_w / self.ref_quat_w.norm(dim=-1, keepdim=True).clamp_min(1e-8)


    def _update_reference(self, dt: float):
        # update buffer (only roll T-1 previsous value, last value is the open-loop reference)
        self.ref_vel_w[:, :-1] = self.ref_vel_w[:, :-1].roll(1, dims=1)
        self.ref_vel_w[:, 0] = self.robot.data.body_lin_vel_w[:, self.link_indx, :]
        self.ref_pos_w[:, :-1] = self.ref_pos_w[:, :-1].roll(1, dims=1)
        # Keep the impedance reference in the same EE frame used by IK, metrics, and debug viz.
        self.ref_pos_w[:, 0] = self.robot.data.body_link_pos_w[:, self.link_indx, :]
        # orientation buffer roll: closed-loop slot reset to current EE pose every step
        self.ref_omega_w[:, :-1] = self.ref_omega_w[:, :-1].roll(1, dims=1)
        self.ref_omega_w[:, 0] = self.robot.data.body_ang_vel_w[:, self.link_indx, :]
        self.ref_quat_w[:, :-1] = self.ref_quat_w[:, :-1].roll(1, dims=1)
        self.ref_quat_w[:, 0] = self.robot.data.body_quat_w[:, self.link_indx, :]
        # integrate
        self._integrate(dt)


    def _update_forces(self, dt: float) -> torch.Tensor:
        """Update the forces based on the current state and the time step.

        This function should be called at every physics step to update the forces.

        Args:
            dt: The time step.
        """
        # resample constant
        if self.use_constant_force:
            self.constant_force.time.add_(dt)
            expire = self.constant_force.expired
            resample = self._env.episode_length_buf[:, None] % self.cfg.constant_force.resample_env_step == 0
            resample = resample & (torch.rand(self.num_envs, 1, device=self.device) < self.cfg.constant_force.resample_probability)
            resample = expire & resample
            constant_force = ConstantForce.sample(
                size=self.num_envs,
                duration_range=self.cfg.constant_force.duration_range,
                force_scales=self.cfg.constant_force.force_scales,
                force_offsets=self.cfg.constant_force.force_offsets,
                device=self.device,
            )
            self.constant_force: ConstantForce = constant_force.where(resample, self.constant_force)

        # resample impulse
        if self.use_impulse_force:
            self.impulse_force.time.add_(dt)
            expire = self.impulse_force.expired
            resample = self._env.episode_length_buf[:, None] % self.cfg.impulse_force.resample_env_step == 0
            resample = resample & (torch.rand(self.num_envs, 1, device=self.device) < self.cfg.impulse_force.resample_probability)
            resample = expire & resample
            impulse_force = ImpulseForce.sample(
                size=self.num_envs,
                duration_range=self.cfg.impulse_force.duration_range,
                x_range=self.cfg.impulse_force.x_range,
                y_range=self.cfg.impulse_force.y_range,
                z_range=self.cfg.impulse_force.z_range,
                device=self.device,
            )
            self.impulse_force: ImpulseForce = impulse_force.where(resample, self.impulse_force)

        # compute total force
        self.force_ext_w[:] = (
            self.constant_force.get_force()
            + self.impulse_force.get_force()
        )

    def _yaw_frame_to_root_frame(self, pose_command_yaw: torch.Tensor) -> torch.Tensor:
        """Convert a gravity-aligned yaw-frame target into the robot root frame."""
        root_quat_w = self.robot.data.root_quat_w
        yaw_quat_w = yaw_quat(root_quat_w)
        pose_command_wrt_world = quat_apply(yaw_quat_w, pose_command_yaw)
        return quat_apply_inverse(root_quat_w, pose_command_wrt_world)

    def _command_to_world(self, pose_command_b: torch.Tensor) -> torch.Tensor:
        """Convert root-frame end-effector targets into world-frame positions."""
        return quat_apply(self.robot.data.root_quat_w, pose_command_b) + self.robot.data.root_pos_w

    def _update_metrics(self):
        # get current end-effector positions in world frame
        ee_pos_w = self.robot.data.body_link_pos_w[:, self.link_indx, :]
        ee_vel_w = self.robot.data.body_lin_vel_w[:, self.link_indx, :]
        ee_quat_w = self.robot.data.body_quat_w[:, self.link_indx, :]
        ee_omega_w = self.robot.data.body_ang_vel_w[:, self.link_indx, :]

        # compute position error
        pos_error = self.pose_command_w - ee_pos_w
        self.metrics["Target Position"] += torch.norm(pos_error, dim=-1)

        # compute orientation error (geodesic angle between target quat and ee quat)
        self.metrics["Target Orientation"] += quat_error_magnitude(self.orient_command_quat_w, ee_quat_w)

        # pick the reference slot based on mode
        if self.integrate_method == "temporal_smoothing":
            slot = self.ref_time_step
        elif self.integrate_method == "open_loop":
            slot = -1
        else:  # closed_loop
            slot = 0

        # reference position / velocity / orientation / angular velocity error vs current EE
        ref_pos_w = self.ref_pos_w[:, slot, :]
        ref_vel_w = self.ref_vel_w[:, slot, :]
        ref_quat_w = self.ref_quat_w[:, slot, :]
        ref_omega_w = self.ref_omega_w[:, slot, :]

        self.metrics[f"Reference Position {self.metric_suffix}"] += torch.norm(ref_pos_w - ee_pos_w, dim=-1)
        self.metrics[f"Reference Velocity {self.metric_suffix}"] += torch.norm(ref_vel_w - ee_vel_w, dim=-1)
        self.metrics[f"Reference Orientation {self.metric_suffix}"] += quat_error_magnitude(ref_quat_w, ee_quat_w)
        self.metrics[f"Reference AngularVelocity {self.metric_suffix}"] += torch.norm(ref_omega_w - ee_omega_w, dim=-1)

        self.metric_counter += 1

    def _resample_command(self, env_ids):
        """Set a fixed forward reach in base-yaw axes, then lock it to world space."""
        # Preserve externally supplied poses during timer resampling and episode resets.
        env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        env_ids = env_ids[~self.external_target_mask[env_ids]]
        if env_ids.numel() == 0:
            return
        # 1. Fixed reach: 0.40 m forward, 0.18 m outward, world height 1.00 m.
        x = torch.full((len(env_ids),), 0.40, device=self.device)
        lateral = 0.18 if self.cfg.link_name == "left_wrist_yaw_link" else -0.18
        y = torch.full_like(x, lateral)
        z = torch.full_like(x, 1.00 - self.cfg.target.fixed_base_z)
        target_yaw = torch.stack([x, y, z], dim=-1)  # (N, 3) in base-yaw frame
        self.pose_command_yaw[env_ids] = target_yaw

        # 2. Lock position to world frame at this moment
        yaw_quat_w = yaw_quat(self.robot.data.root_quat_w[env_ids])
        offset_xy_w = quat_apply(yaw_quat_w, torch.stack([x, y, torch.zeros_like(x)], dim=-1))

        self.pose_command_w[env_ids, 0] = self.robot.data.root_pos_w[env_ids, 0] + offset_xy_w[:, 0]
        self.pose_command_w[env_ids, 1] = self.robot.data.root_pos_w[env_ids, 1] + offset_xy_w[:, 1]
        self.pose_command_w[env_ids, 2] = self.cfg.target.fixed_base_z + z

        # 3. sample orientation: roll/pitch absolute (world), yaw offset from current base heading
        n = len(env_ids)
        roll_t = torch.empty(n, device=self.device).uniform_(*self.cfg.target.roll_range)
        pitch_t = torch.empty(n, device=self.device).uniform_(*self.cfg.target.pitch_range)
        yaw_offset = torch.empty(n, device=self.device).uniform_(*self.cfg.target.yaw_range)
        _, _, base_yaw = euler_xyz_from_quat(self.robot.data.root_quat_w[env_ids])
        target_yaw_w = wrap_to_pi(base_yaw + yaw_offset)

        self.orient_command_rpy_w[env_ids, 0] = roll_t
        self.orient_command_rpy_w[env_ids, 1] = pitch_t
        self.orient_command_rpy_w[env_ids, 2] = target_yaw_w
        self.orient_command_quat_w[env_ids] = quat_from_euler_xyz(roll_t, pitch_t, target_yaw_w)

    def set_target_world(self, position, env_ids=None):
        """Set persistent absolute world XYZ targets in meters for selected environments.

        Accepts one XYZ vector or one per selected environment. The current desired
        orientation is retained. Resampling and resets preserve externally set poses;
        the impedance reference still evolves smoothly from the measured hand state.
        Call before refreshing policy observations when changing targets at runtime.
        """
        if env_ids is None:
            env_ids = slice(None)
        position = torch.as_tensor(position, device=self.device, dtype=self.pose_command_w.dtype)
        selected = self.pose_command_w[env_ids]
        if position.shape != (3,) and position.shape != selected.shape:
            raise ValueError("position must have shape (3,) or (number of selected environments, 3)")
        if not torch.isfinite(position).all():
            raise ValueError("position must contain finite world coordinates")
        self.pose_command_w[env_ids] = position
        self.external_target_mask[env_ids] = True
        self._update_command()


    def _update_command(self):
        """Only recompute base-frame representation; world-frame target is fixed."""
        self.pose_command_b = quat_apply_inverse(
            self.robot.data.root_quat_w,
            self.pose_command_w - self.robot.data.root_pos_w
        )
        # orientation in base frame: q_b = q_root^{-1} * q_w
        self.orient_command_quat_b = quat_mul(
            quat_conjugate(self.robot.data.root_quat_w),
            self.orient_command_quat_w,
        )
        

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization into visualization objects.

        This function is responsible for creating the visualization objects if they don't exist
        and input ``debug_vis`` is True. If the visualization objects exist, the function should
        set their visibility into the stage.
        """
        if debug_vis:
            if not hasattr(self, "current_force_visualizer"):
                self.current_force_visualizer = VisualizationMarkers(self.cfg.current_force_visualizer_cfg)
            if not hasattr(self, "goal_pose_visualizer"):
                # -- goal pose
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                # -- current body pose
                self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
            if not hasattr(self, "reference_pose_visualizer"):
                self.reference_pose_visualizer = VisualizationMarkers(self.cfg.reference_pose_visualizer_cfg)
            # set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
            self.current_force_visualizer.set_visibility(True)
            self.reference_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "current_force_visualizer"):
                self.current_force_visualizer.set_visibility(False)
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)
            if hasattr(self, "reference_pose_visualizer"):
                self.reference_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Callback for debug visualization.

        This function calls the visualization objects and sets the data to visualize into them.
        """
        if not self.robot.is_initialized:
            return
        # force visualization
        body_pos_w = self.robot.data.body_link_pos_w[:, self.link_indx, :].clone()
        arrow_scale, arrow_quat = self._resolve_force_to_arrow(self.force_ext_w)
        self.current_force_visualizer.visualize(body_pos_w, arrow_quat, arrow_scale)

        # position + orientation visualization (FRAME_MARKER_CFG renders axes)
        self.goal_pose_visualizer.visualize(self.pose_command_w[:, :3], orientations=self.orient_command_quat_w)
        ee_pos_w = self.robot.data.body_link_pos_w[:, self.link_indx, :]
        ee_quat_w = self.robot.data.body_quat_w[:, self.link_indx, :]
        self.current_pose_visualizer.visualize(ee_pos_w[:, :3], orientations=ee_quat_w)

        # reference visualization (single slot per mode)
        if self.integrate_method == "temporal_smoothing":
            slot = self.ref_time_step
        elif self.integrate_method == "open_loop":
            slot = -1
        else:  # closed_loop
            slot = 0
        self.reference_pose_visualizer.visualize(
            self.ref_pos_w[:, slot, :], orientations=self.ref_quat_w[:, slot, :]
        )

    def _resolve_force_to_arrow(self, force_w: torch.Tensor):
        """Convert 3D force vector to arrow scale and orientation."""

        default_scale = self.current_force_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(force_w.shape[0], 1)
        # force magnitude
        force_mag = torch.linalg.norm(force_w, dim=1)
        # scale
        arrow_scale[:, 0] *= force_mag * 0.2
        # direction 
        dir = force_w / (force_mag + 1e-6)
        default = torch.zeros_like(dir)
        default[:, 0] = 1.0
        axis = torch.cross(default, dir, dim=-1)
        dot = torch.sum(default * dir, dim=-1).clamp(-1.0, 1.0)
        angle = torch.acos(dot)
        arrow_quat = quat_from_angle_axis(angle, axis)

        return arrow_scale, arrow_quat



@configclass
class ImpedanceCommandCfg(DenseCommandTermCfg):
    """Configuration for an impedance command term."""

    class_type: type = ImpedanceCommand
    """The associated impedance command class to use."""

    substep_update: bool = False
    """Whether to update the command at each substep. Defaults to False."""

    asset_name: str = MISSING
    """Name of the asset in the environment for which the force is generated."""

    link_name: str = MISSING
    """Name of the link or regex expression that the force will be applied to."""

    resampling_time_range: tuple[float, float] = MISSING
    """Time before commands are changed [s]."""

    history_length: int = 32
    """The length of the history to keep for the command. This is used for computing the reference for the impedance control."""

    ref_time_step: int = 24
    """Single time step (1-indexed) into the reference buffer to expose under
    `temporal_smoothing` mode. Must be ≤ history_length. Internally stored as
    `ref_time_step - 1` (0-indexed slot)."""

    force_saturate: float = 80.0
    """The saturation limit for the external force in the impedance control."""

    debug_vis: bool = False
    """Whether to visualize debug information. Defaults to False."""

    @configclass
    class TargetCfg:
        """Configuration for the position target of the impedance command."""

        boundary_dir: str = "reach_boundaries"
        """Directory where the reachability boundary files are stored."""

        left_boundary_file: str = "left_boundary.npz"
        """Filename for the left end-effector reachability boundary.""" 

        right_boundary_file: str = "right_boundary.npz"
        """Filename for the right end-effector reachability boundary."""

        fixed_base_z: float = 0.78
        """The fixed z height of the robot base. The end-effector target will be generated with a z offset relative to this base height."""

        roll_range: tuple[float, float] = (0.0, 0.0)
        """Range to uniformly sample target EE roll (world-frame, radians). Default = identity (no roll variation)."""

        pitch_range: tuple[float, float] = (0.0, 0.0)
        """Range to uniformly sample target EE pitch (world-frame, radians). Default = identity."""

        yaw_range: tuple[float, float] = (0.0, 0.0)
        """Range to uniformly sample target EE yaw OFFSET from current base heading (radians). Default = aligned with base yaw."""

    target: TargetCfg = MISSING
    """Configuration for the position target of the impedance command."""

    @configclass
    class ConstantForceCfg:
        """Configuration for the constant force term."""

        disable: bool = False
        """Whether to disable the constant force. Defaults to False."""

        duration_range: tuple[float, float] = (1.0, 4.0)
        """The range to sample the duration of the force for."""

        force_offsets: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """The offset of the random force in each dimension."""

        force_scales: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """The scale of the random force in each dimension."""

        resample_probability: float = 0.2
        """The probability of resampling the constant force when the time is up. Defaults to 0.2."""

        resample_env_step: int = 10
        """The environment step interval to check for resampling the constant force when the time is up. Defaults to 10."""

    constant_force: ConstantForceCfg = MISSING
    """Configuration for the constant force term."""

    @configclass
    class ImpulseForceCfg:
        """Configuration for the impulse force term."""

        disable: bool = False
        """Whether to disable the impulse force. Defaults to False."""

        duration_range: tuple[float, float] = (0.4, 0.6)
        """The range to sample the duration of the impulse force for."""

        x_range: tuple[float, float] = (20., 80.)
        """The range to sample the x component of the impulse force."""

        y_range: tuple[float, float] = (20., 80.)
        """The range to sample the y component of the impulse force."""

        z_range: tuple[float, float] = (-20., 20.)
        """The range to sample the z component of the impulse force."""

        resample_probability: float = 0.2
        """The probability of resampling the constant force when the time is up. Defaults to 0.2."""

        resample_env_step: int = 10
        """The environment step interval to check for resampling the constant force when the time is up. Defaults to 10."""

    impulse_force: ImpulseForceCfg = MISSING
    """Configuration for the impulse force term."""

    @configclass
    class CompliantParametersCfg:
        """Configuration for the compliant control parameters."""

        linear_kp_range: tuple[float, float] = MISSING
        """The range to sample the linear stiffness for."""

        angular_kp_range: tuple[float, float] = (0.0, 0.0)
        """The range to sample the angular stiffness for. TODO: not implemented yet"""

        virtual_mass_range: list[float] | None = None
        """The range to sample the virtual mass for."""

        virtual_inertia_range: list[float] | None = None
        """Discrete set of virtual rotational inertias to sample from for orientation MSD. None → fixed at 1.0."""

        velocity_norm_clamp: float = 2.0
        """The maximum norm to clamp the reference velocity to. This is used to prevent the reference velocity from becoming too large when the stiffness is low."""

        omega_norm_clamp: float = 4.0
        """Maximum norm clamp for reference angular velocity (rad/s). Set <= 0 to disable."""

        integrate_method: Literal["closed_loop", "open_loop", "temporal_smoothing"] = "temporal_smoothing"
        """The method to integrate the reference position and velocity. "closed_loop" uses the current reference position and velocity to compute the acceleration, while "open_loop" uses the command directly to compute the acceleration without feedback from the current reference. "temporal_smoothing" uses a temporal smoothing approach to integrate the reference position and velocity. Defaults to "temporal_smoothing"."""

    compliant_parameters: CompliantParametersCfg | None = None
    """Configuration for the compliant control parameters."""

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/goal_pose")
    """The configuration for the goal pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    current_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/body_pose"
    )
    """The configuration for the current pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    reference_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/reference_pose"
    )
    """The configuration for the open-loop / closed-loop reference pose marker. Frame marker so reference orientation is visible."""

    current_force_visualizer_cfg: VisualizationMarkersCfg = RED_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Force/current_force"
    )
    """The configuration for the current force visualization marker."""

    # Set the scale of the visualization markers to (0.1, 0.1, 0.1)
    goal_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

    current_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

    # Slightly smaller frame for the reference so it's visually distinguishable from goal/current.
    reference_pose_visualizer_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)

    current_force_visualizer_cfg.markers["arrow"].scale = (0.2, 0.2, 0.8)
    """Set the scale of the current force visualization marker."""
