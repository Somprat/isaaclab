from __future__ import annotations

from dataclasses import MISSING
from typing import Sequence
import omni.log
import torch
from isaaclab.envs.mdp import UniformVelocityCommandCfg,UniformVelocityCommand
from isaaclab.utils import configclass
from isaaclab.managers import CommandTerm
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_apply, quat_apply_inverse, wrap_to_pi, quat_from_euler_xyz, quat_mul, yaw_quat
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.markers import VisualizationMarkers


# ============================================
# Level Vel Command
# ============================================
class UniformLevelVelocityCommand(UniformVelocityCommand):
    cfg: UniformLevelVelocityCommandCfg

    def __init__(self, cfg: UniformLevelVelocityCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.delay_steps = getattr(cfg, 'delay_steps', 0)
        self.steps_since_reset = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        

    @property
    def command(self) -> torch.Tensor:
        mask = self.steps_since_reset >= self.delay_steps
        result = torch.zeros_like(self.vel_command_b)
        result[mask] = self.vel_command_b[mask]
        return result
    
    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None:
            env_ids = slice(None)
        self.steps_since_reset[env_ids] = 0
        return super().reset(env_ids)
    
    def compute(self, dt: float):
        self.steps_since_reset += 1
        super().compute(dt)
    


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = UniformLevelVelocityCommand

    delay_steps: int = 0
    """The number of steps to delay the command."""

    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING
    """The ranges for the limit command."""


# ============================================
# Level Vel Command in Yaw Frame
# ============================================
class UniformLevelYawFrameVelocityCommand(UniformVelocityCommand):
    cfg: UniformLevelYawFrameVelocityCommandCfg

    def __init__(self, cfg: UniformLevelYawFrameVelocityCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.delay_steps = getattr(cfg, 'delay_steps', 0)
        self.steps_since_reset = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        

    @property
    def command(self) -> torch.Tensor:
        mask = self.steps_since_reset >= self.delay_steps
        result = torch.zeros_like(self.vel_command_b)
        result[mask] = self.vel_command_b[mask]
        return result
    
    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None:
            env_ids = slice(None)
        self.steps_since_reset[env_ids] = 0
        return super().reset(env_ids)
    
    def compute(self, dt: float):
        self.steps_since_reset += 1
        super().compute(dt)

    def _update_metrics(self):
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        root_quat_yaw_w = yaw_quat(self.robot.data.root_quat_w)
        root_lin_vel_yaw = quat_apply_inverse(root_quat_yaw_w, self.robot.data.root_lin_vel_w[:, :3])
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - root_lin_vel_yaw[:, :2], dim=-1) / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_w[:, 2]) / max_command_step
        )

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        root_quat_yaw_w = yaw_quat(self.robot.data.root_quat_w)
        root_lin_vel_yaw = quat_apply_inverse(root_quat_yaw_w, self.robot.data.root_lin_vel_w[:, :3])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(root_lin_vel_yaw[:, :2])
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert XY velocity in the yaw-only frame into world-frame arrow orientation."""
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = quat_from_euler_xyz(zeros, zeros, heading_angle)
        arrow_quat = quat_mul(yaw_quat(self.robot.data.root_quat_w), arrow_quat)
        return arrow_scale, arrow_quat
    


@configclass
class UniformLevelYawFrameVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = UniformLevelYawFrameVelocityCommand

    delay_steps: int = 0
    """The number of steps to delay the command."""

    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING
    """The ranges for the limit command."""


# ============================================
# Level Vel in Body Frame
# ============================================
class UniformBodyLevelVelocityCommand(CommandTerm):
    cfg: UniformBodyLevelVelocityCommandCfg

    def __init__(self, cfg: UniformBodyLevelVelocityCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # check configuration
        if self.cfg.heading_command and self.cfg.ranges.heading is None:
            raise ValueError(
                "The velocity command has heading commands active (heading_command=True) but the `ranges.heading`"
                " parameter is set to None."
            )
        if self.cfg.ranges.heading and not self.cfg.heading_command:
            omni.log.warn(
                f"The velocity command has the 'ranges.heading' attribute set to '{self.cfg.ranges.heading}'"
                " but the heading command is not active. Consider setting the flag for the heading command to True."
            )

        if self.cfg.frame_name not in ["body", "world"] or self.cfg.frame_name is None:
            raise ValueError(f"Invalid frame name: {self.cfg.frame_name}")

        # obtain the robot asset
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        # check if body name is valid
        if self.cfg.body_name is None:
            raise ValueError("Body name is not set.")

        if self.cfg.body_name not in self.robot.body_names:
            raise ValueError(f"Body name '{self.cfg.body_name}' not found in robot.")

        # crete buffers to store the command
        # -- command: x vel, y vel, yaw vel, heading
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_standing_env = torch.zeros_like(self.is_heading_env)
        # -- metrics
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)
        # -- delay steps
        self.delay_steps = getattr(cfg, 'delay_steps', 0)
        self.steps_since_reset = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        # -- body index
        self.body_index = self.robot.find_bodies(cfg.body_name)[0][0]
    
    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "UniformBodyLevelVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeading command: {self.cfg.heading_command}\n"
        if self.cfg.heading_command:
            msg += f"\tHeading probability: {self.cfg.rel_heading_envs}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        mask = self.steps_since_reset >= self.delay_steps
        result = torch.zeros_like(self.vel_command_b)
        result[mask] = self.vel_command_b[mask]
        return result

    def _update_metrics(self):
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt

        # get body velocity
        body_quat_w = self.robot.data.body_link_quat_w[:, self.body_index, :]
        body_lin_vel_w = self.robot.data.body_lin_vel_w[:, self.body_index, :]
        body_ang_vel_w = self.robot.data.body_ang_vel_w[:, self.body_index, :]
        body_lin_vel_b = quat_apply_inverse(body_quat_w, body_lin_vel_w)
        body_ang_vel_b = quat_apply_inverse(body_quat_w, body_ang_vel_w)
        # logs data
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - body_lin_vel_b[:, :2], dim=-1) / max_command_step
        )
        if self.cfg.frame_name == "body":
            self.metrics["error_vel_yaw"] += (
                torch.abs(self.vel_command_b[:, 2] - body_ang_vel_b[:, 2]) / max_command_step
            )
        else:
            self.metrics["error_vel_yaw"] += (
                torch.abs(self.vel_command_b[:, 2] - body_ang_vel_w[:, 2]) / max_command_step
            )
    
    def _resample_command(self, env_ids: Sequence[int]):
        # sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        # -- linear velocity - x direction
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        # -- linear velocity - y direction
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        # -- ang vel yaw - rotation around z
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
        # heading target
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            # update heading envs
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        # update standing envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _update_command(self):
        if self.cfg.heading_command:
            # resolve indices of heading envs
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            # compute body heading
            body_forward_w = quat_apply(self.robot.data.body_quat_w[:, self.body_index, :], self.robot.data.FORWARD_VEC_B)
            body_heading_w = torch.atan2(body_forward_w[:, 1], body_forward_w[:, 0])

            # compute angular velocity
            heading_error = wrap_to_pi(self.heading_target[env_ids] - body_heading_w[env_ids])
            self.vel_command_b[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None:
            env_ids = slice(None)
        self.steps_since_reset[env_ids] = 0
        return super().reset(env_ids)
    
    def compute(self, dt: float):
        self.steps_since_reset += 1
        super().compute(dt)
    
    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "goal_vel_visualizer"):
                # -- goal
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                # -- current
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            # set their visibility to true
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        body_pos_w = self.robot.data.body_pos_w[:, self.body_index, :].clone()
        body_pos_w[:, 2] += 0.2
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        body_quat_w = self.robot.data.body_quat_w[:, self.body_index, :]
        body_lin_vel_b = quat_apply_inverse(body_quat_w, self.robot.data.body_lin_vel_w[:, self.body_index, :])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(body_lin_vel_b[:, :2])
        # display markers
        self.goal_vel_visualizer.visualize(body_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(body_pos_w, vel_arrow_quat, vel_arrow_scale)

    """
    Internal helpers.
    """

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        body_quat_w = self.robot.data.body_quat_w[:, self.body_index, :]
        arrow_quat = quat_mul(body_quat_w, arrow_quat)

        return arrow_scale, arrow_quat



@configclass
class UniformBodyLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = UniformBodyLevelVelocityCommand

    body_name: str = MISSING
    """The name of the body for which the command is generated."""

    delay_steps: int = 0
    """The number of steps to delay the command."""

    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING
    """The ranges for the limit command."""

    frame_name: str = "body"
    """The name of the frame for which the command is generated."""