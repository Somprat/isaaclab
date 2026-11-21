from __future__ import annotations

from dataclasses import MISSING
from typing import Sequence

from pathlib import Path
import numpy as np
import omni.log
import torch
from isaaclab.utils import configclass
from isaaclab.managers import CommandTermCfg
from isaaclab.managers import CommandTerm
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.markers import VisualizationMarkers



class EndEffectorPositionCommand(CommandTerm):
    cfg: EndEffectorPositionCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: EndEffectorPositionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.robot_asset_name]
        assert cfg.end_effector_name is not None, "End-effector name must be specified."
        assert cfg.end_effector_name in ["left_wrist_yaw_link", "right_wrist_yaw_link"], "End-effector name must be either 'left_wrist_yaw_link' or 'right_wrist_yaw_link'."
        self.ee_link_name = cfg.end_effector_name
        self.ee_link_indx = self.robot.find_bodies(cfg.end_effector_name)[0][0]
        if self.ee_link_name == "left_wrist_yaw_link":
            boundary_file = cfg.left_boundary_file
        else:
            boundary_file = cfg.right_boundary_file
        self.boundary_dir = Path(__file__).parent / cfg.boundary_dir
        self.boundary_path = self.boundary_dir / boundary_file
        
        # load boundaries
        self.boundaries = self._load_boundary(self.boundary_path)
        
        self.pose_command_yaw = torch.zeros(env.num_envs, 3, device=env.device)
        self.pose_command_b = torch.zeros(env.num_envs, 3, device=env.device)
        self.pose_command_w = torch.zeros(env.num_envs, 3, device=env.device)
        self.metrics["position_error"] = torch.zeros(env.num_envs, device=env.device)

    def __str__(self) -> str:
        msg = "EndEffectorPositionCommand:\n"
        msg += f"\tEnd-effector link: {self.ee_link_name}\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg
    
    def _load_boundary(self, path: Path) -> dict[str, float]:
        data = np.load(path)
        return {
            "radius": float(data["radius"]),
            "z_lower": float(data["z_lower"]),
            "z_upper": float(data["z_upper"]),
        }
    
    """
    Properties
    """
    @property
    def command(self) -> torch.Tensor:
        """The desired end-effector position command. Shape is (num_envs, 6).

        The first three elements correspond to the left end-effector position, followed by the right end-effector position.
        """
        self.pose_command_b = self._yaw_frame_to_root_frame(self.pose_command_yaw)
        return self.pose_command_b
    
    @property
    def command_world(self) -> torch.Tensor:
        """The desired end-effector position command in world frame. Shape is (num_envs, 6).

        The first three elements correspond to the left end-effector position, followed by the right end-effector position.
        """
        self.pose_command_w = self._command_to_world()
        return self.pose_command_w
    
    """
    Implementation specific functions.
    """
    def _yaw_frame_to_root_frame(self, pose_command_yaw: torch.Tensor) -> torch.Tensor:
        """Convert a gravity-aligned yaw-frame target into the robot root frame."""
        root_quat_w = self.robot.data.root_quat_w
        yaw_quat_w = yaw_quat(root_quat_w)
        pose_command_wrt_world = quat_apply(yaw_quat_w, pose_command_yaw)
        return quat_apply_inverse(root_quat_w, pose_command_wrt_world)

    def _command_to_world(self) -> torch.Tensor:
        """Convert root-frame end-effector targets into world-frame positions."""
        pose_command_b = self._yaw_frame_to_root_frame(self.pose_command_yaw)
        return quat_apply(self.robot.data.root_quat_w, pose_command_b) + self.robot.data.root_pos_w

    def _update_metrics(self):
        # compute command in world frame
        self.pose_command_b = self._yaw_frame_to_root_frame(self.pose_command_yaw)
        self.pose_command_w = self._command_to_world()
        # get current end-effector positions in world frame
        ee_pos_w = self.robot.data.body_link_pos_w[:, self.ee_link_indx, :]
        # compute position error
        pos_error = self.pose_command_w - ee_pos_w
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        # sample new end-effector position targets
        r = torch.empty(len(env_ids), device=self.device)
        radius = self.boundaries["radius"]
        z_lower = self.boundaries["z_lower"]
        z_upper = self.boundaries["z_upper"]
        # -- sample in cylindrical coordinates
        rho = torch.sqrt(r.uniform_(0.0, radius**2))
        phi = r.uniform_(0.0, 2 * np.pi)
        z = r.uniform_(z_lower, z_upper)
        # -- convert to Cartesian coordinates
        x = rho * torch.cos(phi)
        y = rho * torch.sin(phi)
        self.pose_command_yaw[env_ids] = torch.stack([x, y, z], dim=-1)
        self.pose_command_b = self._yaw_frame_to_root_frame(self.pose_command_yaw)

    def _update_command(self):
        self.pose_command_b = self._yaw_frame_to_root_frame(self.pose_command_yaw)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                # -- goal pose
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                # -- current body pose
                self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
            # set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        self.pose_command_w = self._command_to_world()
        # update the markers
        # -- goal pose
        self.goal_pose_visualizer.visualize(self.pose_command_w[:, :3], orientations=None)
        # -- current body pose
        ee_pos_w = self.robot.data.body_link_pos_w[:, self.ee_link_indx, :]
        self.current_pose_visualizer.visualize(ee_pos_w[:, :3], orientations=None)




@configclass
class EndEffectorPositionCommandCfg(CommandTermCfg):
    class_type: type = EndEffectorPositionCommand

    robot_asset_name: str = MISSING
    """Name of the robot asset in the environment for which the commands are generated."""

    end_effector_name: str = MISSING
    """Names of the end-effector links in the robot asset for which the commands are generated."""

    boundary_dir: str = "reach_boundaries"
    """Directory where the reachability boundary files are stored."""

    left_boundary_file: str = "left_boundary.npz"
    """Filename for the left end-effector reachability boundary.""" 

    right_boundary_file: str = "right_boundary.npz"
    """Filename for the right end-effector reachability boundary."""

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/goal_pose")
    """The configuration for the goal pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    current_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/body_pose"
    )
    """The configuration for the current pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.1, 0.1, 0.1)
    goal_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    current_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)