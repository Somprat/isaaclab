# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

from isaaclab.assets import RigidObject
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.managers.manager_base import ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_out_of_bounds(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), distance_buffer: float = 3.0
) -> torch.Tensor:
    """Terminate when the actor move too close to the edge of the terrain.

    If the actor moves too close to the edge of the terrain, the termination is activated. The distance
    to the edge of the terrain is calculated based on the size of the terrain and the distance buffer.
    """
    if env.scene.cfg.terrain.terrain_type == "plane":
        return False  # we have infinite terrain because it is a plane
    elif env.scene.cfg.terrain.terrain_type == "generator":
        # obtain the size of the sub-terrains
        terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
        grid_width, grid_length = terrain_gen_cfg.size
        n_rows, n_cols = terrain_gen_cfg.num_rows, terrain_gen_cfg.num_cols
        border_width = terrain_gen_cfg.border_width
        # compute the size of the map
        map_width = n_rows * grid_width + 2 * border_width
        map_height = n_cols * grid_length + 2 * border_width

        # extract the used quantities (to enable type-hinting)
        asset: RigidObject = env.scene[asset_cfg.name]

        # check if the agent is out of bounds
        x_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 0]) > 0.5 * map_width - distance_buffer
        y_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 1]) > 0.5 * map_height - distance_buffer
        return torch.logical_or(x_out_of_bounds, y_out_of_bounds)
    else:
        raise ValueError("Received unsupported terrain type, must be either 'plane' or 'generator'.")


def link_height_below_minimum(
    env: ManagerBasedRLEnv, 
    minimum_height: float, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the specified link's height is below the minimum height.

    This function checks the z-position of a specific body/link and terminates the episode
    if it goes below the specified minimum height threshold. No conditional checks applied.

    Args:
        env: The environment instance.
        minimum_height: The minimum allowed height for the link.
        asset_cfg: The asset configuration specifying the asset and body/link to check.
                  Must specify body_names or body_ids to identify the specific link.

    Returns:
        A boolean tensor indicating which environments should be terminated.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    
    # Get the z-position of the specified bodies
    if isinstance(asset, RigidObject):
        # For RigidObject, use the first body
        body_pos_w = asset.data.body_pos_w[:, 0, 2]
    elif isinstance(asset, Articulation):
        # For Articulation, use the specified body_ids
        body_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids[0], 2]

    # Check if any body is below the minimum height
    height_violation = body_pos_w < minimum_height
    
    return height_violation



def plate_drop_below_pelvis(
    env: ManagerBasedRLEnv,
    plate_asset_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
    robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the plate is dropped below the pelvis.

    This function checks the z-position of the plate and terminates the episode
    if it goes below the pelvis.
    """
    plate_asset: RigidObject = env.scene[plate_asset_cfg.name]
    robot_asset: Articulation = env.scene[robot_asset_cfg.name]

    return plate_asset.data.root_pos_w[:, 2] < robot_asset.data.body_pos_w[:, robot_asset_cfg.body_ids[0], 2]
    

def object_drop_below_plate(
    env: ManagerBasedRLEnv,
    object_asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    plate_asset_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
) -> torch.Tensor:
    """Terminate when the object is dropped below the plate."""
    object_asset: RigidObject = env.scene[object_asset_cfg.name]
    plate_asset: RigidObject = env.scene[plate_asset_cfg.name]
    return object_asset.data.root_pos_w[:, 2] < plate_asset.data.root_pos_w[:, 2]


def object_bad_state(
    env: ManagerBasedRLEnv,
    object_asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    plate_asset_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
    robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link"),
    limit_angle: float = 0.9
) -> torch.Tensor:
    object_asset: RigidObject = env.scene[object_asset_cfg.name]
    plate_asset: RigidObject = env.scene[plate_asset_cfg.name]
    robot_asset: Articulation = env.scene[robot_asset_cfg.name]
    # plate drop
    plate_drop = plate_asset.data.root_pos_w[:, 2] < robot_asset.data.body_pos_w[:, robot_asset_cfg.body_ids[0], 2]
    # object drop
    object_drop = object_asset.data.root_pos_w[:, 2] < plate_asset.data.root_pos_w[:, 2]
    # object bad orientation
    object_bad_orientation = torch.acos(-object_asset.data.projected_gravity_b[:, 2]).abs() > limit_angle
    return plate_drop | object_drop | object_bad_orientation


class DelayTerminationTerm(ManagerTermBase):
    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.term_cfg: TerminationTermCfg = cfg.params.get("term_cfg", None)
        self.termination_delay = cfg.params.get("termination_delay", 0)
        self.step_counter = torch.full(
            (env.num_envs,), -1, dtype=torch.int32, device=env.device
        )
        # resolve term param
        for key, value in self.term_cfg.params.items():
            if isinstance(value, SceneEntityCfg):
                # load the entity
                try:
                    value.resolve(self._env.scene)
                except ValueError as e:
                    raise ValueError(f"Error while parsing '{key}'. {e}")
            self.term_cfg.params[key] = value

    def __call__(
        self, 
        env: ManagerBasedRLEnv, 
        term_cfg:TerminationTermCfg | None = None,
        termination_delay: float = 0
    ) -> torch.Tensor:
       # check termination
       value = self.term_cfg.func(self._env, **self.term_cfg.params).clone()
       
       # update delay step counter
       newly_triggered = value & (self.step_counter < 0)
       self.step_counter[newly_triggered] = 0
       already_triggered = self.step_counter >= 0
       self.step_counter[already_triggered] += 1
       return self.step_counter >= self.termination_delay

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            self.step_counter[:] = -1
        else:
            self.step_counter[env_ids] = -1
