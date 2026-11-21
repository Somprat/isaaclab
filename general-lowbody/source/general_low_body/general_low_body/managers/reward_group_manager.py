# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward manager for computing reward signals for a given world."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from prettytable import PrettyTable
from typing import TYPE_CHECKING, List
from isaaclab.managers import ManagerBase, ManagerTermBase
from isaaclab.managers import RewardTermCfg, RewardManager
from isaaclab.utils import configclass
from dataclasses import MISSING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

@configclass
class RewardGroupTermCfg(RewardTermCfg):
    """Configuration for a reward group term."""
    group_name: str = MISSING
    """The name of the reward group."""


class RewardGroupManager(RewardManager):
    """Manager for computing reward signals for a given world.

    The reward manager computes the total reward as a sum of the weighted reward terms. The reward
    terms are parsed from a nested config class containing the reward manger's settings and reward
    terms configuration.

    The reward terms are parsed from a config class containing the manager's settings and each term's
    parameters. Each reward term should instantiate the :class:`RewardTermCfg` class.

    .. note::

        The reward manager multiplies the reward term's ``weight``  with the time-step interval ``dt``
        of the environment. This is done to ensure that the computed reward terms are balanced with
        respect to the chosen time-step interval in the environment.

    """

    _env: ManagerBasedRLEnv
    """The environment instance."""

    def __init__(self, cfg: object, env: ManagerBasedRLEnv):
        """Initialize the reward manager.

        Args:
            cfg: The configuration object or dictionary (``dict[str, RewardTermCfg]``).
            env: The environment instance.
        """
        super().__init__(cfg, env)
        
        # initialize reward dict
        self._group_names: list[str] = ["lower_body", "upper_body", "residual_whole_body", "shared"]
        for _, (_, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            if term_cfg.group_name not in self._group_names:
                self._group_names.append(term_cfg.group_name)
        self._reward_dict = {
            group_name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device) 
            for group_name in self._group_names
        }


    def compute(self, dt: float) -> dict[str, torch.Tensor]:
        """Computes the reward signal as a weighted sum of individual terms.

        This function calls each reward term managed by the class and adds them to compute the net
        reward signal. It also updates the episodic sums corresponding to individual reward terms.

        Args:
            dt: The time-step interval of the environment.

        Returns:
            The net reward signal of shape (num_envs,).
        """
        # reset computation
        self._reward_buf[:] = 0.0
        for group_name in self._group_names:
            self._reward_dict[group_name][:] = 0.0
        # iterate over all the reward terms
        for term_idx, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            # skip if weight is zero (kind of a micro-optimization)
            if term_cfg.weight == 0.0:
                self._step_reward[:, term_idx] = 0.0
                continue
            # compute term's value
            value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt
            # update total reward
            self._reward_dict[term_cfg.group_name] += value
            # update episodic sum
            self._episode_sums[name] += value

            # Update current reward for this step.
            self._step_reward[:, term_idx] = value / dt
        
        # add shared reward
        shared_reward = self._reward_dict["shared"]
        for group_name in self._group_names:
            if group_name != "shared":
                self._reward_dict[group_name] += shared_reward

        return self._reward_dict

