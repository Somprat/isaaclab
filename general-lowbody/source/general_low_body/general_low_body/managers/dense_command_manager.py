# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command manager for generating and updating commands."""

from __future__ import annotations

import inspect
import torch
import weakref
from abc import abstractmethod
from collections.abc import Sequence
from prettytable import PrettyTable
from typing import TYPE_CHECKING
from dataclasses import MISSING
from isaaclab.utils import configclass
import omni.kit.app

from isaaclab.managers import ManagerBase, ManagerTermBase
from isaaclab.managers import CommandTermCfg
from isaaclab.managers import CommandManager, CommandTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

class DenseCommandTerm(CommandTerm):
    """Dense command term that can be updated at each substep."""
    def __init__(self, cfg: DenseCommandTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    @abstractmethod
    def compute_substep(self, dt: float):
        """Updates the command term at each substep.

        Args:
            dt: The time-step interval of the environment.
        """
        raise NotImplementedError

@configclass
class DenseCommandTermCfg(CommandTermCfg):
    """Configuration for a dense command term."""
    class_type: type[DenseCommandTerm] = MISSING
    """The class type of the command term."""
    substep_update: bool = False
    """Whether the command term should be updated at each substep."""

        
class DenseCommandManager(ManagerBase):
    """Manager for generating commands.

    The command manager is used to generate commands for an agent to execute. It makes it convenient to switch
    between different command generation strategies within the same environment. For instance, in an environment
    consisting of a quadrupedal robot, the command to it could be a velocity command or position command.
    By keeping the command generation logic separate from the environment, it is easy to switch between different
    command generation strategies.

    The command terms are implemented as classes that inherit from the :class:`CommandTerm` class.
    Each command generator term should also have a corresponding configuration class that inherits from the
    :class:`CommandTermCfg` class.
    """

    _env: ManagerBasedRLEnv
    """The environment instance."""

    def __init__(self, cfg: object, env: ManagerBasedRLEnv):
        """Initialize the command manager.

        Args:
            cfg: The configuration object or dictionary (``dict[str, CommandTermCfg]``).
            env: The environment instance.
        """
        # create buffers to parse and store terms
        self._terms: dict[str, DenseCommandTerm | CommandTerm] = dict()
        self._dense_terms: dict[str, DenseCommandTerm] = dict() # terms that are updated at each substep

        # call the base class constructor (this prepares the terms)
        super().__init__(cfg, env)
        # store the commands
        self._commands = dict()
        if self.cfg:
            self.cfg.debug_vis = False
            for term in self._terms.values():
                self.cfg.debug_vis |= term.cfg.debug_vis

    def __str__(self) -> str:
        """Returns: A string representation for the command manager."""
        msg = f"<CommandManager> contains {len(self._terms.values())} active terms.\n"

        # create table for term information
        table = PrettyTable()
        table.title = "Active Command Terms"
        table.field_names = ["Index", "Name", "Type", "Substep Update"]
        # set alignment of table columns
        table.align["Name"] = "l"
        # add info on each term
        for index, (name, term) in enumerate(self._terms.items()):
            table.add_row([index, name, term.__class__.__name__, getattr(term.cfg, 'substep_update', False)])
        # convert table to string
        msg += table.get_string()
        msg += "\n"

        return msg

    """
    Properties.
    """

    @property
    def active_terms(self) -> list[str]:
        """Name of active command terms."""
        return list(self._terms.keys())

    @property
    def has_debug_vis_implementation(self) -> bool:
        """Whether the command terms have debug visualization implemented."""
        # check if function raises NotImplementedError
        has_debug_vis = False
        for term in self._terms.values():
            has_debug_vis |= term.has_debug_vis_implementation
        return has_debug_vis

    """
    Operations.
    """

    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        """Returns the active terms as iterable sequence of tuples.

        The first element of the tuple is the name of the term and the second element is the raw value(s) of the term.

        Args:
            env_idx: The specific environment to pull the active terms from.

        Returns:
            The active terms.
        """

        terms = []
        idx = 0
        for name, term in self._terms.items():
            terms.append((name, term.command[env_idx].cpu().tolist()))
            idx += term.command.shape[1]
        return terms

    def set_debug_vis(self, debug_vis: bool):
        """Sets whether to visualize the command data.

        Args:
            debug_vis: Whether to visualize the command data.

        Returns:
            Whether the debug visualization was successfully set. False if the command
            generator does not support debug visualization.
        """
        for term in self._terms.values():
            term.set_debug_vis(debug_vis)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Reset the command terms and log their metrics.

        This function resets the command counter and resamples the command for each term. It should be called
        at the beginning of each episode.

        Args:
            env_ids: The list of environment IDs to reset. Defaults to None.

        Returns:
            A dictionary containing the information to log under the "Metrics/{term_name}/{metric_name}" key.
        """
        # resolve environment ids
        if env_ids is None:
            env_ids = slice(None)
        # store information
        extras = {}
        for name, term in self._terms.items():
            # reset the command term
            metrics = term.reset(env_ids=env_ids)
            # compute the mean metric value
            for metric_name, metric_value in metrics.items():
                extras[f"Metrics/{name}/{metric_name}"] = metric_value
        # return logged information
        return extras

    def compute(self, dt: float):
        """Updates the commands.

        This function calls each command term managed by the class.

        Args:
            dt: The time-step interval of the environment.

        """
        # iterate over all the command terms
        for term in self._terms.values():
            # compute term's value
            term.compute(dt)
    
    def compute_substep(self, dt: float):
        """Updates the commands at each substep.

        This function calls each command term that requires substep update, managed by the class.

        Args:
            dt: The time-step interval of the environment.

        """
        # iterate over all the command terms that require substep update
        for term in self._dense_terms.values():
            # compute term's value
            term.compute_substep(dt)

    def get_command(self, name: str) -> torch.Tensor:
        """Returns the command for the specified command term.

        Args:
            name: The name of the command term.

        Returns:
            The command tensor of the specified command term.
        """
        return self._terms[name].command

    def get_term(self, name: str) -> DenseCommandTerm:
        """Returns the command term with the specified name.

        Args:
            name: The name of the command term.

        Returns:
            The command term with the specified name.
        """
        return self._terms[name]

    """
    Helper functions.
    """

    def _prepare_terms(self):
        # check if config is dict already
        if isinstance(self.cfg, dict):
            cfg_items = self.cfg.items()
        else:
            cfg_items = self.cfg.__dict__.items()
        # iterate over all the terms
        for term_name, term_cfg in cfg_items:
            # check for non config
            if term_cfg is None:
                continue
            # check for valid config type
            if not isinstance(term_cfg, (DenseCommandTermCfg, CommandTermCfg)):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type DenseCommandTermCfg or CommandTermCfg."
                    f" Received: '{type(term_cfg)}'."
                )
            # create the action term
            term = term_cfg.class_type(term_cfg, self._env)
            # sanity check if term is valid type
            if not isinstance(term, (DenseCommandTerm, CommandTerm)):
                raise TypeError(f"Returned object for the term '{term_name}' is not of type DenseCommandTerm or CommandTerm.")
            # add class to dict
            self._terms[term_name] = term

            # add to dense term dict if substep update is required
            if isinstance(term_cfg, DenseCommandTermCfg) and term_cfg.substep_update:
                self._dense_terms[term_name] = term
