
from __future__ import annotations

import torch
from collections.abc import Callable, Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING, Any
from isaaclab.utils import configclass
from isaaclab.utils.modifiers import ModifierCfg
from isaaclab.utils.noise import NoiseCfg
from isaaclab.managers import ManagerTermBaseCfg, ObservationTermCfg, ManagerBase, ObservationGroupCfg, ManagerTermBase
import inspect
import numpy as np
from prettytable import PrettyTable
from isaaclab.utils import class_to_dict, modifiers
from isaaclab.utils.buffers import CircularBuffer
from general_low_body.utils.delay_buffer import DelayBuffer
from isaaclab.managers import ObservationManager
from isaaclab.utils import class_to_dict, modifiers, noise

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

@configclass
class DelayObservationTermCfg(ObservationTermCfg):
    """Configuration for an observation term."""
    delay_min_lag: int = 0
    """Minimum lag (in steps) for delayed observations. Lag sampled uniformly from
    [min_lag, max_lag]. Convert to ms: lag * (1000 / control_hz)."""
    delay_max_lag: int = 0
    """Maximum lag (in steps) for delayed observations. Use min=max for constant delay."""
    delay_per_env: bool = True
    """If True, each environment samples its own lag. If False, all environments share
    the same lag at each step."""
    delay_hold_prob: float = 0.0
    """Probability of reusing the previous lag instead of resampling. Useful for
    temporally correlated latency patterns."""
    delay_update_period: int = 0
    """Resample lag every N steps (models multi-rate sensors). If 0, update every step."""
    delay_per_env_phase: bool = True
    """If True and update_period > 0, stagger update timing across envs to avoid
    synchronized resampling."""




class DelayObservationManager(ObservationManager):
    

    def __init__(self, cfg: object, env: ManagerBasedEnv):
        """Initialize the observation manager.

        Args:
            cfg: The configuration object or dictionary (``dict[str, ObservationTermCfg]``).
            env: The environment instance.
        """
        super().__init__(cfg, env)


    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        # call all terms that are classes
        for group_name, group_cfg in self._group_obs_class_term_cfgs.items():
            for term_cfg in group_cfg:
                term_cfg.func.reset(env_ids=env_ids)
            # reset terms with history
            for term_name in self._group_obs_term_names[group_name]:
                if term_name in self._group_obs_term_delay_buffer[group_name]:
                    self._group_obs_term_delay_buffer[group_name][term_name].reset(batch_ids=env_ids)
                if term_name in self._group_obs_term_history_buffer[group_name]:
                    self._group_obs_term_history_buffer[group_name][term_name].reset(batch_ids=env_ids)
        # call all modifiers that are classes
        for mod in self._group_obs_class_modifiers:
            mod.reset(env_ids=env_ids)

        # nothing to log here
        return {}


    def compute_group(self, group_name: str, update_history: bool = False) -> torch.Tensor | dict[str, torch.Tensor]:
        """Computes the observations for a given group.

        The observations for a given group are computed by calling the registered functions for each
        term in the group. The functions are called in the order of the terms in the group. The functions
        are expected to return a tensor with shape (num_envs, ...).

        The following steps are performed for each observation term:

        1. Compute observation term by calling the function
        2. Apply custom modifiers in the order specified in :attr:`ObservationTermCfg.modifiers`
        3. Apply corruption/noise model based on :attr:`ObservationTermCfg.noise`
        4. Apply clipping based on :attr:`ObservationTermCfg.clip`
        5. Apply scaling based on :attr:`ObservationTermCfg.scale`
        6. Apply delay based on :attr:`ObservationTermCfg.delay_*` settings

        We apply noise to the computed term first to maintain the integrity of how noise affects the data
        as it truly exists in the real world. If the noise is applied after clipping or scaling, the noise
        could be artificially constrained or amplified, which might misrepresent how noise naturally occurs
        in the data.

        Args:
            group_name: The name of the group for which to compute the observations. Defaults to None,
                in which case observations for all the groups are computed and returned.

        Returns:
            Depending on the group's configuration, the tensors for individual observation terms are
            concatenated along the last dimension into a single tensor. Otherwise, they are returned as
            a dictionary with keys corresponding to the term's name.

        Raises:
            ValueError: If input ``group_name`` is not a valid group handled by the manager.
        """
        # check ig group name is valid
        if group_name not in self._group_obs_term_names:
            raise ValueError(
                f"Unable to find the group '{group_name}' in the observation manager."
                f" Available groups are: {list(self._group_obs_term_names.keys())}"
            )
        # iterate over all the terms in each group
        group_term_names = self._group_obs_term_names[group_name]
        # buffer to store obs per group
        group_obs = dict.fromkeys(group_term_names, None)
        # read attributes for each term
        obs_terms = zip(group_term_names, self._group_obs_term_cfgs[group_name])

        # evaluate terms: compute, add noise, clip, scale, custom modifiers
        for term_name, term_cfg in obs_terms:
            # compute term's value
            obs: torch.Tensor = term_cfg.func(self._env, **term_cfg.params).clone()
            # apply post-processing
            if term_cfg.modifiers is not None:
                for modifier in term_cfg.modifiers:
                    obs = modifier.func(obs, **modifier.params)
            if isinstance(term_cfg.noise, noise.NoiseCfg):
                obs = term_cfg.noise.func(obs, term_cfg.noise)
            elif isinstance(term_cfg.noise, noise.NoiseModelCfg) and term_cfg.noise.func is not None:
                obs = term_cfg.noise.func(obs)
            if term_cfg.clip:
                obs = obs.clip_(min=term_cfg.clip[0], max=term_cfg.clip[1])
            if term_cfg.scale is not None:
                obs = obs.mul_(term_cfg.scale)
            delay_max_lag = getattr(term_cfg, 'delay_max_lag', 0)
            if delay_max_lag > 0:
                delay_buffer = self._group_obs_term_delay_buffer[group_name][term_name]
                delay_buffer.append(obs)
                obs = delay_buffer.compute()
            # Update the history buffer if observation term has history enabled
            if term_cfg.history_length > 0:
                circular_buffer = self._group_obs_term_history_buffer[group_name][term_name]
                if update_history:
                    circular_buffer.append(obs)
                elif circular_buffer._buffer is None:
                    # because circular buffer only exits after the simulation steps,
                    # this guards history buffer from corruption by external calls before simulation start
                    circular_buffer = CircularBuffer(
                        max_len=circular_buffer.max_length,
                        batch_size=circular_buffer.batch_size,
                        device=circular_buffer.device,
                    )
                    circular_buffer.append(obs)

                if term_cfg.flatten_history_dim:
                    group_obs[term_name] = circular_buffer.buffer.reshape(self._env.num_envs, -1)
                else:
                    group_obs[term_name] = circular_buffer.buffer
            else:
                group_obs[term_name] = obs

        # concatenate all observations in the group together
        if self._group_obs_concatenate[group_name]:
            # set the concatenate dimension, account for the batch dimension if positive dimension is given
            return torch.cat(list(group_obs.values()), dim=self._group_obs_concatenate_dim[group_name])
        else:
            return group_obs

    """
    Helper functions.
    """

    def _prepare_terms(self):
        """Prepares a list of observation terms functions."""
        # create buffers to store information for each observation group
        # TODO: Make this more convenient by using data structures.
        self._group_obs_term_names: dict[str, list[str]] = dict()
        self._group_obs_term_dim: dict[str, list[tuple[int, ...]]] = dict()
        self._group_obs_term_cfgs: dict[str, list[ObservationTermCfg]] = dict()
        self._group_obs_class_term_cfgs: dict[str, list[ObservationTermCfg]] = dict()
        self._group_obs_concatenate: dict[str, bool] = dict()
        self._group_obs_concatenate_dim: dict[str, int] = dict()

        self._group_obs_term_history_buffer: dict[str, dict[str, CircularBuffer]] = dict()
        self._group_obs_term_delay_buffer: dict[str, dict[str, DelayBuffer]] = dict()
        # create a list to store modifiers that are classes
        # we store it as a separate list to only call reset on them and prevent unnecessary calls
        self._group_obs_class_modifiers: list[modifiers.ModifierBase] = list()

        # make sure the simulation is playing since we compute obs dims which needs asset quantities
        if not self._env.sim.is_playing():
            raise RuntimeError(
                "Simulation is not playing. Observation manager requires the simulation to be playing"
                " to compute observation dimensions. Please start the simulation before using the"
                " observation manager."
            )

        # check if config is dict already
        if isinstance(self.cfg, dict):
            group_cfg_items = self.cfg.items()
        else:
            group_cfg_items = self.cfg.__dict__.items()
        # iterate over all the groups
        for group_name, group_cfg in group_cfg_items:
            # check for non config
            if group_cfg is None:
                continue
            # check if the term is a curriculum term
            if not isinstance(group_cfg, ObservationGroupCfg):
                raise TypeError(
                    f"Observation group '{group_name}' is not of type 'ObservationGroupCfg'."
                    f" Received: '{type(group_cfg)}'."
                )
            # initialize list for the group settings
            self._group_obs_term_names[group_name] = list()
            self._group_obs_term_dim[group_name] = list()
            self._group_obs_term_cfgs[group_name] = list()
            self._group_obs_class_term_cfgs[group_name] = list()
            group_entry_history_buffer: dict[str, CircularBuffer] = dict()
            group_entry_delay_buffer: dict[str, DelayBuffer] = dict()
            # read common config for the group
            self._group_obs_concatenate[group_name] = group_cfg.concatenate_terms
            self._group_obs_concatenate_dim[group_name] = (
                group_cfg.concatenate_dim + 1 if group_cfg.concatenate_dim >= 0 else group_cfg.concatenate_dim
            )
            # check if config is dict already
            if isinstance(group_cfg, dict):
                group_cfg_items = group_cfg.items()
            else:
                group_cfg_items = group_cfg.__dict__.items()
            # iterate over all the terms in each group
            for term_name, term_cfg in group_cfg_items:
                # skip non-obs settings
                if term_name in [
                    "enable_corruption",
                    "concatenate_terms",
                    "history_length",
                    "flatten_history_dim",
                    "concatenate_dim",
                ]:
                    continue
                # check for non config
                if term_cfg is None:
                    continue
                if not isinstance(term_cfg, ObservationTermCfg):
                    raise TypeError(
                        f"Configuration for the term '{term_name}' is not of type ObservationTermCfg."
                        f" Received: '{type(term_cfg)}'."
                    )
                # resolve common terms in the config
                self._resolve_common_term_cfg(f"{group_name}/{term_name}", term_cfg, min_argc=1)

                # check noise settings
                if not group_cfg.enable_corruption:
                    term_cfg.noise = None
                # check group history params and override terms
                if group_cfg.history_length is not None:
                    term_cfg.history_length = group_cfg.history_length
                    term_cfg.flatten_history_dim = group_cfg.flatten_history_dim
                # add term config to list to list
                self._group_obs_term_names[group_name].append(term_name)
                self._group_obs_term_cfgs[group_name].append(term_cfg)

                # call function the first time to fill up dimensions
                obs_dims = tuple(term_cfg.func(self._env, **term_cfg.params).shape)

                # if scale is set, check if single float or tuple
                if term_cfg.scale is not None:
                    if not isinstance(term_cfg.scale, (float, int, tuple)):
                        raise TypeError(
                            f"Scale for observation term '{term_name}' in group '{group_name}'"
                            f" is not of type float, int or tuple. Received: '{type(term_cfg.scale)}'."
                        )
                    if isinstance(term_cfg.scale, tuple) and len(term_cfg.scale) != obs_dims[1]:
                        raise ValueError(
                            f"Scale for observation term '{term_name}' in group '{group_name}'"
                            f" does not match the dimensions of the observation. Expected: {obs_dims[1]}"
                            f" but received: {len(term_cfg.scale)}."
                        )

                    # cast the scale into torch tensor
                    term_cfg.scale = torch.tensor(term_cfg.scale, dtype=torch.float, device=self._env.device)

                # prepare modifiers for each observation
                if term_cfg.modifiers is not None:
                    # initialize list of modifiers for term
                    for mod_cfg in term_cfg.modifiers:
                        # check if class modifier and initialize with observation size when adding
                        if isinstance(mod_cfg, modifiers.ModifierCfg):
                            # to list of modifiers
                            if inspect.isclass(mod_cfg.func):
                                if not issubclass(mod_cfg.func, modifiers.ModifierBase):
                                    raise TypeError(
                                        f"Modifier function '{mod_cfg.func}' for observation term '{term_name}'"
                                        f" is not a subclass of 'ModifierBase'. Received: '{type(mod_cfg.func)}'."
                                    )
                                mod_cfg.func = mod_cfg.func(cfg=mod_cfg, data_dim=obs_dims, device=self._env.device)

                                # add to list of class modifiers
                                self._group_obs_class_modifiers.append(mod_cfg.func)
                        else:
                            raise TypeError(
                                f"Modifier configuration '{mod_cfg}' of observation term '{term_name}' is not of"
                                f" required type ModifierCfg, Received: '{type(mod_cfg)}'"
                            )

                        # check if function is callable
                        if not callable(mod_cfg.func):
                            raise AttributeError(
                                f"Modifier '{mod_cfg}' of observation term '{term_name}' is not callable."
                                f" Received: {mod_cfg.func}"
                            )

                        # check if term's arguments are matched by params
                        term_params = list(mod_cfg.params.keys())
                        args = inspect.signature(mod_cfg.func).parameters
                        args_with_defaults = [arg for arg in args if args[arg].default is not inspect.Parameter.empty]
                        args_without_defaults = [arg for arg in args if args[arg].default is inspect.Parameter.empty]
                        args = args_without_defaults + args_with_defaults
                        # ignore first two arguments for env and env_ids
                        # Think: Check for cases when kwargs are set inside the function?
                        if len(args) > 1:
                            if set(args[1:]) != set(term_params + args_with_defaults):
                                raise ValueError(
                                    f"Modifier '{mod_cfg}' of observation term '{term_name}' expects"
                                    f" mandatory parameters: {args_without_defaults[1:]}"
                                    f" and optional parameters: {args_with_defaults}, but received: {term_params}."
                                )
                # prepare noise model classes
                if term_cfg.noise is not None and isinstance(term_cfg.noise, noise.NoiseModelCfg):
                    noise_model_cls = term_cfg.noise.class_type
                    if not issubclass(noise_model_cls, noise.NoiseModel):
                        raise TypeError(
                            f"Class type for observation term '{term_name}' NoiseModelCfg"
                            f" is not a subclass of 'NoiseModel'. Received: '{type(noise_model_cls)}'."
                        )
                    # initialize func to be the noise model class instance
                    term_cfg.noise.func = noise_model_cls(
                        term_cfg.noise, num_envs=self._env.num_envs, device=self._env.device
                    )
                    self._group_obs_class_instances.append(term_cfg.noise.func)
                               
                # create delay buffer if delay settings are set
                delay_max_lag = getattr(term_cfg, 'delay_max_lag', 0)
                if delay_max_lag > 0:
                    group_entry_delay_buffer[term_name] = DelayBuffer(
                        min_lag=term_cfg.delay_min_lag,
                        max_lag=term_cfg.delay_max_lag,
                        batch_size=self._env.num_envs,
                        device=self._env.device,
                        per_env=term_cfg.delay_per_env,
                        hold_prob=term_cfg.delay_hold_prob,
                        update_period=term_cfg.delay_update_period,
                        per_env_phase=term_cfg.delay_per_env_phase,
                    )

                # create history buffers and calculate history term dimensions
                if term_cfg.history_length > 0:
                    group_entry_history_buffer[term_name] = CircularBuffer(
                        max_len=term_cfg.history_length, batch_size=self._env.num_envs, device=self._env.device
                    )
                    old_dims = list(obs_dims)
                    old_dims.insert(1, term_cfg.history_length)
                    obs_dims = tuple(old_dims)
                    if term_cfg.flatten_history_dim:
                        obs_dims = (obs_dims[0], np.prod(obs_dims[1:]))

                self._group_obs_term_dim[group_name].append(obs_dims[1:])

                # add term in a separate list if term is a class
                if isinstance(term_cfg.func, ManagerTermBase):
                    self._group_obs_class_term_cfgs[group_name].append(term_cfg)
                    # call reset (in-case above call to get obs dims changed the state)
                    term_cfg.func.reset()
            # add history buffers for each group
            self._group_obs_term_history_buffer[group_name] = group_entry_history_buffer
            self._group_obs_term_delay_buffer[group_name] = group_entry_delay_buffer
