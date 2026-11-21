# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Expand a walking checkpoint for online hand observations without training.

Writes a fresh output directory, including env/agent params for playback. The
source checkpoint is never modified. Unsupported state shapes fail explicitly.
"""

import argparse
import copy
import json
from dataclasses import fields
from pathlib import Path

import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from play_g1_reference import load_playback_config

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import to_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.core.velocity.mdp.hand_actions import HandIKJointPositionActionCfg
from isaaclab_tasks.core.velocity.mdp.hand_commands import heading_feedback, ik_joint_errors
from isaaclab_tasks.utils import resolve_task_config


def make_model(agent, group, width):
    """Construct the saved actor or critic architecture with a new input width."""
    options = to_dict(getattr(agent, group))
    options.pop("class_name")
    obs = TensorDict({"policy": torch.zeros(1, width)}, batch_size=[1])
    return MLPModel(obs, agent.obs_groups, group, 29 if group == "actor" else 1, **options)


def expand_state(source, target):
    """Copy all state, adding zero input weights and neutral normalization slots."""
    if source.keys() != target.keys():
        raise ValueError("Checkpoint model keys differ from the configured MLP")
    for name, value in source.items():
        dest = target[name]
        if dest.shape == value.shape:
            target[name] = value.clone()
        elif name == "mlp.0.weight" and value.shape[1] == 115 and dest.shape[1] == 143:
            dest.zero_()
            dest[:, :115] = value
        elif name in ("obs_normalizer._mean", "obs_normalizer._var", "obs_normalizer._std"):
            if value.shape != (1, 115) or dest.shape != (1, 143):
                raise ValueError(f"Unsupported normalization shape: {name}")
            dest[:, :115] = value
        else:
            raise ValueError(f"Unsupported tensor shape: {name}: {value.shape} -> {dest.shape}")
    return target


def transfer(checkpoint, output):
    if output.exists():
        raise ValueError("Choose a fresh transfer output directory")
    old, agent, preset = load_playback_config(checkpoint, device="cpu")
    if preset != "lipm":
        raise ValueError("Transfer requires a walking-only lipm checkpoint")
    cfg, _ = resolve_task_config(
        "Isaac-Velocity-Flat-G1", "rsl_rl_cfg_entry_point", overrides=["presets=lipm_online_hands"]
    )
    # Restore original physics, resets, walking command, reward and observation settings.
    cfg.rewards.feet_slide = copy.deepcopy(old.rewards.feet_slide)
    cfg.from_dict(to_dict(old))
    # Saved walking rewards overwrite preset weights: reapply the IK-arm exclusion.
    cfg.rewards.joint_deviation_arms.weight = 0.0
    action = old.actions.joint_pos
    cfg.actions.joint_pos = HandIKJointPositionActionCfg(
        **{f.name: getattr(action, f.name) for f in fields(action) if f.init and f.name != "class_type"}
    )
    cfg.observations.policy.ik_joint_errors = ObsTerm(func=ik_joint_errors)
    cfg.observations.policy.heading_feedback = ObsTerm(func=heading_feedback)
    old_obs, new_obs = to_dict(old.observations.policy), to_dict(cfg.observations.policy)
    if list(new_obs) != list(old_obs) + [
        "hand_targets_and_errors", "ik_joint_errors", "heading_feedback"
    ]:
        raise ValueError("New observations must append to the original observation order")
    if any(new_obs[key] != value for key, value in old_obs.items()):
        raise ValueError("Original observation definitions changed")
    source = torch.load(checkpoint, map_location="cpu", weights_only=False)
    result = copy.deepcopy(source)
    torch.manual_seed(42)
    inputs = torch.randn(64, 115)
    extended = torch.cat((inputs, torch.randn(64, 28)), dim=-1)
    for group in ("actor", "critic"):
        original = make_model(agent, group, 115).eval()
        expanded = make_model(agent, group, 143).eval()
        original.load_state_dict(source[f"{group}_state_dict"])
        expanded.load_state_dict(expand_state(source[f"{group}_state_dict"], expanded.state_dict()))
        with torch.inference_mode():
            a = original(TensorDict({"policy": inputs}, batch_size=[64]))
            b = expanded(TensorDict({"policy": extended}, batch_size=[64]))
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
        result[f"{group}_state_dict"] = expanded.state_dict()
    # Standard runner.load expects an optimizer dict. Keep groups, discard every
    # accumulated moment/step; parameter count and ordering have not changed.
    result["optimizer_state_dict"]["state"] = {}
    result["iter"] = 0
    result["infos"] = {"source_checkpoint": str(checkpoint.resolve()), "transfer": "lipm_online_hands"}
    cfg.sim.device = agent.device = "cuda:0"
    output.mkdir(parents=True)
    dump_yaml(str(output / "params/env.yaml"), cfg)
    dump_yaml(str(output / "params/agent.yaml"), agent)
    torch.save(result, output / "model_0.pt")
    (output / "transfer.json").write_text(json.dumps({
        "source": str(checkpoint.resolve()), "old_observations": 115, "new_observations": 143,
        "actions": 29, "actor_critic_parity": "passed", "optimizer": "fresh", "iteration": 0,
    }, indent=2) + "\n")
    print(f"Transferred and verified actor/critic parity: {output / 'model_0.pt'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    transfer(args.checkpoint.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
