# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Opt-in deployment-compatible observations/actions for ONNX actor fine-tuning."""

from isaaclab.envs import mdp
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from ...mdp import reference_observations as reference

SDK_JOINT_NAMES = (
    [
        f"{side}_{joint}_joint"
        for side in ("left", "right")
        for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
    ]
    + [f"waist_{axis}_joint" for axis in ("yaw", "roll", "pitch")]
    + [
        f"{side}_{joint}_joint"
        for side in ("left", "right")
        for joint in (
            "shoulder_pitch",
            "shoulder_roll",
            "shoulder_yaw",
            "elbow",
            "wrist_roll",
            "wrist_pitch",
            "wrist_yaw",
        )
    ]
)
BASE_OBSERVATIONS = (
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "joint_pos_rel",
    "joint_vel_rel",
    "last_action",
)


@configclass
class DeploymentPolicyObservationsCfg(ObservationGroupCfg):
    # Term-major history: five successive frames per term, exactly as deploy.yaml.
    base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel, history_length=5, scale=0.2)
    projected_gravity = ObservationTermCfg(func=mdp.projected_gravity, history_length=5)
    velocity_commands = ObservationTermCfg(
        func=mdp.generated_commands, params={"command_name": "base_velocity"}, history_length=5
    )
    joint_pos_rel = ObservationTermCfg(func=mdp.joint_pos_rel, history_length=5)
    joint_vel_rel = ObservationTermCfg(func=mdp.joint_vel_rel, history_length=5, scale=0.05)
    last_action = ObservationTermCfg(func=mdp.last_action, history_length=5)
    # Extra channels have zero weights at initialization; PPO can learn them.
    stride_length = ObservationTermCfg(func=reference.stride_length)
    gait_phase = ObservationTermCfg(func=reference.gait_phase)
    reference_joint_pos = ObservationTermCfg(func=reference.reference_joint_positions)
    reference_joint_vel = ObservationTermCfg(func=reference.reference_joint_velocities)
    reference_root_orientation = ObservationTermCfg(func=reference.reference_root_orientation)
    reference_root_height = ObservationTermCfg(func=reference.reference_root_height)
    step_length = None  # Opt-in appended channel; existing checkpoints retain their input layout.

    def __post_init__(self):
        self.concatenate_terms = True
        self.enable_corruption = False


@configclass
class DeploymentObservationsCfg:
    policy = DeploymentPolicyObservationsCfg()


def configure_onnx_finetuning(env_cfg, agent_cfg, deployment: dict) -> list[str]:
    """Adapt only this config instance to the published actor's deployment contract.

    Reference command, rewards, reset logic and Newton physics remain inherited
    from the reference preset. The default/reference presets are not modified.
    """
    if tuple(deployment["observations"]) != BASE_OBSERVATIONS:
        raise ValueError("Unsupported ONNX observation schema/order; expected the published six-term G1 policy.")
    mapping = deployment["joint_ids_map"]
    if sorted(mapping) != list(range(29)):
        raise ValueError("Expected a permutation of the 29 SDK joint indices.")
    names = [SDK_JOINT_NAMES[i] for i in mapping]
    env_cfg.observations = DeploymentObservationsCfg()
    if deployment.get("mpc_step_length_observation", False):
        env_cfg.observations.policy.step_length = ObservationTermCfg(func=reference.step_length)

    for name, spec in deployment["observations"].items():
        if spec["history_length"] != 5:
            raise ValueError("Expected five frames of history for every deployment observation.")
        term = getattr(env_cfg.observations.policy, name)
        scale = spec.get("scale", 1.0)
        term.scale = tuple(scale) if isinstance(scale, list) else scale
        term.clip = tuple(spec["clip"]) if spec.get("clip") is not None else None
    for name in ("joint_pos_rel", "joint_vel_rel"):
        getattr(env_cfg.observations.policy, name).params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=names, preserve_order=True)
        }
    default = deployment["default_joint_pos"]
    env_cfg.scene.robot.init_state.joint_pos = dict(zip(names, default, strict=True))
    env_cfg.scene.robot.init_state.joint_vel = {".*": 0.0}
    spec = deployment["actions"]["JointPositionAction"]
    action = env_cfg.actions.joint_pos
    action.joint_names = names
    action.preserve_order = True
    action.scale = dict(zip(names, spec["scale"], strict=True))
    action.offset = dict(zip(names, spec["offset"], strict=True))
    action.use_default_offset = False
    if spec.get("clip") is not None:
        raise ValueError("Deployment action clipping is unsupported; expected null.")
    motors = env_cfg.scene.robot.actuators["motors"]
    motors.stiffness = dict(zip(SDK_JOINT_NAMES, deployment["stiffness"], strict=True))
    motors.damping = dict(zip(SDK_JOINT_NAMES, deployment["damping"], strict=True))
    if abs(env_cfg.sim.dt * env_cfg.decimation - deployment["step_dt"]) > 1e-8:
        raise ValueError("Policy timestep differs from the ONNX deployment timestep.")
    agent_cfg.actor.hidden_dims = [512, 256, 128]
    agent_cfg.actor.activation = "elu"
    agent_cfg.actor.obs_normalization = False
    return names
