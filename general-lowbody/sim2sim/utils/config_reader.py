from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(slots=True)
class ConfigReader:
    config: dict[str, Any]
    xml_path: str
    simulation_duration: int
    simulation_dt: float
    control_decimation: int
    policy_joints: list[str]
    policy_lower_body_joints: list[str]
    policy_upper_body_joints: list[str]
    upper_body_joint2motor_idx: list[int]
    upper_body_kps: np.ndarray
    upper_body_kds: np.ndarray
    upper_body_default_pos: np.ndarray
    lower_body_joint2motor_idx: list[int]
    lower_body_kps: np.ndarray
    lower_body_kds: np.ndarray
    lower_body_default_pos: np.ndarray
    whole_body_joint2motor_idx: list[int]
    whole_body_default_pos: np.ndarray
    num_upper_body_actions: int
    num_lower_body_actions: int
    num_actions: int
    upper_body_limit_lower: np.ndarray
    upper_body_limit_upper: np.ndarray
    lower_body_limit_lower: np.ndarray
    lower_body_limit_upper: np.ndarray
    soft_limit_factor: float
    upper_body_soft_limit: tuple[np.ndarray, np.ndarray]
    lower_body_soft_limit: tuple[np.ndarray, np.ndarray]
    default_angles: np.ndarray
    ang_vel_scale: float
    dof_pos_scale: float
    dof_vel_scale: float
    upper_body_action_scale: float | np.ndarray
    lower_body_action_scale: float | np.ndarray
    cmd_scale: np.ndarray
    cmd: np.ndarray
    history_len: int
    num_obs: int

    @classmethod
    def from_yaml(cls, config_path: str | Path, actuator_config: str = "mimic") -> "ConfigReader":
        config_path = Path(config_path)
        with config_path.open("r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        actuator = config["actuator_config"][actuator_config]
        whole_body_default_pos = np.array(config["whole_body_default_pos"], dtype=np.float32)
        upper_body_limit_lower = np.asarray(config["upper_body_limit_lower"])
        upper_body_limit_upper = np.asarray(config["upper_body_limit_upper"])
        lower_body_limit_lower = np.asarray(config["lower_body_limit_lower"])
        lower_body_limit_upper = np.asarray(config["lower_body_limit_upper"])
        soft_limit_factor = config["soft_limit_factor"]

        upper_body_action_scale = actuator["upper_body_action_scale"]
        lower_body_action_scale = actuator["lower_body_action_scale"]
        if isinstance(upper_body_action_scale, list):
            upper_body_action_scale = np.array(upper_body_action_scale, dtype=np.float32)
        if isinstance(lower_body_action_scale, list):
            lower_body_action_scale = np.array(lower_body_action_scale, dtype=np.float32)

        return cls(
            config=config,
            xml_path=config["xml_path"],
            simulation_duration=config["simulation_duration"],
            simulation_dt=config["simulation_dt"],
            control_decimation=config["control_decimation"],
            policy_joints=config["policy_joints"],
            policy_lower_body_joints=config["policy_lower_body_joints"],
            policy_upper_body_joints=config["policy_upper_body_joints"],
            upper_body_joint2motor_idx=config["upper_body_joint2motor_idx"],
            upper_body_kps=np.array(actuator["upper_body_kps"], dtype=np.float32),
            upper_body_kds=np.array(actuator["upper_body_kds"], dtype=np.float32),
            upper_body_default_pos=np.array(config["upper_body_default_pos"], dtype=np.float32),
            lower_body_joint2motor_idx=config["lower_body_joint2motor_idx"],
            lower_body_kps=np.array(actuator["lower_body_kps"], dtype=np.float32),
            lower_body_kds=np.array(actuator["lower_body_kds"], dtype=np.float32),
            lower_body_default_pos=np.array(config["lower_body_default_pos"], dtype=np.float32),
            whole_body_joint2motor_idx=config["whole_body_joint2motor_idx"],
            whole_body_default_pos=whole_body_default_pos,
            num_upper_body_actions=config["num_upper_actions"],
            num_lower_body_actions=config["num_lower_actions"],
            num_actions=config["num_actions"],
            upper_body_limit_lower=upper_body_limit_lower,
            upper_body_limit_upper=upper_body_limit_upper,
            lower_body_limit_lower=lower_body_limit_lower,
            lower_body_limit_upper=lower_body_limit_upper,
            upper_body_soft_limit=cls.compute_soft_limit(
                upper_body_limit_lower, upper_body_limit_upper, soft_factor=soft_limit_factor
            ),
            lower_body_soft_limit=cls.compute_soft_limit(
                lower_body_limit_lower, lower_body_limit_upper, soft_factor=soft_limit_factor
            ),
             soft_limit_factor=soft_limit_factor,
            default_angles=whole_body_default_pos.copy(),
            ang_vel_scale=config["ang_vel_scale"],
            dof_pos_scale=config["dof_pos_scale"],
            dof_vel_scale=config["dof_vel_scale"],
            upper_body_action_scale=upper_body_action_scale,
            lower_body_action_scale=lower_body_action_scale,
            cmd_scale=np.array(config["cmd_scale"], dtype=np.float32),
            cmd=np.array(config["cmd_init"], dtype=np.float32),
            num_obs=config["num_obs"],
            history_len=config["history_len"],
        )

    @staticmethod
    def compute_soft_limit(
        lower_limit: np.ndarray,
        upper_limit: np.ndarray,
        soft_factor: float = 0.9,
    ) -> tuple[np.ndarray, np.ndarray]:
        center = (upper_limit + lower_limit) / 2.0
        half_range = (upper_limit - lower_limit) / 2.0
        soft_lower = center - half_range * soft_factor
        soft_upper = center + half_range * soft_factor
        return (soft_lower, soft_upper)
