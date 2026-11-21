# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record G1 reference-policy inference and evaluate MPC, LIPM, or LIPM-plus-hands tracking."""

import argparse
import csv
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path



import numpy as np
import torch
import warp as wp
import yaml
from scipy.spatial.transform import Rotation

wp.config.enable_backward = False

from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

from isaaclab.app import launch_simulation
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from isaaclab.utils import to_dict
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.core.velocity.config.g1.mpc_model import MPC_URDF
from isaaclab_tasks.core.velocity.mdp import lipm_mdp
from isaaclab_tasks.core.velocity.mdp.lipm_commands import robot_com
from isaaclab_tasks.utils import resolve_task_config


class SavedConfigLoader(yaml.SafeLoader):
    """Accept only the tuple/slice tags emitted by local training configuration dumps."""


SavedConfigLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple", lambda loader, node: tuple(loader.construct_sequence(node))
)
SavedConfigLoader.add_constructor(
    "tag:yaml.org,2002:python/object/apply:builtins.slice",
    lambda loader, node: slice(*loader.construct_sequence(node)),
)


def load_playback_config(checkpoint: Path, preset: str = "auto", motion: Path | None = None, device: str | None = None):
    """Restore saved settings into the correct concrete preset before creating managers."""
    if not checkpoint.is_file():
        raise ValueError(f"Checkpoint does not exist: {checkpoint}")
    params = checkpoint.parent / "params"
    saved_env = yaml.load((params / "env.yaml").read_text(), Loader=SavedConfigLoader)
    command_type = saved_env["commands"]["base_velocity"]["class_type"].split(":")[-1]
    if command_type == "LipmMotionCommand":
        policy_obs = saved_env["observations"]["policy"]
        if policy_obs.get("hand_targets_and_errors") is not None:
            detected = "lipm_online_hands"
        else:
            detected = "lipm_hands" if policy_obs.get("arm_targets") is not None else "lipm"
    elif command_type == "ReferenceMotionCommand":
        detected = "reference"
    else:
        raise ValueError(f"Unsupported saved reference command: {command_type}")
    if preset != "auto" and preset != detected:
        raise ValueError(f"Checkpoint uses presets={detected}, not requested presets={preset}")
    cfg, agent = resolve_task_config(
        "Isaac-Velocity-Flat-G1", "rsl_rl_cfg_entry_point", overrides=[f"presets={detected}"]
    )
    deployment_path = params / "deployment.yaml"
    if deployment_path.is_file():
        from isaaclab_tasks.core.velocity.config.g1.onnx_finetune import configure_onnx_finetuning

        configure_onnx_finetuning(cfg, agent, yaml.safe_load(deployment_path.read_text()))
    # Restore legacy material/event types before loading older MPC configurations.
    if "compliant_contact_stiffness" in saved_env["sim"]["physics_material"]:
        from isaaclab_tasks.core.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg

        legacy = G1FlatEnvCfg()
        cfg.sim.physics_material = legacy.sim.physics_material
        cfg.scene.terrain.physics_material = legacy.scene.terrain.physics_material
        if saved_env["events"].get("physics_material") is not None:
            cfg.events.physics_material = legacy.events.physics_material
    # Older LIPM runs used the generic slide reward, now disabled by the preset.
    # Recreate its typed configuration so nested saved fields can be restored.
    if saved_env.get("rewards", {}).get("feet_slide") is not None and cfg.rewards.feet_slide is None:
        from isaaclab_tasks.core.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg

        cfg.rewards.feet_slide = G1FlatEnvCfg().rewards.feet_slide
    # Recreate the optional typed term before restoring a residual-policy configuration.
    if saved_env.get("observations", {}).get("policy", {}).get("ik_joint_errors") is not None:
        from isaaclab.managers import ObservationTermCfg
        from isaaclab_tasks.core.velocity.mdp.hand_commands import ik_joint_errors

        cfg.observations.policy.ik_joint_errors = ObservationTermCfg(func=ik_joint_errors)
    if saved_env.get("observations", {}).get("policy", {}).get("heading_feedback") is not None:
        from isaaclab.managers import ObservationTermCfg
        from isaaclab_tasks.core.velocity.mdp.hand_commands import heading_feedback

        cfg.observations.policy.heading_feedback = ObservationTermCfg(func=heading_feedback)
    cfg.from_dict(saved_env)
    agent.from_dict(yaml.load((params / "agent.yaml").read_text(), Loader=SavedConfigLoader))
    if motion is not None:
        if not motion.is_file() or motion.suffix != ".npz":
            raise ValueError("--motion must name an existing NPZ clip")
        cfg.commands.base_velocity.motion_directory = str(motion.resolve())
    cfg.play_mode()
    cfg.scene.num_envs = 1
    if device is not None:
        cfg.sim.device = device
    agent.device = cfg.sim.device
    cfg.seed = agent.seed
    return cfg, agent, detected


def load_gait_labels(motion_path: Path):
    """Normalize MPC labels and LIP sample-held gait schedules to one timestamped format."""
    if not motion_path.is_file():
        raise ValueError("Gait evaluation requires a single reference NPZ")
    with np.load(motion_path, allow_pickle=False) as data:
        if "schema_version" not in data:
            labels = {key: data[key].copy() for key in ("step_conditioning", "condition_stage", "is_transition")}
            labels["time"] = np.arange(len(labels["condition_stage"])) / float(data["fps"].item())
            return labels
        time = data["time"].astype(float)
        length = data["desired_step_length"]
        duration = np.broadcast_to(data["step_duration"], time.shape)
        schedule = data["gait_schedule"] if "gait_schedule" in data else np.empty((0, 3))
        stages = np.zeros(len(time), dtype=int)
        transition = np.zeros(len(time), dtype=bool)
        if len(schedule):
            stages = (np.searchsorted(schedule[:, 0], time, side="right") - 1).clip(0, len(schedule) - 1)
            transition = ~(np.isclose(length, schedule[stages, 1]) & np.isclose(duration, schedule[stages, 2]))
        if "startup_duration" in data:
            transition |= time - time[0] < float(data["startup_duration"].item())
        return {
            "time": time - time[0],
            "step_conditioning": np.column_stack((length, duration)),
            "condition_stage": stages,
            "is_transition": transition.astype(int),
        }


def resolve_hand_points(body_names: list[str], hand_frames: list[str], urdf: Path = MPC_URDF):
    """Resolve each hand-frame origin to a simulated body plus a fixed local offset.

    Matched G1 assets merge the rubber-hand fixed links into wrist bodies. Follow
    only fixed joints, so a missing articulated body cannot silently use a wrong offset.
    """
    joints = {joint.find("child").get("link"): joint for joint in ET.parse(urdf).getroot().findall("joint")}
    indices, offsets = [], []
    for frame in hand_frames:
        current, transform, visited = frame, np.eye(4), set()
        while current not in body_names:
            if current in visited or current not in joints or joints[current].get("type") != "fixed":
                raise ValueError(f"Cannot resolve hand frame {frame} through fixed joints to a simulated body")
            visited.add(current)
            joint = joints[current]
            origin = joint.find("origin")
            local = np.eye(4)
            if origin is not None:
                local[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                local[:3, :3] = Rotation.from_euler(
                    "xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
                ).as_matrix()
            transform = local @ transform
            current = joint.find("parent").get("link")
        indices.append(body_names.index(current))
        offsets.append(transform[:3, 3])
    return indices, np.asarray(offsets)


class LipmTrackingEvaluator:
    """Measure current-state errors before stepping, avoiding lookahead and auto-reset bias."""

    def __init__(self, command):
        self.command = command
        self.robot = command.robot
        if command.has_arm_reference:
            self.torso_id = self.robot.body_names.index(command.library.hand_target_frame)
            self.hand_ids, offsets = resolve_hand_points(self.robot.body_names, command.library.hand_frame_names)
            self.hand_offsets = torch.tensor(offsets, dtype=torch.float32, device=command.device)
            self.waist_ids = [i for i, name in enumerate(self.robot.joint_names) if name.startswith("waist_")]

    def sample(self, episode_length: torch.Tensor, step_dt: float):
        command, robot = self.command, self.robot
        times = command.start_times + episode_length * step_dt
        ref = command.library.sample(command.clip_ids, times)
        q = command.rotation
        desired_com = quat_apply(q, ref["com_pos"] - command.anchor) + command.origin
        desired_feet = (
            quat_apply(q[:, None].expand(-1, 2, -1), ref["foot_pos"] - command.anchor[:, None])
            + command.origin[:, None]
        )
        com, vel = robot_com(robot)
        feet_error = (command.foot_positions() - desired_feet).norm(dim=-1)
        row = {
            "reference_time_s": float(times[0]),
            "clip_id": int(command.clip_ids[0]),
            "com_position_error_m": (com - desired_com).norm(dim=-1)[0].item(),
            "com_velocity_error_m_s": (vel - quat_apply(q, ref["com_vel"])).norm(dim=-1)[0].item(),
            "left_foot_position_error_m": feet_error[0, 0].item(),
            "right_foot_position_error_m": feet_error[0, 1].item(),
        }
        if command.has_arm_reference:
            for field, units in (("pos", "rad"), ("vel", "rad_s")):
                actual = getattr(robot.data, f"joint_{field}").torch[:, command.arm_joint_ids]
                row[f"arm_joint_{field}_rmse_{units}"] = (
                    (actual - ref[f"arm_joint_{field}"]).square().mean(-1).sqrt()[0].item()
                )
            body_pos, body_quat = robot.data.body_link_pos_w.torch, robot.data.body_link_quat_w.torch
            hand_world = body_pos[:, self.hand_ids] + quat_apply(
                body_quat[:, self.hand_ids], self.hand_offsets.unsqueeze(0)
            )
            hand_torso = quat_apply_inverse(
                body_quat[:, self.torso_id, None].expand(-1, len(self.hand_ids), -1),
                hand_world - body_pos[:, self.torso_id, None],
            )
            torso_q = body_quat[:, self.torso_id]
            torso_lin = robot.data.body_link_lin_vel_w.torch[:, self.torso_id]
            torso_ang = robot.data.body_link_ang_vel_w.torch[:, self.torso_id]
            for axis, value in zip("xyz", torso_lin[0].cpu().tolist()):
                row[f"torso_world_velocity_{axis}_m_s"] = value
            angles = Rotation.from_quat(torso_q[0].cpu().numpy()).as_euler("xyz")
            for axis, value in zip(("roll", "pitch", "yaw"), angles):
                row[f"torso_{axis}_rad"] = float(value)
            for axis, value in zip("xyz", quat_apply_inverse(torso_q, torso_ang)[0].cpu().tolist()):
                row[f"torso_angular_velocity_{axis}_rad_s"] = value
            row["torso_angular_speed_rad_s"] = torso_ang.norm(dim=-1)[0].item()
            for index in self.waist_ids:
                name = robot.joint_names[index]
                row[f"{name}_position_rad"] = robot.data.joint_pos.torch[0, index].item()
                row[f"{name}_velocity_rad_s"] = robot.data.joint_vel.torch[0, index].item()
            offset_world = quat_apply(body_quat[:, self.hand_ids], self.hand_offsets.unsqueeze(0))
            hand_velocity = robot.data.body_link_lin_vel_w.torch[:, self.hand_ids] + torch.cross(
                robot.data.body_link_ang_vel_w.torch[:, self.hand_ids], offset_world, dim=-1
            )
            carried_rotation = torch.cross(
                torso_ang[:, None].expand_as(hand_world), hand_world - body_pos[:, self.torso_id, None], dim=-1
            )
            relative_velocity_world = hand_velocity - torso_lin[:, None] - carried_rotation
            relative_velocity = quat_apply_inverse(
                torso_q[:, None].expand(-1, len(self.hand_ids), -1), relative_velocity_world
            )
            for index, side in enumerate(command.library.hand_names):
                for kind, vector in (
                    ("world_velocity", hand_velocity),
                    ("torso_relative_velocity", relative_velocity),
                    ("torso_rotation_velocity", carried_rotation),
                ):
                    for axis, value in zip("xyz", vector[0, index].cpu().tolist()):
                        row[f"{side}_hand_{kind}_{axis}_m_s"] = value
                    row[f"{side}_hand_{kind}_speed_m_s"] = vector[0, index].norm().item()
                for axis, value in zip("xyz", hand_world[0, index].cpu().tolist()):
                    row[f"{side}_hand_world_{axis}_m"] = value
                row[f"{side}_hand_position_error_m"] = (
                    (hand_torso - ref["hand_pos_torso"]).norm(dim=-1)[0, index].item()
                )
                for kind, positions in (("actual", hand_torso), ("target", ref["hand_pos_torso"])):
                    for axis, value in zip("xyz", positions[0, index].cpu().tolist()):
                        row[f"{side}_hand_{kind}_{axis}_torso_m"] = value
        return row


def write_tracking_report(output: Path, trace: list[dict]):
    with (output / "reference_tracking.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(trace[0]))
        writer.writeheader()
        writer.writerows(trace)
    metrics = {}
    for key in trace[0]:
        if key.endswith(("_error_m", "_error_m_s", "_rmse_rad", "_rmse_rad_s")):
            values = np.array([row[key] for row in trace])
            metrics[key] = {
                "mean": float(values.mean()),
                "rmse": float(np.sqrt(np.mean(values**2))),
                "max": float(values.max()),
            }
    report = {
        "sample_count": len(trace),
        "sampling_note": (
            "Pre-action states compared with the current reference time, excluding terminal auto-reset states."
        ),
        "hand_frame": "torso_link" if any("_hand_" in key for key in trace[0]) else None,
        "metrics": metrics,
    }
    motion = {}
    for key in trace[0]:
        if key.startswith(("torso_", "waist_")) or "_velocity_" in key or "heading_error" in key:
            values = np.asarray([row[key] for row in trace])
            motion[key] = {
                "mean": float(values.mean()),
                "rms": float(np.sqrt(np.mean(values**2))),
                "min": float(values.min()),
                "max": float(values.max()),
                "p95_abs": float(np.quantile(np.abs(values), 0.95)),
            }
    report["motion_metrics"] = motion
    report["motion_note"] = (
        "Torso Euler angles are world-frame XYZ radians. "
        "torso_angular_velocity_* is torso-frame; "
        "torso_world_angular_velocity_* is world-frame (rad/s). "
        "Waist yaw velocity is joint-relative (rad/s). "
        "Hand world velocity = torso-origin translation + torso rotation cross hand offset "
        "+ arm motion relative to torso. Relative velocity includes rotating-frame correction. "
        "Component speed magnitudes are not additive or causal percentages."
    )
    (output / "tracking_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--motion", type=Path, help="Override the saved reference with one NPZ clip.")
    parser.add_argument(
        "--preset",
        choices=("auto", "reference", "lipm", "lipm_hands", "lipm_online_hands"),
        default="auto",
        help="Infer from saved env.yaml by default; an explicit preset must match the checkpoint.",
    )
    parser.add_argument("--device", help="Override both simulation and policy device, e.g. cuda:0.")
    parser.add_argument("--evaluate-gait", action="store_true")
    parser.add_argument(
        "--hand-orientation", action="store_true", help="Enable torso-relative hand orientation IK for online-hands playback."
    )
    parser.add_argument(
        "--immediate-arm-placement", action="store_true",
        help="Evaluate a direct-arm checkpoint with immediate IK endpoints and zero arm joint velocities.",
    )
    parser.add_argument(
        "--record-full-state", action="store_true",
        help="Save control-rate root pose, measured joint angles, joint commands, and hand targets for MuJoCo diagnostics.",
    )
    parser.add_argument("--duration", type=float, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh output directory.")
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("Duration must be finite and positive.")
    checkpoint = args.checkpoint.resolve()
    try:
        cfg, agent, preset = load_playback_config(checkpoint, args.preset, args.motion, args.device)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.immediate_arm_placement:
        if preset != "lipm_online_hands" or not cfg.actions.joint_pos.direct_arm_placement:
            parser.error("--immediate-arm-placement requires a direct-arm online-hands checkpoint")
        cfg.actions.joint_pos.immediate_arm_placement = True
    if args.hand_orientation:
        if preset != "lipm_online_hands":
            parser.error("--hand-orientation requires an online-hands checkpoint")
        cfg.actions.joint_pos.track_orientation = True
    if preset == "lipm_online_hands":
        metadata = torch.load(checkpoint, map_location="cpu", weights_only=False).get("infos") or {}
        control_steps = metadata.get("online_hands_control_steps")
        action_cfg = cfg.actions.joint_pos
        hand_cfg = cfg.commands.hand_position
        if action_cfg.ramp_steps and control_steps is None:
            parser.error("Online-hands checkpoint is missing saved IK ramp progress")
        if "online_hands_ik_influence" in metadata:
            action_cfg.ik_influence = metadata["online_hands_ik_influence"]
        elif action_cfg.ramp_steps:
            action_cfg.ik_influence *= min(1.0, control_steps / action_cfg.ramp_steps)
        action_cfg.ramp_steps = 0
        action_cfg.follow_reach_curriculum = False
        if "online_hands_reach_fraction" in metadata:
            fraction = metadata["online_hands_reach_fraction"]
        elif hand_cfg.curriculum_steps:
            if control_steps is None and hand_cfg.reach_fraction != hand_cfg.initial_reach_fraction:
                parser.error("Online-hands checkpoint is missing saved reach curriculum progress")
            progress = min(1.0, (control_steps or 0) / hand_cfg.curriculum_steps)
            fraction = hand_cfg.initial_reach_fraction + progress * (
                hand_cfg.reach_fraction - hand_cfg.initial_reach_fraction
            )
        else:
            fraction = hand_cfg.reach_fraction
        hand_cfg.initial_reach_fraction = hand_cfg.reach_fraction = fraction
        hand_cfg.curriculum_steps = 0
    torch.manual_seed(agent.seed)
    steps = round(args.duration / (cfg.sim.dt * cfg.decimation))
    if steps < (2 if args.evaluate_gait else 1):
        parser.error("Duration is shorter than the required number of control steps.")
    motion_path = Path(cfg.commands.base_velocity.motion_directory)
    labels = load_gait_labels(motion_path) if args.evaluate_gait else None
    args.output.mkdir(parents=True)
    cfg.sim.render_interval = cfg.decimation
    cfg.sim.default_visualizer_cfg = NewtonGLVisualizerCfg(
        headless=True,
        eye=(2.5, -3, 1.6),
        lookat=(0, 0, 0.65),
        window_width=960,
        window_height=720,
        streaming_view=False,
    )
    cfg.video_recorders = [
        VideoRecorderCfg(
            source="viz:newton_gl",
            output_dir=str(args.output.resolve()),
            output_filename_prefix=f"g1_reference_{checkpoint.stem}",
            video_length=steps,
            video_interval=0,
            step_offset=0,
            fps=round(1 / (cfg.sim.dt * cfg.decimation)),
        )
    ]
    with launch_simulation(cfg, {"headless": True}):
        from rsl_rl.runners import OnPolicyRunner

        from isaaclab.envs import ManagerBasedRLEnv

        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

        env = ManagerBasedRLEnv(cfg=cfg)
        try:
            wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
            runner = OnPolicyRunner(wrapped, to_dict(agent), log_dir=None, device=agent.device)
            runner.load(str(checkpoint))
            policy = runner.get_inference_policy(device=env.device)
            observations = wrapped.get_observations()
            resets = 0
            termination_events = []
            trace = []
            tracking_trace = []
            recorded_frames = []
            recorded_targets = []
            recorded_orientations = []
            recorded_joint_names = None
            command_term = env.command_manager.get_term("base_velocity")
            tracker = LipmTrackingEvaluator(command_term) if preset in ("lipm", "lipm_hands", "lipm_online_hands") else None
            if args.evaluate_gait:
                robot = env.scene["robot"]
                sensor = env.scene["contact_forces"]
                feet = ["left_ankle_roll_link", "right_ankle_roll_link"]
                foot_ids = [robot.body_names.index(n) for n in feet]
                contact_ids = [sensor.body_names.index(n) for n in feet]
            with torch.inference_mode():
                for step in range(steps):
                    pos = env.scene["robot"].data.root_pos_w.torch[0].cpu().tolist()
                    env.sim.set_camera_view(eye=(pos[0] + 2.5, pos[1] - 3, 1.6), target=(pos[0], pos[1], 0.65))
                    if tracker is not None:
                        tracking_row = tracker.sample(env.episode_length_buf, env.step_dt)
                        if preset == "lipm_online_hands":
                            tracking_row["root_heading_error_rad"] = float(lipm_mdp.heading_error(env)[0])
                            tracking_row["torso_heading_error_rad"] = float(
                                lipm_mdp.heading_error(env, body_name="torso_link")[0]
                            )
                            robot = env.scene["robot"]
                            torso_id = robot.body_names.index("torso_link")
                            angular_local = lipm_mdp.torso_angular_velocity(env)[0]
                            angular_world = robot.data.body_link_ang_vel_w.torch[0, torso_id]
                            for axis, value in zip("xyz", angular_local.cpu().tolist()):
                                tracking_row[f"torso_angular_velocity_{axis}_rad_s"] = value
                            for axis, value in zip("xyz", angular_world.cpu().tolist()):
                                tracking_row[f"torso_world_angular_velocity_{axis}_rad_s"] = value
                            tracking_row["torso_angular_speed_rad_s"] = float(angular_local.norm())
                            waist_id = robot.joint_names.index("waist_yaw_joint")
                            tracking_row["waist_yaw_position_rad"] = float(robot.data.joint_pos.torch[0, waist_id])
                            tracking_row["waist_yaw_velocity_rad_s"] = float(robot.data.joint_vel.torch[0, waist_id])
                            # Compare measured joints with the command held over the preceding physics step.
                            # The initial/reset sample has no preceding policy command.
                            tracking_row["arm_joint_command_valid"] = bool(env.episode_length_buf[0] > 0)
                            arm_suffixes = (
                                "shoulder_pitch_joint", "shoulder_roll_joint", "shoulder_yaw_joint",
                                "elbow_joint", "wrist_roll_joint", "wrist_pitch_joint", "wrist_yaw_joint",
                            )
                            for side in ("left", "right"):
                                joint_names = [f"{side}_{suffix}" for suffix in arm_suffixes]
                                joint_ids = [robot.joint_names.index(name) for name in joint_names]
                                measured = robot.data.joint_pos.torch[0, joint_ids].cpu().tolist()
                                commanded = robot.data.joint_pos_target.torch[0, joint_ids].cpu().tolist()
                                for name, actual_angle, target_angle in zip(joint_names, measured, commanded):
                                    tracking_row[f"{name}_actual_rad"] = actual_angle
                                    tracking_row[f"{name}_commanded_rad"] = target_angle
                                    tracking_row[f"{name}_tracking_error_rad"] = actual_angle - target_angle
                            # Implicit-actuator torque telemetry is a PD estimate, not a force-sensor reading.
                            # Sampled at control rate; peaks between samples may be missed.
                            for side in ("left", "right"):
                                joint_names = [f"{side}_{suffix}" for suffix in arm_suffixes]
                                joint_ids = [robot.joint_names.index(name) for name in joint_names]
                                computed = robot.actuators.computed_effort.torch[0, joint_ids].cpu().tolist()
                                applied = robot.actuators.applied_effort.torch[0, joint_ids].cpu().tolist()
                                limits = robot.data.joint_effort_limits.torch[0, joint_ids].cpu().tolist()
                                for name, requested, delivered, limit in zip(joint_names, computed, applied, limits):
                                    tracking_row[f"{name}_computed_torque_nm"] = requested
                                    tracking_row[f"{name}_applied_torque_nm"] = delivered
                                    tracking_row[f"{name}_computed_minus_applied_torque_nm"] = requested - delivered
                                    tracking_row[f"{name}_torque_limit_nm"] = limit
                                    tracking_row[f"{name}_torque_margin_nm"] = limit - abs(delivered)
                            hands = env.command_manager.get_term("hand_position")
                            actual = hands.positions()
                            torso_pos = robot.data.body_link_pos_w.torch[:, torso_id]
                            torso_quat = robot.data.body_link_quat_w.torch[:, torso_id]
                            rotation = torso_quat[:, None].expand(-1, 2, -1)

                            actual_world = torso_pos[:, None] + quat_apply(rotation, actual)
                            target_world = torso_pos[:, None] + quat_apply(rotation, hands.targets)



                            for index, side in enumerate(("left", "right")):
                                actual_z = actual_world[0, index, 2].item()
                                target_z = target_world[0, index, 2].item()

                                tracking_row[f"{side}_hand_actual_z_world_m"] = actual_z
                                tracking_row[f"{side}_hand_target_z_world_m"] = target_z
                                tracking_row[f"{side}_hand_z_world_error_m"] = actual_z - target_z

                                error = (hands.targets[:, index] - actual[:, index]).norm(dim=-1)
                                tracking_row[f"{side}_hand_position_error_m"] = error[0].item()
                                for kind, values in (("actual", actual), ("target", hands.targets)):
                                    for axis, value in zip("xyz", values[0, index].cpu().tolist()):
                                        tracking_row[f"{side}_hand_{kind}_{axis}_torso_m"] = value
                            tracking_row["ik_influence"] = env.action_manager.get_term("joint_pos").ik_influence
                            tracking_row["reach_fraction"] = float(hands.episode_reach_fraction[0])
                            nominal = hands.targets.new_tensor((hands.cfg.left_position, hands.cfg.right_position))
                            for index, side in enumerate(("left", "right")):
                                tracking_row[f"{side}_hand_target_reach_distance_m"] = float(
                                    (hands.targets[0, index] - nominal[index]).norm()
                                )
                        tracking_row.update(time_s=step * env.step_dt, episode=resets)
                        tracking_trace.append(tracking_row)
                    if args.evaluate_gait:
                        # Sample before stepping: these states have not been replaced by auto-reset.
                        reference_time = (
                            float(command_term.start_times[0]) + int(env.episode_length_buf[0]) * env.step_dt
                        )
                        frame = int(
                            np.clip(
                                np.searchsorted(labels["time"], reference_time, side="right") - 1,
                                0,
                                len(labels["time"]) - 1,
                            )
                        )
                        stage = int(labels["condition_stage"][frame])
                        applied = labels["step_conditioning"][frame]
                        x, y, z, w = robot.data.root_link_quat_w.torch[0].cpu().tolist()
                        row = {
                            "time_s": step * env.step_dt,
                            "segment": resets * (int(labels["condition_stage"].max()) + 1) + stage + 1,
                            "target_vx_m_s": float(command_term.command[0, 0]),
                            "policy_input_vx_m_s": float(command_term.command[0, 0]),
                            "actual_vx_m_s": robot.data.root_lin_vel_b.torch[0, 0].item(),
                            "height_m": robot.data.root_link_pos_w.torch[0, 2].item(),
                            "upright": -robot.data.projected_gravity_b.torch[0, 2].item(),
                        }
                        for side, index in zip(("left", "right"), foot_ids):
                            for axis, value in zip(
                                ("x", "y", "z"), robot.data.body_link_pos_w.torch[0, index].cpu().tolist()
                            ):
                                row[f"{side}_foot_{axis}_m"] = value
                        for side, index in zip(("left", "right"), contact_ids):
                            row[f"{side}_foot_normal_force_n"] = sensor.data.net_normal_forces_w.torch[
                                0, index, 2
                            ].item()
                        row.update(
                            root_x_m=pos[0],
                            root_y_m=pos[1],
                            heading_rad=math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),
                            yaw_rate_rad_s=robot.data.root_ang_vel_w.torch[0, 2].item(),
                            actual_vy_body_m_s=robot.data.root_lin_vel_b.torch[0, 1].item(),
                            actual_vx_world_m_s=robot.data.root_lin_vel_w.torch[0, 0].item(),
                            actual_vy_world_m_s=robot.data.root_lin_vel_w.torch[0, 1].item(),
                            episode=resets,
                            reference_time_s=reference_time,
                            condition_stage=stage,
                            is_transition=int(labels["is_transition"][frame]),
                            commanded_step_length_m=float(applied[0]),
                            commanded_step_duration_s=float(applied[1]),
                        )
                        if tracker is not None:
                            row.update(tracking_row)
                        trace.append(row)
                    if args.record_full_state:
                        robot = env.scene["robot"]
                        if recorded_joint_names is None:
                            recorded_joint_names = list(robot.joint_names)
                        root_pose = torch.cat((
                            robot.data.root_link_pos_w.torch[0],
                            robot.data.root_link_quat_w.torch[0],
                        ))
                        measured = robot.data.joint_pos.torch[0]
                        commanded = robot.data.joint_pos_target.torch[0]
                        recorded_frames.append(torch.cat((root_pose, measured, commanded)).cpu().numpy().copy())
                        hands = env.command_manager.get_term("hand_position")
                        hands._refresh_targets()
                        torso_id = hands.torso_id
                        torso_pos = robot.data.body_link_pos_w.torch[:, torso_id, None]
                        torso_quat = robot.data.body_link_quat_w.torch[:, torso_id, None].expand(-1, 2, -1)
                        recorded_targets.append((torso_pos + quat_apply(torso_quat, hands.targets))[0].cpu().numpy().copy())
                        recorded_orientations.append(
                            quat_mul(torso_quat, hands.orientation_targets)[0].cpu().numpy().copy()
                        )
                    actions = policy(observations)
                    if not torch.isfinite(actions).all():
                        raise RuntimeError("Non-finite policy action.")
                    observations, rewards, done, _ = wrapped.step(actions)
                    if bool(done.any()):
                        terms = [
                            name
                            for name in env.termination_manager.active_terms
                            if bool(env.termination_manager.get_term(name)[0])
                        ]
                        termination_events.append({"time_s": (step + 1) * env.step_dt, "terms": terms})
                    resets += int(done.sum())
                    if not torch.isfinite(observations["policy"]).all():
                        raise RuntimeError("Non-finite observation.")
                    if step % 250 == 0:
                        print(f"Inference step {step}/{steps}; resets={resets}", flush=True)
            if args.record_full_state:
                np.savez_compressed(
                    args.output / "full_state_replay.npz",
                    frames=np.asarray(recorded_frames),
                    joint_names=np.asarray(recorded_joint_names),
                    dt=env.step_dt,
                    targets_world=np.asarray(recorded_targets),
                    orientations_world_xyzw=np.asarray(recorded_orientations),
                )
            (args.output / "inference.json").write_text(
                json.dumps(
                    {
                        "checkpoint": str(checkpoint),
                        "preset": preset,
                        "duration_s": args.duration,
                        "steps": steps,
                        "resets": resets,
                        "termination_events": termination_events,
                        "training": False,
                "immediate_arm_placement_override": args.immediate_arm_placement,
                        "motion": cfg.commands.base_velocity.motion_directory,
                        "random_start": False,
                        "num_envs": 1,
                    },
                    indent=2,
                )
                + "\n"
            )
            if tracking_trace:
                write_tracking_report(args.output, tracking_trace)
        finally:
            env.close()
    if args.evaluate_gait:
        from evaluate_g1_29dof_gait import summarize_gait

        with (args.output / "velocity_tracking.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(trace[0]))
            writer.writeheader()
            writer.writerows(trace)
        failures = [
            event
            for event in termination_events
            if any(term in ("fallen", "bad_orientation") for term in event["terms"])
        ]
        metadata = {
            "physics_backend": "newton_mjwarp",
            "duration_s": args.duration,
            "completed_duration_s": steps * env.step_dt,
            "seed": agent.seed,
            "control_dt_s": env.step_dt,
            "settle_time_s": 2.0,
            "fall": failures or None,
            "termination_events": termination_events,
            "training": False,
                "immediate_arm_placement_override": args.immediate_arm_placement,
            "checkpoint": str(checkpoint),
            "motion": str(motion_path.resolve()),
            "preset": preset,
            "command_note": (
                "Velocity command is the policy's lookahead command; "
                "gait labels and errors use current-state timestamps."
            ),
            "sampling_note": "State sampled before policy action, excluding post-step auto-reset states. "
            "Transitions excluded from gait summary.",
        }
        (args.output / "evaluation_run.json").write_text(json.dumps(metadata, indent=2) + "\n")
        summarize_gait(args.output / "velocity_tracking.csv", 2.0)


if __name__ == "__main__":
    main()
