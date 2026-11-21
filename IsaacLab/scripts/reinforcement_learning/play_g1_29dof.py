# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record Unitree's published 29-DOF G1 velocity policy at 0.2, 0.5 and 1.0 m/s.

Run from the repository root on an RTX GPU:
    HEADLESS=1 uv run --no-sync python scripts/reinforcement_learning/play_g1_29dof.py

The first invocation downloads pinned official weights and USD assets. This is an
inference-only environment; the existing G1 task and checkpoint are untouched.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_REVISION = "4960b84732b0c2ec593dccbfe963fda1bcd7b1e3"
ASSET_REVISION = "323e350252b9c3aee9c40acbfdad84f6ce46a5ac"
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


def download_assets(directory: Path) -> None:
    """Cache the official inference bundle and its four self-contained USD files."""
    policy_base = (
        f"https://raw.githubusercontent.com/unitreerobotics/unitree_rl_lab/{POLICY_REVISION}"
        "/deploy/robots/g1_29dof/config/policy/velocity/v0"
    )
    asset_base = (
        f"https://huggingface.co/datasets/unitreerobotics/unitree_model/resolve/{ASSET_REVISION}"
        "/G1/29dof/usd/g1_29dof_rev_1_0"
    )
    files = {"policy.onnx": f"{policy_base}/exported/policy.onnx", "deploy.yaml": f"{policy_base}/params/deploy.yaml"}
    for relative in (
        "g1_29dof_rev_1_0.usd",
        "configuration/g1_29dof_rev_1_0_base.usd",
        "configuration/g1_29dof_rev_1_0_physics.usd",
        "configuration/g1_29dof_rev_1_0_sensor.usd",
    ):
        files[f"assets/{relative}"] = f"{asset_base}/{relative}"
    manifest = {}
    for relative, url in files.items():
        destination = directory / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            print(f"Downloading {relative}", flush=True)
            temporary = destination.with_suffix(destination.suffix + ".part")
            try:
                with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
        manifest[relative] = {"url": url, "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}
    (directory / "sources.json").write_text(json.dumps(manifest, indent=2) + "\n")


def make_env_cfg(
    directory: Path, deployment: dict, output: Path, steps: int, evaluate_gait: bool = False, backend: str = "physx"
):
    """Build the robot with the published policy's joint order, pose, and gains."""
    from isaaclab_physx.physics import PhysxCfg
    from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg
    from isaaclab_visualizers.kit import KitVisualizerCfg

    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg
    from isaaclab.envs import ManagerBasedEnvCfg, mdp
    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
    from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.utils import configclass

    policy_names = [SDK_JOINT_NAMES[i] for i in deployment["joint_ids_map"]]
    initial_pose = dict(zip(policy_names, deployment["default_joint_pos"], strict=True))
    # The deployment gains are in SDK motor order; the poses/actions are in policy order.
    stiffness = dict(zip(SDK_JOINT_NAMES, deployment["stiffness"], strict=True))
    damping = dict(zip(SDK_JOINT_NAMES, deployment["damping"], strict=True))
    effort, velocity = {}, {}
    for name in SDK_JOINT_NAMES:
        if "hip_roll" in name or "knee" in name:
            effort[name], velocity[name] = 139.0, 20.0
        elif "hip_pitch" in name or "hip_yaw" in name or name == "waist_yaw_joint":
            effort[name], velocity[name] = 88.0, 32.0
        elif "wrist_pitch" in name or "wrist_yaw" in name:
            effort[name], velocity[name] = 5.0, 22.0
        else:
            effort[name], velocity[name] = 25.0, 37.0

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(
            prim_path="/World/Ground",
            spawn=sim_utils.GroundPlaneCfg(
                color=(0.35, 0.35, 0.35),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0, dynamic_friction=1.0, restitution=0.0
                ),
            ),
        )
        light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))
        robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(directory / "assets/g1_29dof_rev_1_0.usd"),
                activate_contact_sensors=evaluate_gait,
                rigid_props=PhysxRigidBodyCfg(
                    disable_gravity=False,
                    retain_accelerations=False,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=1000.0,
                    max_depenetration_velocity=1.0,
                ),
                articulation_props=PhysxArticulationCfg(
                    enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.8), joint_pos=initial_pose, joint_vel={".*": 0.0}
            ),
            soft_joint_pos_limit_factor=0.9,
            actuators={
                "motors": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness=stiffness,
                    damping=damping,
                    joint_effort_limit=effort,
                    joint_velocity_limit=velocity,
                    armature=0.01,
                )
            },
        )

    @configclass
    class StateObservations(ObservationGroupCfg):
        angular_velocity = ObservationTermCfg(func=mdp.base_ang_vel)

    @configclass
    class ActionsCfg:
        joints = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=policy_names,
            preserve_order=True,
            scale=dict(zip(policy_names, deployment["actions"]["JointPositionAction"]["scale"], strict=True)),
            offset=dict(zip(policy_names, deployment["actions"]["JointPositionAction"]["offset"], strict=True)),
            use_default_offset=False,
        )

    scene = SceneCfg(num_envs=1, env_spacing=2.5)
    if evaluate_gait:
        from isaaclab.sensors import ContactSensorCfg

        scene.feet_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*_ankle_roll_link",
            update_period=deployment["step_dt"],
        )

    cfg = ManagerBasedEnvCfg(
        scene=scene,
        decimation=4,
        observations={"state": StateObservations()},
        actions=ActionsCfg(),
        video_recorders=[
            VideoRecorderCfg(
                source="viz:kit",
                output_dir=str(output),
                output_filename_prefix="g1_29dof_walk",
                video_length=steps,
                video_interval=0,
                step_offset=0,
                fps=50,
            )
        ],
        seed=42,
    )
    cfg.sim.physics = PhysxCfg()
    cfg.sim.dt = deployment["step_dt"] / cfg.decimation
    cfg.sim.render_interval = cfg.decimation
    cfg.sim.default_visualizer_cfg = KitVisualizerCfg(
        origin_type="asset",
        origin_track_path="robot",
        eye=(3.0, -3.0, 1.8),
        lookat=(0.0, 0.0, 0.1),
        window_width=1280,
        window_height=720,
    )
    if backend == "mjwarp":
        from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

        from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg

        from isaaclab_tasks.core.velocity.velocity_env_cfg import RoughPhysicsCfg

        cfg.sim.physics = RoughPhysicsCfg().newton_mjwarp
        cfg.sim.physics.solver_cfg.njmax = 300
        cfg.sim.physics.solver_cfg.nconmax = 100
        cfg.scene.ground.spawn.physics_material = RigidBodyMaterialBaseCfg(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0
        )
        cfg.sim.default_visualizer_cfg = NewtonGLVisualizerCfg(
            headless=True,
            eye=(3.0, -3.0, 1.8),
            lookat=(0.0, 0.0, 0.65),
            window_width=1280,
            window_height=720,
            streaming_view=False,
        )
        cfg.video_recorders[0].source = "viz:newton_gl"
    return cfg, policy_names


def run(args, deployment: dict) -> None:
    """Execute the speed sequence, recording measured velocity and failures."""
    import numpy as np
    import onnxruntime as ort
    import torch

    from isaaclab.app import launch_simulation

    steps = round(args.duration / deployment["step_dt"])
    cfg, policy_names = make_env_cfg(args.assets, deployment, args.output, steps, args.evaluate_gait, args.backend)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    policy = ort.InferenceSession(str(args.assets / "policy.onnx"), options, providers=["CPUExecutionProvider"])
    input_name = policy.get_inputs()[0].name
    if policy.get_inputs()[0].shape[-1] != 480 or policy.get_outputs()[0].shape[-1] != 29:
        raise ValueError("Expected the official 480-input / 29-output velocity policy.")
    args.output.mkdir(parents=True, exist_ok=True)
    targets = (0.2, 0.5, 1.0)
    # Fixed feedforward calibration for this model/policy and the three requested speeds.
    # Keep target and actual policy input distinct in the recording's measurements.
    policy_inputs = targets if args.raw_commands else (0.27, 0.53, 1.05)
    samples = [[] for _ in targets]
    fall = None
    completed_steps = 0
    with launch_simulation(cfg, args):
        from isaaclab.envs import ManagerBasedEnv

        env = ManagerBasedEnv(cfg=cfg)
        try:
            env.reset()
            robot = env.scene["robot"]
            if set(robot.joint_names) != set(policy_names):
                raise ValueError(f"Robot/policy joints differ: {robot.joint_names}")
            joint_ids = [robot.joint_names.index(name) for name in policy_names]
            if args.evaluate_gait:
                feet = ["left_ankle_roll_link", "right_ankle_roll_link"]
                foot_ids = [robot.body_names.index(name) for name in feet]
                contacts = env.scene["feet_contact"]
                contact_ids = [contacts.body_names.index(name) for name in feet]
            default = np.asarray(deployment["default_joint_pos"], dtype=np.float32)[None, :]
            last_action = np.zeros((1, 29), dtype=np.float32)
            histories = {}
            with (args.output / "velocity_tracking.csv").open("w", newline="") as stream, torch.inference_mode():
                writer = csv.writer(stream)
                writer.writerow(
                    [
                        "time_s",
                        "segment",
                        "target_vx_m_s",
                        "policy_input_vx_m_s",
                        "actual_vx_m_s",
                        "height_m",
                        "upright",
                    ]
                    + (
                        [f"{side}_foot_{axis}_m" for side in ("left", "right") for axis in ("x", "y", "z")]
                        + [
                            "left_foot_normal_force_n",
                            "right_foot_normal_force_n",
                            "root_x_m",
                            "root_y_m",
                            "heading_rad",
                            "yaw_rate_rad_s",
                            "actual_vy_body_m_s",
                            "actual_vx_world_m_s",
                            "actual_vy_world_m_s",
                        ]
                        if args.evaluate_gait
                        else []
                    )
                )
                for step in range(steps):
                    segment = min(step * 3 // steps, 2)
                    command = np.array([[policy_inputs[segment], 0.0, 0.0]], dtype=np.float32)
                    values = {
                        "base_ang_vel": robot.data.root_ang_vel_b.torch.cpu().numpy(),
                        "projected_gravity": robot.data.projected_gravity_b.torch.cpu().numpy(),
                        "velocity_commands": command,
                        "joint_pos_rel": robot.data.joint_pos.torch[:, joint_ids].cpu().numpy() - default,
                        "joint_vel_rel": robot.data.joint_vel.torch[:, joint_ids].cpu().numpy(),
                        "last_action": last_action,
                    }
                    for name, spec in deployment["observations"].items():
                        value = values[name] * np.asarray(spec["scale"], dtype=np.float32)
                        if name not in histories:
                            histories[name] = np.repeat(value[:, None, :], spec["history_length"], axis=1)
                        else:
                            histories[name] = np.concatenate((histories[name][:, 1:], value[:, None, :]), axis=1)
                    observation = np.concatenate(
                        [histories[name].reshape(1, -1) for name in deployment["observations"]], axis=1
                    )
                    # run the policy with the created observation
                    last_action = policy.run(None, {input_name: observation.astype(np.float32)})[0]
                    if not np.isfinite(last_action).all():
                        raise RuntimeError("Policy returned non-finite joint targets.")

                    if args.backend == "mjwarp":
                        pos = robot.data.root_link_pos_w.torch[0].cpu().tolist()
                        env.sim.set_camera_view(eye=(pos[0] + 3, pos[1] - 3, 1.8), target=(pos[0], pos[1], 0.65))
                    env.step(torch.from_numpy(last_action).to(env.device))
                    completed_steps = step + 1
                    # IsaacLab uses xyzw quaternions; heading is the world yaw of the base.
                    x, y, z, w = robot.data.root_link_quat_w.torch[0].cpu().tolist()
                    heading = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
                    actual = robot.data.root_lin_vel_b.torch[0, 0].item()
                    height = robot.data.root_link_pos_w.torch[0, 2].item()
                    upright = -robot.data.projected_gravity_b.torch[0, 2].item()
                    writer.writerow(
                        [
                            (step + 1) * env.step_dt,
                            segment + 1,
                            targets[segment],
                            policy_inputs[segment],
                            actual,
                            height,
                            upright,
                        ]
                        + (
                            robot.data.body_link_pos_w.torch[0, foot_ids].cpu().numpy().reshape(-1).tolist()
                            + contacts.data.net_normal_forces_w.torch[0, contact_ids, 2].cpu().numpy().tolist()
                            + robot.data.root_link_pos_w.torch[0, :2].cpu().numpy().tolist()
                            + [
                                heading,
                                robot.data.root_ang_vel_w.torch[0, 2].item(),
                                robot.data.root_lin_vel_b.torch[0, 1].item(),
                                robot.data.root_lin_vel_w.torch[0, 0].item(),
                                robot.data.root_lin_vel_w.torch[0, 1].item(),
                            ]
                            if args.evaluate_gait
                            else []
                        )
                    )
                    samples[segment].append(actual)
                    if height < 0.4 or upright < 0.5:
                        fall = {"time_s": completed_steps * env.step_dt, "height_m": height, "upright": upright}
                        print(f"Stopping after fall: {fall}", flush=True)
                        break
                    if step % 100 == 0:
                        print(
                            f"t={step * env.step_dt:.2f}s target={targets[segment]:.1f} "
                            f"policy_input={policy_inputs[segment]:.2f} actual={actual:.3f} m/s",
                            flush=True,
                        )
        finally:
            env.close()
        metadata = {
            "policy_sha256": hashlib.sha256((args.assets / "policy.onnx").read_bytes()).hexdigest(),
            "physics_backend": "newton_mjwarp" if args.backend == "mjwarp" else "isaacsim_physx",
            "duration_s": args.duration,
            "completed_duration_s": completed_steps * deployment["step_dt"],
            "segment_duration_s": args.duration / 3,
            "settle_time_s": args.settle_time,
            "seed": 42,
            "control_dt_s": deployment["step_dt"],
            "targets_m_s": targets,
            "policy_inputs_m_s": policy_inputs,
            "contact_on_n": 10,
            "contact_off_n": 5,
            "contact_debounce_s": 0.06,
            "fall": fall,
            "asset": str(args.assets / "assets/g1_29dof_rev_1_0.usd"),
            "training": False,
        }
        (args.output / "evaluation_run.json").write_text(json.dumps(metadata, indent=2) + "\n")
        summary = []
        for speed, policy_input, velocities in zip(targets, policy_inputs, samples):
            if not velocities:
                continue
            # Exclude the first second after each abrupt command change for steady-state metrics.
            settled = np.asarray(velocities[min(50, len(velocities) // 2) :])
            summary.append(
                {
                    "target_m_s": speed,
                    "policy_input_m_s": policy_input,
                    "mean_actual_m_s": float(settled.mean()),
                    "mean_absolute_error_m_s": float(np.abs(settled - speed).mean()),
                }
            )
        print(json.dumps(summary, indent=2), flush=True)
        print(f"Video and velocity measurements: {args.output}", flush=True)
        if args.evaluate_gait:
            from evaluate_g1_29dof_gait import summarize_gait

            summarize_gait(args.output / "velocity_tracking.csv", args.settle_time)


def main() -> None:
    """Download assets or launch a complete twenty-second playback."""
    from isaaclab.app import add_launcher_args

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=REPO_ROOT / "checkpoints/g1-29dof")
    parser.add_argument("--output", type=Path, default=REPO_ROOT.parents[1] / "unitree_rl_lab/videos/g1-29dof")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--backend", choices=("physx", "mjwarp"), default="physx")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--evaluate-gait", action="store_true", help="Record foot contacts and summarize gait metrics.")
    parser.add_argument(
        "--raw-commands", action="store_true", help="Send target speeds without feedforward calibration."
    )
    parser.add_argument(
        "--settle-time", type=float, default=2.0, help="Seconds excluded after each speed change for gait."
    )
    add_launcher_args(parser)
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration < 3.0:
        parser.error("--duration must be finite and at least 3 seconds.")
    if args.evaluate_gait and (not math.isfinite(args.settle_time) or not 0 <= args.settle_time < args.duration / 3):
        parser.error("--settle-time must be finite, nonnegative, and shorter than each speed segment.")
    args.assets, args.output = args.assets.resolve(), args.output.resolve()
    if not args.download_only and args.output.exists():
        parser.error("Choose a new output directory to preserve existing results.")
    download_assets(args.assets)
    if not args.download_only:
        import yaml

        deployment = yaml.safe_load((args.assets / "deploy.yaml").read_text())
        run(args, deployment)


if __name__ == "__main__":
    main()
