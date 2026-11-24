"""Measure the LIPM G1 preset's actual reset state for reference generation.

With --height-calibration, the robot is instead placed at the root pose and joint
positions of a wb_humanoid_mpc torso-origin height calibration (schema 2) before
measuring. The posture also becomes the nominal-PD default for the contact check.
"""

import argparse
import json
import math
from pathlib import Path

import torch
import warp as wp

wp.config.enable_backward = False

from isaaclab.app import launch_simulation
from isaaclab.utils.math import quat_apply
from isaaclab_tasks.core.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg
from isaaclab_tasks.core.velocity.mdp.lipm_commands import robot_com
from isaaclab_tasks.utils import resolve_task_config

# MPC contact frame foot_*_contact, expressed in *_ankle_roll_link (g1_29dof URDF).
MPC_CONTACT_OFFSET = (0.035, 0.0, -0.035)


def quat_xyzw_from_zyx(yaw, pitch, roll):
    """OCS2 base pose Euler angles (R = Rz(yaw) Ry(pitch) Rx(roll)) to an xyzw quaternion."""
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def load_calibration(path):
    calibration = json.loads(path.read_text())
    if (
        calibration.get("kind") != "g1_torso_origin_height_calibration"
        or calibration.get("schema_version") != 2
    ):
        raise ValueError(f"{path} is not a schema 2 g1_torso_origin_height_calibration file")
    if not calibration.get("success"):
        raise ValueError(f"{path} did not converge (status: {calibration.get('status')})")
    return calibration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination JSON; defaults to initial_states/g1_reset.json without --height-calibration",
    )
    parser.add_argument(
        "--height-calibration",
        type=Path,
        default=None,
        help="wb_humanoid_mpc torso-origin height calibration JSON; place the robot at its root pose and joints",
    )
    parser.add_argument(
        "--keep-contact-frame-height",
        action="store_true",
        help="With --height-calibration, keep the MPC root z instead of shifting it so the mean MPC contact frame height is 0",
    )
    parser.add_argument(
        "--contact-check-steps",
        type=int,
        default=5,
        help="Nominal-PD steps used to verify loading after capturing the reset",
    )
    args = parser.parse_args()
    if args.contact_check_steps < 1:
        parser.error("contact-check-steps must be positive")
    if args.output is None:
        if args.height_calibration is not None:
            parser.error("--output is required with --height-calibration")
        args.output = Path(__file__).resolve().parents[1] / "initial_states/g1_reset.json"
    cfg, _ = resolve_task_config(
        "Isaac-Velocity-Flat-G1", "rsl_rl_cfg_entry_point", overrides=["presets=lipm"]
    )
    # Measure the reset without needing a reference NPZ to exist first.
    foot_cfg = cfg.commands.base_velocity
    nominal = G1FlatEnvCfg()
    cfg.commands = nominal.commands
    cfg.observations = nominal.observations
    cfg.rewards = {}
    cfg.terminations = {}
    cfg.scene.num_envs = 1
    cfg.seed = 42
    calibration = None
    if args.height_calibration is not None:
        calibration = load_calibration(args.height_calibration)
        x, y, z, yaw, pitch, roll = calibration["base_pose_xyz_yaw_pitch_roll"]
        # LIP heights are measured from the MPC contact frames; put them on z=0.
        z_shift = 0.0 if args.keep_contact_frame_height else -float(calibration["mean_contact_frame_height"])
        target_root_pos = (x, y, z + z_shift)
        target_root_quat = quat_xyzw_from_zyx(yaw, pitch, roll)
        target_joints = dict(zip(calibration["joint_names"], calibration["joint_pos"], strict=True))
        # The reset events restore init_state exactly (zero ranges, joint scale 1), and
        # the nominal action offsets and PD targets then hold this posture.
        cfg.scene.robot.init_state.pos = target_root_pos
        cfg.scene.robot.init_state.rot = target_root_quat
        cfg.scene.robot.init_state.joint_pos = target_joints
        cfg.scene.robot.init_state.joint_vel = {".*": 0.0}
    with launch_simulation(cfg, {"headless": True}):
        from isaaclab.envs import ManagerBasedRLEnv

        env = ManagerBasedRLEnv(cfg=cfg)
        try:
            env.reset()
            robot = env.scene["robot"]
            foot_ids = [
                robot.body_names.index(name) for name in foot_cfg.foot_body_names
            ]
            offsets = torch.tensor(foot_cfg.foot_offsets, device=env.device)
            origin = env.scene.env_origins[0]
            com, velocity = robot_com(robot)
            torso_id = robot.body_names.index("torso_link")
            torso_com = robot.data.body_com_pos_w.torch[0, torso_id] - origin
            soles = (
                robot.data.body_pos_w.torch[0, foot_ids]
                + quat_apply(robot.data.body_quat_w.torch[0, foot_ids], offsets)
                - origin
            )
            # These four 5 mm-radius spheres are the matched URDF's collision geometry.
            centers = torch.tensor(
                [
                    [-0.05, 0.025, -0.03],
                    [-0.05, -0.025, -0.03],
                    [0.12, 0.03, -0.03],
                    [0.12, -0.03, -0.03],
                ],
                device=env.device,
            )
            q = robot.data.body_quat_w.torch[0, foot_ids, None].expand(-1, 4, -1)
            points = quat_apply(q, centers[None].expand(2, -1, -1))
            points += robot.data.body_pos_w.torch[0, foot_ids, None] - origin
            minimum_collision_height = points[..., 2].min(dim=1).values - 0.005
            placement = None
            if calibration is not None:
                q_target = torch.tensor(
                    [target_joints[name] for name in robot.joint_names], device=env.device
                )
                q_error = (robot.data.joint_pos.torch[0] - q_target).abs().max().item()
                root_error = (
                    robot.data.root_pos_w.torch[0] - origin
                    - torch.tensor(target_root_pos, device=env.device)
                ).abs().max().item()
                if q_error > 1e-5 or root_error > 1e-6:
                    raise RuntimeError(
                        f"Placement mismatch: joint {q_error:.3e} rad, root {root_error:.3e} m"
                    )
                foot_q = robot.data.body_quat_w.torch[0, foot_ids]
                mpc_contact = (
                    robot.data.body_pos_w.torch[0, foot_ids]
                    + quat_apply(
                        foot_q,
                        torch.tensor(MPC_CONTACT_OFFSET, device=env.device).expand(2, -1),
                    )
                    - origin
                )
                measured_torso = robot.data.body_pos_w.torch[0, torso_id] - origin
                placement = {
                    "source": str(args.height_calibration.resolve()),
                    "requested_torso_origin_height": calibration["requested_torso_origin_height"],
                    "calibration_torso_origin_height": calibration["measured_torso_origin_height"],
                    "calibration_whole_body_com_height": calibration["whole_body_com_height"],
                    "calibration_mean_contact_frame_height": calibration["mean_contact_frame_height"],
                    "calibration_lip_com_height": calibration["whole_body_com_height"]
                    - calibration["mean_contact_frame_height"],
                    "calibration_contact_frame_pos": calibration["contact_frame_pos"],
                    "root_z_shift": z_shift,
                    "base_pose_xyz_yaw_pitch_roll": calibration["base_pose_xyz_yaw_pitch_roll"],
                    "max_joint_placement_error": q_error,
                    "max_root_placement_error": root_error,
                    "mpc_contact_offset_in_ankle_roll": list(MPC_CONTACT_OFFSET),
                    "isaac_mpc_contact_frame_pos": mpc_contact.cpu().tolist(),
                    "torso_origin_height_minus_contact_height": (
                        measured_torso[2] - mpc_contact[:, 2].mean()
                    ).item(),
                    "com_height_minus_contact_height": (
                        com[0, 2] - origin[2] - mpc_contact[:, 2].mean()
                    ).item(),
                    "foot_tilt_rad": (
                        2 * torch.asin(foot_q[:, :2].norm(dim=-1).clamp(max=1.0))
                    ).cpu().tolist(),
                }
            state = {
                "schema_version": 1,
                "frame": "world_z_up",
                "state_kind": (
                    "height_calibration_placement_before_physics"
                    if calibration is not None
                    else "reset_before_physics"
                ),
                "task": "Isaac-Velocity-Flat-G1",
                "preset": "lipm",
                "asset": str(cfg.scene.robot.spawn.usd_path),
                "root_pos": (robot.data.root_pos_w.torch[0] - origin).cpu().tolist(),
                "root_quat_xyzw": robot.data.root_quat_w.torch[0].cpu().tolist(),
                "com_pos": (com[0] - origin).cpu().tolist(),
                "com_vel": velocity[0].cpu().tolist(),
                "torso_body_name": "torso_link",
                "torso_com_pos": torso_com.cpu().tolist(),
                "torso_link_pos": (robot.data.body_pos_w.torch[0, torso_id] - origin).cpu().tolist(),
                "foot_names": ["left", "right"],
                "foot_pos": soles.cpu().tolist(),
                "foot_offsets": offsets.cpu().tolist(),
                "foot_contact_points": (
                    points - torch.tensor([0.0, 0.0, 0.005], device=env.device)
                )
                .cpu()
                .tolist(),
                "minimum_foot_collision_height": minimum_collision_height.cpu().tolist(),
                "joint_names": robot.joint_names,
                "joint_pos": robot.data.joint_pos.torch[0].cpu().tolist(),
                "joint_vel": robot.data.joint_vel.torch[0].cpu().tolist(),
                "notes": "World axes relative to environment origin. State is captured before physics; contact-check samples are subsequent nominal-PD measurements, not a replacement for the reset state.",
            }
            if placement is not None:
                state["height_calibration"] = placement
                state["notes"] += (
                    " Root pose and joints are placed from the height calibration; foot_pos uses the"
                    " Isaac sole offset, while the MPC contact frame is 3.5 cm further forward"
                    " (height_calibration.isaac_mpc_contact_frame_pos)."
                )
            assert torch.isfinite(com).all() and torch.isfinite(soles).all()
            if (minimum_collision_height < -1e-6).any():
                raise ValueError(
                    f"Reset feet penetrate the plane: {minimum_collision_height.tolist()}"
                )
            sensor = env.scene["contact_forces"]
            sensor_ids = [
                sensor.body_names.index(name) for name in foot_cfg.foot_body_names
            ]
            contact_samples = []
            for i in range(args.contact_check_steps):
                env.step(
                    torch.zeros(
                        (1, env.action_manager.total_action_dim), device=env.device
                    )
                )
                forces = sensor.data.net_forces_w.torch[0, sensor_ids]
                contact_samples.append(
                    {
                        "time": (i + 1) * env.step_dt,
                        "foot_force_world": forces.cpu().tolist(),
                        "loaded_over_10N": (forces.norm(dim=-1) > 10).cpu().tolist(),
                    }
                )
            state["contact_check"] = {
                "samples": contact_samples,
                "both_feet_loaded": any(
                    all(s["loaded_over_10N"]) for s in contact_samples
                ),
                "total_mass_kg": robot.data.body_mass.torch[0].sum().item(),
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(state, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        k: state[k]
                        for k in (
                            "root_pos",
                            "com_pos",
                            "com_vel",
                            "torso_com_pos",
                            "torso_link_pos",
                            "foot_pos",
                            "minimum_foot_collision_height",
                            "contact_check",
                        )
                    }
                    | ({"height_calibration": placement} if placement else {}),
                    indent=2,
                )
            )
            print(f"Saved measured state: {args.output.resolve()}")
        finally:
            env.close()


if __name__ == "__main__":
    main()
