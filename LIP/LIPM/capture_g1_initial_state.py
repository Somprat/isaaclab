"""Measure the LIPM G1 preset's actual reset state for reference generation."""

import argparse
import json
from pathlib import Path

import torch
import warp as wp

wp.config.enable_backward = False

from isaaclab.app import launch_simulation
from isaaclab.utils.math import quat_apply
from isaaclab_tasks.core.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg
from isaaclab_tasks.core.velocity.mdp.lipm_commands import robot_com
from isaaclab_tasks.utils import resolve_task_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "initial_states/g1_reset.json",
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
            state = {
                "schema_version": 1,
                "frame": "world_z_up",
                "state_kind": "reset_before_physics",
                "task": "Isaac-Velocity-Flat-G1",
                "preset": "lipm",
                "asset": str(cfg.scene.robot.spawn.usd_path),
                "root_pos": (robot.data.root_pos_w.torch[0] - origin).cpu().tolist(),
                "root_quat_xyzw": robot.data.root_quat_w.torch[0].cpu().tolist(),
                "com_pos": (com[0] - origin).cpu().tolist(),
                "com_vel": velocity[0].cpu().tolist(),
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
                            "foot_pos",
                            "minimum_foot_collision_height",
                            "contact_check",
                        )
                    },
                    indent=2,
                )
            )
            print(f"Saved measured state: {args.output.resolve()}")
        finally:
            env.close()


if __name__ == "__main__":
    main()
