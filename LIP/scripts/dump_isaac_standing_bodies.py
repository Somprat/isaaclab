"""Diagnostic: dump per-body kinematics/inertials of the IsaacLab G1 (lipm preset) in its reset configuration.

The reset configuration is written explicitly from LIP/initial_states/g1_reset.json (root pose + all joints),
then read back without stepping physics. Output is consumed by compare_isaac_urdf_standing.py.
"""

import argparse
import json
from pathlib import Path

import torch
import warp as wp

wp.config.enable_backward = False

from isaaclab.app import launch_simulation
from isaaclab_tasks.core.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg
from isaaclab_tasks.core.velocity.mdp.lipm_commands import robot_com
from isaaclab_tasks.utils import resolve_task_config

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=ROOT / "initial_states/g1_reset.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    state = json.loads(args.state.read_text())

    cfg, _ = resolve_task_config("Isaac-Velocity-Flat-G1", "rsl_rl_cfg_entry_point", overrides=["presets=lipm"])
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
            origin = env.scene.env_origins[0]
            dev = env.device
            # Exact configuration: root pose and named joints from the state file, zero velocity.
            pose = torch.tensor([*state["root_pos"], *state["root_quat_xyzw"]], device=dev)[None]
            pose[:, :3] += origin
            by_name = dict(zip(state["joint_names"], state["joint_pos"]))
            q = torch.tensor([[by_name[n] for n in robot.joint_names]], device=dev)
            robot.write_root_pose_to_sim(pose)
            robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=dev))
            robot.write_joint_state_to_sim(q, torch.zeros_like(q))
            env.scene.update(0.0)
            measured_q = robot.data.joint_pos.torch[0]
            max_joint_error = (measured_q - q[0]).abs().max().item()
            com, _ = robot_com(robot)
            out = {
                "source": "IsaacLab Isaac-Velocity-Flat-G1 lipm preset; explicit state write, no physics step",
                "asset": str(cfg.scene.robot.spawn.usd_path),
                "state_file": str(args.state.resolve()),
                "max_joint_write_error": max_joint_error,
                "root_pos": (robot.data.root_pos_w.torch[0] - origin).cpu().tolist(),
                "root_quat_xyzw": robot.data.root_quat_w.torch[0].cpu().tolist(),
                "joint_names": list(robot.joint_names),
                "joint_pos": measured_q.cpu().tolist(),
                "body_names": list(robot.body_names),
                "body_pos": (robot.data.body_pos_w.torch[0] - origin).cpu().tolist(),
                "body_quat_xyzw": robot.data.body_quat_w.torch[0].cpu().tolist(),
                "body_com_pos": (robot.data.body_com_pos_w.torch[0] - origin).cpu().tolist(),
                "body_mass": robot.data.body_mass.torch[0].cpu().tolist(),
                "whole_body_com": (com[0] - origin).cpu().tolist(),
                "foot_sole_offset_in_ankle_roll": [0.0, 0.0, -0.035],
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(out, indent=2) + "\n")
            print(f"max joint write error {max_joint_error:.3e}; whole CoM {out['whole_body_com']}; saved {args.output}")
        finally:
            env.close()


if __name__ == "__main__":
    main()
