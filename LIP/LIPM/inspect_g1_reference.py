"""Render G1/LIPM reference geometry and a nominal-PD startup diagnostic over SSH.

Run with the IsaacLab uv environment. This is not trained-policy playback.
"""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import warp as wp
from PIL import Image, ImageDraw, ImageFont

wp.config.enable_backward = False

from isaaclab import sim as sim_utils
from isaaclab.app import launch_simulation
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import quat_apply
from isaaclab_tasks.core.velocity.mdp.lipm_commands import robot_com
from isaaclab_tasks.utils import resolve_task_config
from isaaclab_visualizers.newton import NewtonGLVisualizerCfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--motion",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "generated_trajectories/step_020_g1_startup.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "previews/g1_lipm",
    )
    parser.add_argument("--duration", type=float, default=2.0)
    args = parser.parse_args()
    if not np.isfinite(args.duration) or args.duration <= 0:
        parser.error("duration must be positive and finite")
    args.output.mkdir(parents=True, exist_ok=True)
    cfg, _ = resolve_task_config(
        "Isaac-Velocity-Flat-G1", "rsl_rl_cfg_entry_point", overrides=["presets=lipm"]
    )
    cfg.commands.base_velocity.motion_directory = str(args.motion.resolve())
    cfg.scene.num_envs = 1
    cfg.seed = 42
    # Keep a fall visible instead of automatically resetting the diagnostic.
    cfg.terminations.fallen = None
    cfg.terminations.bad_orientation = None
    cfg.sim.default_visualizer_cfg = NewtonGLVisualizerCfg(
        headless=True,
        eye=(1.65, -2.0, 1.3),
        lookat=(0.25, 0, 0.65),
        window_width=960,
        window_height=720,
        streaming_view=False,
    )
    orange = (255, 145, 40)
    cyan = (45, 220, 240)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 18)
    small = ImageFont.truetype(font_path, 15)
    title_font = ImageFont.truetype(font_path, 23)
    records = []
    video_path = args.output / "g1_lipm_comparison.mp4"
    with launch_simulation(cfg, {"headless": True, "visualizer": "newton_gl"}):
        from isaaclab.envs import ManagerBasedRLEnv

        env = ManagerBasedRLEnv(cfg=cfg)
        try:
            env.reset()
            command = env.command_manager.get_term("base_velocity")
            duration = min(
                args.duration, float(command.library.durations[0]) - env.step_dt
            )
            if duration <= 0:
                raise ValueError(
                    "Reference is too short for the environment control timestep"
                )
            markers = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/World/Visuals/LipmInspection",
                    markers={
                        "desired": sim_utils.SphereCfg(
                            radius=0.025,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=tuple(c / 255 for c in orange)
                            ),
                        ),
                        "actual": sim_utils.SphereCfg(
                            radius=0.018,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=tuple(c / 255 for c in cyan)
                            ),
                        ),
                    },
                )
            )
            viz = next(
                v for v in env.sim.visualizers if v.cfg.visualizer_type == "newton_gl"
            )
            fps = round(1 / env.step_dt)
            with imageio.get_writer(
                video_path, fps=fps, codec="libx264", quality=8, macro_block_size=16
            ) as writer:
                for i in range(round(duration / env.step_dt) + 1):
                    if i:
                        observation, reward, _, _, _ = env.step(
                            torch.zeros((1, 29), device=env.device)
                        )
                        assert observation["policy"].shape == (1, 115)
                        assert (
                            torch.isfinite(observation["policy"]).all()
                            and torch.isfinite(reward).all()
                        )
                    t = i * env.step_dt
                    # Sample the CURRENT state for comparison. Training's observation target is one step ahead.
                    sample = command.library.sample(
                        command.clip_ids, torch.full((1,), t, device=env.device)
                    )
                    q = command.rotation
                    ref_com = (
                        quat_apply(q, sample["com_pos"] - command.anchor)
                        + command.origin
                    )
                    ref_feet = (
                        quat_apply(
                            q[:, None].expand(-1, 2, -1),
                            sample["foot_pos"] - command.anchor[:, None],
                        )
                        + command.origin[:, None]
                    )
                    ref_vel = quat_apply(q, sample["com_vel"])
                    com, vel = robot_com(command.robot)
                    feet = command.foot_positions()
                    markers.visualize(
                        translations=torch.cat(
                            (ref_com, ref_feet[0], com, feet[0]), dim=0
                        ),
                        marker_indices=[0, 0, 0, 1, 1, 1],
                    )
                    env.sim.forward()
                    frame = viz.render_rgb_array()
                    row = {
                        "time": t,
                        "planned_support": sample["support_mask"][0].cpu().tolist(),
                        "actual_com": com[0].cpu().tolist(),
                        "desired_com": ref_com[0].cpu().tolist(),
                        "actual_com_velocity": vel[0].cpu().tolist(),
                        "desired_com_velocity": ref_vel[0].cpu().tolist(),
                        "actual_feet": feet[0].cpu().tolist(),
                        "desired_feet": ref_feet[0].cpu().tolist(),
                    }
                    records.append(row)
                    canvas = Image.new("RGB", (1440, 720), (19, 25, 35))
                    canvas.paste(
                        Image.fromarray(frame[:, :, :3]).resize((960, 720)), (0, 0)
                    )
                    draw = ImageDraw.Draw(canvas)
                    draw.rectangle((0, 0, 960, 87), fill=(19, 25, 35))
                    draw.text(
                        (22, 12),
                        "G1 + LIPM reference inspection",
                        font=title_font,
                        fill="white",
                    )
                    draw.text(
                        (22, 49),
                        "Nominal-pose PD only | No learned tracking policy | Playback: 0.5x",
                        font=font,
                        fill=(215, 220, 230),
                    )
                    x = 983
                    draw.text(
                        (x, 16),
                        f"Simulation time: {t:.2f} s",
                        font=title_font,
                        fill="white",
                    )
                    draw.text(
                        (x, 60), "Orange: desired soles / CoM", font=font, fill=orange
                    )
                    draw.text(
                        (x, 88), "Cyan: measured soles / CoM", font=font, fill=cyan
                    )
                    dc, ac = row["desired_com"], row["actual_com"]
                    dv, av = (
                        np.linalg.norm(row["desired_com_velocity"][:2]),
                        np.linalg.norm(row["actual_com_velocity"][:2]),
                    )
                    lines = [
                        f"CoM height: {ac[2]:.3f} / {dc[2]:.3f} m",
                        f"CoM speed: {av:.3f} / {dv:.3f} m/s",
                        f"Step command: {sample['desired_step_length'][0].item():.2f} m",
                        f"Actual sole Z: {row['actual_feet'][0][2]:.3f} / {row['actual_feet'][1][2]:.3f} m",
                        "CoM values: actual / desired",
                    ]
                    for j, line in enumerate(lines):
                        draw.text(
                            (x, 140 + j * 29), line, font=small, fill=(225, 230, 235)
                        )
                    # Top view exposes reference/actual markers even when the robot mesh occludes them.
                    box = (995, 355, 1415, 655)
                    draw.rectangle(box, outline=(100, 110, 120), width=1)
                    draw.text(
                        (x, 316),
                        "Top view: X forward, Y left (meters)",
                        font=small,
                        fill="white",
                    )

                    def project(point):
                        return (
                            int(1025 + (point[0] + 0.25) / 1.75 * 350),
                            int(505 - point[1] / 0.65 * 130),
                        )

                    for points, color in (
                        ([dc, *row["desired_feet"]], orange),
                        ([ac, *row["actual_feet"]], cyan),
                    ):
                        for label, point in zip(("CoM", "L", "R"), points):
                            px, py = project(point)
                            px, py = max(1000, min(1400, px)), max(365, min(640, py))
                            draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=color)
                            draw.text((px + 7, py - 10), label, font=small, fill=color)
                    draw.text(
                        (x, 675),
                        "Diagnostic only; no feasibility claim.",
                        font=small,
                        fill=(185, 195, 210),
                    )
                    rgb = np.asarray(canvas)
                    if i == 0:
                        canvas.save(args.output / "startup.png")
                        for _ in range(fps * 2):
                            writer.append_data(rgb)
                    for _ in range(2):
                        writer.append_data(rgb)
            initial = records[0]
            summary = {
                "mode": "nominal_pose_PD_no_trained_policy",
                "motion": str(args.motion.resolve()),
                "sample_count": len(records),
                "simulation_duration": records[-1]["time"],
                "initial": initial,
                "initial_com_height_error_m": initial["actual_com"][2]
                - initial["desired_com"][2],
                "initial_horizontal_velocity_error_mps": (
                    np.array(initial["actual_com_velocity"])
                    - initial["desired_com_velocity"]
                )[:2].tolist(),
                "initial_actual_foot_separation_m": float(
                    np.linalg.norm(
                        np.array(initial["actual_feet"])[0, :2]
                        - np.array(initial["actual_feet"])[1, :2]
                    )
                ),
                "initial_desired_foot_separation_m": float(
                    np.linalg.norm(
                        np.array(initial["desired_feet"])[0, :2]
                        - np.array(initial["desired_feet"])[1, :2]
                    )
                ),
                "max_desired_foot_separation_m": max(
                    float(
                        np.linalg.norm(
                            np.array(r["desired_feet"])[0, :2]
                            - np.array(r["desired_feet"])[1, :2]
                        )
                    )
                    for r in records
                ),
            }
            (args.output / "comparison.json").write_text(
                json.dumps(records, indent=2) + "\n"
            )
            (args.output / "summary.json").write_text(
                json.dumps(summary, indent=2) + "\n"
            )
            print(json.dumps(summary, indent=2))
            print(f"Saved: {video_path}")
        finally:
            env.close()


if __name__ == "__main__":
    main()
