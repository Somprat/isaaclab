"""Direct shoulder-pitch target experiment: PD motor physics, no IK or policy."""

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from replay_hand_horizontal_hold_torque import ROOT, TorqueReplay

TIMES = np.arange(8, dtype=float)
OFFSETS = np.array([0.0, 0.0, 0.04, 0.13, 0.27, 0.36, 0.40, 0.40])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=ROOT / "LIP/previews/perfect-arm-replay-source"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--physics-dt", type=float, default=0.001)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument(
        "--final-offset",
        type=float,
        help="Override the original schedule: smooth reach from 1 to 7 seconds, then hold this offset in radians.",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh output directory")
    if not np.isfinite(args.duration) or args.duration <= 0:
        parser.error("Duration must be positive and finite")
    if args.final_offset is not None and not np.isfinite(args.final_offset):
        parser.error("Final offset must be finite")

    def command_offset(time):
        if args.final_offset is None:
            return float(np.interp(time, TIMES, OFFSETS))
        phase = np.clip((time - 1.0) / 6.0, 0.0, 1.0)
        blend = phase**3 * (10.0 + phase * (-15.0 + 6.0 * phase))
        return float(args.final_offset * blend)

    replay = TorqueReplay(args.source, args.physics_dt)
    n = round(args.duration / replay.dt)
    if n < 1 or n > len(replay.frames):
        parser.error("Duration must fit the source recording")
    args.output.mkdir(parents=True)
    initial = replay.data.qpos[replay.arm_q].copy()
    names = [replay.names[s] for s in replay.arm_slots]
    shoulder = [names.index(s + "_shoulder_pitch_joint") for s in ("left", "right")]
    rows = []
    renderer = (
        None if args.no_video else mujoco.Renderer(replay.model, height=720, width=960)
    )
    writer = (
        None
        if args.no_video
        else imageio.get_writer(
            args.output / "shoulder_actuator.mp4",
            fps=round(1 / replay.dt),
            codec="libx264",
            quality=8,
        )
    )
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation, camera.distance = -55, -15, 2.5
    try:
        for i in range(n):
            t = i * replay.dt
            desired = initial.copy()
            offset = command_offset(t)
            desired[shoulder] += offset
            replay.data.ctrl[replay.arm_act] = desired
            replay.drive_body(t)
            mujoco.mj_forward(replay.model, replay.data)
            root, lower, angle = replay.body_errors(t)
            row = {
                "time_s": t,
                "command_offset_rad": offset,
                "root_tracking_error_m": root,
                "lower_joint_max_error_rad": lower,
                "root_orientation_error_rad": angle,
            }
            for j, name in enumerate(names):
                actual = replay.data.qpos[replay.arm_q[j]]
                vel = replay.data.qvel[replay.arm_v[j]]
                requested = replay.kp * (desired[j] - actual) - replay.kd * vel
                applied = replay.data.actuator_force[replay.arm_act[j]]
                np.testing.assert_allclose(
                    applied,
                    np.clip(requested, -replay.limit[j], replay.limit[j]),
                    atol=1e-9,
                )
                for suffix, value in [
                    ("desired_rad", desired[j]),
                    ("actual_rad", actual),
                    ("actual_offset_rad", actual - initial[j]),
                    ("error_rad", actual - desired[j]),
                    ("velocity_rad_s", vel),
                    ("requested_nm", requested),
                    ("applied_nm", applied),
                ]:
                    row[name + "_" + suffix] = float(value)
            rows.append(row)
            if renderer is not None:
                camera.lookat[:] = [*replay.data.qpos[:2], 0.82]
                renderer.update_scene(replay.data, camera)
                image = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, 960, 65), fill=(15, 20, 25))
                draw.text(
                    (12, 8),
                    f"DIRECT SHOULDER PITCH | NO IK / NO POLICY | {t:.2f} s",
                    fill="white",
                )
                draw.text(
                    (12, 27),
                    f"Command: initial + {offset:.3f} rad | Kp=40, Kd=10, torque limit=25 Nm",
                    fill="white",
                )
                draw.text(
                    (12, 46),
                    "Actual offsets L/R: "
                    + " / ".join(
                        f"{row[names[j] + '_actual_offset_rad']:.3f}" for j in shoulder
                    )
                    + " rad",
                    fill="white",
                )
                writer.append_data(np.asarray(image))
            for k in range(replay.substeps):
                replay.advance(t + k * args.physics_dt)
            if i % 250 == 0:
                print(f"Frame {i}/{n}", flush=True)
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
    with (args.output / "tracking.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {
        "source": str(args.source.resolve()),
        "policy": False,
        "ik": False,
        "physics_dt": args.physics_dt,
        "control_dt": replay.dt,
        "duration_s": n * replay.dt,
        "schedule": {float(t): command_offset(t) for t in TIMES},
        "interpolation": (
            "linear, sampled at control ticks; held at +0.40 rad after 6 s"
            if args.final_offset is None
            else "quintic smooth reach from 1 to 7 s, then held; sampled at control ticks"
        ),
        "other_arm_joints": "PD hold at initial measured angles",
        "kp": replay.kp,
        "kd": replay.kd,
        "body": "Recorded trajectory followed by external compliant constraints; contacts disabled",
        "shoulders": {},
    }
    for side, j in zip(("left", "right"), shoulder):
        name = names[j]
        after = [r for r in rows if r["time_s"] >= 7]
        summary["shoulders"][side] = {
            "initial_rad": float(initial[j]),
            "final_command_rad": rows[-1][name + "_desired_rad"],
            "final_actual_rad": rows[-1][name + "_actual_rad"],
            "final_actual_offset_rad": rows[-1][name + "_actual_offset_rad"],
            "post7_mean_abs_error_rad": float(
                np.mean([abs(r[name + "_error_rad"]) for r in after])
            )
            if after
            else None,
            "peak_abs_applied_nm": max(abs(r[name + "_applied_nm"]) for r in rows),
            "fraction_at_torque_limit": float(
                np.mean(
                    [
                        abs(r[name + "_applied_nm"]) >= 0.99 * replay.limit[j]
                        for r in rows
                    ]
                )
            ),
        }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    t = [r["time_s"] for r in rows]
    for col, (side, j) in enumerate(zip(("left", "right"), shoulder)):
        name = names[j]
        axs[0, col].plot(
            t, [r["command_offset_rad"] for r in rows], label="Command offset"
        )
        axs[0, col].plot(
            t, [r[name + "_actual_offset_rad"] for r in rows], label="Actual offset"
        )
        axs[0, col].set_title(side.capitalize() + " shoulder pitch")
        axs[0, col].set_ylabel("Offset from initial (rad)")
        axs[0, col].legend()
        axs[1, col].plot(t, [r[name + "_error_rad"] for r in rows])
        axs[1, col].set_ylabel("Actual − desired (rad)")
        axs[2, col].plot(t, [r[name + "_applied_nm"] for r in rows])
        axs[2, col].set_ylabel("Applied torque (Nm)")
        axs[2, col].set_xlabel("Time (s)")
    for ax in axs.flat:
        ax.grid(alpha=0.2)
        ax.axvline(7, color="black", ls="--", lw=0.7)
    fig.tight_layout()
    fig.savefig(args.output / "tracking.png", dpi=150)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
