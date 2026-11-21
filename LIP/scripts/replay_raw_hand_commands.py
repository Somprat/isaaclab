"""Kinematic G1 diagnostic: exact recorded joint commands or unrestricted pose IK.

No policy execution or physics integration. Pelvis, legs and waist use recorded
poses. IK uses recorded Cartesian pose targets, including orientation capture.
"""

import argparse
import csv
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("commands", "all-commands", "ik", "matched-ik"), required=True)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.iterations < 1:
        parser.error("Use a fresh output directory and positive iteration count.")
    root = Path(__file__).resolve().parents[2]
    raw = np.load(args.source / "full_state_replay.npz", allow_pickle=False)
    frames, names, dt = raw["frames"], raw["joint_names"].tolist(), float(raw["dt"])
    targets, orientations = raw["targets_world"], raw["orientations_world_xyzw"]
    reference = list(csv.DictReader((args.source / "reference_tracking.csv").open()))
    xml = (
        root
        / "wb_humanoid_mpc/robot_models/unitree_g1/g1_description/urdf/g1_29dof.xml"
    )
    tree = ET.parse(xml)
    tree.getroot().find("compiler").set("meshdir", str(xml.parent.parent / "meshes"))
    visual = tree.getroot().find("visual")
    if visual is None:
        visual = ET.SubElement(tree.getroot(), "visual")
    settings = visual.find("global")
    if settings is None:
        settings = ET.SubElement(visual, "global")
    settings.set("offwidth", "960")
    settings.set("offheight", "720")
    model = mujoco.MjModel.from_xml_string(
        ET.tostring(tree.getroot(), encoding="unicode")
    )
    data = mujoco.MjData(model)
    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names
    ]
    assert min(joint_ids) >= 0
    addresses = model.jnt_qposadr[joint_ids]
    arms = [
        i
        for i, name in enumerate(names)
        if any(x in name for x in ("shoulder", "elbow", "wrist"))
    ]
    assert len(arms) == 14
    arm_addresses = addresses[arms]
    # Direct qpos assignment plus mj_forward does not enforce these limits either;
    # disable them explicitly to make the diagnostic configuration unambiguous.
    model.jnt_limited[:] = False
    model.actuator_ctrllimited[:] = False
    model.actuator_forcelimited[:] = False
    torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    sides = []
    for side in ("left", "right"):
        slots = [i for i in arms if names[i].startswith(side + "_")]
        body = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, side + "_wrist_yaw_link"
        )
        sides.append(
            (
                side,
                body,
                addresses[slots],
                model.jnt_dofadr[np.array(joint_ids)[slots]],
                np.array([0.0415, 0.003 if side == "left" else -0.003, 0.0]),
            )
        )

    def hand(body, offset):
        return data.xpos[body] + data.xmat[body].reshape(3, 3) @ offset

    args.output.mkdir(parents=True)
    renderer = None if args.no_video else mujoco.Renderer(model, height=720, width=960)
    writer = (
        None
        if args.no_video
        else imageio.get_writer(
            args.output / "replay.mp4",
            fps=round(1 / dt),
            codec="libx264",
            quality=8,
            macro_block_size=1,
        )
    )
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth, camera.elevation, camera.distance = -55, -15, 2.5
    previous = frames[0, 7 : 7 + len(names)][arms].copy()
    rows, validation = [], []
    try:
        for i, frame in enumerate(frames):
            data.qpos[:3] = frame[:3]
            data.qpos[3:7] = frame[[6, 3, 4, 5]]
            data.qpos[addresses] = frame[7 : 7 + len(names)]
            mujoco.mj_forward(model, data)
            for side, body, _, _, offset in sides:
                measured = data.xmat[torso].reshape(3, 3).T @ (
                    hand(body, offset) - data.xpos[torso]
                )
                logged = np.array(
                    [
                        float(reference[i][f"{side}_hand_actual_{a}_torso_m"])
                        for a in "xyz"
                    ]
                )
                validation.append(float(np.linalg.norm(measured - logged)))
            if args.mode == "all-commands":
                # Root motion has no joint-position target; anchor it to the recorded pose.
                data.qpos[addresses] = frame[7 + len(names) : 7 + 2 * len(names)]
                mujoco.mj_forward(model, data)
            elif args.mode == "commands":
                if i > 0:  # First frame has no preceding actuator command.
                    data.qpos[arm_addresses] = frame[
                        7 + len(names) : 7 + 2 * len(names)
                    ][arms]
                mujoco.mj_forward(model, data)
            else:
                data.qpos[arm_addresses] = previous
                mujoco.mj_forward(model, data)
                for side_index, (_, body, qpos_ids, dof_ids, offset) in enumerate(
                    sides
                ):
                    desired_quat = orientations[i, side_index][[3, 0, 1, 2]]
                    for _ in range(args.iterations):
                        point = hand(body, offset)
                        rot_error = np.zeros(3)
                        mujoco.mju_subQuat(rot_error, desired_quat, data.xquat[body])
                        error = np.r_[
                            targets[i, side_index] - point,
                            data.xmat[body].reshape(3, 3) @ rot_error,
                        ]
                        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
                        mujoco.mj_jac(model, data, jp, jr, point, body)
                        jac = np.vstack((jp[:, dof_ids], jr[:, dof_ids]))
                        correction = jac.T @ np.linalg.solve(
                            jac @ jac.T + 0.05**2 * np.eye(6), error
                        )
                        if args.mode == "matched-ik":
                            correction = np.clip(correction, -0.05, 0.05)
                            side_joint_ids = [joint_ids[slot] for slot in arms if names[slot].startswith(side + "_")]
                            ranges = model.jnt_range[side_joint_ids]
                            center = ranges.mean(axis=1)
                            half_range = 0.9 * (ranges[:, 1] - ranges[:, 0]) / 2
                            data.qpos[qpos_ids] = np.clip(
                                data.qpos[qpos_ids] + correction,
                                center - half_range,
                                center + half_range,
                            )
                        else:
                            data.qpos[qpos_ids] += correction
                        mujoco.mj_forward(model, data)
                previous = data.qpos[arm_addresses].copy()
            row = {"time_s": i * dt}
            for side_index, (side, body, _, _, offset) in enumerate(sides):
                point = hand(body, offset)
                error = point - targets[i, side_index]
                row[f"{side}_position_error_m"] = float(np.linalg.norm(error))
                rot_error = np.zeros(3)
                mujoco.mju_subQuat(
                    rot_error,
                    orientations[i, side_index][[3, 0, 1, 2]],
                    data.xquat[body],
                )
                row[f"{side}_orientation_error_rad"] = float(np.linalg.norm(rot_error))
                for axis, value in zip("xyz", error):
                    row[f"{side}_world_error_{axis}_m"] = float(value)
            for slot in arms:
                row[names[slot]] = float(data.qpos[addresses[slot]])
            rows.append(row)
            if renderer is not None:
                camera.lookat[:] = [frame[0], frame[1], 0.82]
                renderer.update_scene(data, camera)
                for j, target in enumerate(targets[i]):
                    scene = renderer.scene
                    if scene.ngeom < scene.maxgeom:
                        mujoco.mjv_initGeom(
                            scene.geoms[scene.ngeom],
                            mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([0.018, 0.018, 0.018]),
                            target,
                            np.eye(3).reshape(-1),
                            np.array([1.0, 0.25 if j == 0 else 0.8, 0.1, 0.8]),
                        )
                        scene.ngeom += 1
                image = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, 960, 65), fill=(15, 20, 25))
                title = {
                    "commands": "RECORDED ARM COMMANDS; MEASURED BODY POSE",
                    "all-commands": "RECORDED ALL-JOINT COMMANDS; MEASURED PELVIS POSE",
                    "ik": "UNRESTRICTED POSE IK - ORIGINAL CARTESIAN TARGETS",
                    "matched-ik": "ONLINE IK; MEASURED BODY MOTION; WORLD POSE TARGETS",
                }[args.mode]
                draw.text((12, 8), f"{title} | {i * dt:.2f} s", fill="white")
                draw.text(
                    (12, 27),
                    "Recorded pelvis; no policy or dynamics. Spheres = hand targets.",
                    fill="white",
                )
                draw.text(
                    (12, 46),
                    f"Position error L/R: {row['left_position_error_m'] * 100:.1f} / {row['right_position_error_m'] * 100:.1f} cm",
                    fill="white",
                )
                writer.append_data(np.asarray(image))
            if i % 250 == 0:
                print(f"{args.mode}: {i}/{len(frames)}", flush=True)
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
    with (args.output / "errors.csv").open("w") as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    metadata = {
        "source": str(args.source.resolve()),
        "mode": args.mode,
        "frames": len(frames),
        "duration_s": len(frames) * dt,
        "ik_iterations_per_frame": args.iterations if args.mode in ("ik", "matched-ik") else None,
        "max_model_position_disagreement_m": max(validation),
        "note": "Matched-ik mode re-solves online from the previous displayed arms, recorded measured body pose, and recorded world hand targets with DLS damping 0.05, per-joint 0.05 rad correction cap, and 0.9 soft joint ranges. No physics or policy runs during replay. All-commands mode applies all recorded joint position targets; commands mode applies only arm targets.",
        "post7": {},
    }
    for side in ("left", "right"):
        e = np.array([r[f"{side}_position_error_m"] for r in rows if r["time_s"] >= 7])
        metadata["post7"][side] = {
            "mean_error_m": float(e.mean()),
            "max_error_m": float(e.max()),
        }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
