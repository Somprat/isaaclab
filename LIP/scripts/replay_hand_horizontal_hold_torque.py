"""Pose IK with torque-driven arms and recorded body motion enforced by driven constraints.

The root mocap weld and leg/waist joint equalities are external support, not a
walking controller. Their finite tracking error is measured. No arm state is
assigned after initialization. The original XML and kinematic replay are untouched.
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
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline

ROOT = Path(__file__).resolve().parents[2]
SUFFIXES = (
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow",
    "wrist_roll",
    "wrist_pitch",
    "wrist_yaw",
)


class TorqueReplay:
    def __init__(
        self,
        source,
        physics_dt=0.001,
        kp=40.0,
        kd=10.0,
        reach_end_s=14.0,
        preserve_position=False,
    ):
        if not np.isfinite(reach_end_s) or reach_end_s <= 1.0:
            raise ValueError(
                "Reach end must be finite and later than the 1-second initial hold"
            )
        self.reach_end_s = float(reach_end_s)
        self.preserve_position = preserve_position
        self.source = Path(source)
        raw = np.load(self.source / "full_state_replay.npz", allow_pickle=False)
        self.frames = raw["frames"]
        self.names = raw["joint_names"].tolist()
        self.dt = float(raw["dt"])
        self.substeps = round(self.dt / physics_dt)
        if self.substeps < 1 or not np.isclose(self.substeps * physics_dt, self.dt):
            raise ValueError(
                "Physics timestep must divide the recording control timestep"
            )
        self.times = np.arange(len(self.frames)) * self.dt
        self.position_spline = CubicSpline(self.times, self.frames[:, :3], axis=0)
        self.rotation_spline = RotationSpline(
            self.times, Rotation.from_quat(self.frames[:, 3:7])
        )
        self.joint_spline = CubicSpline(
            self.times, self.frames[:, 7 : 7 + len(self.names)], axis=0
        )
        self.reference = list(
            csv.DictReader((self.source / "reference_tracking.csv").open())
        )
        if len(self.reference) != len(self.frames):
            raise ValueError("Recording/CSV lengths disagree")
        arm_names = [
            f"{side}_{suffix}_joint"
            for side in ("left", "right")
            for suffix in SUFFIXES
        ]
        self.arm_slots = [self.names.index(name) for name in arm_names]
        self.body_slots = [i for i in range(len(self.names)) if i not in self.arm_slots]
        xml = (
            ROOT
            / "wb_humanoid_mpc/robot_models/unitree_g1/g1_description/urdf/g1_29dof.xml"
        )
        tree = ET.parse(xml)
        root = tree.getroot()
        root.find("compiler").set("meshdir", str(xml.parent.parent / "meshes"))
        option = ET.SubElement(
            root,
            "option",
            timestep=str(physics_dt),
            integrator="implicitfast",
            iterations="100",
            tolerance="1e-10",
        )
        # External body support replaces ground contact. Exclude contacts to isolate
        # arm inertial/gravity/actuator response; native arm joint limits remain.
        ET.SubElement(option, "flag", contact="disable")
        visual = root.find("visual")
        if visual is None:
            visual = ET.SubElement(root, "visual")
        global_cfg = visual.find("global")
        if global_cfg is None:
            global_cfg = ET.SubElement(visual, "global")
        global_cfg.set("offwidth", "960")
        global_cfg.set("offheight", "720")
        ET.SubElement(
            root.find("worldbody"),
            "body",
            name="recorded_pelvis",
            mocap="true",
            pos="0 0 0.793",
        )
        eq = ET.SubElement(root, "equality")
        constraint_cfg = {"solref": "0.002 1", "solimp": "0.9999 0.9999 0.001"}
        ET.SubElement(
            eq,
            "weld",
            name="recorded_root",
            body1="pelvis",
            body2="recorded_pelvis",
            relpose="0 0 0 1 0 0 0",
            **constraint_cfg,
        )
        for slot in self.body_slots:
            name = self.names[slot]
            ET.SubElement(
                eq,
                "joint",
                name="recorded_" + name,
                joint1=name,
                polycoef="0 0 0 0 0",
                **constraint_cfg,
            )
        actuators = root.find("actuator")
        for motor in list(actuators):
            name = motor.get("joint")
            actuators.remove(motor)
            if name in arm_names:
                limit = 5.0 if "wrist_pitch" in name or "wrist_yaw" in name else 25.0
                # Native affine actuator evaluates the PD torque inside MuJoCo and
                # lets implicitfast account for velocity feedback in integration.
                ET.SubElement(
                    actuators,
                    "general",
                    name=name,
                    joint=name,
                    gainprm=str(kp),
                    biastype="affine",
                    biasprm=f"0 {-kp} {-kd}",
                    forcelimited="true",
                    forcerange=f"{-limit} {limit}",
                )
        self.model = mujoco.MjModel.from_xml_string(
            ET.tostring(root, encoding="unicode")
        )
        self.data = mujoco.MjData(self.model)
        self.reference_data = mujoco.MjData(self.model)
        self.jids = np.array([self.model.joint(n).id for n in self.names])
        self.qids = self.model.jnt_qposadr[self.jids]
        self.vids = self.model.jnt_dofadr[self.jids]
        self.arm_q = self.qids[self.arm_slots]
        self.arm_v = self.vids[self.arm_slots]
        self.arm_act = np.array([self.model.actuator(n).id for n in arm_names])
        self.body_eq = np.array(
            [
                self.model.equality("recorded_" + self.names[s]).id
                for s in self.body_slots
            ]
        )
        self.torso = self.model.body("torso_link").id
        self.hands = [
            self.model.body(s + "_wrist_yaw_link").id for s in ("left", "right")
        ]
        self.offsets = np.array([[0.0415, 0.003, 0], [0.0415, -0.003, 0]])
        ranges = self.model.jnt_range[self.jids[self.arm_slots]]
        self.soft_lo = ranges.mean(axis=1) - 0.45 * (ranges[:, 1] - ranges[:, 0])
        self.soft_hi = ranges.mean(axis=1) + 0.45 * (ranges[:, 1] - ranges[:, 0])
        self.kp, self.kd = kp, kd
        self.limit = self.model.actuator_forcerange[self.arm_act, 1]
        self.held_xy = self.held_quat = None
        self.q_des = self.frames[0, 7 : 7 + len(self.names)][self.arm_slots].copy()
        self.data.qpos[:3] = self.frames[0, :3]
        self.data.qpos[3:7] = self.frames[0, [6, 3, 4, 5]]
        self.data.qpos[self.qids] = self.frames[0, 7 : 7 + len(self.names)]
        # Estimate initial generalized velocities from the recording, including
        # quaternion-aware base rotation. Later states are exclusively simulated.
        q1 = self.data.qpos.copy()
        epsilon = 1e-5
        q1[:3] = self.position_spline(epsilon)
        q1[3:7] = self.rotation_spline(epsilon).as_quat()[[3, 0, 1, 2]]
        q1[self.qids] = self.joint_spline(epsilon)
        mujoco.mj_differentiatePos(
            self.model, self.data.qvel, epsilon, self.data.qpos, q1
        )
        self.drive_body(0.0)
        self.data.ctrl[self.arm_act] = self.q_des
        mujoco.mj_forward(self.model, self.data)
        self.max_fk_error = 0.0
        # Undo each recorded torso rotation BEFORE retiming. XY are offsets in
        # world axes; Z is absolute world height, independent of torso bobbing.
        path = []
        for i in range(len(self.frames)):
            world = self.recorded_targets(i)
            world[:, :2] -= self.reference_data.xpos[self.torso, :2]
            path.append(world)
        self.target_path = np.asarray(path)

    def drive_body(self, time):
        # Smooth interpolation, evaluated at every physics substep; no qpos/qvel writes.
        time = min(time, self.times[-1])
        self.data.mocap_pos[0] = self.position_spline(time)
        self.data.mocap_quat[0] = self.rotation_spline(time).as_quat()[[3, 0, 1, 2]]
        desired = self.joint_spline(time)[self.body_slots]
        self.model.eq_data[self.body_eq, 0] = (
            desired - self.model.qpos0[self.qids[self.body_slots]]
        )

    def points(self, data=None):
        data = self.data if data is None else data
        return np.array(
            [
                data.xpos[b] + data.xmat[b].reshape(3, 3) @ o
                for b, o in zip(self.hands, self.offsets)
            ]
        )

    def recorded_targets(self, i):
        frame, d = self.frames[i], self.reference_data
        d.qpos[:3] = frame[:3]
        d.qpos[3:7] = frame[[6, 3, 4, 5]]
        d.qpos[self.qids] = frame[7 : 7 + len(self.names)]
        mujoco.mj_forward(self.model, d)
        rotation = d.xmat[self.torso].reshape(3, 3)
        local = np.array(
            [
                [
                    float(self.reference[i][f"{s}_hand_target_{a}_torso_m"])
                    for a in "xyz"
                ]
                for s in ("left", "right")
            ]
        )
        targets = d.xpos[self.torso] + local @ rotation.T
        self.max_fk_error = max(
            self.max_fk_error,
            max(
                abs(
                    self.points(d)[j, 2]
                    - float(self.reference[i][f"{s}_hand_actual_z_world_m"])
                )
                for j, s in enumerate(("left", "right"))
            ),
        )
        return targets

    def target_time(self, time):
        if time <= 1.0:
            return max(0.0, float(time))
        return float(
            1.0 + 6.0 * np.clip((time - 1.0) / (self.reach_end_s - 1.0), 0.0, 1.0)
        )

    def targets(self, i):
        source_time = self.target_time(self.times[i])
        targets = np.array(
            [
                [
                    np.interp(source_time, self.times, self.target_path[:, h, a])
                    for a in range(3)
                ]
                for h in range(2)
            ]
        )
        targets[:, :2] += self.data.xpos[self.torso, :2]
        if self.held_xy is None and self.times[i] >= self.reach_end_s:
            self.held_xy = self.points()[:, :2] - self.data.xpos[self.torso, :2]
            self.held_quat = self.data.xquat[self.hands].copy()
        if self.held_xy is not None and not self.preserve_position:
            targets[:, :2] = self.data.xpos[self.torso, :2] + self.held_xy
        orientations = (
            np.repeat(self.data.xquat[self.torso][None], 2, axis=0)
            if self.held_quat is None
            else self.held_quat.copy()
        )
        return targets, orientations

    def solve_ik(self, targets, orientations):
        increments = []
        for hand in range(2):
            sl = slice(hand * 7, (hand + 1) * 7)
            body = self.hands[hand]
            point = self.points()[hand]
            jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
            mujoco.mj_jac(self.model, self.data, jp, jr, point, body)
            rotation_error = np.zeros(3)
            mujoco.mju_subQuat(
                rotation_error, orientations[hand], self.data.xquat[body]
            )
            error = np.r_[
                targets[hand] - point,
                self.data.xmat[body].reshape(3, 3) @ rotation_error,
            ]
            jac = np.vstack((jp[:, self.arm_v[sl]], jr[:, self.arm_v[sl]]))
            delta = jac.T @ np.linalg.solve(jac @ jac.T + 0.05**2 * np.eye(6), error)

            caps = np.full(7, 0.05)
            caps[0] = 0.30

            delta = np.clip(delta, -caps, caps)
            self.q_des[sl] = np.clip(
                self.data.qpos[self.arm_q[sl]] + delta,
                self.soft_lo[sl],
                self.soft_hi[sl],
            )
            increments.extend(delta)
        self.data.ctrl[self.arm_act] = self.q_des
        return np.array(increments)

    def advance(self, time):
        self.drive_body(time)
        mujoco.mj_step(self.model, self.data)
        if (
            not np.isfinite(self.data.qpos).all()
            or not np.isfinite(self.data.qvel).all()
        ):
            raise RuntimeError("Nonfinite dynamics")

    def body_errors(self, time):
        time = min(time, self.times[-1])
        root_error = np.linalg.norm(self.data.qpos[:3] - self.position_spline(time))
        joint_error = np.max(
            abs(
                self.data.qpos[self.qids[self.body_slots]]
                - self.joint_spline(time)[self.body_slots]
            )
        )
        actual = Rotation.from_quat(self.data.qpos[[4, 5, 6, 3]])
        angle = (self.rotation_spline(time).inv() * actual).magnitude()
        return root_error, joint_error, angle


def run(args):
    if args.output.exists():
        raise ValueError("Choose a fresh output directory")
    replay = TorqueReplay(
        args.source,
        args.physics_dt,
        reach_end_s=args.reach_end,
        preserve_position=args.preserve_position,
    )
    n = (
        len(replay.frames)
        if args.duration is None
        else min(len(replay.frames), round(args.duration / replay.dt))
    )
    if n < 1:
        raise ValueError("Duration must contain at least one control frame")
    args.output.mkdir(parents=True)
    renderer = (
        None if args.no_video else mujoco.Renderer(replay.model, height=720, width=960)
    )
    writer = (
        None
        if args.no_video
        else imageio.get_writer(
            args.output / "torque_replay.mp4",
            fps=round(1 / replay.dt),
            codec="libx264",
            quality=8,
        )
    )
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation, camera.distance = -55, -15, 2.5
    rows, subrows = [], []
    try:
        for i in range(n):
            t = replay.times[i]
            mujoco.mj_forward(replay.model, replay.data)
            targets, orientations = replay.targets(i)
            increments = replay.solve_ik(targets, orientations)
            row = {
                "time_s": t,
                "source_target_time_s": replay.target_time(t),
                "hold_active": replay.held_xy is not None,
            }
            for h, side in enumerate(("left", "right")):
                error = replay.points()[h] - targets[h]
                row[f"{side}_position_error_m"] = float(np.linalg.norm(error))
                for a, value in zip("xyz", error):
                    row[f"{side}_world_error_{a}_m"] = value
            root_err, joint_err, angle_err = replay.body_errors(t)
            row.update(
                root_tracking_error_m=root_err,
                lower_joint_max_error_rad=joint_err,
                root_orientation_error_rad=angle_err,
            )
            for j, slot in enumerate(replay.arm_slots):
                name = replay.names[slot]
                row[name + "_actual_rad"] = replay.data.qpos[replay.arm_q[j]]
                row[name + "_desired_rad"] = replay.q_des[j]
                row[name + "_ik_increment_rad"] = increments[j]
            rows.append(row)
            if renderer is not None:
                camera.lookat[:] = [*replay.data.qpos[:2], 0.82]
                renderer.update_scene(replay.data, camera)
                for target in targets:
                    scene = renderer.scene
                    mujoco.mjv_initGeom(
                        scene.geoms[scene.ngeom],
                        mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.full(3, 0.018),
                        target,
                        np.eye(3).reshape(-1),
                        np.array([1.0, 0.4, 0.1, 0.8]),
                    )
                    scene.ngeom += 1
                image = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, 960, 65), fill=(15, 20, 25))
                draw.text(
                    (12, 8),
                    f"TORQUE-DRIVEN ARMS | shoulder pitch cap 0.30; others 0.05 rad | {t:.2f} s",
                    fill="white",
                )
                draw.text(
                    (12, 27),
                    "Recorded body driven by constraints; arm states evolved by physics; no policy.",
                    fill="white",
                )
                draw.text(
                    (12, 46),
                    f"Hand errors L/R: {row['left_position_error_m'] * 100:.1f}/{row['right_position_error_m'] * 100:.1f} cm | {'Original position' if replay.preserve_position else 'Captured XY'}; orientation hold {replay.reach_end_s:g} s",
                    fill="white",
                )
                writer.append_data(np.asarray(image))
            for sub in range(replay.substeps):
                time = t + sub * args.physics_dt
                replay.drive_body(time)
                mujoco.mj_forward(replay.model, replay.data)
                requested = (
                    replay.kp * (replay.q_des - replay.data.qpos[replay.arm_q])
                    - replay.kd * replay.data.qvel[replay.arm_v]
                )
                applied = replay.data.actuator_force[replay.arm_act].copy()
                np.testing.assert_allclose(
                    applied, np.clip(requested, -replay.limit, replay.limit), atol=1e-9
                )
                sr = {"time_s": time}
                for j, slot in enumerate(replay.arm_slots):
                    name = replay.names[slot]
                    sr[name + "_velocity_rad_s"] = replay.data.qvel[replay.arm_v[j]]
                    sr[name + "_requested_nm"] = requested[j]
                    sr[name + "_applied_nm"] = applied[j]
                subrows.append(sr)
                replay.advance(time)
            if i % 250 == 0:
                print(
                    f"Frame {i}/{n}; hand error {row['left_position_error_m']:.3f}/{row['right_position_error_m']:.3f} m",
                    flush=True,
                )
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
    for name, values in (("tracking.csv", rows), ("torques.csv", subrows)):
        with (args.output / name).open("w") as stream:
            w = csv.DictWriter(stream, fieldnames=list(values[0]))
            w.writeheader()
            w.writerows(values)
    metadata = {
        "source": str(args.source.resolve()),
        "duration_s": n * replay.dt,
        "physics_dt": args.physics_dt,
        "control_dt": replay.dt,
        "kp": replay.kp,
        "kd": replay.kd,
        "ik_step_cap (shoulder)": 0.30,
        "ik_step_cap_rad (except shoulder)": 0.05,
        "dls_damping": 0.05,
        "persistent_accumulation": False,
        "body_drive": "Cubic position/joint splines and RotationSpline drive mocap root weld and lower-body joint equalities; compliant external support, not free walking.",
        "arm_dynamics": "Native torque-limited affine PD actuators; implicitfast integration. Contacts disabled for isolated arm test; native joint limits enabled.",
        "reach_timing": {
            "initial_hold_s": 1.0,
            "source_reach_end_s": 7.0,
            "reach_end_s": replay.reach_end_s,
            "capture_time_s": replay.reach_end_s,
        },
        "target_retiming": "Interpolate recorded world-axis XY offsets and absolute world Z at slowed source time; attach XY to current simulated torso. Body recording runs at normal speed.",
        "preserve_original_position": replay.preserve_position,
        "hold": (
            f"Original world-axis XY offsets and absolute world Z endpoint retained; measured world orientation captured at {replay.reach_end_s:g} seconds."
            if replay.preserve_position
            else f"Measured simulated XY offsets and world orientations captured at {replay.reach_end_s:g} seconds; retimed original world Z retained."
        ),
        "max_model_z_disagreement_m": replay.max_fk_error,
        "body_tracking": {
            k: float(max(r[k] for r in rows))
            for k in (
                "root_tracking_error_m",
                "lower_joint_max_error_rad",
                "root_orientation_error_rad",
            )
        },
        "post_hold": {},
    }
    after = [r for r in rows if r["time_s"] >= replay.reach_end_s]
    if after:
        metadata["post_hold"] = {
            s: {
                "mean_position_error_m": float(
                    np.mean([r[s + "_position_error_m"] for r in after])
                ),
                "max_position_error_m": float(
                    max(r[s + "_position_error_m"] for r in after)
                ),
            }
            for s in ("left", "right")
        }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=ROOT / "LIP/previews/perfect-arm-replay-source"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--physics-dt", type=float, default=0.001)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument(
        "--preserve-position",
        action="store_true",
        help="Keep original Cartesian position targets after the reach; capture orientation only.",
    )
    parser.add_argument(
        "--reach-end",
        type=float,
        default=14.0,
        help="Finish the retimed reach and capture the hold at this second.",
    )
    run(parser.parse_args())
