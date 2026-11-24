"""Export a LIP reference NPZ as a plain CSV that the wb_humanoid_mpc container can read.

The container mounts only wb_humanoid_mpc/, so the file is written there (default
wb_humanoid_mpc/lip_references/<npz stem>.csv; generated_motions/ is root-owned
because the container creates it, so the host cannot write there).

Format: '#' lines hold metadata as 'key: value'; the first non-'#' line is the
comma-separated header, then one row per sample. Units are SI, world z-up.
Feet are exported at the MPC contact frames (foot_*_contact): the LIP/Isaac sole
point moved +0.035 m along the walking heading (LIP has no foot yaw; feet
are assumed to point along the constant heading).
The calibrated posture (base pose and joints) is written as metadata for the MPC initial state,
with the root lifted by -mean_contact_frame_height so the contact frames sit at z=0 like LIP.

Checks: schema 3, world_z_up, feet ordered left/right, time starting at 0 with uniform
dt, valid support mask (never flight; stance feet on the ground), constant CoM
height equal to a successful schema-2 height calibration inside wb_humanoid_mpc/.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
WB_ROOT = WORKSPACE / "wb_humanoid_mpc"
CONTAINER_WB_ROOT = Path("/wb_humanoid_mpc_ws/src/wb_humanoid_mpc")
MPC_CONTACT_FORWARD = 0.035  # foot_*_contact x offset in *_ankle_roll_link relative to the Isaac sole point
TOL = 1e-6


def fail(message):
    raise SystemExit(f"export_lip_to_mpc: {message}")


def container_path(path: Path) -> str:
    return str(CONTAINER_WB_ROOT / path.resolve().relative_to(WB_ROOT))


def check(d):
    for key in ("schema_version", "frame", "foot_names", "time", "dt", "com_pos", "com_vel", "foot_pos",
                "foot_vel", "support_mask", "heading", "com_height", "com_height_source", "height_calibration_source"):
        if key not in d.files:
            fail(f"NPZ lacks '{key}'")
    if int(d["schema_version"]) != 3:
        fail(f"schema_version {int(d['schema_version'])} != 3")
    if str(d["frame"]) != "world_z_up":
        fail(f"frame {d['frame']} != world_z_up")
    if d["foot_names"].tolist() != ["left", "right"]:
        fail(f"foot_names {d['foot_names'].tolist()} != ['left', 'right']")

    t, dt = d["time"], float(d["dt"])
    n = len(t)
    if n < 2 or abs(t[0]) > 1e-9:
        fail("time must start at 0 with at least two samples")
    steps = np.diff(t)
    if not np.allclose(steps, dt, atol=1e-9, rtol=0):
        fail(f"time is not uniformly spaced at dt={dt} (diff range {steps.min()}..{steps.max()})")
    shapes = {"com_pos": (n, 3), "com_vel": (n, 3), "foot_pos": (n, 2, 3), "foot_vel": (n, 2, 3), "support_mask": (n, 2)}
    for key, shape in shapes.items():
        if d[key].shape != shape:
            fail(f"{key} shape {d[key].shape} != {shape}")
        if key != "support_mask" and not np.isfinite(d[key]).all():
            fail(f"{key} has non-finite values")

    support = d["support_mask"]
    if support.dtype != bool:
        fail(f"support_mask dtype {support.dtype} is not bool")
    if not support.any(axis=1).all():
        fail(f"support_mask has flight samples at t={t[~support.any(axis=1)][:5]}")
    feet_z = d["foot_pos"][..., 2]
    if np.abs(feet_z[support]).max() > TOL:
        fail(f"stance foot off the ground by {np.abs(feet_z[support]).max():.3e} m")
    if feet_z.min() < -TOL:
        fail(f"foot below the ground: {feet_z.min():.3e} m")
    if not support[0].all():
        fail("reference must start in double support")

    if str(d["com_height_source"]) != "height_calibration":
        fail(f"com_height_source is '{d['com_height_source']}', expected height_calibration")
    cal_path = Path(str(d["height_calibration_source"]))
    if not cal_path.is_file():
        fail(f"height calibration {cal_path} not found")
    try:
        cal_path.resolve().relative_to(WB_ROOT)
    except ValueError:
        fail(f"height calibration {cal_path} is outside {WB_ROOT}; the container could not see it")
    cal = json.loads(cal_path.read_text())
    if cal.get("kind") != "g1_torso_origin_height_calibration" or cal.get("schema_version") != 2 or not cal.get("success"):
        fail(f"{cal_path} is not a successful schema-2 height calibration")
    lip_height = float(cal["whole_body_com_height"]) - float(cal["mean_contact_frame_height"])
    com_z = d["com_pos"][:, 2]
    if np.ptp(com_z) > TOL or abs(com_z[0] - lip_height) > TOL or abs(float(d["com_height"]) - lip_height) > TOL:
        fail(f"CoM height {com_z.min():.6f}..{com_z.max():.6f} does not match calibration {lip_height:.6f}")
    return cal_path, cal, lip_height


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("npz", type=Path, help="LIP reference NPZ (demo_LIPM_3D_vt.py output)")
    parser.add_argument("--output", type=Path, help="CSV path inside wb_humanoid_mpc/")
    parser.add_argument("--dt", type=float, help="Output sample period [s]; an integer multiple of the NPZ dt (default: NPZ dt)")
    args = parser.parse_args()

    d = np.load(args.npz)
    cal_path, cal, lip_height = check(d)
    src_dt = float(d["dt"])
    stride = 1 if args.dt is None else round(args.dt / src_dt)
    if stride < 1 or abs(stride * src_dt - (args.dt or src_dt)) > 1e-9:
        fail(f"--dt {args.dt} is not a positive integer multiple of {src_dt}")
    idx = np.arange(0, len(d["time"]), stride)
    if idx[-1] != len(d["time"]) - 1:
        idx = np.append(idx, len(d["time"]) - 1)  # keep the final sample; the last interval may be shorter

    output = args.output or WB_ROOT / "lip_references" / f"{args.npz.stem}.csv"
    try:
        output.resolve().relative_to(WB_ROOT)
    except ValueError:
        fail(f"output {output} must be inside {WB_ROOT}")

    heading = float(d["heading"])
    forward = np.array([np.cos(heading), np.sin(heading), 0.0])
    contact = d["foot_pos"] + MPC_CONTACT_FORWARD * forward
    columns = ["time", "left_contact", "right_contact",
               "com_x", "com_y", "com_z", "com_vx", "com_vy", "com_vz",
               "left_x", "left_y", "left_z", "right_x", "right_y", "right_z",
               "left_vx", "left_vy", "left_vz", "right_vx", "right_vy", "right_vz"]
    rows = np.column_stack([
        d["time"][idx], d["support_mask"][idx].astype(int),
        d["com_pos"][idx], d["com_vel"][idx],
        contact[idx, 0], contact[idx, 1], d["foot_vel"][idx, 0], d["foot_vel"][idx, 1],
    ])

    meta = {
        "format": "lip_reference_csv_v2",
        "frame": "world_z_up",
        "units": "m, m/s, s",
        "source_npz": str(args.npz.resolve()),
        "source_npz_sha256": hashlib.sha256(args.npz.read_bytes()).hexdigest(),
        "height_calibration": container_path(cal_path),
        "requested_torso_origin_height": cal["requested_torso_origin_height"],
        "lip_com_height": f"{lip_height:.9f}",
        "heading": f"{heading:.9f}",
        "foot_point": f"MPC foot_*_contact frame = LIP sole + {MPC_CONTACT_FORWARD} m along heading",
        "contact_flags": "1 = stance (LIP support_mask), 0 = swing",
        "source_dt": src_dt,
        "dt": stride * src_dt,
        "samples": len(idx),
        "duration": float(d["time"][-1]),
    }
    base = list(map(float, cal["base_pose_xyz_yaw_pitch_roll"]))
    base[2] -= float(cal["mean_contact_frame_height"])
    if len(base) != 6 or len(cal["joint_names"]) != len(cal["joint_pos"]):
        fail("calibration base pose or joint arrays are malformed")
    meta["initial_base_pose_xyz_yaw_pitch_roll"] = ",".join(f"{v:.10g}" for v in base)
    meta["initial_joint_names"] = ",".join(cal["joint_names"])
    meta["initial_joint_pos"] = ",".join(f"{float(v):.10g}" for v in cal["joint_pos"])
    meta["initial_pose_source"] = "height_calibration posture, root z - mean_contact_frame_height"
    if "initial_state_source" in d.files:
        meta["initial_state_source"] = str(d["initial_state_source"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for key, value in meta.items():
            f.write(f"# {key}: {value}\n")
        f.write(",".join(columns) + "\n")
        for row in rows:
            f.write(",".join([f"{row[0]:.6f}", str(int(row[1])), str(int(row[2]))] + [f"{v:.9f}" for v in row[3:]]) + "\n")
    print(f"Wrote {len(idx)} samples (dt {stride * src_dt:.4f} s, {meta['duration']:.3f} s) to {output}")
    print(f"Container path: {container_path(output)}")


if __name__ == "__main__":
    main()
