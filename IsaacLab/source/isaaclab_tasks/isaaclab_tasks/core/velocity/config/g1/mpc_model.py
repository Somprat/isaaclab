# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Source-model contracts for the MPC-matched G1 reference asset."""

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from isaaclab.paths import ISAACLAB_ROOT

MPC_URDF = Path(ISAACLAB_ROOT).parent / "wb_humanoid_mpc/robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf"
MPC_ASSET_DIR = Path(ISAACLAB_ROOT) / "checkpoints/g1-29dof/mpc-matched-assets"
MPC_USD = MPC_ASSET_DIR / "g1_29dof/g1_29dof.usda"


def _origin(element):
    result = np.eye(4)
    if element is not None:
        result[:3, 3] = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
        result[:3, :3] = Rotation.from_euler("xyz", np.fromstring(element.get("rpy", "0 0 0"), sep=" ")).as_matrix()
    return result


def read_urdf_model(path=MPC_URDF):
    """Aggregate fixed-link inertias in the surviving movable-link frames.

    Inertias are about each composite body's COM, expressed in its link frame.
    This aggregation is independently compared with Pinocchio during validation.
    """
    root = ET.parse(path).getroot()
    links = {link.get("name"): link for link in root.findall("link")}
    joints = root.findall("joint")
    root_name = next(name for name in links if name not in {j.find("child").get("link") for j in joints})
    children = {name: [] for name in links}
    for joint in joints:
        children[joint.find("parent").get("link")].append(joint)
    bodies, active = {}, {}

    def walk(name, owner, local, world):
        if owner not in bodies:
            bodies[owner] = {
                "mass": 0.0,
                "first_moment": np.zeros(3),
                "inertia_origin": np.zeros((3, 3)),
                "position_zero": world[:3, 3].tolist(),
                "rotation_zero": world[:3, :3].tolist(),
                "collision_count": 0,
            }
        body = bodies[owner]
        link = links[name]
        body["collision_count"] += len(link.findall("collision"))
        inertial = link.find("inertial")
        if inertial is not None:
            mass = float(inertial.find("mass").get("value"))
            transform = local @ _origin(inertial.find("origin"))
            rotation, com = transform[:3, :3], transform[:3, 3]
            values = inertial.find("inertia").attrib
            tensor = np.array(
                [
                    [float(values["ixx"]), float(values["ixy"]), float(values["ixz"])],
                    [float(values["ixy"]), float(values["iyy"]), float(values["iyz"])],
                    [float(values["ixz"]), float(values["iyz"]), float(values["izz"])],
                ]
            )
            body["mass"] += mass
            body["first_moment"] += mass * com
            body["inertia_origin"] += rotation @ tensor @ rotation.T + mass * (
                com @ com * np.eye(3) - np.outer(com, com)
            )
        for joint in children[name]:
            child = joint.find("child").get("link")
            origin = _origin(joint.find("origin"))
            if joint.get("type") == "fixed":
                walk(child, owner, local @ origin, world @ origin)
            elif joint.get("type") == "revolute":
                limits = joint.find("limit").attrib
                active[joint.get("name")] = {
                    "parent": owner,
                    "child": child,
                    "origin": (local @ origin).tolist(),
                    "axis": np.fromstring(joint.find("axis").get("xyz"), sep=" ").tolist(),
                    **{key: float(limits[key]) for key in ("lower", "upper", "effort", "velocity")},
                }
                walk(child, child, np.eye(4), world @ origin)
            else:
                raise ValueError(f"Unsupported source joint type: {joint.get('type')}")

    walk(root_name, root_name, np.eye(4), np.eye(4))
    for body in bodies.values():
        mass = body["mass"]
        com = body.pop("first_moment") / mass
        inertia = body.pop("inertia_origin") - mass * (com @ com * np.eye(3) - np.outer(com, com))
        body["com"], body["inertia"] = com.tolist(), inertia.tolist()
    return {
        "source_urdf": str(Path(path).resolve()),
        "source_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "total_mass": sum(b["mass"] for b in bodies.values()),
        "root": root_name,
        "bodies": bodies,
        "joints": active,
    }


def forward_kinematics(model, joint_positions, root_pose):
    transforms = {model["root"]: np.eye(4)}
    transforms[model["root"]][:3, 3] = root_pose[:3]
    transforms[model["root"]][:3, :3] = Rotation.from_quat(root_pose[3:7]).as_matrix()
    pending = dict(model["joints"])
    while pending:
        progressed = False
        for name, joint in list(pending.items()):
            if joint["parent"] not in transforms:
                continue
            rotation = np.eye(4)
            rotation[:3, :3] = Rotation.from_rotvec(np.asarray(joint["axis"]) * joint_positions[name]).as_matrix()
            transforms[joint["child"]] = transforms[joint["parent"]] @ np.asarray(joint["origin"]) @ rotation
            del pending[name]
            progressed = True
        if not progressed:
            raise ValueError("Source joint graph has missing/cyclic parents.")
    return transforms


def validate_mpc_asset():
    """Fail before training if the matched asset or its source has changed."""
    import json

    manifest_path = MPC_ASSET_DIR / "model_manifest.json"
    if not MPC_USD.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            "Build the MPC-matched asset with scripts/reinforcement_learning/prepare_g1_mpc_asset.py."
        )
    manifest = json.loads(manifest_path.read_text())
    if hashlib.sha256(MPC_URDF.read_bytes()).hexdigest() != manifest["source_sha256"]:
        raise RuntimeError("MPC URDF changed; rebuild the matched asset before training.")
    for relative, expected in manifest["asset_files_sha256"].items():
        path = MPC_ASSET_DIR / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Matched asset changed or is missing: {path}; rebuild before training.")
    for relative, expected in manifest["source_meshes_sha256"].items():
        path = MPC_URDF.parent.parent / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"MPC mesh changed or is missing: {path}; rebuild before training.")


def validate_mpc_model_event(env, env_ids):
    """Startup guard for newly configured matched reference environments only."""
    if Path(env.cfg.scene.robot.spawn.usd_path).resolve() == MPC_USD.resolve():
        validate_mpc_asset()
