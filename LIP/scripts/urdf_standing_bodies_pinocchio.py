"""Diagnostic: evaluate the wb_humanoid_mpc G1 URDF (Pinocchio) at the exact configuration in an Isaac body dump.

Runs inside the wb-mpc-dev container (Pinocchio). Root pose and joints are set by name from the Isaac dump.
Reports, per moving body: link-frame origin, URDF link-only inertial CoM, and Pinocchio joint-body CoM
(fixed children merged, as in the MPC model and Isaac's merge_fixed_joints conversion); plus whole-body CoM and
foot reference points (Isaac sole offset and MPC contact-frame offset).
usage: python3 urdf_standing_bodies_pinocchio.py ISAAC_JSON URDF TASK_INFO OUTPUT_JSON
"""

import json
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin

isaac_path, urdf_path, task_path, output_path = sys.argv[1:5]
isaac = json.load(open(isaac_path))
model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
data = model.createData()

q = pin.neutral(model)
q[:3] = isaac["root_pos"]
q[3:7] = isaac["root_quat_xyzw"]  # Pinocchio free-flyer quaternion is x, y, z, w
joints = dict(zip(isaac["joint_names"], isaac["joint_pos"]))
for jid in range(2, model.njoints):
    name = model.names[jid]
    q[model.joints[jid].idx_q] = joints.pop(name)
assert not joints, f"Isaac joints missing in URDF: {joints}"
pin.forwardKinematics(model, data, q)
pin.updateFramePlacements(model, data)
whole_com = pin.centerOfMass(model, data, q, False)

links = {l.get("name"): l for l in ET.parse(urdf_path).getroot().findall("link")}


def link_inertial(name):
    inertial = links[name].find("inertial")
    xyz = np.array([float(v) for v in inertial.find("origin").get("xyz").split()])
    return xyz, float(inertial.find("mass").get("value"))


task = open(task_path).read()
block = re.search(r"contact_frame_translation\s*\{([^}]*)\}", task).group(1)
contact_offset = np.array([float(re.search(rf"\b{a}\s+(-?[\d.eE+-]+)", block).group(1)) for a in "xyz"])

bodies = {}
for name in isaac["body_names"]:
    frame = model.frames[model.getFrameId(name, pin.BODY)]
    oMf = data.oMf[model.getFrameId(name, pin.BODY)]
    jid = frame.parentJoint if hasattr(frame, "parentJoint") else frame.parent
    offset, mass = link_inertial(name)
    entry = {
        "link_pos": oMf.translation.tolist(),
        "link_only_com": oMf.act(offset).tolist(),
        "link_only_mass": mass,
    }
    if frame.placement.isIdentity() and model.frames[model.getFrameId(name, pin.BODY)].name == name:
        inertia = model.inertias[jid]
        entry["merged_com"] = data.oMi[jid].act(inertia.lever).tolist()
        entry["merged_mass"] = float(inertia.mass)
    bodies[name] = entry

feet = {}
for side in ("left", "right"):
    oMf = data.oMf[model.getFrameId(f"{side}_ankle_roll_link", pin.BODY)]
    feet[side] = {
        "isaac_sole_point": oMf.act(np.array(isaac["foot_sole_offset_in_ankle_roll"])).tolist(),
        "mpc_contact_frame": oMf.act(contact_offset).tolist(),
    }
out = {
    "source": "Pinocchio " + pin.__version__ + " on wb_humanoid_mpc URDF; exact Isaac root pose and joints",
    "urdf": urdf_path,
    "total_mass": float(sum(model.inertias[j].mass for j in range(model.njoints))),
    "mpc_contact_offset_in_ankle_roll": contact_offset.tolist(),
    "whole_body_com": whole_com.tolist(),
    "bodies": bodies,
    "feet": feet,
}
json.dump(out, open(output_path, "w"), indent=2)
print(f"whole CoM {whole_com}, torso link-only CoM {bodies['torso_link']['link_only_com']}, merged {bodies['torso_link'].get('merged_com')}")
