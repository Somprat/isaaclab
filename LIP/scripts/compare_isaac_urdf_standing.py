"""Diagnostic: compare Isaac and wb_humanoid_mpc URDF (Pinocchio) bodies in the same standing configuration.

Inputs come from dump_isaac_standing_bodies.py and urdf_standing_bodies_pinocchio.py. Writes a height
comparison JSON/CSV and a PNG overlay (side + front view) with a per-body height-difference panel.
"""

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dir", type=Path, default=ROOT / "LIP/reports/isaac-vs-mpc-standing")
parser.add_argument("--urdf", type=Path, default=ROOT / "wb_humanoid_mpc/robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf")
args = parser.parse_args()
isaac = json.loads((args.dir / "isaac_bodies.json").read_text())
urdf = json.loads((args.dir / "urdf_bodies.json").read_text())

names = isaac["body_names"]
I_link = dict(zip(names, np.array(isaac["body_pos"])))
I_com = dict(zip(names, np.array(isaac["body_com_pos"])))
I_mass = dict(zip(names, isaac["body_mass"]))
U = {n: {k: np.array(v) if isinstance(v, list) else v for k, v in b.items()} for n, b in urdf["bodies"].items()}

rows = []
for n in names:
    rows.append({
        "body": n,
        "isaac_link_z": I_link[n][2], "urdf_link_z": U[n]["link_pos"][2],
        "link_pos_diff_norm": float(np.linalg.norm(I_link[n] - U[n]["link_pos"])),
        "isaac_com_z": I_com[n][2], "urdf_merged_com_z": U[n]["merged_com"][2], "urdf_link_only_com_z": U[n]["link_only_com"][2],
        "merged_com_diff_norm": float(np.linalg.norm(I_com[n] - U[n]["merged_com"])),
        "isaac_com_minus_link_only_z": float(I_com[n][2] - U[n]["link_only_com"][2]),
        "isaac_mass": I_mass[n], "urdf_merged_mass": U[n]["merged_mass"], "urdf_link_only_mass": U[n]["link_only_mass"],
    })
with open(args.dir / "body_comparison.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

wb_i, wb_u = np.array(isaac["whole_body_com"]), np.array(urdf["whole_body_com"])
t = "torso_link"
feet_i = {s: I_link[f"{s}_ankle_roll_link"] + 0 for s in ("left", "right")}
summary = {
    "configuration": "Isaac reset state written exactly to both models (root pose + 29 joints by name); kinematic, no physics",
    "max_joint_write_error_isaac": isaac["max_joint_write_error"],
    "total_mass": {"isaac": float(sum(isaac["body_mass"])), "urdf": urdf["total_mass"]},
    "pelvis_z": {"isaac": float(I_link["pelvis"][2]), "urdf": float(U["pelvis"]["link_pos"][2])},
    "whole_body_com_z": {"isaac": float(wb_i[2]), "urdf": float(wb_u[2]), "diff": float(wb_i[2] - wb_u[2]),
                         "xyz_diff_norm": float(np.linalg.norm(wb_i - wb_u))},
    "torso_link": {
        "link_origin_z": {"isaac": float(I_link[t][2]), "urdf": float(U[t]["link_pos"][2])},
        "isaac_body_com_z": float(I_com[t][2]),
        "urdf_merged_body_com_z": float(U[t]["merged_com"][2]),
        "urdf_link_only_com_z (used by height calibration)": float(U[t]["link_only_com"][2]),
        "isaac_minus_link_only_z": float(I_com[t][2] - U[t]["link_only_com"][2]),
        "mass": {"isaac": I_mass[t], "urdf_merged": U[t]["merged_mass"], "urdf_link_only": U[t]["link_only_mass"]},
    },
    "max_link_pos_diff": max(r["link_pos_diff_norm"] for r in rows),
    "max_merged_com_diff": max(r["merged_com_diff_norm"] for r in rows),
    "bodies_where_isaac_com_differs_from_link_only_by_1mm": [r["body"] for r in rows if abs(r["isaac_com_minus_link_only_z"]) > 1e-3],
    "feet": {s: {"isaac_sole_point_z": urdf["feet"][s]["isaac_sole_point"][2], "mpc_contact_frame": urdf["feet"][s]["mpc_contact_frame"],
                 "isaac_sole_point": urdf["feet"][s]["isaac_sole_point"]} for s in ("left", "right")},
    "mpc_contact_offset_in_ankle_roll": urdf["mpc_contact_offset_in_ankle_roll"],
}
(args.dir / "height_comparison.json").write_text(json.dumps(summary, indent=2) + "\n")

# Skeleton edges from URDF moving joints.
edges = []
for j in ET.parse(args.urdf).getroot().findall("joint"):
    p, c = j.find("parent").get("link"), j.find("child").get("link")
    if p in I_link and c in I_link:
        edges.append((p, c))

fig = plt.figure(figsize=(16, 8.5), constrained_layout=True)
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25])
for k, (axis, label) in enumerate(((0, "X (forward) [m]"), (1, "Y (left) [m]"))):
    ax = fig.add_subplot(gs[k])
    for p, c in edges:
        ax.plot([I_link[p][axis], I_link[c][axis]], [I_link[p][2], I_link[c][2]], color="#1f77b4", lw=5, alpha=0.35, solid_capstyle="round")
        ax.plot([U[p]["link_pos"][axis], U[c]["link_pos"][axis]], [U[p]["link_pos"][2], U[c]["link_pos"][2]], color="#d62728", lw=1.4, ls="--")
    ax.plot([], [], color="#1f77b4", lw=5, alpha=0.35, label="Isaac links")
    ax.plot([], [], color="#d62728", lw=1.4, ls="--", label="MPC/URDF links (Pinocchio)")
    ax.scatter(I_com[t][axis], I_com[t][2], s=140, marker="o", facecolors="none", edgecolors="#1f77b4", lw=2,
               label=f"Isaac torso CoM z={I_com[t][2]:.4f}", zorder=5)
    ax.scatter(U[t]["merged_com"][axis], U[t]["merged_com"][2], s=40, marker="x", color="#d62728",
               label=f"URDF merged torso CoM z={U[t]['merged_com'][2]:.4f}", zorder=6)
    ax.scatter(U[t]["link_only_com"][axis], U[t]["link_only_com"][2], s=70, marker="^", color="#ff7f0e",
               label=f"URDF link-only torso CoM z={U[t]['link_only_com'][2]:.4f}", zorder=6)
    ax.scatter(wb_i[axis], wb_i[2], s=160, marker="o", facecolors="none", edgecolors="k", lw=2, label=f"Isaac whole CoM z={wb_i[2]:.4f}", zorder=5)
    ax.scatter(wb_u[axis], wb_u[2], s=50, marker="+", color="#2ca02c", lw=2, label=f"URDF whole CoM z={wb_u[2]:.4f}", zorder=6)
    for s in ("left", "right"):
        sole, contact = np.array(urdf["feet"][s]["isaac_sole_point"]), np.array(urdf["feet"][s]["mpc_contact_frame"])
        ax.scatter(sole[axis], sole[2], s=30, marker="s", color="#1f77b4", zorder=6)
        ax.scatter(contact[axis], contact[2], s=30, marker="D", color="#d62728", zorder=6)
    ax.scatter([], [], s=30, marker="s", color="#1f77b4", label="Isaac sole point")
    ax.scatter([], [], s=30, marker="D", color="#d62728", label="MPC contact frame")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_aspect("equal")
    ax.set_xlabel(label)
    ax.set_ylabel("Z [m]")
    ax.set_title(("Side" if axis == 0 else "Front") + " view, identical configuration")
    ax.grid(alpha=0.3)
    if k == 0:
        ax.legend(fontsize=7.5, loc="lower left")

ax = fig.add_subplot(gs[2])
y = np.arange(len(names))
ax.barh(y - 0.2, [1e3 * (I_com[n][2] - U[n]["merged_com"][2]) for n in names], height=0.4, color="#2ca02c",
        label="Isaac body CoM z − URDF merged body CoM z")
ax.barh(y + 0.2, [1e3 * (I_com[n][2] - U[n]["link_only_com"][2]) for n in names], height=0.4, color="#ff7f0e",
        label="Isaac body CoM z − URDF link-only CoM z")
ax.set_yticks(y, names, fontsize=7)
ax.invert_yaxis()
ax.set_xlabel("Height difference [mm]")
ax.axvline(0, color="k", lw=0.8)
ax.grid(axis="x", alpha=0.3)
ax.legend(fontsize=8, loc="lower right")
ax.set_title(f"Per-body CoM height differences\nmax link-origin diff {summary['max_link_pos_diff']*1e3:.4f} mm, "
             f"whole CoM Δz {summary['whole_body_com_z']['diff']*1e3:.4f} mm")
fig.suptitle("G1 standing: Isaac vs MPC/URDF (Pinocchio) at Isaac reset configuration", fontsize=13)
fig.savefig(args.dir / "isaac_vs_mpc_standing.png", dpi=150)
print(json.dumps(summary, indent=2))
