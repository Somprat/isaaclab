import argparse
from pathlib import Path
import time
import numpy as np
import viser
from trimesh.scene import Scene
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf
import jax.numpy as jnp
from jaxlie import SO3

import pyroki as pk
from ik import solve_ik_with_base_multiple_targets_g1

# ── Shared config ─────────────────────────────────────────────────────────
BASE_HEIGHT    = 0.78
LEFT_FOOT_POS  = np.array([0.0,  0.1, 0.0])
RIGHT_FOOT_POS = np.array([0.0, -0.1, 0.0])
FOOT_WXYZ      = np.array([1.0, 0.0, 0.0, 0.0])
LEFT_EE        = "left_wrist_yaw_link"
RIGHT_EE       = "right_wrist_yaw_link"

# Boundary computation
N_DIRECTIONS = 500
MAX_REACH    = 3
N_STEPS      = 5
CYLINDER_SURFACE_SAMPLES = 48
CYLINDER_HEIGHT_SAMPLES  = 24
DEFAULT_BOUNDARY_DIR = Path(__file__).parent / "reach_boundaries"
# ─────────────────────────────────────────────────────────────────────────


# ── Collision visualization helpers (from test.py) ────────────────────────
def _get_link_frame_paths(scene: Scene, prefix: str) -> dict[str, str]:
    result: dict[str, str] = {}
    base = scene.graph.base_frame
    parents = scene.graph.transforms.parents
    result[base] = prefix
    for frame_name in parents.keys():
        frames, current = [], frame_name
        while current != base:
            frames.append(current)
            current = parents[current]
        frames.append(prefix)
        result[frame_name] = "/".join(frames[::-1])
    return result


def _create_collision_handles(server, urdf, robot_coll, root_node_name="/base"):
    handles = []
    link_paths = _get_link_frame_paths(urdf.scene, f"{root_node_name}/visual")
    for link_name, mesh in robot_coll.get_link_collision_meshes().items():
        if mesh.is_empty or link_name not in link_paths:
            continue
        h = server.scene.add_mesh_simple(
            f"{link_paths[link_name]}/coll_{link_name}",
            vertices=mesh.vertices.astype(np.float32),
            faces=mesh.faces,
            color=(0.0, 1.0, 0.0),
            opacity=0.4,
        )
        h.visible = False
        handles.append(h)
    return handles


# ── Boundary computation ──────────────────────────────────────────────────
def fibonacci_sphere(n: int) -> np.ndarray:
    golden = (1 + np.sqrt(5)) / 2
    i     = np.arange(n)
    phi   = np.arccos(1 - 2 * (i + 0.5) / n)
    theta = 2 * np.pi * i / golden
    return np.stack([np.sin(phi)*np.cos(theta),
                     np.sin(phi)*np.sin(theta),
                     np.cos(phi)], axis=1)


def ee_world_pos(robot, ee_link, cfg, base_pos, base_wxyz) -> np.ndarray:
    ee_idx = robot.links.names.index(ee_link)
    Ts     = robot.forward_kinematics(jnp.array(cfg))
    R      = np.array(SO3(jnp.array(base_wxyz)).as_matrix())
    return base_pos + R @ np.array(Ts[ee_idx, 4:])


def compute_boundary(
    robot,
    robot_coll,
    ee_link,
    directions,
    xy_max: float | None = None,
    xy_max_weight: float = 50.0,
) -> np.ndarray:
    cfg_rest  = np.array(robot.joint_var_cls(0).default_factory())
    init_pos  = np.array([0.0, 0.0, BASE_HEIGHT])
    init_wxyz = np.array([1.0, 0.0, 0.0, 0.0])

    # Center = robot base frame origin in world frame.
    center = np.array([0.0, 0.0, BASE_HEIGHT])
    print(f"  center = {center.round(3)}")

    boundary = []
    t0 = time.time()
    for i, d in enumerate(directions):
        base_pos, base_wxyz, cfg = init_pos.copy(), init_wxyz.copy(), cfg_rest.copy()
        for step in range(1, N_STEPS + 1):
            target = center + MAX_REACH * (step / N_STEPS) * d
            target[2] = max(target[2], 0.01)

            # Reset to neutral for every direction — no cross-direction warm-start.
            base_pos, base_wxyz, cfg = solve_ik_with_base_multiple_targets_g1(
                robot=robot,
                target_link_names=["left_ankle_roll_link", "right_ankle_roll_link", ee_link],
                target_positions=np.array([LEFT_FOOT_POS, RIGHT_FOOT_POS, target]),
                target_wxyzs=np.array([FOOT_WXYZ, FOOT_WXYZ, FOOT_WXYZ]),
                pos_weights=np.array([500.0, 500.0, 50.0]),
                ori_weights=np.array([0.0, 0.0, 0.0]),
                flat_foot_link_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                flat_foot_weight=50.0,
                robot_coll=robot_coll,
                prev_pos=base_pos,
                prev_wxyz=base_wxyz,
                prev_cfg=cfg,
                xy_max=xy_max,
                xy_max_weight=xy_max_weight,
            )
        boundary.append(ee_world_pos(robot, ee_link, cfg, base_pos, base_wxyz))

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate    = (i + 1) / elapsed
            eta     = (len(directions) - i - 1) / rate
            print(f"  [{i+1}/{len(directions)}]  {rate:.1f} dir/s, ETA {eta:.0f}s")

    return np.array(boundary)


def height_colors(pts, lo, hi) -> np.ndarray:
    t = (pts[:, 2] - pts[:, 2].min()) / (pts[:, 2].max() - pts[:, 2].min() + 1e-9)
    c = (1 - t[:, None]) * np.array(lo) + t[:, None] * np.array(hi)
    return (c * 255).clip(0, 255).astype(np.uint8)


def fit_cylindrical_boundary(boundary_world: np.ndarray, base_center_world: np.ndarray) -> dict[str, float]:
    rel = boundary_world - base_center_world[None, :]
    radial = np.linalg.norm(rel[:, :2], axis=1)
    return {
        "radius": float(radial.max()),
        "z_lower": float(rel[:, 2].min()),
        "z_upper": float(rel[:, 2].max()),
    }


def yaw_only_wxyz(wxyz: np.ndarray) -> np.ndarray:
    R = np.array(SO3(jnp.array(wxyz)).as_matrix())
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])


def save_cylindrical_boundary(path: Path, boundary_data: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        radius=np.array(boundary_data["radius"], dtype=np.float32),
        z_lower=np.array(boundary_data["z_lower"], dtype=np.float32),
        z_upper=np.array(boundary_data["z_upper"], dtype=np.float32),
    )


def load_cylindrical_boundary(path: Path) -> dict[str, float]:
    data = np.load(path)
    return {
        "radius": float(data["radius"]),
        "z_lower": float(data["z_lower"]),
        "z_upper": float(data["z_upper"]),
    }


def cylinder_surface_points_local(radius: float, z_lower: float, z_upper: float) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, CYLINDER_SURFACE_SAMPLES, endpoint=False)
    z_vals = np.linspace(z_lower, z_upper, CYLINDER_HEIGHT_SAMPLES)
    tt, zz = np.meshgrid(theta, z_vals, indexing="xy")
    side = np.stack([
        radius * np.cos(tt).reshape(-1),
        radius * np.sin(tt).reshape(-1),
        zz.reshape(-1),
    ], axis=1)

    cap_r = np.sqrt(np.linspace(0.0, 1.0, CYLINDER_HEIGHT_SAMPLES)) * radius
    cap_t, cap_rr = np.meshgrid(theta, cap_r, indexing="xy")
    top = np.stack([
        cap_rr.reshape(-1) * np.cos(cap_t).reshape(-1),
        cap_rr.reshape(-1) * np.sin(cap_t).reshape(-1),
        np.full(cap_rr.size, z_upper),
    ], axis=1)
    bottom = np.stack([
        cap_rr.reshape(-1) * np.cos(cap_t).reshape(-1),
        cap_rr.reshape(-1) * np.sin(cap_t).reshape(-1),
        np.full(cap_rr.size, z_lower),
    ], axis=1)
    return np.concatenate([side, top, bottom], axis=0)


def transform_points(points_local: np.ndarray, base_pos: np.ndarray, base_wxyz: np.ndarray) -> np.ndarray:
    R = np.array(SO3(jnp.array(base_wxyz)).as_matrix())
    return points_local @ R.T + base_pos[None, :]


# ── Main ──────────────────────────────────────────────────────────────────
def main(xy_max: float | None, xy_max_weight: float, boundary_dir: Path):
    urdf       = load_robot_description("g1_description")
    robot      = pk.Robot.from_urdf(urdf)
    robot_coll = pk.collision.RobotCollision.from_urdf(urdf)
    directions = fibonacci_sphere(N_DIRECTIONS)
    base_center_world = np.array([0.0, 0.0, BASE_HEIGHT])

    print("=== Computing left arm boundary ===")
    left_boundary = compute_boundary(
        robot,
        robot_coll,
        LEFT_EE,
        directions,
        xy_max=xy_max,
        xy_max_weight=xy_max_weight,
    )
    left_boundary_fit = fit_cylindrical_boundary(left_boundary, base_center_world)
    left_boundary_path = boundary_dir / "left_boundary.npz"
    save_cylindrical_boundary(left_boundary_path, left_boundary_fit)
    left_boundary_fit = load_cylindrical_boundary(left_boundary_path)
    print(f"  saved left cylindrical boundary to {left_boundary_path}")
    print(f"Left Boundary: {left_boundary_fit}")

    print("\n=== Computing right arm boundary ===")
    right_boundary = compute_boundary(
        robot,
        robot_coll,
        RIGHT_EE,
        directions,
        xy_max=xy_max,
        xy_max_weight=xy_max_weight,
    )
    right_boundary_fit = fit_cylindrical_boundary(right_boundary, base_center_world)
    right_boundary_path = boundary_dir / "right_boundary.npz"
    save_cylindrical_boundary(right_boundary_path, right_boundary_fit)
    right_boundary_fit = load_cylindrical_boundary(right_boundary_path)
    print(f"  saved right cylindrical boundary to {right_boundary_path}")
    print(f"Right Boundary: {right_boundary_fit}")

    # ── Viser setup ───────────────────────────────────────────────────────
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=4, height=4)

    base_frame = server.scene.add_frame("/base", show_axes=False)
    base_frame.position = (0.0, 0.0, BASE_HEIGHT)
    base_frame.wxyz = (1.0, 0.0, 0.0, 0.0)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")

    # Boundary point clouds
    left_cloud  = server.scene.add_point_cloud("/boundary/left",  points=left_boundary,
        colors=height_colors(left_boundary,  (0.1, 0.4, 1.0), (0.6, 0.9, 1.0)), point_size=0.015)
    right_cloud = server.scene.add_point_cloud("/boundary/right", points=right_boundary,
        colors=height_colors(right_boundary, (1.0, 0.3, 0.1), (1.0, 0.8, 0.4)), point_size=0.015)
    left_cylinder_local = cylinder_surface_points_local(**left_boundary_fit)
    right_cylinder_local = cylinder_surface_points_local(**right_boundary_fit)
    left_cylinder = server.scene.add_point_cloud(
        "/boundary_fit/left",
        points=transform_points(left_cylinder_local, base_center_world, np.array([1.0, 0.0, 0.0, 0.0])),
        colors=np.tile(np.array([[120, 210, 255]], dtype=np.uint8), (len(left_cylinder_local), 1)),
        point_size=0.01,
    )
    right_cylinder = server.scene.add_point_cloud(
        "/boundary_fit/right",
        points=transform_points(right_cylinder_local, base_center_world, np.array([1.0, 0.0, 0.0, 0.0])),
        colors=np.tile(np.array([[255, 190, 120]], dtype=np.uint8), (len(right_cylinder_local), 1)),
        point_size=0.01,
    )

    # Interactive hand targets — initialize at rest-pose EE positions
    cfg_rest = np.array(robot.joint_var_cls(0).default_factory())
    Ts_rest  = robot.forward_kinematics(jnp.array(cfg_rest))
    l_ee_idx = robot.links.names.index(LEFT_EE)
    r_ee_idx = robot.links.names.index(RIGHT_EE)
    l_rest   = tuple(np.array(Ts_rest[l_ee_idx, 4:]) + np.array([0.0, 0.0, BASE_HEIGHT]))
    r_rest   = tuple(np.array(Ts_rest[r_ee_idx, 4:]) + np.array([0.0, 0.0, BASE_HEIGHT]))

    left_hand_target  = server.scene.add_transform_controls("/left_hand",  scale=0.2,
                                                             position=l_rest, wxyz=(1,0,0,0))
    right_hand_target = server.scene.add_transform_controls("/right_hand", scale=0.2,
                                                             position=r_rest, wxyz=(1,0,0,0))

    # GUI
    timing_handle     = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)
    show_left_bnd     = server.gui.add_checkbox("Show left boundary",  initial_value=True)
    show_right_bnd    = server.gui.add_checkbox("Show right boundary", initial_value=True)
    show_fit_handle   = server.gui.add_checkbox("Show fitted cylinder", initial_value=True)
    show_coll_handle  = server.gui.add_checkbox("Show collision bodies", initial_value=False)
    coll_handles      = _create_collision_handles(server, urdf, robot_coll)

    # IK state
    prev_pos  = np.array([0.0, 0.0, BASE_HEIGHT])
    prev_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
    prev_cfg  = cfg_rest.copy()

    # ── Interactive loop ──────────────────────────────────────────────────
    while True:
        t0 = time.time()
        prev_pos, prev_wxyz, prev_cfg = solve_ik_with_base_multiple_targets_g1(
            robot=robot,
            target_link_names=[
                "left_ankle_roll_link", "right_ankle_roll_link",
                LEFT_EE, RIGHT_EE,
            ],
            target_positions=np.array([
                LEFT_FOOT_POS, RIGHT_FOOT_POS,
                left_hand_target.position, right_hand_target.position,
            ]),
            target_wxyzs=np.array([
                FOOT_WXYZ, FOOT_WXYZ,
                left_hand_target.wxyz, right_hand_target.wxyz,
            ]),
            pos_weights=np.array([500.0, 500.0, 5.0, 5.0]),
            ori_weights=np.array([0.0, 0.0, 0.0, 0.0]),
            flat_foot_link_names=["left_ankle_roll_link", "right_ankle_roll_link"],
            flat_foot_weight=50.0,
            robot_coll=robot_coll,
            prev_pos=prev_pos,
            prev_wxyz=prev_wxyz,
            prev_cfg=prev_cfg,
            xy_max=xy_max,
            xy_max_weight=xy_max_weight,
        )

        urdf_vis.update_cfg(prev_cfg)
        base_frame.position = prev_pos
        base_frame.wxyz     = prev_wxyz
        yaw_wxyz = yaw_only_wxyz(prev_wxyz)
        left_cylinder.points = transform_points(left_cylinder_local, prev_pos, yaw_wxyz)
        right_cylinder.points = transform_points(right_cylinder_local, prev_pos, yaw_wxyz)

        timing_handle.value  = 0.99 * timing_handle.value + 0.01 * (time.time() - t0) * 1000
        left_cloud.visible   = show_left_bnd.value
        right_cloud.visible  = show_right_bnd.value
        left_cylinder.visible = show_fit_handle.value
        right_cylinder.visible = show_fit_handle.value
        for h in coll_handles:
            h.visible = show_coll_handle.value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="No XY limit — full reachable boundary in all directions.")
    parser.add_argument("--xy-max", type=float, default=0.45,
                        help="Optional world-frame XY radius limit in metres. Ignored with --full.")
    parser.add_argument("--xy-max-weight", type=float, default=500.0,
                        help="Penalty weight for the optional XY radius limit.")
    parser.add_argument("--boundary-dir", type=Path, default=DEFAULT_BOUNDARY_DIR,
                        help="Directory where fitted base-local boundaries are saved.")
    args = parser.parse_args()
    xy_max = None if args.full else args.xy_max
    main(xy_max, args.xy_max_weight, args.boundary_dir)
