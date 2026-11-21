"""Humanoid IK with fixed feet and floating base."""

import time
import viser
import yourdfpy
from robot_descriptions.loaders.yourdfpy import load_robot_description
import numpy as np
from trimesh.scene import Scene

import pyroki as pk
from pyroki.collision import RobotCollision
from viser.extras import ViserUrdf
from ik import solve_ik_with_base_multiple_targets_g1

BASE_HEIGHT = 0.78
FOOT_POSITIONS = 0.2
ARM_LENGTH = 0.45


def _get_link_frame_paths(scene: Scene, prefix: str) -> dict[str, str]:
    result: dict[str, str] = {}
    base = scene.graph.base_frame
    parents = scene.graph.transforms.parents
    result[base] = prefix
    for frame_name in parents.keys():
        frames = []
        current = frame_name
        while current != base:
            frames.append(current)
            current = parents[current]
        frames.append(prefix)
        result[frame_name] = "/".join(frames[::-1])
    return result


def _create_collision_handles(
    server: viser.ViserServer,
    urdf: yourdfpy.URDF,
    robot_coll: RobotCollision,
    root_node_name: str = "/base",
) -> list[viser.SceneNodeHandle]:
    handles: list[viser.SceneNodeHandle] = []
    link_paths = _get_link_frame_paths(urdf.scene, f"{root_node_name}/visual")
    for link_name, mesh in robot_coll.get_link_collision_meshes().items():
        if mesh.is_empty or link_name not in link_paths:
            continue
        handle = server.scene.add_mesh_simple(
            f"{link_paths[link_name]}/coll_{link_name}",
            vertices=mesh.vertices.astype(np.float32),
            faces=mesh.faces,
            color=(0.0, 1.0, 0.0),
            opacity=0.4,
        )
        handle.visible = False
        handles.append(handle)
    return handles


def main():
    urdf = load_robot_description("g1_description")

    robot = pk.Robot.from_urdf(urdf)
    robot_coll = pk.collision.RobotCollision.from_urdf(urdf)

    server = viser.ViserServer()
    base_frame = server.scene.add_frame("/base", show_axes=False)
    server.scene.add_grid("/ground", width=2, height=2)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")

    init_base_pos = np.array([0.0, 0.0, 0.78])
    init_base_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
    base_frame.position = init_base_pos
    base_frame.wxyz = init_base_wxyz

    # Fixed feet positions in world frame (on the ground).
    left_foot_pos = np.array([0.0, 0.1, 0.0])
    right_foot_pos = np.array([0.0, -0.1, 0.0])
    foot_wxyz = np.array([1.0, 0.0, 0.0, 0.0])

    # Interactive hand targets.
    left_hand_target = server.scene.add_transform_controls(
        "/left_hand", scale=0.2, position=(0.0, 0.1, 0.7), wxyz=(1, 0, 0, 0)
    )
    right_hand_target = server.scene.add_transform_controls(
        "/right_hand", scale=0.2, position=(0.0, -0.1, 0.7), wxyz=(1, 0, 0, 0)
    )
    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)
    show_coll_handle = server.gui.add_checkbox("Show Collision Bodies", initial_value=False)

    coll_handles = _create_collision_handles(server, urdf, robot_coll, root_node_name="/base")

    prev_pos = init_base_pos.copy()
    prev_wxyz = init_base_wxyz.copy()
    prev_cfg = np.array(robot.joint_var_cls(0).default_factory())

    while True:
        start_time = time.time()
        base_pos, base_wxyz, prev_cfg = solve_ik_with_base_multiple_targets_g1(
            robot=robot,
            target_link_names=[
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ],
            target_positions=np.array([
                left_foot_pos,
                right_foot_pos,
                left_hand_target.position,
                right_hand_target.position,
            ]),
            target_wxyzs=np.array([
                foot_wxyz,
                foot_wxyz,
                left_hand_target.wxyz,
                right_hand_target.wxyz,
            ]),
            pos_weights=np.array([500.0, 500.0, 5.0, 5.0]),
            ori_weights=np.array([0.0, 0.0, 0.0, 0.0]),
            flat_foot_link_names=["left_ankle_roll_link", "right_ankle_roll_link"],
            flat_foot_weight=50.0,
            robot_coll=robot_coll,
            prev_pos=prev_pos,
            prev_wxyz=prev_wxyz,
            prev_cfg=prev_cfg,
        )
        prev_pos = base_pos
        prev_wxyz = base_wxyz

        elapsed_time = time.time() - start_time
        timing_handle.value = 0.99 * timing_handle.value + 0.01 * (elapsed_time * 1000)

        urdf_vis.update_cfg(prev_cfg)
        base_frame.position = base_pos
        base_frame.wxyz = base_wxyz

        for h in coll_handles:
            h.visible = show_coll_handle.value


if __name__ == "__main__":
    main()
