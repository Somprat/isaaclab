"""Render saved MPC poses on the repository's G1 model (kinematic replay)."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
import imageio.v2 as imageio
import mujoco
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('motion', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.json').exists():
        raise FileExistsError(f'Refusing to overwrite {args.output} or its report')
    model_path = Path(__file__).resolve().parents[1] / 'robot_models/unitree_g1/g1_description/urdf/g1_29dof.xml'
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    with np.load(args.motion, allow_pickle=False) as clip:
        decode = lambda names: [n.decode() if isinstance(n, bytes) else str(n) for n in names]
        names = decode(clip['joint_names'])
        bodies = decode(clip['body_names'])
        root = bodies.index('pelvis')
        positions = clip['body_pos_w'][:, root].copy()
        quats = clip['body_quat_w'][:, root].copy()  # NPZ and MuJoCo both use wxyz.
        joints = clip['joint_pos'].copy()
        fps = float(np.asarray(clip['fps']).reshape(-1)[0])
    addresses = []
    for name in names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f'Model has no joint {name}')
        addresses.append(model.jnt_qposadr[jid])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    camera = mujoco.MjvCamera()
    camera.distance = 3.0
    camera.azimuth = 135
    camera.elevation = -15
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        with imageio.get_writer(str(args.output), fps=fps) as writer:
            for i in range(len(joints)):
                data.qpos[:3] = positions[i]
                data.qpos[3:7] = quats[i] / np.linalg.norm(quats[i])
                data.qpos[addresses] = joints[i]
                mujoco.mj_forward(model, data)  # Set saved poses; do not integrate physics.
                camera.lookat[:] = positions[i]
                camera.lookat[2] = 0.65
                renderer.update_scene(data, camera=camera)
                writer.append_data(renderer.render())
    quats /= np.linalg.norm(quats, axis=-1, keepdims=True)
    upright = 1 - 2 * (quats[:, 1] ** 2 + quats[:, 2] ** 2)
    report = {
        'mode': 'kinematic saved-pose replay; not an independent physics stability test',
        'motion': str(args.motion.resolve()), 'frames': len(joints), 'fps': fps,
        'min_pelvis_height_m': float(positions[:, 2].min()),
        'max_pelvis_tilt_deg': float(np.degrees(np.arccos(np.clip(upright, -1, 1))).max()),
        'saved_pose_fall_threshold_crossed': bool(np.any((positions[:, 2] < .35) | (upright < np.cos(1.0)))),
        'fall_threshold': 'pelvis below 0.35 m or tilt above 1 radian',
    }
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
