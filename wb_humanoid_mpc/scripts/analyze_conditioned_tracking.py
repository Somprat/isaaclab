#!/usr/bin/env python3
"""Measure achieved COM speed and same-foot stride in conditioned NPZ exports.

Run inside the MPC container after sourcing /bin/ros_setup.sh (NumPy and the
ROS Pinocchio Python bindings are required). This does not run MPC or simulation.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pinocchio as pin


def analyze(path, model, data):
    with np.load(path, allow_pickle=False) as d:
        fps = float(d['fps'][0])
        positions = d['body_pos_w'][:, 0]
        quaternions = d['body_quat_w'][:, 0]
        w, x, y, z = quaternions.T
        yaw = np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
        indices = [model.idx_qs[model.getJointId(str(name))] for name in d['joint_names'].astype(str)]
        com = []
        for frame in range(len(positions)):
            q = pin.neutral(model)
            q[:3] = positions[frame]
            q[3:7] = quaternions[frame, [1, 2, 3, 0]]
            q[indices] = d['joint_pos'][frame]
            com.append(pin.centerOfMass(model, data, q).copy())
        velocity = np.gradient(np.asarray(com), 1/fps, axis=0)
        forward = np.cos(yaw)*velocity[:, 0] + np.sin(yaw)*velocity[:, 1]
        names = list(d['body_names'].astype(str))
        flags = d['contact_flags'].astype(bool)
        stages = []
        for stage in np.unique(d['condition_stage']):
            ix = np.flatnonzero((d['condition_stage'] == stage) & (d['is_transition'] == 0))
            if not len(ix):
                continue
            tail = ix[len(ix)//2:]
            target_speed, target_stride = map(float, d['conditioning_target'][ix[0]])
            strides = []
            for contact, side in enumerate(('left', 'right')):
                foot = d['body_pos_w'][:, names.index(side + '_ankle_roll_link')]
                touchdown = np.flatnonzero(~flags[:-1, contact] & flags[1:, contact]) + 1
                for a, b in zip(touchdown[:-1], touchdown[1:]):
                    if (d['condition_stage'][a] == stage and d['condition_stage'][b] == stage
                            and not d['is_transition'][a:b+1].any()):
                        strides.append(float(np.dot(foot[b, :2] - foot[a, :2], [np.cos(yaw[a]), np.sin(yaw[a])])))
            stages.append({
                'stage': int(stage), 'target_speed_mps': target_speed, 'target_stride_m': target_stride,
                'plateau_mean_speed_mps': float(forward[ix].mean()),
                'late_plateau_mean_speed_mps': float(forward[tail].mean()),
                'late_plateau_speed_rmse_mps': float(np.sqrt(np.mean((forward[tail] - target_speed)**2))),
                'mean_measured_full_stride_m': float(np.mean(strides)) if strides else None,
                'measured_stride_count': len(strides),
            })
        finite = all(np.isfinite(d[k]).all() for k in d.files if d[k].dtype.kind in 'fiu')
        return {'file': path.name, 'frames': len(positions), 'fps': fps, 'all_finite': finite,
                'pelvis_height_range_m': [float(positions[:, 2].min()), float(positions[:, 2].max())],
                'stages': stages}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--json', type=Path, help='Optional destination for detailed metrics.')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    urdf = repo / 'robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf'
    model = pin.buildModelFromUrdf(str(urdf), pin.JointModelFreeFlyer())
    data = model.createData()
    files = sorted(args.directory.glob('*.npz'))
    if not files:
        parser.error('No NPZ files found.')
    reports = [analyze(path, model, data) for path in files]
    for report in reports:
        print(report['file'], flush=True)
        for stage in report['stages']:
            print(f"  target {stage['target_speed_mps']:.3f}: plateau {stage['plateau_mean_speed_mps']:.3f}, "
                  f"late {stage['late_plateau_mean_speed_mps']:.3f} m/s; "
                  f"stride {stage['mean_measured_full_stride_m']}", flush=True)
    if args.json:
        args.json.write_text(json.dumps(reports, indent=2) + '\n')


if __name__ == '__main__':
    main()
