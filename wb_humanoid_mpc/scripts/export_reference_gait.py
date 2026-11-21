#!/usr/bin/env python3
"""Export gait measurements directly from an MPC NPZ; no policy or simulation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def write_csv(path, rows):
    if rows:
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('motion', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--settle-time', type=float, default=2.0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a fresh output directory.')
    with np.load(args.motion, allow_pickle=False) as source:
        d = {key: source[key].copy() for key in source.files}
    names = d['body_names'].astype(str).tolist()
    contacts = d['contact_names'].astype(str).tolist()
    root = names.index('pelvis')
    feet = [names.index(side + '_ankle_roll_link') for side in ('left', 'right')]
    ci = [contacts.index('foot_l_contact'), contacts.index('foot_r_contact')]
    time = d['sample_time'].astype(float)
    if not np.all(np.diff(time) > 0):
        raise ValueError('Sample times must increase.')
    q = d['body_quat_w'][:, root].astype(float)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T  # MPC stores wxyz.
    forward = np.column_stack((1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y)))
    lateral = np.column_stack((2*(x*y-w*z), 1-2*(x*x+z*z), 2*(y*z+w*x)))
    vel = d['body_lin_vel_w'][:, root].astype(float)
    pos = d['body_pos_w'][:, root]
    heading = np.unwrap(np.arctan2(forward[:, 1], forward[:, 0]))
    vx = (vel*forward).sum(axis=1)
    vy = (vel*lateral).sum(axis=1)
    upright = 1-2*(x*x+y*y)
    flags = d['contact_flags'][:, ci].astype(bool)
    events = {side: np.flatnonzero(~flags[:-1, j] & flags[1:, j])+1
              for j, side in enumerate(('left', 'right'))}
    trace, touchdowns, strides, steps, summaries = [], [], [], [], []
    for i, t in enumerate(time):
        row = dict(time_s=float(t), segment=int(d['condition_stage'][i])+1,
                   target_vx_m_s=float(d['conditioning'][i, 0]), actual_vx_m_s=float(vx[i]),
                   height_m=float(pos[i, 2]), upright=float(upright[i]))
        for j, side in enumerate(('left', 'right')):
            for axis, value in zip('xyz', d['body_pos_w'][i, feet[j]]):
                row[f'{side}_foot_{axis}_m'] = float(value)
            row[f'{side}_foot_contact_planned'] = int(flags[i, j])
        row.update(root_x_m=float(pos[i, 0]), root_y_m=float(pos[i, 1]),
                   heading_rad=float(heading[i]), actual_vy_body_m_s=float(vy[i]),
                   actual_vx_world_m_s=float(vel[i, 0]), actual_vy_world_m_s=float(vel[i, 1]),
                   is_transition=int(d['is_transition'][i]), command_time_s=float(d['command_time'][i]),
                   commanded_step_length_m=float(d['step_conditioning'][i, 0]),
                   commanded_step_duration_s=float(d['step_conditioning'][i, 1]))
        trace.append(row)
    for stage in np.unique(d['condition_stage']):
        stage_mask = d['condition_stage'] == stage
        start = d['command_time'][np.flatnonzero(stage_mask)[0]]
        mask = stage_mask & ~d['is_transition'].astype(bool) & (time >= start+args.settle_time)
        ids = np.flatnonzero(mask)
        if len(ids) < 2:
            continue
        segment_events, segment_strides, segment_steps = [], [], []
        counts = {}
        for j, side in enumerate(('left', 'right')):
            valid = [i for i in events[side] if mask[i]]
            counts[side] = len(valid)
            for i in valid:
                row = dict(segment=int(stage)+1, side=side, time_s=float(time[i]),
                           x_m=float(d['body_pos_w'][i, feet[j], 0]),
                           y_m=float(d['body_pos_w'][i, feet[j], 1]))
                touchdowns.append(row)
                segment_events.append((i, j, row))
            for first, last in zip(valid[:-1], valid[1:]):
                if not mask[first:last+1].all():
                    continue
                delta = d['body_pos_w'][last, feet[j], :2]-d['body_pos_w'][first, feet[j], :2]
                row = dict(segment=int(stage)+1, side=side, start_time_s=float(time[first]),
                           end_time_s=float(time[last]), stride_period_s=float(time[last]-time[first]),
                           stride_length_m=float(np.linalg.norm(delta)), stride_forward_m=float(delta[0]))
                strides.append(row)
                segment_strides.append(row)
        segment_events.sort()
        for (a, j, prev), (b, k, cur) in zip(segment_events[:-1], segment_events[1:]):
            if j == k or not mask[a:b+1].all():
                continue
            # Signed displacement along the fixed initial world heading, not the
            # Euclidean separation (which would include lateral foot spacing).
            delta = np.array([cur['x_m']-prev['x_m'], cur['y_m']-prev['y_m']])
            initial_heading = heading[0]
            row = dict(segment=int(stage)+1, side=cur['side'], start_time_s=float(time[a]),
                       end_time_s=float(time[b]), step_duration_s=float(time[b]-time[a]),
                       step_length_m=float(delta @ np.array([np.cos(initial_heading), np.sin(initial_heading)])))
            steps.append(row)
            segment_steps.append(row)
        lengths = np.array([s['stride_length_m'] for s in segment_strides])
        periods = np.array([s['stride_period_s'] for s in segment_strides])
        summaries.append(dict(
            segment=int(stage)+1, target_vx_m_s=float(d['conditioning'][ids, 0].mean()),
            settle_time_s=args.settle_time, measurement_duration_s=float(time[ids[-1]]-time[ids[0]]),
            mean_actual_vx_m_s=float(vx[ids].mean()),
            speed_mae_m_s=float(np.abs(vx[ids]-d['conditioning'][ids, 0]).mean()),
            speed_std_m_s=float(vx[ids].std()), left_touchdowns=counts['left'], right_touchdowns=counts['right'],
            complete_strides=len(periods),
            mean_stride_length_m=float(lengths.mean()) if len(lengths) else None,
            stride_length_std_m=float(lengths.std()) if len(lengths) else None,
            mean_stride_period_s=float(periods.mean()) if len(periods) else None,
            stride_period_cv=float(periods.std()/periods.mean()) if len(periods) else None,
            cadence_steps_min=float(120/periods.mean()) if len(periods) else None,
            commanded_step_length_m=float(d['step_conditioning'][ids, 0].mean()),
            commanded_step_duration_s=float(d['step_conditioning'][ids, 1].mean()),
            measured_step_length_m=float(np.mean([s['step_length_m'] for s in segment_steps])) if segment_steps else None,
            measured_step_duration_s=float(np.mean([s['step_duration_s'] for s in segment_steps])) if segment_steps else None,
            root_lateral_displacement_m=float(pos[ids[-1], 1]-pos[ids[0], 1]),
            heading_change_deg=float(np.degrees(heading[ids[-1]]-heading[ids[0]])),
        ))
    args.output.mkdir(parents=True)
    for name, rows in [('velocity_tracking', trace), ('gait_touchdowns', touchdowns),
                       ('gait_strides', strides), ('gait_steps', steps), ('gait_summary', summaries)]:
        write_csv(args.output/(name+'.csv'), rows)
    (args.output/'gait_summary.json').write_text(json.dumps(summaries, indent=2, allow_nan=False)+'\n')
    metadata = dict(source='MPC-generated trajectory; no policy inference or independent physics simulation',
                    motion=str(args.motion.resolve()), sha256=hashlib.sha256(args.motion.read_bytes()).hexdigest(),
                    frames=len(time), fps=float(d['fps'].item()),
                    touchdown_detection='Rising edges of planned contact_flags, quantized to the NPZ sampling rate',
                    foot_position='Recorded ankle body origin, as in the baseline CSV; not the sole contact frame',
                    min_pelvis_height_m=float(pos[:, 2].min()),
                    max_pelvis_tilt_deg=float(np.degrees(np.arccos(np.clip(upright, -1, 1))).max()))
    (args.output/'evaluation_run.json').write_text(json.dumps(metadata, indent=2)+'\n')
    report = ['# Generated MPC reference gait', '', metadata['source'], '',
              '| Stage | Command speed | Measured body-forward speed | Command step length | Measured step length | Command duration | Measured duration |',
              '|---:|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        values=[s[k] for k in ['target_vx_m_s','mean_actual_vx_m_s','commanded_step_length_m',
                              'measured_step_length_m','commanded_step_duration_s','measured_step_duration_s']]
        report.append('| '+str(s['segment'])+' | '+' | '.join('N/A' if v is None else f'{v:.3f}' for v in values)+' |')
    report += ['', 'Units: meters, seconds, m/s. Startup/transition frames and the first two seconds of each stage are excluded.',
               '', metadata['touchdown_detection']+'. '+metadata['foot_position']+'.',
               'Timing follows the prescribed MPC schedule; it is not independently verified contact timing in physics.',
               'Baseline uses measured physics contact forces, so these are descriptive comparisons, not equivalent stability tests.',
               f"Minimum pelvis height: {metadata['min_pelvis_height_m']:.3f} m; maximum tilt: {metadata['max_pelvis_tilt_deg']:.2f} degrees."]
    (args.output/'gait_report.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
