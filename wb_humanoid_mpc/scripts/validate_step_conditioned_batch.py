#!/usr/bin/env python3
"""Validate exact-command batch files and summarize duplicates and achieved gait.

Run inside the prepared container after generate_step_conditioned_batch.py.
This checks exported references; it is not an independent physics evaluation.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    manifest_path = directory / 'generation_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['status'] != 'generated':
        parser.error('Generation has not completed successfully.')
    expected = manifest['num_motions']
    files = sorted(directory.glob('reference_*.npz'))
    if len(files) != expected:
        parser.error(f'Expected {expected} completed NPZ clips, found {len(files)}.')
    exporter = Path(__file__).with_name('export_reference_gait.py')
    qc = directory / 'quality_checks'
    qc.mkdir(exist_ok=True)
    results = []
    exact_groups = defaultdict(list)
    near_groups = []
    representatives = []
    for path in files:
        errors = []
        with np.load(path, allow_pickle=False) as d:
            arrays = {k: d[k] for k in d.files}
        for key, value in arrays.items():
            if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
                errors.append(f'Non-finite {key}')
        if len(arrays['joint_pos']) != 601 or not np.isclose(float(arrays['fps'].reshape(-1)[0]), 30):
            errors.append('Unexpected frame count or FPS')
        if not np.allclose(np.diff(arrays['sample_time']), 1/30, atol=1e-8):
            errors.append('Unexpected sample times')
        applied = np.unique(np.round(arrays['step_conditioning'], 5), axis=0)
        target = np.array([[0,.8],[.1,.8],[.2,.8],[.3,.8]])
        if applied.shape != target.shape or not np.allclose(applied, target):
            errors.append('Applied step commands differ from requested batch')
        planned = arrays['planned_steps']
        changes = planned[np.r_[True, abs(np.diff(planned[:,4]))>1e-8]][:,[0,4,5]]
        expected_changes = np.array([[0,0,.8],[.8,.1,.8],[7.2,.2,.8],[13.6,.3,.8]])
        if changes.shape != expected_changes.shape or not np.allclose(changes, expected_changes):
            errors.append('Unexpected planned command changes')
        pose = np.concatenate([arrays[k].reshape(-1) for k in ('joint_pos','body_pos_w','body_quat_w')])
        digest = hashlib.sha256(pose.tobytes()).hexdigest()
        exact_groups[digest].append(path.name)
        group = next((i for i, r in enumerate(representatives) if np.max(abs(pose-r)) <= 1e-5), None)
        if group is None:
            representatives.append(pose)
            near_groups.append([path.name])
        else:
            near_groups[group].append(path.name)
        names = arrays['body_names'].astype(str).tolist()
        root = names.index('pelvis')
        height = arrays['body_pos_w'][:,root,2]
        quat = arrays['body_quat_w'][:,root].astype(float)
        quat /= np.linalg.norm(quat,axis=1,keepdims=True)
        tilt = np.degrees(np.arccos(np.clip(1-2*(quat[:,1]**2+quat[:,2]**2),-1,1)))
        if height.min() < .69 or tilt.max() > 30:
            errors.append('Posture outside batch screening limits (0.69 m / 30 degrees)')
        gait_dir = qc / path.stem
        result = subprocess.run([sys.executable,str(exporter),str(path),'--output',str(gait_dir)],
                                stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True)
        gait = []
        if result.returncode:
            errors.append('Gait export failed: '+result.stderr[-500:])
        else:
            gait = json.loads((gait_dir/'gait_summary.json').read_text())
            if len(gait) != 3:
                errors.append('Expected three measured gait stages')
            for stage, length in zip(gait,[.1,.2,.3]):
                if abs(stage['measured_step_length_m']-length) > .02:
                    errors.append('Measured mean step-length error exceeds 2 cm')
                if abs(stage['measured_step_duration_s']-.8) > 1/30+1e-6:
                    errors.append('Measured mean duration error exceeds one video frame')
        results.append({'file':path.name,'passed':not errors,'errors':errors,'pose_sha256':digest,
                        'frames':len(height),'min_pelvis_height_m':float(height.min()),
                        'max_pelvis_tilt_deg':float(tilt.max()),'gait':gait})
    report = {'clips':len(results),'passed':sum(r['passed'] for r in results),
              'failed':sum(not r['passed'] for r in results),'nominal_total_seconds':20*len(results),
              'total_frames':sum(r['frames'] for r in results),'exact_unique_pose_trajectories':len(exact_groups),
              'near_unique_pose_trajectories_at_1e_minus5':len(near_groups),
              'exact_duplicate_groups':[v for v in exact_groups.values() if len(v)>1],
              'near_duplicate_groups':near_groups,'minimum_pelvis_height_m':min(r['min_pelvis_height_m'] for r in results),
              'maximum_pelvis_tilt_deg':max(r['max_pelvis_tilt_deg'] for r in results),
              'validation_scope':'Export integrity, command schedule, measured gait and posture screening; no independent physics stability certification.',
              'results':results}
    (directory/'validation_report.json').write_text(json.dumps(report,indent=2)+'\n')
    manifest['status'] = 'validated' if report['failed']==0 else 'validation_failed'
    manifest['validation_report'] = 'validation_report.json'
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('results','exact_duplicate_groups','near_duplicate_groups')},indent=2))
    if report['failed']:
        raise SystemExit('Some clips failed validation; inspect validation_report.json.')


if __name__ == '__main__':
    main()
