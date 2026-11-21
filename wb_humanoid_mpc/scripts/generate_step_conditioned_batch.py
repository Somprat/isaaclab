#!/usr/bin/env python3
"""Generate exact-command, no-ramp MPC clips inside the prepared ROS container.

Run from the container repository after sourcing ROS and the workspace setup.
This intentionally preserves identical commands; different seeds need not yield
unique neutral-body trajectories. It does not train or evaluate an RL policy.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import subprocess
import time

PAIRS = ('0.10,0.8', '0.20,0.8', '0.30,0.8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-motions', type=int, default=50)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--output-dir', type=Path, default=Path('generated_motions/step_no_ramp_50_exact'))
    args = parser.parse_args()
    if args.num_motions < 1 or not 1 <= args.workers <= 4:
        parser.error('Use positive motion counts and 1–4 workers.')
    if not 0 <= args.seed <= 2**32 - args.num_motions:
        parser.error('The complete seed range must fit uint32.')
    repo = Path(__file__).resolve().parents[1]
    output = (repo / args.output_dir).resolve()
    if not output.is_relative_to(repo):
        parser.error('Output must be inside the mounted repository.')
    if output.exists():
        parser.error('Choose a fresh output directory; existing data will not be overwritten.')
    output.mkdir(parents=True)
    (output / 'logs').mkdir()
    task = output / 'task_snapshot.info'
    reference = output / 'reference_snapshot.info'
    shutil.copyfile(repo / 'robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task_random_reference.info', task)
    shutil.copyfile(repo / 'robot_models/unitree_g1/g1_centroidal_mpc/config/command/reference_random_reference.info', reference)
    manifest = {'status': 'running', 'num_motions': args.num_motions, 'duration_s': 20,
                'fps': 30, 'step_conditions_length_m_duration_s': [[.1,.8],[.2,.8],[.3,.8]],
                'requested_stage_boundaries_s': [20/3,40/3], 'settling_s': .8,
                'applied_stage_changes_s': [7.2,13.6], 'no_step_transitions': True,
                'upper_body_mode': 'neutral', 'workers': args.workers,
                'duplicate_trajectories_explicitly_accepted': True, 'clips': []}
    manifest_path = output / 'generation_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    start = time.monotonic()

    def generate(index):
        seed = args.seed + index
        filename = f'reference_{index:04d}.npz'
        command = ['ros2', 'run', 'humanoid_centroidal_mpc_ros2',
                   'humanoid_centroidal_mpc_random_reference_generator',
                   '--task-file', str(task), '--reference-file', str(reference),
                   '--duration', '20', '--fps', '30', '--no-step-transitions',
                   '--upper-body-mode', 'neutral', '--num-motions', '1',
                   '--max-attempts-per-motion', '1', '--seed', str(seed),
                   '--output', str(output / filename)]
        for pair in PAIRS:
            command += ['--step-condition', pair]
        with (output / 'logs' / f'reference_{index:04d}.log').open('w') as log:
            result = subprocess.run(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT)
        return {'index': index, 'seed': seed, 'file': filename, 'exit_code': result.returncode,
                'command': command, 'file_exists': (output / filename).is_file()}

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(generate, i) for i in range(args.num_motions)]
        for future in as_completed(futures):
            clip = future.result()
            manifest['clips'].append(clip)
            manifest['clips'].sort(key=lambda row: row['index'])
            manifest['elapsed_s'] = time.monotonic() - start
            manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
            good = sum(c['exit_code'] == 0 and c['file_exists'] for c in manifest['clips'])
            print(f"Completed {len(manifest['clips'])}/{args.num_motions}; successful {good}; elapsed {manifest['elapsed_s']:.0f}s", flush=True)
    manifest['status'] = ('generated' if all(c['exit_code'] == 0 and c['file_exists'] for c in manifest['clips']) else 'failed')
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    if manifest['status'] == 'failed':
        raise SystemExit('Some clips failed. Inspect generation_manifest.json and per-clip logs.')
    print(f'Generated {args.num_motions} clips in {output}', flush=True)


if __name__ == '__main__':
    main()
