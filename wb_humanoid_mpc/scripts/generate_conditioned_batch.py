#!/usr/bin/env python3
"""Launch conditioned MPC exports in the prepared Docker container.

Defaults: 50 motions, 20 seconds each, 30 Hz. Each motion allocates 80%
of command frames equally to the three velocity/full-stride pairs below,
with the remaining 20% used for startup and inter-stage transitions.
"""

import argparse
import math
from pathlib import Path
import shlex
import subprocess


CONTAINER_REPO = Path('/wb_humanoid_mpc_ws/src/wb_humanoid_mpc')
DEFAULT_PAIRS = ('0.2,0.20', '0.5,0.40', '1.0,0.60')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-motions', type=int, default=50)
    parser.add_argument('--duration', type=float, default=20.0, help='Seconds per motion (default: 20).')
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--max-attempts-per-motion', type=int, default=20)
    parser.add_argument('--container', default='wb-mpc-dev')
    parser.add_argument('--output-dir', type=Path, default=Path('generated_motions/conditioned_50'),
                        help='Output directory relative to the repository.')
    parser.add_argument('--dry-run', action='store_true', help='Print the command without running Docker or MPC.')
    args = parser.parse_args()
    if args.num_motions < 1 or args.max_attempts_per_motion < 1:
        parser.error('Motion and attempt counts must be positive.')
    if not all(math.isfinite(x) and x > 0 for x in (args.duration, args.fps)):
        parser.error('Duration and FPS must be finite and positive.')
    if not 0 <= args.seed <= 2**32 - 1:
        parser.error('Seed must fit an unsigned 32-bit integer.')
    if args.seed + args.num_motions * args.max_attempts_per_motion - 1 > 2**32 - 1:
        parser.error('The batch seed range would overflow an unsigned 32-bit integer.')
    repo = Path(__file__).resolve().parents[1]
    output_dir = (repo / args.output_dir).resolve()
    if not output_dir.is_relative_to(repo):
        parser.error('Output directory must be inside this repository, which is mounted in Docker.')
    output = output_dir / 'conditioned_mpc_reference.npz'
    expected = ([output] if args.num_motions == 1 else
                [output.with_name(f'{output.stem}_{i:04d}.npz') for i in range(args.num_motions)])
    if any(path.exists() for path in expected):
        parser.error('Output files already exist; choose a fresh --output-dir to avoid overwriting data.')
    generator_args = [
        'ros2', 'run', 'humanoid_centroidal_mpc_ros2',
        'humanoid_centroidal_mpc_random_reference_generator',
        '--num-motions', str(args.num_motions), '--duration', str(args.duration),
        '--fps', str(args.fps), '--seed', str(args.seed),
        '--max-attempts-per-motion', str(args.max_attempts_per_motion),
        '--plateau-ratio', '0.8',
        '--output', str(CONTAINER_REPO / output.relative_to(repo)),
    ]
    for pair in DEFAULT_PAIRS:
        generator_args.extend(['--condition', pair])
    # Pass CLI values as positional arguments, never interpolate them into shell code.
    command = [
        'docker', 'exec', args.container, 'bash', '-lc',
        'source /bin/ros_setup.sh && source /wb_humanoid_mpc_ws/install/setup.bash && exec "$@"',
        'conditioned-batch', *generator_args,
    ]
    print(f'{args.num_motions} motions, {args.duration:g} seconds each, {args.fps:g} Hz', flush=True)
    print('Pairs (m/s, full stride m): ' + '; '.join(DEFAULT_PAIRS), flush=True)
    print('80% command plateaus, divided equally; 20% startup/inter-stage transitions.', flush=True)
    print(f'Host output: {output_dir}', flush=True)
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
