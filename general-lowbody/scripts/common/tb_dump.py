#!/usr/bin/env python3
"""Dump tensorboard scalar logs from a single rsl_rl run to CSV.

Reads `events.out.tfevents.*` (the same data tensorboard shows) and writes
flat CSVs Claude can read directly. One run per invocation; output is written
inside the run directory.

Usage:
    python scripts/common/tb_dump.py logs/rsl_rl/<task>/<run>
    python scripts/common/tb_dump.py <run_dir> --out somewhere_else
    python scripts/common/tb_dump.py <run_dir> --max-points 0   # no downsampling

Output (default <run_dir>/tb_dump/):
    summary.csv        one row per tag: tag, n, last_step, last, max, min, mean, mean_tail10pct
                       (always uses FULL data, not downsampled)
    scalars.csv        long form: tag, step, wall_time, value
                       (downsampled to --max-points per curve; default 500)
"""

from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def find_event_files(run_dir: Path):
    return sorted(run_dir.glob("**/events.out.tfevents.*"))


def load_run(run_dir: Path):
    """Return {tag: [(step, wall_time, value), ...]} for all scalars."""
    ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    return {tag: [(e.step, e.wall_time, e.value) for e in ea.Scalars(tag)] for tag in tags}


def downsample(rows, max_points):
    if max_points <= 0 or len(rows) <= max_points:
        return rows
    stride = max(1, len(rows) // max_points)
    return rows[::stride]


def stats(values):
    n = len(values)
    if n == 0:
        return dict(n=0, last=None, max=None, min=None, mean=None, mean_tail=None)
    last = values[-1]
    mx, mn = max(values), min(values)
    mean = sum(values) / n
    tail_n = max(1, n // 10)
    mean_tail = sum(values[-tail_n:]) / tail_n
    return dict(n=n, last=last, max=mx, min=mn, mean=mean, mean_tail=mean_tail)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="Path to a single rsl_rl run directory")
    ap.add_argument("--out", default=None,
                    help="Output directory (default: <run_dir>/tb_dump)")
    ap.add_argument("--max-points", type=int, default=500,
                    help="Downsample each curve to ≤N points in scalars.csv "
                         "(0 = no downsample). summary.csv always uses full data. Default 500.")
    args = ap.parse_args()

    run_dir = Path(args.run).resolve()
    if not run_dir.is_dir():
        print(f"[error] not a directory: {run_dir}", file=sys.stderr)
        sys.exit(1)
    if not find_event_files(run_dir):
        print(f"[error] no tensorboard event files under {run_dir}", file=sys.stderr)
        sys.exit(1)

    out = Path(args.out) if args.out else run_dir / "tb_dump"
    out.mkdir(parents=True, exist_ok=True)

    print(f"[load] {run_dir}")
    data = load_run(run_dir)
    if not data:
        print(f"[error] no scalar tags found in {run_dir}", file=sys.stderr)
        sys.exit(1)

    with open(out / "scalars.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "step", "wall_time", "value"])
        for tag, rows in data.items():
            for step, wt, val in downsample(rows, args.max_points):
                w.writerow([tag, step, wt, val])

    summary_rows = []
    for tag, rows in data.items():
        values = [v for _, _, v in rows]
        steps = [s for s, _, _ in rows]
        s = stats(values)
        summary_rows.append({
            "tag": tag,
            "n": s["n"],
            "last_step": steps[-1] if steps else None,
            "last": s["last"],
            "max": s["max"],
            "min": s["min"],
            "mean": s["mean"],
            "mean_tail10pct": s["mean_tail"],
        })
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    print(f"\nDone. Output → {out}")
    print(f"  tags: {len(data)}    summary rows: {len(summary_rows)}")


if __name__ == "__main__":
    main()
