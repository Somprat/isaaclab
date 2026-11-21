# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Summarize footfall traces recorded by play_g1_29dof.py --evaluate-gait.

Run independently with ``uv run python scripts/reinforcement_learning/evaluate_g1_29dof_gait.py TRACE.csv``.
Contact detection uses 10 N on / 5 N off hysteresis and requires 60 ms of stable
contact state. Complete strides must start and end within a settled speed segment.
Stride length is the horizontal displacement between successive same-foot touchdowns,
not the instantaneous separation of the two feet. Step frequency counts both feet.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _touchdowns(times: np.ndarray, forces: np.ndarray) -> list[int]:
    """Find stance onsets with force hysteresis and a 60 ms debounce in both directions."""
    stance = False
    candidate = None
    events = []
    for index, force in enumerate(forces):
        transition = force <= 5.0 if stance else force >= 10.0
        if not transition:
            candidate = None
        elif candidate is None:
            candidate = index
        elif times[index] - times[candidate] >= 0.06 - 1e-9:
            stance = not stance
            if stance:
                events.append(candidate)
            candidate = None
    return events


def summarize_gait(path: Path, settle_time: float = 2.0) -> list[dict]:
    """Write summaries, complete strides, touchdown events, and a plot next to a recorded trace."""
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("The gait trace is empty.")
    data = {key: np.asarray([float(row[key]) for row in rows]) for key in rows[0]}
    times = data["time_s"]
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("Gait samples must have strictly increasing timestamps.")
    dt = float(np.median(np.diff(times)))
    episodes = data.get("episode", np.zeros(len(times)))
    events = {side: [] for side in ("left", "right")}
    for episode in np.unique(episodes):
        indices = np.flatnonzero(episodes == episode)
        for side in events:
            events[side].extend(
                indices[_touchdowns(times[indices], data[f"{side}_foot_normal_force_n"][indices])].tolist()
            )
    summaries, strides, touchdowns = [], [], []
    for segment in np.unique(data["segment"]):
        segment_ids = np.flatnonzero(data["segment"] == segment)
        start = times[segment_ids[0]] - dt
        mask = (data["segment"] == segment) & (times >= start + settle_time)
        if "is_transition" in data:
            mask &= data["is_transition"] == 0
        ids = np.flatnonzero(mask)
        if len(ids) < 2:
            continue  # A fall can end a run before a segment has settled.
        segment_strides = []
        foot_counts = {}
        for side, indices in events.items():
            valid = [index for index in indices if mask[index]]
            foot_counts[side] = len(valid)
            for index in valid:
                touchdowns.append(
                    {
                        "segment": int(segment),
                        "side": side,
                        "time_s": times[index],
                        "x_m": data[f"{side}_foot_x_m"][index],
                        "y_m": data[f"{side}_foot_y_m"][index],
                    }
                )
            for first, last in zip(valid[:-1], valid[1:]):
                if not mask[first : last + 1].all():
                    continue
                displacement = np.array(
                    [data[f"{side}_foot_{axis}_m"][last] - data[f"{side}_foot_{axis}_m"][first] for axis in ("x", "y")]
                )
                stride = {
                    "segment": int(segment),
                    "side": side,
                    "start_time_s": times[first],
                    "end_time_s": times[last],
                    "stride_period_s": times[last] - times[first],
                    "stride_length_m": float(np.linalg.norm(displacement)),
                    "stride_forward_m": float(displacement[0]),
                }
                strides.append(stride)
                segment_strides.append(stride)
        lengths = np.asarray([stride["stride_length_m"] for stride in segment_strides])
        periods = np.asarray([stride["stride_period_s"] for stride in segment_strides])
        actual = data["actual_vx_m_s"][mask]
        target = float(data["target_vx_m_s"][ids].mean())
        summary = {
            "segment": int(segment),
            "target_vx_m_s": target,
            "policy_input_vx_m_s": float(data["policy_input_vx_m_s"][ids[0]]),
            "settle_time_s": settle_time,
            "measurement_duration_s": times[ids[-1]] - times[ids[0]],
            "mean_actual_vx_m_s": float(actual.mean()),
            "speed_mae_m_s": float(np.abs(actual - data["target_vx_m_s"][ids]).mean()),
            "speed_std_m_s": float(actual.std()),
            "left_touchdowns": foot_counts["left"],
            "right_touchdowns": foot_counts["right"],
            "complete_strides": len(periods),
            "mean_stride_length_m": float(lengths.mean()) if len(lengths) else None,
            "stride_length_std_m": float(lengths.std()) if len(lengths) else None,
            "mean_stride_period_s": float(periods.mean()) if len(periods) else None,
            "stride_period_cv": float(periods.std() / periods.mean()) if len(periods) else None,
            "step_frequency_hz": float(2.0 / periods.mean()) if len(periods) else None,
            "cadence_steps_min": float(120.0 / periods.mean()) if len(periods) else None,
        }
        summary["root_forward_displacement_m"] = float(data["root_x_m"][ids[-1]] - data["root_x_m"][ids[0]])
        summary["root_lateral_displacement_m"] = float(data["root_y_m"][ids[-1]] - data["root_y_m"][ids[0]])
        if "heading_rad" in data:
            heading = np.unwrap(data["heading_rad"][ids])
            summary.update(
                heading_start_deg=float(np.degrees(heading[0])),
                heading_end_deg=float(np.degrees(heading[-1])),
                heading_change_deg=float(np.degrees(heading[-1] - heading[0])),
                mean_lateral_body_velocity_m_s=float(data["actual_vy_body_m_s"][ids].mean()),
                mean_yaw_rate_rad_s=float(data["yaw_rate_rad_s"][ids].mean()),
            )
        if "commanded_step_length_m" in data:
            summary["commanded_step_length_m"] = float(data["commanded_step_length_m"][ids].mean())
            summary["commanded_step_duration_s"] = float(data["commanded_step_duration_s"][ids].mean())
            summary["measured_step_length_m_approx"] = float(lengths.mean() / 2) if len(lengths) else None
            summary["measured_step_duration_s_approx"] = float(periods.mean() / 2) if len(periods) else None
        summaries.append(summary)

    for name, records in (("gait_summary", summaries), ("gait_strides", strides), ("gait_touchdowns", touchdowns)):
        if records:
            with (path.parent / f"{name}.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
    (path.parent / "gait_summary.json").write_text(json.dumps(summaries, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summaries, indent=2), flush=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    axes[0].plot(times, data["actual_vx_m_s"], label="Measured forward speed", linewidth=0.8)
    axes[0].plot(times, data["target_vx_m_s"], label="Target speed", linestyle="--")
    axes[0].set_ylabel("Speed (m/s)")
    for side in events:
        axes[1].plot(times, data[f"{side}_foot_z_m"], label=side, linewidth=0.8)
        axes[2].plot(times, data[f"{side}_foot_normal_force_n"], label=side, linewidth=0.8)
        for index in events[side]:
            axes[1].scatter(times[index], data[f"{side}_foot_z_m"][index], s=9)
    axes[1].set_ylabel("Foot origin height (m)")
    axes[2].set_ylabel("Foot normal force (N)")
    axes[2].set_xlabel("Time (s)")
    for axis in axes:
        axis.legend(loc="upper left")
        axis.grid(alpha=0.2)
    figure.suptitle("G1 29-DOF velocity and gait evaluation")
    figure.savefig(path.parent / "gait_trace.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    targets = [summary["target_vx_m_s"] for summary in summaries]
    speeds = [summary["mean_actual_vx_m_s"] for summary in summaries]
    axes[0].errorbar(targets, speeds, yerr=[summary["speed_std_m_s"] for summary in summaries], fmt="o-")
    axes[0].plot(targets, targets, linestyle="--", color="gray", label="Perfect tracking")
    axes[0].set(xlabel="Target speed (m/s)", ylabel="Actual speed (m/s)")
    axes[0].legend()
    valid = [summary for summary in summaries if summary["complete_strides"] > 0]
    axes[1].errorbar(
        [summary["mean_actual_vx_m_s"] for summary in valid],
        [summary["mean_stride_length_m"] for summary in valid],
        yerr=[summary["stride_length_std_m"] for summary in valid],
        fmt="o-",
    )
    axes[1].set(xlabel="Actual speed (m/s)", ylabel="Stride length (m)")
    axes[2].plot(
        [summary["mean_actual_vx_m_s"] for summary in valid],
        [summary["step_frequency_hz"] for summary in valid],
        "o-",
    )
    axes[2].set(xlabel="Actual speed (m/s)", ylabel="Step frequency (steps/s)")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle("G1 gait across speeds (error bars: sample standard deviation)")
    figure.savefig(path.parent / "gait_speed_summary.png", dpi=160)
    plt.close(figure)
    run_path = path.parent / "evaluation_run.json"
    run = json.loads(run_path.read_text()) if run_path.exists() else {}
    report = [
        "# G1 29-DOF gait evaluation",
        "",
        f"Backend: {run.get('physics_backend', 'unknown')}. Completed duration: {times[-1]:.2f} s.",
        f"Fall recorded: {run['fall'] if run.get('fall') else ('no' if 'fall' in run else 'unknown')}."
        " No training performed.",
        f"Minimum pelvis height: {data['height_m'].min():.3f} m. "
        f"Maximum tilt: {np.degrees(np.arccos(np.clip(data['upright'], -1, 1))).max():.2f} degrees.",
        f"Policy inputs (m/s): {run.get('policy_inputs_m_s', 'see trace')}. Seed: {run.get('seed', 'unknown')}.",
        "",
        "| Target m/s | Actual body-forward m/s | Full stride m | Step interval s (approx.) "
        "| Steps/min | Lateral displacement m |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        length = f"{s['mean_stride_length_m']:.3f}" if s["mean_stride_length_m"] is not None else "N/A"
        period = f"{s['mean_stride_period_s'] / 2:.3f}" if s["mean_stride_period_s"] is not None else "N/A"
        cadence = f"{s['cadence_steps_min']:.1f}" if s["cadence_steps_min"] is not None else "N/A"
        report.append(
            f"| {s['target_vx_m_s']:.1f} | {s['mean_actual_vx_m_s']:.3f} | {length} | {period} "
            f"| {cadence} | {s['root_lateral_displacement_m']:.3f} |"
        )
    report.extend(
        [
            "",
            f"The first {settle_time:g} seconds of each speed segment are excluded."
            " Stride means successive same-foot touchdowns"
            "; step length and duration are approximately half the stride values for symmetric alternating gait.",
            "",
            "Forward speed is measured in the body frame."
            " Lateral displacement is measured in world coordinates."
            " Heading/lateral velocity fields in gait_summary.csv help distinguish turning from sideways motion.",
            "",
            "Touchdowns use 10 N on / 5 N off force hysteresis and 60 ms debounce at 50 Hz."
            " Foot positions are ankle body origins."
            " This is one flat-ground run without perturbations;"
            " it does not establish robustness or contact slip.",
            "",
            "See gait_summary.csv, gait_strides.csv, gait_touchdowns.csv, velocity_tracking.csv,"
            " gait_trace.png, gait_speed_summary.png, and evaluation_run.json.",
        ]
    )
    if "commanded_step_length_m" in data:
        report.extend(
            [
                "",
                "## Reference command comparison",
                "",
                "Measured step values below are half the same-foot stride means, assuming alternating symmetric steps.",
                "Transitions are excluded; contact detection restarts after each episode reset.",
                "",
                "| Segment | Commanded step m | Measured step m (approx.) "
                "| Commanded duration s | Measured duration s (approx.) |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for s in summaries:
            length = s["measured_step_length_m_approx"]
            period = s["measured_step_duration_s_approx"]
            measured = f"{length:.3f}" if length is not None else "N/A"
            interval = f"{period:.3f}" if period is not None else "N/A"
            report.append(
                f"| {s['segment']} | {s['commanded_step_length_m']:.3f} | {measured} "
                f"| {s['commanded_step_duration_s']:.3f} | {interval} |"
            )
        report.append(
            "\nThese are policy-inference measurements, not kinematic replay. "
            "The policy was trained earlier and does not receive step duration directly."
        )
    (path.parent / "gait_report.md").write_text("\n".join(report) + "\n")
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--settle-time", type=float, default=2.0)
    args = parser.parse_args()
    if not np.isfinite(args.settle_time) or args.settle_time < 0:
        parser.error("--settle-time must be finite and nonnegative.")
    summarize_gait(args.trace, args.settle_time)
