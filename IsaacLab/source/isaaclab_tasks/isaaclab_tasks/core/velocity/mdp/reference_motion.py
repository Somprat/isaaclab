# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Device-resident, batched sampling of conditioned MPC motion clips."""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def contact_phase(flags: np.ndarray) -> np.ndarray:
    """Unwrapped cycle phase, anchored at consecutive left-only support onsets.

    Phase zero is the start of left-only support, not a clock reset at a speed
    transition. Linear interpolation follows recorded gait cycles. The partial
    first/last cycles are extrapolated with the nearest observed cycle period.
    """
    left_only = (flags[:, 0] != 0) & (flags[:, 1] == 0)
    starts = np.flatnonzero(left_only & ~np.r_[False, left_only[:-1]])
    if len(starts) < 2:
        raise ValueError("A phase-conditioned clip needs at least two left-only support onsets.")
    frames = np.arange(len(flags))
    cycles = np.interp(frames, starts, np.arange(len(starts), dtype=float))
    before, after = frames < starts[0], frames > starts[-1]
    cycles[before] = (frames[before] - starts[0]) / (starts[1] - starts[0])
    cycles[after] = len(starts) - 1 + (frames[after] - starts[-1]) / (starts[-1] - starts[-2])
    return (2 * np.pi * cycles).astype(np.float32)


class ReferenceMotionLibrary:
    """Concatenate clips once; sample all environments with device tensor indexing.

    Names define correspondence; file and articulation ordering may differ. Times
    are relative to each clip's first stored state. Conditioning stays associated
    with the state row produced by that command (the MPC exporter logs post-step
    states). All clips must share the same joint/body name sets and contain the
    conditioning metadata and two-foot contact flags.
    """

    def __init__(self, directory: str, joint_names: list[str], body_names: list[str], device: str):
        source = Path(directory).expanduser()
        files = [source] if source.is_file() and source.suffix == ".npz" else sorted(source.glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"No NPZ reference motions in {directory}")
        fields = {
            name: []
            for name in (
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
                "conditioning",
                "phase",
            )
        }
        offsets, lengths, fps_values = [], [], []
        offset = 0
        for path in files:
            with np.load(path, allow_pickle=False) as data:
                joints = data["joint_names"].astype(str).tolist()
                bodies = data["body_names"].astype(str).tolist()
                if len(joints) != len(joint_names) or set(joints) != set(joint_names):
                    raise ValueError(f"Reference/robot joint names differ: {path}")
                if not set(body_names).issubset(bodies):
                    raise ValueError(f"Missing reference bodies in {path}: {set(body_names) - set(bodies)}")
                ji = [joints.index(name) for name in joint_names]
                bi = [bodies.index(name) for name in body_names]
                n = len(data["joint_pos"])
                fps = float(data["fps"].item())
                if n < 2 or not np.isfinite(fps) or fps <= 0:
                    raise ValueError(f"Invalid reference frame count/fps: {path}")
                contacts = data["contact_names"].astype(str).tolist()
                flags = data["contact_flags"][:, [contacts.index("foot_l_contact"), contacts.index("foot_r_contact")]]
                values = {
                    "joint_pos": data["joint_pos"][:, ji],
                    "joint_vel": data["joint_vel"][:, ji],
                    "conditioning": data["conditioning"],
                    "phase": contact_phase(flags),
                }
                for name in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
                    values[name] = data[name][:, bi]
                # MPC files are wxyz; this checkout's asset/math APIs are xyzw.
                # Convert once on ingestion, not repeatedly at simulation writes.
                values["body_quat_w"] = values["body_quat_w"][..., [1, 2, 3, 0]]
                if values["conditioning"].shape != (n, 2):
                    raise ValueError(f"Expected [velocity, full_stride] conditioning in {path}")
                if any(len(value) != n or not np.isfinite(value).all() for value in values.values()):
                    raise ValueError(f"Invalid/nonfinite reference arrays: {path}")
                for name, value in values.items():
                    fields[name].append(value)
                offsets.append(offset)
                lengths.append(n)
                fps_values.append(fps)
                offset += n
        self.files = files
        self.data = {
            name: torch.tensor(np.concatenate(values), dtype=torch.float32, device=device)
            for name, values in fields.items()
        }
        self.offsets = torch.tensor(offsets, dtype=torch.long, device=device)
        self.lengths = torch.tensor(lengths, dtype=torch.long, device=device)
        self.fps = torch.tensor(fps_values, dtype=torch.float32, device=device)
        self.durations = (self.lengths - 1) / self.fps

    def sample(self, clip_ids: torch.Tensor, times: torch.Tensor) -> dict[str, torch.Tensor]:
        """Sample arbitrary control times with shortest-arc xyzw interpolation."""
        frame = torch.minimum(times.clamp_min(0) * self.fps[clip_ids], (self.lengths[clip_ids] - 1).float())
        local = frame.floor().long()
        start = self.offsets[clip_ids] + local
        end = self.offsets[clip_ids] + torch.minimum(local + 1, self.lengths[clip_ids] - 1)
        alpha = frame - local
        result = {}
        for name, values in self.data.items():
            a, b = values[start], values[end]
            blend = alpha.reshape((-1,) + (1,) * (a.ndim - 1))
            if name == "body_quat_w":
                a, b = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
                dot = (a * b).sum(-1, keepdim=True)
                b = torch.where(dot < 0, -b, b)
                dot = dot.abs().clamp(0, 1)
                theta = torch.acos(dot)
                spherical = (torch.sin((1 - blend) * theta) * a + torch.sin(blend * theta) * b) / torch.sin(
                    theta
                ).clamp_min(1e-8)
                result[name] = F.normalize(torch.where(dot > 0.9995, torch.lerp(a, b, blend), spherical), dim=-1)
            else:
                result[name] = torch.lerp(a, b, blend)
        return result
