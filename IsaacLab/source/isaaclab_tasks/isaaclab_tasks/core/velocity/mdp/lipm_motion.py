# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Device-resident LIPM references (schemas 1–3) with optional arm IK extension v1."""

from pathlib import Path

import numpy as np
import torch


def _read_names(data, key: str, file: Path) -> list[str]:
    """Read a nonempty, unique string array from an NPZ's name metadata."""
    value = np.asarray(data[key])
    if value.ndim != 1 or value.size == 0 or value.dtype.kind not in "US":
        raise ValueError(f"{file}: {key} must be a nonempty string array")
    names = value.astype(str).tolist()
    if any(not name.strip() for name in names) or len(set(names)) != len(names):
        raise ValueError(f"{file}: {key} must contain unique, nonempty names")
    return names


class LipmMotionLibrary:
    """Sample independent clips without interpolating across clip boundaries.

    Times are seconds relative to the first stored sample, which may be at dt.
    Phase is converted from wrapped cycle fractions to unwrapped radians.
    Support is a right-continuous sample-and-hold signal, never interpolated.

    The optional ``arm_reference_schema_version=1`` extension preserves the LIPM
    schema and adds arm joint positions/velocities and torso-frame hand targets.
    Supply ``joint_names=robot.joint_names`` to order the controlled joint subset
    in articulation order. ``arm_joint_ids`` then indexes that subset in the
    robot, while sampled ``arm_joint_pos/vel`` have only those joint columns.
    Without robot names, the first clip's joint order is used and IDs are None.
    ``hand_names`` is always ordered left, right (or the selected single hand).
    Metadata lives on the library, not in the sampled tensor dictionary.

    All clips must either omit arms or share the same arm joint set, hand set
    and hand frame definitions. Their stored column order may differ.
    """

    def __init__(self, path: str, device: str, joint_names: list[str] | None = None):
        if joint_names is not None and (
            not joint_names
            or any(not isinstance(name, str) or not name.strip() for name in joint_names)
            or len(set(joint_names)) != len(joint_names)
        ):
            raise ValueError("Robot joint_names must contain unique, nonempty names")
        source = Path(path).expanduser()
        self.files = [source] if source.is_file() else sorted(source.glob("*.npz"))
        if not self.files:
            raise ValueError(f"No LIPM NPZ clips at {source}")
        self.clips = []
        self.has_arm_reference = False
        self.arm_joint_names: list[str] = []
        self.arm_joint_ids: torch.Tensor | None = None
        self.hand_names: list[str] = []
        self.hand_frame_names: list[str] = []
        self.hand_target_frame: str | None = None
        durations = []
        for file in self.files:
            with np.load(file, allow_pickle=False) as data:
                version = data["schema_version"].item()
                if version not in (1, 2, 3) or data["frame"].item() != "world_z_up":
                    raise ValueError(f"{file}: expected LIPM schema 1, 2 or 3, world_z_up")
                if data["foot_names"].tolist() != ["left", "right"]:
                    raise ValueError(f"{file}: foot order must be left, right")
                time = np.asarray(data["time"], dtype=np.float64)
                if time.ndim != 1 or len(time) < 2 or not np.isfinite(time).all() or not (np.diff(time) > 0).all():
                    raise ValueError(f"{file}: time must be finite and strictly increasing")
                n = len(time)
                shapes = {
                    "com_pos": (n, 3),
                    "com_vel": (n, 3),
                    "foot_pos": (n, 2, 3),
                    "support_mask": (n, 2),
                    "phase": (n,),
                    "desired_step_length": (n,),
                    "desired_step_width": (n,),
                }
                if version >= 2:
                    shapes["foot_vel"] = (n, 2, 3)
                if version == 3:
                    shapes["step_duration"] = (n,)
                clip = {"time": torch.tensor(time - time[0], dtype=torch.float32, device=device)}
                for key, shape in shapes.items():
                    value = np.asarray(data[key])
                    if value.shape != shape or not np.isfinite(value).all():
                        raise ValueError(f"{file}: invalid {key}, expected finite {shape}")
                    if key == "support_mask":
                        count = value.sum(-1)
                        valid_count = count == 1 if version == 1 else (count >= 1) & (count <= 2)
                        if not np.isin(value, [0, 1]).all() or not valid_count.all():
                            requirement = "exactly one foot" if version == 1 else "one or two feet"
                            raise ValueError(f"{file}: support_mask must select {requirement}")
                    elif key == "phase":
                        if not ((value >= 0) & (value < 1)).all():
                            raise ValueError(f"{file}: phase must be cycle fractions in [0,1)")
                        value = np.unwrap(value * (2 * np.pi))
                    elif key == "step_duration" and (value <= 0).any():
                        raise ValueError(f"{file}: step_duration must be positive")
                    elif key.startswith("desired_step_") and (value < 0).any():
                        raise ValueError(f"{file}: {key} must be nonnegative")
                    clip[key] = torch.tensor(
                        value, dtype=torch.bool if key == "support_mask" else torch.float32, device=device
                    )
                for key in ("step_duration", "heading"):
                    if key in clip:
                        continue
                    value = float(data[key].item())
                    if not np.isfinite(value) or (key == "step_duration" and value <= 0):
                        raise ValueError(f"{file}: invalid {key}")
                    clip[key] = torch.tensor(value, dtype=torch.float32, device=device)
                clip["schema_version"] = version
                has_arms = "arm_reference_schema_version" in data
                if not has_arms and any(key.startswith(("arm_", "hand_")) for key in data.files):
                    raise ValueError(f"{file}: arm/hand fields require arm_reference_schema_version")
                if self.clips and has_arms != self.has_arm_reference:
                    raise ValueError(f"{file}: cannot mix clips with and without arm references")
                self.has_arm_reference = has_arms
                if has_arms:
                    clip.update(self._load_arm_reference(data, file, n, device, joint_names))
                self.clips.append(clip)
                durations.append(time[-1] - time[0])
        self.durations = torch.tensor(durations, dtype=torch.float32, device=device)

    def _load_arm_reference(self, data, file: Path, n: int, device: str, joint_names: list[str] | None):
        required = {
            "arm_joint_names",
            "arm_joint_pos",
            "arm_joint_vel",
            "hand_names",
            "hand_frame_names",
            "hand_target_frame",
            "hand_pos_torso",
        }
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{file}: missing arm reference fields: {', '.join(sorted(missing))}")
        version = np.asarray(data["arm_reference_schema_version"])
        if version.shape != () or version.dtype.kind not in "iu" or version.item() != 1:
            raise ValueError(f"{file}: expected arm_reference_schema_version=1")
        frame = np.asarray(data["hand_target_frame"])
        if frame.shape != () or frame.dtype.kind not in "US" or frame.astype(str).item() != "torso_link":
            raise ValueError(f"{file}: hand_target_frame must be torso_link")
        stored_joints = _read_names(data, "arm_joint_names", file)
        stored_hands = _read_names(data, "hand_names", file)
        stored_frames = _read_names(data, "hand_frame_names", file)
        if not set(stored_hands).issubset({"left", "right"}):
            raise ValueError(f"{file}: hand_names must select left and/or right")
        if len(stored_frames) != len(stored_hands):
            raise ValueError(f"{file}: hand_frame_names must match hand_names")
        if not self.clips:
            if joint_names is None:
                self.arm_joint_names = stored_joints
            else:
                missing_joints = set(stored_joints).difference(joint_names)
                if missing_joints:
                    raise ValueError(f"{file}: arm joints missing from robot: {', '.join(sorted(missing_joints))}")
                self.arm_joint_names = [name for name in joint_names if name in stored_joints]
                self.arm_joint_ids = torch.tensor(
                    [joint_names.index(name) for name in self.arm_joint_names], dtype=torch.long, device=device
                )
            self.hand_names = [name for name in ("left", "right") if name in stored_hands]
            self.hand_frame_names = [stored_frames[stored_hands.index(name)] for name in self.hand_names]
            self.hand_target_frame = "torso_link"
        if set(stored_joints) != set(self.arm_joint_names) or set(stored_hands) != set(self.hand_names):
            raise ValueError(f"{file}: all clips must share the same arm joint and hand sets")
        ji = [stored_joints.index(name) for name in self.arm_joint_names]
        hi = [stored_hands.index(name) for name in self.hand_names]
        if [stored_frames[index] for index in hi] != self.hand_frame_names:
            raise ValueError(f"{file}: hand frame definitions differ between clips")
        result = {}
        for key, shape, indices in (
            ("arm_joint_pos", (n, len(stored_joints)), ji),
            ("arm_joint_vel", (n, len(stored_joints)), ji),
            ("hand_pos_torso", (n, len(stored_hands), 3), hi),
        ):
            value = np.asarray(data[key])
            if value.shape != shape or value.dtype.kind not in "fiu" or not np.isfinite(value).all():
                raise ValueError(f"{file}: invalid {key}, expected finite {shape}")
            result[key] = torch.tensor(value[:, indices], dtype=torch.float32, device=device)
        return result

    def sample(self, clip_ids: torch.Tensor, times: torch.Tensor) -> dict[str, torch.Tensor]:
        """Sample support discretely; schema 2 uses velocity-consistent cubic Hermite.

        This preserves zero swing endpoint velocities even when control and export
        timesteps differ. Schema 1 retains its original walking interpolation.
        Arm positions always use cubic Hermite with analytically differentiated
        velocities. Hand positions interpolate linearly in torso coordinates;
        they do not receive a world-space alignment transform here.
        """
        result = {}
        for index, clip in enumerate(self.clips):
            mask = clip_ids == index
            ids = mask.nonzero(as_tuple=True)[0]
            t = times[ids].clamp(0, self.durations[index])
            hi = torch.searchsorted(clip["time"], t.contiguous(), right=True).clamp(1, len(clip["time"]) - 1)
            lo = hi - 1
            alpha = (t - clip["time"][lo]) / (clip["time"][hi] - clip["time"][lo])
            for key, value in clip.items():
                if key in ("time", "schema_version"):
                    continue
                if value.ndim == 0:
                    sampled = value.expand(len(ids))
                elif key == "support_mask" or (
                    clip["schema_version"] == 3
                    and key in ("step_duration", "desired_step_length", "desired_step_width")
                ):
                    nearest = torch.searchsorted(clip["time"], t.contiguous(), right=True).sub(1).clamp_min(0)
                    sampled = value[nearest]
                else:
                    weight = alpha.reshape((-1,) + (1,) * (value.ndim - 1))
                    sampled = torch.lerp(value[lo], value[hi], weight)
                if key not in result:
                    result[key] = torch.empty(
                        (len(clip_ids), *sampled.shape[1:]), device=times.device, dtype=sampled.dtype
                    )
                result[key][ids] = sampled
            velocity_pairs = []
            if "foot_vel" in clip:
                velocity_pairs.extend((("com_pos", "com_vel"), ("foot_pos", "foot_vel")))
            if "arm_joint_vel" in clip:
                velocity_pairs.append(("arm_joint_pos", "arm_joint_vel"))
            for position_key, velocity_key in velocity_pairs:
                p0, p1 = clip[position_key][lo], clip[position_key][hi]
                v0, v1 = clip[velocity_key][lo], clip[velocity_key][hi]
                shape = (-1,) + (1,) * (p0.ndim - 1)
                u = alpha.reshape(shape)
                h = (clip["time"][hi] - clip["time"][lo]).reshape(shape)
                result[position_key][ids] = (
                    (2 * u**3 - 3 * u**2 + 1) * p0
                    + (u**3 - 2 * u**2 + u) * h * v0
                    + (-2 * u**3 + 3 * u**2) * p1
                    + (u**3 - u**2) * h * v1
                )
                result[velocity_key][ids] = (
                    (6 * u**2 - 6 * u) * p0 / h
                    + (3 * u**2 - 4 * u + 1) * v0
                    + (-6 * u**2 + 6 * u) * p1 / h
                    + (3 * u**2 - 2 * u) * v1
                )
        return result
