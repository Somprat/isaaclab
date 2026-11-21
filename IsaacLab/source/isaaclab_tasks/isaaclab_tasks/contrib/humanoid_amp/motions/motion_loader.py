# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os

import numpy as np
import torch


class MotionLoader:
    """
    Helper class to load and sample motion data from NumPy-file format.
    """

    def __init__(self, motion_file: str, device: torch.device) -> None:
        """Load a motion file and initialize the internal variables.

        Args:
            motion_file: Motion file path to load. Accepts AMP field names (such as
                ``dof_positions``) and MPC aliases (such as ``joint_pos``). Body rotations
                must use wxyz quaternions. Joint and body order is preserved; use
                :meth:`get_dof_index` and :meth:`get_body_index` to map to a robot.
            device: The device to which to load the data.

        Raises:
            AssertionError: If the specified motion file doesn't exist.
        """
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        self.device = device
        with np.load(motion_file, allow_pickle=False) as data:
            # Keep the existing AMP names authoritative when both formats are present.
            dof_names_key = "dof_names" if "dof_names" in data else "joint_names"
            self._dof_names = data[dof_names_key].astype(str).tolist()
            self._body_names = data["body_names"].astype(str).tolist()

            # MPC exports use joint_* and world-frame body_*_w field names.
            motion_fields = {
                "dof_positions": "joint_pos",
                "dof_velocities": "joint_vel",
                "body_positions": "body_pos_w",
                "body_rotations": "body_quat_w",
                "body_linear_velocities": "body_lin_vel_w",
                "body_angular_velocities": "body_ang_vel_w",
            }
            for name, alias in motion_fields.items():
                key = name if name in data else alias
                setattr(self, name, torch.tensor(data[key], dtype=torch.float32, device=self.device))

            # Accept both the scalar AMP fps and the one-element MPC fps array.
            self.dt = 1.0 / float(data["fps"].item())
        self.num_frames = self.dof_positions.shape[0]
        self.duration = self.dt * (self.num_frames - 1)
        print(f"Motion loaded ({motion_file}): duration: {self.duration} sec, frames: {self.num_frames}")

    @property
    def dof_names(self) -> list[str]:
        """Skeleton DOF names."""
        return self._dof_names

    @property
    def body_names(self) -> list[str]:
        """Skeleton rigid body names."""
        return self._body_names

    @property
    def num_dofs(self) -> int:
        """Number of skeleton's DOFs."""
        return len(self._dof_names)

    @property
    def num_bodies(self) -> int:
        """Number of skeleton's rigid bodies."""
        return len(self._body_names)

    def _interpolate(
        self,
        a: torch.Tensor,
        *,
        b: torch.Tensor | None = None,
        blend: torch.Tensor | None = None,
        start: np.ndarray | None = None,
        end: np.ndarray | None = None,
    ) -> torch.Tensor:
        """Linear interpolation between consecutive values.

        Args:
            a: The first value. Shape is (N, X) or (N, M, X).
            b: The second value. Shape is (N, X) or (N, M, X).
            blend: Interpolation coefficient between 0 (a) and 1 (b).
            start: Indexes to fetch the first value. If both, ``start`` and ``end` are specified,
                the first and second values will be fetches from the argument ``a`` (dimension 0).
            end: Indexes to fetch the second value. If both, ``start`` and ``end` are specified,
                the first and second values will be fetches from the argument ``a`` (dimension 0).

        Returns:
            Interpolated values. Shape is (N, X) or (N, M, X).
        """
        if start is not None and end is not None:
            return self._interpolate(a=a[start], b=a[end], blend=blend)
        if a.ndim >= 2:
            blend = blend.unsqueeze(-1)
        if a.ndim >= 3:
            blend = blend.unsqueeze(-1)
        return (1.0 - blend) * a + blend * b

    def _slerp(
        self,
        q0: torch.Tensor,
        *,
        q1: torch.Tensor | None = None,
        blend: torch.Tensor | None = None,
        start: np.ndarray | None = None,
        end: np.ndarray | None = None,
    ) -> torch.Tensor:
        """Interpolation between consecutive rotations (Spherical Linear Interpolation).

        Args:
            q0: The first quaternion (wxyz). Shape is (N, 4) or (N, M, 4).
            q1: The second quaternion (wxyz). Shape is (N, 4) or (N, M, 4).
            blend: Interpolation coefficient between 0 (q0) and 1 (q1).
            start: Indexes to fetch the first quaternion. If both, ``start`` and ``end` are specified,
                the first and second quaternions will be fetches from the argument ``q0`` (dimension 0).
            end: Indexes to fetch the second quaternion. If both, ``start`` and ``end` are specified,
                the first and second quaternions will be fetches from the argument ``q0`` (dimension 0).

        Returns:
            Interpolated quaternions. Shape is (N, 4) or (N, M, 4).
        """
        if start is not None and end is not None:
            return self._slerp(q0=q0[start], q1=q0[end], blend=blend)
        if q0.ndim >= 2:
            blend = blend.unsqueeze(-1)
        if q0.ndim >= 3:
            blend = blend.unsqueeze(-1)

        q0 = torch.nn.functional.normalize(q0, dim=-1)
        q1 = torch.nn.functional.normalize(q1, dim=-1)
        dot = (q0 * q1).sum(dim=-1, keepdim=True)
        q1 = torch.where(dot < 0.0, -q1, q1)
        dot = dot.abs().clamp(0.0, 1.0)
        theta = torch.acos(dot)
        sin_theta = torch.sin(theta).clamp_min(1e-8)
        spherical = (torch.sin((1.0 - blend) * theta) * q0 + torch.sin(blend * theta) * q1) / sin_theta
        linear = (1.0 - blend) * q0 + blend * q1
        return torch.nn.functional.normalize(torch.where(dot > 0.9995, linear, spherical), dim=-1)

    def _compute_frame_blend(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the indexes of the first and second values, as well as the blending time
        to interpolate between them and the given times.

        Args:
            times: Times, between 0 and motion duration, to sample motion values.
                Specified times will be clipped to fall within the range of the motion duration.

        Returns:
            First value indexes, Second value indexes, and blending time between 0 (first value) and 1 (second value).
        """
        times = np.asarray(times, dtype=np.float64)
        if times.ndim != 1 or not np.isfinite(times).all():
            raise ValueError("Motion times must be a finite one-dimensional array.")
        frame = np.clip(times, 0.0, self.duration) / self.dt
        index_0 = np.floor(frame).astype(int).clip(0, self.num_frames - 1)
        index_1 = np.minimum(index_0 + 1, self.num_frames - 1)
        blend = np.clip(frame - index_0, 0.0, 1.0)
        return index_0, index_1, blend

    def sample_times(self, num_samples: int, duration: float | None = None) -> np.ndarray:
        """Sample random motion times uniformly.

        Args:
            num_samples: Number of time samples to generate.
            duration: Maximum motion duration to sample.
                If not defined samples will be within the range of the motion duration.

        Raises:
            AssertionError: If the specified duration is longer than the motion duration.

        Returns:
            Time samples, between 0 and the specified/motion duration.
        """
        duration = self.duration if duration is None else duration
        assert duration <= self.duration, (
            f"The specified duration ({duration}) is longer than the motion duration ({self.duration})"
        )
        return duration * np.random.uniform(low=0.0, high=1.0, size=num_samples)

    def sample(
        self,
        num_samples: int,
        times: np.ndarray | None = None,
        duration: float | None = None,
        *,
        env_origins: torch.Tensor | None = None,
        target_headings: torch.Tensor | None = None,
        anchor_times: np.ndarray | None = None,
        root_body_name: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample motion data.

        Args:
            num_samples: Number of time samples to generate. If ``times`` is defined, this parameter is ignored.
            times: Motion time used for sampling.
                If not defined, motion data will be random sampled uniformly in time.
            duration: Maximum motion duration to sample.
                If not defined, samples will be within the range of the motion duration.
                If ``times`` is defined, this parameter is ignored.

            env_origins: Optional world offsets of shape (N, 3). The root XY at
                ``anchor_times`` is placed at each environment's XY origin.
                Reference height is preserved and shifted by the origin's Z.
            target_headings: Optional desired root yaw at the anchor, shape (N,),
                in radians. Rotates every body pose and world-frame velocity.
                If omitted, the original heading is preserved.
            anchor_times: Fixed reset/reference times per environment, shape (N,).
                Defaults to clip start. Keep these fixed as playback advances.
            root_body_name: Root body used for alignment; defaults to the first
                body in the file (pelvis in MPC G1 exports).

        Returns:
            A tuple containing sampled motion data:
                - DOF positions (with shape (N, num_dofs))
                - DOF velocities (with shape (N, num_dofs))
                - Body positions (with shape (N, num_bodies, 3))
                - Body rotations (with shape (N, num_bodies, 4), as wxyz quaternion)
                - Body linear velocities (with shape (N, num_bodies, 3))
                - Body angular velocities (with shape (N, num_bodies, 3))
        """
        times = self.sample_times(num_samples, duration) if times is None else times
        index_0, index_1, blend = self._compute_frame_blend(times)
        blend = torch.tensor(blend, dtype=torch.float32, device=self.device)

        sampled = (
            self._interpolate(self.dof_positions, blend=blend, start=index_0, end=index_1),
            self._interpolate(self.dof_velocities, blend=blend, start=index_0, end=index_1),
            self._interpolate(self.body_positions, blend=blend, start=index_0, end=index_1),
            self._slerp(self.body_rotations, blend=blend, start=index_0, end=index_1),
            self._interpolate(self.body_linear_velocities, blend=blend, start=index_0, end=index_1),
            self._interpolate(self.body_angular_velocities, blend=blend, start=index_0, end=index_1),
        )
        if env_origins is not None or target_headings is not None:
            sampled = self._align_sample(sampled, env_origins, target_headings, anchor_times, root_body_name)
        return sampled

    def sample_steps(
        self,
        steps: np.ndarray,
        step_dt: float,
        start_times: np.ndarray | float = 0.0,
        *,
        env_origins: torch.Tensor | None = None,
        target_headings: torch.Tensor | None = None,
        anchor_times: np.ndarray | None = None,
        root_body_name: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample at policy steps, without changing playback speed or source frame rate.

        ``step_dt`` is the policy/control timestep in seconds (for example,
        ``env.step_dt``), not the physics substep. For resets into the middle of
        a clip, pass the same saved per-environment ``start_times`` as
        ``anchor_times`` to keep the alignment fixed throughout the episode.
        Alignment arguments have the same meaning as in :meth:`sample`.
        """
        if not np.isfinite(step_dt) or step_dt <= 0.0:
            raise ValueError("Policy timestep must be finite and positive.")
        steps = np.asarray(steps, dtype=np.float64)
        if steps.ndim != 1 or not np.isfinite(steps).all() or np.any(steps < 0):
            raise ValueError("Policy steps must be a finite nonnegative one-dimensional array.")
        times = np.asarray(start_times, dtype=np.float64) + steps * step_dt
        return self.sample(
            len(steps),
            times=times,
            env_origins=env_origins,
            target_headings=target_headings,
            anchor_times=anchor_times,
            root_body_name=root_body_name,
        )

    def _align_sample(
        self,
        sampled: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        env_origins: torch.Tensor | None,
        target_headings: torch.Tensor | None,
        anchor_times: np.ndarray | None,
        root_body_name: str | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply one fixed horizontal root transform per sampled environment."""
        dof_pos, dof_vel, positions, rotations, lin_vel, ang_vel = sampled
        count = positions.shape[0]
        origins = (
            torch.zeros((count, 3), device=self.device)
            if env_origins is None
            else torch.as_tensor(env_origins, device=self.device, dtype=positions.dtype)
        )
        if origins.shape != (count, 3) or not torch.isfinite(origins).all():
            raise ValueError("env_origins must have shape (samples, 3) and contain finite values.")
        anchor_times = np.zeros(count) if anchor_times is None else np.asarray(anchor_times, dtype=np.float64)
        if anchor_times.shape != (count,):
            raise ValueError("anchor_times must have shape (samples,).")
        start, end, blend = self._compute_frame_blend(anchor_times)
        blend = torch.tensor(blend, dtype=torch.float32, device=self.device)
        root = self.get_body_index([root_body_name or self.body_names[0]])[0]
        anchor_pos = self._interpolate(self.body_positions[:, root], start=start, end=end, blend=blend).clone()
        # Preserve absolute reference height above the ground; only rebase horizontal displacement.
        anchor_pos[:, 2] = 0.0
        if target_headings is None:
            delta = torch.zeros(count, device=self.device)
        else:
            headings = torch.as_tensor(target_headings, device=self.device, dtype=positions.dtype)
            if headings.shape != (count,) or not torch.isfinite(headings).all():
                raise ValueError("target_headings must have shape (samples,) and be finite radians.")
            q = self._slerp(self.body_rotations[:, root], start=start, end=end, blend=blend)
            w, x, y, z = q.unbind(-1)
            delta = headings - torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        c, s = torch.cos(delta)[:, None], torch.sin(delta)[:, None]

        def rotate(v: torch.Tensor) -> torch.Tensor:
            return torch.stack((c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1], v[..., 2]), dim=-1)

        positions = rotate(positions - anchor_pos[:, None]) + origins[:, None]
        c_half, s_half = torch.cos(delta / 2.0)[:, None], torch.sin(delta / 2.0)[:, None]
        w, x, y, z = rotations.unbind(-1)
        rotations = torch.stack(
            (c_half * w - s_half * z, c_half * x - s_half * y, c_half * y + s_half * x, c_half * z + s_half * w), dim=-1
        )
        return dof_pos, dof_vel, positions, rotations, rotate(lin_vel), rotate(ang_vel)

    def get_dof_index(self, dof_names: list[str]) -> list[int]:
        """Get skeleton DOFs indexes by DOFs names.

        Args:
            dof_names: List of DOFs names.

        Raises:
            AssertionError: If the specified DOFs name doesn't exist.

        Returns:
            List of DOFs indexes.
        """
        indexes = []
        for name in dof_names:
            assert name in self._dof_names, f"The specified DOF name ({name}) doesn't exist: {self._dof_names}"
            indexes.append(self._dof_names.index(name))
        return indexes

    def get_body_index(self, body_names: list[str]) -> list[int]:
        """Get skeleton body indexes by body names.

        Args:
            dof_names: List of body names.

        Raises:
            AssertionError: If the specified body name doesn't exist.

        Returns:
            List of body indexes.
        """
        indexes = []
        for name in body_names:
            assert name in self._body_names, f"The specified body name ({name}) doesn't exist: {self._body_names}"
            indexes.append(self._body_names.index(name))
        return indexes


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Motion file")
    args, _ = parser.parse_known_args()

    motion = MotionLoader(args.file, "cpu")

    print("- number of frames:", motion.num_frames)
    print("- number of DOFs:", motion.num_dofs)
    print("- number of bodies:", motion.num_bodies)
