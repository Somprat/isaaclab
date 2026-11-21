# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU-only sampling contracts: time interpolation and batched frame transforms."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

_PATH = Path(__file__).resolve().parents[2] / "isaaclab_tasks/contrib/humanoid_amp/motions/motion_loader.py"
_SPEC = importlib.util.spec_from_file_location("motion_loader_under_test", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


@pytest.fixture
def motion(tmp_path):
    positions = np.zeros((3, 2, 3), dtype=np.float32)
    positions[:, 0] = [[10, 2, 1], [11, 2, 1], [12, 2, 1]]
    positions[:, 1] = positions[:, 0] + [1, 0, 0]
    yaw = np.array([0, 0.001, np.pi / 2])
    quats = np.zeros((3, 2, 4), dtype=np.float32)
    quats[:, :, 0] = np.cos(yaw[:, None] / 2)
    quats[:, :, 3] = np.sin(yaw[:, None] / 2)
    quats[2] *= -1  # Same orientation, opposite quaternion representation.
    path = tmp_path / "motion.npz"
    np.savez(
        path,
        dof_names=["joint"],
        body_names=["root", "tip"],
        dof_positions=np.array([[0], [2], [10]], dtype=np.float32),
        dof_velocities=np.zeros((3, 1), dtype=np.float32),
        body_positions=positions,
        body_rotations=quats,
        body_linear_velocities=np.tile([1.0, 0.0, 0.0], (3, 2, 1)),
        body_angular_velocities=np.tile([0.0, 0.0, 2.0], (3, 2, 1)),
        fps=np.array(2.0),
    )
    return _MODULE.MotionLoader(str(path), torch.device("cpu"))


def test_sampling_clamps_and_uses_enclosing_frames(motion):
    sampled = motion.sample(3, times=np.array([0.3, -1.0, 2.0]))
    torch.testing.assert_close(sampled[0][:, 0], torch.tensor([1.2, 0.0, 10.0]))
    assert all(torch.isfinite(value).all() for value in sampled)


def test_quaternion_interpolation_respects_time_and_sign(motion):
    sampled = motion.sample(2, times=np.array([0.1, 0.75]))
    expected_yaw = np.array([0.0002, (0.001 + np.pi / 2) / 2])
    expected = torch.tensor(
        np.stack([np.cos(expected_yaw / 2), np.zeros(2), np.zeros(2), np.sin(expected_yaw / 2)], axis=-1),
        dtype=torch.float32,
    )
    torch.testing.assert_close(sampled[3][:, 0], expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(torch.linalg.vector_norm(sampled[3], dim=-1), torch.ones((2, 2)))


def test_policy_steps_use_control_time_not_source_fps(motion):
    sampled = motion.sample_steps(np.array([0, 15, 50]), step_dt=0.02)
    torch.testing.assert_close(sampled[0][:, 0], torch.tensor([0.0, 1.2, 10.0]))
    with pytest.raises(ValueError, match="timestep"):
        motion.sample_steps(np.array([0]), step_dt=0)


def test_alignment_rebases_xy_preserves_height_and_rotates_world_vectors(motion):
    sampled = motion.sample(
        2,
        times=np.array([0.5, 0.5]),
        env_origins=torch.tensor([[100.0, 200.0, 3.0], [-5.0, 7.0, 0.0]]),
        target_headings=torch.tensor([np.pi / 2, 0.001]),
        anchor_times=np.array([0.0, 0.5]),
        root_body_name="root",
    )
    torch.testing.assert_close(sampled[2][0], torch.tensor([[100.0, 201.0, 4.0], [100.0, 202.0, 4.0]]))
    torch.testing.assert_close(sampled[2][1], torch.tensor([[-5.0, 7.0, 1.0], [-4.0, 7.0, 1.0]]))
    torch.testing.assert_close(sampled[4][0], torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(sampled[5][0], torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]]))
    yaw = np.pi / 2 + 0.001
    torch.testing.assert_close(
        sampled[3][0, 0], torch.tensor([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], dtype=torch.float32)
    )
    raw = motion.sample(1, times=np.array([0.5]))
    torch.testing.assert_close(raw[2][0, 0], torch.tensor([11.0, 2.0, 1.0]))
