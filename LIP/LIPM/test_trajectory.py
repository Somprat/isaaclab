"""Physical and boundary contracts for generated startup/swing references."""

import numpy as np
import pytest
from trajectory import generate_reference, smooth_swing


@pytest.mark.parametrize("length,period,heading", [(0.2, 0.4, 0.0), (0.1, 0.5, np.pi / 2)])
def test_startup_matches_reset_and_requested_first_landing(length, period, heading):
    rotation = np.array([[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]])
    com = np.r_[rotation @ [0.0237138, 0.000072], 0.714414]
    feet = np.array([[-0.026, 0.1185065, 0.0], [-0.026, -0.1185065, 0.0]])
    feet[:, :2] = feet[:, :2] @ rotation.T
    data = generate_reference(
        com,
        [0.0, 0.0],
        feet,
        step_length=length,
        step_width=0.237013,
        step_duration=period,
        heading=heading,
        duration=2.0,
    )
    n = round(float(data["startup_duration"] / data["dt"]))
    landing = n + round(period / float(data["dt"]))
    np.testing.assert_allclose(data["com_pos"][0], com, atol=1e-12)
    np.testing.assert_allclose(data["com_vel"][0], 0, atol=1e-12)
    np.testing.assert_allclose(data["foot_pos"][: n + 1], np.broadcast_to(feet, (n + 1, 2, 3)), atol=1e-12)
    assert data["support_mask"][:n].all()
    assert data["support_mask"][n].tolist() == [True, False]
    assert data["support_mask"][landing].tolist() == [False, True]
    expected = feet[0, :2] + rotation @ [length, -0.237013]
    np.testing.assert_allclose(data["foot_pos"][landing, 1, :2], expected, atol=1e-9)
    np.testing.assert_allclose(data["foot_vel"][data["support_mask"]], 0, atol=1e-12)
    # Continuous position, velocity and acceleration at the startup/single-support join.
    for axis in range(2):
        p = np.polynomial.Polynomial(data["startup_coefficients"][:, axis])
        d = float(data["startup_duration"])
        np.testing.assert_allclose(data["com_pos"][n, axis], p(1), atol=1e-12)
        np.testing.assert_allclose(data["com_vel"][n, axis], p.deriv()(1) / d, atol=1e-12)
        np.testing.assert_allclose(data["com_acc"][n, axis], p.deriv(2)(1) / d**2, atol=1e-10)
    assert data["startup_zmp_margin"] >= 0.003 - 1e-7
    assert data["startup_max_acceleration"] <= 0.35 * 9.81 + 1e-7
    # Independent LIPM equation check, including the double-support startup.
    np.testing.assert_allclose(
        data["com_pos"][:, :2] - data["com_acc"][:, :2] * com[2] / 9.81,
        data["zmp"][:, :2],
        atol=1e-12,
    )
    for value in data.values():
        if value.dtype.kind in "fc":
            assert np.isfinite(value).all()


def test_swing_has_rest_endpoints_and_correct_derivatives():
    start, target = np.array([0.1, -0.1, 0.0]), np.array([0.3, -0.1, 0.0])
    p, v, a = smooth_swing(start, target, np.array([0.0, 0.5, 1.0]), 0.4, 0.05)
    np.testing.assert_allclose(p[[0, 2]], [start, target], atol=1e-12)
    np.testing.assert_allclose(v[[0, 2]], 0, atol=1e-12)
    np.testing.assert_allclose(a[[0, 2]], 0, atol=1e-12)
    assert p[1, 2] == pytest.approx(0.05)
    u, eps = 0.31, 1e-5
    before = smooth_swing(start, target, u - eps, 0.4, 0.05)
    after = smooth_swing(start, target, u + eps, 0.4, 0.05)
    middle = smooth_swing(start, target, u, 0.4, 0.05)
    np.testing.assert_allclose((after[0] - before[0]) / (2 * eps * 0.4), middle[1], atol=1e-7)
    np.testing.assert_allclose((after[1] - before[1]) / (2 * eps * 0.4), middle[2], atol=1e-6)


def test_infeasible_startup_is_rejected():
    with pytest.raises(ValueError, match="No feasible"):
        generate_reference(
            [2.0, 0.0, 0.7],
            [0.0, 0.0],
            [[0.0, 0.1, 0.0], [0.0, -0.1, 0.0]],
            step_length=0.2,
            step_width=0.2,
            step_duration=0.4,
            startup_duration=0.8,
            max_startup_duration=0.8,
            duration=2.0,
        )


def test_scheduled_gait_preserves_dynamics_and_changes_only_at_touchdown():
    data = generate_reference(
        [0.0237138, 0.000072, 0.714414],
        [0.0, 0.0],
        [[-0.026, 0.1185065, 0.0], [-0.026, -0.1185065, 0.0]],
        step_length=0.2,
        step_width=0.237013,
        step_duration=0.4,
        duration=20,
        dt=0.005,
        schedule=[[0, 0.1, 0.35], [20 / 3, 0.2, 0.4], [40 / 3, 0.3, 0.45]],
        transition_steps=2,
    )
    np.testing.assert_allclose(
        data["applied_gait_changes"],
        [
            [0.8, 0.1, 0.35],
            [6.75, 0.15, 0.375],
            [7.125, 0.2, 0.4],
            [13.525, 0.25, 0.425],
            [13.95, 0.3, 0.45],
        ],
        atol=1e-10,
    )
    changes = np.flatnonzero(np.diff(data["desired_step_length"])) + 1
    assert np.all(np.any(data["support_mask"][changes] != data["support_mask"][changes - 1], axis=1))
    np.testing.assert_allclose(data["foot_vel"][changes], 0, atol=1e-12)
    # Exact LIPM evolution across touchdowns catches resetting/jumping the CoM.
    start = round(float(data["startup_duration"] / data["dt"]))
    p, v = data["com_pos"][start:, :2], data["com_vel"][start:, :2]
    zmp = data["zmp"][start:-1, :2]
    w = np.sqrt(9.81 / 0.714414)
    h = float(data["dt"])
    np.testing.assert_allclose(p[1:], zmp + (p[:-1] - zmp) * np.cosh(w * h) + v[:-1] * np.sinh(w * h) / w, atol=1e-10)
    np.testing.assert_allclose(v[1:], (p[:-1] - zmp) * w * np.sinh(w * h) + v[:-1] * np.cosh(w * h), atol=1e-10)
    assert np.max(np.linalg.norm(np.diff(data["foot_pos"], axis=0), axis=-1)) < 0.03
    assert data["time"][-1] == pytest.approx(20)
