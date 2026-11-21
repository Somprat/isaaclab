"""Behavioral checks for the driven-body arm dynamics diagnostic."""

import importlib.util
from pathlib import Path

import mujoco
import numpy as np

MODULE = (
    Path(__file__).resolve().parents[1]
    / "scripts/replay_hand_horizontal_hold_torque.py"
)
spec = importlib.util.spec_from_file_location("hand_torque_replay", MODULE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_prescribed_body_does_not_overwrite_arm_state_and_motor_is_limited_pd():
    replay = module.TorqueReplay(module.ROOT / "LIP/previews/perfect-arm-replay-source")
    data, model = replay.data, replay.model
    # Moving the external body reference must not teleport the dynamic robot.
    before_q, before_v = data.qpos.copy(), data.qvel.copy()
    replay.drive_body(0.5)
    np.testing.assert_array_equal(data.qpos, before_q)
    np.testing.assert_array_equal(data.qvel, before_v)
    replay.drive_body(0.0)
    # Different signs and large commands exercise force saturation as well as PD.
    delta = np.linspace(-2.0, 2.0, 14)
    data.ctrl[replay.arm_act] = data.qpos[replay.arm_q] + delta
    data.qvel[replay.arm_v] = np.linspace(-0.2, 0.2, 14)
    mujoco.mj_forward(model, data)
    expected = np.clip(
        40.0 * delta - 10.0 * data.qvel[replay.arm_v], -replay.limit, replay.limit
    )
    np.testing.assert_allclose(
        data.actuator_force[replay.arm_act], expected, atol=1e-10
    )
    assert np.any(np.isclose(abs(expected), replay.limit))
    # Restore the initial recording state before the integration check.
    data.qpos[:] = before_q
    data.qvel[:] = before_v
    mujoco.mj_forward(model, data)
    targets, orientations = replay.targets(0)
    targets[:, 2] += 0.1
    increments = replay.solve_ik(targets, orientations)
    caps = np.full(14, 0.05)
    caps[[0, 7]] = 0.30
    assert np.all(abs(increments) <= caps + 1e-12)
    np.testing.assert_array_equal(data.qpos, before_q)
    old_arm = data.qpos[replay.arm_q].copy()
    desired = replay.q_des.copy()
    for i in range(replay.substeps):
        replay.advance(i * model.opt.timestep)
    assert np.max(abs(data.qpos[replay.arm_q] - old_arm)) > 1e-6
    assert np.max(abs(data.qpos[replay.arm_q] - desired)) > 1e-4
    assert np.isfinite(data.qvel).all()


def test_retimed_world_offsets_keep_body_time_and_capture_at_fourteen():
    replay = module.TorqueReplay(
        module.ROOT / "LIP/previews/perfect-arm-replay-source", reach_end_s=14.0
    )
    assert replay.target_time(0.0) == 0.0
    assert replay.target_time(1.0) == 1.0
    assert replay.target_time(7.5) == 4.0
    assert replay.target_time(14.0) == 7.0
    assert replay.target_time(18.0) == 7.0
    source_i = round(4.0 / replay.dt)
    world = replay.recorded_targets(source_i)
    expected = world.copy()
    expected[:, :2] -= replay.reference_data.xpos[replay.torso, :2]
    # A different CURRENT torso translation and rotation must not rotate the
    # earlier world-axis offsets or move their absolute world-height target.
    replay.data.qpos[:3] += [2.0, -1.0, 0.3]
    replay.data.qpos[3:7] = [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]
    mujoco.mj_forward(replay.model, replay.data)
    expected[:, :2] += replay.data.xpos[replay.torso, :2]
    actual, _ = replay.targets(round(7.5 / replay.dt))
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    assert replay.held_xy is None
    replay.targets(round(13.98 / replay.dt))
    assert replay.held_xy is None
    replay.targets(round(14.0 / replay.dt))
    assert replay.held_xy is not None
    replay.drive_body(7.5)
    np.testing.assert_allclose(replay.data.mocap_pos[0], replay.position_spline(7.5))


def test_original_position_survives_capture_and_follows_only_horizontal_translation():
    replay = module.TorqueReplay(
        module.ROOT / "LIP/previews/perfect-arm-replay-source", preserve_position=True
    )
    initial_arm = replay.data.qpos[replay.arm_q].copy()
    target, orientation = replay.targets(round(14.0 / replay.dt))
    assert replay.held_quat is not None
    expected = replay.target_path[round(7.0 / replay.dt)].copy()
    expected[:, :2] += replay.data.xpos[replay.torso, :2]
    np.testing.assert_allclose(target, expected)
    assert (
        np.linalg.norm(target - replay.points()) > 0.1
    )  # Failed reach must not replace the target.
    replay.data.qpos[:3] += [1.0, 2.0, 0.3]
    replay.data.qpos[replay.arm_q] = initial_arm + 0.1
    mujoco.mj_forward(replay.model, replay.data)
    later, later_orientation = replay.targets(round(15.0 / replay.dt))
    np.testing.assert_allclose(later, target + [1.0, 2.0, 0.0], atol=1e-10)
    np.testing.assert_allclose(later_orientation, orientation)
