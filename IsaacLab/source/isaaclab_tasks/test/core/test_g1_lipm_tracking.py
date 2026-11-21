# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LIPM NPZ contract: clocks, phase wrap, frames and physical tracking quantities."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from isaaclab.managers import ObservationManager, RewardManager, SceneEntityCfg
from isaaclab.paths import ISAACLAB_ROOT
from isaaclab.utils.io import dump_yaml

from isaaclab_tasks.core.velocity.mdp import lipm_mdp
from isaaclab_tasks.core.velocity.mdp.lipm_commands import LipmMotionCommand, LipmMotionCommandCfg, robot_com
from isaaclab_tasks.core.velocity.mdp.lipm_motion import LipmMotionLibrary
from isaaclab_tasks.utils import resolve_task_config


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "lipm.npz"
    t = np.arange(4) * 0.1
    com = np.column_stack((1 + t, np.zeros(4), np.full(4, 0.6)))
    feet = np.zeros((4, 2, 3))
    feet[:, :, 0] = 1
    feet[:, 0, 1] = 0.1
    feet[:, 1, 1] = -0.1
    np.savez(
        path,
        schema_version=1,
        frame="world_z_up",
        foot_names=["left", "right"],
        time=t + 0.02,
        com_pos=com,
        com_vel=np.tile([1.0, 0.0, 0.0], (4, 1)),
        foot_pos=feet,
        support_mask=[[1, 0], [1, 0], [0, 1], [0, 1]],
        phase=[0.9, 0.95, 0.0, 0.05],
        desired_step_length=np.full(4, 0.2),
        desired_step_width=np.full(4, 0.2),
        step_duration=0.4,
        heading=0.0,
    )
    return path


def test_npz_clock_phase_wrap_support_and_clamped_endpoints(clip):
    library = LipmMotionLibrary(str(clip), "cpu")
    sample = library.sample(torch.zeros(4, dtype=torch.long), torch.tensor([-1.0, 0.15, 0.2, 2.0]))
    torch.testing.assert_close(sample["com_pos"][:, 0], torch.tensor([1.0, 1.15, 1.2, 1.3]))
    torch.testing.assert_close(sample["phase"][1], torch.tensor(1.95 * np.pi, dtype=torch.float32))
    assert sample["support_mask"].tolist() == [[True, False], [True, False], [False, True], [False, True]]
    torch.testing.assert_close(library.durations, torch.tensor([0.3]))


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("foot_names", ["right", "left"], "foot order"),
        ("time", [0.0, 0.1, 0.1, 0.3], "strictly increasing"),
        ("support_mask", [[1, 1]] * 4, "exactly one foot"),
    ],
)
def test_invalid_export_fails_before_training(clip, field, value, match):
    with np.load(clip, allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    data[field] = value
    np.savez(clip, **data)
    with pytest.raises(ValueError, match=match):
        LipmMotionLibrary(str(clip), "cpu")


@pytest.fixture
def command_env(request):
    num_envs = getattr(request, "param", 1)

    def wrap(value):
        tensor = torch.tensor(value, dtype=torch.float32)
        return SimpleNamespace(torch=tensor.repeat(num_envs, *((1,) * (tensor.ndim - 1))))

    q = [0.0, 0.0, 2**-0.5, 2**-0.5]  # xyzw, 90-degree yaw
    data = SimpleNamespace(
        root_quat_w=wrap([q]),
        body_quat_w=wrap([[q, q]]),
        body_link_quat_w=wrap([[q, q]]),
        body_link_lin_vel_w=wrap([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
        body_link_ang_vel_w=wrap([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
        body_mass=wrap([[1.0, 3.0]]),
        body_com_pos_w=wrap([[[10.0, 20.0, 2.5], [10.0, 20.0, 3.5]]]),
        body_com_lin_vel_w=wrap([[[0.0, 0.0, 0.0], [0.0, 4.0, 0.0]]]),
        body_pos_w=wrap([[[9.9, 20.0, 2.035], [10.1, 20.0, 2.035]]]),
        joint_pos=wrap([[0.0, 0.0, 0.0, 0.0]]),
        joint_vel=wrap([[0.0, 0.0, 0.0, 0.0]]),
        default_joint_pos=wrap([[0.7, -0.1, 1.4, 0.2]]),
        default_joint_vel=wrap([[0.0, 0.0, 0.0, 0.0]]),
        root_lin_vel_b=wrap([[0.0, 0.0, 0.0]]),
        root_ang_vel_b=wrap([[0.0, 0.0, 0.0]]),
        projected_gravity_b=wrap([[0.0, 0.0, -1.0]]),
    )
    robot = SimpleNamespace(
        data=data,
        body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
        joint_names=["waist_yaw_joint", "left_elbow_joint", "left_knee_joint", "right_elbow_joint"],
    )
    sensor = SimpleNamespace(
        body_names=robot.body_names, data=SimpleNamespace(net_forces_w=wrap([[[0.0, 0.0, 50.0], [0.0, 0.0, 0.0]]]))
    )

    class Scene(dict):
        env_origins = torch.tensor([[10.0, 20.0, 2.0]]).repeat(num_envs, 1)

    return SimpleNamespace(
        scene=Scene(robot=robot, contact_forces=sensor),
        num_envs=num_envs,
        device="cpu",
        step_dt=0.1,
        episode_length_buf=torch.zeros(num_envs, dtype=torch.long),
        sim=SimpleNamespace(is_playing=lambda: True),
        action_manager=SimpleNamespace(action=torch.zeros(num_envs, 4)),
    )


def test_command_alignment_com_observations_rewards_and_exhaustion(clip, command_env):
    env = command_env
    robot = env.scene["robot"]
    command = LipmMotionCommand(LipmMotionCommandCfg(motion_directory=str(clip)), env)
    assert not command.has_arm_reference
    assert command.arm_joint_ids is None
    assert "arm_joint_position_rmse" not in command.metrics
    env.command_manager = SimpleNamespace(get_term=lambda name: command)
    command.reset(torch.arange(1))
    # Real articulated CoM, not root or unweighted body mean.
    com, vel = robot_com(robot)
    torch.testing.assert_close(com, torch.tensor([[10.0, 20.0, 3.25]]))
    torch.testing.assert_close(vel, torch.tensor([[0.0, 3.0, 0.0]]))
    torch.testing.assert_close(command.reference["com_pos"], torch.tensor([[10.0, 20.1, 2.6]]))
    torch.testing.assert_close(command.reference["com_vel"], torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-6, rtol=0)
    targets = lipm_mdp.lipm_targets(env)
    assert targets.shape == (1, 19)
    # A world +Y error becomes forward +X in the heading frame.
    torch.testing.assert_close(targets[:, 6:9], torch.tensor([[0.1, 0.0, -0.65]]), atol=2e-6, rtol=0)
    torch.testing.assert_close(lipm_mdp.foot_position_tracking(env), torch.ones(1), atol=1e-6, rtol=0)
    torch.testing.assert_close(lipm_mdp.support_tracking(env), torch.ones(1))
    assert lipm_mdp.com_position_tracking(env).item() < 0.01
    assert not lipm_mdp.reference_finished(env).item()
    env.episode_length_buf[:] = 1
    command.compute(env.step_dt)
    torch.testing.assert_close(command.reference["com_pos"][:, 1], torch.tensor([20.2]))
    env.episode_length_buf[:] = 3
    assert lipm_mdp.reference_finished(env).item()
    cfg, _ = resolve_task_config("Isaac-Velocity-Flat-G1", "rsl_rl_cfg_entry_point", overrides=["presets=lipm"])
    assert isinstance(cfg.commands.base_velocity, LipmMotionCommandCfg)
    assert cfg.rewards.joint_position_tracking is None
    assert cfg.events.reset_robot_joints is not None


def test_separate_swing_height_ignores_drift_and_stance_height(clip, command_env):
    env = command_env
    command = LipmMotionCommand(LipmMotionCommandCfg(motion_directory=str(clip)), env)
    env.command_manager = SimpleNamespace(get_term=lambda name: command)
    command.reset(torch.arange(1))
    cfg, _ = resolve_task_config("Isaac-Velocity-Flat-G1", "rsl_rl_cfg_entry_point", overrides=["presets=lipm_hands"])
    rewards = RewardManager({"horizontal": cfg.rewards.lipm_feet, "swing": cfg.rewards.lipm_swing_height}, env)
    command.reference["foot_pos"][:] = command.foot_positions()
    # Right swing at 5 cm, left stance has an unrelated large Z error.
    command.reference["support_mask"][:] = torch.tensor([[True, False]])
    command.reference["foot_pos"][0, 0, 2] += 0.5
    command.reference["foot_pos"][0, 1, 2] += 0.05
    robot = env.scene["robot"]
    robot.data.body_pos_w.torch[0, 1, 2] += 0.05
    torch.testing.assert_close(rewards.compute(env.step_dt), torch.tensor([0.3]))
    # A meter of XY drift removes horizontal credit, but preserves full lift credit.
    command.reference["foot_pos"][..., 0] += 1.0
    torch.testing.assert_close(rewards.compute(env.step_dt), torch.tensor([0.1]))
    # Missing the swing height by one width earns exp(-1), not stance credit.
    robot.data.body_pos_w.torch[0, 1, 2] -= 0.02
    torch.testing.assert_close(
        rewards.compute(env.step_dt), torch.tensor([0.1 * np.exp(-1)], dtype=torch.float32), atol=1e-6, rtol=0
    )
    # Switch support: only the left height matters now.
    command.reference["support_mask"][:] = torch.tensor([[False, True]])
    command.reference["foot_pos"][0, 0, 2] = command.foot_positions()[0, 0, 2]
    torch.testing.assert_close(rewards.compute(env.step_dt), torch.tensor([0.1]))
    command.reference["support_mask"][:] = True
    torch.testing.assert_close(rewards.compute(env.step_dt), torch.zeros(1))


def test_com_relative_feet_preserve_tracking_under_common_drift(clip, command_env):
    env = command_env
    command = LipmMotionCommand(LipmMotionCommandCfg(motion_directory=str(clip)), env)
    env.command_manager = SimpleNamespace(get_term=lambda name: command)
    command.reset(torch.arange(1))
    cfg, _ = resolve_task_config("Isaac-Velocity-Flat-G1", "rsl_rl_cfg_entry_point", overrides=["presets=lipm_hands"])
    rewards = RewardManager({"relative": cfg.rewards.lipm_feet_com_relative}, env)
    command.reference["foot_pos"][:] = command.foot_positions()
    command.reference["com_pos"][:] = robot_com(command.robot)[0]
    torch.testing.assert_close(rewards.compute(env.step_dt), torch.tensor([0.1]))
    # Translate actual soles and all body CoMs together; the world target stays fixed.
    data = command.robot.data
    drift = torch.tensor([1.0, -0.5, 0.0])
    data.body_pos_w.torch += drift
    data.body_com_pos_w.torch += drift
    torch.testing.assert_close(rewards.compute(env.step_dt), torch.tensor([0.1]))
    assert lipm_mdp.foot_horizontal_position_tracking(env).item() < 1e-10
    # Moving feet relative to the body must still reduce reward despite common drift.
    data.body_pos_w.torch[..., 1] += 0.12
    torch.testing.assert_close(
        rewards.compute(env.step_dt), torch.tensor([0.1 * np.exp(-1)], dtype=torch.float32), atol=1e-6, rtol=0
    )
    # Vertical errors belong to the separate swing-height reward.
    data.body_pos_w.torch[..., 2] += 0.3
    torch.testing.assert_close(
        rewards.compute(env.step_dt), torch.tensor([0.1 * np.exp(-1)], dtype=torch.float32), atol=1e-6, rtol=0
    )
    with pytest.raises(ValueError, match="finite and positive"):
        lipm_mdp.foot_com_relative_tracking(env, std=0.0)


def test_phase_contact_penalties_and_sole_velocity(clip, command_env):
    env = command_env
    command = LipmMotionCommand(LipmMotionCommandCfg(motion_directory=str(clip)), env)
    env.command_manager = SimpleNamespace(get_term=lambda name: command)
    command.reset(torch.arange(1))
    cfg, _ = resolve_task_config("Isaac-Velocity-Flat-G1", "rsl_rl_cfg_entry_point", overrides=["presets=lipm_hands"])
    rewards = RewardManager(
        {"loaded_swing": cfg.rewards.lipm_loaded_swing, "stance_slide": cfg.rewards.lipm_stance_slide}, env
    )
    data = env.scene["robot"].data
    forces = env.scene["contact_forces"].data.net_forces_w.torch
    command.reference["support_mask"][:] = torch.tensor([[True, False]])
    forces[0, :, 2] = 50.0
    # Left link is stationary but rotates: omega cross (0, 0, -0.035) = (-0.07, 0, 0).
    data.body_link_ang_vel_w.torch[0, 0, 1] = 2.0
    data.body_link_lin_vel_w.torch[0, 1, 0] = 0.4
    torch.testing.assert_close(command.foot_velocities()[0, 0], torch.tensor([-0.07, 0.0, 0.0]))
    torch.testing.assert_close(rewards.compute(env.step_dt), torch.tensor([-0.107]))
    command._update_metrics()
    torch.testing.assert_close(command.metrics["contact_agreement"], torch.tensor([0.5]))
    torch.testing.assert_close(command.metrics["loaded_swing_fraction"], torch.ones(1))
    torch.testing.assert_close(command.metrics["stance_slide_speed_m_s"], torch.tensor([0.07]))
    torch.testing.assert_close(command.metrics["sole_speed_m_s"], torch.tensor([0.235]))
    # Airborne swing movement has no penalty; airborne stance has no sliding penalty.
    forces.zero_()
    torch.testing.assert_close(rewards.compute(env.step_dt), torch.zeros(1))
    forces[0, :, 2] = 50.0
    command.reference["support_mask"][:] = torch.tensor([[False, True]])
    torch.testing.assert_close(rewards.compute(env.step_dt), torch.tensor([-0.14]))
    # Double support averages both stance speeds and has no loaded-swing penalty.
    command.reference["support_mask"][:] = True
    torch.testing.assert_close(rewards.compute(env.step_dt), torch.tensor([-0.0235]))


def test_schema2_double_support_and_velocity_consistent_sampling(clip):
    with np.load(clip, allow_pickle=False) as source:
        fields = {key: source[key] for key in source.files}
    fields["schema_version"] = 2
    fields["support_mask"] = [[1, 1], [1, 0], [0, 1], [1, 0]]
    fields["foot_vel"] = np.zeros_like(fields["foot_pos"])
    # Independent cubic p(t)=t^3 on the first interval; derivative 3t^2.
    t = np.arange(4) * 0.1
    fields["foot_pos"][:, 1, 0] = t**3
    fields["foot_vel"][:, 1, 0] = 3 * t**2
    np.savez(clip, **fields)
    library = LipmMotionLibrary(str(clip), "cpu")
    samples = library.sample(torch.zeros(3, dtype=torch.long), torch.tensor([0.0, 0.05, 0.1]))
    assert samples["support_mask"].tolist() == [[True, True], [True, True], [True, False]]
    torch.testing.assert_close(samples["foot_pos"][1, 1, 0], torch.tensor(0.05**3))
    torch.testing.assert_close(samples["foot_vel"][:, 1, 0], torch.tensor([0.0, 3 * 0.05**2, 3 * 0.1**2]))
    command = SimpleNamespace(reference=samples, cfg=SimpleNamespace(foot_body_names=["left", "right"]))
    sensor = SimpleNamespace(
        body_names=["right", "left"],
        data=SimpleNamespace(
            net_forces_w=torch.tensor(
                [
                    [[0.0, 0.0, 80.0], [0.0, 0.0, 80.0]],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 80.0]],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 80.0]],
                ]
            )
        ),
    )
    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda name: command), scene={"contact_forces": sensor}
    )
    torch.testing.assert_close(lipm_mdp.support_tracking(env), torch.tensor([1.0, 0.5, 1.0]))
    fields["support_mask"][0] = [0, 0]
    np.savez(clip, **fields)
    with pytest.raises(ValueError, match="one or two feet"):
        LipmMotionLibrary(str(clip), "cpu")


def test_scheduled_parameters_are_held_until_boundary_and_duration_is_validated(clip):
    with np.load(clip, allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    data.update(
        schema_version=3,
        foot_vel=np.zeros((4, 2, 3)),
        step_duration=np.array([0.35, 0.35, 0.4, 0.4]),
        desired_step_length=np.array([0.1, 0.1, 0.2, 0.2]),
    )
    np.savez(clip, **data)
    library = LipmMotionLibrary(str(clip), "cpu")
    sample = library.sample(torch.zeros(3, dtype=torch.long), torch.tensor([0.199, 0.2, 1.0]))
    torch.testing.assert_close(sample["step_duration"], torch.tensor([0.35, 0.4, 0.4]))
    torch.testing.assert_close(sample["desired_step_length"], torch.tensor([0.1, 0.2, 0.2]))
    data["step_duration"][1] = 0
    np.savez(clip, **data)
    with pytest.raises(ValueError, match="step_duration must be positive"):
        LipmMotionLibrary(str(clip), "cpu")


@pytest.fixture
def arm_clip(clip):
    with np.load(clip, allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    t = np.array([0.0, 0.1, 0.25, 0.4])
    data.update(
        time=t + 0.02,
        arm_reference_schema_version=1,
        arm_joint_names=["right_elbow_joint", "left_elbow_joint"],
        arm_joint_pos=np.column_stack((2 + t**3, 1 + 2 * t**3)),
        arm_joint_vel=np.column_stack((3 * t**2, 6 * t**2)),
        hand_names=["right", "left"],
        hand_frame_names=["right_rubber_hand", "left_rubber_hand"],
        hand_target_frame="torso_link",
        hand_pos_torso=np.stack(
            (np.column_stack((t, -np.ones(4), t)), np.column_stack((2 * t, np.ones(4), t))), axis=1
        ),
    )
    np.savez(clip, **data)
    return clip


@pytest.mark.parametrize("command_env", [2], indirect=True)
def test_command_arm_targets_share_walking_clip_clock_and_partial_reset(arm_clip, command_env):
    with np.load(arm_clip, allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    # Make CoM progress an independent clock oracle on these nonuniform knots.
    t = data["time"] - data["time"][0]
    data["com_pos"][:, 0] = 1 + t
    np.savez(arm_clip, **data)
    data["com_pos"][:, 0] = 1 + 2 * t
    data["com_vel"][:, 0] = 2
    data["arm_joint_pos"] += 10
    data["hand_pos_torso"] += 2
    data["desired_step_length"][:] = 0.4
    np.savez(arm_clip.parent / "second.npz", **data)

    env = command_env
    command = LipmMotionCommand(LipmMotionCommandCfg(motion_directory=str(arm_clip.parent)), env)
    assert command.has_arm_reference
    assert command.arm_joint_ids.tolist() == [1, 3]
    command.reset(torch.arange(2))
    # Reset samples one control step ahead, with both target families from the same clip.
    offset = command.clip_ids.float() * 10
    torch.testing.assert_close(command.reference["arm_joint_pos"], torch.stack((1.002 + offset, 2.001 + offset), -1))
    torch.testing.assert_close(command.reference["com_pos"][:, 1], 20.1 + command.clip_ids.float() * 0.1)

    # Metrics use only selected arm joints and the pre-update reference. Large
    # errors on waist/knee joints must not contribute to the arm RMSE.
    robot = env.scene["robot"]
    robot.data.joint_pos.torch[:] = 100
    robot.data.joint_pos.torch[:, command.arm_joint_ids] = command.reference["arm_joint_pos"] + 0.25
    command.clip_ids[:] = torch.tensor([0, 1])
    env.episode_length_buf[:] = torch.tensor([1, 2])
    command.compute(env.step_dt)
    torch.testing.assert_close(command.metrics["arm_joint_position_rmse"], torch.full((2,), 0.25))
    torch.testing.assert_close(command.reference["arm_joint_pos"], torch.tensor([[1.016, 2.008], [11.054, 12.027]]))
    torch.testing.assert_close(
        command.reference["arm_joint_vel"], torch.tensor([[0.24, 0.12], [0.54, 0.27]]), atol=2e-5, rtol=1e-5
    )
    torch.testing.assert_close(command.reference["com_pos"], torch.tensor([[10.0, 20.2, 2.6], [10.0, 20.6, 2.6]]))
    assert command.reference["support_mask"].tolist() == [[True, False], [False, True]]
    torch.testing.assert_close(command.reference["desired_step_length"], torch.tensor([0.2, 0.4]))
    # Neither the 90-degree world yaw nor the environment origin applies to hands.
    torch.testing.assert_close(
        command.reference["hand_pos_torso"],
        torch.tensor([[[0.4, 1.0, 0.2], [0.2, -1.0, 0.2]], [[2.6, 3.0, 2.3], [2.3, 1.0, 2.3]]]),
    )

    other_reference = {key: value[1].clone() for key, value in command.reference.items()}
    other_velocity_command = command.command[1].clone()
    env.episode_length_buf[0] = 0
    command.reset(torch.tensor([0]))
    reset_offset = command.clip_ids[0].float() * 10
    torch.testing.assert_close(command.reference["arm_joint_pos"][0], torch.tensor([1.002, 2.001]) + reset_offset)
    torch.testing.assert_close(command.reference["com_pos"][0, 1], 20.1 + command.clip_ids[0].float() * 0.1)
    for key, value in other_reference.items():
        torch.testing.assert_close(command.reference[key][1], value)
    torch.testing.assert_close(command.command[1], other_velocity_command)


def test_arm_mapping_and_velocity_consistent_sampling_across_reordered_clips(arm_clip):
    with np.load(arm_clip, allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    for key in ("arm_joint_names", "hand_names", "hand_frame_names"):
        data[key] = data[key][::-1]
    for key in ("arm_joint_pos", "arm_joint_vel", "hand_pos_torso"):
        data[key] = data[key][:, ::-1]
    data["arm_joint_pos"] += 10
    np.savez(arm_clip.parent / "second.npz", **data)
    library = LipmMotionLibrary(
        str(arm_clip.parent),
        "cpu",
        joint_names=["waist_yaw_joint", "left_elbow_joint", "left_knee_joint", "right_elbow_joint"],
    )
    assert library.has_arm_reference
    assert library.arm_joint_names == ["left_elbow_joint", "right_elbow_joint"]
    assert library.arm_joint_ids.tolist() == [1, 3]
    assert library.hand_names == ["left", "right"]
    assert library.hand_frame_names == ["left_rubber_hand", "right_rubber_hand"]
    assert library.hand_target_frame == "torso_link"
    sample = library.sample(torch.tensor([0, 0, 0, 1]), torch.tensor([-1.0, 0.2, 1.0, 0.2]))
    # Independent cubic trajectories, sampled between nonuniform source knots.
    torch.testing.assert_close(
        sample["arm_joint_pos"], torch.tensor([[1.0, 2.0], [1.016, 2.008], [1.128, 2.064], [11.016, 12.008]])
    )
    torch.testing.assert_close(
        sample["arm_joint_vel"],
        torch.tensor([[0.0, 0.0], [0.24, 0.12], [0.96, 0.48], [0.24, 0.12]]),
        atol=2e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(sample["hand_pos_torso"][1], torch.tensor([[0.4, 1.0, 0.2], [0.2, -1.0, 0.2]]))
    torch.testing.assert_close(sample["hand_pos_torso"][3], sample["hand_pos_torso"][1])
    # A batch using only the second clip must still have fully initialized results.
    second_only = library.sample(torch.tensor([1]), torch.tensor([0.2]))
    torch.testing.assert_close(second_only["arm_joint_pos"][0], sample["arm_joint_pos"][3])


def test_single_hand_reference_without_robot_names_uses_stored_joint_order(arm_clip):
    with np.load(arm_clip, allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    for key in ("arm_joint_names", "hand_names", "hand_frame_names"):
        data[key] = data[key][:1]
    for key in ("arm_joint_pos", "arm_joint_vel", "hand_pos_torso"):
        data[key] = data[key][:, :1]
    np.savez(arm_clip, **data)
    library = LipmMotionLibrary(str(arm_clip), "cpu")
    assert library.arm_joint_ids is None
    assert library.arm_joint_names == ["right_elbow_joint"]
    assert library.hand_names == ["right"]
    sample = library.sample(torch.tensor([0]), torch.tensor([0.2]))
    torch.testing.assert_close(sample["arm_joint_pos"], torch.tensor([[2.008]]))
    torch.testing.assert_close(sample["arm_joint_vel"], torch.tensor([[0.12]]))
    torch.testing.assert_close(sample["hand_pos_torso"], torch.tensor([[[0.2, -1.0, 0.2]]]))


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("arm_reference_schema_version", 2, "arm_reference_schema_version=1"),
        ("arm_reference_schema_version", None, "require arm_reference_schema_version"),
        ("arm_joint_vel", None, "missing arm reference fields: arm_joint_vel"),
        ("arm_joint_names", ["left_elbow_joint"] * 2, "unique, nonempty names"),
        ("arm_joint_names", ["left_elbow_joint", "unknown_joint"], "arm joints missing from robot"),
        ("arm_joint_pos", np.full((4, 2), np.nan), "invalid arm_joint_pos"),
        ("arm_joint_vel", np.zeros((3, 2)), "invalid arm_joint_vel"),
        ("hand_names", ["right", "middle"], "hand_names must select"),
        ("hand_frame_names", ["right_rubber_hand"], "hand_frame_names must match"),
        ("hand_target_frame", "world", "hand_target_frame must be torso_link"),
        ("hand_pos_torso", np.zeros((4, 3)), "invalid hand_pos_torso"),
    ],
)
def test_invalid_arm_reference_fails_before_sampling(arm_clip, field, value, match):
    with np.load(arm_clip, allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    if value is None:
        del data[field]
    else:
        data[field] = value
    np.savez(arm_clip, **data)
    with pytest.raises(ValueError, match=match):
        LipmMotionLibrary(str(arm_clip), "cpu", joint_names=["left_elbow_joint", "right_elbow_joint"])


@pytest.mark.parametrize("incompatibility", ["no_arms", "joint_set", "hand_set", "hand_frame"])
def test_incompatible_arm_clip_library_is_rejected(arm_clip, incompatibility):
    with np.load(arm_clip, allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    if incompatibility == "no_arms":
        data = {key: value for key, value in data.items() if not key.startswith(("arm_", "hand_"))}
        match = "cannot mix clips"
    elif incompatibility == "joint_set":
        data["arm_joint_names"] = ["right_elbow_joint", "left_wrist_pitch_joint"]
        match = "same arm joint and hand sets"
    elif incompatibility == "hand_set":
        data["hand_names"] = ["right"]
        data["hand_frame_names"] = ["right_rubber_hand"]
        data["hand_pos_torso"] = data["hand_pos_torso"][:, :1]
        match = "same arm joint and hand sets"
    else:
        data["hand_frame_names"] = ["right_wrist_yaw_link", "left_rubber_hand"]
        match = "hand frame definitions differ"
    np.savez(arm_clip.parent / "second.npz", **data)
    with pytest.raises(ValueError, match=match):
        LipmMotionLibrary(str(arm_clip.parent), "cpu")


@pytest.mark.parametrize("command_env", [2], indirect=True)
def test_hands_preset_observation_and_reward_managers(arm_clip, command_env):
    env = command_env
    cfg, _ = resolve_task_config("Isaac-Velocity-Flat-G1", "rsl_rl_cfg_entry_point", overrides=["presets=lipm_hands"])
    cfg.commands.base_velocity.motion_directory = str(arm_clip)
    command = LipmMotionCommand(cfg.commands.base_velocity, env)
    env.command_manager = SimpleNamespace(get_term=lambda name: command)
    command.reset(torch.arange(2))
    data = env.scene["robot"].data
    # Perfect tracking in env 0; +/- one reward width on each arm in env 1.
    # Large errors on unrelated joints must not affect the tracking rewards.
    data.joint_pos.torch[:] = 100
    data.joint_vel.torch[:] = 100
    data.joint_pos.torch[:, command.arm_joint_ids] = command.reference["arm_joint_pos"]
    data.joint_vel.torch[:, command.arm_joint_ids] = command.reference["arm_joint_vel"]
    data.joint_pos.torch[1, command.arm_joint_ids] += torch.tensor([0.3, -0.3])
    data.joint_vel.torch[1, command.arm_joint_ids] += torch.tensor([1.0, -1.0])

    observations = ObservationManager(cfg.observations, env).compute()["policy"]
    assert observations.shape == (2, 44)  # 4-joint fixture: 40 existing + 2 positions + 2 velocities.
    torch.testing.assert_close(observations[:, -4:], torch.tensor([[1.102, 1.801, 0.06, 0.03]]).repeat(2, 1))
    walking_cfg, _ = resolve_task_config("Isaac-Velocity-Flat-G1", "rsl_rl_cfg_entry_point", overrides=["presets=lipm"])
    walking_observations = ObservationManager(walking_cfg.observations, env).compute()["policy"]
    torch.testing.assert_close(observations[:, :40], walking_observations)
    # Resolve the walking preset after the hands preset to catch shared-config mutation.
    assert walking_cfg.rewards.joint_deviation_arms.func is not cfg.rewards.joint_deviation_arms.func
    torch.testing.assert_close(
        lipm_mdp.arm_joint_position_tracking(env), torch.tensor([1.0, np.exp(-1)], dtype=torch.float32)
    )
    torch.testing.assert_close(
        lipm_mdp.arm_joint_velocity_tracking(env), torch.tensor([1.0, np.exp(-1)], dtype=torch.float32)
    )
    rewards = RewardManager({"position": cfg.rewards.lipm_arm_position, "velocity": cfg.rewards.lipm_arm_velocity}, env)
    torch.testing.assert_close(
        rewards.compute(env.step_dt), torch.tensor([0.11, 0.11 * np.exp(-1)], dtype=torch.float32)
    )
    # All walking objectives remain usable alongside the arm terms.
    for name in ("lipm_feet", "lipm_com_position", "lipm_com_velocity", "lipm_support"):
        term = getattr(cfg.rewards, name)
        values = term.func(env, **term.params)
        assert values.shape == (2,) and torch.isfinite(values).all()


def test_single_hand_preset_keeps_opposite_arm_posture_penalty(arm_clip, command_env):
    with np.load(arm_clip, allow_pickle=False) as source:
        fields = {key: source[key] for key in source.files}
    for key in ("arm_joint_names", "hand_names", "hand_frame_names"):
        fields[key] = fields[key][:1]  # Right-hand reference only.
    for key in ("arm_joint_pos", "arm_joint_vel", "hand_pos_torso"):
        fields[key] = fields[key][:, :1]
    np.savez(arm_clip, **fields)
    env = command_env
    cfg, _ = resolve_task_config("Isaac-Velocity-Flat-G1", "rsl_rl_cfg_entry_point", overrides=["presets=lipm_hands"])
    cfg.commands.base_velocity.motion_directory = str(arm_clip)
    command = LipmMotionCommand(cfg.commands.base_velocity, env)
    env.command_manager = SimpleNamespace(get_term=lambda name: command)
    command.reset(torch.tensor([0]))
    data = env.scene["robot"].data
    data.joint_pos.torch[:] = torch.tensor([[100.0, 0.4, 100.0, 100.0]])
    initial_pos = data.joint_pos.torch.clone()
    initial_ref = command.reference["arm_joint_pos"].clone()
    term = cfg.rewards.joint_deviation_arms
    # Resolve the fixture's two elbow IDs directly; the simulator resolves the
    # preset's shoulder/elbow name selectors for the full G1 asset.
    penalty = term.func(env, asset_cfg=SceneEntityCfg("robot", joint_ids=[1, 3]))
    torch.testing.assert_close(penalty, torch.tensor([0.5]))
    torch.testing.assert_close(data.joint_pos.torch, initial_pos)
    torch.testing.assert_close(command.reference["arm_joint_pos"], initial_ref)
    observations = ObservationManager(cfg.observations, env).compute()["policy"]
    assert observations.shape == (1, 42)
    torch.testing.assert_close(observations[:, -2:], torch.tensor([[1.801, 0.03]]))


def test_hands_preset_rejects_walking_only_reference_at_observation_initialization(clip, command_env):
    cfg, _ = resolve_task_config("Isaac-Velocity-Flat-G1", "rsl_rl_cfg_entry_point", overrides=["presets=lipm_hands"])
    cfg.commands.base_velocity.motion_directory = str(clip)
    env = command_env
    command = LipmMotionCommand(cfg.commands.base_velocity, env)
    env.command_manager = SimpleNamespace(get_term=lambda name: command)
    command.reset(torch.tensor([0]))
    with pytest.raises(ValueError, match="arm-augmented LIP NPZ"):
        ObservationManager(cfg.observations, env)


@pytest.mark.parametrize(
    "reward,std",
    [(lipm_mdp.arm_joint_position_tracking, 0.0), (lipm_mdp.arm_joint_velocity_tracking, float("nan"))],
)
def test_arm_tracking_rejects_invalid_reward_width(reward, std, command_env):
    with pytest.raises(ValueError, match="std must be finite and positive"):
        reward(command_env, std=std)


@pytest.fixture(scope="module")
def reference_playback():
    path = Path(ISAACLAB_ROOT) / "scripts/reinforcement_learning/play_g1_reference.py"
    spec = importlib.util.spec_from_file_location("g1_reference_playback", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("preset", ["reference", "lipm", "lipm_hands"])
def test_playback_restores_concrete_saved_preset(reference_playback, tmp_path, arm_clip, command_env, preset):
    cfg, agent = resolve_task_config(
        "Isaac-Velocity-Flat-G1", "rsl_rl_cfg_entry_point", overrides=[f"presets={preset}"]
    )
    dump_yaml(str(tmp_path / "params/env.yaml"), cfg)
    dump_yaml(str(tmp_path / "params/agent.yaml"), agent)
    checkpoint = tmp_path / "model_1499.pt"
    checkpoint.touch()  # Config restoration does not read model weights.
    restored, restored_agent, detected = reference_playback.load_playback_config(
        checkpoint, motion=arm_clip, device="cpu"
    )
    assert detected == preset and type(restored) is type(cfg)
    assert type(restored.commands.base_velocity) is type(cfg.commands.base_velocity)
    assert restored.scene.num_envs == 1
    assert restored.sim.device == restored_agent.device == "cpu"
    assert restored.commands.base_velocity.motion_directory == str(arm_clip.resolve())
    if preset == "reference":
        assert not restored.commands.base_velocity.random_start
    else:
        assert not hasattr(restored.commands.base_velocity, "random_start")
    if preset == "lipm_hands":
        # Saved functions are import strings until the managers resolve them.
        command = LipmMotionCommand(restored.commands.base_velocity, command_env)
        command_env.command_manager = SimpleNamespace(get_term=lambda name: command)
        command.reset(torch.tensor([0]))
        observations = ObservationManager(restored.observations, command_env).compute()["policy"]
        assert observations.shape == (1, 44)
        torch.testing.assert_close(observations[:, -4:], torch.tensor([[1.102, 1.801, 0.06, 0.03]]))
    wrong_preset = "reference" if preset != "reference" else "lipm_hands"
    with pytest.raises(ValueError, match="Checkpoint uses presets="):
        reference_playback.load_playback_config(checkpoint, preset=wrong_preset)


def test_playback_gait_labels_use_lip_timestamps_and_blend_flags(reference_playback, clip):
    with np.load(clip, allow_pickle=False) as source:
        fields = {key: source[key] for key in source.files}
    fields.update(
        schema_version=3,
        foot_vel=np.zeros((4, 2, 3)),
        time=np.array([0.0, 0.1, 0.25, 0.4]),
        desired_step_length=np.array([0.1, 0.1, 0.15, 0.2]),
        step_duration=np.array([0.5, 0.5, 0.475, 0.45]),
        gait_schedule=np.array([[0.0, 0.1, 0.5], [0.2, 0.2, 0.45]]),
        startup_duration=0.1,
    )
    np.savez(clip, **fields)
    labels = reference_playback.load_gait_labels(clip)
    np.testing.assert_array_equal(labels["time"], [0.0, 0.1, 0.25, 0.4])
    np.testing.assert_array_equal(labels["condition_stage"], [0, 0, 1, 1])
    np.testing.assert_array_equal(labels["is_transition"], [1, 0, 1, 0])
    np.testing.assert_allclose(labels["step_conditioning"][2], [0.15, 0.475])
    # Older MPC exports keep their own explicit labels and FPS-based timeline.
    np.savez(clip, step_conditioning=[[0.1, 0.5], [0.2, 0.45]], condition_stage=[0, 1], is_transition=[0, 1], fps=20)
    labels = reference_playback.load_gait_labels(clip)
    np.testing.assert_array_equal(labels["time"], [0.0, 0.05])
    np.testing.assert_array_equal(labels["condition_stage"], [0, 1])


def test_playback_tracking_uses_current_time_and_physical_hand_points(
    reference_playback, arm_clip, command_env, tmp_path
):
    env = command_env
    robot = env.scene["robot"]
    # Add measured wrist/torso poses with a translated, 90-degree-yaw torso.
    # Use the real matched URDF's fixed hand offsets (41.5 mm forward, +/-3 mm lateral).
    robot.body_names += ["torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link"]
    q = robot.data.body_quat_w.torch[:, :1].expand(-1, 3, -1)
    robot.data.body_link_quat_w = SimpleNamespace(torch=torch.cat((robot.data.body_quat_w.torch, q), dim=1))
    robot.data.body_link_pos_w = SimpleNamespace(
        torch=torch.cat(
            (robot.data.body_pos_w.torch, torch.tensor([[[10.0, 20.0, 1.0], [10.0, 20.2, 1.1], [10.2, 20.2, 1.1]]])),
            dim=1,
        )
    )
    # Rigid torso/hand rotation plus translation: relative hand velocity must vanish.
    positions = robot.data.body_link_pos_w.torch
    angular = torch.zeros_like(positions)
    angular[..., 2] = 2.0
    linear = torch.cross(angular, positions - positions[:, 2:3], dim=-1)
    linear[..., 0] += 0.3
    robot.data.body_link_lin_vel_w = SimpleNamespace(torch=linear)
    robot.data.body_link_ang_vel_w = SimpleNamespace(torch=angular)
    with np.load(arm_clip, allow_pickle=False) as source:
        fields = {key: source[key] for key in source.files}
    # Stored order is right,left; left has exactly 5 cm Euclidean hand error.
    fields["hand_pos_torso"][:] = [[0.2415, -0.203, 0.1], [0.2715, -0.037, 0.1]]
    np.savez(arm_clip, **fields)
    command = LipmMotionCommand(LipmMotionCommandCfg(motion_directory=str(arm_clip)), env)
    command.reset(torch.tensor([0]))
    env.episode_length_buf[:] = 2
    command.compute(env.step_dt)  # Command holds t=0.3 targets; actual state below matches t=0.2.
    robot.data.joint_pos.torch[:, command.arm_joint_ids] = torch.tensor([[1.016, 2.008]])
    robot.data.joint_vel.torch[:, command.arm_joint_ids] = torch.tensor([[0.24, 0.12]])
    row = reference_playback.LipmTrackingEvaluator(command).sample(env.episode_length_buf, env.step_dt)
    assert row["reference_time_s"] == pytest.approx(0.2)
    assert row["arm_joint_pos_rmse_rad"] == pytest.approx(0.0, abs=1e-6)
    assert row["arm_joint_vel_rmse_rad_s"] == pytest.approx(0.0, abs=2e-5)
    assert row["left_hand_position_error_m"] == pytest.approx(0.05, abs=3e-6)
    assert row["right_hand_position_error_m"] == pytest.approx(0.0, abs=3e-6)
    assert row["left_hand_actual_x_torso_m"] == pytest.approx(0.2415, abs=3e-6)
    assert row["left_hand_actual_y_torso_m"] == pytest.approx(0.003, abs=3e-6)
    assert row["torso_yaw_rad"] == pytest.approx(np.pi / 2, abs=1e-6)
    assert row["torso_angular_speed_rad_s"] == pytest.approx(2.0)
    assert row["left_hand_torso_relative_velocity_speed_m_s"] == pytest.approx(0.0, abs=3e-6)
    assert row["left_hand_world_velocity_speed_m_s"] > 0.1
    # Exercise registered hand rewards with translated/rotated torso and merged wrists.
    env.command_manager = SimpleNamespace(get_term=lambda name: command)
    cfg, _ = resolve_task_config("Isaac-Velocity-Flat-G1", "rsl_rl_cfg_entry_point", overrides=["presets=lipm_hands"])
    rewards = RewardManager(
        {"left": cfg.rewards.lipm_left_hand_position, "right": cfg.rewards.lipm_right_hand_position}, env
    )
    torch.testing.assert_close(
        rewards.compute(env.step_dt),
        torch.tensor([0.05 * (np.exp(-0.25) + 1)], dtype=torch.float32),
        atol=1e-6,
        rtol=0,
    )
    # Missing the left target badly must not suppress the right hand's independent credit.
    command.reference["hand_pos_torso"][:, 0, 0] += 1.0
    torch.testing.assert_close(rewards.compute(env.step_dt), torch.tensor([0.05]), atol=1e-6, rtol=0)
    # Unmerged hand bodies use their origins, with no wrist offset applied twice.
    robot.body_names.append("right_rubber_hand")
    right_point = robot.data.body_link_pos_w.torch[:, 4:5] + torch.tensor([[[0.003, 0.0415, 0.0]]])
    robot.data.body_link_pos_w.torch = torch.cat((robot.data.body_link_pos_w.torch, right_point), dim=1)
    torch.testing.assert_close(lipm_mdp.hand_position_tracking(env, "right"), torch.ones(1), atol=1e-6, rtol=0)
    command.library.hand_names = ["right"]
    command.library.hand_frame_names = ["right_rubber_hand"]
    command.reference["hand_pos_torso"] = command.reference["hand_pos_torso"][:, 1:2]
    torch.testing.assert_close(lipm_mdp.hand_position_tracking(env, "left"), torch.zeros(1))
    torch.testing.assert_close(lipm_mdp.hand_position_tracking(env, "right"), torch.ones(1), atol=1e-6, rtol=0)
    # A rigid-body frame needs no offset; missing movable frames are rejected.
    ids, offsets = reference_playback.resolve_hand_points(robot.body_names, ["left_wrist_yaw_link"])
    assert ids == [3]
    np.testing.assert_array_equal(offsets, [[0, 0, 0]])
    with pytest.raises(ValueError, match="Cannot resolve hand frame"):
        reference_playback.resolve_hand_points(["torso_link"], ["left_rubber_hand"])
    reference_playback.write_tracking_report(tmp_path, [row])
    summary = json.loads((tmp_path / "tracking_summary.json").read_text())
    assert summary["metrics"]["left_hand_position_error_m"]["mean"] == pytest.approx(0.05, abs=3e-6)
    assert summary["hand_frame"] == "torso_link"
    assert summary["motion_metrics"]["torso_angular_speed_rad_s"]["rms"] == pytest.approx(2.0)
    assert (tmp_path / "reference_tracking.csv").is_file()
