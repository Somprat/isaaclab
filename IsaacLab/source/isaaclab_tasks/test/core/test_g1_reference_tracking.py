# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU contracts for reference correspondence, command clocks, and tracking rewards."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from isaaclab_tasks.core.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg
from isaaclab_tasks.core.velocity.config.g1.mpc_model import read_urdf_model
from isaaclab_tasks.core.velocity.mdp import reference_observations as obs
from isaaclab_tasks.core.velocity.mdp import reference_rewards as rew
from isaaclab_tasks.core.velocity.mdp.reference_commands import ReferenceMotionCommand, ReferenceMotionCommandCfg
from isaaclab_tasks.core.velocity.mdp.reference_motion import ReferenceMotionLibrary
from isaaclab_tasks.utils import resolve_task_config


def write_clip(path, fps=2.0):
    frames = np.arange(9, dtype=np.float32)
    body = np.zeros((9, 2, 3), dtype=np.float32)
    body[:, 0, 0] = 10 + frames / 10
    body[:, 0, 2] = 1
    body[:, 1] = body[:, 0] + [0.5, 0, 0]
    quat = np.zeros((9, 2, 4), dtype=np.float32)
    quat[:, :, 0] = 1
    quat[1::2] *= -1
    flags = np.array([[1, 0], [1, 1], [0, 1], [1, 1]] * 2 + [[1, 0]], dtype=np.uint8)
    np.savez(
        path,
        joint_names=["j0", "j1"],
        body_names=["pelvis", "tip"],
        joint_pos=np.stack((2 * frames, -2 * frames), -1),
        joint_vel=np.ones((9, 2), dtype=np.float32),
        body_pos_w=body,
        body_quat_w=quat,
        body_lin_vel_w=np.tile([0.2, 0.0, 0.0], (9, 2, 1)),
        body_ang_vel_w=np.zeros((9, 2, 3), dtype=np.float32),
        conditioning=np.stack((np.full(9, 0.5), np.full(9, 0.4)), -1),
        contact_flags=flags,
        contact_names=["foot_l_contact", "foot_r_contact"],
        fps=np.array([fps]),
    )


@pytest.fixture
def clips(tmp_path):
    write_clip(tmp_path / "a.npz")
    return tmp_path


def test_batched_library_reorders_names_and_respects_per_clip_fps(clips):
    write_clip(clips / "b.npz", fps=4.0)
    library = ReferenceMotionLibrary(str(clips), ["j1", "j0"], ["tip", "pelvis"], "cpu")
    sample = library.sample(torch.tensor([0, 1]), torch.tensor([0.25, 0.25]))
    torch.testing.assert_close(sample["joint_pos"], torch.tensor([[-1.0, 1.0], [-2.0, 2.0]]))
    torch.testing.assert_close(sample["body_pos_w"][:, 0, 0], torch.tensor([10.55, 10.6]))
    torch.testing.assert_close(sample["phase"], torch.tensor([np.pi / 4, np.pi / 2], dtype=torch.float32))
    torch.testing.assert_close(sample["body_quat_w"].norm(dim=-1), torch.ones(2, 2))
    torch.testing.assert_close(sample["body_quat_w"][0, 0].abs(), torch.tensor([0.0, 0.0, 0.0, 1.0]))
    end = library.sample(torch.tensor([0]), torch.tensor([100.0]))
    torch.testing.assert_close(end["joint_pos"], torch.tensor([[-16.0, 16.0]]))
    with pytest.raises(ValueError, match="joint names"):
        ReferenceMotionLibrary(str(clips), ["wrong"], ["pelvis"], "cpu")


@pytest.fixture
def command_env(clips):
    def wrap(value):
        return SimpleNamespace(torch=value)

    data = SimpleNamespace(
        joint_pos=wrap(torch.zeros(2, 2)),
        joint_vel=wrap(torch.zeros(2, 2)),
        default_joint_pos=wrap(torch.zeros(2, 2)),
        root_pos_w=wrap(torch.zeros(2, 3)),
        root_quat_w=wrap(torch.tensor([[1.0, 0, 0, 0]]).repeat(2, 1)),
        body_pos_w=wrap(torch.zeros(2, 2, 3)),
    )
    robot = SimpleNamespace(joint_names=["j1", "j0"], body_names=["tip", "pelvis"], data=data)

    def write_root(root_pose, env_ids):
        data.root_pos_w.torch[env_ids] = root_pose[:, :3]
        data.root_quat_w.torch[env_ids] = root_pose[:, 3:7]

    def write_velocity(root_velocity, env_ids):
        robot.reset_link_velocity = root_velocity.clone()

    def write_joints(position, velocity, env_ids):
        data.joint_pos.torch[env_ids] = position
        data.joint_vel.torch[env_ids] = velocity

    robot.write_root_link_pose_to_sim_index = write_root
    robot.write_root_link_velocity_to_sim_index = write_velocity
    robot.write_joint_state_to_sim = write_joints

    class Scene(dict):
        env_origins = torch.tensor([[100.0, 200.0, 0.0], [-10.0, 5.0, 2.0]])

    env = SimpleNamespace(
        scene=Scene(robot=robot),
        num_envs=2,
        device="cpu",
        step_dt=0.1,
        episode_length_buf=torch.zeros(2, dtype=torch.long),
    )
    cfg = ReferenceMotionCommandCfg(motion_directory=str(clips), random_start=False, reset_height_offset=0)
    command = ReferenceMotionCommand(cfg, env)
    env.command_manager = SimpleNamespace(get_term=lambda name: command)
    command.reset(torch.arange(2))
    return command, env


def test_command_reset_and_reward_use_the_same_next_control_frame(command_env):
    command, env = command_env
    torch.testing.assert_close(command.robot.data.root_quat_w.torch, torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1))
    torch.testing.assert_close(
        command.robot.data.root_pos_w.torch, torch.tensor([[100.0, 200.0, 1.0], [-10.0, 5.0, 3.0]])
    )
    torch.testing.assert_close(command.reference["joint_pos"], torch.tensor([[-0.4, 0.4], [-0.4, 0.4]]))
    torch.testing.assert_close(command.command, torch.tensor([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]))
    torch.testing.assert_close(obs.stride_length(env), torch.tensor([[0.4], [0.4]]))
    torch.testing.assert_close(obs.step_length(env), torch.tensor([[0.2], [0.2]]))
    torch.testing.assert_close(
        obs.gait_phase(env), torch.tensor([[np.sin(np.pi / 10), np.cos(np.pi / 10)]]).float().repeat(2, 1)
    )
    # Reward is evaluated after physics, before the command manager advances.
    command.robot.data.joint_pos.torch[:] = command.reference["joint_pos"]
    torch.testing.assert_close(rew.joint_position_tracking(env, 0.3), torch.ones(2))
    env.episode_length_buf[:] = 1
    command.compute(0.1)
    torch.testing.assert_close(command.reference["joint_pos"], torch.tensor([[-0.8, 0.8], [-0.8, 0.8]]))
    env.episode_length_buf[:] = 40
    assert rew.reference_finished(env).all()


def test_tracking_rewards_decrease_with_error_and_quaternion_sign_is_irrelevant(command_env):
    command, env = command_env
    robot = command.robot.data
    robot.joint_pos.torch[:] = command.reference["joint_pos"] + 0.3
    torch.testing.assert_close(rew.joint_position_tracking(env, 0.3), torch.full((2,), np.exp(-1)))
    robot.joint_vel.torch[:] = command.reference["joint_vel"] + 3.0
    torch.testing.assert_close(rew.joint_velocity_tracking(env, 3.0), torch.full((2,), np.exp(-1)))
    robot.root_quat_w.torch[:] = -command.reference["body_quat_w"][:, command.root_index]
    torch.testing.assert_close(rew.root_orientation_tracking(env, 0.4), torch.ones(2))
    robot.root_pos_w.torch[:] = command.reference["body_pos_w"][:, command.root_index]
    robot.body_pos_w.torch[:] = command.reference["body_pos_w"]
    torch.testing.assert_close(rew.body_position_tracking(env, 0.15, ["tip"]), torch.ones(2))
    robot.body_pos_w.torch[:, 0, 0] += 0.15
    torch.testing.assert_close(
        rew.body_position_tracking(env, 0.15, ["tip"]), torch.full((2,), np.exp(-1)), atol=1e-5, rtol=1e-5
    )


def test_reference_is_a_preset_of_the_existing_task_and_default_is_preserved():
    base, _ = resolve_task_config("Isaac-Velocity-Flat-G1", "rsl_rl_cfg_entry_point", overrides=[])
    cfg, _ = resolve_task_config("Isaac-Velocity-Flat-G1", "rsl_rl_cfg_entry_point", overrides=["presets=reference"])
    assert type(base) is G1FlatEnvCfg
    assert base.commands.base_velocity.class_type is not ReferenceMotionCommand
    assert not hasattr(base.observations.policy, "stride_length")
    assert base.commands.base_velocity.ranges.lin_vel_x == (0.0, 0.5)
    assert cfg.commands.base_velocity.class_type is ReferenceMotionCommand
    assert cfg.observations.policy.stride_length.func is obs.stride_length
    assert cfg.observations.policy.gait_phase.func is obs.gait_phase
    assert cfg.rewards.joint_position_tracking.func is rew.joint_position_tracking
    assert cfg.events.reset_base is None and cfg.events.reset_robot_joints is None
    # Reference shaping augments, rather than replaces, velocity walking.
    for name in ("track_lin_vel_xy_exp", "track_ang_vel_z_exp", "feet_air_time", "feet_slide", "action_rate_l2"):
        assert getattr(cfg.rewards, name).func is getattr(base.rewards, name).func
        assert getattr(cfg.rewards, name).weight == getattr(base.rewards, name).weight
    assert cfg.scene.contact_forces is not None
    assert cfg.actions.to_dict() == base.actions.to_dict()


def test_source_model_merges_fixed_mass_and_rotated_inertia_about_com(tmp_path):
    # Independent compound-body result: 2 kg at x=0 and 1 kg at x=1,
    # with the second body's unequal principal axes rotated by 90 degrees.
    path = tmp_path / "compound.urdf"
    path.write_text("""<robot name="compound">
      <link name="base"><inertial><mass value="2"/>
        <inertia ixx="0.2" ixy="0" ixz="0" iyy="0.3" iyz="0" izz="0.4"/>
      </inertial></link>
      <link name="fixed"><inertial><mass value="1"/>
        <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.2" iyz="0" izz="0.3"/>
      </inertial></link>
      <joint name="weld" type="fixed"><parent link="base"/><child link="fixed"/>
        <origin xyz="1 0 0" rpy="0 0 1.5707963267948966"/>
      </joint></robot>""")
    body = read_urdf_model(path)["bodies"]["base"]
    assert body["mass"] == 3
    np.testing.assert_allclose(body["com"], [1 / 3, 0, 0], atol=1e-12)
    np.testing.assert_allclose(body["inertia"], np.diag([0.4, 16 / 15, 41 / 30]), atol=1e-12)
