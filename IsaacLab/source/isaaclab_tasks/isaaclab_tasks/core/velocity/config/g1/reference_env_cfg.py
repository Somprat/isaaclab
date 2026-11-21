# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reference-tracking preset of Isaac-Velocity-Flat-G1; no separate task ID."""

import math
from dataclasses import fields
from pathlib import Path

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import mdp
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.paths import ISAACLAB_ROOT
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils import configclass, replace

from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets import G1_29DOF_CFG

from ...mdp import hand_commands, lipm_mdp
from ...mdp import reference_observations as obs
from ...mdp import reference_rewards as rew
from ...mdp.hand_actions import HandIKJointPositionActionCfg
from ...mdp.lipm_commands import LipmMotionCommandCfg
from ...mdp.reference_commands import ReferenceMotionCommandCfg
from .flat_env_cfg import G1FlatEnvCfg
from .mpc_model import MPC_USD, validate_mpc_model_event


@configclass
class ReferenceObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        stride_length = ObsTerm(func=obs.stride_length)
        gait_phase = ObsTerm(func=obs.gait_phase)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        reference_joint_pos = ObsTerm(func=obs.reference_joint_positions)
        reference_joint_vel = ObsTerm(func=obs.reference_joint_velocities)
        reference_root_orientation = ObsTerm(func=obs.reference_root_orientation)
        reference_root_height = ObsTerm(func=obs.reference_root_height)

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    policy = PolicyCfg()


@configclass
class ReferenceRewardsCfg:
    joint_position_tracking = RewTerm(func=rew.joint_position_tracking, weight=2.0, params={"std": 0.3})
    joint_velocity_tracking = RewTerm(func=rew.joint_velocity_tracking, weight=0.5, params={"std": 3.0})
    root_height_tracking = RewTerm(func=rew.root_height_tracking, weight=0.5, params={"std": 0.1})
    root_orientation_tracking = RewTerm(func=rew.root_orientation_tracking, weight=0.5, params={"std": 0.4})
    body_position_tracking = RewTerm(
        func=rew.body_position_tracking,
        weight=1.0,
        params={
            "std": 0.15,
            "body_names": [
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "torso_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ],
        },
    )


@configclass
class ReferenceCommandsCfg:
    base_velocity = ReferenceMotionCommandCfg(
        motion_directory=str(Path(ISAACLAB_ROOT).parent / "wb_humanoid_mpc/generated_motions/conditioned_50_fixed")
    )


@configclass
class ReferenceTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    reference_finished = DoneTerm(func=rew.reference_finished, time_out=True)
    fallen = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.35})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})


@configclass
class G1ReferenceFlatEnvCfg(G1FlatEnvCfg):
    """Existing flat walking task with synchronized motion-backed commands/rewards."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = replace(G1_29DOF_CFG, prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.usd_path = str(MPC_USD)
        self.scene.contact_forces.prim_path = "{ENV_REGEX_NS}/Robot/Geometry/.*"
        self.scene.robot.init_state.rot = (0.0, 0.0, 0.0, 1.0)
        # Keep locomotion PD gains, while physical effort/velocity limits and
        # zero rotor armature follow the exact MPC URDF.
        self.scene.robot.actuators = {
            "motors": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness={
                    ".*_hip_.*": 100.0,
                    ".*_knee_joint": 150.0,
                    ".*_ankle_.*": 40.0,
                    "waist_.*": 200.0,
                    ".*_shoulder_.*": 40.0,
                    ".*_elbow_joint": 40.0,
                    ".*_wrist_.*": 40.0,
                },
                damping={
                    ".*_hip_.*": 2.0,
                    ".*_knee_joint": 4.0,
                    ".*_ankle_.*": 2.0,
                    "waist_.*": 5.0,
                    ".*_shoulder_.*": 10.0,
                    ".*_elbow_joint": 10.0,
                    ".*_wrist_.*": 10.0,
                },
                joint_effort_limit={
                    ".*_hip_roll_joint": 88.0,
                    ".*_knee_joint": 139.0,
                    ".*_hip_(pitch|yaw)_joint": 88.0,
                    "waist_yaw_joint": 88.0,
                    "waist_(roll|pitch)_joint": 50.0,
                    ".*_ankle_.*": 50.0,
                    ".*_shoulder_.*": 25.0,
                    ".*_elbow_joint": 25.0,
                    ".*_wrist_roll_joint": 25.0,
                    ".*_wrist_(pitch|yaw)_joint": 5.0,
                },
                joint_velocity_limit={
                    ".*_hip_roll_joint": 32.0,
                    ".*_knee_joint": 20.0,
                    ".*_hip_(pitch|yaw)_joint": 32.0,
                    "waist_yaw_joint": 32.0,
                    "waist_(roll|pitch)_joint": 37.0,
                    ".*_ankle_.*": 37.0,
                    ".*_shoulder_.*": 37.0,
                    ".*_elbow_joint": 37.0,
                    ".*_wrist_roll_joint": 37.0,
                    ".*_wrist_(pitch|yaw)_joint": 22.0,
                },
                armature=0.0,
            )
        }
        material = RigidBodyMaterialBaseCfg(
            static_friction=0.4,
            dynamic_friction=0.4,
            restitution=0.0,
        )
        self.scene.robot.spawn.physics_material = material
        self.scene.terrain.physics_material = material
        self.sim.physics_material = material
        self.events.physics_material = None
        self.events.mpc_model_contract = EventTermCfg(func=validate_mpc_model_event, mode="startup")
        self.sim.physics.newton_mjwarp.solver_cfg.njmax = 300
        self.sim.physics.newton_mjwarp.solver_cfg.nconmax = 100
        self.observations = ReferenceObservationsCfg()
        # Keep the flat walking rewards and their weights; add imitation terms.
        tracking_rewards = ReferenceRewardsCfg()
        for name, term in vars(tracking_rewards).items():
            setattr(self.rewards, name, term)
        # These baseline selectors name fingers/torso joints absent on the 29-DOF
        # model. Preserve their penalty functions/weights where a joint exists.
        self.rewards.joint_deviation_fingers = None
        self.rewards.joint_deviation_torso.params["asset_cfg"].joint_names = "waist_.*"
        self.rewards.joint_deviation_arms.params["asset_cfg"].joint_names = [".*_shoulder_.*", ".*_elbow_joint"]
        self.commands = ReferenceCommandsCfg()
        self.terminations = ReferenceTerminationsCfg()
        # ReferenceMotionCommand resets both root and joints from the same frame.
        # Fix source friction; postpone disturbances/mass randomization until the
        # basic tracking behavior has been validated in physics.
        self.events.reset_base = None
        self.events.reset_robot_joints = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.actions.joint_pos.scale = 0.5

    def play_mode(self):
        # No pretrained velocity-policy assumptions or extra command fields here.
        self.commands.base_velocity.random_start = False
        self.scene.num_envs = 1


@configclass
class LipmObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        targets = ObsTerm(func=lipm_mdp.lipm_targets)

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    policy = PolicyCfg()


@configclass
class LipmHandsObservationsCfg(LipmObservationsCfg):
    @configclass
    class PolicyCfg(LipmObservationsCfg.PolicyCfg):
        arm_targets = ObsTerm(func=lipm_mdp.arm_targets)

    policy = PolicyCfg()


@configclass
class LipmCommandsCfg:
    base_velocity = LipmMotionCommandCfg(
        motion_directory=str(Path(ISAACLAB_ROOT).parent / "LIP/generated_trajectories/step_020_g1_startup.npz")
    )


@configclass
class G1LipmFlatEnvCfg(G1ReferenceFlatEnvCfg):
    """Task-space LIPM tracking with learned joint control; select presets=lipm."""

    def __post_init__(self):
        super().__post_init__()
        # Calibrated for this matched G1 asset and nominal joint pose: the lowest
        # foot collision surfaces meet z=0 (within floating-point tolerance).
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.78420225)
        self.scene.robot.init_state.lin_vel = (0.0, 0.0, 0.0)
        self.scene.robot.init_state.ang_vel = (0.0, 0.0, 0.0)
        self.commands = LipmCommandsCfg()
        self.observations = LipmObservationsCfg()
        # No joint/root trajectories exist in LIPM NPZs.
        for name in vars(ReferenceRewardsCfg()):
            setattr(self.rewards, name, None)
        self.rewards.lipm_feet = RewTerm(
            func=lipm_mdp.foot_horizontal_position_tracking, weight=2.0, params={"std": 0.08}
        )
        self.rewards.lipm_swing_height = RewTerm(func=lipm_mdp.swing_height_tracking, weight=1.0, params={"std": 0.02})
        self.rewards.lipm_feet_com_relative = RewTerm(
            func=lipm_mdp.foot_com_relative_tracking, weight=1.0, params={"std": 0.12}
        )
        self.rewards.lipm_com_position = RewTerm(func=lipm_mdp.com_position_tracking, weight=1.0, params={"std": 0.10})
        self.rewards.lipm_com_velocity = RewTerm(func=lipm_mdp.com_velocity_tracking, weight=0.5, params={"std": 0.30})
        self.rewards.lipm_support = RewTerm(func=lipm_mdp.support_tracking, weight=0.5)
        self.rewards.lipm_loaded_swing = RewTerm(func=lipm_mdp.loaded_swing_penalty, weight=-1.0)
        self.rewards.lipm_stance_slide = RewTerm(func=lipm_mdp.stance_slide_penalty, weight=-1.0)

        # use the existing joint error reward. But we use specficallly for the waist
        self.rewards.waist_velocity = RewTerm(
            func=mdp.joint_vel_l2,
            weight=-0.005,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_.*"])},
        )
        # Replace the generic link-speed penalty with phase-aligned sole speed.
        self.rewards.feet_slide = None
        # The LIPM clock owns contact timing, and its velocity target is CoM velocity.
        self.rewards.feet_air_time = None
        self.rewards.track_lin_vel_xy_exp = None
        self.terminations.reference_finished = DoneTerm(func=lipm_mdp.reference_finished, time_out=True)
        # No reference joint reset is available: use a deterministic nominal pose.
        defaults = G1FlatEnvCfg()
        self.events.reset_base = defaults.events.reset_base
        self.events.reset_robot_joints = defaults.events.reset_robot_joints
        for range_name in ("pose_range", "velocity_range"):
            ranges = self.events.reset_base.params[range_name]
            self.events.reset_base.params[range_name] = dict.fromkeys(ranges, (0.0, 0.0))
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)

    def play_mode(self):
        self.scene.num_envs = 1


@configclass
class G1LipmHandsFlatEnvCfg(G1LipmFlatEnvCfg):
    """LIPM walking plus offline arm IK targets; select presets=lipm_hands.

    Generate the augmented NPZ with LIP/LIPM/export_hand_ik_reference.py first.
    Episodes start at the nominal pose; train with agent.init_at_random_ep_len=false.
    The arm terms fail at observation initialization if the NPZ has no arm targets.
    """

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.motion_directory = str(
            Path(ISAACLAB_ROOT).parent / "LIP/generated_trajectories/steps_010_020_030_hands.npz"
        )
        self.observations = LipmHandsObservationsCfg()
        self.rewards.lipm_arm_position = RewTerm(
            func=lipm_mdp.arm_joint_position_tracking, weight=1.0, params={"std": 0.3}
        )
        self.rewards.lipm_arm_velocity = RewTerm(
            func=lipm_mdp.arm_joint_velocity_tracking, weight=0.1, params={"std": 1.0}
        )
        self.rewards.lipm_left_hand_position = RewTerm(
            func=lipm_mdp.hand_position_tracking, weight=0.5, params={"side": "left", "std": 0.10}
        )
        self.rewards.lipm_right_hand_position = RewTerm(
            func=lipm_mdp.hand_position_tracking, weight=0.5, params={"side": "right", "std": 0.10}
        )
        # Retain the original penalty's weight and joint selection, but exclude
        # referenced arm joints so nominal posture does not compete with reaching.
        self.rewards.joint_deviation_arms.func = lipm_mdp.unreferenced_joint_deviation_l1


@configclass
class LipmOnlineHandsObservationsCfg(LipmObservationsCfg):
    @configclass
    class PolicyCfg(LipmObservationsCfg.PolicyCfg):
        hand_targets_and_errors = ObsTerm(func=hand_commands.hand_targets_and_errors)
        # Enabled during a fresh residual-policy transfer; legacy 127-input policies stay compatible.
        ik_joint_errors = None
        heading_feedback = None

    policy = PolicyCfg()


@configclass
class LipmOnlineHandsCommandsCfg(LipmCommandsCfg):
    hand_position = hand_commands.HandPositionCommandCfg(
        initial_reach_fraction=0.1,
        reach_fraction=1.0,
        curriculum_steps=10000,
    )


@configclass
class G1LipmOnlineHandsFlatEnvCfg(G1LipmFlatEnvCfg):
    """Progressive forward hand reaches, zero hand rewards, and initially disabled online IK.

    Appends 12 observations to the walking policy's 115. Checkpoint input-layer
    transfer is required before training; ordinary resume is not compatible.
    Hand goals require no arm-augmented NPZ. Increase action ik_influence gradually
    to enable online IK. Hand reach distance ramps from 10% to 100% over 10000
    control steps; each episode holds its sampled difficulty until reset.
    """

    def __post_init__(self):
        super().__post_init__()
        # Preserve every original joint-action option and its ordering.
        original_action = self.actions.joint_pos
        self.actions.joint_pos = HandIKJointPositionActionCfg(
            **{
                field.name: getattr(original_action, field.name)
                for field in fields(original_action)
                if field.init and field.name != "class_type"
            }
        )
        # Arms are owned by online IK; nominal arm posture must not oppose reaching.
        self.rewards.joint_deviation_arms.weight = 0.0
        self.rewards.online_heading = RewTerm(func=lipm_mdp.heading_tracking, weight=2.0)
        self.rewards.online_torso_heading = RewTerm(
            func=lipm_mdp.heading_tracking, weight=1.0, params={"body_name": "torso_link"}
        )
        # Enabled explicitly by the fine-tuning launcher; legacy saved policies retain zero weight.
        self.rewards.online_torso_upright = RewTerm(
            func=lipm_mdp.torso_upright_tracking, weight=0.0, params={"tolerance": math.radians(8)}
        )
        self.rewards.online_torso_yaw_motion = RewTerm(func=lipm_mdp.torso_yaw_motion_penalty, weight=-0.05)
        self.rewards.online_waist_yaw_motion = RewTerm(func=lipm_mdp.waist_yaw_motion_penalty, weight=-0.02)
        self.rewards.online_arm_joint_error = RewTerm(func=hand_commands.ik_joint_error_penalty, weight=0.0)
        self.commands = LipmOnlineHandsCommandsCfg()
        self.observations = LipmOnlineHandsObservationsCfg()
        self.rewards.online_left_hand_position = RewTerm(
            func=hand_commands.hand_position_tracking, weight=1.0, params={"side": "left", "std": 0.20}
        )
        self.rewards.online_right_hand_position = RewTerm(
            func=hand_commands.hand_position_tracking, weight=1.0, params={"side": "right", "std": 0.20}
        )
        self.rewards.online_left_hand_position_penalty = RewTerm(
            func=hand_commands.hand_position_penalty, weight=-1.0, params={"side": "left"}
        )
        self.rewards.online_right_hand_position_penalty = RewTerm(
            func=hand_commands.hand_position_penalty, weight=-1.0, params={"side": "right"}
        )
        self.rewards.online_left_hand_orientation = RewTerm(
            func=hand_commands.hand_orientation_tracking, weight=1.0, params={"side": "left", "std": 0.5}
        )
        self.rewards.online_right_hand_orientation = RewTerm(
            func=hand_commands.hand_orientation_tracking, weight=1.0, params={"side": "right", "std": 0.5}
        )
        self.rewards.online_left_hand_orientation_penalty = RewTerm(
            func=hand_commands.hand_orientation_penalty, weight=-0.5, params={"side": "left"}
        )
        self.rewards.online_right_hand_orientation_penalty = RewTerm(
            func=hand_commands.hand_orientation_penalty, weight=-0.5, params={"side": "right"}
        )
        self.rewards.online_left_hand_linear_velocity = RewTerm(
            func=hand_commands.hand_linear_velocity_tracking, weight=0.5, params={"side": "left", "std": 0.1}
        )
        self.rewards.online_right_hand_linear_velocity = RewTerm(
            func=hand_commands.hand_linear_velocity_tracking, weight=0.5, params={"side": "right", "std": 0.1}
        )


@configclass
class G1FlatEnvModes(PresetCfg):
    """Select a reference mode with presets while keeping the task ID/default."""

    lipm_online_hands = G1LipmOnlineHandsFlatEnvCfg()
    lipm_hands = G1LipmHandsFlatEnvCfg()
    lipm = G1LipmFlatEnvCfg()
    reference = G1ReferenceFlatEnvCfg()
    default = G1FlatEnvCfg()
