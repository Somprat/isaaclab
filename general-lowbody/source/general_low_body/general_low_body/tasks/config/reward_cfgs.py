from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewardTerm
from general_low_body import mdp
from isaaclab.managers import SceneEntityCfg
import math
from general_low_body.assets import G1BodyCfg


@configclass
class LocomotionRewardCfg:
    
    # track_lin_vel_xy = RewardTerm(
    #     func=mdp.track_lin_vel_xy_yaw_frame_exp,
    #     weight=1.0,
    #     params={
    #         "command_name": "base_velocity", 
    #         "std": math.sqrt(0.25),
    #     },
    # )
    # track_ang_vel_z = RewardTerm(
    #     func=mdp.track_ang_vel_z_world_exp, 
    #     weight=1.0, 
    #     params={
    #         "command_name": "base_velocity", 
    #         "std": math.sqrt(0.25),
    #     }
    # )
    penalty_lin_vel_z = RewardTerm(
        func=mdp.lin_vel_z_l2,
        weight=-2.0,
    )
    penalty_ang_vel_xy = RewardTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.05,
    )
    penalty_robot_flat_orientation = RewardTerm(
        func=mdp.flat_orientation_l2,
        weight=-5.0,
    )
    penalty_dof_pos_waist = RewardTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["waist.*"]),
        }
    )
    penalty_dof_pos_legs = RewardTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )
    penalty_lower_body_dof_pos_limits = RewardTerm(
        func=mdp.joint_pos_limits,
        weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=G1BodyCfg.LOWER_BODY_JOINTS),
        }
    )
    penalty_lower_body_dof_acc = RewardTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=G1BodyCfg.LOWER_BODY_JOINTS),
        }
    )
    penalty_lower_body_dof_vel = RewardTerm(
        func=mdp.joint_vel_l2,
        weight=-0.001,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=G1BodyCfg.LOWER_BODY_JOINTS),
        }
    )
    penalty_lower_body_dof_torque = RewardTerm(
        func=mdp.joint_torques_l2,
        weight=-2e-5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=G1BodyCfg.LOWER_BODY_JOINTS),
        }
    )
    penalty_lower_body_action_rate = RewardTerm(
        func=mdp.lower_body_action_rate,
        weight=-0.05,
    )
    penalty_robot_base_height = RewardTerm(
        func=mdp.base_height_l2,
        weight=-10,
        params={
            "target_height": G1BodyCfg.TARGET_BASE_HEIGHT,
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        }
    )
    feet_slide = RewardTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("robot_contact_sensor", body_names=".*ankle_roll.*"),
        },
    )
    # feet_clearance = RewardTerm(
    #     func=mdp.foot_clearance_reward,
    #     weight=1.0,
    #     params={
    #         "std": 0.05,
    #         "tanh_mult": 2.0,
    #         "target_height": G1BodyCfg.TARGET_FEET_HEIGHT,
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
    #     },
    # )
    feet_impact = RewardTerm(
        func=mdp.FootImpactL2Reward,
        weight=0.5,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "std": 0.5,
        }
    )
    # feet_soft_landing = RewardTerm(
    #     func=mdp.feet_soft_landing,
    #     weight=-1e-3,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("robot_contact_sensor", body_names=".*ankle_roll.*"),
    #         "command_name": "base_velocity",
    #         "command_threshold": 0.1,
    #     },
    # )
    lower_body_undesired_contacts = RewardTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg(
                "robot_contact_sensor", 
                body_names=G1BodyCfg.LOWER_BODY_LINKS,
            ),
        },
    )
    alive_reward = RewardTerm(
        func=mdp.is_alive,
        weight=0.15,
    )
    
@configclass
class LeftArmIKRewardCfg(LocomotionRewardCfg):

    penalty_robot_base_height = None
    penalty_robot_flat_orientation = None
    penalty_lin_vel_z = None
    penalty_ang_vel_xy = None

    penalty_dof_pos_waist = RewardTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["waist.*"]),
        }
    )

    penalty_left_ee_position = RewardTerm(
        func=mdp.penalty_ee_reference_position_l2,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=G1BodyCfg.LEFT_EE_NAME),
            "command_name": "left_ee_position",
        }
    )

    penalty_left_ee_orientation = RewardTerm(
        func=mdp.penalty_ee_reference_orientation_l2,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=G1BodyCfg.LEFT_EE_NAME),
            "command_name": "left_ee_position",
        }
    )

    reward_left_ee_position = RewardTerm(
        func=mdp.reward_ee_reference_position_exp,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=G1BodyCfg.LEFT_EE_NAME),
            "command_name": "left_ee_position",
            "std": 0.2,
        }
    )

    reward_left_ee_velocity_exp = RewardTerm(
        func=mdp.reward_ee_reference_velocity_exp,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=G1BodyCfg.LEFT_EE_NAME),
            "command_name": "left_ee_position",
            "std": 1.0,
        }
    )

    reward_left_ee_orientation = RewardTerm(
        func=mdp.reward_ee_reference_orientation_exp,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=G1BodyCfg.LEFT_EE_NAME),
            "command_name": "left_ee_position",
            "std": 0.5,
        }
    )

    reward_left_ee_ang_velocity_exp = RewardTerm(
        func=mdp.reward_ee_reference_angular_velocity_exp,
        weight=0.25,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=G1BodyCfg.LEFT_EE_NAME),
            "command_name": "left_ee_position",
            "std": 2.0,
        }
    )


@configclass
class DualArmIKRewardCfg(LocomotionRewardCfg):

    penalty_robot_base_height = None
    penalty_robot_flat_orientation = None
    penalty_lin_vel_z = None
    penalty_ang_vel_xy = None

    penalty_dof_pos_waist = RewardTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["waist.*"]),
        }
    )

    penalty_left_ee_position = RewardTerm(
        func=mdp.penalty_ee_reference_position_l2,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=G1BodyCfg.LEFT_EE_NAME),
            "command_name": "left_ee_position",
        }
    )

    penalty_right_ee_position = RewardTerm(
        func=mdp.penalty_ee_reference_position_l2,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=G1BodyCfg.RIGHT_EE_NAME),
            "command_name": "right_ee_position",
        }
    )

    penalty_left_ee_orientation = RewardTerm(
        func=mdp.penalty_ee_reference_orientation_l2,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=G1BodyCfg.LEFT_EE_NAME),
            "command_name": "left_ee_position",
        }
    )

    penalty_right_ee_orientation = RewardTerm(
        func=mdp.penalty_ee_reference_orientation_l2,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=G1BodyCfg.RIGHT_EE_NAME),
            "command_name": "right_ee_position",
        }
    )

    reward_left_ee_position = RewardTerm(
        func=mdp.reward_ee_reference_position_exp,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=G1BodyCfg.LEFT_EE_NAME),
            "command_name": "left_ee_position",
            "std": 0.2,
        }
    )

    reward_right_ee_position = RewardTerm(
        func=mdp.reward_ee_reference_position_exp,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=G1BodyCfg.RIGHT_EE_NAME),
            "command_name": "right_ee_position",
            "std": 0.2,
        }
    )

    reward_left_ee_velocity_exp = RewardTerm(
        func=mdp.reward_ee_reference_velocity_exp,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=G1BodyCfg.LEFT_EE_NAME),
            "command_name": "left_ee_position",
            "std": 1.0,
        }
    )

    reward_right_ee_velocity_exp = RewardTerm(
        func=mdp.reward_ee_reference_velocity_exp,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=G1BodyCfg.RIGHT_EE_NAME),
            "command_name": "right_ee_position",
            "std": 1.0,
        }
    )

    reward_left_ee_orientation = RewardTerm(
        func=mdp.reward_ee_reference_orientation_exp,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=G1BodyCfg.LEFT_EE_NAME),
            "command_name": "left_ee_position",
            "std": 0.5,
        }
    )

    reward_right_ee_orientation = RewardTerm(
        func=mdp.reward_ee_reference_orientation_exp,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=G1BodyCfg.RIGHT_EE_NAME),
            "command_name": "right_ee_position",
            "std": 0.5,
        }
    )

    reward_left_ee_ang_velocity_exp = RewardTerm(
        func=mdp.reward_ee_reference_angular_velocity_exp,
        weight=0.25,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=G1BodyCfg.LEFT_EE_NAME),
            "command_name": "left_ee_position",
            "std": 2.0,
        }
    )

    reward_right_ee_ang_velocity_exp = RewardTerm(
        func=mdp.reward_ee_reference_angular_velocity_exp,
        weight=0.25,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=G1BodyCfg.RIGHT_EE_NAME),
            "command_name": "right_ee_position",
            "std": 2.0,
        }
    )
