from isaaclab.utils import configclass
from general_low_body import mdp
from general_low_body.assets import G1BodyCfg
import math

@configclass
class LocomotionCommandsCfg:
    """Command specifications for the MDP."""
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 8.0),
        rel_standing_envs=0.05,
        rel_heading_envs=0.25,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        delay_steps=0,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.1, 0.1), heading=(-math.pi, math.pi),
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.5, 0.5), heading=(-math.pi, math.pi),
        ),
    )


@configclass
class LeftArmIKCommandCfg:
    # base_velocity = mdp.UniformLevelYawFrameVelocityCommandCfg(
    #     asset_name="robot",
    #     resampling_time_range=(3.0, 8.0),
    #     rel_standing_envs=0.05,
    #     rel_heading_envs=0.25,
    #     heading_command=True,
    #     heading_control_stiffness=0.5,
    #     debug_vis=True,
    #     delay_steps=0,
    #     ranges=mdp.UniformLevelYawFrameVelocityCommandCfg.Ranges(
    #         lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.1, 0.1), heading=(-math.pi, math.pi),
    #     ),
    #     limit_ranges=mdp.UniformLevelYawFrameVelocityCommandCfg.Ranges(
    #         lin_vel_x=(-0.5, 1.0), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.5, 0.5), heading=(-math.pi, math.pi),
    #     ),
    # )

    # left_ee_position = mdp.EndEffectorPositionCommandCfg(
    #     robot_asset_name="robot",
    #     end_effector_name=G1BodyCfg.LEFT_EE_NAME,
    #     resampling_time_range=(3.0, 10.0),
    #     debug_vis=True
    # )

    left_ee_position = mdp.ImpedanceCommandCfg(
        asset_name="robot",
        link_name=G1BodyCfg.LEFT_EE_NAME,
        resampling_time_range=(3.0, 10.0),
        debug_vis=True,
        substep_update=True,
        history_length=32,
        ref_time_step=32,
        force_saturate=40,
        target=mdp.ImpedanceCommandCfg.TargetCfg(
            fixed_base_z=G1BodyCfg.TARGET_BASE_HEIGHT,
            roll_range=(-0.3, 0.3),
            pitch_range=(-0.3, 0.3),    
            yaw_range=(-0.5, 0.5),  
        ),
        constant_force=mdp.ImpedanceCommandCfg.ConstantForceCfg(
            disable=False,
            duration_range=(1.0, 4.0),
            force_offsets=(0.0, 0.0, 0.0),
            force_scales = (5, 5, 5),
            resample_probability=0.2,
            resample_env_step=10,
        ),
        impulse_force=mdp.ImpedanceCommandCfg.ImpulseForceCfg(
            disable=False,
            duration_range=(0.2, 0.5),
            x_range = (1, 5),
            y_range = (1, 5),
            z_range = (-1, 1),
            resample_probability=0.1,
            resample_env_step=20,
        ),
        compliant_parameters=mdp.ImpedanceCommandCfg.CompliantParametersCfg(
            linear_kp_range=(12.0, 24.0),
            angular_kp_range=(12.0, 24.0),
            virtual_mass_range = (1.0, 1.0, 1.0),
            virtual_inertia_range=(1.0, 1.0),
            integrate_method="open_loop",
            velocity_norm_clamp=0.5,
            omega_norm_clamp=1.0,
        )
    )


@configclass
class DualArmIKCommandCfg:
    left_ee_position = mdp.ImpedanceCommandCfg(
        asset_name="robot",
        link_name=G1BodyCfg.LEFT_EE_NAME,
        resampling_time_range=(3.0, 10.0),
        debug_vis=True,
        substep_update=True,
        history_length=32,
        ref_time_step=32,
        force_saturate=40,
        target=mdp.ImpedanceCommandCfg.TargetCfg(
            fixed_base_z=G1BodyCfg.TARGET_BASE_HEIGHT,
            roll_range=(-0.3, 0.3),
            pitch_range=(-0.3, 0.3),    
            yaw_range=(-0.5, 0.5),  
        ),
        constant_force=mdp.ImpedanceCommandCfg.ConstantForceCfg(
            disable=False,
            duration_range=(1.0, 4.0),
            force_offsets=(0.0, 0.0, 0.0),
            force_scales = (5, 5, 5),
            resample_probability=0.2,
            resample_env_step=10,
        ),
        impulse_force=mdp.ImpedanceCommandCfg.ImpulseForceCfg(
            disable=False,
            duration_range=(0.2, 0.5),
            x_range = (1, 5),
            y_range = (1, 5),
            z_range = (-1, 1),
            resample_probability=0.1,
            resample_env_step=20,
        ),
        compliant_parameters=mdp.ImpedanceCommandCfg.CompliantParametersCfg(
            linear_kp_range=(12.0, 24.0),
            angular_kp_range=(12.0, 24.0),
            virtual_mass_range = (1.0, 1.0, 1.0),
            virtual_inertia_range=(1.0, 1.0),
            integrate_method="open_loop",
            velocity_norm_clamp=0.5,
            omega_norm_clamp=1.0,
        )
    )

    right_ee_position = left_ee_position.replace(link_name=G1BodyCfg.RIGHT_EE_NAME)