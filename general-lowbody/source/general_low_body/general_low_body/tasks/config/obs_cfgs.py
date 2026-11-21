from isaaclab.utils import configclass
from general_low_body import mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from general_low_body.utils import *
from isaaclab.managers import SceneEntityCfg

@configclass
class RandomUpperBodyActionObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2), clip=(-100, 100))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05), clip=(-100, 100))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"}, clip=(-100, 100))
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01), clip=(-100, 100))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5), clip=(-100, 100))
        upper_body_action = ObsTerm(func=mdp.last_action, params={"action_name": "UpperBodyAction"}, clip=(-100, 100))
        lower_body_action = ObsTerm(func=mdp.last_action, params={"action_name": "LowerBodyAction"}, clip=(-100, 100))

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"}, clip=(-100, 100))
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100))
        upper_body_action = ObsTerm(func=mdp.last_action, params={"action_name": "UpperBodyAction"}, clip=(-100, 100))
        lower_body_action = ObsTerm(func=mdp.last_action, params={"action_name": "LowerBodyAction"}, clip=(-100, 100))
        def __post_init__(self):
            self.history_length = 5

    # privileged observations
    critic: CriticCfg = CriticCfg()



@configclass
class LeftArmIKObservationsCfg:
    """Observation specifications for the MDP."""
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2), clip=(-100, 100))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05), clip=(-100, 100))
        # velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"}, clip=(-100, 100))
        left_ee_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "left_ee_position"}, clip=(-100, 100))
        left_ee_ref_pos = ObsTerm(func=mdp.impedance_ref_pos_b, params={"command_name": "left_ee_position"}, clip=(-100, 100))
        left_ee_ref_vel = ObsTerm(func=mdp.impedance_ref_vel_b, params={"command_name": "left_ee_position"}, clip=(-100, 100))
        left_ee_ref_quat = ObsTerm(func=mdp.impedance_ref_quat_b, params={"command_name": "left_ee_position"}, clip=(-100, 100))
        left_ee_ref_ang_vel = ObsTerm(func=mdp.impedance_ref_ang_vel_b, params={"command_name": "left_ee_position"}, clip=(-100, 100))
        left_ee_external_force = ObsTerm(func=mdp.impedance_external_force_b, params={"command_name": "left_ee_position"}, clip=(-100, 100))
        left_ee_impedance_parameters = ObsTerm(func=mdp.impedance_parameters, params={"command_name": "left_ee_position"}, clip=(-100, 100))
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01), clip=(-100, 100))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5), clip=(-100, 100))
        lower_body_action = ObsTerm(func=mdp.last_action, params={"action_name": "LowerBodyAction"}, clip=(-100, 100))

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100))
        #velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"}, clip=(-100, 100))
        left_ee_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "left_ee_position"}, clip=(-100, 100))
        left_ee_ref_pos = ObsTerm(func=mdp.impedance_ref_pos_b, params={"command_name": "left_ee_position"}, clip=(-100, 100))
        left_ee_ref_vel = ObsTerm(func=mdp.impedance_ref_vel_b, params={"command_name": "left_ee_position"}, clip=(-100, 100))
        left_ee_ref_quat = ObsTerm(func=mdp.impedance_ref_quat_b, params={"command_name": "left_ee_position"}, clip=(-100, 100))
        left_ee_ref_ang_vel = ObsTerm(func=mdp.impedance_ref_ang_vel_b, params={"command_name": "left_ee_position"}, clip=(-100, 100))
        left_ee_external_force = ObsTerm(func=mdp.impedance_external_force_b, params={"command_name": "left_ee_position"}, clip=(-100, 100))
        left_ee_impedance_parameters = ObsTerm(func=mdp.impedance_parameters, params={"command_name": "left_ee_position"}, clip=(-100, 100))
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100))
        lower_body_action = ObsTerm(func=mdp.last_action, params={"action_name": "LowerBodyAction"}, clip=(-100, 100))
        def __post_init__(self):
            self.history_length = 5

    # privileged observations
    critic: CriticCfg = CriticCfg()



@configclass
class DualArmIKObservationsCfg:
    """Observation specifications for the MDP."""
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2), clip=(-100, 100))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05), clip=(-100, 100))
        # velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"}, clip=(-100, 100))
        left_ee_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "left_ee_position"}, clip=(-100, 100))
        left_ee_ref_pos = ObsTerm(func=mdp.impedance_ref_pos_b, params={"command_name": "left_ee_position"}, clip=(-100, 100))
        left_ee_ref_vel = ObsTerm(func=mdp.impedance_ref_vel_b, params={"command_name": "left_ee_position"}, clip=(-100, 100))
        left_ee_ref_quat = ObsTerm(func=mdp.impedance_ref_quat_b, params={"command_name": "left_ee_position"}, clip=(-100, 100))
        left_ee_ref_ang_vel = ObsTerm(func=mdp.impedance_ref_ang_vel_b, params={"command_name": "left_ee_position"}, clip=(-100, 100))
        left_ee_external_force = ObsTerm(func=mdp.impedance_external_force_b, params={"command_name": "left_ee_position"}, clip=(-100, 100))
        left_ee_impedance_parameters = ObsTerm(func=mdp.impedance_parameters, params={"command_name": "left_ee_position"}, clip=(-100, 100))

        right_ee_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "right_ee_position"}, clip=(-100, 100))
        right_ee_ref_pos = ObsTerm(func=mdp.impedance_ref_pos_b, params={"command_name": "right_ee_position"}, clip=(-100, 100))
        right_ee_ref_vel = ObsTerm(func=mdp.impedance_ref_vel_b, params={"command_name": "right_ee_position"}, clip=(-100, 100))
        right_ee_ref_quat = ObsTerm(func=mdp.impedance_ref_quat_b, params={"command_name": "right_ee_position"}, clip=(-100, 100))
        right_ee_ref_ang_vel = ObsTerm(func=mdp.impedance_ref_ang_vel_b, params={"command_name": "right_ee_position"}, clip=(-100, 100))
        right_ee_external_force = ObsTerm(func=mdp.impedance_external_force_b, params={"command_name": "right_ee_position"}, clip=(-100, 100))
        right_ee_impedance_parameters = ObsTerm(func=mdp.impedance_parameters, params={"command_name": "right_ee_position"}, clip=(-100, 100))

        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01), clip=(-100, 100))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5), clip=(-100, 100))
        lower_body_action = ObsTerm(func=mdp.last_action, params={"action_name": "LowerBodyAction"}, clip=(-100, 100))

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100))

        left_ee_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "left_ee_position"}, clip=(-100, 100))
        left_ee_ref_pos = ObsTerm(func=mdp.impedance_ref_pos_b, params={"command_name": "left_ee_position"}, clip=(-100, 100))
        left_ee_ref_vel = ObsTerm(func=mdp.impedance_ref_vel_b, params={"command_name": "left_ee_position"}, clip=(-100, 100))
        left_ee_ref_quat = ObsTerm(func=mdp.impedance_ref_quat_b, params={"command_name": "left_ee_position"}, clip=(-100, 100))
        left_ee_ref_ang_vel = ObsTerm(func=mdp.impedance_ref_ang_vel_b, params={"command_name": "left_ee_position"}, clip=(-100, 100))
        left_ee_external_force = ObsTerm(func=mdp.impedance_external_force_b, params={"command_name": "left_ee_position"}, clip=(-100, 100))
        left_ee_impedance_parameters = ObsTerm(func=mdp.impedance_parameters, params={"command_name": "left_ee_position"}, clip=(-100, 100))

        right_ee_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "right_ee_position"}, clip=(-100, 100))
        right_ee_ref_pos = ObsTerm(func=mdp.impedance_ref_pos_b, params={"command_name": "right_ee_position"}, clip=(-100, 100))
        right_ee_ref_vel = ObsTerm(func=mdp.impedance_ref_vel_b, params={"command_name": "right_ee_position"}, clip=(-100, 100))
        right_ee_ref_quat = ObsTerm(func=mdp.impedance_ref_quat_b, params={"command_name": "right_ee_position"}, clip=(-100, 100))
        right_ee_ref_ang_vel = ObsTerm(func=mdp.impedance_ref_ang_vel_b, params={"command_name": "right_ee_position"}, clip=(-100, 100))
        right_ee_external_force = ObsTerm(func=mdp.impedance_external_force_b, params={"command_name": "right_ee_position"}, clip=(-100, 100))
        right_ee_impedance_parameters = ObsTerm(func=mdp.impedance_parameters, params={"command_name": "right_ee_position"}, clip=(-100, 100))

        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100))
        lower_body_action = ObsTerm(func=mdp.last_action, params={"action_name": "LowerBodyAction"}, clip=(-100, 100))
        def __post_init__(self):
            self.history_length = 5

    # privileged observations
    critic: CriticCfg = CriticCfg()