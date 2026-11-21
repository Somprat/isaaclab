from isaaclab.utils import configclass
from general_low_body import mdp
from general_low_body.assets import G1BodyCfg, G1_MIMIC_ACTION_SCALE
import re
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg




MIMIC_UPPER_BODY_ACTION_SCALE = {}
MIMIC_LOWER_BODY_ACTION_SCALE = {}
for joint_name in G1BodyCfg.UPPER_BODY_JOINTS:
    for regex_key, value in G1_MIMIC_ACTION_SCALE.items():
        if re.fullmatch(regex_key, joint_name):
            MIMIC_UPPER_BODY_ACTION_SCALE[joint_name] = value

for joint_name in G1BodyCfg.LOWER_BODY_JOINTS:
    for regex_key, value in G1_MIMIC_ACTION_SCALE.items():
        if re.fullmatch(regex_key, joint_name):
            MIMIC_LOWER_BODY_ACTION_SCALE[joint_name] = value

UPPER_BODY_ACTION_SCALE = 0.25
LOWER_BODY_ACTION_SCALE = 0.25

@configclass
class G1ActionsCfg:
    """Action specifications for the MDP."""

    UpperBodyAction = mdp.RandomJointPosToLimitsActionCfg(
        asset_name="robot", 
        joint_names=G1BodyCfg.UPPER_BODY_JOINTS, 
        preserve_order=True
    )

    LowerBodyAction = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=G1BodyCfg.LOWER_BODY_JOINTS, 
        scale=LOWER_BODY_ACTION_SCALE, 
        use_default_offset=True,
        preserve_order=True,
        clip={".*": (-100.0, 100.0)}
    )


@configclass
class G1RandomUpperActionsCfg:
    UpperBodyAction = mdp.RandomJointPosToLimitsActionCfg(
        asset_name="robot", 
        joint_names=G1BodyCfg.UPPER_BODY_JOINTS, 
        preserve_order=True
    )

    LowerBodyAction = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=G1BodyCfg.LOWER_BODY_JOINTS, 
        scale=MIMIC_LOWER_BODY_ACTION_SCALE, 
        use_default_offset=True,
        preserve_order=True,
        clip={".*": (-100.0, 100.0)}
    )

@configclass
class G1LeftArmIKCfg:
    LeftArmAction = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=G1BodyCfg.LEFT_ARM_JOINTS,
        body_name=G1BodyCfg.LEFT_EE_NAME,
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        scale=1.0,
    )
    RightArmAction = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=G1BodyCfg.RIGHT_ARM_JOINTS, 
        scale=UPPER_BODY_ACTION_SCALE, 
        use_default_offset=True,
        preserve_order=True,
        clip={".*": (-100.0, 100.0)}
    )
    LowerBodyAction = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=G1BodyCfg.LOWER_BODY_JOINTS, 
        scale=MIMIC_LOWER_BODY_ACTION_SCALE, 
        use_default_offset=True,
        preserve_order=True,
        clip={".*": (-100.0, 100.0)}
    )

@configclass
class G1LeftArmDLSIKCfg(G1LeftArmIKCfg):
    LeftArmAction = mdp.G1DLSIKActionCfg(
        asset_name="robot",
        mode="left_arm",
        joint_names=G1BodyCfg.LEFT_ARM_JOINTS,
        body_names=[G1BodyCfg.LEFT_EE_NAME],
        # mjlab defaults. Tune if tracking is too soft or jittery.
        damping=0.02,
        max_dq=1.0,
        position_weight=2.0,
        orientation_weight=0.5,
        joint_limit_weight=1.0,
        posture_weight=0.01,
    )
    RightArmAction = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=G1BodyCfg.RIGHT_ARM_JOINTS, 
        scale=UPPER_BODY_ACTION_SCALE, 
        use_default_offset=True,
        preserve_order=True,
        clip={".*": (-100.0, 100.0)}
    )
    LowerBodyAction = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=G1BodyCfg.LOWER_BODY_JOINTS, 
        scale=MIMIC_LOWER_BODY_ACTION_SCALE, 
        use_default_offset=True,
        preserve_order=True,
        clip={".*": (-100.0, 100.0)}
    )


@configclass
class G1LeftArmCuroboIKCfg:
    """Left-arm cuRobo IK + right arm / lower body joint position. PLAY/training-capable.

    Same 7-D pose action layout as ``G1LeftArmIKCfg`` and ``G1LeftArmPinkIKCfg`` —
    a policy trained with diff-IK can swap in here directly.

    cuRobo runs on GPU with batched cuda-graph IK, so unlike Pink IK this scales
    to num_envs=8192 in principle (though we have not benchmarked training-rate
    throughput yet).
    """

    LeftArmAction = mdp.G1CuroboIKActionCfg(
        asset_name="robot",
        mode="left_arm",
        joint_names=G1BodyCfg.LEFT_ARM_JOINTS,
        body_names=[G1BodyCfg.LEFT_EE_NAME],
        num_seeds=1,
        debug_log_every=0,
        health_log_every=0,
        use_cuda_graph=False,
    )

    RightArmAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=G1BodyCfg.RIGHT_ARM_JOINTS,
        scale=UPPER_BODY_ACTION_SCALE,
        use_default_offset=True,
        preserve_order=True,
        clip={".*": (-100.0, 100.0)},
    )
    LowerBodyAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=G1BodyCfg.LOWER_BODY_JOINTS,
        scale=MIMIC_LOWER_BODY_ACTION_SCALE,
        use_default_offset=True,
        preserve_order=True,
        clip={".*": (-100.0, 100.0)},
    )

@configclass
class G1DualArmCuroboIKCfg:
    """Both arms driven by a single cuRobo IK solver call + lower body joint pos.

    Action layout (14-D arm slice, left then right):
        ``[left_pos(3), left_quat(4), right_pos(3), right_quat(4)]`` in pelvis frame.
    Lower body slice stays 15-D joint position. Total policy ``action_dim = 29``.

    The env step is expected to read ``UpperBodyAction.mode == "both_arm"`` and
    stitch the policy's arm/lower slices accordingly (instead of the left/right
    split used by the single-arm cfgs above).

    Single-solver dual-arm IK has lower fixed overhead than two parallel
    solvers and shares one CUDA graph / one set of sync barriers.
    """

    UpperBodyAction = mdp.G1CuroboIKActionCfg(
        asset_name="robot",
        mode="both_arm",
        joint_names=G1BodyCfg.LEFT_ARM_JOINTS + G1BodyCfg.RIGHT_ARM_JOINTS,
        body_names=[G1BodyCfg.LEFT_EE_NAME, G1BodyCfg.RIGHT_EE_NAME],
        num_seeds=1,
        debug_log_every=0,
        health_log_every=500,
        use_cuda_graph=False,
    )
    LowerBodyAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=G1BodyCfg.LOWER_BODY_JOINTS,
        scale=MIMIC_LOWER_BODY_ACTION_SCALE,
        use_default_offset=True,
        preserve_order=True,
        clip={".*": (-100.0, 100.0)},
    )


@configclass
class G1DualArmDLSIKCfg:
    """Dual-arm damped-least-squares IK + lower body joint position. Drop-in
    replacement for ``G1DualArmCuroboIKCfg`` — same 14-D arm slice layout,
    same env step path — but the IK is a single-step DLS instead of cuRobo's
    LBFGS, so it doesn't carry cuRobo's long-run CUDA-stability tax.

    Three DLS-specific mitigations vs. Isaac Lab's stock diff-IK
    (see ``G1DLSIKActionCfg`` docstring):

    - ``max_dq=0.5`` clamps per-step joint delta → no whipping when
      the impedance reference is unreachable.
    - ``joint_limit_weight`` is a soft penalty inside the solver →
      arms slide into joint limits instead of bouncing off them.
    - ``posture_weight + posture_target`` regularises the 14-DOF
      null-space → no drift between symmetric IK branches.
    """

    UpperBodyAction = mdp.G1DLSIKActionCfg(
        asset_name="robot",
        mode="both_arm",
        joint_names=G1BodyCfg.LEFT_ARM_JOINTS + G1BodyCfg.RIGHT_ARM_JOINTS,
        body_names=[G1BodyCfg.LEFT_EE_NAME, G1BodyCfg.RIGHT_EE_NAME],
        # mjlab defaults. Tune if tracking is too soft or jittery.
        damping=0.01,
        adaptive_damping=0.3,
        max_dq=1.0,
        position_weight=2.0,
        orientation_weight=0.5,
        joint_limit_weight=1.0,
        posture_weight=0.01,
    )
    LowerBodyAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=G1BodyCfg.LOWER_BODY_JOINTS,
        scale=LOWER_BODY_ACTION_SCALE,
        use_default_offset=True,
        preserve_order=True,
        clip={".*": (-100.0, 100.0)},
    )