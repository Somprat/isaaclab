import torch
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.pink_task_space_actions import PinkInverseKinematicsAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass




class G1PinkInverseKinematicsAction(PinkInverseKinematicsAction):
    """G1 variant of Isaac Lab Pink IK action with two fixes:

    1. `actions_clone[:, -0:]` slicing bug when num_hand_joints == 0 (Python's `-0 == 0` makes the
       slice grab the whole action tensor, which then gets concatenated to the IK joint output →
       shape mismatch in apply_actions).
    2. Bypass the env_origin → base_link static transformation. ImpedanceCommand emits target poses
       in the CURRENT base (pelvis) frame, not env_origin frame, and the robot moves during play so
       the init-time `base_link_frame_in_env_origin` capture is stale anyway. Pink's pinocchio FK
       treats the URDF root (= pelvis after locking other joints) as the world, which is exactly the
       frame we want — so we override the transform to identity and let the action pass through.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        identity = torch.eye(4, device=self.device).unsqueeze(0).expand(env.num_envs, -1, -1).clone()
        self.base_link_frame_in_env_origin = identity

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        if self.hand_joint_dim == 0:
            self._target_hand_joint_positions = actions.new_zeros(actions.shape[0], 0)


@configclass
class G1PinkInverseKinematicsActionCfg(PinkInverseKinematicsActionCfg):
    class_type: type[ActionTerm] = G1PinkInverseKinematicsAction


def setup_pink_ik_urdf(actions_cfg, scene_cfg, urdf_link_prefix: str = "g1_29dof_rev_1_0_"):
    """Convert robot USD to URDF, lock non-IK joints, set URDF paths on the Pink action term,
    and ensure Pink-controlled joints have entries in ``init_state.joint_pos``
    (PinkIKController requires this; Isaac Lab otherwise defaults missing joints to 0.0).

    Args:
        urdf_link_prefix: USD→URDF conversion can prepend the asset's prim name to link names
            (e.g. ``g1_29dof_rev_1_0_left_wrist_yaw_link``). If non-empty, this prefix is stripped
            from every link element and every parent/child reference so the URDF link names match
            the names used elsewhere in the codebase (``G1BodyCfg.LEFT_EE_NAME`` etc.). Set to ``""``
            to disable.

    Call from env_cfg.__post_init__ AFTER ``super().__post_init__()`` so ``scene.robot`` is configured.
    """
    import tempfile
    import xml.etree.ElementTree as ET
    import isaaclab.controllers.utils as ControllerUtils

    temp_urdf_dir = tempfile.gettempdir()
    urdf_path, mesh_path = ControllerUtils.convert_usd_to_urdf(
        scene_cfg.robot.spawn.usd_path, temp_urdf_dir, force_conversion=True
    )
    ControllerUtils.change_revolute_to_fixed(
        urdf_path, actions_cfg.LeftArmAction.ik_urdf_fixed_joint_names
    )

    if urdf_link_prefix:
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        n_stripped = 0
        for link in root.findall("link"):
            name = link.get("name", "")
            if name.startswith(urdf_link_prefix):
                link.set("name", name[len(urdf_link_prefix):])
                n_stripped += 1
        for joint in root.findall("joint"):
            for tag in ("parent", "child"):
                ref = joint.find(tag)
                if ref is not None:
                    link_attr = ref.get("link", "")
                    if link_attr.startswith(urdf_link_prefix):
                        ref.set("link", link_attr[len(urdf_link_prefix):])
        tree.write(urdf_path)
        print(f"[setup_pink_ik_urdf] stripped prefix '{urdf_link_prefix}' from {n_stripped} link names in {urdf_path}")

    actions_cfg.LeftArmAction.controller.urdf_path = urdf_path
    actions_cfg.LeftArmAction.controller.mesh_path = mesh_path

    # PinkIKController needs every controlled joint to match a key in init_state.joint_pos.
    # Fill in any missing ones with 0.0 (Isaac Lab's default for unspecified joints).
    joint_pos = scene_cfg.robot.init_state.joint_pos
    if joint_pos is None:
        scene_cfg.robot.init_state.joint_pos = joint_pos = {}
    for joint_name in actions_cfg.LeftArmAction.pink_controlled_joint_names:
        joint_pos.setdefault(joint_name, 0.0)