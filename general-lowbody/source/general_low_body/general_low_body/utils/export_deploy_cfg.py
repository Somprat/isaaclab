import numpy as np
import os
import yaml

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import class_to_dict
from isaaclab.utils.string import resolve_matching_names
from general_low_body.assets import G1BodyCfg

def format_value(x):
    if isinstance(x, float):
        return float(f"{x:.3g}")
    elif isinstance(x, list):
        return [format_value(i) for i in x]
    elif isinstance(x, dict):
        return {k: format_value(v) for k, v in x.items()}
    else:
        return x



def export_mujoco_cfg(env: ManagerBasedRLEnv, log_dir):
    asset: Articulation = env.scene["robot"]
    

    cfg = {}  # noqa: SIM904
    cfg['policy_joints'] = asset.data.joint_names
    cfg['policy_lower_body_joints'] = G1BodyCfg.LOWER_BODY_JOINTS
    cfg['policy_upper_body_joints'] = G1BodyCfg.UPPER_BODY_JOINTS
    cfg['num_actions'] = G1BodyCfg.NUM_ACTIONS
    cfg['num_upper_body_actions'] = len(G1BodyCfg.UPPER_BODY_JOINTS)
    cfg['num_lower_body_actions'] = len(G1BodyCfg.LOWER_BODY_JOINTS)

    # joint ids mapping
    joint_sdk_names = env.cfg.scene.robot.joint_sdk_names
    whole_body_joint2motor_idx, _ = resolve_matching_names(asset.data.joint_names, joint_sdk_names, preserve_order=True)
    upper_body_joint2motor_idx, _ = resolve_matching_names(G1BodyCfg.UPPER_BODY_JOINTS, joint_sdk_names, preserve_order=True)
    lower_body_joint2motor_idx, _ = resolve_matching_names(G1BodyCfg.LOWER_BODY_JOINTS, joint_sdk_names, preserve_order=True)


    # joint default pos
    whole_body_default_pos = asset.data.default_joint_pos[0].detach().cpu().numpy().tolist()
    upper_body_idx = [asset.data.joint_names.index(joint_name) for joint_name in G1BodyCfg.UPPER_BODY_JOINTS]
    lower_body_idx = [asset.data.joint_names.index(joint_name) for joint_name in G1BodyCfg.LOWER_BODY_JOINTS]
    
    # joint kp
    stiffness = asset.data.default_joint_stiffness[0].detach().cpu().numpy().tolist()
    
    # joint kd
    damping = asset.data.default_joint_damping[0].detach().cpu().numpy().tolist()
    

    # write to file
    cfg['whole_body_joint2motor_idx'] = whole_body_joint2motor_idx
    cfg['whole_body_default_pos'] = whole_body_default_pos

    cfg['upper_body_joint2motor_idx'] = upper_body_joint2motor_idx
    cfg['upper_body_kps'] = [stiffness[i] for i in upper_body_idx]
    cfg['upper_body_kds'] = [damping[i] for i in upper_body_idx]
    cfg['upper_body_default_pos'] = [whole_body_default_pos[i] for i in upper_body_idx]


    cfg['lower_body_joint2motor_idx'] = lower_body_joint2motor_idx
    cfg['lower_body_kps'] = [stiffness[i] for i in lower_body_idx]
    cfg['lower_body_kds'] = [damping[i] for i in lower_body_idx]
    cfg['lower_body_default_pos'] = [whole_body_default_pos[i] for i in lower_body_idx]

    # --- actions ---
    action_names = env.action_manager.active_terms
    action_terms = zip(action_names, env.action_manager._terms.values())
    cfg["actions"] = {}
    for action_name, action_term in action_terms:
        # skip upper body action
        if action_name == "UpperBodyAction":
            continue
        term_cfg = action_term.cfg.copy()
        if isinstance(term_cfg.scale, float):
            term_cfg.scale = [term_cfg.scale for _ in range(action_term.action_dim)]
        else:  # dict
            term_cfg.scale = action_term._scale[0].detach().cpu().numpy().tolist()

        if term_cfg.clip is not None:
            term_cfg.clip = action_term._clip[0].detach().cpu().numpy().tolist()

        if action_name in ["JointPositionAction", "JointVelocityAction"]:
            if term_cfg.use_default_offset:
                term_cfg.offset = action_term._offset[0].detach().cpu().numpy().tolist()
            else:
                term_cfg.offset = [0.0 for _ in range(action_term.action_dim)]

        # clean cfg
        term_cfg = term_cfg.to_dict()

        for _ in ["class_type", "asset_name", "debug_vis"]:
            del term_cfg[_]
        cfg["actions"][action_name] = term_cfg

        if action_term._joint_ids == slice(None):
            cfg["actions"][action_name]["joint_ids"] = None
        else:
            cfg["actions"][action_name]["joint_ids"] = action_term._joint_ids

        if action_name == "UpperBodyAction":
            cfg['upper_body_action_scale'] = term_cfg['scale']
        elif action_name == "LowerBodyAction":
            cfg['lower_body_action_scale'] = term_cfg['scale']

    # save config file
    filename = os.path.join(log_dir, "params", "mujoco.yaml")
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    if not isinstance(cfg, dict):
        cfg = class_to_dict(cfg)
    cfg = format_value(cfg)
    with open(filename, "w") as f:
        yaml.dump(cfg, f, default_flow_style=None, sort_keys=False)

    
    