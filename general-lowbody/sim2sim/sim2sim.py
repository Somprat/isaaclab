import time
import argparse
import os
import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml
from collections import deque
from utils.rotation_helper import transform_imu_data_pelvis_to_torso, quat_apply, get_gravity_orientation
from utils.circular_buffer import CircularBuffer
from utils.config_reader import ConfigReader
from scipy.spatial.transform import Rotation
from utils.random_action import RandomJointActionGenerator
import threading
import select
import sys
from pathlib import Path
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).parent.resolve()
CONFIG_PATH = PROJECT_ROOT / 'configs'
ACTUATOR_CONFIG = 'mimic'
CHECKPOINT = PROJECT_ROOT / 'checkpoints' / 'policy.onnx'



def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


def compute_soft_limit(lower_limit, upper_limit, soft_factor=0.9):
    center = (upper_limit + lower_limit) / 2.0
    half_range = (upper_limit - lower_limit) / 2.0
    soft_lower = center - half_range * soft_factor
    soft_upper = center + half_range * soft_factor
    return (soft_lower, soft_upper) 

def check_joint_limit(action, soft_limit):
    lower_violations = action < soft_limit[0]
    upper_violations = action > soft_limit[1]
    
    within_limit = np.all((action >= soft_limit[0]) & (action <= soft_limit[1]))
    if not within_limit:
        violated_indices = np.where(lower_violations | upper_violations)[0]
        
        print(f"Joint limit violations detected:")
        for idx in violated_indices:
            if lower_violations[idx]:
                print(f"  Joint {idx}: {action[idx]:.4f} < lower_limit {soft_limit[0][idx]:.4f} "
                      f"(violation: {action[idx] - soft_limit[0][idx]:.4f})")
            if upper_violations[idx]:
                print(f"  Joint {idx}: {action[idx]:.4f} > upper_limit {soft_limit[1][idx]:.4f} "
                      f"(violation: {action[idx] - soft_limit[1][idx]:.4f})")
    return within_limit 

def apply_action(
    num_actions,
    d,
    target_lower_pos,
    target_upper_pos,
    lower_body_joint2motor_idx,
    upper_body_joint2motor_idx,
    lower_body_kps,
    upper_body_kps,
    lower_body_kds,
    upper_body_kds,
):
    """Apply action to the robot"""
    q_t = d.qpos[7: 7 + num_actions]
    dq_t = d.qvel[6: 6 + num_actions]
    tau = np.zeros(num_actions)
    tau_lower = pd_control(
        target_lower_pos, # in sim
        q_t[lower_body_joint2motor_idx],
        lower_body_kps,
        np.zeros_like(lower_body_kds),
        dq_t[lower_body_joint2motor_idx], 
        lower_body_kds)
    for i in range(len(tau_lower)):
        motor_idx = lower_body_joint2motor_idx[i]
        tau[motor_idx] = tau_lower[i]

    tau_upper = pd_control(
        target_upper_pos, # in sim
        q_t[upper_body_joint2motor_idx],
        upper_body_kps,
        np.zeros_like(upper_body_kds),
        dq_t[upper_body_joint2motor_idx], 
        upper_body_kds)
    for i in range(len(tau_upper)):
        motor_idx = upper_body_joint2motor_idx[i]
        tau[motor_idx] = tau_upper[i]
    return tau

    
if __name__ == "__main__":
    
    # load config
    cfg = ConfigReader.from_yaml(CONFIG_PATH / 'g1_mujoco.yaml', actuator_config=ACTUATOR_CONFIG)

    # load checkpoint
    policy = ort.InferenceSession(CHECKPOINT)

    # define context variables
    random_action_generator = RandomJointActionGenerator(num_joints=cfg.num_upper_body_actions, soft_limit=cfg.upper_body_soft_limit)
    upper_action = np.zeros(cfg.num_upper_body_actions, dtype=np.float32)
    lower_action = np.zeros(cfg.num_lower_body_actions, dtype=np.float32)
    ang_vel_buffer = CircularBuffer(max_len=cfg.history_len, data_shape=(3,))
    proj_gravity_buffer = CircularBuffer(max_len=cfg.history_len, data_shape=(3,))
    vel_command_buffer = CircularBuffer(max_len=cfg.history_len, data_shape=(3,))
    joint_pos_buffer = CircularBuffer(max_len=cfg.history_len, data_shape=(cfg.num_actions,))
    joint_vel_buffer = CircularBuffer(max_len=cfg.history_len, data_shape=(cfg.num_actions,))
    upper_action_buffer = CircularBuffer(max_len=cfg.history_len, data_shape=(cfg.num_upper_body_actions,))
    lower_action_buffer = CircularBuffer(max_len=cfg.history_len, data_shape=(cfg.num_lower_body_actions,))

    obs = np.zeros(cfg.num_obs, dtype=np.float32)
    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(str(PROJECT_ROOT / cfg.xml_path))
    d = mujoco.MjData(m)
    m.opt.timestep = cfg.simulation_dt


    policy_to_xml = []
    for i in range(1, m.njnt):
        jname = mujoco.mj_id2name(m, 3, i)
        if jname in cfg.policy_joints:
            idx = cfg.policy_joints.index(jname)
            policy_to_xml.append(idx)

    xml_to_policy = []
    for i in range(len(policy_to_xml)):
        idx = policy_to_xml.index(i)
        xml_to_policy.append(idx)


    default_angles = cfg.default_angles[policy_to_xml]
    target_dof_pos = default_angles.copy()

    target_upper_pos = cfg.upper_body_default_pos.copy()
    target_lower_pos = cfg.lower_body_default_pos.copy()
    
    # ============================================
    # Move to default pos
    # ============================================
    # reset robot
    for i in range(20):
        tau = apply_action(
            cfg.num_actions,
            d,
            cfg.lower_body_default_pos,
            cfg.upper_body_default_pos,
            cfg.lower_body_joint2motor_idx,
            cfg.upper_body_joint2motor_idx,
            cfg.lower_body_kps,
            cfg.upper_body_kps,
            cfg.lower_body_kds,
            cfg.upper_body_kds,
        )
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    

    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = 0  
        viewer.cam.distance = 1.5   
        viewer.cam.elevation = -15
          
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        try:
            while viewer.is_running() and time.time() - start < cfg.simulation_duration:
                step_start = time.time()
                tau = apply_action(
                    cfg.num_actions,
                    d,
                    target_lower_pos,
                    target_upper_pos,
                    cfg.lower_body_joint2motor_idx,
                    cfg.upper_body_joint2motor_idx,
                    cfg.lower_body_kps,
                    cfg.upper_body_kps,
                    cfg.lower_body_kds,
                    cfg.upper_body_kds,
                )

                d.ctrl[:] = tau
                # mj_step can be replaced with code that also evaluates
                # a policy and applies a control signal before stepping the physics.
                mujoco.mj_step(m, d)

                counter += 1
                if counter % cfg.control_decimation == 0:
                    # Apply control signal here.

                    # create observation
                    qj = d.qpos[7: 7 + cfg.num_actions]
                    dqj = d.qvel[6: 6 + cfg.num_actions]
                    quat = d.qpos[3:7]
                    omega = d.qvel[3:6]

                    qj = (qj - default_angles) * cfg.dof_pos_scale
                    dqj = dqj * cfg.dof_vel_scale
                    waist_yaw = d.qpos[7: 7 + cfg.num_actions][12]
                    waist_yaw_omega = d.qvel[6: 6 + cfg.num_actions][12]
                    gravity_orientation = get_gravity_orientation(quat)
                    omega = omega * cfg.ang_vel_scale
                    count = counter * cfg.simulation_dt

                    # append to buffer
                    ang_vel_buffer.append(omega.copy())
                    proj_gravity_buffer.append(gravity_orientation.copy())
                    vel_command_buffer.append(cfg.cmd.copy() * cfg.cmd_scale)
                    joint_pos_buffer.append(qj[xml_to_policy].copy())
                    joint_vel_buffer.append(dqj[xml_to_policy].copy())
                    upper_action_buffer.append(upper_action.copy())
                    lower_action_buffer.append(lower_action.copy())
                    obs = np.concatenate([
                        ang_vel_buffer.buffer.flatten(),
                        proj_gravity_buffer.buffer.flatten(),
                        vel_command_buffer.buffer.flatten(),
                        joint_pos_buffer.buffer.flatten(),
                        joint_vel_buffer.buffer.flatten(),
                        upper_action_buffer.buffer.flatten(),
                        lower_action_buffer.buffer.flatten(),
                    ], axis=0)
                    obs = np.clip(obs, -100, 100)
                    obs = obs.astype(np.float32).reshape(1, -1)



                    # apply action
                    upper_action = random_action_generator.sample()

                    lower_action = policy.run(None, {"obs": obs})[0].squeeze(0)
                    lower_action = np.clip(lower_action, -100, 100)
        
                    target_lower_pos = lower_action * cfg.lower_body_action_scale + cfg.lower_body_default_pos
                    target_upper_pos = random_action_generator.processed_action
                    

                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
        except KeyboardInterrupt:
            print("\nSimulation interrupted by user")
        except Exception as e:
            print(f"\nError during simulation: {e}")
        finally:
            time.sleep(0.1)
print("Exiting gracefully...")
time.sleep(0.2)
print("Done")