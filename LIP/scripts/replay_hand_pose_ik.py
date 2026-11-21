import argparse
from contextlib import nullcontext
import os
os.environ['MUJOCO_GL']='egl'
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw
import csv,json
root=Path('/home/choon/isaaclab')
source=root/'LIP/previews/perfect-arm-replay-source'
parser = argparse.ArgumentParser(description="Replay recorded body motion with position-and-orientation arm DLS IK.")
parser.add_argument("--output", type=Path, default=root / "LIP/previews/recomputed-pose-ik-yaw-lock")
parser.add_argument("--no-video", action="store_true", help="Write numeric replay results without rendering a video.")
args = parser.parse_args()
out = args.output
if out.exists():
    parser.error("Choose a fresh output directory")
out.mkdir(parents=True)
raw=np.load(source/'full_state_replay.npz');frames=raw['frames'];names=raw['joint_names'].tolist();dt=float(raw['dt']);n=len(names)
xml=root/'wb_humanoid_mpc/robot_models/unitree_g1/g1_description/urdf/g1_29dof.xml'
tree=ET.parse(xml);tree.getroot().find('compiler').set('meshdir',str(xml.parent.parent/'meshes'))
visual=tree.getroot().find('visual')
if visual is None:visual=ET.SubElement(tree.getroot(),'visual')
global_el=visual.find('global')
if global_el is None:global_el=ET.SubElement(visual,'global')
global_el.set('offwidth','960');global_el.set('offheight','720')
model=mujoco.MjModel.from_xml_string(ET.tostring(tree.getroot(),encoding='unicode'));data=mujoco.MjData(model)
addresses=[int(model.jnt_qposadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)]) for name in names]
assert all(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)>=0 for name in names)
arm=[i for i,name in enumerate(names) if any(k in name for k in ('shoulder','elbow','wrist'))];assert len(arm)==14
camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE;camera.azimuth=-55;camera.elevation=-15;camera.distance=2.5
renderer = None if args.no_video else mujoco.Renderer(model, height=720, width=960)
rows=[]
with (source/'reference_tracking.csv').open() as f:reference=list(csv.DictReader(f))
validation=[]
video=out/'exact_arm_commands.mp4'
replay_arm_angles = frames[0, 7:7+n][arm].copy()
q_hold = None
q_hold_time = None
wrist_yaw_slots = [arm.index(names.index(side + "_wrist_yaw_joint")) for side in ("left", "right")]
wrist_yaw_qpos = [addresses[arm[slot]] for slot in wrist_yaw_slots]

video_context = nullcontext(None) if args.no_video else imageio.get_writer(video, fps=round(1/dt), codec='libx264', quality=8, macro_block_size=1)
with video_context as writer:
 for i,frame in enumerate(frames):
  data.qpos[:3]=frame[:3];data.qpos[3:7]=frame[[6,3,4,5]]
  actual=frame[7:7+n];command=frame[7+n:7+2*n]
  data.qpos[addresses]=actual;mujoco.mj_forward(model,data)
  def hand_xyz(side):
   bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,side+'_wrist_yaw_link')
   return data.xpos[bid]+data.xmat[bid].reshape(3,3)@np.array([.0415,.003 if side=='left' else -.003,0])
  validation.extend([abs(hand_xyz(side)[2]-float(reference[i][side+'_hand_actual_z_world_m'])) for side in ['left','right']])
  # Recover the original desired world positions from the recorded torso frame.
  # At this point, data still contains the recorded actual robot pose.
  torso_id = mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_BODY, "torso_link"
  )
  torso_rotation = data.xmat[torso_id].reshape(3, 3)
  world_targets = {}

  for side in ("left", "right"):
      target_torso = np.array([
          float(reference[i][f"{side}_hand_target_{axis}_torso_m"])
          for axis in "xyz"
      ])
      world_targets[side] = (
          data.xpos[torso_id] + torso_rotation @ target_torso
      )

  # Preserve the arms from the previous replay frame.
  # Pelvis, legs, and waist retain their recorded poses.
  for slot, j in enumerate(arm):
      data.qpos[addresses[j]] = replay_arm_angles[slot]
  mujoco.mj_forward(model, data)

  # Capture the current replay arm state once, before the seven-second IK update.
  if q_hold is None and i * dt >= 7.0:
      q_hold = replay_arm_angles.copy()
      q_hold_time = i * dt

  # Reapply the captured angles explicitly, independent of the recorded pose.
  if q_hold is not None:
      data.qpos[wrist_yaw_qpos] = q_hold[wrist_yaw_slots]
      mujoco.mj_forward(model, data)

  # One differential-IK update per frame, without persistent accumulation.
  for side in ("left", "right"):
      side_joints = [
          j for j in arm
          if names[j].startswith(side + "_")
          and (q_hold is None or names[j] != side + "_wrist_yaw_joint")
      ]
      joint_ids = [
          mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, names[j])
          for j in side_joints
      ]
      dof_ids = [int(model.jnt_dofadr[jid]) for jid in joint_ids]
      qpos_ids = [addresses[j] for j in side_joints]

      hand_position = hand_xyz(side)
      body_id = mujoco.mj_name2id(
          model, mujoco.mjtObj.mjOBJ_BODY, side + "_wrist_yaw_link"
      )

      jac_position = np.zeros((3, model.nv))
      jac_rotation = np.zeros((3, model.nv))
      mujoco.mj_jac(
          model, data, jac_position, jac_rotation,
          hand_position, body_id,
      )

      # Identity hand-link orientation relative to the torso: mirrored palms inward.
      # MuJoCo quaternions use wxyz. mju_subQuat returns the shortest rotation
      # in the current local frame, so rotate it into world axes for mj_jac.
      orientation_error_local = np.zeros(3)
      mujoco.mju_subQuat(orientation_error_local, data.xquat[torso_id], data.xquat[body_id])
      orientation_error_world = data.xmat[body_id].reshape(3, 3) @ orientation_error_local
      jac = np.vstack((jac_position[:, dof_ids], jac_rotation[:, dof_ids]))
      error = np.concatenate((world_targets[side] - hand_position, orientation_error_world))

      # Same DLS damping and per-update correction limit as the saved run.
      correction = jac.T @ np.linalg.solve(
          jac @ jac.T + 0.05**2 * np.eye(6), error
      )
      correction = np.clip(correction, -0.05, 0.05)

      # Match the saved 0.9 soft joint-limit factor.
      ranges = model.jnt_range[joint_ids]
      center = ranges.mean(axis=1)
      half_range = 0.9 * (ranges[:, 1] - ranges[:, 0]) / 2

      data.qpos[qpos_ids] = np.clip(
          data.qpos[qpos_ids] + correction,
          center - half_range,
          center + half_range,
      )
      mujoco.mj_forward(model, data)

  replay_arm_angles = data.qpos[
      [addresses[j] for j in arm]
  ].copy()
  if q_hold is not None:
      np.testing.assert_array_equal(data.qpos[wrist_yaw_qpos], q_hold[wrist_yaw_slots])
  row = {"time_s": i * dt, "wrist_yaw_locked": q_hold is not None}
  for side, address in zip(("left", "right"), wrist_yaw_qpos):
      row[f"{side}_wrist_yaw_rad"] = float(data.qpos[address])
  for side in ("left", "right"):
   for axis, value in zip("xyz", hand_xyz(side)):
    row[f"{side}_hand_world_{axis}_m"] = float(value)
  rows.append(row)
  if renderer is not None:
   camera.lookat[:]=[frame[0],frame[1],.82]
   renderer.update_scene(data,camera)
   img=Image.fromarray(renderer.render());draw=ImageDraw.Draw(img)
   draw.rectangle((0,0,960,58),fill=(15,20,25))
   draw.text((14,10),f'POSE IK - NO ACCUMULATION | {i*dt:5.2f} s | No dynamics or policy during replay',fill='white')
   draw.text((14,32),f'Recorded body; wrist yaw {"LOCKED" if q_hold is not None else "free until 7 s"}; remaining joints use pose IK.',fill='white')
   writer.append_data(np.asarray(img))
  if i%250==0:print('Processed',i,flush=True)
if renderer is not None:
 renderer.close()
with (out/'kinematic_hand_positions.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
metadata={'source_checkpoint':str(root/'IsaacLab/logs/rsl_rl/g1_lipm_online_hands_finetune/run_5ogXH686/training/model_499.pt'),'source_recording':str(source),'frames':len(frames),'fps':round(1/dt),'duration_s':len(frames)*dt,'arm_joints':[names[i] for i in arm],'max_actual_hand_world_z_model_disagreement_m':max(validation),'orientation_target':'identity hand-link orientation relative to current torso (palms inward)', 'q_hold_capture_time_s':q_hold_time, 'locked_joints':['left_wrist_yaw_joint','right_wrist_yaw_joint'], 'wrist_yaw_hold_rad':None if q_hold is None else q_hold[wrist_yaw_slots].tolist(), 'video_written':not args.no_video, 'description':'Closed-loop kinematic diagnostic: recorded pelvis, legs, waist and world targets; arm position-and-orientation DLS IK recomputed from previous replay arm state each frame. At seven seconds wrist yaw is captured and fixed on both arms, and its columns are removed from subsequent IK solves. Remaining arm joints continue pose IK. Full q_hold is also saved; no null-space posture correction is applied. No persistent reference accumulation, physics integration, or policy execution. Damping 0.05, per-joint increment cap 0.05 rad, soft-limit factor 0.9, one solve per recorded control frame.'}
if q_hold is not None:
 np.savez(out / 'q_hold.npz', q_hold=q_hold.reshape(2, 7), joint_names=np.array([names[j] for j in arm]), time_s=q_hold_time)
(out/'replay_metadata.json').write_text(json.dumps(metadata,indent=2))
print(json.dumps(metadata,indent=2),flush=True)
if not args.no_video:
 print(video,flush=True)
