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
parser = argparse.ArgumentParser(description="Replay recorded body motion with arm IK holding world horizontal position and orientation after seven seconds.")
parser.add_argument("--output", type=Path, default=root / "LIP/previews/recomputed-horizontal-hand-hold")
parser.add_argument("--no-video", action="store_true", help="Write numeric replay results without rendering a video.")
parser.add_argument("--hold-frame", choices=("torso_translation", "world"), default="torso_translation",
                    help="Default: fixed horizontal offset from torso in world axes; world: fixed room location.")
parser.add_argument("--preserve-position", action="store_true",
                    help="Keep original recorded position targets after seven seconds; capture orientation only.")
parser.add_argument("--preserve-orientation", action="store_true",
                    help="Keep the original torso-relative orientation target throughout the replay.")
parser.add_argument("--initial-yaw-orientation", action="store_true",
                    help="Keep a fixed world hand orientation aligned with the initial torso yaw, with zero pitch and roll.")
parser.add_argument("--world-identity-orientation", action="store_true",
                    help="Keep hand world roll, pitch, and yaw at zero throughout, independently of initial torso pose.")
parser.add_argument("--torso-following-z", action="store_true",
                    help="Use original Z trajectory minus initial torso height as a world-axis offset from current torso height.")
parser.add_argument("--x-velocity-filter-tau", type=float,
                    help="Advance a world X anchor with low-pass torso X velocity (seconds; zero disables smoothing). Requires --preserve-position.")
args = parser.parse_args()
if args.x_velocity_filter_tau is not None:
    if not np.isfinite(args.x_velocity_filter_tau) or args.x_velocity_filter_tau < 0:
        parser.error("--x-velocity-filter-tau must be finite and nonnegative")
    if not args.preserve_position:
        parser.error("--x-velocity-filter-tau requires --preserve-position")
if sum((args.initial_yaw_orientation, args.preserve_orientation, args.world_identity_orientation)) > 1:
    parser.error("Choose one orientation mode")
fixed_world_orientation = args.initial_yaw_orientation or args.world_identity_orientation
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

print("World gravity:", model.opt.gravity)
print("Body gravity compensation:", model.body_gravcomp)
print("Joint actuator gravity compensation:", model.jnt_actgravcomp)
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
held_positions = {}
held_quaternions = {}
held_torso_position = None
initial_torso_z = None
previous_torso_x = None
world_x_anchor = None
filtered_torso_velocity_x = 0.0
torso_velocity_x = 0.0
fixed_world_quaternion = np.array([1.0, 0.0, 0.0, 0.0]) if args.world_identity_orientation else None
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
  if initial_torso_z is None:
      initial_torso_z = float(data.xpos[torso_id, 2])
  if args.initial_yaw_orientation and fixed_world_quaternion is None:
      yaw = np.arctan2(torso_rotation[1, 0], torso_rotation[0, 0])
      fixed_world_quaternion = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
  if args.x_velocity_filter_tau is not None:
      torso_x = float(data.xpos[torso_id, 0])
      if previous_torso_x is None:
          world_x_anchor = torso_x
      else:
          torso_velocity_x = (torso_x - previous_torso_x) / dt
          alpha = 1.0 if args.x_velocity_filter_tau == 0 else -np.expm1(-dt / args.x_velocity_filter_tau)
          filtered_torso_velocity_x += alpha * (torso_velocity_x - filtered_torso_velocity_x)
          world_x_anchor += filtered_torso_velocity_x * dt
      previous_torso_x = torso_x
  world_targets = {}

  for side in ("left", "right"):
      target_torso = np.array([
          float(reference[i][f"{side}_hand_target_{axis}_torso_m"])
          for axis in "xyz"
      ])
      world_targets[side] = (
          data.xpos[torso_id] + torso_rotation @ target_torso
      )

      if args.x_velocity_filter_tau is not None:
          # Keep the original reach offset while advancing the world anchor causally.
          world_targets[side][0] += world_x_anchor - data.xpos[torso_id, 0]
      if args.torso_following_z:
          world_targets[side][2] += data.xpos[torso_id, 2] - initial_torso_z

  # Preserve the arms from the previous replay frame.
  # Pelvis, legs, and waist retain their recorded poses.
  for slot, j in enumerate(arm):
      data.qpos[addresses[j]] = replay_arm_angles[slot]
  mujoco.mj_forward(model, data)

  # Capture the current replay arm state once, before the seven-second IK update.
  if q_hold is None and i * dt >= 7.0 and not (args.preserve_position and (args.preserve_orientation or fixed_world_orientation)):
      q_hold = replay_arm_angles.copy()
      q_hold_time = i * dt
      held_torso_position = data.xpos[torso_id].copy()
      for side in ("left", "right"):
          bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, side + "_wrist_yaw_link")
          held_positions[side] = hand_xyz(side).copy()
          if not (args.preserve_orientation or fixed_world_orientation):
              held_quaternions[side] = data.xquat[bid].copy()

  # Hold horizontal targets in the selected frame; retain recorded vertical goals.
  if q_hold is not None and not args.preserve_position:
      for side in ("left", "right"):
          world_targets[side][:2] = held_positions[side][:2]
          if args.hold_frame == "torso_translation":
              world_targets[side][:2] += data.xpos[torso_id, :2] - held_torso_position[:2]

  orientation_targets = {
      side: fixed_world_quaternion if fixed_world_orientation else held_quaternions.get(side, data.xquat[torso_id])
      for side in ("left", "right")
  }

  # One differential-IK update per frame, without persistent accumulation.
  for side in ("left", "right"):
      side_joints = [
          j for j in arm
          if names[j].startswith(side + "_")
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

      # Track the selected hand-link orientation with mirrored palms inward.
      # MuJoCo quaternions use wxyz. mju_subQuat returns the shortest rotation
      # in the current local frame, so rotate it into world axes for mj_jac.
      orientation_error_local = np.zeros(3)
      mujoco.mju_subQuat(orientation_error_local, orientation_targets[side], data.xquat[body_id])
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

      q_des = np.clip(
        data.qpos[qpos_ids] + correction,
        center - half_range,
        center + half_range,
      )
    #   q = data.qpos[arm_pos_ids]
    #   qd = data.qvel[arm_dof_ids]

    #   tau_requested = 40.0 * (arm_q_des - q) - 10.0 * qd
    #   tau_applied = np.clip(tau_requested, - arm_torque_limits, arm_torque_limits)

    #   data.ctrl[arm_actuator_ids] = tau_applied
      data.qpos[qpos_ids] = q_des
  mujoco.mj_forward(model, data)

    #   mujoco.mj_forward(model, data)

  replay_arm_angles = data.qpos[
      [addresses[j] for j in arm]
  ].copy()
  row = {"time_s": i * dt, "horizontal_hold_active": q_hold is not None}
  if args.x_velocity_filter_tau is not None:
      row["torso_velocity_x_m_s"] = torso_velocity_x
      row["filtered_torso_velocity_x_m_s"] = filtered_torso_velocity_x
      row["world_x_anchor_m"] = world_x_anchor
  for axis, value in zip("xyz", data.xpos[torso_id]):
      row[f"torso_world_{axis}_m"] = float(value)
  for side in ("left", "right"):
      bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, side + "_wrist_yaw_link")
      orientation_error = np.zeros(3)
      mujoco.mju_subQuat(orientation_error, orientation_targets[side], data.xquat[bid])
      for axis, value in zip("wxyz", orientation_targets[side]):
          row[f"{side}_target_orientation_world_{axis}"] = float(value)
      row[f"{side}_orientation_error_rad"] = float(np.linalg.norm(orientation_error))
      row[f"{side}_horizontal_error_m"] = float(np.linalg.norm(hand_xyz(side)[:2] - world_targets[side][:2]))
      for axis, value in zip("xyz", world_targets[side]):
          row[f"{side}_target_world_{axis}_m"] = float(value)
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
   target_label = "Original position targets" if args.preserve_position else "Measured XY hold"
   orientation_label = "Original torso-relative orientation" if args.preserve_orientation else (
       "World orientation held" if q_hold is not None else "Torso-relative orientation until 7 s"
   )
   if args.initial_yaw_orientation:
       orientation_label = "Fixed world orientation from initial torso yaw"
   if args.world_identity_orientation:
       orientation_label = "Fixed world identity orientation"
   if args.torso_following_z:
       target_label = "XYZ offsets from torso in world axes"
   if args.x_velocity_filter_tau is not None:
       target_label = f"World X / velocity filter {args.x_velocity_filter_tau:g}s"
   draw.text((14,32),f'{target_label} | {orientation_label}',fill='white')
   writer.append_data(np.asarray(img))
  if i%250==0:print('Processed',i,flush=True)
if renderer is not None:
 renderer.close()
with (out/'kinematic_hand_positions.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
metadata = {
    "source_recording": str(source), "frames": len(frames), "duration_s": len(frames)*dt,
    "hold_time_s": q_hold_time, "hold_frame": args.hold_frame, "video_written": not args.no_video,
    "preserve_original_position": args.preserve_position,
    "preserve_original_orientation": args.preserve_orientation,
    "held_world_positions": {k:v.tolist() for k,v in held_positions.items()},
    "held_world_quaternions_wxyz": {k:v.tolist() for k,v in held_quaternions.items()},
    "description": "Recorded body motion; all seven joints per arm remain in pose DLS IK. At seven seconds capture XY and world orientation, retaining recorded world Z targets. In torso_translation mode, XY follows torso translation in world axes; in world mode XY stays at the captured room location. No dynamics, policy inference, persistent reference accumulation, joint locking, or posture correction. Rigid G1 rubber hands; no articulated fingers.",
    "post_hold_errors": {
        side: {
            "horizontal_rms_m": float(np.sqrt(np.mean([r[f"{side}_horizontal_error_m"]**2 for r in rows if r["time_s"] >= 7.0]))),
            "horizontal_max_m": max(r[f"{side}_horizontal_error_m"] for r in rows if r["time_s"] >= 7.0),
            "orientation_max_deg": float(np.rad2deg(max(r[f"{side}_orientation_error_rad"] for r in rows if r["time_s"] >= 7.0)))
        } for side in ("left", "right")
    }
}
if args.preserve_position:
 metadata["description"] = (
     "Recorded body motion from perfect-arm-replay-source (run_5ogXH686). "
     "Original recorded position targets retained for the entire replay; only measured world orientation "
     "is held after seven seconds. One pose DLS IK update per frame, all arm joints capped at 0.05 rad "
     "and placed exactly. No physics stepping or policy inference."
 )
if args.preserve_orientation:
 metadata["description"] = (
     "Recorded body motion from perfect-arm-replay-source (run_5ogXH686). "
     + ("Original recorded position targets retained throughout. " if args.preserve_position
        else "Measured XY captured at seven seconds; requested Z retained. ")
     + "Original identity hand orientation relative to the current torso retained throughout; "
     "no world-orientation capture. Fresh pose DLS IK each frame, 0.05 rad joint correction cap, "
     "exact joint placement, no physics stepping or policy inference."
 )
if args.initial_yaw_orientation:
 metadata["initial_yaw_orientation_world_wxyz"] = fixed_world_quaternion.tolist()
 metadata["description"] = (
     "Recorded body motion from perfect-arm-replay-source (run_5ogXH686). "
     + ("Original recorded position targets retained throughout. " if args.preserve_position
        else "Measured XY captured at seven seconds; requested Z retained. ")
     + "Hand world orientation fixed from initial torso yaw with zero pitch and roll, "
     "without seven-second orientation capture. Fresh pose DLS IK each frame, "
     "0.05 rad joint correction cap, exact joint placement, no physics stepping or policy inference."
 )
if args.world_identity_orientation:
 metadata["world_identity_orientation_wxyz"] = fixed_world_quaternion.tolist()
 metadata["description"] = (
     "Recorded body motion from perfect-arm-replay-source (run_5ogXH686). "
     + ("Original recorded position targets retained throughout. " if args.preserve_position
        else "Measured XY captured at seven seconds; requested Z retained. ")
     + "Hand world orientation fixed at identity (zero roll, pitch, yaw), independent of initial torso pose. "
     "No seven-second orientation capture. Fresh pose DLS IK each frame, 0.05 rad joint correction cap, "
     "exact joint placement, no physics stepping or policy inference."
 )
metadata["torso_following_z"] = args.torso_following_z
metadata["initial_torso_z_m"] = initial_torso_z
if args.torso_following_z:
 metadata["description"] += (
     " Z target follows current torso height with the original recorded Z trajectory minus initial torso height "
     "as its offset; X/Y target definitions are retained."
 )
metadata["x_velocity_filter_tau_s"] = args.x_velocity_filter_tau
if args.x_velocity_filter_tau is not None:
 metadata["description"] += (
     " World X anchor starts at initial torso X and integrates causally filtered torso X velocity "
     "(backward position differences, exponential low-pass, initial filtered velocity zero). "
     "The original reach X offset is added to this anchor."
 )
 metadata["x_anchor_max_lag_m"] = max(abs(r["world_x_anchor_m"] - r["torso_world_x_m"]) for r in rows)
for side in ("left", "right"):
 errors = [
     np.linalg.norm([r[f"{side}_hand_world_{a}_m"] - r[f"{side}_target_world_{a}_m"] for a in "xyz"])
     for r in rows if r["time_s"] >= 7.0
 ]
 metadata["post_hold_errors"][side]["position_mean_m"] = float(np.mean(errors))
 metadata["post_hold_errors"][side]["position_max_m"] = float(np.max(errors))
if q_hold is not None:
 np.savez(out / 'q_hold.npz', q_hold=q_hold.reshape(2, 7), joint_names=np.array([names[j] for j in arm]), time_s=q_hold_time)
(out/'replay_metadata.json').write_text(json.dumps(metadata,indent=2))
print(json.dumps(metadata,indent=2),flush=True)
if not args.no_video:
 print(video,flush=True)
