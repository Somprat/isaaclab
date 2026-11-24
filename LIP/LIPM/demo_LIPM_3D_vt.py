import numpy as np
import os
import matplotlib
if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter, writers
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from trajectory import generate_reference

parser = argparse.ArgumentParser(description="Step-length-conditioned LIPM demo")
for name, default, help_text in (
    ("step-length", 0.20, "Desired alternating-step length [m]"),
    ("step-width", None, "Lateral foot separation [m]"),
    ("step-duration", 0.40, "Time between alternating touchdowns [s]"),
    ("com-height", 0.60, "Constant CoM height [m]"),
    ("heading", 0.0, "World travel heading [rad]"),
    ("swing-height", 0.05, "Swing clearance [m]"),
):
    parser.add_argument("--" + name, type=float, default=default, help=help_text)
parser.add_argument("--initial-velocity", type=float, default=None, help="Initial forward speed [m/s]; defaults to rest with startup, otherwise length / duration")
parser.add_argument("--headless", action="store_true", help="Save animation without opening a window")
parser.add_argument("--no-render", action="store_true", help="Roll out without plots or video")
parser.add_argument("--output", type=Path, default=Path("LIPM_trajectory.npz"), help="Trajectory NPZ destination (saved even with --no-render)")
parser.add_argument("--duration", type=float, default=10.0, help="Rollout duration [s]")
parser.add_argument("--initial-state", type=Path, help="Measured G1 reset JSON; overrides CoM state and initial foot XY. Feet are projected to the ground plane.")
parser.add_argument("--torso-com-height", type=float, default=None, help="Desired torso_link CoM height [m]; converted to whole-robot CoM height with the reset offset (whole CoM z - torso CoM z) and overrides --com-height and the initial-state CoM height")
parser.add_argument("--height-calibration", type=Path, help="wb_humanoid_mpc torso-origin height calibration JSON (humanoid_centroidal_mpc_height_calibration). Its whole_body_com_height overrides --com-height and the --initial-state CoM height; --initial-state still supplies CoM XY/velocity and feet.")
parser.add_argument("--startup-duration", type=float, default=0.8, help="Minimum two-foot load-transfer duration [s]; 0 disables startup")
parser.add_argument("--max-startup-duration", type=float, default=2.0, help="Longest startup allowed if feasibility requires more time [s]")
parser.add_argument("--gait-schedule", type=Path, help="JSON with duration, transition_steps and schedule rows")
args = parser.parse_args()
schedule = None
transition_steps = 2
if args.gait_schedule:
    config = json.loads(args.gait_schedule.read_text())
    args.duration = float(config["duration"])
    transition_steps = config.get("transition_steps", 2)
    schedule = [[row["time"], row["step_length"], row["step_duration"]] for row in config["schedule"]]
    from trajectory import validate_schedule
    schedule = validate_schedule(schedule, args.duration, transition_steps)
    args.step_length, args.step_duration = schedule[0, 1:]

measured_state = None
if args.initial_state is not None:
    measured_state = json.loads(args.initial_state.read_text())
    if measured_state.get("schema_version") != 1 or measured_state.get("frame") != "world_z_up":
        parser.error("initial-state must use schema 1 and world_z_up")
    if measured_state.get("foot_names") != ["left", "right"]:
        parser.error("initial-state feet must be ordered left, right")
    for key, shape in (("com_pos", (3,)), ("com_vel", (3,)), ("foot_pos", (2, 3))):
        value = np.asarray(measured_state[key], dtype=float)
        if value.shape != shape or not np.isfinite(value).all():
            parser.error(f"invalid initial-state {key}; expected finite {shape}")
    if abs(measured_state["com_vel"][2]) > 1e-6:
        parser.error("constant-height LIPM requires zero initial vertical velocity")
    args.com_height = float(measured_state["com_pos"][2])
    if args.step_width is None:
        lateral = np.array([-np.sin(args.heading), np.cos(args.heading)])
        feet = np.asarray(measured_state["foot_pos"])
        args.step_width = abs(float((feet[0, :2] - feet[1, :2]) @ lateral))
# Measured at G1 lipm reset: whole CoM z 0.714414 m, torso_link CoM z 1.018010 m.
TORSO_TO_COM_Z = -0.303596
if args.height_calibration is not None and args.torso_com_height is not None:
    parser.error("use either --height-calibration or --torso-com-height, not both")
height_source = "initial_state" if measured_state is not None else "com_height_argument"
if args.height_calibration is not None:
    calibration = json.loads(args.height_calibration.read_text())
    if calibration.get("kind") != "g1_torso_origin_height_calibration" or calibration.get("schema_version") != 2:
        parser.error("height-calibration must be a schema 2 g1_torso_origin_height_calibration file")
    if not calibration.get("success"):
        parser.error(f"height calibration did not converge (status: {calibration.get('status')})")
    args.com_height = float(calibration["whole_body_com_height"]) - float(calibration["mean_contact_frame_height"])
    height_source = "height_calibration"
    print(f"Height calibration {args.height_calibration}: requested torso origin z={calibration['requested_torso_origin_height']:.6f} m, "
          f"measured {calibration['measured_torso_origin_height']:.6f} m -> whole-body CoM height {args.com_height:.6f} m")
if args.torso_com_height is not None:
    height_source = "torso_com_height_offset"
    if measured_state is not None and "torso_com_pos" in measured_state:
        TORSO_TO_COM_Z = float(measured_state["com_pos"][2]) - float(measured_state["torso_com_pos"][2])
    args.com_height = args.torso_com_height + TORSO_TO_COM_Z
    print(f"Torso CoM height {args.torso_com_height:.6f} m -> whole-robot CoM height "
          f"{args.com_height:.6f} m (offset {TORSO_TO_COM_Z:+.6f} m)")
if args.step_width is None:
    args.step_width = 0.20
if args.headless:
    plt.switch_backend("Agg")
for name, value in vars(args).items():
    if isinstance(value, float) and not np.isfinite(value):
        parser.error(f"{name} must be finite")
if args.duration <= args.step_duration:
    parser.error("duration must exceed one step duration")
if args.step_duration < 0.04 or args.step_width <= 0 or args.com_height <= 0 or args.step_length < 0 or args.swing_height < 0:
    parser.error("Require duration >= 0.04 s, positive width/height, nonnegative length/clearance")
samples_per_step = max(2, round(args.step_duration / 0.02))


class Ball:
    def __init__(self, size=10, shape='o'):
        self.scatter, = ax.plot([], [], [], shape, markersize=size, animated=False)

    def update(self, pos):
        # draw ball
        self.scatter.set_data_3d([pos[0]], [pos[1]], [pos[2]])

class Line:
    def __init__(self, size=1, color='g'):
        self.line, = ax.plot([], [], [], linewidth=size, color=color, animated=False)

    def update(self, pos):
        # draw line
        self.line.set_xdata(pos[0,:])
        self.line.set_ydata(pos[1,:])
        self.line.set_3d_properties(np.asarray(pos[2,:]))

class LIPM_3D_Animate():
    def __init__(self):
        self.origin = Ball(size=2, shape='ko')
        self.COM_trajectory = Line(size=1, color='g')
        self.COM_head = Ball(size=2, shape='ro')

        self.left_foot = Ball(size=5, shape='bo')
        self.right_foot = Ball(size=5, shape='mo')

        self.left_leg = Line(size=3, color='b')
        self.right_leg = Line(size=3, color='m')
        self.COM = Ball(size=16, shape='ro')


    def update(self, COM_pos, COM_pos_trajectory, left_foot_pos, right_foot_pos):
        self.origin.update([0, 0, 0])
        self.COM.update(COM_pos)
        self.COM_trajectory.update(COM_pos_trajectory)
        self.COM_head.update(COM_pos_trajectory[:,-1])

        self.left_foot.update(left_foot_pos)
        self.right_foot.update(right_foot_pos)

        pos_1 = np.zeros((3,2))
        pos_1[:,0] = COM_pos
        pos_1[:,1] = left_foot_pos
        self.left_leg.update(pos_1)

        pos_2 = np.zeros((3,2))
        pos_2[:,0] = COM_pos
        pos_2[:,1] = right_foot_pos
        self.right_leg.update(pos_2)

        artists = []
        artists.append(self.origin.scatter)
        artists.append(self.left_leg.line)
        artists.append(self.right_leg.line)
        artists.append(self.COM.scatter)
        artists.append(self.COM_trajectory.line)
        artists.append(self.COM_head.scatter)
        artists.append(self.left_foot.scatter)
        artists.append(self.right_foot.scatter)

        # automatic set the x, y view limitation 
        if COM_pos[0] >= 3.0:
            ax.set_xlim(-1.0 + COM_pos[0] - 3.0, 4.0 + COM_pos[0] - 3.0)
        elif COM_pos[0] <= 0:
            ax.set_xlim(-1.0 + COM_pos[0], 4.0 + COM_pos[0])

        if COM_pos[1] >= 1.5:
            ax.set_ylim(-2 + COM_pos[1] - 1.5, 2 + COM_pos[1] - 1.5)
        elif COM_pos[1] <= -1.5:
            ax.set_ylim(-2 + COM_pos[1] + 1.5, 2 + COM_pos[1] + 1.5)

        ax.set_zlim(-0.01, 1.)

        artists.append(ax)

        return artists 

def ani_3D_init():
    return [] 

def ani_3D_update(i):
    COM_pos = [COM_pos_x[i], COM_pos_y[i], LIPM_model.zc]
    COM_pos_trajectory = np.zeros((3, i))
    COM_pos_trajectory[0,:] = COM_pos_x[0:i]
    COM_pos_trajectory[1,:] = COM_pos_y[0:i]
    COM_pos_trajectory[2,:] = np.zeros((1,i))

    left_foot_pos = [left_foot_pos_x[i], left_foot_pos_y[i], left_foot_pos_z[i]]
    right_foot_pos = [right_foot_pos_x[i], right_foot_pos_y[i], right_foot_pos_z[i]]

    artists = LIPM_3D_ani.update(COM_pos, COM_pos_trajectory, left_foot_pos, right_foot_pos)

    return artists 

def ani_2D_init():
    COM_traj_ani.set_data(COM_pos_x[0:0], COM_pos_y[0:0])
    COM_pos_ani.set_data([COM_pos_x[0]], [COM_pos_y[0]])
    left_foot_pos_ani.set_data([left_foot_pos_x[0]], [left_foot_pos_y[0]])
    right_foot_pos_ani.set_data([right_foot_pos_x[0]], [right_foot_pos_y[0]])

    return [COM_pos_ani, COM_traj_ani, left_foot_pos_ani, right_foot_pos_ani]

def ani_2D_update(i):
    COM_traj_ani.set_data(COM_pos_x[0:i], COM_pos_y[0:i])
    COM_pos_ani.set_data([COM_pos_x[i]], [COM_pos_y[i]])
    left_foot_pos_ani.set_data([left_foot_pos_x[i]], [left_foot_pos_y[i]])
    right_foot_pos_ani.set_data([right_foot_pos_x[i]], [right_foot_pos_y[i]])

    ani_text_COM_pos.set_text(COM_pos_str % (COM_pos_x[i], COM_pos_y[i]))

    # # automatic set the x, y view limitation 
    bx.set_xlim(-2.0 + COM_pos_x[i], 3.0 + COM_pos_x[i])
    bx.set_ylim(-0.8 + COM_pos_y[i], 0.8 + COM_pos_y[i])

    return [COM_pos_ani, COM_traj_ani, left_foot_pos_ani, right_foot_pos_ani, ani_text_COM_pos, bx]
   

def COM_vel_2D_init():
    COM_vel_x_ani.set_data(np.linspace(0, 1, 0), COM_vel_x[0:0])
    COM_vel_y_ani.set_data(np.linspace(0, 1, 0), COM_vel_y[0:0])
    return [COM_vel_x_ani, COM_vel_y_ani]

def COM_vel_2D_update(i):
    COM_vel_x_ani.set_data(np.linspace(0, i-1, i), COM_vel_x[0:i])
    COM_vel_y_ani.set_data(np.linspace(0, i-1, i), COM_vel_y[0:i])
    COM_dvel_x_ani.set_data(np.linspace(0, i-1, i), COM_dvel_x[0:i])
    COM_dvel_y_ani.set_data(np.linspace(0, i-1, i), COM_dvel_y[0:i])
    return [COM_vel_x_ani, COM_vel_y_ani, cx]

def step_params_2D_init():
    step_length_ani.set_data(np.linspace(0, 1, 0), step_length[0:0])
    step_width_ani.set_data(np.linspace(0, 1, 0), step_width[0:0])
    dstep_length_ani.set_data(np.linspace(0, 1, 0), dstep_length[0:0])
    dstep_width_ani.set_data(np.linspace(0, 1, 0), dstep_width[0:0])
    return [step_length_ani, step_width_ani]

def step_params_2D_update(i):
    step_length_ani.set_data(np.linspace(0, i-1, i), step_length[0:i])
    step_width_ani.set_data(np.linspace(0, i-1, i), step_width[0:i])
    dstep_length_ani.set_data(np.linspace(0, i-1, i), dstep_length[0:i])
    dstep_width_ani.set_data(np.linspace(0, i-1, i), dstep_width[0:i])
    return [step_length_ani, step_width_ani, dx]


# %% ---------------------------------------------------------------- LIPM control
print('\n--------- Program start from here ...')

COM_pos_x = list()
COM_pos_y = list()
COM_vel_x = list()
COM_vel_y = list()
COM_dvel_x = list()
COM_dvel_y = list()
left_foot_pos_x = list()
left_foot_pos_y = list()
left_foot_pos_z = list()
right_foot_pos_x = list()
right_foot_pos_y = list()
right_foot_pos_z = list()
step_length = list()
dstep_length = list()
step_width = list() 
dstep_width = list()

# Initialize the 3D LIPM (with initial COM position, velocity and foot position)
# COM_pos_0 = [-0.4, 0.2, 1.0]
# COM_v0 = [1.0, -0.01]
COM_pos_0 = [0., 0., args.com_height]
initial_speed = (0.0 if args.startup_duration > 0 else args.step_length / args.step_duration) if args.initial_velocity is None else args.initial_velocity
COM_v0 = initial_speed * np.array([np.cos(args.heading), np.sin(args.heading)])

left_foot_pos = [0., args.step_width / 2, 0]
right_foot_pos = [0., -args.step_width / 2, 0]
if measured_state is not None:
    COM_pos_0 = list(measured_state["com_pos"])
    COM_pos_0[2] = args.com_height  # measured height unless --height-calibration or --torso-com-height is given
    COM_v0 = np.asarray(measured_state["com_vel"][:2])
    # Single-support LIPM assumes feet on the plane; retain measured XY exactly.
    left_foot_pos = [*measured_state["foot_pos"][0][:2], 0.0]
    right_foot_pos = [*measured_state["foot_pos"][1][:2], 0.0]
    print(f"Initialized from {args.initial_state}; CoM height={args.com_height:.6f} m, "
          f"step width={args.step_width:.6f} m. Projected sole heights to the ground plane.")
fields = generate_reference(
    COM_pos_0, COM_v0, [left_foot_pos, right_foot_pos],
    step_length=args.step_length, step_width=args.step_width, step_duration=args.step_duration,
    heading=args.heading, swing_height=args.swing_height, duration=args.duration,
    dt=0.005 if schedule is not None else args.step_duration / samples_per_step, startup_duration=args.startup_duration,
    max_startup_duration=args.max_startup_duration,
    contact_points=measured_state.get("foot_contact_points") if measured_state else None,
    schedule=schedule, transition_steps=transition_steps,
)
fields["initial_state_source"] = np.array(str(args.initial_state.resolve()) if args.initial_state else "manual")
fields["com_height_source"] = np.array(height_source)
fields["height_calibration_source"] = np.array(str(args.height_calibration.resolve()) if args.height_calibration else "none")
fields["initial_foot_projection"] = np.array("ground_z_zero")
count = len(fields["time"])
LIPM_model = SimpleNamespace(dt=float(fields["dt"]), zc=args.com_height)
total_time = float(fields["time"][-1])
COM_pos_x, COM_pos_y = fields["com_pos"][:, 0].tolist(), fields["com_pos"][:, 1].tolist()
COM_vel_x, COM_vel_y = fields["com_vel"][:, 0].tolist(), fields["com_vel"][:, 1].tolist()
COM_dvel_x = fields["com_vel"][:, 0].tolist()
COM_dvel_y = fields["com_vel"][:, 1].tolist()
left_foot_pos_x, left_foot_pos_y, left_foot_pos_z = fields["foot_pos"][:, 0].T.tolist()
right_foot_pos_x, right_foot_pos_y, right_foot_pos_z = fields["foot_pos"][:, 1].T.tolist()
delta = fields["foot_pos"][:, 1, :2] - fields["foot_pos"][:, 0, :2]
step_length = np.abs(delta @ np.array([np.cos(args.heading), np.sin(args.heading)])).tolist()
step_width = np.abs(delta @ np.array([-np.sin(args.heading), np.cos(args.heading)])).tolist()
dstep_length, dstep_width = fields["desired_step_length"].tolist(), fields["desired_step_width"].tolist()
print(f"Startup: {float(fields['startup_duration']):.2f} s; "
      f"ZMP support margin: {float(fields['startup_zmp_margin']):.4f} m")
for name, value in fields.items():
    if value.dtype.kind in "fc" and not np.isfinite(value).all():
        raise ValueError(f"Nonfinite trajectory field: {name}")
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("wb") as output_file:
    np.savez_compressed(output_file, **fields)
print(f"Saved {count} trajectory samples: {args.output.resolve()}")
if schedule is not None:
    print("Applied gait changes [time, length, duration]:")
    print(fields["applied_gait_changes"])

if args.no_render:
    print(f"Generated {len(COM_pos_x)} samples; desired step length={args.step_length:g} m")
    raise SystemExit(0)

# ------------------------------------------------- animation plot
data_len = len(COM_pos_x)
print('--------- plot')
from matplotlib import gridspec
fig = plt.figure(figsize=(10, 10))
spec = gridspec.GridSpec(nrows=2, ncols=2, height_ratios=[2.5, 1])
ax = fig.add_subplot(spec[0,0], projection ='3d')
# ax.set_aspect('equal') # bugs
ax.set_xlim(-1.0, 4.0)
ax.set_ylim(-2.0, 2.0)
ax.set_zlim(-0.01, 1.)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')

# view angles
ax.view_init(20, -150)
LIPM_3D_ani = LIPM_3D_Animate()
# ani_3D = FuncAnimation(fig, ani_3D_update, frames=range(1, data_len), init_func=ani_3D_init, interval=1.0/LIPM_model.dt, blit=False, repeat=True)

# Add 2D plot for COM trajectory
bx = fig.add_subplot(spec[1,0], autoscale_on=False)
bx.set_xlim(-0.5, 5.0)
bx.set_ylim(-0.8, 0.8)
bx.set_aspect('equal')
bx.set_xlabel('x (m)')
bx.set_ylabel('y (m)')
bx.grid(ls='--')

COM_pos_str = 'COM = (%.2f, %.2f)'
ani_text_COM_pos = bx.text(0.05, 0.9, '', transform=bx.transAxes)


original_ani, = bx.plot(0, 0, marker='o', markersize=2, color='k')
left_foot_pos_ani, = bx.plot([], [], 'o', lw=2, color='b')
COM_traj_ani, = bx.plot(COM_pos_x[0], COM_pos_y[0], markersize=2, color='g')
COM_pos_ani, = bx.plot(COM_pos_x[0], COM_pos_y[0], marker='o', markersize=6, color='r')
left_foot_pos_ani, = bx.plot([], [], 'o', markersize=10, color='b')
right_foot_pos_ani, = bx.plot([], [], 'o', markersize=10, color='m')

# ani_2D = FuncAnimation(fig=fig, init_func=ani_2D_init, func=ani_2D_update, frames=range(1, data_len), interval=1.0/LIPM_model.dt, blit=False, repeat=True)

# Add CoM velocity plot
cx = fig.add_subplot(spec[0,1])
cx.set_xlim(0, total_time/LIPM_model.dt)
cx.set_ylim(min(min(COM_vel_x), min(COM_vel_y))-0.1, max(max(COM_vel_x), max(COM_vel_y))+0.1)
cx.set_xlabel('time (s)')
cx.set_ylabel('CoM velocity (m/s)')
cx.set_xticks(np.linspace(0, total_time/LIPM_model.dt, 6), labels=[f'{t:g}' for t in np.linspace(0, total_time, 6)])
cx.grid(ls='--')

COM_vel_x_ani, = cx.plot([], [], color='k', label='CoM velocity x')
COM_dvel_x_ani, = cx.plot([], [], color='k', linestyle='--', label='desired CoM velocity x')
COM_vel_y_ani, = cx.plot([], [], color='purple', label='CoM velocity y')
COM_dvel_y_ani, = cx.plot([], [], color='purple', linestyle='--', label='desired CoM velocity y')
cx.legend(loc='upper right')

# COM_vel_2D = FuncAnimation(fig=fig, init_func=COM_vel_2D_init, func=COM_vel_2D_update, frames=range(1, data_len), interval=1.0/LIPM_model.dt, blit=False, repeat=True)

# Add analysis plot
dx = fig.add_subplot(spec[1,1])
dx.set_xlim(0, total_time/LIPM_model.dt)
dx.set_ylim(min(min(step_length), min(step_width))-0.1, max(max(step_length), max(step_width))+0.1)
dx.set_xlabel('time (s)')
dx.set_ylabel('scale')
dx.set_xticks(np.linspace(0, total_time/LIPM_model.dt, 6), labels=[f'{t:g}' for t in np.linspace(0, total_time, 6)])
dx.grid(ls='--')

step_length_ani, = dx.plot([], [], color='gray', label='step length')
dstep_length_ani, = dx.plot([], [], color='gray', linestyle='--', label='desired step length')
step_width_ani, = dx.plot([], [], color='cyan', label='step width')
dstep_width_ani, = dx.plot([], [], color='cyan', linestyle='--', label='desired step width')
dx.legend(loc='upper right')

# step_params_2D = FuncAnimation(fig=fig, init_func=step_params_2D_init, func=step_params_2D_update, frames=range(1, data_len), interval=1.0/LIPM_model.dt, blit=False, repeat=True)

# * Combine all the animations
def _init_func():
    artist1 = ani_3D_init()
    artist2 = ani_2D_init()
    artist3 = COM_vel_2D_init()
    artist4 = step_params_2D_init()
    return artist1 + artist2 + artist3 + artist4

def _update_func(i):
    artist1 = ani_3D_update(i)
    artist2 = ani_2D_update(i)
    artist3 = COM_vel_2D_update(i)
    artist4 = step_params_2D_update(i)
    return artist1 + artist2 + artist3 + artist4

anim = FuncAnimation(fig=fig, init_func=_init_func, func=_update_func, frames=range(1, data_len), interval=1000 * LIPM_model.dt, blit=False, repeat=False)

# * Save the animation
print("--------- Play the animation")
if matplotlib.get_backend().lower() != "agg":
    plt.show()

print("--------- Save the animation")
if writers.is_available("ffmpeg"):
    filepath = os.path.join(os.getcwd(), "LIPM_vt.mp4")
    writer = FFMpegWriter(fps=1.0/LIPM_model.dt, codec="libx264")
else:
    filepath = os.path.join(os.getcwd(), "LIPM_vt.gif")
    writer = PillowWriter(fps=1.0/LIPM_model.dt)
    print("FFmpeg unavailable; saving GIF instead.")
anim.save(filepath, writer=writer)
print(f"Saved animation: {filepath}")

print('---------  Program terminated')
