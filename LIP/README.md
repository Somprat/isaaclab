# ModelBasedFootstepPlanning-IROS2024
This repository is an open-sourced code for the following paper presented at the 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS).

#### Title: Integrating Model-Based Footstep Planning with Model-Free Reinforcement Learning for Dynamic Legged Locomotion <br/> 
Paper Link: [https://ieeexplore.ieee.org/abstract/document/10801468](https://ieeexplore.ieee.org/abstract/document/10801468) <br/>
Video Link: [https://youtu.be/Z0E9AKt6RFo](https://youtu.be/Z0E9AKt6RFo)

### Installation ###
1. Create a new python virtual env with python 3.8 using Anaconda
2. Clone this repo
3. Install humanoidGym Requirements:
```bash
pip install -r requirements.txt
```
4. Install Isaac Gym
   - Download and install Isaac Gym Preview 4 (Preview 3 should still work) from https://developer.nvidia.com/isaac-gym
     - Extract the zip package
     - Copy the `isaacgym` folder, and place it in a new location
   - Install `issacgym/python` requirements
   ```bash
   cd <issacgym_location>/python
   pip install -e .
   ```
5. Install humanoidGym
    - go back to the humanoidGym repo, and install it.
    ```bash
    pip install -e .
    ```
---
## User Manual ##
## 1. Linear Inverted Pendulum Model (LIPM) Animation 
All the LIP model-related code is in the `LIPM` folder.
These codes are modified from [BipedalWalkingRobots](https://github.com/chauby/BipedalWalkingRobots) for the Center of Mass (CoM) velocity tracking task.

By running the code below, you should be able to get the following videos:
```bash
python LIPM/demo_LIPM_3D_vt.py
```

The demo accepts fixed gait parameters directly (example values are not validated
G1 settings):
```bash
python LIPM/demo_LIPM_3D_vt.py --step-length 0.20 --step-width 0.20 \
  --step-duration 0.40 --com-height 0.60 --heading 0 --swing-height 0.05
```
`--step-length` is the desired distance between alternating support feet, not a
full stride or an exact landing constraint. It remains fixed throughout the
rollout. Heading is in world radians; lengths are meters and duration is seconds.
The initial forward speed defaults to length / duration; override it using
`--initial-velocity`. Use `--no-render` to run without plots/video. CoM and foot
initial states still need to match your robot. This parameterizes the standalone
planner only; it does not connect it to Isaac Lab training.

<div align="center">
  <img src="https://github.com/user-attachments/assets/edff8522-b9d5-42d3-80af-37c0f0d50758">
</div>

By running the code below, you should be able to get the following videos and images:
```bash
python LIPM/demo_LIPM_3D_vt_analysis.py
```

<div align="center">
  <img src="https://github.com/user-attachments/assets/2e4e1600-2b34-4181-aaea-63e2288e85e7">
  <img width = "60%" src="https://github.com/user-attachments/assets/236682a4-8beb-4e46-b2a5-e4fe76a71978">
  <img width = "60%" src="https://github.com/user-attachments/assets/bd0f33c4-7ba8-4403-9647-2d1e61091263">
</div>

## 2. MIT Humanoid in IsaacGym
### Train ###  
```bash
python gym/scripts/train.py --task=humanoid_controller
```
-  To run on CPU add the following arguments: `--sim_device=cpu`, `--rl_device=cpu` (sim on CPU and rl on GPU is possible).
-  To run headless (no rendering) add `--headless`.
- **Important**: To improve performance, once the training starts press `v` to stop the rendering. You can then enable it later to check the progress.
- The trained policy is saved in `gym/logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`. Where `<experiment_name>` and `<run_name>` are defined in the train config.
-  The following command line arguments override the values set in the config files:
 - --task TASK: Task name.
 - --resume:   Resume training from a checkpoint
 - --experiment_name EXPERIMENT_NAME: Name of the experiment to run or load.
 - --run_name RUN_NAME:  Name of the run.
 - --load_run LOAD_RUN:   Name of the run to load when resume=True. If -1: will load the last run.
 - --checkpoint CHECKPOINT:  Saved model checkpoint number. If -1: will load the last checkpoint.
 - --num_envs NUM_ENVS:  Number of environments to create.
 - --seed SEED:  Random seed.
 - --max_iterations MAX_ITERATIONS:  Maximum number of training iterations.
 - --original_cfg:  Use configs stored in the saved files associated with the loaded policy instead of the current one in envs.

### Play (a trained policy) ###  
```bash
python gym/scripts/play.py --task=humanoid_controller
```
- By default the loaded policy is the last model of the last run of the experiment folder.
- Other runs/model iteration can be selected by setting `--load_run` and `--checkpoint`.
- You would need around 3,000 iterations of training to obtain a well-behaved policy.

<div align="center">
  <img src="https://github.com/user-attachments/assets/52acb865-057a-48bf-8b6c-4ec7f3415bca">
</div>

## 3. Deploy the policy to robot hardware
This repository does not include a code stack for deploying a policy to MIT Humanoid hardware.
Please check the [Cheetah-Software](https://github.com/mit-biomimetics/Cheetah-Software) for our lab's hardware code stack.

To deploy the trained policy, you would need to set `EXPORT_POLICY=TRUE` in the `humanoidGym/scripts/play.py` script.
Then you would get a `policy.onnx` file to run on C++ code.

---
### System Info ###
Operating System: Ubuntu 22.04 <br/>
GPU: Geforce 3090

---
### Troubleshooting ###
1. If you get the following error: `ImportError: libpython3.8m.so.1.0: cannot open shared object file: No such file or directory`, do: `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH`

---
### Acknowledgement ###
We would appreciate it if you would cite it in academic publications:
```
@inproceedings{lee2024integrating,
  title={Integrating model-based footstep planning with model-free reinforcement learning for dynamic legged locomotion},
  author={Lee, Ho Jae and Hong, Seungwoo and Kim, Sangbae},
  booktitle={2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={11248--11255},
  year={2024},
  organization={IEEE}
}
```

### Export step-conditioned task-space references

Every run of `LIPM/demo_LIPM_3D_vt.py` now saves a compressed NPZ, including
when `--no-render` is selected. The default is `LIPM_trajectory.npz` in the
working directory. An existing output file is replaced.

```bash
python LIPM/demo_LIPM_3D_vt.py --step-length 0.20 --duration 10 \
  --no-render --output generated_trajectories/step_020.npz
```

Load with `np.load(path, allow_pickle=False)`. Core fields (schema 1 and 2) include:

- `time` (N,), `dt` scalar: seconds; the first sample is at t=0 (older exports started at dt).
- `com_pos`, `com_vel` (N,3): world position [m] and analytical velocity [m/s].
- `foot_pos` (N,2,3): world foot positions [m], ordered left then right.
- `support_mask` (N,2): intended supporting feet, not measured contact. Schema 2
  supports two-foot startup and changes support at touchdown. Legacy schema 1
  labelled the old support at the touchdown sample.
- `phase` (N,): full-cycle fraction in [0,1), left support starts at 0 and right
  support at 0.5. This is the planner clock, not contact estimation.
- `desired_step_length`, `desired_step_width` (N,): nominal gait commands [m].
- Scalar `step_duration`, `heading`, `com_height`, `swing_height`, plus initial
  horizontal `initial_velocity` (2,), `frame`, `foot_names`, `schema_version`.

Coordinates are world Z-up on flat ground. The clip contains no joint targets,
root orientations, torques, or measured contacts. It needs a task-space command
adapter and foot/CoM rewards for Isaac Lab; it is not compatible with the existing
full-body motion loader without that adapter. Example gait settings have not
been validated for G1. `--headless` saves animation without a desktop;
`--no-render` skips animation while retaining the NPZ export.

### Inspect a reference against G1 (headless MP4)

From the parent workspace directory, using the local IsaacLab environment:

```bash
uv run --project IsaacLab --no-sync python LIP/LIPM/inspect_g1_reference.py \
  --motion LIP/generated_trajectories/step_020.npz \
  --output LIP/previews/g1_lipm --duration 2
```

This writes `g1_lipm_comparison.mp4`, `startup.png`, `summary.json`, and the
per-frame `comparison.json`. Orange markers are reference soles/CoM; cyan markers
are measurements. The clip includes a startup hold and half-speed playback.
The robot uses nominal-pose PD with zero policy actions: this is a geometry and
startup diagnostic, not a trained tracking demonstration. Falls remain visible
instead of resetting. Reference and actual points are compared at the same time
(the diagnostic does not display the policy's one-step-ahead target). Current
output files at the selected destination are replaced on rerun.

### Regenerate from the measured G1 reset pose

The `lipm` preset uses a calibrated 0.78420225 m root height and zero initial root/joint velocities.
Capture its mass-weighted CoM and sole geometry before physics, then generate:

```bash
uv run --project IsaacLab --no-sync python LIP/LIPM/capture_g1_initial_state.py \
  --output LIP/initial_states/g1_reset.json
uv run --project IsaacLab --no-sync python LIP/LIPM/demo_LIPM_3D_vt.py \
  --initial-state LIP/initial_states/g1_reset.json \
  --step-length 0.20 --step-duration 0.40 --duration 10 --no-render \
  --output LIP/generated_trajectories/step_020_g1_reset.npz
uv run --project IsaacLab --no-sync python LIP/LIPM/inspect_g1_reference.py \
  --motion LIP/generated_trajectories/step_020_g1_reset.npz \
  --output LIP/previews/g1_lipm_reset
```

`--initial-state` overrides initial CoM position and velocity. Initial foot XY
comes from the measurement; measured sole Z is projected to the ground plane
because this LIPM assumes flat-ground support. Unless explicitly supplied,
`--step-width` is inferred from measured foot separation along the lateral axis.
The exact initial CoM state is included at t=0 in the NPZ. The source JSON path
and foot projection convention are recorded in the export metadata.

Both feet start at the ground plane within floating-point tolerance. This is
not a settled single-support stance: the planner still assumes left-foot support
initially. Matching geometry/velocity does not solve the standing-to-walking
support transfer. The capture tool checks all four collision-sphere bottoms per
foot for the matched G1 URDF, then records five nominal-PD physics steps of contact
forces (configurable with `--contact-check-steps`). Those contact-check samples do
not replace the initial state saved before physics. It can run before a reference
NPZ exists.

Before generating a new NPZ, choose step length, step duration and swing height;
step width defaults to the measured lateral foot separation. Rerun state capture
when changing the robot model, reset joint pose, sole offsets, or root height.
The currently measured CoM height is 0.714414 m. Both feet load during 0.02–0.08 s
in the short diagnostic, but lose load at 0.10 s: this checks contact onset, not
settled standing or a stable walking policy.

### Smooth two-foot startup (current default)

The reusable generator is `LIPM/trajectory.py`. Every new export now includes a
planned two-foot load-transfer phase followed by smooth swing motion. Existing
NPZs do not change automatically. The Isaac Lab default is now
`generated_trajectories/step_020_g1_startup.npz`, regenerated from the measured G1
reset. Its startup is 0.8 s; the first landing occurs at 1.2 s.

```bash
uv run --project IsaacLab --no-sync python LIP/LIPM/demo_LIPM_3D_vt.py \
  --initial-state LIP/initial_states/g1_reset.json \
  --step-length 0.20 --step-duration 0.40 --startup-duration 0.8 \
  --duration 10 --no-render \
  --output LIP/generated_trajectories/step_020_g1_startup.npz
```

The startup endpoint is solved automatically from the desired first landing.
`--startup-duration` is the minimum; it can increase to `--max-startup-duration`
(default 2 s) to satisfy support/friction checks. `--startup-duration 0` disables
it for legacy comparisons. With startup enabled, manual initial velocity defaults
to zero; a measured-state JSON overrides it. Duration includes startup.

New exports use schema 2. Alongside the existing fields, they contain analytical
`foot_vel`, `foot_acc`, `com_acc`, `zmp`, `startup_duration`, support polygon and
startup feasibility metadata. `support_mask` may be `[true,true]` during startup;
it never indicates flight. Schema-2 playback preserves smooth velocities between
samples. The original schema-1 files remain readable by the updated adapter.
The LIPM startup ZMP is checked inside the foot support hull, but whole-body
reachability and learned tracking still require simulation evaluation.

### Scheduled step length and duration

Edit `gait_schedules/steps_010_020_030.json`. Times are seconds from clip start,
including the initial weight transfer. `transition_steps: 2` applies the halfway
setting at the first touchdown at/after the request and the final setting at the
next touchdown. `1` applies the full setting immediately at that first touchdown;
it is more abrupt and has not been validated on the trained robot. Changes never
retime a swing already in progress. If another request arrives during a blend,
the next touchdown starts blending from the currently active settings.

From `/home/choon/isaaclab`, regenerate with:

```bash
IsaacLab/.venv/bin/python LIP/LIPM/demo_LIPM_3D_vt.py \
  --initial-state LIP/initial_states/g1_reset.json \
  --gait-schedule LIP/gait_schedules/steps_010_020_030.json \
  --no-render --output LIP/generated_trajectories/steps_010_020_030.npz
```

The JSON overrides `--duration`, `--step-length`, and `--step-duration`. Width,
height and reset geometry retain the existing settings. Scheduled exports use
schema 3 with per-sample length/duration commands, held until each touchdown by
the IsaacLab loader. They use a fixed 5 ms sampling interval; step durations are
rounded to that interval. `gait_schedule` stores requested [time, length, duration]
rows; `applied_gait_changes` stores actual setting changes. The initial row is the
first swing start (the first settings also apply during startup).

For the supplied schedule, settings change at 6.750 / 7.125 s and 13.525 / 13.950 s.
The first full-setting steps land at 7.525 and 14.400 s. These are nominal length
commands; transient landing distances can differ under XCoM placement. CoM
position/velocity remain continuous, while support-switch acceleration changes
are still part of the underlying single-support LIPM. This is not a whole-body
feasibility guarantee.

The existing fixed-gait reference and training preset default are unchanged.
Select this NPZ with the training override:
`env.commands.base_velocity.motion_directory=/home/choon/isaaclab/LIP/generated_trajectories/steps_010_020_030.npz`.
Retraining/fine-tuning is required to establish tracking of the scheduled gait.
