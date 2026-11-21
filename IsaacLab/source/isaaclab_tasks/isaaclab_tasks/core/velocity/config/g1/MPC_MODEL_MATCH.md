# G1 model parity for MPC reference training

The reference preset uses a separate asset generated from the same
`g1_description/urdf/g1_29dof.urdf` that the MPC exporter loads. The URDF in the
running ROS installation and the source file have the same SHA256:
`157d9820e6aa3f5b9d92246e9c1b2710e31461a12cb2eb29605fe3755142e25e`.
The original pretrained asset and `policy.onnx` are not edited.

## What was different

The original USD totals approximately 33.34 kg; the MPC URDF totals 35.11514202 kg.
The significant mass/inertia differences are in the waist and torso:

| Composite link | Previous USD mass (kg) | MPC mass (kg) |
| --- | ---: | ---: |
| waist_yaw_link | 0.214 | 0.244 |
| waist_roll_link | 0.086 | 0.047 |
| torso_link, including fixed attachments | 7.817 | 9.600 |

The wrist/hand composite inertias already matched. Waist/torso frame origins
also differed by 9–10 mm. The previous reference preset overrode some source
motor limits, added 0.01 rotor armature and used 0.8/0.6 robot friction rather
than the MPC contact coefficient of 0.4.

## Current matched setup

- USD: `checkpoints/g1-29dof/mpc-matched-assets/g1_29dof/g1_29dof.usda`.
- Fixed URDF joints are merged, preserving composite mass, COM and inertia.
- All 29 joint frames, axes, position, velocity and effort limits follow the URDF.
- Rotor armature is zero, as in the MPC rigid-body model.
- Collision shapes derive from the URDF; mesh collisions are convex hulls.
- Ground and every robot shape have friction 0.4 and zero restitution.
- Material/mass/push randomization is disabled for this initial integration.
- Locomotion PD gains and the chosen position-action interface remain controller
  settings; ONNX fine-tuning retains its deployment gains, motor order and scales.

The contact sensor follows the converted asset's nested Geometry hierarchy.
At startup, a guard checks source URDF, source mesh and generated USD fingerprints.
Rebuild if any fingerprint changes. Only the reference preset is affected; the
normal walking preset and original checkpoint assets are unchanged. Older saved
runs retain the asset path recorded in their configuration snapshots.

## Validation

Independent fixed-link aggregation agrees with Pinocchio to 1.4e-17 kg m^2 for
this source model. Runtime checks use the robot actually loaded into Newton:

| Quantity | Maximum discrepancy |
| --- | ---: |
| Per-body mass | 3.82e-7 kg |
| Per-body COM | 7.32e-9 m |
| Per-body inertia tensor entry | 3.05e-9 kg m^2 |
| Link position across five reference poses | 1.04e-6 m |
| Link orientation across five reference poses | 8.00e-7 rad |
| Full mass-matrix entry against Pinocchio | 5.73e-7 |
| Gravity generalized force against Pinocchio | 1.97e-5 |

The matrix/gravity check evaluates the Newton solver's MuJoCo model at five
poses from the saved motion. It explicitly reorders joint coordinates and
converts Pinocchio's local free-base linear velocity convention to MuJoCo's
world linear velocity convention. All source joint limits match within float
precision. All loaded shape friction values are 0.400000006 and armature is zero.

The imported ONNX actor was also checked for 100 inference-only simulation steps
with the new asset: history/action contracts and initial action parity passed.
No PPO optimizer updates were run during this model-matching work.

## Rebuild

Run from `/home/choon/isaaclab/IsaacLab`:

```bash
uv run --no-sync --with isaacsim-asset-isolated==6.1.0.0 \
  --extra-index-url https://pypi.nvidia.com \
  python scripts/reinforcement_learning/prepare_g1_mpc_asset.py
```

This regenerates only the dedicated matched asset directory. It does not alter
the MPC source URDF, source meshes, saved NPZ files or original pretrained USD.
The isolated importer is used without changing the main uv environment.

The fine-tuning script uses the matched model automatically for new runs.
The one-off Pinocchio exporters, physics audit scripts and trial outputs were
removed after validation; the measured results above remain the validation
record. The runtime source/asset fingerprint guard remains enabled.

## Remaining differences

This establishes rigid-body parameter and free-motion dynamics parity, not
identical contact trajectories. MPC integrates a centroidal model with planned
contact modes/wrenches and soft friction/contact constraints. Newton/MuJoCo-Warp
resolves physical contact, impacts and actuator commands. In particular:

- A planned support foot can be unloaded in the simulator.
- MPC's support rectangle approximates the URDF's rounded contact spheres.
- MPC torque costs/URDF limits do not constitute a validated torque controller.
- Matching dynamics does not guarantee the saved reference is physically
  trackable or that an initial policy will balance.

Earlier PD/balance-controller failure measurements used the old asset and must
not be interpreted as measurements of the newly matched setup. PPO training is
still needed to evaluate the intended learned reference tracker.
