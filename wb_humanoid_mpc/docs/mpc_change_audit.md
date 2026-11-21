# MPC change audit

Audit against the wb_humanoid_mpc repository's current HEAD. This is an inventory,
not a claim that every diff was authored during the latest task. The workspace
already contained modifications when the pelvis-height request began.

## Solver and dynamics

- `lib/ocs2_ros2`: git status is clean. No SQP, HPIPM, integration, or other OCS2
  solver implementation was edited.
- `humanoid_nmpc/humanoid_centroidal_mpc/src/dynamics/`: no changes against HEAD.
  State/control dimensions and the centroidal dynamics equations were not rewritten.
- Solver configuration in `task_random_reference.info` *was* changed: SQP
  iterations increased from 1 to 5. The current setting is 5; 10-iteration
  configurations were used during unsuccessful trials and are not the final default.
- `DynamicsHelperFunctions.cpp` has a pre-existing Pinocchio-version compatibility
  branch (`parentJoint` versus `parent`), not a new force/dynamics formulation.
- The pre-existing target-calculator diff changes momentum/rate conversion and
  yaw indexing. These affect reference generation across modes and deserve a
  separate compatibility review; they are not required merely to accept L/T inputs.

## Every modified original tracked file (21)

Paths below are relative to wb_humanoid_mpc.

| Original file | Reason for its current diff | Already modified before pelvis-height work? |
|---|---|---|
| `README.md` | Generator usage documentation | Yes |
| `docker/Dockerfile` | Pinocchio/ROS environment compatibility | Yes |
| `humanoid_nmpc/humanoid_centroidal_mpc/CMakeLists.txt` | Register posture and stance-feedback tests | No |
| `humanoid_nmpc/humanoid_centroidal_mpc/include/humanoid_centroidal_mpc/command/CentroidalMpcTargetTrajectoriesCalculator.h` | Conditioning-mode reference-smoothing switch | Yes |
| `humanoid_nmpc/humanoid_centroidal_mpc/src/command/CentroidalMpcTargetTrajectoriesCalculator.cpp` | Smoothing switch; pre-existing momentum normalization/yaw-rate reference corrections | Yes |
| `humanoid_nmpc/humanoid_centroidal_mpc/src/CentroidalMpcInterface.cpp` | Register optional pelvis/knee penalties and lateral swing constraint | No |
| `humanoid_nmpc/humanoid_centroidal_mpc/src/constraint/ZeroVelocityConstraintCppAd.cpp` | Step-mode stance X/Y anchors and fixed-heading feedback | No |
| `humanoid_nmpc/humanoid_centroidal_mpc/src/cost/CentroidalMpcEndEffectorFootCost.cpp` | Step-conditioned X/Y targets; later fixed-heading yaw cost and versioned AD cache | Yes; later extended |
| `humanoid_nmpc/humanoid_centroidal_mpc_ros2/CMakeLists.txt` | Reference/conditioning/step-plan tests | Yes |
| `humanoid_nmpc/humanoid_centroidal_mpc_ros2/src/CentroidalMpcRandomReferenceGenerator.cpp` | CLI L/T conditioning, metadata, neutral upper-body mode, correct startup order, fixed step heading | Yes; later extended |
| `humanoid_nmpc/humanoid_common_mpc/CMakeLists.txt` | Pinocchio library discovery/link compatibility | Yes |
| `humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/constraint/EndEffectorKinematicsTwistConstraint.h` | Optional full-orientation reference for stance | No |
| `humanoid_nmpc/humanoid_common_mpc/src/constraint/EndEffectorKinematicsTwistConstraint.cpp` | Consistent full-orientation residual/Jacobian when explicitly supplied | No |
| `humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h` | Store/share immutable step plan | Yes |
| `humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/swing_foot_planner/SplineCpg.h` | Optional smooth vertical swing profile | No |
| `humanoid_nmpc/humanoid_common_mpc/src/swing_foot_planner/SplineCpg.cpp` | C2 swing-height profile with smooth endpoints | No |
| `humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h` | Opt-in smoothSwing config | No |
| `humanoid_nmpc/humanoid_common_mpc/src/swing_foot_planner/SwingTrajectoryPlanner.cpp` | Load/use smoothSwing and validate zero endpoint velocities | No |
| `humanoid_nmpc/humanoid_common_mpc/src/cost/StateInputQuadraticCost.cpp` | Step-mode smooth vertical load reference during double support | No |
| `humanoid_nmpc/humanoid_common_mpc/src/pinocchio_model/DynamicsHelperFunctions.cpp` | Pinocchio-version compatibility only | Yes |
| `robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task_random_reference.info` | Existing velocity weights plus later posture, swing, force-transfer, heading, and iteration settings | Yes; later extended |

## Added implementation files

- `humanoid_nmpc/humanoid_centroidal_mpc_ros2/include/humanoid_centroidal_mpc_ros2/MotionConditioning.h`
- `humanoid_nmpc/humanoid_centroidal_mpc_ros2/include/humanoid_centroidal_mpc_ros2/StepConditioning.h`
- `humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/swing_foot_planner/StepReferencePlan.h`
- `humanoid_nmpc/humanoid_centroidal_mpc/include/humanoid_centroidal_mpc/constraint/PelvisHeightConstraint.h`
- `humanoid_nmpc/humanoid_centroidal_mpc/include/humanoid_centroidal_mpc/constraint/KneeFlexionConstraint.h`
- `humanoid_nmpc/humanoid_centroidal_mpc/include/humanoid_centroidal_mpc/constraint/SwingFootLateralConstraint.h`

There are also new tests, generation/measurement/replay scripts, documentation,
Docker setup files, and generated artifacts. They are not original MPC files.
The unsuccessful full-horizontal swing constraint was removed.

## Scope assessment

Independent step length/duration control needs command scheduling, fixed foot
anchors/targets, a prescribed contact schedule, reference-manager plumbing,
exported metadata and tests. It does not require new dynamics or solver code.

The later pelvis/knee penalties, smooth swing profile, force-transfer cost,
stance corrections, lateral swing constraint, heading cost and configuration
tuning go beyond that minimal conditioning change. They followed subsequent
requests to improve posture/startup/sideways motion. They should be separately
reviewable and optional, rather than presented as necessary for L/T conditioning.

Most new mechanics are gated by a step plan or explicit task-file flags.
However, full legacy equivalence has not been demonstrated: target-calculator
corrections are shared, the generator task configuration is shared across its
modes, and shared planner/constraint code was edited. Passing selected tests
and generating clips is not proof of unchanged legacy behavior.

No additional tuning or rollback was performed as part of this audit.
