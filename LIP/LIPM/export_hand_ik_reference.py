"""Append offline arm IK references to a LIP NPZ, without changing its walking data.

Run in the IsaacLab uv environment (no simulator or GPU is required)::

    cd /home/choon/isaaclab/IsaacLab
    uv run --no-sync python ../LIP/LIPM/export_hand_ik_reference.py \
        --motion ../LIP/generated_trajectories/steps_010_020_030_first050.npz \
        --output ../LIP/generated_trajectories/steps_010_020_030_hands.npz --seed 42

Defaults are loaded from LIP/hand_schedules/random_reach.json. Use --config to
select another JSON; explicit command-line options override its settings.
JSON keys match the CLI names with underscores. Distances are meters, times
seconds, joint angles radians and joint speeds rad/s. Random forward/outward/up
ranges are offsets from the reset hand positions; outward is mirrored for the
right hand. A null left_target/right_target selects random sampling. After
start_delay and reach_duration, the goal is held for the remainder of the clip.

Each selected hand approaches one goal and holds it for the rest of the clip.
Targets are positions of the URDF rubber-hand frame origins in ``torso_link``
coordinates (X forward, Y left, Z up). Orientations are unconstrained. Goals
default to seeded Cartesian offsets from the reset hands; explicit torso-frame
goals can be supplied with --left-target / --right-target. Only the selected
arm's seven joints participate in each solve; waist and legs stay fixed for FK.
Pinocchio supplies FK/Jacobians; IsaacLab DifferentialIKController supplies DLS.

The source schema_version and every original array are preserved. The extension
has its own arm_reference_schema_version=1, arm_joint_names, arm_joint_pos/vel
[T,J], hand_names, hand_frame_names, hand_pos_torso [T,H,3], and FK diagnostics.
All arm rows use the original time array. The current LIP adapter ignores these
extra fields: arm observations/rewards still need a separate integration change.

Validation covers sampled kinematic reachability, position/velocity limits and
finite values. It does not establish collision-free motion or dynamic balance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np


LIP_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = LIP_ROOT.parent
DEFAULT_URDF = WORKSPACE / "wb_humanoid_mpc/robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf"
DEFAULT_HAND_SCHEDULE = LIP_ROOT / "hand_schedules/random_reach.json"
ARM_SUFFIXES = (
    "shoulder_pitch_joint", "shoulder_roll_joint", "shoulder_yaw_joint",
    "elbow_joint", "wrist_roll_joint", "wrist_pitch_joint", "wrist_yaw_joint",
)


class ArmIK:
    """Fixed-torso FK and an existing position-only DLS controller for one arm."""

    def __init__(self, urdf, initial_state, side, args):
        import pinocchio as pin
        import torch
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

        self.pin, self.torch = pin, torch
        self.model = pin.buildModelFromUrdf(str(urdf))
        self.data = self.model.createData()
        names = initial_state["joint_names"]
        values = np.asarray(initial_state["joint_pos"], dtype=float)
        if len(set(names)) != len(names) or values.shape != (len(names),) or not np.isfinite(values).all():
            raise ValueError("Reset state must contain unique joint names and finite matching joint_pos")
        if set(names) != set(self.model.names[1:]):
            raise ValueError("Reset joint names do not match the URDF")
        self.q = pin.neutral(self.model)
        for name, value in zip(names, values):
            joint = self.model.joints[self.model.getJointId(name)]
            if joint.nq != 1 or joint.nv != 1:
                raise ValueError(f"Expected one-DOF G1 joint: {name}")
            self.q[joint.idx_q] = value
        self.names = [f"{side}_{suffix}" for suffix in ARM_SUFFIXES]
        joints = [self.model.joints[self.model.getJointId(name)] for name in self.names]
        self.q_ids = np.array([j.idx_q for j in joints])
        self.v_ids = np.array([j.idx_v for j in joints])
        self.lower = self.model.lowerPositionLimit[self.q_ids] + args.joint_limit_margin
        self.upper = self.model.upperPositionLimit[self.q_ids] - args.joint_limit_margin
        self.speed_limits = np.minimum(self.model.velocityLimit[self.v_ids], args.max_joint_speed)
        self.initial = self.q[self.q_ids].copy()
        if np.any(self.lower >= self.upper) or np.any(self.initial < self.lower) or np.any(self.initial > self.upper):
            raise ValueError(f"{side}: reset arm pose violates joint limits/margin")
        velocities = np.asarray(initial_state["joint_vel"], dtype=float)
        if velocities.shape != values.shape or not np.isfinite(velocities).all():
            raise ValueError("Invalid reset joint_vel")
        if np.any(np.abs(velocities[[names.index(n) for n in self.names]]) > 1e-8):
            raise ValueError("This exporter requires stationary reset arms")
        self.hand_frame = f"{side}_rubber_hand"
        for frame in ("torso_link", self.hand_frame):
            if not self.model.existFrame(frame):
                raise ValueError(f"URDF is missing frame {frame}")
        self.hand_id = self.model.getFrameId(self.hand_frame)
        self.torso_id = self.model.getFrameId("torso_link")
        self.controller = DifferentialIKController(
            DifferentialIKControllerCfg(
                command_type="position", use_relative_mode=False, ik_method="dls",
                ik_params={"lambda_val": args.damping},
            ),
            num_envs=1, device="cpu",
        )
        self.args = args

    def tensor(self, value):
        return self.torch.as_tensor(np.asarray(value).copy(), dtype=self.torch.float32).unsqueeze(0)

    def kinematics(self, arm_q):
        self.q[self.q_ids] = arm_q
        pin = self.pin
        pin.computeJointJacobians(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)
        torso, hand = self.data.oMf[self.torso_id], self.data.oMf[self.hand_id]
        rotation = torso.rotation.T
        position = rotation @ (hand.translation - torso.translation)
        quaternion = pin.Quaternion(rotation @ hand.rotation).coeffs()
        jacobian = pin.getFrameJacobian(
            self.model, self.data, self.hand_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )[:, self.v_ids].copy()
        # The torso is fixed with respect to the controlled arm joints. Rotate
        # both Jacobian blocks into the same torso axes used by the target.
        jacobian[:3] = rotation @ jacobian[:3]
        jacobian[3:] = rotation @ jacobian[3:]
        return position.copy(), quaternion.copy(), jacobian

    def solve(self, target, seed):
        q = seed.copy()
        for iteration in range(self.args.max_iterations + 1):
            position, quaternion, jacobian = self.kinematics(q)
            error = float(np.linalg.norm(target - position))
            if error <= self.args.position_tolerance:
                return q, position, iteration
            if iteration == self.args.max_iterations:
                break
            self.controller.set_command(self.tensor(target), ee_quat=self.tensor(quaternion))
            proposed = self.controller.compute(
                self.tensor(position), self.tensor(quaternion), self.tensor(jacobian), self.tensor(q)
            )[0].numpy().astype(float)
            if not np.isfinite(proposed).all():
                raise ValueError("IK produced nonfinite joints")
            delta = proposed - q
            delta *= min(1.0, self.args.max_ik_step / max(float(np.max(np.abs(delta))), 1e-12))
            q = np.clip(q + delta, self.lower, self.upper)
        raise ValueError(f"IK failed to converge: position error {error:.6g} m")


def load_motion(path):
    """Validate using the existing adapter, then retain every original array."""
    from isaaclab_tasks.core.velocity.mdp.lipm_motion import LipmMotionLibrary

    if not path.is_file() or path.suffix != ".npz":
        raise ValueError("--motion must be a single existing NPZ file")
    LipmMotionLibrary(str(path), "cpu")
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: source[key].copy() for key in source.files}
    if any(key.startswith(("arm_", "hand_")) for key in arrays):
        raise ValueError("Source already contains arm/hand fields; use an original LIP reference")
    return arrays


def generate_arm(solver, times, side, rng, args):
    start, _, _ = solver.kinematics(solver.initial)
    explicit = getattr(args, f"{side}_target")
    u = np.clip((times - times[0] - args.start_delay) / args.reach_duration, 0, 1)
    blend = u**3 * (10 - 15 * u + 6 * u**2)
    last_failure = ""
    for attempt in range(1, (1 if explicit is not None else args.max_attempts) + 1):
        if explicit is not None:
            goal = np.asarray(explicit, dtype=float)
        else:
            offset = np.array([rng.uniform(*bounds) for bounds in
                               (args.forward_range, args.outward_range, args.up_range)])
            offset[1] *= 1 if side == "left" else -1
            goal = start + offset
        targets = start + blend[:, None] * (goal - start)
        q = solver.initial.copy()
        positions, achieved, iterations = [], [], []
        try:
            # Quickly reject unreachable endpoints before processing the clip.
            solver.solve(goal, q)
            for target in targets:
                q, actual, count = solver.solve(target, q)
                positions.append(q.copy())
                achieved.append(actual)
                iterations.append(count)
            positions, achieved = np.asarray(positions), np.asarray(achieved)
            velocities = np.gradient(positions, times, axis=0, edge_order=1)
            segment_speed = np.abs(np.diff(positions, axis=0) / np.diff(times)[:, None])
            if (segment_speed > solver.speed_limits + 1e-6).any() or (
                np.abs(velocities) > solver.speed_limits + 1e-6
            ).any():
                raise ValueError("Joint speed limit exceeded; increase --reach-duration or reduce goal range")
            errors = np.linalg.norm(achieved - targets, axis=-1)
            if not all(np.isfinite(a).all() for a in (positions, velocities, achieved, errors)):
                raise ValueError("Nonfinite reference")
            if (positions < solver.lower - 1e-7).any() or (positions > solver.upper + 1e-7).any():
                raise ValueError("Joint position limit exceeded")
            if errors.max() > args.position_tolerance:
                raise ValueError("FK residual exceeds position tolerance")
            return {
                "joint_pos": positions, "joint_vel": velocities, "target": targets,
                "actual": achieved, "error": errors, "goal": goal,
                "iterations": np.asarray(iterations, dtype=np.int32), "attempts": attempt,
            }
        except ValueError as exc:
            last_failure = str(exc)
    raise ValueError(f"{side} reference rejected: {last_failure}")


def save_npz(path, arrays, overwrite):
    """Publish a complete archive, preserving an existing output unless requested."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)  # Atomic create; fails if another writer created path.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_hand_schedule(path):
    """Read typed reach defaults; reject misspelled keys and unsupported frames."""
    config = json.loads(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("Hand schedule must be a JSON object")
    if type(config.get("schema_version")) is not int or config["schema_version"] != 1:
        raise ValueError("Hand schedule requires schema_version=1")
    if config.get("target_frame") != "torso_link":
        raise ValueError("Hand schedule requires target_frame='torso_link'")
    floats = {
        "start_delay", "reach_duration", "position_tolerance", "joint_limit_margin",
        "max_joint_speed", "max_ik_step", "damping",
    }
    integers = {"seed", "max_iterations", "max_attempts"}
    vectors = {"forward_range": 2, "outward_range": 2, "up_range": 2,
               "left_target": 3, "right_target": 3}
    allowed = floats | integers | vectors.keys() | {"schema_version", "target_frame", "hands"}
    if unknown := config.keys() - allowed:
        raise ValueError(f"Unknown hand schedule keys: {', '.join(sorted(unknown))}")
    for name in floats | integers:
        if name not in config:
            continue
        value = config[name]
        numeric = type(value) in (int, float) and np.isfinite(value)
        if not numeric or (name in integers and type(value) is not int):
            raise ValueError(f"Hand schedule {name} must be a finite {'integer' if name in integers else 'number'}")
    for name, size in vectors.items():
        if name not in config or (name.endswith("_target") and config[name] is None):
            continue
        value = config[name]
        if not isinstance(value, list) or len(value) != size or not all(
            type(v) in (int, float) and np.isfinite(v) for v in value
        ):
            raise ValueError(f"Hand schedule {name} must contain {size} finite numbers")
    if "hands" in config and config["hands"] not in ("left", "right", "both"):
        raise ValueError("Hand schedule hands must be 'left', 'right', or 'both'")
    return {key: value for key, value in config.items() if key not in ("schema_version", "target_frame")}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_HAND_SCHEDULE,
                        help="Hand schedule JSON; CLI options override it (default: %(default)s)")
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--initial-state", type=Path, default=LIP_ROOT / "initial_states/g1_reset.json")
    parser.add_argument("--hands", choices=("left", "right", "both"), default="both")
    parser.add_argument("--left-target", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--right-target", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-delay", type=float, default=0.8, help="Seconds from the first LIP sample")
    parser.add_argument("--reach-duration", type=float, default=2.0)
    parser.add_argument("--forward-range", type=float, nargs=2, default=(0.08, 0.18), metavar=("MIN", "MAX"))
    parser.add_argument("--outward-range", type=float, nargs=2, default=(0.0, 0.06), metavar=("MIN", "MAX"))
    parser.add_argument("--up-range", type=float, nargs=2, default=(0.04, 0.12), metavar=("MIN", "MAX"))
    parser.add_argument("--position-tolerance", type=float, default=1e-5, help="FK position error tolerance in meters")
    parser.add_argument("--joint-limit-margin", type=float, default=0.02, help="Radians inside URDF limits")
    parser.add_argument("--max-joint-speed", type=float, default=2.0, help="rad/s, additionally capped by URDF limits")
    parser.add_argument("--max-ik-step", type=float, default=0.1, help="Maximum radians per solver iteration")
    parser.add_argument("--damping", type=float, default=0.01)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--max-attempts", type=int, default=20, help="Random goal attempts per selected hand")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        parser.set_defaults(**load_hand_schedule(args.config))
    except (OSError, ValueError) as exc:
        parser.error(f"Cannot load hand schedule {args.config}: {exc}")
    # Parse again with JSON defaults so explicitly supplied CLI values win.
    args = parser.parse_args(argv)
    for name in ("reach_duration", "position_tolerance", "max_joint_speed", "max_ik_step", "damping"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    for name in ("start_delay", "joint_limit_margin"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
    if args.max_iterations < 1 or args.max_attempts < 1 or args.seed < 0:
        parser.error("Iteration/attempt limits must be positive and seed nonnegative")
    for name in ("forward_range", "outward_range", "up_range"):
        bounds = getattr(args, name)
        if not np.isfinite(bounds).all() or bounds[0] > bounds[1]:
            parser.error(f"Invalid --{name.replace('_', '-')} bounds")
    for side in ("left", "right"):
        target = getattr(args, f"{side}_target")
        if target is not None and (not np.isfinite(target).all() or args.hands not in (side, "both")):
            parser.error(f"--{side}-target requires a finite target and that hand selected")
    if args.motion.resolve() == args.output.resolve():
        parser.error("Output must differ from the source LIP NPZ")
    if args.output.suffix != ".npz":
        parser.error("Output must end in .npz")
    if args.output.exists() and not args.overwrite:
        parser.error("Output already exists; use a new path or --overwrite")
    return args


def main():
    args = parse_args()
    arrays = load_motion(args.motion)
    times = np.asarray(arrays["time"], dtype=float)
    if times[-1] - times[0] < args.start_delay + args.reach_duration:
        raise ValueError("Reference is too short for --start-delay plus --reach-duration")
    if not np.any((times - times[0] > args.start_delay) & (
        times - times[0] < args.start_delay + args.reach_duration
    )):
        raise ValueError("Reference has no samples inside the reach; use a finer LIP timestep")
    initial_state = json.loads(args.initial_state.read_text())
    sides = ["left", "right"] if args.hands == "both" else [args.hands]
    results, solvers = [], []
    for side in sides:
        solver = ArmIK(args.urdf, initial_state, side, args)
        # Each hand has an independent seed stream, unchanged by --hands selection.
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, int(side == "right")]))
        result = generate_arm(solver, times, side, rng, args)
        results.append(result)
        solvers.append(solver)
        print(f"{side}: goal={result['goal'].tolist()}, attempts={result['attempts']}, "
              f"max FK error={result['error'].max():.3g} m", flush=True)
    settings = {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()}
    metadata = {
        "settings": settings,
        "source_npz_sha256": hashlib.sha256(args.motion.read_bytes()).hexdigest(),
        "source_urdf_sha256": hashlib.sha256(args.urdf.read_bytes()).hexdigest(),
        "initial_state_sha256": hashlib.sha256(args.initial_state.read_bytes()).hexdigest(),
        "hand_schedule_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "solver": "isaaclab.controllers.DifferentialIKController/dls",
        "kinematics": "pinocchio", "pinocchio_version": solvers[0].pin.__version__,
        "orientation_constrained": False,
        "validation": "sampled FK residuals, joint position margins, finite values, joint speeds",
        "collision_checked": False, "dynamics_checked": False,
        "attempts": {side: r["attempts"] for side, r in zip(sides, results)},
    }
    arrays.update({
        "arm_reference_schema_version": np.array(1, dtype=np.int64),
        "arm_joint_names": np.asarray([name for solver in solvers for name in solver.names]),
        "arm_joint_pos": np.concatenate([r["joint_pos"] for r in results], axis=1),
        "arm_joint_vel": np.concatenate([r["joint_vel"] for r in results], axis=1),
        "hand_names": np.asarray(sides),
        "hand_frame_names": np.asarray([solver.hand_frame for solver in solvers]),
        "hand_target_frame": np.array("torso_link"),
        "hand_pos_torso": np.stack([r["target"] for r in results], axis=1),
        "hand_fk_pos_torso": np.stack([r["actual"] for r in results], axis=1),
        "hand_goal_pos_torso": np.stack([r["goal"] for r in results]),
        "arm_ik_position_error": np.stack([r["error"] for r in results], axis=1),
        "arm_ik_iterations": np.stack([r["iterations"] for r in results], axis=1),
        "arm_generation_metadata": np.array(json.dumps(metadata, sort_keys=True)),
    })
    save_npz(args.output, arrays, args.overwrite)
    print(f"Saved {len(times)} frames to {args.output.resolve()}")


if __name__ == "__main__":
    main()
