"""
Solves floating-base IK with multiple end-effector targets for the G1 humanoid.
"""

from typing import Sequence

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
from jaxls import Cost, VarValues
import numpy as onp
import pyroki as pk


def _flat_foot_residual(
    vals: VarValues,
    robot: pk.Robot,
    joint_var,
    base_var,
    foot_link_index: jax.Array,
    weight: float,
) -> jax.Array:
    """Penalize foot pitch and roll by constraining foot z-axis to be world-up."""
    joint_cfg = vals[joint_var]
    T_world_base = vals[base_var]
    Ts_base_link = robot.forward_kinematics(joint_cfg)
    T_base_foot = jaxlie.SE3(Ts_base_link[foot_link_index])
    T_world_foot = T_world_base @ T_base_foot
    # Foot z-axis in world frame; we want it = [0, 0, 1].
    foot_z_world = T_world_foot.rotation().as_matrix() @ jnp.array([0.0, 0.0, 1.0])
    # Only penalize x and y components (pitch/roll tilt), ignore z (yaw is free).
    return (foot_z_world[:2] * weight).flatten()


flat_foot_cost = Cost.factory(_flat_foot_residual)


def _xy_position_limit_residual(
    vals: VarValues,
    robot: pk.Robot,
    joint_var,
    base_var,
    link_index: jax.Array,
    xy_max: float,
    weight: float,
) -> jax.Array:
    """Penalize XY overflow for a world-frame link position."""
    joint_cfg = vals[joint_var]
    T_world_base = vals[base_var]
    Ts_base_link = robot.forward_kinematics(joint_cfg)
    T_base_link = jaxlie.SE3(Ts_base_link[link_index])
    T_world_link = T_world_base @ T_base_link
    link_xy = T_world_link.translation()[:2]
    xy_norm = jnp.linalg.norm(link_xy)
    overflow = jnp.maximum(0.0, xy_norm - xy_max)
    return jnp.array([weight * overflow])


xy_position_limit_cost = Cost.factory(_xy_position_limit_residual)


def solve_ik_with_base_multiple_targets_g1(
    robot: pk.Robot,
    target_link_names: Sequence[str],
    target_wxyzs: onp.ndarray,
    target_positions: onp.ndarray,
    prev_pos: onp.ndarray,
    prev_wxyz: onp.ndarray,
    prev_cfg: onp.ndarray,
    pos_weights: onp.ndarray | None = None,
    ori_weights: onp.ndarray | None = None,
    robot_coll: pk.collision.RobotCollision | None = None,
    flat_foot_link_names: Sequence[str] = (),
    flat_foot_weight: float = 50.0,
    xy_max: float | None = None,
    xy_max_weight: float = 50.0,
) -> tuple[onp.ndarray, onp.ndarray, onp.ndarray]:
    """
    Solves floating-base IK for a G1 humanoid with multiple end-effector targets.

    Target positions and orientations are in world frame.

    Args:
        robot: PyRoKi Robot.
        target_link_names: List of link names to control.
        target_wxyzs: Shape (num_targets, 4). Target orientations in world frame.
        target_positions: Shape (num_targets, 3). Target positions in world frame.
        prev_pos: Shape (3,). Previous base position.
        prev_wxyz: Shape (4,). Previous base orientation.
        prev_cfg: Shape (num_actuated_joints,). Previous joint configuration.
        pos_weights: Shape (num_targets,). Per-target position weights. Defaults to 5.0 for all.
        ori_weights: Shape (num_targets,). Per-target orientation weights. Defaults to 1.0 for all.
        xy_max: Optional world-frame XY radius limit applied to non-foot target links.
        xy_max_weight: Weight for the optional XY radius penalty.

    Returns:
        base_pos: Shape (3,). Solved base position.
        base_wxyz: Shape (4,). Solved base orientation.
        cfg: Shape (num_actuated_joints,). Solved joint configuration.
    """
    num_targets = len(target_link_names)
    assert target_positions.shape == (num_targets, 3)
    assert target_wxyzs.shape == (num_targets, 4)
    assert prev_pos.shape == (3,) and prev_wxyz.shape == (4,)
    assert prev_cfg.shape == (robot.joints.num_actuated_joints,)

    if pos_weights is None:
        pos_weights = onp.full(num_targets, 5.0)
    if ori_weights is None:
        ori_weights = onp.full(num_targets, 1.0)
    assert pos_weights.shape == (num_targets,)
    assert ori_weights.shape == (num_targets,)

    target_link_indices = [robot.links.names.index(name) for name in target_link_names]
    flat_foot_indices = [robot.links.names.index(name) for name in flat_foot_link_names]
    constrained_link_indices = [
        link_idx for link_idx in target_link_indices if link_idx not in set(flat_foot_indices)
    ]

    base_pose, cfg = _solve_ik_jax(
        robot,
        jnp.array(target_wxyzs),
        jnp.array(target_positions),
        jnp.array(target_link_indices),
        jnp.array(prev_pos),
        jnp.array(prev_wxyz),
        jnp.array(prev_cfg),
        jnp.array(pos_weights),
        jnp.array(ori_weights),
        robot_coll,
        jnp.array(flat_foot_indices, dtype=jnp.int32),
        flat_foot_weight,
        jnp.array(constrained_link_indices, dtype=jnp.int32),
        xy_max,
        xy_max_weight,
    )

    base_pos = onp.array(base_pose.translation())
    base_wxyz = onp.array(base_pose.rotation().wxyz)
    return base_pos, base_wxyz, onp.array(cfg)


@jdc.jit
def _solve_ik_jax(
    robot: pk.Robot,
    target_wxyz: jax.Array,          # (num_targets, 4)
    target_position: jax.Array,       # (num_targets, 3)
    target_link_indices: jax.Array,   # (num_targets,)
    prev_pos: jax.Array,
    prev_wxyz: jax.Array,
    prev_cfg: jax.Array,
    pos_weights: jax.Array,           # (num_targets,)
    ori_weights: jax.Array,           # (num_targets,)
    robot_coll: pk.collision.RobotCollision | None,
    flat_foot_indices: jax.Array,   # (num_flat_feet,)
    flat_foot_weight: float,
    constrained_link_indices: jax.Array,  # (num_constrained_links,)
    xy_max: float | None,
    xy_max_weight: float,
) -> tuple[jaxlie.SE3, jax.Array]:
    JointVar = robot.joint_var_cls

    def retract_fn(transform: jaxlie.SE3, delta: jax.Array) -> jaxlie.SE3:
        return jaxls.SE3Var.retract_fn(transform, delta)

    class BaseSE3Var(
        jaxls.Var[jaxlie.SE3],
        default_factory=lambda: jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3(prev_wxyz), prev_pos
        ),
        tangent_dim=jaxlie.SE3.tangent_dim,
        retract_fn=retract_fn,
    ): ...

    joint_var = JointVar(0)
    base_var = BaseSE3Var(0)

    target_poses = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(target_wxyz), target_position
    )  # batch shape: (num_targets,)

    costs = []
    num_targets = target_link_indices.shape[0]
    for i in range(num_targets):
        costs.append(
            pk.costs.pose_cost_with_base(
                robot,
                joint_var,
                base_var,
                jax.tree.map(lambda x: x[i], target_poses),
                target_link_indices[i],
                pos_weight=pos_weights[i],
                ori_weight=ori_weights[i],
            )
        )

    num_joints = robot.joints.num_actuated_joints
    costs.append(
        pk.costs.rest_with_base_cost(
            joint_var,
            base_var,
            prev_cfg,
            weight=jnp.array([0.1] * num_joints + [1.0] * 3 + [0.5] * 3),
        )
    )
    costs.append(pk.costs.limit_constraint(robot, joint_var))
    for i in range(flat_foot_indices.shape[0]):
        costs.append(
            flat_foot_cost(robot, joint_var, base_var, flat_foot_indices[i], flat_foot_weight)
        )
    if xy_max is not None:
        for i in range(constrained_link_indices.shape[0]):
            costs.append(
                xy_position_limit_cost(
                    robot,
                    joint_var,
                    base_var,
                    constrained_link_indices[i],
                    xy_max,
                    xy_max_weight,
                )
            )
    if robot_coll is not None:
        costs.append(
            pk.costs.self_collision_cost(
                robot,
                robot_coll=robot_coll,
                joint_var=joint_var,
                margin=0.02,
                weight=10.0,
            )
        )

    sol = (
        jaxls.LeastSquaresProblem(costs=costs, variables=[joint_var, base_var])
        .analyze()
        .solve(
            initial_vals=jaxls.VarValues.make([joint_var.with_value(prev_cfg), base_var]),
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=10.0),
        )
    )
    return sol[base_var], sol[joint_var]
