"""Damped least-squares (DLS) IK action term, modeled on mjlab's
``DifferentialIKAction`` but rewritten on top of Isaac Lab / PhysX.

Why this exists (vs ``G1CuroboIKAction`` / Isaac Lab's stock diff-IK):

- **Doesn't hang.** Single-step DLS uses no autograd backward, no LBFGS
  iteration, no CUDA graph capture, and no Warp kernel cache. Empirically
  cuRobo IK destabilises long (5-15 h) training on Blackwell + Isaac Sim 5;
  this implementation should behave like Isaac Lab's native diff-IK from
  a CUDA-stability standpoint.

- **Doesn't whip / convulse on unreachable targets.** Three explicit
  mitigations that Isaac Lab's stock diff-IK lacks:

  1. ``max_dq`` clamps the per-step joint displacement so an unreachable
     goal can't translate into a multi-rad joint kick.
  2. ``joint_limit_weight`` adds a soft penalty *inside* the DLS solver
     so the optimiser steers joints away from limits before they hit the
     hard PhysX clip.
  3. ``posture_weight`` + ``posture_target`` regularises null-space DOF
     for redundant arms (esp. dual-arm 14-DOF) so they settle to a
     natural pose instead of drifting between symmetric IK branches.

Action layout (matches ``G1CuroboIKAction``): per controlled EE, a 7-D
``[pos_x, pos_y, pos_z, qw, qx, qy, qz]`` absolute pose target in the
robot pelvis frame. Single-arm mode -> ``action_dim=7``; both-arm mode
-> ``action_dim=14`` (left then right). The env step computes pelvis-frame
references from the impedance command and the action manager hands them
directly to this term.

Math: joint-space DLS normal equations,
``(J^T W J + λ²I + diag(w_lim² + w_post²)) dq = J^T W dx + ...``
solved once per env step via a single batched ``torch.linalg.solve``.
"""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, Literal

import torch

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    axis_angle_from_quat,
    quat_apply,
    quat_conjugate,
    quat_mul,
)

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv


class G1DLSIKAction(ActionTerm):
    """Single-step damped-least-squares IK for one or two G1 end-effectors.

    See module docstring. Designed to be a drop-in replacement for
    :class:`G1CuroboIKAction` — same ``mode`` / ``body_names`` /
    ``joint_names`` schema, same 7-D-per-EE action layout, so the
    surrounding env step doesn't need to change.
    """

    cfg: G1DLSIKActionCfg
    _asset: "Articulation"

    def __init__(self, cfg: "G1DLSIKActionCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)

        # Validate mode against body_names / joint_names lengths.
        expected_eefs = 2 if self.cfg.mode == "both_arm" else 1
        if len(self.cfg.body_names) != expected_eefs:
            raise ValueError(
                f"G1DLSIKAction: mode='{self.cfg.mode}' expects "
                f"{expected_eefs} entries in body_names, got {self.cfg.body_names}."
            )
        if len(self.cfg.joint_names) != 7 * expected_eefs:
            raise ValueError(
                f"G1DLSIKAction: mode='{self.cfg.mode}' expects "
                f"{7 * expected_eefs} joint_names, got {len(self.cfg.joint_names)}."
            )
        self._num_eefs = expected_eefs

        # Isaac Lab joint indices for the controlled arm joints (7 or 14).
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=True
        )
        self._num_joints = len(self._joint_ids)

        # Per-EE body indices + the same indices with the +6 floating-base
        # offset that PhysX's get_jacobians() expects. (Fixed-base
        # articulations drop the base body from the Jacobian, so the
        # body_idx is shifted by -1 instead.)
        self._ee_body_idxs: list[int] = []
        self._jacobi_body_idxs: list[int] = []
        for name in self.cfg.body_names:
            ids, _ = self._asset.find_bodies(name)
            if not ids:
                raise ValueError(
                    f"G1DLSIKAction: body_names entry '{name}' not found in articulation."
                )
            body_idx = ids[0]
            self._ee_body_idxs.append(body_idx)
            if self._asset.is_fixed_base:
                self._jacobi_body_idxs.append(body_idx - 1)
            else:
                self._jacobi_body_idxs.append(body_idx)
        if self._asset.is_fixed_base:
            self._jacobi_joint_ids = list(self._joint_ids)
        else:
            self._jacobi_joint_ids = [i + 6 for i in self._joint_ids]

        # Buffers.
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._joint_targets = torch.zeros(self.num_envs, self._num_joints, device=self.device)

        # Joint limit bounds (soft limits include a small margin from hard limits).
        # Cached at init under the assumption that limits don't change per env.
        limits = self._asset.data.soft_joint_pos_limits  # (N, num_total_joints, 2)
        self._joint_lower = limits[:, self._joint_ids, 0].contiguous()  # (N, num_joints)
        self._joint_upper = limits[:, self._joint_ids, 1].contiguous()

        # Posture target. Default to the asset's default_joint_pos for our
        # joints; allow an override list/tuple matching joint_names order.
        q_default = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        if self.cfg.posture_target is not None:
            if len(self.cfg.posture_target) != self._num_joints:
                raise ValueError(
                    f"G1DLSIKAction: posture_target length {len(self.cfg.posture_target)} "
                    f"!= num_joints {self._num_joints}."
                )
            override = torch.tensor(
                self.cfg.posture_target, device=self.device, dtype=q_default.dtype
            )
            q_default[:] = override
        self._posture_target = q_default  # (N, num_joints)

        # Pre-compute weight squares.
        self._w_pos2 = self.cfg.position_weight ** 2
        self._w_ori2 = self.cfg.orientation_weight ** 2
        self._w_lim2 = self.cfg.joint_limit_weight ** 2
        self._w_post2 = self.cfg.posture_weight ** 2
        self._lam2 = max(self.cfg.damping, 1e-6) ** 2

        # Identity for batched DLS damping (precomputed).
        eye = torch.eye(self._num_joints, device=self.device)
        self._eye = eye.unsqueeze(0).expand(self.num_envs, -1, -1).contiguous()

        # Whether the goal needs a pelvis->world transform. Goals from
        # impedance command live in the articulation root (pelvis) frame;
        # our Jacobian and EE pose are in world frame, so we always
        # transform to world before computing the pose error.
        self._goal_frame_is_pelvis = True

        print(
            f"[G1DLSIKAction mode={self.cfg.mode}] {self._num_eefs} EE(s)="
            f"{self.cfg.body_names} | joints={self._num_joints}"
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def action_dim(self) -> int:
        return 7 * self._num_eefs

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------
    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions[:] = actions
        # Real work happens in apply_actions, after the action manager
        # has settled (DLS depends on the current joint state which is
        # only stable after process_actions of every term has run).

    def apply_actions(self):
        # Gather per-EE goal pose slices from the 7*n_eefs action vector.
        # Each slice is (pos, quat) in PELVIS frame.
        N = self.num_envs
        root_pos_w = self._asset.data.root_pos_w  # (N, 3)
        root_quat_w = self._asset.data.root_quat_w  # (N, 4)

        # Stack per-EE (6,) errors into (N, 6*n_eefs).
        # Stack per-EE (6, num_joints) Jacobians into (N, 6*n_eefs, num_joints).
        # Use list + cat — n_eefs is at most 2, allocations are tiny.
        errors_per_eef: list[torch.Tensor] = []
        jacobians_per_eef: list[torch.Tensor] = []

        full_jacobian = self._asset.root_physx_view.get_jacobians()
        # full_jacobian shape: (N, num_bodies, 6, num_dofs_phx).

        for i in range(self._num_eefs):
            goal_pos_b = self._processed_actions[:, 7 * i : 7 * i + 3]
            goal_quat_b = self._processed_actions[:, 7 * i + 3 : 7 * i + 7]

            # Pelvis frame -> world frame.
            goal_pos_w = root_pos_w + quat_apply(root_quat_w, goal_pos_b)
            goal_quat_w = quat_mul(root_quat_w, goal_quat_b)

            # Current EE pose in world frame.
            ee_pos_w = self._asset.data.body_pos_w[:, self._ee_body_idxs[i]]
            ee_quat_w = self._asset.data.body_quat_w[:, self._ee_body_idxs[i]]

            # Pose error in world frame.
            pos_err = goal_pos_w - ee_pos_w  # (N, 3)
            q_err = quat_mul(goal_quat_w, quat_conjugate(ee_quat_w))  # (N, 4)
            # Force the scalar part non-negative so axis-angle picks the short
            # rotation (no >pi flips that whip the arm around).
            q_err = torch.where(q_err[:, :1] < 0, -q_err, q_err)
            rot_err = axis_angle_from_quat(q_err)  # (N, 3)
            err_world = torch.cat([pos_err, rot_err], dim=-1)  # (N, 6)
            errors_per_eef.append(err_world)

            # Jacobian for this EE, sliced to our joints, in world frame.
            jac = full_jacobian[:, self._jacobi_body_idxs[i], :, :][:, :, self._jacobi_joint_ids]
            # shape (N, 6, num_joints)
            jacobians_per_eef.append(jac)

        if self._num_eefs == 1:
            dx = errors_per_eef[0]  # (N, 6)
            J = jacobians_per_eef[0]  # (N, 6, num_joints)
        else:
            dx = torch.cat(errors_per_eef, dim=-1)  # (N, 12)
            J = torch.cat(jacobians_per_eef, dim=1)  # (N, 12, num_joints)

        # Joint-space normal equations. Split linear/angular rows so we can
        # weight position vs orientation independently:
        #   JTJ = wp2 * J_lin^T J_lin + wo2 * J_ang^T J_ang
        #   JTdx = wp2 * J_lin^T pos_err + wo2 * J_ang^T rot_err
        # Each EE contributes 3 linear + 3 angular rows, so the linear and
        # angular row indices are interleaved when we stack multiple EEs.
        # We split the stacked J / dx into linear and angular halves per EE
        # and accumulate. Also stash per-EE position-error squared norm for
        # the adaptive damping term below.
        JTJ = torch.zeros(N, self._num_joints, self._num_joints, device=self.device)
        JTdx = torch.zeros(N, self._num_joints, device=self.device)
        pos_err_sq_per_eef: list[torch.Tensor] = []
        for i in range(self._num_eefs):
            jac = jacobians_per_eef[i]
            err = errors_per_eef[i]
            jac_lin = jac[:, :3, :]  # (N, 3, num_joints)
            jac_ang = jac[:, 3:, :]  # (N, 3, num_joints)
            err_lin = err[:, :3]  # (N, 3)
            err_ang = err[:, 3:]  # (N, 3)
            JTJ.add_(self._w_pos2 * torch.einsum("bti,btj->bij", jac_lin, jac_lin))
            JTJ.add_(self._w_ori2 * torch.einsum("bti,btj->bij", jac_ang, jac_ang))
            JTdx.add_(self._w_pos2 * torch.einsum("bti,bt->bi", jac_lin, err_lin))
            JTdx.add_(self._w_ori2 * torch.einsum("bti,bt->bi", jac_ang, err_ang))
            pos_err_sq_per_eef.append((err_lin ** 2).sum(dim=-1))  # (N,)

        # Joint state for limit / posture residuals.
        q = self._asset.data.joint_pos[:, self._joint_ids]  # (N, num_joints)

        # Soft joint-limit penalty: only active for joints that have
        # actually crossed a soft limit. ``r_limit`` is the signed
        # distance back to the limit (negative when above upper, positive
        # when below lower) and ``violated`` is a 0/1 mask per joint.
        if self._w_lim2 > 0.0:
            over_upper = (self._joint_upper - q).clamp(max=0.0)  # <=0
            under_lower = (self._joint_lower - q).clamp(min=0.0)  # >=0
            r_limit = over_upper + under_lower  # (N, num_joints)
            violated = (r_limit != 0).to(JTJ.dtype)  # (N, num_joints)
            # Add to diagonal of JTJ and to JTdx (identity Jacobian rows).
            diag = JTJ.diagonal(dim1=-2, dim2=-1)
            diag.add_(self._w_lim2 * violated)
            JTdx.add_(self._w_lim2 * violated * r_limit)

        # Posture regularisation: pulls joints toward posture_target.
        # Always-on regardless of distance to target (small weight by default).
        if self._w_post2 > 0.0:
            r_posture = self._posture_target - q
            diag = JTJ.diagonal(dim1=-2, dim2=-1)
            diag.add_(self._w_post2)
            JTdx.add_(self._w_post2 * r_posture)

        # DLS damping. Base term is always λ². Adaptive term grows with the
        # per-EE position error squared, applied only to that arm's joints
        # (joints i*7..i*7+7 in cfg order, since the cfg pairs body_names[i]
        # with joint_names[7i:7i+7]). The net effect is:
        #   - target reachable → pos_err small → damping ≈ λ² → sharp tracking
        #   - target unreachable → pos_err large → damping ↑ → step shrinks
        #     smoothly so the arm settles at the closest reachable point
        #     instead of bouncing between DLS direction estimates near the
        #     workspace boundary.
        # We add the lam2 + adaptive term to the diagonal in one shot.
        diag = JTJ.diagonal(dim1=-2, dim2=-1)  # view, in-place writes OK
        if self.cfg.adaptive_damping > 0.0:
            adaptive_lam2 = torch.full(
                (N, self._num_joints), self._lam2, device=self.device, dtype=JTJ.dtype
            )
            for i in range(self._num_eefs):
                start = 7 * i
                end = 7 * (i + 1)
                # broadcast (N,) → (N, 7) over this arm's joint columns
                adaptive_lam2[:, start:end].add_(
                    (self.cfg.adaptive_damping * pos_err_sq_per_eef[i]).unsqueeze(-1)
                )
            diag.add_(adaptive_lam2)
        else:
            diag.add_(self._lam2)

        # Solve and clamp the per-step joint delta.
        dq = torch.linalg.solve(JTJ, JTdx.unsqueeze(-1)).squeeze(-1)
        if self.cfg.max_dq > 0.0:
            dq = dq.clamp(-self.cfg.max_dq, self.cfg.max_dq)

        self._joint_targets[:] = q + dq
        self._asset.set_joint_position_target(self._joint_targets, self._joint_ids)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self._raw_actions.zero_()
        else:
            self._raw_actions[env_ids] = 0.0


@configclass
class G1DLSIKActionCfg(ActionTermCfg):
    """Config for :class:`G1DLSIKAction`."""

    class_type: type[ActionTerm] = G1DLSIKAction

    mode: Literal["left_arm", "right_arm", "both_arm"] = "left_arm"
    """Which arm(s) this action term controls. Schema matches
    :class:`G1CuroboIKActionCfg` so the env step doesn't care which
    backend IK is in use."""

    joint_names: list[str] = field(default_factory=list)
    """Articulation joints the action term writes targets to, in this
    exact order. 7 entries for single-arm modes, 14 for both-arm
    (left 7 then right 7)."""

    body_names: list[str] = field(default_factory=list)
    """End-effector body names. Length 1 for single-arm, 2 for both-arm
    (left first, then right). Each EE consumes a 7-D pose slice from the
    action vector."""

    # ------------------------------------------------------------------
    # DLS solver knobs (defaults are mjlab's).
    # ------------------------------------------------------------------
    damping: float = 0.05
    """Damping coefficient λ (added as λ²I to the normal equations).
    Larger = more stable near singularities, smaller = sharper tracking."""

    adaptive_damping: float = 1.0
    """Extra damping that scales with per-EE position error squared.
    Effective damping for arm i = λ² + adaptive_damping · ‖pos_err_i‖².
    This is the "stay smoothly at closest reachable position when target
    is unreachable" knob — without it, DLS jitters at the workspace
    boundary as the Newton direction flips between near-singular Jacobian
    estimates. Set 0 to disable (recovers vanilla mjlab behavior).

    Heuristic: at pos_err = 0.5 m the extra damping is 0.25 (vs λ² = 0.0025
    by default), so the arm slides 100 more smoothly when far from the
    target than when within reach. At pos_err = 1 cm it adds 1e-4 (≈4% of
    base) — effectively a no-op for normal tracking."""

    max_dq: float = 0.5
    """Maximum per-step joint displacement (rad/step). The single most
    important parameter against the "arm whips when target is
    unreachable" failure mode. Set to 0 to disable clamping."""

    position_weight: float = 1.0
    """Weight on position residual in the DLS system."""

    orientation_weight: float = 1.0
    """Weight on orientation (axis-angle) residual in the DLS system.
    Set to 0 for position-only IK."""

    joint_limit_weight: float = 5.0
    """Soft joint-limit penalty weight (0 = disabled). When a joint goes
    past its soft limit the solver gets an extra residual that pulls
    it back toward the legal region, before PhysX has to hard-clip."""

    posture_weight: float = 0.01
    """Posture regularisation weight (0 = disabled). Pulls joints toward
    ``posture_target`` each step; small values are enough to kill
    null-space drift on redundant arms without affecting task tracking."""

    posture_target: list[float] | None = None
    """Per-joint posture target in ``joint_names`` order. If ``None``,
    defaults to the articulation's ``default_joint_pos`` for the
    controlled joints."""
