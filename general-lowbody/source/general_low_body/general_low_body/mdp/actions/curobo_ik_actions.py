"""GPU-batched IK action term backed by cuRobo (Aug 2025 / IsaacLab-pinned commit).

We use the IsaacLab-tested commit ``ebb71702...`` of cuRobo rather than the
2026 release because the 2026 release requires warp-lang >= 1.10 which is
incompatible with IsaacSim's bundled omni.warp.core (1.5.0). The Aug 2025
commit only uses warp APIs available in 0.9 → 1.13, so it works with whatever
warp IsaacSim ships and we don't need any pre-import shims.

Action layout: per controlled EE, ``[pos_x, pos_y, pos_z, qw, qx, qy, qz]`` in
the robot base (pelvis) frame. Single-arm mode has ``action_dim=7``; both-arm
mode has ``action_dim=14`` (left then right). The ``mode`` cfg field is
exposed so the env step can read it and decide which slots of the policy
output go to this action term.

Setup pattern mirrors Pink IK: call ``setup_curobo_ik_urdf`` from the env
cfg's ``__post_init__`` to convert the G1 USD to a URDF on disk and stamp the
path onto the action cfg. The action term then builds the cuRobo solver from
that URDF at ``__init__`` time.
"""

from __future__ import annotations

import os
import time
from dataclasses import field
from typing import TYPE_CHECKING, Literal

import torch

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

try:
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose
    from curobo.types.robot import JointState as CuroboJointState
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
except ModuleNotFoundError as exc:
    if exc.name != "curobo" and not exc.name.startswith("curobo."):
        raise
    TensorDeviceType = None


if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv


class G1CuroboIKAction(ActionTerm):
    """cuRobo IK for one or two end-effectors on the G1.

    Mode is set by ``cfg.mode`` and must match the lengths of
    ``cfg.body_names`` / ``cfg.joint_names``:

    - ``"left_arm"`` / ``"right_arm"``: 1 EE, 7 joints, ``action_dim=7``.
    - ``"both_arm"``: 2 EEs (left then right), 14 joints (left 7 + right 7),
      ``action_dim=14``. A single cuRobo solver call returns both arms;
      the primary (left) EE uses ``goal_pose`` and the right EE is passed
      through ``link_poses``.

    Per-EE action slice: ``[pos_x, pos_y, pos_z, qw, qx, qy, qz]`` in pelvis
    frame. Each EE's slice is independently transformed pelvis→base_link.
    """

    cfg: G1CuroboIKActionCfg
    _asset: "Articulation"

    def __init__(self, cfg: "G1CuroboIKActionCfg", env: "ManagerBasedEnv"):
        if TensorDeviceType is None:
            raise ImportError("cuRobo is required only for cuRobo IK tasks; use G1-Dual-Arm-IK for DLS.")
        super().__init__(cfg, env)

        if not cfg.urdf_path:
            raise RuntimeError(
                "G1CuroboIKAction: cfg.urdf_path is unset. Call "
                "`mdp.setup_curobo_ik_urdf(self.actions, self.scene)` in your "
                "env_cfg.__post_init__ after super().__post_init__()."
            )

        # Validate mode against body_names / joint_names lengths.
        expected_eefs = 2 if self.cfg.mode == "both_arm" else 1
        if len(self.cfg.body_names) != expected_eefs:
            raise ValueError(
                f"G1CuroboIKAction: mode='{self.cfg.mode}' expects "
                f"{expected_eefs} entries in body_names, got {self.cfg.body_names}."
            )
        if len(self.cfg.joint_names) != 7 * expected_eefs:
            raise ValueError(
                f"G1CuroboIKAction: mode='{self.cfg.mode}' expects "
                f"{7 * expected_eefs} joint_names, got {len(self.cfg.joint_names)}."
            )
        self._num_eefs = expected_eefs

        # Isaac Lab joint indices for the controlled arm joints (7 or 14).
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=True
        )
        self._num_joints = len(self._joint_ids)
        # Cache EE body indices (one per controlled EE) for the diagnostic FK
        # comparison. Index 0 is the primary EE (cuRobo's ee_link).
        self._ee_body_idxs: list[int | None] = []
        for name in self.cfg.body_names:
            try:
                ee_ids, _ = self._asset.find_bodies(name)
                self._ee_body_idxs.append(ee_ids[0] if ee_ids else None)
            except Exception:
                self._ee_body_idxs.append(None)
        # Cache the body index for cuRobo's chain base. ImpedanceCommand gives
        # goals in pelvis frame (= articulation root frame); cuRobo computes IK
        # in its own base_link frame. When they differ (typical case:
        # base_link="torso_link" to keep waist out of the IK chain), we use
        # this body's measured world pose each step to transform the goal
        # pelvis_frame → world → base_link_frame.
        base_ids, _ = self._asset.find_bodies(self.cfg.base_link_name)
        if not base_ids:
            raise ValueError(
                f"G1CuroboIKAction: cfg.base_link_name='{self.cfg.base_link_name}' "
                f"not found in articulation. Use a body name from `_asset.body_names`."
            )
        self._base_body_idx = base_ids[0]
        self._needs_base_frame_transform = self.cfg.base_link_name != "pelvis"

        # Build the cuRobo solver. CUDA graphs (use_cuda_graph=True) cache the
        # solver pipeline at the first solve_batch call for max_batch=num_envs.
        self._tensor_args = TensorDeviceType(device=torch.device(self.device))
        self._ik = self._build_ik_solver()

        # cuRobo's active-joint order vs cfg.joint_names order may differ. Map
        # both directions once at init.
        ik_joint_names = list(self._ik.kinematics.joint_names)
        missing = set(self.cfg.joint_names) - set(ik_joint_names)
        if missing:
            raise ValueError(
                f"G1CuroboIKAction: cfg.joint_names not active in cuRobo "
                f"solver (likely locked or absent): {missing}. "
                f"cuRobo active joints: {ik_joint_names}"
            )
        # One-line sanity print so we can verify in startup logs that both
        # arms' joints landed in the cuRobo IK chain (especially important
        # in both_arm mode where the secondary arm is added via link_names).
        try:
            link_names_in_kin = list(self._ik.kinematics.link_names)
        except Exception:
            link_names_in_kin = []
        print(
            f"[G1CuroboIKAction mode={self.cfg.mode}] active DOFs="
            f"{len(ik_joint_names)} joints={ik_joint_names} | "
            f"tracked links={link_names_in_kin}"
        )
        self._ik_joint_names = ik_joint_names
        # _curobo_to_cfg_idx[i] = position in ik_joint_names of cfg.joint_names[i]
        self._curobo_to_cfg_idx = torch.tensor(
            [ik_joint_names.index(n) for n in self.cfg.joint_names],
            device=self.device, dtype=torch.long,
        )

        # Action / output buffers.
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._joint_targets = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._joint_targets[:] = self._asset.data.joint_pos[:, self._joint_ids]

        self._step_count = 0
        self._last_success_rate = 0.0

        # ------------------------------------------------------------------
        # Health logger — designed so a single training run that freezes
        # leaves enough on-disk evidence to root-cause the freeze:
        #  - GPU memory series (allocated / reserved / peak) → leak vs not
        #  - solve_batch wall-time (avg / max / p99 per window) → optimizer
        #    divergence or scheduler stall
        #  - NaN counters on inputs and IK outputs → poisoning detection
        #  - pos_err / success_rate / joint-target norms → silent IK quality
        #    degradation before the kernel actually hangs
        # The file is line-buffered so the row just before freeze IS on disk
        # even when the process needs SIGKILL.
        # Also: faulthandler.register(SIGUSR1) gives us a "stack trace on
        # demand" — before SIGKILLing a stuck process, send SIGUSR1 first;
        # if Python is alive (only CUDA stuck) the trace lands in stderr.
        self._health_file = None
        self._t0 = time.time()
        self._solve_ms_window: list[float] = []
        self._pos_err_mm_window: list[float] = []
        self._pos_err_mm_max_in_window = 0.0
        self._success_window: list[float] = []
        self._tgt_norm_window: list[float] = []
        self._tgt_norm_max_in_window = 0.0
        self._nan_in_total = 0
        self._nan_out_total = 0

        # Register SIGUSR1 → Python stack-trace handler unconditionally.
        # cuRobo freezes typically leave Python alive while CUDA is stuck;
        # without this handler, `kill -USR1 <pid>` defaults to terminating
        # the process (Linux default for SIGUSR1) and we lose all diagnostic
        # info. Keep this independent of `health_log_every` — the cost is
        # zero when the signal is not raised, and it's the one tool that
        # works when everything else (logging, prints) is also stuck.
        try:
            import faulthandler, signal  # noqa: E401
            faulthandler.register(signal.SIGUSR1)
            print(
                f"[G1CuroboIKAction] SIGUSR1 handler installed; "
                f"`kill -USR1 {os.getpid()}` will dump Python stack trace."
            )
        except Exception as e:
            print(f"[G1CuroboIKAction] faulthandler setup failed: {e}")

        if self.cfg.health_log_every > 0:
            stamp = int(self._t0)
            log_path = f"/tmp/curobo_health_pid{os.getpid()}_{stamp}.tsv"
            self._health_file = open(log_path, "w", buffering=1)
            self._health_file.write(
                "step\twallclock\telapsed_s\trss_mb\tgpu_alloc_mb\tgpu_reserved_mb"
                "\tgpu_peak_alloc_mb\tavg_solve_ms\tmax_solve_ms\tp99_solve_ms"
                "\tnan_in_total\tnan_out_total\tpos_err_mean_mm\tpos_err_max_mm"
                "\tsuccess_rate\ttgt_norm_mean\ttgt_norm_max"
                "\talloc_retries\treserved_curr_mb\n"
            )
            print(f"[G1CuroboIKAction] health log → {log_path}")

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
        from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_conjugate, quat_mul

        self._raw_actions[:] = actions
        self._processed_actions[:] = actions

        # Current measured joint positions for all active arm joints (7 or 14).
        q_lab = self._asset.data.joint_pos[:, self._joint_ids]  # (N, num_joints) cfg order

        # play.py wraps env.step in torch.inference_mode(), but cuRobo's
        # gradient IK uses autograd. Step out, clone every input so we hand
        # cuRobo regular autograd-tracked tensors, then clone the output back
        # before exiting so it's safe for the rest of the env step.
        with torch.inference_mode(False), torch.enable_grad():
            # Per-EE pelvis-frame goal slices for the diagnostic print (same
            # frame as ee_pos_b further down).
            goal_pos_pelvis: list[torch.Tensor] = []
            goal_quat_pelvis: list[torch.Tensor] = []
            # Per-EE base_link-frame goal (after pelvis→base transform if
            # base_link != "pelvis"). Index 0 is the primary EE (cuRobo's
            # ee_link, passed via goal_pose); indices 1+ are passed via
            # link_poses keyed by body_names[i].
            goal_pos_base: list[torch.Tensor] = []
            goal_quat_base: list[torch.Tensor] = []

            # ImpedanceCommand emits in pelvis frame, but cuRobo's IK is in
            # base_link frame. When base_link != "pelvis" (typical case:
            # base_link="torso_link" so waist is out of the IK chain), apply
            # T_pelvis_in_base_link^-1 ∘ goal_in_pelvis:
            #   pelvis → world → base_link
            # using the articulation's measured root (= pelvis) pose and the
            # measured base_link body pose this step. This way cuRobo gets a
            # target consistent with the actual runtime waist state.
            if self._needs_base_frame_transform:
                root_pos_w = self._asset.data.root_pos_w
                root_quat_w = self._asset.data.root_quat_w
                base_pos_w = self._asset.data.body_pos_w[:, self._base_body_idx]
                base_quat_w = self._asset.data.body_quat_w[:, self._base_body_idx]

            for i in range(self._num_eefs):
                pp = self._processed_actions[:, 7 * i : 7 * i + 3].clone()
                pq = self._processed_actions[:, 7 * i + 3 : 7 * i + 7].clone()
                goal_pos_pelvis.append(pp)
                goal_quat_pelvis.append(pq)
                if self._needs_base_frame_transform:
                    pp_w = root_pos_w + quat_apply(root_quat_w, pp)
                    pq_w = quat_mul(root_quat_w, pq)
                    pp_b = quat_apply_inverse(base_quat_w, pp_w - base_pos_w).clone()
                    pq_b = quat_mul(quat_conjugate(base_quat_w), pq_w).clone()
                else:
                    pp_b = pp
                    pq_b = pq
                goal_pos_base.append(pp_b)
                goal_quat_base.append(pq_b)

            # Permute Isaac Lab joint order → cuRobo joint order via scatter:
            #   q_curobo[ik_idx[i]] = q_lab[i]
            q_curobo = torch.empty_like(q_lab)
            q_curobo[:, self._curobo_to_cfg_idx] = q_lab
            q_curobo = q_curobo.clone()

            goal_pose = Pose(position=goal_pos_base[0], quaternion=goal_quat_base[0])
            link_poses: dict | None = None
            if self._num_eefs > 1:
                link_poses = {
                    self.cfg.body_names[i]: Pose(
                        position=goal_pos_base[i], quaternion=goal_quat_base[i]
                    )
                    for i in range(1, self._num_eefs)
                }

            # Old API: seed_config shape is (num_seeds, batch, dof). With
            # num_seeds=1 and warm-starting from current_state, this gives us
            # the smoothness benefit of differential IK without its local-min
            # failure mode (the downstream LBFGS still runs).
            seed_config = q_curobo.unsqueeze(0)  # (1, N, num_joints)

            # Health: count NaN/Inf entering the solver. If any input is bad,
            # cuRobo's optimizer can spin forever. This counter is the most
            # decisive signal for NaN-poisoning freezes.
            if self._health_file is not None:
                nan_q = (~torch.isfinite(q_curobo).all(dim=-1)).sum().item()
                nan_g = 0
                for pb in goal_pos_base:
                    nan_g += (~torch.isfinite(pb).all(dim=-1)).sum().item()
                self._nan_in_total += int(nan_q + nan_g)

            # Time the solve. perf_counter is wall-clock; without CUDA sync
            # this measures kernel-launch overhead (and any blocking sync
            # cuRobo does internally). On the periodic flush step below we DO
            # sync so the window aggregates land on real GPU times.
            t_solve_start = time.perf_counter()
            result = self._ik.solve_batch(
                goal_pose=goal_pose,
                retract_config=q_curobo,
                seed_config=seed_config,
                return_seeds=1,
                num_seeds=self.cfg.num_seeds,
                link_poses=link_poses,
            )
            t_solve_ms = (time.perf_counter() - t_solve_start) * 1000.0

            # result.solution shape: (batch, return_seeds=1, active_dof)
            sol = result.solution.squeeze(1).detach().clone()
            success = result.success.detach().clone() if result.success is not None else None
            pos_err = result.position_error.detach().clone() if result.position_error is not None else None
            rot_err = result.rotation_error.detach().clone() if result.rotation_error is not None else None
            # Explicit del to drop refs to the IKResult dataclass — its
            # js_solution / goal_pose / seed fields are not detached and could
            # otherwise keep autograd graph alive across the for-loop boundary.
            del result

        # Permute cuRobo joint order → cfg.joint_names order, then sanitize
        # before handing targets to PhysX. This is intentionally shared by
        # single-arm (7 DOF) and dual-arm (14 DOF) modes: dual-arm failures now
        # reach sim.step with the pre-step CUDA sync already complete, so the
        # remaining likely trigger is a rare finite-but-unsafe target for the
        # GPU articulation solver.
        candidate = sol[:, self._curobo_to_cfg_idx]
        candidate = torch.where(torch.isfinite(candidate), candidate, q_lab)
        if self.cfg.fallback_to_current_on_failure and success is not None:
            success_mask = success.reshape(self.num_envs, -1).any(dim=-1)
            candidate = torch.where(success_mask.unsqueeze(-1), candidate, q_lab)
        if self.cfg.clamp_to_joint_limits:
            joint_limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
            candidate = torch.clamp(candidate, joint_limits[..., 0], joint_limits[..., 1])
        if self.cfg.max_joint_target_delta is not None and self.cfg.max_joint_target_delta > 0.0:
            max_delta = self.cfg.max_joint_target_delta
            candidate = q_lab + torch.clamp(candidate - q_lab, -max_delta, max_delta)
        self._joint_targets[:] = candidate

        if success is not None:
            self._last_success_rate = success.float().mean().item()
        self._step_count += 1

        # Health: accumulate per-step samples. Every value is a Python float
        # to avoid keeping CUDA tensor refs alive in the window lists.
        if self._health_file is not None:
            self._solve_ms_window.append(t_solve_ms)
            nan_o = (~torch.isfinite(sol).all(dim=-1)).sum().item()
            self._nan_out_total += int(nan_o)
            if pos_err is not None:
                pe = pos_err.float()  # avoid f16 surprises
                self._pos_err_mm_window.append(pe.mean().item() * 1000.0)
                pe_max_mm = pe.max().item() * 1000.0
                if pe_max_mm > self._pos_err_mm_max_in_window:
                    self._pos_err_mm_max_in_window = pe_max_mm
            if success is not None:
                self._success_window.append(success.float().mean().item())
            tgt_norm = self._joint_targets.norm(dim=-1)
            self._tgt_norm_window.append(tgt_norm.mean().item())
            tgt_max = tgt_norm.max().item()
            if tgt_max > self._tgt_norm_max_in_window:
                self._tgt_norm_max_in_window = tgt_max

        # Periodic flush. We sync once here so the window's avg/max times
        # reflect actual GPU work, not kernel-launch queue depth. Cost: ~5ms
        # every N steps. Memory snapshot also reset so the peak we report is
        # truly the per-window peak (catches single-spike allocations).
        if self._health_file is not None and self._step_count % self.cfg.health_log_every == 0:
            torch.cuda.synchronize()
            elapsed = time.time() - self._t0
            rss_mb = -1.0
            try:
                with open(f"/proc/{os.getpid()}/status", "r") as f:
                    for ln in f:
                        if ln.startswith("VmRSS:"):
                            rss_mb = int(ln.split()[1]) / 1024.0
                            break
            except Exception:
                pass
            alloc_mb = torch.cuda.memory_allocated() / 1e6
            reserved_mb = torch.cuda.memory_reserved() / 1e6
            peak_mb = torch.cuda.max_memory_allocated() / 1e6
            # Allocator fragmentation probe: num_alloc_retries grows whenever
            # cudaMalloc fails and the caching allocator has to free + retry —
            # a textbook sign of fragmentation. reserved_curr_mb is the live
            # caching-allocator reservation (vs peak_mb which is the peak
            # *allocated* in the past window).
            try:
                mem_stats = torch.cuda.memory_stats()
                alloc_retries = int(mem_stats.get("num_alloc_retries", 0))
                reserved_curr_mb = mem_stats.get("reserved_bytes.all.current", 0) / 1e6
            except Exception:
                alloc_retries = -1
                reserved_curr_mb = -1.0
            times = sorted(self._solve_ms_window)
            avg_ms = sum(times) / len(times) if times else 0.0
            max_ms = times[-1] if times else 0.0
            p99_ms = times[max(0, int(len(times) * 0.99) - 1)] if times else 0.0
            pe_mean = (
                sum(self._pos_err_mm_window) / len(self._pos_err_mm_window)
                if self._pos_err_mm_window else 0.0
            )
            pe_max = self._pos_err_mm_max_in_window
            sr = sum(self._success_window) / len(self._success_window) if self._success_window else 0.0
            tn_mean = sum(self._tgt_norm_window) / len(self._tgt_norm_window) if self._tgt_norm_window else 0.0
            tn_max = self._tgt_norm_max_in_window
            self._health_file.write(
                f"{self._step_count}\t{time.time():.1f}\t{elapsed:.1f}\t{rss_mb:.0f}"
                f"\t{alloc_mb:.0f}\t{reserved_mb:.0f}\t{peak_mb:.0f}"
                f"\t{avg_ms:.2f}\t{max_ms:.2f}\t{p99_ms:.2f}"
                f"\t{self._nan_in_total}\t{self._nan_out_total}"
                f"\t{pe_mean:.2f}\t{pe_max:.2f}\t{sr:.3f}"
                f"\t{tn_mean:.3f}\t{tn_max:.3f}"
                f"\t{alloc_retries}\t{reserved_curr_mb:.0f}\n"
            )
            # buffering=1 already flushes on newline, this is belt-and-suspenders
            self._health_file.flush()
            self._solve_ms_window.clear()
            self._pos_err_mm_window.clear()
            self._pos_err_mm_max_in_window = 0.0
            self._success_window.clear()
            self._tgt_norm_window.clear()
            self._tgt_norm_max_in_window = 0.0
            torch.cuda.reset_peak_memory_stats()

        # Optional diagnostic — per-EE goal vs measured pose in pelvis frame
        # so a frame mismatch would show as a fixed offset.
        if self.cfg.debug_log_every > 0 and self._step_count % self.cfg.debug_log_every == 0:
            from isaaclab.utils.math import quat_apply_inverse, quat_conjugate, quat_mul
            root_pos_w = self._asset.data.root_pos_w
            root_quat_w = self._asset.data.root_quat_w
            pos_err_mm = (pos_err.mean().item() * 1000.0) if pos_err is not None else float("nan")
            rot_err_v = rot_err.mean().item() if rot_err is not None else float("nan")
            for i, name in enumerate(self.cfg.body_names):
                ee_b_str = "n/a"
                idx = self._ee_body_idxs[i]
                if idx is not None:
                    ee_pos_w = self._asset.data.body_pos_w[:, idx]
                    ee_quat_w = self._asset.data.body_quat_w[:, idx]
                    ee_pos_b = quat_apply_inverse(root_quat_w, ee_pos_w - root_pos_w)
                    ee_quat_b = quat_mul(quat_conjugate(root_quat_w), ee_quat_w)
                    ee_b_str = (
                        f"ee_pos_b={ee_pos_b[0].tolist()} ee_quat_b={ee_quat_b[0].tolist()}"
                    )
                print(
                    f"[cuRobo-IK step {self._step_count} ee={name}] "
                    f"goal_pos_b={goal_pos_pelvis[i][0].tolist()} "
                    f"goal_quat_b={goal_quat_pelvis[i][0].tolist()} | "
                    f"{ee_b_str}"
                )
            print(
                f"[cuRobo-IK step {self._step_count}] "
                f"pos_err={pos_err_mm:.2f}mm rot_err={rot_err_v:.4f}rad "
                f"success={self._last_success_rate:.2f} "
                f"|q_tgt|={self._joint_targets[0].norm().item():.3f} "
                f"|q_meas|={q_lab[0].norm().item():.3f}"
            )

    def apply_actions(self):
        self._asset.set_joint_position_target(self._joint_targets, self._joint_ids)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self._raw_actions.zero_()
            self._joint_targets[:] = self._asset.data.joint_pos[:, self._joint_ids]
        else:
            self._raw_actions[env_ids] = 0.0
            self._joint_targets[env_ids] = self._asset.data.joint_pos[env_ids][:, self._joint_ids]

    def __del__(self):
        # Best-effort: flush the last health row + close the file when
        # the action term is garbage-collected (normal training exit).
        try:
            f = getattr(self, "_health_file", None)
            if f is not None:
                f.flush()
                f.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _build_ik_solver(self) -> IKSolver:
        """Build the cuRobo IK solver from our URDF + a hand-crafted robot_cfg dict.

        ``ee_link`` is set from ``cfg.body_names[0]`` (primary EE). For both-arm
        mode we also declare the secondary EE(s) in ``kinematics.link_names``
        so they're tracked for FK and addressable via ``solve_batch(link_poses=...)``.
        Any joints in the chain that should be pinned (e.g. waist when
        base_link="pelvis") go in ``cfg.lock_joint_names``.
        """
        # Silence cuRobo's per-solve INFO spam ("Updating safety params" /
        # "Updating optimizer params" every step). The train path imports
        # something that flips Python's root logger to INFO, so cuRobo's
        # debug-level chatter leaks through; play doesn't import that and
        # the messages stay hidden. Forcing cuRobo's logger to WARNING here
        # cuts the noise without losing real warnings/errors.
        import logging
        logging.getLogger("curobo").setLevel(logging.WARNING)

        # cspace.joint_names enumerates ALL joints in the kinematic chain;
        # locked joints are still listed here and pinned via lock_joints.
        all_chain_joints = list(self.cfg.lock_joint_names) + list(self.cfg.joint_names)
        n_dof = len(all_chain_joints)
        lock_joints = {j: 0.0 for j in self.cfg.lock_joint_names}

        robot_cfg = {
            "robot_cfg": {
                "kinematics": {
                    "urdf_path": self.cfg.urdf_path,
                    "asset_root_path": "",
                    "base_link": self.cfg.base_link_name,
                    "ee_link": self.cfg.body_names[0],
                    # link_names lists all FK-tracked links. The primary
                    # ee_link is added automatically; for both-arm mode the
                    # secondary EE(s) need to be listed explicitly so they
                    # become addressable via solve_batch(link_poses={...}).
                    "link_names": list(self.cfg.body_names[1:]),
                    "collision_link_names": [],
                    "collision_spheres": {},
                    "collision_sphere_buffer": 0.0,
                    "self_collision_ignore": {},
                    "self_collision_buffer": {},
                    "mesh_link_names": [],
                    "lock_joints": lock_joints,
                    "extra_links": {},
                    "use_global_cumul": True,
                    "cspace": {
                        "joint_names": all_chain_joints,
                        # retract_config is per-call now; this is just a fallback.
                        "retract_config": [0.0] * n_dof,
                        "null_space_weight": [1.0] * n_dof,
                        "cspace_distance_weight": [1.0] * n_dof,
                        "max_acceleration": 15.0,
                        "max_jerk": 500.0,
                    },
                },
            }
        }

        ik_cfg = IKSolverConfig.load_from_robot_config(
            robot_cfg=robot_cfg,
            tensor_args=self._tensor_args,
            num_seeds=self.cfg.num_seeds,
            position_threshold=self.cfg.position_tolerance,
            rotation_threshold=self.cfg.orientation_tolerance,
            use_cuda_graph=self.cfg.use_cuda_graph,
            self_collision_check=False,
            self_collision_opt=False,
            regularization=True,
            collision_checker_type=None,
        )
        return IKSolver(ik_cfg)


@configclass
class G1CuroboIKActionCfg(ActionTermCfg):
    """Config for :class:`G1CuroboIKAction`."""

    class_type: type[ActionTerm] = G1CuroboIKAction

    mode: Literal["left_arm", "right_arm", "both_arm"] = "left_arm"
    """Which arm(s) this action term controls. Exposed for the env step / wrapper
    to decide how to compose the policy action vector:

    - ``"left_arm"`` / ``"right_arm"``: single solver, ``action_dim=7``,
      ``body_names`` has 1 entry, ``joint_names`` has 7 entries.
    - ``"both_arm"``: single solver call returns both arms, ``action_dim=14``
      (left pose 7-D then right pose 7-D), ``body_names`` has 2 entries
      (left EE then right EE), ``joint_names`` has 14 entries
      (left-arm 7 then right-arm 7).

    The action term validates these length invariants at init time."""

    joint_names: list[str] = field(default_factory=list)
    """Articulation joints the action term writes targets to, in this exact order.
    7 entries for single-arm modes, 14 for both-arm (left 7 then right 7)."""

    body_names: list[str] = field(default_factory=list)
    """End-effector body names. First entry becomes cuRobo's primary ``ee_link``
    (goal_pose target); any further entries are tracked via FK and addressed
    through ``solve_batch(link_poses={name: Pose})``. Length must match ``mode``
    (1 for single-arm, 2 for both-arm)."""

    base_link_name: str = "torso_link"
    """Base link of cuRobo's IK chain. Default is "torso_link" so the chain
    contains only the controlled arm joints (waist is upstream and excluded).
    The action term transforms pelvis-frame impedance goals into this link's
    frame each step using its measured runtime pose, so the IK is always
    consistent with the actual waist state."""

    lock_joint_names: list[str] = field(default_factory=list)
    """Joints in the kinematic chain (between base_link and the EEs) that should
    be pinned at 0. With ``base_link_name="torso_link"`` this is typically empty;
    with ``base_link_name="pelvis"`` it would include the 3 waist joints."""

    urdf_path: str = ""
    """Filled in by :func:`setup_curobo_ik_urdf` at env_cfg post-init."""

    # Solver knobs
    num_seeds: int = 1
    """Single-seed warm-started LM gives smooth, IK-branch-stable tracking.
    Bump to 4-12 if you need to escape unreachable local minima."""

    position_tolerance: float = 0.005
    orientation_tolerance: float = 0.05
    use_cuda_graph: bool = True
    """Old cuRobo handles fixed-iteration LBFGS internally so CUDA graphs are
    safe at the default. Required for any sensible large-batch throughput."""

    clamp_to_joint_limits: bool = True
    """Clamp cuRobo output to IsaacLab soft joint limits before writing the
    target to PhysX."""

    max_joint_target_delta: float | None = 0.25
    """Maximum per-env-step joint target jump in radians. This keeps cuRobo
    branch flips / far best-effort solutions from becoming one-frame PD
    impulses during the next PhysX step. Set to None or <=0 to disable."""

    fallback_to_current_on_failure: bool = False
    """If True, envs where cuRobo reports failure hold the measured joint
    position instead of using the best-effort solution. Leave False by default
    because cuRobo often marks usable workspace-edge solutions as failed."""

    debug_log_every: int = 0
    """If > 0, print IK pose-error / joint-norm diagnostics every N solver
    calls. 0 disables. Useful for sanity-checking frame alignment."""

    health_log_every: int = 0
    """If > 0, write a TSV row of GPU memory + solve-time + NaN + pose-error
    stats to ``/tmp/curobo_health_pid<PID>_<EPOCH>.tsv`` every N solver calls.
    Set this for long training runs to make freezes diagnosable. Columns:

      step, wallclock, elapsed_s, rss_mb, gpu_alloc_mb, gpu_reserved_mb,
      gpu_peak_alloc_mb, avg_solve_ms, max_solve_ms, p99_solve_ms,
      nan_in_total, nan_out_total, pos_err_mean_mm, pos_err_max_mm,
      success_rate, tgt_norm_mean, tgt_norm_max

    A freeze leaves the last row on disk; load it in pandas to plot the
    series and identify whether memory leaked, solve-time grew, or NaN
    appeared before the kernel hung. Also installs SIGUSR1 → Python stack
    trace (use before SIGKILL to capture the call site, if Python is still
    alive while the CUDA driver is stuck)."""


def setup_curobo_ik_urdf(
    actions_cfg,
    scene_cfg,
    urdf_link_prefix: str = "g1_29dof_rev_1_0_",
    root_link_rename: tuple[str, str] = ("g1_29dof_rev_1_0", "pelvis"),
):
    """Convert the G1 USD to a URDF on disk and stamp the path onto the
    cuRobo action cfg. Call from ``env_cfg.__post_init__`` after
    ``super().__post_init__()`` so ``scene.robot`` is configured.

    The USD→URDF conversion does two things we have to undo:
    1. Prefixes every non-root link with the asset prim name (we strip it).
    2. Uses the asset prim name as the URDF root link, even though our USD
       refers to that frame as ``pelvis`` (we rename the root link so the
       URDF matches ``G1BodyCfg.PELVIS_NAME`` and the impedance-command
       base frame).

    Args:
        urdf_link_prefix: Strip this prefix from every non-root link name
            and from every parent/child reference. Set ``""`` to disable.
        root_link_rename: ``(old, new)`` — rename the URDF root link from
            the asset prim name (``g1_29dof_rev_1_0``) to ``pelvis``. Set
            both to ``""`` to disable.
    """
    import tempfile
    import xml.etree.ElementTree as ET
    import isaaclab.controllers.utils as ControllerUtils

    temp_urdf_dir = tempfile.gettempdir()
    urdf_path, _mesh_path = ControllerUtils.convert_usd_to_urdf(
        scene_cfg.robot.spawn.usd_path, temp_urdf_dir, force_conversion=True
    )

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    n_stripped = 0
    if urdf_link_prefix:
        for link in root.findall("link"):
            name = link.get("name", "")
            if name.startswith(urdf_link_prefix):
                link.set("name", name[len(urdf_link_prefix):])
                n_stripped += 1
        for joint in root.findall("joint"):
            for tag in ("parent", "child"):
                ref = joint.find(tag)
                if ref is not None:
                    link_attr = ref.get("link", "")
                    if link_attr.startswith(urdf_link_prefix):
                        ref.set("link", link_attr[len(urdf_link_prefix):])

    n_renamed = 0
    old_root, new_root = root_link_rename
    if old_root and new_root and old_root != new_root:
        for link in root.findall("link"):
            if link.get("name") == old_root:
                link.set("name", new_root)
                n_renamed += 1
        for joint in root.findall("joint"):
            for tag in ("parent", "child"):
                ref = joint.find(tag)
                if ref is not None and ref.get("link") == old_root:
                    ref.set("link", new_root)

    tree.write(urdf_path)
    print(
        f"[setup_curobo_ik_urdf] urdf={urdf_path} | "
        f"stripped prefix '{urdf_link_prefix}' from {n_stripped} links | "
        f"renamed root '{old_root}' → '{new_root}' ({n_renamed} link(s))"
    )

    # Stamp urdf_path onto every G1CuroboIKActionCfg attribute on the actions
    # cfg. This is attribute-name agnostic so it works whether the action term
    # is named LeftArmAction (G1LeftArmCuroboIKCfg) or UpperBodyAction
    # (G1BothArmCuroboIKCfg) or anything else.
    n_set = 0
    set_names: list[str] = []
    for attr_name in dir(actions_cfg):
        if attr_name.startswith("_"):
            continue
        attr = getattr(actions_cfg, attr_name, None)
        if isinstance(attr, G1CuroboIKActionCfg):
            attr.urdf_path = urdf_path
            n_set += 1
            set_names.append(attr_name)
    if n_set == 0:
        raise RuntimeError(
            "setup_curobo_ik_urdf: no G1CuroboIKActionCfg attribute found on "
            "actions_cfg. Make sure your env_cfg uses one of the cuRobo action "
            "cfgs (e.g. G1LeftArmCuroboIKCfg, G1BothArmCuroboIKCfg)."
        )
    print(f"[setup_curobo_ik_urdf] set urdf_path on: {set_names}")
