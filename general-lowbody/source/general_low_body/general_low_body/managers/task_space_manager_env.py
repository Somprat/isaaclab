import os
import time
from typing import Sequence
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg, ManagerBasedEnv
from .reward_group_manager import RewardGroupManager
from isaaclab.managers import CommandManager, TerminationManager, CurriculumManager
from isaaclab.managers.action_manager import ActionManager
from isaaclab.managers.observation_manager import ObservationManager
from isaaclab.managers.recorder_manager import RecorderManager
from isaaclab.managers.event_manager import EventManager
import torch
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.assets import Articulation
from general_low_body import mdp
from .dense_command_manager import DenseCommandManager
from isaaclab.managers import RewardManager
from general_low_body.mdp import ImpedanceCommand

class TaskSpaceManagerEnv(ManagerBasedRLEnv):
    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.robot: Articulation = self.scene["robot"]
        self._cuda_sync_debug = os.getenv("GLB_CUDA_SYNC_DEBUG", "0") == "1"
        self._cuda_sync_debug_file = None
        if self._cuda_sync_debug:
            log_path = f"/tmp/glb_cuda_sync_pid{os.getpid()}_{int(time.time())}.tsv"
            self._cuda_sync_debug_file = open(log_path, "w", buffering=1)
            self._cuda_sync_debug_file.write("rl_step\tsim_step\tstage\tevent\twallclock\n")
            print(f"[TaskSpaceManagerEnv] CUDA sync debug log -> {log_path}")

        # Periodic torch.cuda.empty_cache() to probe whether the long-run freeze
        # is driven by PyTorch caching-allocator fragmentation. cuRobo allocates
        # many short-lived autograd intermediates per solve; over hours, the
        # caching allocator can fragment in a way that downstream PhysX kernel
        # launches deadlock on (sync barriers don't reach into the allocator).
        # Set ``GLB_EMPTY_CACHE_EVERY=0`` to disable. Default 100 env steps
        # (~2s at 50Hz) — empty_cache is ~1ms, so per-hour overhead is ~2s.
        try:
            self._empty_cache_every = int(os.getenv("GLB_EMPTY_CACHE_EVERY", "100"))
        except ValueError:
            self._empty_cache_every = 0
        if self._empty_cache_every > 0:
            print(
                f"[TaskSpaceManagerEnv] torch.cuda.empty_cache() every "
                f"{self._empty_cache_every} env steps"
            )



        # Check for fixed base
        if "LowerBodyAction" not in self.action_manager.active_terms:
            self.lower_body_action_dim = 0 # Fixed Base
        else:
            self.lower_body_action_dim = self.action_manager.get_term("LowerBodyAction").action_dim

        # ------------------------------------------------------------------
        # arm_mode is derived from which EE impedance commands are active.
        # Commands are the source of truth: they declare which EEs the task
        # is actually tracking. Action terms must then match this layout.
        # ------------------------------------------------------------------
        self.use_left_ee_command = "left_ee_position" in self.command_manager.active_terms
        self.use_right_ee_command = "right_ee_position" in self.command_manager.active_terms
        if self.use_left_ee_command and self.use_right_ee_command:
            self.arm_mode = "both_arm"
        elif self.use_left_ee_command:
            self.arm_mode = "left_arm"
        elif self.use_right_ee_command:
            self.arm_mode = "right_arm"
        else:
            raise ValueError(
                "TaskSpaceManagerEnv requires at least one of left_ee_position / "
                "right_ee_position EE impedance commands."
            )

        # Validate the action manager's arm terms collectively consume the
        # [left(7), right(7)] upper-body slice that step() constructs every
        # tick. Two valid layouts:
        #   1) UpperBodyAction (14-D, dual-arm cuRobo) — only valid in both_arm.
        #   2) LeftArmAction (7-D) + RightArmAction (7-D) — valid in any mode;
        #      the slot for a side without an EE command is filled with zeros.
        active = self.action_manager.active_terms
        if "UpperBodyAction" in active:
            ub_dim = self.action_manager.get_term("UpperBodyAction").action_dim
            if ub_dim != 14:
                raise ValueError(
                    f"UpperBodyAction must have action_dim=14 (got {ub_dim})."
                )
            if self.arm_mode != "both_arm":
                raise ValueError(
                    f"UpperBodyAction (dual-arm) requires both left_ee_position "
                    f"AND right_ee_position commands (current arm_mode={self.arm_mode})."
                )
        elif not ("LeftArmAction" in active and "RightArmAction" in active):
            raise ValueError(
                "TaskSpaceManagerEnv requires either UpperBodyAction (dual-arm cuRobo) "
                "or LeftArmAction + RightArmAction in the action manager."
            )

    def _ee_reference_pose_b(self, side: str) -> torch.Tensor:
        """7-D ``[pos(3), quat(4)]`` reference for one EE from the impedance
        command, or zeros if no command term is configured for that side.
        Selects open-loop / closed-loop / temporal slot based on the
        command's ``integrate_method``."""
        if side == "left":
            use = self.use_left_ee_command
            term_name = "left_ee_position"
        elif side == "right":
            use = self.use_right_ee_command
            term_name = "right_ee_position"
        else:
            raise ValueError(f"side must be 'left' or 'right', got {side}")
        if not use:
            return torch.zeros(self.num_envs, 7, device=self.device)
        cmd: ImpedanceCommand = self.command_manager.get_term(term_name)
        mode = cmd.integrate_method
        if mode == "open_loop":
            pos = cmd.reference_position_openloop_b
            quat = cmd.reference_orientation_quat_openloop_b
        elif mode == "closed_loop":
            pos = cmd.reference_position_closedloop_b
            quat = cmd.reference_orientation_quat_closedloop_b
        else:
            pos = cmd.reference_position_temporal_b
            quat = cmd.reference_orientation_quat_temporal_b
        return torch.cat([pos, quat], dim=-1)

    def load_managers(self):
        # note: this order is important since observation manager needs to know the command and action managers
        # and the reward manager needs to know the termination manager
        # -- command manager
        self.command_manager: DenseCommandManager = DenseCommandManager(self.cfg.commands, self)
        print("[INFO] Command Manager: ", self.command_manager)

        # -- event manager (we print it here to make the logging consistent)
        print("[INFO] Event Manager: ", self.event_manager)
        # -- recorder manager
        self.recorder_manager = RecorderManager(self.cfg.recorders, self)
        print("[INFO] Recorder Manager: ", self.recorder_manager)
        # -- action manager
        self.action_manager = ActionManager(self.cfg.actions, self)
        print("[INFO] Action Manager: ", self.action_manager)
        # -- observation manager
        self.observation_manager = ObservationManager(self.cfg.observations, self)
        print("[INFO] Observation Manager:", self.observation_manager)

        # prepare the managers
        # -- termination manager
        self.termination_manager = TerminationManager(self.cfg.terminations, self)
        print("[INFO] Termination Manager: ", self.termination_manager)
        # -- reward manager
        self.reward_manager = RewardManager(self.cfg.rewards, self)
        print("[INFO] Reward Manager: ", self.reward_manager)
        # -- curriculum manager
        self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        print("[INFO] Curriculum Manager: ", self.curriculum_manager)

        # setup the action and observation spaces for Gym
        self._configure_gym_env_spaces()

        # perform events at the start of the simulation
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")

    def zero_external_wrenches(self) -> None:
        for asset in self.scene.articulations.values():
            if getattr(asset, "has_external_wrench", False):
                asset._external_force_b.zero_()
                asset._external_torque_b.zero_()
                asset._external_wrench_positions_b.zero_()
                asset.uses_external_wrench_positions = False
                asset.has_external_wrench = False

    def _cuda_sync_barrier(self, stage: str) -> None:
        """Force CUDA work to complete at the cuRobo/PhysX handoff.

        The left-arm cuRobo task can silently deadlock in PhysX unless the
        cuRobo/action CUDA work and the PhysX step are ordered explicitly.
        Keep the sync itself always-on; GLB_CUDA_SYNC_DEBUG only controls the
        breadcrumb TSV around the sync points.
        """
        if self._cuda_sync_debug_file is not None:
            self._cuda_sync_debug_file.write(
                f"{self.common_step_counter}\t{self._sim_step_counter}\t{stage}\tbefore_sync\t{time.time():.6f}\n"
            )
        torch.cuda.synchronize()
        if self._cuda_sync_debug_file is not None:
            self._cuda_sync_debug_file.write(
                f"{self.common_step_counter}\t{self._sim_step_counter}\t{stage}\tafter_sync\t{time.time():.6f}\n"
            )

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        # clip actions
        action = action.to(self.device)
        action_clip = getattr(self.cfg, "action_clip", float("inf"))
        action = torch.clip(action, -action_clip, action_clip)

        # The policy's upper-body slice is REPLACED by the impedance references
        # every step (the policy doesn't choose arm targets in this task-space
        # env — only the lower body is policy-controlled).
        #
        # Action layout is always [left(7), right(7), lower]:
        # - UpperBodyAction (14-D) consumes left+right as one slice.
        # - LeftArmAction (7-D) + RightArmAction (7-D) consume them separately.
        # The action manager routes by term order, so a single cat is enough.
        left_arm_action = self._ee_reference_pose_b("left")
        right_arm_action = self._ee_reference_pose_b("right")
        if self.lower_body_action_dim > 0:
            lower_body_action = action[:, -self.lower_body_action_dim:]
            action = torch.cat([left_arm_action, right_arm_action, lower_body_action], dim=-1)
        else:
            action = torch.cat([left_arm_action, right_arm_action], dim=-1)

        # process actions
        self.action_manager.process_action(action.to(self.device))
        self._cuda_sync_barrier("after_process_action")

        # Periodic empty_cache after cuRobo solve + sync. Done here (not at
        # step start) so the cuRobo intermediates that just got freed by the
        # autograd graph teardown actually return to the OS pool, BEFORE
        # PhysX's sim.step claims a large block.
        if (
            self._empty_cache_every > 0
            and self.common_step_counter % self._empty_cache_every == 0
        ):
            torch.cuda.empty_cache()

        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # zero out external wrenches before stepping
            self.zero_external_wrenches()
            # update substep commands
            self.command_manager.compute_substep(dt=self.physics_dt)
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            self._cuda_sync_barrier("before_sim_step")
            # simulate
            self.sim.step(render=False)
            self._cuda_sync_barrier("after_sim_step")
            # Isaac Lab 2.2 has no recorder hook between physics substeps.
            if hasattr(self.recorder_manager, "record_post_physics_decimation_step"):
                self.recorder_manager.record_post_physics_decimation_step()
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            # update observations for recording if needed
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            # trigger recorder terms for pre-reset calls
            self.recorder_manager.record_pre_reset(reset_env_ids)

            self._reset_idx(reset_env_ids)

            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

            # trigger recorder terms for post-reset calls
            self.recorder_manager.record_post_reset(reset_env_ids)

        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute(update_history=True)

        # clip observations
        obs_clip = getattr(self.cfg, "obs_clip", float("inf"))
        for key, value in self.obs_buf.items():
            self.obs_buf[key] = torch.clip(value, -obs_clip, obs_clip)
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
