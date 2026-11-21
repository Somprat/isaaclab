# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Online arm IK with actuator targets or prescribed arm states during body physics."""

import math

import torch

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse


class HandIKJointPositionAction(JointPositionAction):
    """Keep policy joint ordering; blend policy arm targets toward online IK.

    At influence zero this is exactly JointPositionAction, including clipping.
    At influence one, bounded policy corrections are added to differential IK.
    Legs and waist always retain the original joint-position mapping. One owner
    writes all final joint targets. IK runs once per control step, not substep.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if not math.isfinite(cfg.ik_influence) or not 0 <= cfg.ik_influence <= 1:
            raise ValueError("ik_influence must be in [0, 1]")
        if not isinstance(cfg.ramp_steps, int) or cfg.ramp_steps < 0:
            raise ValueError("ramp_steps must be a nonnegative control-step count")
        for name in ("max_ik_step", "max_reference_lead", "damping"):
            if not math.isfinite(getattr(cfg, name)) or getattr(cfg, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(cfg.arm_residual_limit) or cfg.arm_residual_limit < 0:
            raise ValueError("arm_residual_limit must be finite and nonnegative")
        if cfg.immediate_arm_placement and not cfg.direct_arm_placement:
            raise ValueError("Immediate arm placement requires direct_arm_placement")
        self._arms = []
        suffixes = (
            "shoulder_pitch_joint",
            "shoulder_roll_joint",
            "shoulder_yaw_joint",
            "elbow_joint",
            "wrist_roll_joint",
            "wrist_pitch_joint",
            "wrist_yaw_joint",
        )
        for side in ("left", "right"):
            names = [f"{side}_{suffix}" for suffix in suffixes]
            joint_ids = [self._asset.joint_names.index(name) for name in names]
            slots = [self._joint_names.index(name) for name in names]
            controller = DifferentialIKController(
                DifferentialIKControllerCfg(
                    command_type="pose" if cfg.track_orientation else "position",
                    use_relative_mode=False,
                    ik_method="dls",
                    ik_params={"lambda_val": cfg.damping},
                ),
                num_envs=self.num_envs,
                device=self.device,
            )
            self._arms.append((joint_ids, slots, controller))
        self._ik_references = [torch.zeros(self.num_envs, 7, device=self.device) for _ in self._arms]
        self._reference_valid = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        self.q_hold = torch.zeros(self.num_envs, 2, 7, device=self.device)
        """Measured left/right arm positions captured at the first control step at or after seven seconds."""
        self.q_hold_captured = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        """Whether each environment has captured its hold posture in the current episode."""
        if cfg.direct_arm_placement:
            manager = env.sim.physics_manager
            if not hasattr(manager, "register_post_step_callback"):
                raise ValueError("Direct arm placement requires the Newton physics backend")
            if env.cfg.sim.physics.num_substeps != 1 or env.cfg.sim.physics.use_cuda_graph:
                raise ValueError("Direct arm placement requires single-substep eager physics")
            if not cfg.ik_only_arms:
                raise ValueError("Direct arm placement requires ik_only_arms")
            self._arm_ids = [joint for joints, _, _ in self._arms for joint in joints]
            self._arm_slots = [slot for _, slots, _ in self._arms for slot in slots]
            self._arm_start = self._asset.data.joint_pos.torch[:, self._arm_ids].clone()
            self._arm_goal = self._arm_start.clone()
            self._arm_velocity = torch.zeros_like(self._arm_start)
            self._placement_step = 0
            manager.set_decimation(1)
            env._physics_handles_decimation = False
            manager.register_post_step_callback(self._restore_arm_state)

    @property
    def ik_influence(self):
        """Optional linear ramp uses global environment steps, independent of resets."""
        if self.cfg.ik_only_arms:
            return 1.0
        progress = 1.0 if self.cfg.ramp_steps == 0 else min(1.0, self._env.common_step_counter / self.cfg.ramp_steps)
        if self.cfg.follow_reach_curriculum:
            hands = self._env.command_manager.get_term(self.cfg.command_name)
            # No episode receives stronger IK than its sampled reach difficulty.
            progress = torch.minimum(
                hands.episode_reach_fraction[:, None], hands.targets.new_full((self.num_envs, 1), progress)
            )
        return self.cfg.ik_influence * progress

    def joint_reference_errors(self):
        """14 measured-minus-IK errors, left then right; zero for invalid/reset references."""
        errors = []
        for hand, (joint_ids, _, _) in enumerate(self._arms):
            current = self._asset.data.joint_pos.torch[:, joint_ids]
            error = current - self._ik_references[hand]
            errors.append(torch.where(self._reference_valid[:, hand, None], error, 0.0))
        return torch.cat(errors, dim=-1)

    def process_actions(self, actions):
        super().process_actions(actions)
        capture = (~self.q_hold_captured) & (self._env.episode_length_buf * self._env.step_dt >= 7.0)
        for hand, (joint_ids, _, _) in enumerate(self._arms):
            current = self._asset.data.joint_pos.torch[:, joint_ids]
            self.q_hold[:, hand] = torch.where(capture[:, None], current, self.q_hold[:, hand])
        self.q_hold_captured |= capture
        influence = self.ik_influence
        if not torch.is_tensor(influence) and influence == 0:
            self._reference_valid.zero_()
            return  # Preserve the original mapping and avoid any Jacobian work.
        command = self._env.command_manager.get_term(self.cfg.command_name)
        if command.robot is not self._asset:
            raise ValueError("Hand command and action must use the same robot")
        data = self._asset.data
        command.capture_horizontal_hold()
        actual = command.positions()
        actual_orientations = command.orientations() if self.cfg.track_orientation else None
        torso_q = data.body_link_quat_w.torch[:, command.torso_id]
        for hand, (joint_ids, slots, controller) in enumerate(self._arms):
            body_id = command.body_ids[hand]
            jac_body = body_id - 1 if self._asset.is_fixed_base else body_id
            columns = [j + self._asset.num_base_dofs for j in joint_ids]
            # Owned copy: never rotate or offset the shared data-layer Jacobian.
            jac = data.body_link_jacobian_w.torch[:, jac_body, :, columns].clone()
            lever = quat_apply(data.body_link_quat_w.torch[:, body_id], command.offsets[hand].expand(self.num_envs, -1))
            jac[:, :3] += torch.cross(jac[:, 3:], lever[:, :, None].expand(-1, -1, len(joint_ids)), dim=1)
            rotation = torso_q[:, None].expand(-1, len(joint_ids), -1)
            # Only arm columns participate: these joints cannot move torso_link.
            jac[:, :3] = quat_apply_inverse(rotation, jac[:, :3].transpose(1, 2)).transpose(1, 2)
            jac[:, 3:] = quat_apply_inverse(rotation, jac[:, 3:].transpose(1, 2)).transpose(1, 2)
            current = data.joint_pos.torch[:, joint_ids]
            identity = current.new_zeros(self.num_envs, 4)
            identity[:, 3] = 1
            if self.cfg.track_orientation:
                orientation = actual_orientations[:, hand]
                pose_target = torch.cat((command.targets[:, hand], command.orientation_targets[:, hand]), dim=-1)
                controller.set_command(pose_target, ee_quat=orientation)
            else:
                orientation = identity
                controller.set_command(command.targets[:, hand], ee_quat=orientation)
            solved = controller.compute(actual[:, hand], orientation, jac, current)
            valid = torch.isfinite(solved).all(-1, keepdim=True)

            ik_caps = 0.05
            correction = torch.where(valid, solved-current, 0.0)
            correction = torch.clamp(correction, min=-ik_caps, max=ik_caps)

            if self.cfg.persistent_reference and not self.cfg.ik_only_arms:
                previous = torch.where(self._reference_valid[:, hand, None], self._ik_references[hand], current)
                # Scale integration during blending, and bound target lead to avoid
                # accumulating an unreachable reference while the policy dominates.
                ik_target = (previous + correction * influence).clamp(
                    min=current - self.cfg.max_reference_lead,
                    max=current + self.cfg.max_reference_lead,
                )
            else:
                ik_target = current + correction
            limits = data.soft_joint_pos_limits.torch[:, joint_ids]
            target = ik_target.clamp(min=limits[..., 0], max=limits[..., 1])
            active = valid & (influence > 0)
            self._ik_references[hand][:] = torch.where(active, target, current)
            self._reference_valid[:, hand] = active[:, 0]
            if self.cfg.ik_only_arms:
                self._processed_actions[:, slots] = torch.where(valid, target, current)
                continue
            original = self._processed_actions[:, slots]
            # Policy outputs become bounded radian corrections, without the nominal-pose offset.
            residual = self.cfg.arm_residual_limit * torch.tanh(self.raw_actions[:, slots])
            corrected = (target + residual).clamp(min=limits[..., 0], max=limits[..., 1])
            blended = torch.lerp(original, corrected, influence)
            if self.cfg.arm_residual_limit > 0:
                blended = blended.clamp(min=limits[..., 0], max=limits[..., 1])
            self._processed_actions[:, slots] = torch.where(valid, blended, original)

    def apply_actions(self):
        """Submit body commands and place arms at the start of each physics substep."""
        if not self.cfg.direct_arm_placement:
            super().apply_actions()
            return
        if self._placement_step == 0:
            self._arm_start.copy_(self._asset.data.joint_pos.torch[:, self._arm_ids])
            self._arm_goal.copy_(self._processed_actions[:, self._arm_slots])
            if self.cfg.immediate_arm_placement:
                self._arm_velocity.zero_()
            else:
                self._arm_velocity.copy_((self._arm_goal - self._arm_start) / self._env.step_dt)
        self._write_arm_state()
        super().apply_actions()
        self._asset.set_joint_velocity_target_index(target=self._arm_velocity, joint_ids=self._arm_ids)

    def _write_arm_state(self):
        if self.cfg.immediate_arm_placement:
            position = self._arm_goal
        else:
            fraction = self._placement_step / self._env.cfg.decimation
            position = torch.lerp(self._arm_start, self._arm_goal, fraction)
        self._asset.write_joint_state_to_sim_index(
            position=position, velocity=self._arm_velocity, joint_ids=self._arm_ids
        )
        self._env.sim.forward()

    def _restore_arm_state(self):
        """Project arms onto their prescribed trajectory before sensors and observations."""
        self._placement_step += 1
        self._write_arm_state()
        self._placement_step %= self._env.cfg.decimation

    def reset(self, env_ids=None):
        """Re-seed only reset environments from measured joints on their next action."""
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._reference_valid[ids] = False
        self.q_hold_captured[ids] = False
        self.q_hold[ids] = 0.0


@configclass
class HandIKJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = HandIKJointPositionAction
    command_name: str = "hand_position"
    direct_arm_placement: bool = False
    """Prescribe interpolated arm states at each physics substep using online IK endpoints."""
    immediate_arm_placement: bool = False
    """Apply each IK endpoint immediately and hold it with zero joint velocity between solves."""
    ik_only_arms: bool = False
    """Give IK all arm commands, bypassing policy residuals, blending, ramps, and reference accumulation."""
    ik_influence: float = 0.0
    """Blend fraction in [0, 1]. Zero preserves the pretrained action mapping."""
    follow_reach_curriculum: bool = False
    """Cap each environment's IK ramp by its episode reach fraction."""
    ramp_steps: int = 0
    """Control steps to reach ik_influence; zero uses a constant blend."""
    track_orientation: bool = False
    """Solve full hand poses using torso-relative command orientations; false retains position-only IK."""
    persistent_reference: bool = True
    """Accumulate measured-state IK corrections into a bounded persistent reference."""
    max_reference_lead: float = 0.30
    """Maximum reference lead relative to measured arm joint positions (rad)."""
    max_ik_step: float = 0.05
    """Maximum IK reference increment per control step before influence scaling (rad)."""
    arm_residual_limit: float = 0.0
    """Maximum policy correction around IK in radians; zero preserves legacy IK-only control."""
    residual_limit: float = 0.0
    """Unused legacy field retained so older saved env.yaml files still load."""
    damping: float = 0.05
