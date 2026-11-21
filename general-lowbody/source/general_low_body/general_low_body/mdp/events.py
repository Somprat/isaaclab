import math
from typing import Optional, Sequence
import torch
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.envs import DirectRLEnv, ManagerBasedEnv
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from pxr import Gf, Sdf, UsdGeom, Vt
import omni
import isaaclab.sim as sim_utils
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_mul, quat_apply_inverse
from general_low_body.assets import G1BodyCfg


class UpperBodyVibration(ManagerTermBase):
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot", body_names=G1BodyCfg.UPPER_BODY_LINKS))
        self.robot_asset: Articulation = env.scene[self.asset_cfg.name]
        # resolve body indices
        if self.asset_cfg.body_ids == slice(None):
            self.upper_body_link_ids = torch.arange(self.robot_asset.num_bodies, dtype=torch.int, device=self.device)
        else:
            self.upper_body_link_ids = torch.tensor(self.asset_cfg.body_ids, dtype=torch.int, device=self.device)
        self.num_links = len(self.upper_body_link_ids)
        
        self.T_k_range = cfg.params.get("T_k_range", (0.64, 1.28))
        self.p_k_range = cfg.params.get("p_k_range", (-100.0, 100.0)) # impulse
        self.s_k_range = cfg.params.get("s_k_range", (-10.0, 10.0)) # oscillation amplitude
        self.phi_k_range = cfg.params.get("phi_k_range", (-math.pi, math.pi)) # phase shift
        self.impulse_std = cfg.params.get("impulse_std", 0.01) # standard deviation of the impulse
        self.num_signals = cfg.params.get("num_signals", 1)

        self.T_k = torch.zeros(self.num_envs, self.num_signals, 1, device=self.device)
        self.p_k = torch.zeros(self.num_envs, self.num_signals, 6, device=self.device)
        self.s_k = torch.zeros(self.num_envs, self.num_signals, 6, device=self.device)
        self.phi_k = torch.zeros(self.num_envs, self.num_signals, 1, device=self.device)
        self.acc_base = torch.zeros(self.num_envs, 6, device=self.device)
        self.vel_base = torch.zeros(self.num_envs, 6, device=self.device)
        self.t = torch.zeros(self.num_envs, 1, device=self.device)

        
        # sample initial values
        self._resample(torch.arange(self.num_envs, device=self.device))


    def __call__(
        self,
        env: DirectRLEnv | ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=G1BodyCfg.UPPER_BODY_LINKS),
        T_k_range: tuple[float, float] = (0.64, 1.28),
        p_k_range: tuple[float, float] = (-100.0, 100.0),
        s_k_range: tuple[float, float] = (-10.0, 10.0),
        phi_k_range: tuple[float, float] = (-math.pi, math.pi),
        impulse_std: float = 0.01, 
        num_signals: int = 1,
    ):
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=self.device)
        else:
            env_ids = env_ids.to(self.device)

        self.t[env_ids] += env.step_dt
        t = self.t[env_ids].unsqueeze(1)
        T_k = self.T_k[env_ids]     # (N_envs, N_sig, 1)
        p_k = self.p_k[env_ids]     # (N_envs, N_sig, 6)
        s_k = self.s_k[env_ids]     # (N_envs, N_sig, 6)
        phi_k = self.phi_k[env_ids] # (N_envs, N_sig, 1)
        
        # swing component
        arg = 2.0 * math.pi * t / T_k + phi_k
        acc_base_swing = s_k * torch.sin(arg)

        # impulse component
        t_in_cycle = torch.fmod(t, T_k)
        impulse_mean = T_k / 2.0
        g_t = torch.exp(-torch.square(t_in_cycle - impulse_mean) / (2.0 * self.impulse_std ** 2))
        acc_base_impulse = p_k * g_t

        # update base acceleration and velocity
        self.acc_base[env_ids] = torch.sum(acc_base_swing, dim=1) + torch.sum(acc_base_impulse, dim=1)
        self.vel_base[env_ids] += self.acc_base[env_ids] * env.step_dt

        # get upper body link states
        masses = self.robot_asset.data._root_physx_view.get_masses()
        link_mass = masses.to(self.device)[env_ids[:, None], self.upper_body_link_ids].unsqueeze(-1)
        link_lin_vel_w = self.robot_asset.data.body_lin_vel_w[env_ids[:, None], self.upper_body_link_ids, :]
        link_pos_w = self.robot_asset.data.body_pos_w[env_ids[:, None], self.upper_body_link_ids, :]
        inertias = self.robot_asset.data._root_physx_view.get_inertias()
        link_inertia = inertias.to(self.device)[env_ids[:, None], self.upper_body_link_ids, :]
        link_inertia = link_inertia.view(len(env_ids), self.num_links, 3, 3)

        # get local states
        root_pos_w = self.robot_asset.data.root_pos_w[env_ids, :].unsqueeze(1)
        root_quat_w = self.robot_asset.data.root_quat_w[env_ids, :].unsqueeze(1)
        root_vel_w = self.robot_asset.data.root_lin_vel_w[env_ids, :].unsqueeze(1)
        root_quat_w_expanded = root_quat_w.expand(-1, self.num_links, -1)
        link_pos_b = quat_apply_inverse(root_quat_w_expanded, link_pos_w - root_pos_w)
        link_vel_b = quat_apply_inverse(root_quat_w_expanded, link_lin_vel_w - root_vel_w)

        acc_base = self.acc_base[env_ids, :].unsqueeze(1)
        vel_base = self.vel_base[env_ids, :].unsqueeze(1)

        # compute force
        lin_acc_b = acc_base[..., :3]
        ang_acc_b = acc_base[..., 3:]
        ang_vel_b = vel_base[..., 3:]

        link_force_b = link_mass * (-lin_acc_b - torch.cross(ang_acc_b, link_pos_b, dim=-1) 
                                  - torch.cross(ang_vel_b, torch.cross(ang_vel_b, link_pos_b, dim=-1), dim=-1)
                                  - 2.0 * torch.cross(ang_vel_b, link_vel_b, dim=-1))
        
        link_torque_b = -torch.matmul(link_inertia, ang_acc_b.unsqueeze(-1)).squeeze(-1) \
              - torch.cross(ang_vel_b, torch.matmul(link_inertia, ang_vel_b.unsqueeze(-1)).squeeze(-1), dim=-1)


        # apply force and torque
        self.robot_asset.set_external_force_and_torque(link_force_b, link_torque_b, env_ids=env_ids, body_ids=self.upper_body_link_ids)
        return None

    def _resample(self, env_ids: torch.Tensor):
        n = len(env_ids)
        shape_1d = (n, self.num_signals, 1)
        shape_6d = (n, self.num_signals, 6)
        # resample period
        log_T_min = torch.log(torch.tensor(self.T_k_range[0], device=self.device))
        log_T_max = torch.log(torch.tensor(self.T_k_range[1], device=self.device))
        log_T = (log_T_max - log_T_min) * torch.rand(shape_1d, device=self.device) + log_T_min
        self.T_k[env_ids] = torch.exp(log_T)

        # resample impulse
        p_min, p_max = self.p_k_range
        self.p_k[env_ids] = (p_max - p_min) * torch.rand(shape_6d, device=self.device) + p_min

        # resample oscillation amplitude
        s_min, s_max = self.s_k_range
        self.s_k[env_ids] = (s_max - s_min) * torch.rand(shape_6d, device=self.device) + s_min

        # resample phase shift
        phi_min, phi_max = self.phi_k_range
        self.phi_k[env_ids] = (phi_max - phi_min) * torch.rand(shape_1d, device=self.device) + phi_min

        # reset
        self.t[env_ids] = 0.0
        self.acc_base[env_ids] = 0.0
        self.vel_base[env_ids] = 0.0

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._resample(env_ids)
        # zero force and torque
        zero_vec = torch.zeros(len(env_ids), self.num_links, 3, device=self.device)
        self.robot_asset.set_external_force_and_torque(zero_vec, zero_vec, env_ids=env_ids, body_ids=self.upper_body_link_ids)