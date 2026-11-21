import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.envs.common import ViewerCfg

from .config import *
from general_low_body.assets import G1_MIMIC_DELAYED_CFG
from general_low_body import mdp



@configclass
class G1RandomUpperActionEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: G1RobotSceneCfg = G1RobotSceneCfg(num_envs=8192, env_spacing=2.5, replicate_physics=True)
    # Basic settings
    observations: RandomUpperBodyActionObservationsCfg = RandomUpperBodyActionObservationsCfg()
    actions: G1RandomUpperActionsCfg = G1RandomUpperActionsCfg()
    commands: LocomotionCommandsCfg = LocomotionCommandsCfg()
    commands.base_velocity.ranges = commands.base_velocity.limit_ranges
    # MDP settings
    rewards: LocomotionRewardCfg = LocomotionRewardCfg()
    terminations: LocomotionTerminationsCfg = LocomotionTerminationsCfg()
    events: G1RobotEventCfg = G1RobotEventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    curriculum.lin_vel_cmd_levels = None
    curriculum.ang_vel_cmd_levels = None

    action_clip = G1BodyCfg.ACTION_CLIP
    obs_clip = G1BodyCfg.OBS_CLIP

    viewer: ViewerCfg = ViewerCfg(
        origin_type="asset_root",
        asset_name="robot",
        eye=[2.5, -2.0, 1.8],
        lookat=[0.0, 0.0, 0.0],
    )


    # body keys
    body_keys = ['upper_body', 'lower_body']

    def __post_init__(self):
        """Post initialization."""
        self.scene.robot = G1_MIMIC_DELAYED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.robot_contact_sensor.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt



@configclass
class G1RandomUpperActionPlayEnvCfg(G1RandomUpperActionEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        
