from isaaclab.utils import configclass
from general_low_body import mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from general_low_body.utils import *
from general_low_body.assets import G1BodyCfg
from isaaclab.managers import SceneEntityCfg



@configclass
class LocomotionTerminationsCfg:
    """Termination terms for locomotion MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(
        func=mdp.link_height_below_minimum, 
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "minimum_height": G1BodyCfg.TERMINAL_HEIGHT
        }
    )
