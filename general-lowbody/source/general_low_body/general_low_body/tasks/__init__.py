import gymnasium as gym
from general_low_body import agents



"""G1 Random Upper Action"""
gym.register(
    id="G1-Random-Upper-Action",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.random_upper_env_cfg:G1RandomUpperActionEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.random_upper_env_cfg:G1RandomUpperActionPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RandomUpperActionRunnerCfg",
    },
)

"""G1 Left Arm IK"""
gym.register(
    id="G1-Left-Arm-IK",
    entry_point="general_low_body.managers.task_space_manager_env:TaskSpaceManagerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.left_arm_ik_env_cfg:G1LeftArmIKEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.left_arm_ik_env_cfg:G1LeftArmIKPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1LeftArmIKRunnerCfg",
    },
)



"""G1 Left Arm IK"""
gym.register(
    id="G1-Fixed-Base",
    entry_point="general_low_body.managers.task_space_manager_env:TaskSpaceManagerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fixed_base_env_cfg:G1FixedBaseEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.fixed_base_env_cfg:G1FixedBasePlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1LeftArmIKRunnerCfg",
    },
)


"""G1 Left Arm cuRobo IK"""
gym.register(
    id="G1-Left-Arm-Curobo-IK",
    entry_point="general_low_body.managers.task_space_manager_env:TaskSpaceManagerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.left_arm_ik_env_cfg:G1LeftArmCuroboIKEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.left_arm_ik_env_cfg:G1LeftArmCuroboIKPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1LeftArmIKRunnerCfg",
    },
)



"""G1 Dual Arm cuRobo IK"""
gym.register(
    id="G1-Dual-Arm-IK",
    entry_point="general_low_body.managers.task_space_manager_env:TaskSpaceManagerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dual_arm_ik_env_cfg:G1DualArmIKEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.dual_arm_ik_env_cfg:G1DualArmIKPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1DualArmIKRunnerCfg",
    },
)