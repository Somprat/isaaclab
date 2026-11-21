
"""
G1 Robot Constants and Configuration

This module contains joint names, body part definitions, and action sequences
for the G1 humanoid robot.
"""
class G1BodyCfg:

    REFERENCE_BODY = "torso_link"
    PELVIS_NAME = "pelvis"
    CAMERA_NAME = "d435_link"
    LEFT_EE_NAME = "left_wrist_yaw_link"
    RIGHT_EE_NAME = "right_wrist_yaw_link"
    TARGET_BASE_HEIGHT = 0.78
    TARGET_FEET_HEIGHT = 0.12
    TERMINAL_HEIGHT = 0.2
    DT = 0.005
    DECIMATION = 4
    CONTROLLER_DELAY_STEP = 4

    
    # ============================================
    # Clip
    # ============================================
    
    ACTION_CLIP = 100
    OBS_CLIP = 100

    # ============================================
    # Plate Offset
    # ============================================
    
    #PLATE_OFFSET = [0.29792, 0.0, 0.14100] # x, y, z offset from pelvis

    PLATE_OFFSET = [0.30225, 0.0, 0.14100] # x, y, z offset from pelvis
    PLATE_TERMINAL_HEIGHT = 0.7

    # ============================================
    # Lower Body
    # ============================================
    WAIST_JOINTS = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]

    HIPS_JOINTS = [".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_joint"]

    FEET_JOINTS = [".*_ankle_pitch_joint", ".*_ankle_roll_joint"]

    FEET_BODY_NAME = ".*_ankle_roll_link"

    LOWER_BODY_JOINTS = [
        'waist_yaw_joint', 
        'waist_roll_joint', 
        'waist_pitch_joint',
        'left_hip_pitch_joint', 
        'right_hip_pitch_joint', 
        'left_hip_roll_joint', 
        'right_hip_roll_joint', 
        'left_hip_yaw_joint', 
        'right_hip_yaw_joint', 
        'left_knee_joint', 
        'right_knee_joint',
        'left_ankle_pitch_joint', 
        'right_ankle_pitch_joint', 
        'left_ankle_roll_joint', 
        'right_ankle_roll_joint'
    ]

    LOWER_BODY_LINKS = [
        "pelvis",
        #"pelvis_contour_link",
        ".*_hip_pitch_link",
        ".*_hip_roll_link",
        ".*_hip_yaw_link",
        ".*_knee_link",
        "waist_yaw_link",
        "waist_roll_link",
        "torso_link",
    ] # NOTE: only for undesired contacts

    # ============================================
    # Upper Body
    # ============================================
    UPPER_BODY_JOINTS = [
        'left_shoulder_pitch_joint', 
        'right_shoulder_pitch_joint', 
        'left_shoulder_roll_joint', 
        'right_shoulder_roll_joint', 
        'left_shoulder_yaw_joint', 
        'right_shoulder_yaw_joint', 
        'left_elbow_joint', 
        'right_elbow_joint', 
        'left_wrist_roll_joint', 
        'right_wrist_roll_joint', 
        'left_wrist_pitch_joint', 
        'right_wrist_pitch_joint', 
        'left_wrist_yaw_joint', 
        'right_wrist_yaw_joint'
    ]

    LEFT_ARM_JOINTS = [
        'left_shoulder_pitch_joint', 
        'left_shoulder_roll_joint', 
        'left_shoulder_yaw_joint', 
        'left_elbow_joint', 
        'left_wrist_roll_joint', 
        'left_wrist_pitch_joint', 
        'left_wrist_yaw_joint'
    ]

    RIGHT_ARM_JOINTS = [
        'right_shoulder_pitch_joint', 
        'right_shoulder_roll_joint', 
        'right_shoulder_yaw_joint', 
        'right_elbow_joint', 
        'right_wrist_roll_joint', 
        'right_wrist_pitch_joint', 
        'right_wrist_yaw_joint'
    ]

    UPPER_BODY_LINKS = [
        ".*_shoulder_pitch_link",
        ".*_shoulder_roll_link",
        ".*_shoulder_yaw_link",
        ".*_elbow_link",
        ".*_wrist_roll_link",
        ".*_wrist_pitch_link",
        ".*_wrist_yaw_link",
        "torso_link",
    ] # NOTE: only for undesired contacts

    LEFT_ARM_BODY_LINKS = [
        "left_shoulder_pitch_link",
        "left_shoulder_roll_link",
        "left_shoulder_yaw_link",
        "left_elbow_link",
        "left_wrist_roll_link",
        "left_wrist_pitch_link",
        "left_wrist_yaw_link",
    ]

    RIGHT_ARM_BODY_LINKS = [
        "right_shoulder_pitch_link",
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
        "right_elbow_link",
        "right_wrist_roll_link",
        "right_wrist_pitch_link",
        "right_wrist_yaw_link",
    ]

    ACTION_DIM = {
        "upper_body": len(UPPER_BODY_JOINTS),
        "lower_body": len(LOWER_BODY_JOINTS),
    }
    NUM_ACTIONS = sum(ACTION_DIM.values())


    
    # ============================================
    # EE default pose
    # ============================================
    LEFT_EE_DEFAULT_POS = [0.2452, 0.1486, 0.0512]
    RIGHT_EE_DEFAULT_POS = [0.2452, -0.1486, 0.0512]

    LEFT_EE_DEFAULT_QUAT = [1.0, 0.0, 0.0, 0.0]
    RIGHT_EE_DEFAULT_QUAT = [1.0, 0.0, 0.0, 0.0]

    
    
    
    # ============================================
    # Joint SDK Names
    # ============================================
    JOINT_SDK_NAMES = [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]
    



