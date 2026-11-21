import numpy as np
from scipy.spatial.transform import Rotation as R



def quat_apply(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Apply a quaternion rotation to a vector.

    Args:
        quat: The quaternion in (w, x, y, z). Shape is (4,).
        vec: The vector in (x, y, z). Shape is (3,).

    Returns:
        The rotated vector in (x, y, z). Shape is (3,).
    """
    # extract components from quaternion
    w = quat[0]
    xyz = quat[1:]  # (x, y, z) components
    
    # compute cross products
    t = np.cross(xyz, vec) * 2
    return vec + w * t + np.cross(xyz, t)


def quat_apply_inverse(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Apply an inverse quaternion rotation to a vector.

    Args:
        quat: The quaternion in (w, x, y, z). Shape is (4,).
        vec: The vector in (x, y, z). Shape is (3,).

    Returns:
        The rotated vector in (x, y, z). Shape is (3,).
    """
    # extract components from quaternion
    w = quat[0]
    xyz = quat[1:]  # (x, y, z) components
    
    # compute cross products
    t = np.cross(xyz, vec) * 2
    return vec - w * t + np.cross(xyz, t)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Computes the conjugate of a quaternion.

    Args:
        q: The quaternion orientation in (w, x, y, z). Shape is (4,).

    Returns:
        The conjugate quaternion in (w, x, y, z). Shape is (4,).
    """
    return q * np.array([1, -1, -1, -1])


def quat_inv(q: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Computes the inverse of a quaternion.

    Args:
        q: The quaternion orientation in (w, x, y, z). Shape is (4,).
        eps: A small value to avoid division by zero. Defaults to 1e-9.

    Returns:
        The inverse quaternion in (w, x, y, z). Shape is (4,).
    """
    return quat_conjugate(q) / np.maximum(np.sum(q**2), eps)


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions together.

    Args:
        q1: The first quaternion in (w, x, y, z). Shape is (4,).
        q2: The second quaternion in (w, x, y, z). Shape is (4,).

    Returns:
        The product of the two quaternions in (w, x, y, z). Shape is (4,).

    Raises:
        ValueError: Input shapes of q1 and q2 are not matching.
    """
    # check input is correct
    if q1.shape != q2.shape:
        msg = f"Expected input quaternion shape mismatch: {q1.shape} != {q2.shape}."
        raise ValueError(msg)
    
    # extract components from quaternions
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    
    # perform multiplication
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    return np.array([w, x, y, z])


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def transform_imu_data(waist_yaw, waist_yaw_omega, imu_quat, imu_omega):
    RzWaist = R.from_euler("z", waist_yaw).as_matrix()
    R_torso = R.from_quat([imu_quat[1], imu_quat[2], imu_quat[3], imu_quat[0]]).as_matrix()
    R_pelvis = np.dot(R_torso, RzWaist.T)
    w = np.dot(RzWaist, imu_omega[0]) - np.array([0, 0, waist_yaw_omega])
    return R.from_matrix(R_pelvis).as_quat()[[3, 0, 1, 2]], w

def transform_imu_data_pelvis_to_torso(waist_yaw, waist_yaw_omega, pelvis_quat, pelvis_omega):
    """
    Transform IMU data from pelvis frame to torso frame
    
    Args:
        waist_yaw: waist yaw angle (rotation from torso to pelvis)
        waist_yaw_omega: waist yaw angular velocity
        pelvis_quat: quaternion in pelvis frame [w, x, y, z]
        pelvis_omega: angular velocity in pelvis frame
    
    Returns:
        torso_quat: quaternion in torso frame [w, x, y, z]
        torso_omega: angular velocity in torso frame
    """
    # Create rotation matrix from torso to pelvis (around z-axis)
    RzWaist = R.from_euler("z", waist_yaw).as_matrix()
    
    # Get pelvis orientation matrix
    R_pelvis = R.from_quat([pelvis_quat[1], pelvis_quat[2], pelvis_quat[3], pelvis_quat[0]]).as_matrix()
    
    # Transform to torso frame: R_torso = R_pelvis * RzWaist
    R_torso = np.dot(R_pelvis, RzWaist)
    
    # Transform angular velocity to torso frame
    # torso_omega = RzWaist^T * pelvis_omega + [0, 0, waist_yaw_omega]
    torso_omega = np.dot(RzWaist.T, pelvis_omega) + np.array([0, 0, waist_yaw_omega])
    
    # Convert rotation matrix back to quaternion [w, x, y, z]
    torso_quat = R.from_matrix(R_torso).as_quat()[[3, 0, 1, 2]]
    
    return torso_quat, torso_omega



def is_torso_bad_orientation(waist_euler_xyz, pelvis_quat, limit_angle_deg):
    R_Waist = R.from_euler("xyz", waist_euler_xyz).as_matrix()
    R_pelvis = R.from_quat([pelvis_quat[1], pelvis_quat[2], pelvis_quat[3], pelvis_quat[0]]).as_matrix()
    torso_rot_world = np.dot(R_pelvis, R_Waist)

    torso_quat_world = R.from_matrix(torso_rot_world).as_quat()
    torso_quat_world = np.array([torso_quat_world[3], torso_quat_world[0], torso_quat_world[1], torso_quat_world[2]])
    torso_proj = get_gravity_orientation(torso_quat_world)
    tilt_angle = np.arccos(-torso_proj[2])
    
    limit_angle_rad = np.deg2rad(limit_angle_deg)
    is_bad = np.abs(tilt_angle) > limit_angle_rad
    
    return is_bad


def is_object_bad_orientation(waist_euler_xyz, pelvis_quat, object_camera_quat, limit_angle_deg):
    
    R_Waist = R.from_euler("xyz", waist_euler_xyz).as_matrix()
    R_pelvis_w = R.from_quat([pelvis_quat[1], pelvis_quat[2], pelvis_quat[3], pelvis_quat[0]]).as_matrix()
    R_torso_w = np.dot(R_pelvis_w, R_Waist)
    R_camera_torso = R.from_euler("xyz", np.array([0, 0.8307767239493009, 0])).as_matrix()
    R_object_camera = R.from_quat(np.array([object_camera_quat[1], object_camera_quat[2], object_camera_quat[3],object_camera_quat[0]])).as_matrix()

    R_camera_w = np.dot(R_torso_w, R_camera_torso)
    R_object_w = np.dot(R_camera_w, R_object_camera)
    object_quat_world = R.from_matrix(R_object_w).as_quat()
    object_quat_world = np.array([object_quat_world[3], object_quat_world[0], object_quat_world[1] ,object_quat_world[2]])
    object_proj = get_gravity_orientation(object_quat_world)
    tilt_angle = np.arccos(-object_proj[2])
    
    limit_angle_rad = np.deg2rad(limit_angle_deg)
    is_bad = np.abs(tilt_angle) > limit_angle_rad
    
    return is_bad

def get_object_gravity_orientation(object_camera_quat, waist_euler_xyz, pelvis_quat):
    R_Waist = R.from_euler("xyz", waist_euler_xyz).as_matrix()
    R_pelvis_w = R.from_quat([pelvis_quat[1], pelvis_quat[2], pelvis_quat[3], pelvis_quat[0]]).as_matrix()
    R_torso_w = np.dot(R_pelvis_w, R_Waist)
    R_camera_torso = R.from_euler("xyz", np.array([0, 0.8307767239493009, 0])).as_matrix()
    R_object_camera = R.from_quat(np.array([object_camera_quat[1], object_camera_quat[2], object_camera_quat[3],object_camera_quat[0]])).as_matrix()

    R_camera_w = np.dot(R_torso_w, R_camera_torso)
    R_object_w = np.dot(R_camera_w, R_object_camera)
    object_quat_world = R.from_matrix(R_object_w).as_quat()
    object_quat_world = np.array([object_quat_world[3], object_quat_world[0], object_quat_world[1] ,object_quat_world[2]])
    object_proj = get_gravity_orientation(object_quat_world)
    return object_proj