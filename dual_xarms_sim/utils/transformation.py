from scipy.spatial.transform import Rotation as R
import numpy as np


def construct_adjoint_matrix(tcp_pose):
    """
    Construct the adjoint matrix for a spatial velocity vector
    :args: tcp_pose: (x, y, z, qx, qy, qz, qw)
    """
    rotation = R.from_quat(tcp_pose[3:]).as_matrix()
    translation = np.array(tcp_pose[:3])
    skew_matrix = np.array(
        [
            [0, -translation[2], translation[1]],
            [translation[2], 0, -translation[0]],
            [-translation[1], translation[0], 0],
        ]
    )
    adjoint_matrix = np.zeros((6, 6))
    adjoint_matrix[:3, :3] = rotation
    adjoint_matrix[3:, 3:] = rotation
    adjoint_matrix[3:, :3] = skew_matrix @ rotation
    return adjoint_matrix


def construct_homogeneous_matrix(tcp_pose):
    """
    Construct the homogeneous transformation matrix from given pose.
    args: tcp_pose: (x, y, z, qx, qy, qz, qw)
    """
    rotation = R.from_quat(tcp_pose[3:]).as_matrix()
    translation = np.array(tcp_pose[:3])
    T = np.zeros((4, 4))
    T[:3, :3] = rotation
    T[:3, 3] = translation
    T[3, 3] = 1
    return T

def compute_relative_poses_batch(left_poses, right_poses):
    """
    Computes the left end-effector poses relative to the right end-effector poses in a batched manner.

    Parameters:
        left_poses  : np.ndarray of shape (B, 7) where each row is [x, y, z, qx, qy, qz, qw]
        right_poses : np.ndarray of shape (B, 7) where each row is [x, y, z, qx, qy, qz, qw]

    Returns:
        relative_poses: np.ndarray of shape (B, 7) where each row is the relative pose [x, y, z, qx, qy, qz, qw]
    """
    # Extract translations and quaternions for both left and right poses
    t_left = left_poses[:, :3]    # Shape: (B, 3)
    quat_left = left_poses[:, 3:]  # Shape: (B, 4)

    t_right = right_poses[:, :3]    # Shape: (B, 3)
    quat_right = right_poses[:, 3:]  # Shape: (B, 4)

    # Create batch Rotation objects
    R_left = R.from_quat(quat_left)   # Each row is treated as one rotation.
    R_right = R.from_quat(quat_right)

    # Compute the inverse rotations of the right poses (batched)
    R_right_inv = R_right.inv()

    # Compute the relative rotation: R_relative = R_right_inv * R_left
    # This performs elementwise multiplication for the batch.
    R_relative = R_right_inv * R_left
    quat_relative = R_relative.as_quat()  # Shape: (B, 4)

    # Compute the relative translation:
    # First, find the difference in translations (in world frame)
    delta_t = t_left - t_right  # Shape: (B, 3)
    # Rotate this difference into the right arm's frame.
    t_relative = R_right_inv.apply(delta_t)  # Shape: (B, 3)

    # Concatenate the relative translation and rotation to get the full relative pose
    relative_poses = np.hstack([t_relative, quat_relative], dtype=np.float32)  # Shape: (B, 7)

    return relative_poses

def compute_relative_velocities_batch(left_vel, right_vel, left_pose, right_pose):
    """
    Computes the left end-effector velocities relative to the right end-effector frame 
    in a batched (vectorized) manner.

    Parameters:
        left_vel : np.ndarray of shape (B, 6)
                   Each row is [vx, vy, vz, wx, wy, wz] for the left end-effector in world frame.
        right_vel: np.ndarray of shape (B, 6)
                   Each row is [vx, vy, vz, wx, wy, wz] for the right end-effector in world frame.
        left_pose: np.ndarray of shape (B, 7)
                   Each row is [x, y, z, qx, qy, qz, qw] for the left end-effector in world frame.
        right_pose: np.ndarray of shape (B, 7)
                   Each row is [x, y, z, qx, qy, qz, qw] for the right end-effector in world frame.

    Returns:
        relative_vel: np.ndarray of shape (B, 6)
                      Each row is the left end-effector's velocity relative to the right end-effector,
                      expressed in the right end-effector's frame as [vx, vy, vz, wx, wy, wz].
    """
    # Extract positions from the poses
    t_left = left_pose[:, :3]    # Shape: (B, 3)
    t_right = right_pose[:, :3]  # Shape: (B, 3)

    # Extract rotations from the right pose (we need these to transform into right's frame)
    quat_right = right_pose[:, 3:]  # Shape: (B, 4) in [qx, qy, qz, qw] order.
    R_right = R.from_quat(quat_right)
    R_right_inv = R_right.inv()     # This will rotate vectors from world into the right TCP's frame.

    # Extract linear and angular velocity components
    v_left    = left_vel[:, :3]   # Linear velocity of left TCP in world frame.
    omega_left = left_vel[:, 3:]  # Angular velocity of left TCP in world frame.

    v_right    = right_vel[:, :3]   # Linear velocity of right TCP in world frame.
    omega_right = right_vel[:, 3:]  # Angular velocity of right TCP in world frame.

    # Compute the difference in positions (from right to left)
    delta_t = t_left - t_right  # Shape: (B, 3)

    # Account for the velocity of a point displaced by delta_t due to the right's angular velocity.
    # (i.e. if the right arm rotates, a point offset from its origin would have an extra velocity)
    v_due_to_right_rotation = np.cross(omega_right, delta_t)

    # Compute the relative linear velocity in world frame:
    v_rel_world = v_left - (v_right + v_due_to_right_rotation)

    # Rotate the relative linear velocity into the right TCP's frame:
    v_rel_right = R_right_inv.apply(v_rel_world)

    # For angular velocity, the relative angular velocity in world frame is simply:
    omega_rel_world = omega_left - omega_right
    # Rotate into the right TCP's frame:
    omega_rel_right = R_right_inv.apply(omega_rel_world)

    # Concatenate linear and angular parts to get the full relative velocity (6D)
    relative_vel = np.hstack([v_rel_right, omega_rel_right]).astype(np.float32)

    return relative_vel

def transform_tcp_to_ee(arm_tcp_pose):
    """
    Transforms the tool center point (TCP) pose back to the end-effector (EE) pose.

    Parameters:
        arm_tcp_pose - TCP pose vector after modifications (x, y, z, qw, qx, qy, qz)

    Returns:
        reversed_ee_pose - Transformed back EE pose
    """
    # Extract the original rotation and compute its inverse
    rotation = R.from_quat(arm_tcp_pose[3:], scalar_first=True)
    # Calculate the original offset translation, then negate it
    offset_tcp_to_ee = np.array([0, 0, -0.215]) # 0.255 old long gripper
    tool_translation_world = rotation.apply(offset_tcp_to_ee)

    # Reverse the translation and apply the inverse rotation to find the original EE position
    ee_pose = arm_tcp_pose.copy()
    ee_pose[0:3] += tool_translation_world

    return ee_pose