import numpy as np


def convert_pose(pose: np.ndarray) -> np.ndarray:
    """Convert the pose between OpenCV and OpenGL format by flipping y- and z-axis.
    Args:
        pose: A 4x4 numpy array that represents a 6-DoF pose.
    Returns:
        A converted `pose`.
    """
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    pose = np.matmul(pose, flip_yz)
    return pose
