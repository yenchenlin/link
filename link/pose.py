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


def skew(x: np.ndarray) -> np.ndarray:
    """
    Args:
        x: A vector of shape (3,).

    Returns:
        The skew symmetric matrix of shape (3, 3).
    """
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


def compute_essential_matrix(
    extrinsic_A: np.ndarray, extrinsic_B: np.ndarray
) -> np.ndarray:
    """
    Args:
        extrinsic_A: First camera extrinsic of shape (4, 4).
        extrinsic_B: Second camera extrinsic of shape (4, 4).

    Returns:
        The fundamental matrix of shape (3, 3).
    """
    relative = extrinsic_B.dot(np.linalg.inv(extrinsic_A))
    R = relative[:3, :3]
    T = relative[:3, -1]
    S = skew(T)
    E = np.dot(S, R)
    return E


def compute_fundamental_matrix(
    extrinsic_A: np.ndarray,
    extrinsic_B: np.ndarray,
    intrinsic_A: np.ndarray,
    intrinsic_B: np.ndarray,
) -> np.ndarray:
    """
    Args:
        extrinsic_A: First camera extrinsic of shape (4, 4).
        extrinsic_B: Second camera extrinsic of shape (4, 4).
        intrinsic_A: First camera intrinsic of shape (3, 3).
        intrinsic_B: Second camera intrinsic of shape (3, 3).

    Returns:
        The essential matrix of shape (3, 3).
    """
    E = compute_essential_matrix(extrinsic_A, extrinsic_B)
    F = np.linalg.inv(intrinsic_B).T.dot(E).dot(np.linalg.inv(intrinsic_A))
    return F
