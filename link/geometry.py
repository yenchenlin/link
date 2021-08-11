import numpy as np
from .linalg import skew


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