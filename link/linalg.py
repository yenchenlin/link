import numpy as np


def skew(x: np.ndarray) -> np.ndarray:
    """
    Args:
        x: An array of shape (3,).

    Returns:
        The skew symmetric array of shape (3, 3).
    """
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
