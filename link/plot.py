import cv2
import matplotlib.cm as cm
import numpy as np


def sinebow(h):
    """A cyclic and uniform colormap, see http://basecase.org/env/on-rainbows."""
    f = lambda x: np.sin(np.pi * x) ** 2
    return np.stack([f(3 / 6 - h), f(5 / 6 - h), f(7 / 6 - h)], -1)


def visualize_depth(
    depth,
    acc=None,
    near=None,
    far=None,
    curve_fn=lambda x: np.log(x + np.finfo(np.float32).eps),
    modulus=0,
    colormap=None,
):
    """Visualize a depth map.
    Args:
        depth: A depth map.
        acc: An accumulation map, in [0, 1].
        near: The depth of the near plane, if None then just use the min().
        far: The depth of the far plane, if None then just use the max().
        curve_fn: A curve function that gets applied to `depth`, `near`, and `far`
        before the rest of visualization. Good choices: x, 1/(x+eps), log(x+eps).
        modulus: If > 0, mod the normalized depth by `modulus`. Use (0, 1].
        colormap: A colormap function. If None (default), will be set to
        matplotlib's viridis if modulus==0, sinebow otherwise.
    Returns:
        An RGB visualization of `depth`.
    """
    # If `near` or `far` are None, identify the min/max non-NaN values.
    eps = np.finfo(np.float32).eps
    near = near or np.min(np.nan_to_num(depth, np.inf)) + eps
    far = far or np.max(np.nan_to_num(depth, -np.inf)) - eps

    # Wrap the values around if requested.
    if modulus > 0:
        # Curve all values.
        depth, near, far = [curve_fn(x) for x in [depth, near, far]]
        value = np.mod(depth, modulus) / modulus
        colormap = colormap or sinebow
    else:
        # Scale to [0, 1].
        value = np.nan_to_num(np.clip((depth - near) / (far - near), 0, 1))
        colormap = colormap or cm.get_cmap("viridis")

    vis = colormap(value)[:, :, :3]

    # Set non-accumulated pixels to white.
    if acc is not None:
        vis = vis * acc[:, :, None] + (1 - acc)[:, :, None]

    return vis


def draw_line(img: np.ndarray, line: np.ndarray, color=[255, 0, 0], thickness=1):
    """
    Args:
        img: The image to be drawn.
        line: An array of shape (3,). It stores [a, b, c],
          the parameters of a 2D line ax + by + c = 0.
        color: A list of shape (3,). The line's color.

    Returns:
        The skewed matrix, a np.ndarray of shape (3, 3).
    """
    H, W, _ = img.shape
    x0, y0 = map(int, [0, -line[2] / line[1]])
    x1, y1 = map(int, [W, -(line[2] + line[0] * W) / line[1]])
    img = cv2.line(img, (x0, y0), (x1, y1), color, thickness)
    return img


def remap_using_flow_fields(
    image: np.ndarray, disp_x: np.ndarray, disp_y: np.ndarray, 
    interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT):
    """
    Create a new image where the value of each pixel (u, v) comes from 
    (u+disp_x[u, v], v+disp_y[u, v])
    For eacg pixel (u, v) in the new image, 

    Reference:
    https://stackoverflow.com/questions/46520123/how-do-i-use-opencvs-remap-function

    Args:
        image: image to be remapped. Shape (H, W, C).
        disp_x: displacement in the horizontal direction for each pixel. Shape: (H, W).
        disp_y: displacement in the vertical direction for each pixel. Shape: (H, W).
        interpolation
        border_mode
    Returns:
        remapped image, a np.ndarray of shape (H, W, C).
    """
    H, W = disp_x.shape[:2]

    # Create the new image
    X, Y = np.meshgrid(np.linspace(0, W - 1, W),
                       np.linspace(0, H - 1, H))
    map_x = (X+disp_x).astype(np.float32)
    map_y = (Y+disp_y).astype(np.float32)
    remapped_image = cv2.remap(image, map_x, map_y, interpolation=interpolation, borderMode=border_mode)

    return remapped_image
