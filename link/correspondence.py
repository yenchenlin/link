import argparse

import cv2
import numpy as np


def colormap_from_heatmap(
    h,  # numpy array [H, W]
    normalize=False,  # whether or not to normalize to [0,1]
):  # np.ndarray [H, W, 3] 'rgb' ordering

    h_255 = None
    if normalize:
        h_255 = np.uint8(255 * h / np.max(h))
    else:
        h_255 = np.uint8(255 * h)

    colormap = cv2.applyColorMap(h_255, cv2.COLORMAP_JET)
    colormap_rgb = np.zeros_like(colormap)
    colormap_rgb[:, :, 0] = colormap[:, :, 0]
    colormap_rgb[:, :, 2] = colormap[:, :, 2]
    return colormap_rgb


def draw_reticle(img, u, v, label_color):
    """
    Draws a reticle on the image at the given (u,v) position
    :param img:
    :type img:
    :param u:
    :type u:
    :param v:
    :type v:
    :param label_color:
    :type label_color:
    :return:
    :rtype:
    """
    # cast to int
    u = int(u)
    v = int(v)

    white = (255, 255, 255)
    cv2.circle(img, (u, v), 10, label_color, 1)
    cv2.circle(img, (u, v), 11, white, 1)
    cv2.circle(img, (u, v), 12, label_color, 1)
    cv2.line(img, (u, v + 1), (u, v + 3), white, 1)
    cv2.line(img, (u + 1, v), (u + 3, v), white, 1)
    cv2.line(img, (u, v - 1), (u, v - 3), white, 1)
    cv2.line(img, (u - 1, v), (u - 3, v), white, 1)


class CorrespondenceVisualization:
    def __init__(self, rgb_A, feature_A, rgb_B, feature_B, pos=False):
        self.rgb_A = rgb_A
        self.feature_A = feature_A
        self.rgb_B = rgb_B
        self.feature_B = feature_B
        self.pos = pos

        self._paused = False
        cv2.namedWindow("source")

    def mouse_callback(self, event, u, v, flags, param):

        """
        For each network, find the best match in the target image to point highlighted
        with reticle in the source image. Displays the result
        :return:
        :rtype:
        """
        if self._paused:
            return

        self.find_best_match(u, v)

    def find_best_match(self, u, v):

        """
        For each network, find the best match in the target image to point highlighted
        with reticle in the source image. Displays the result
        :return:
        :rtype:
        """
        if self._paused:
            return

        # [2,1] = [2, N] with N = 1
        uv = np.ones([2, 1], dtype=np.int64)
        uv[0, 0] = u
        uv[1, 0] = v

        # Show img a
        rgb_A_with_reticle = np.copy(self.rgb_A)
        draw_reticle(rgb_A_with_reticle, u, v, [0, 255, 0])
        cv2.imshow("source", rgb_A_with_reticle)

        # Find match.
        H, W, C = self.feature_B.shape

        if self.pos:
            # L2 Norm
            feature_diff = np.linalg.norm(
                self.feature_A[v, u][None, None, :] - self.feature_B, axis=-1
            )
        else:
            # Cosine distance
            normalized_feature_A = self.feature_A[v, u][None, None, :] / (
                np.linalg.norm(self.feature_A[v, u][None, None, :], axis=2)[:, :, None]
                + 1e-8
            )
            normalized_feature_B = self.feature_B / (
                np.linalg.norm(self.feature_B, axis=2)[:, :, None] + 1e-8
            )
            feature_diff = np.linalg.norm(
                normalized_feature_A - normalized_feature_B, axis=-1
            )

        vu_pred = np.unravel_index(feature_diff.argmin(), feature_diff.shape)

        # Show img b
        rgb_B_with_reticle = np.copy(self.rgb_B)
        draw_reticle(rgb_B_with_reticle, vu_pred[1], vu_pred[0], [0, 255, 0])
        cv2.imshow("target", rgb_B_with_reticle)

        # Show heatmap
        heatmap_rgb = colormap_from_heatmap(
            np.exp(-1 * feature_diff / 0.1), normalize=True
        )
        heatmap_blend = (heatmap_rgb * 0.6 + self.rgb_B * 0.4).astype(np.uint8)  # blend

        cv2.imshow("heatmap", heatmap_blend)
        # self._image_save_dict['target'] = rgb_B_with_reticle

    def run(self):
        cv2.imshow("source", self.rgb_A)
        cv2.imshow("target", self.rgb_B)
        cv2.setMouseCallback("source", self.mouse_callback)

        while True:
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break
            elif k == ord("p"):
                if self._paused:
                    print("un pausing")
                    self._paused = False
                else:
                    print("pausing")
                    self._paused = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--image_A", type=str, help="the path of image A")
    parser.add_argument("--image_B", type=str, help="the path of image B")
    parser.add_argument("--feature_A", type=str, help="the path of feature A")
    parser.add_argument("--feature_B", type=str, help="the path of feature B")
    parser.add_argument("--pos", action="store_true", help="only use positions.")
    args = parser.parse_args()

    rgb_A = cv2.cvtColor(cv2.imread(args.image_A), cv2.COLOR_BGR2RGB)
    rgb_B = cv2.cvtColor(cv2.imread(args.image_B), cv2.COLOR_BGR2RGB)
    feature_A = np.load(args.feature_A)
    feature_B = np.load(args.feature_B)
    if args.pos:
        feature_A = feature_A[:, :, :3]
        feature_B = feature_B[:, :, :3]

    corr_vis = CorrespondenceVisualization(
        rgb_A, feature_A, rgb_B, feature_B, pos=args.pos
    )
    corr_vis.run()

    cv2.destroyAllWindows()
