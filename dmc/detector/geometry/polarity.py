import numpy as np

from dmc.debug import DebugSink, NullSink, CvDebugSink
from dmc.data import LPattern
from dmc.viz import draw_sampled_border


class PolarityChecker:
    def __init__(self, debug: DebugSink = NullSink()):
        self.debug = debug

    def has_inverted_polarity(self, sample_img: np.ndarray, l_pattern: LPattern) -> bool:
        v1 = np.array(l_pattern.vertex1)
        v2 = np.array(l_pattern.vertex2)
        c = np.array(l_pattern.corner)

        arm1 = v1 - c
        arm2 = v2 - c
        arm1_len = np.linalg.norm(arm1)
        arm2_len = np.linalg.norm(arm2)
        arm1_hat = arm1 / arm1_len
        arm2_hat = arm2 / arm2_len

        arm_mean = self.scan_along_boundary(sample_img, arm1_hat, arm2_hat, np.array(c), int(arm1_len), 10, offset=5,
                                            direction=1)
        bg_mean = self.scan_along_boundary(sample_img, arm1_hat, arm2_hat, np.array(c), int(arm1_len), 10, offset=5,
                                           direction=-1)

        # self.debug.log(f"[border-fitter] flat boundary around arm 1: {flat_boundary_arm1}")
        # self.debug.log(f"[border-fitter] flat boundary around arm 2: {flat_boundary_arm2}")
        #
        # black_boundary_arm1 = np.where(flat_boundary_arm1 == 0)
        # black_boundary_arm2 = np.where(flat_boundary_arm2 == 0)
        #
        # self.debug.log(f"[border-fitter] black pixels around arm 1: {black_boundary_arm1}")
        # self.debug.log(f"[border-fitter] black pixels around arm 2: {black_boundary_arm2}")

        inverted = False
        contrast = arm_mean - bg_mean

        if abs(contrast) < 0.1 * 255:
            inverted = False
        else:
            inverted = contrast > 0

        # black_percentage_threshold = 0.7
        #
        # if black_boundary_arm1[0].shape[0] > black_percentage_threshold * flat_boundary_arm1.shape[0] and \
        #         black_boundary_arm2[0].shape[0] > black_percentage_threshold * flat_boundary_arm2.shape[0]:
        #     inverted = True

        return inverted

    def scan_along_boundary(
            self,
            sample_img: np.ndarray,
            u_hat: np.ndarray,
            v_hat: np.ndarray,
            origin: np.ndarray,
            u_len: int,
            v_len: int,
            offset: int,
            direction: int
    ):
        # offset the origin by a margin
        origin = origin + offset * (v_hat * direction)

        us = np.arange(u_len, dtype=float)
        vs = np.arange(v_len, dtype=float)
        pts = (origin[None, None, :]
               + us[:, None, None] * u_hat[None, None, :]
               + (direction * vs)[None, :, None] * v_hat[None, None, :])
        rows = np.clip(np.round(pts[..., 1]).astype(int), 0, sample_img.shape[0] - 1)
        cols = np.clip(np.round(pts[..., 0]).astype(int), 0, sample_img.shape[1] - 1)

        if isinstance(self.debug, CvDebugSink):
            for v in range(v_len):
                coords = [(x, y) for x, y in zip(list(cols[:, v]), list(rows[:, v]))]
                draw_sampled_border(sample_img.copy(), coords, None, debug=self.debug)

        grid = sample_img[rows, cols].astype(float)
        return grid.mean()
        # flat = np.array([])
        # for v in range(v_len):
        #     signal = np.array([
        #         int(sample_img[
        #                 max(0, min(int(round((origin + u * u_hat + v * v_hat)[1])), sample_img.shape[0] - 1)),
        #                 max(0, min(int(round((origin + u * u_hat + v * v_hat)[0])), sample_img.shape[1] - 1))
        #             ])
        #         for u in range(u_len)
        #     ])
        #
        #     self.debug.log(f"[border-fitter] signal: {signal}")
        #
        #     flat = np.concatenate((flat, signal))
        #
        #     scan_coords = []
        #     for u in range(u_len):
        #         coords = origin + u * u_hat + v * v_hat
        #         row = int(round(coords[1]))
        #         col = int(round(coords[0]))
        #
        #         row = max(0, min(row, sample_img.shape[0] - 1))
        #         col = max(0, min(col, sample_img.shape[1] - 1))
        #         scan_coords.append((col, row))
        #
        #     draw_sampled_border(sample_img.copy(), scan_coords, None, debug=self.debug)

        # return flat
