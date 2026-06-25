from typing import Optional

import cv2 as cv
import numpy as np

from dmc.config import DecoderConfig
from dmc.data import Decoded
from dmc.debug import DebugSink, NullSink
from dmc.decoder.grid_estimation import GridEstimator
from dmc.detector import DetectionResult
from dmc.utils import valid_shape
from dmc.viz import draw_module_numbers, draw_module_grid


class Decoder:
    def __init__(self, config: DecoderConfig = DecoderConfig(), debug: DebugSink = NullSink()):
        self.config = config
        self.estimator = GridEstimator(margin=self.config.estimator_margin, debug=debug)
        self.debug = debug

    def decode(self, image: np.ndarray, detection: DetectionResult) -> Optional[Decoded]:
        if not isinstance(image, np.ndarray):
            raise ValueError("Input image should be a numpy array")

        if len(image.shape) != 3:
            raise ValueError("Input image should have 3 channels")

        warp = detection.rectify(image, output_size=self.config.output_size)

        if warp is None:
            return None

        grid_vis = warp.copy()

        gray = cv.medianBlur(cv.cvtColor(warp, cv.COLOR_BGR2GRAY), self.config.smoothing)

        self.debug.show("rectified", gray)
        self.debug.pause()

        h, w = gray.shape[:2]

        grid = self.estimator.estimate_grid(gray, inverted=detection.is_inverted)

        if grid is not None:
            col_centres, row_centres = grid
            bits = self.estimator.sample_matrix(gray, grid[0], grid[1], inverted=detection.is_inverted)
            matrix = bits[1:-1, 1:-1]

            draw_module_grid(grid_vis, col_centres, row_centres)
            grid_vis = draw_module_numbers(grid_vis, col_centres, row_centres)
            self.debug.show("final grid", grid_vis)
            self.debug.pause()
        else:
            pitch, score = self.estimator.estimate_pitch(gray)

            if pitch is not None:
                # width and height on which the estimator ran and found the pitch
                w_eff = w - 2 * self.estimator.margin
                h_eff = h - 2 * self.estimator.margin
                nx = int(round(w_eff / pitch))
                ny = int(round(h_eff / pitch))
                matrix = self.estimator.get_matrix_data(gray, w / nx, h / ny, ny, nx)
            else:
                matrix = None

        if matrix is None or not valid_shape(matrix.shape[0], matrix.shape[1], self.config.valid_sizes):
            return None

        self.debug.pause()

        codewords = self.estimator.ecc200_codewords_from_data_modules(matrix)

        data_codewords = []
        for cw in codewords:
            if cw == 129:  # padding, stop
                break
            if 1 <= cw <= 128:
                data_codewords.append(chr(cw - 1))
            elif 130 <= cw <= 229:
                data_codewords.append(f"{cw - 130:02d}")

        text = "".join(data_codewords)

        return Decoded(detection=detection, text=text, codewords=codewords, matrix=matrix)
