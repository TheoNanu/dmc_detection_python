import cv2 as cv
import numpy as np

class ModuleSampler:
    @staticmethod
    def draw_grid(image: np.ndarray, horizontal_pitch: float, vertical_pitch: float) -> None:
        h, w = image.shape[:2]

        y = 0.0
        while y <= h:
            cv.line(image, (0, int(y)), (w, int(y)), (0, 255, 0), 1)
            y += vertical_pitch

        x = 0.0
        while x <= w:
            cv.line(image, (int(x), 0), (int(x), h), (0, 255, 0), 1)
            x += horizontal_pitch

    @staticmethod
    def get_matrix_data(image: np.ndarray, horizontal_pitch: float, vertical_pitch: float, rows: int, cols: int) -> np.ndarray:
        result = np.zeros(shape=(rows - 2, cols - 2), dtype=np.uint8)

        for y in range(1, rows - 1):
            for x in range(1, cols - 1):
                module = image[int(y * vertical_pitch):int(y * vertical_pitch + vertical_pitch),
                                int(x * horizontal_pitch):int(x * horizontal_pitch + horizontal_pitch)]

                if module.size == 0:
                    continue

                median = np.median(module)
                if median > 150:
                    result[y - 1][x - 1] = 0
                else:
                    result[y - 1][x - 1] = 1

        return result