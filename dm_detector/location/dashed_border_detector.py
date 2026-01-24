import cv2 as cv
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

from .l_finder_detector import LPattern

@dataclass
class DataMatrixLocation:
    l_pattern: LPattern
    upper_border: Tuple[int, int, int, int]
    right_border: Tuple[int, int, int, int]
    bounding_box: Tuple[int, int, int, int]

class DashedBorderDetector:

    def __init__(self, tau: int = 5, edge_threshold: int = 50):
        self.tau = tau
        self.edge_threshold = edge_threshold

    def get_detection_regions(self, l_pattern: LPattern,
                              img_shape: Tuple[int, int]) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
        x1, y1 = l_pattern.vertex1
        x3, y3 = l_pattern.vertex2
        len1, len2 = l_pattern.len1, l_pattern.len2
        tau = self.tau
        img_h, img_w = img_shape

        upper_x = max(0, int(x1 - tau))
        upper_y = max(0, int(y1 - tau - (len1 - len2)))
        upper_w = min(img_w - upper_x, int(len1 + 2 * tau))
        upper_h = min(img_h - upper_y, int(len1 - len2 + 2 * tau))

        right_x = min(img_w - 1, int(x3 + tau))
        right_y = max(0, int(y3 - tau))
        right_w = min(img_w - right_x, int(len1 - len2 + 2 * tau))
        right_h = min(img_h - right_y, int(len1 + 2 * tau))

        upper_w = max(1, upper_w)
        upper_h = max(1, upper_h)
        right_w = max(1, right_w)
        right_h = max(1, right_h)

        return (upper_x, upper_y, upper_w, upper_h), (right_x, right_y, right_w, right_h)

    @staticmethod
    def scan_edge_points(edge_img: np.ndarray,
                         region: Tuple[int, int, int, int],
                         direction: str = 'horizontal') -> Tuple[int, int]:
        x, y, w, h = region

        if x < 0 or y < 0 or x + w > edge_img.shape[1] or y + h > edge_img.shape[0]:
            return 0, 0

        roi = edge_img[y:y+h, x:x+w]

        if direction == 'horizontal':
            edge_counts = np.sum(roi > 0, axis=1)
        else:
            edge_counts = np.sum(roi > 0, axis=0)

        if len(edge_counts) == 0:
            return 0, 0

        dashed_idx = int(np.argmax(edge_counts))
        solid_idx = int(np.argmin(edge_counts))

        return dashed_idx, solid_idx

    @staticmethod
    def _auto_canny(image: np.ndarray, sigma: float = 0.33) -> np.ndarray:
        v = np.median(image)
        lower = max(0, int((1.0 - sigma) * v))
        upper = min(255, int((1.0 + sigma) * v))
        if lower < 50: lower = 50
        if upper < 100: upper = 100
        return cv.Canny(image, lower, upper)

    def detect(self, gray_img: np.ndarray, l_pattern: LPattern) -> Optional[DataMatrixLocation]:
        edges = self._auto_canny(gray_img)

        upper_region, right_region = self.get_detection_regions(l_pattern, gray_img.shape)

        upper_dashed_row, _ = self.scan_edge_points(edges, upper_region, 'horizontal')
        right_dashed_col, _ = self.scan_edge_points(edges, right_region, 'vertical')

        x1, y1 = l_pattern.vertex1
        x2, y2 = l_pattern.corner
        x3, y3 = l_pattern.vertex2

        upper_border_y = upper_region[1] + upper_dashed_row
        right_border_x = right_region[0] + right_dashed_col

        min_x = min(int(x1), int(x2), int(x3))
        min_y = min(int(upper_border_y), int(y1), int(y2), int(y3))
        max_x = max(int(right_border_x), int(x1), int(x2), int(x3))
        max_y = max(int(y1), int(y2), int(y3))

        bounding_box = (min_x, min_y, max_x - min_x, max_y - min_y)

        return DataMatrixLocation(
            l_pattern=l_pattern,
            upper_border=upper_region,
            right_border=right_region,
            bounding_box=bounding_box
        )

    @staticmethod
    def draw_detection_regions(image: np.ndarray,
                               upper_region: Tuple[int, int, int, int],
                               right_region: Tuple[int, int, int, int]) -> np.ndarray:
        result = image.copy()

        x, y, w, h = upper_region
        cv.rectangle(result, (x, y), (x + w, y + h), (255, 255, 0), 1)

        x, y, w, h = right_region
        cv.rectangle(result, (x, y), (x + w, y + h), (0, 255, 255), 1)

        return result

    @staticmethod
    def draw_location(image: np.ndarray,
                      location: DataMatrixLocation,
                      color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        result = image.copy()
        x, y, w, h = location.bounding_box
        cv.rectangle(result, (x, y), (x + w, y + h), color, 2)
        return result