import cv2 as cv
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from dm_detector.location.l_finder_detector import LPattern
from dm_detector.location.dashed_border_detector import DataMatrixLocation

@dataclass
class PreciseLocation:
    vertices: List[Tuple[float, float]]
    center: Tuple[float, float]
    angle: float
    size: Tuple[float, float]

    def get_ordered_vertices(self) -> List[Tuple[int, int]]:
        return [(int(v[0]), int(v[1])) for v in self.vertices]

class BorderFitter:

    def __init__(self):
        pass

    def fit(self, gray_img: np.ndarray, l_pattern: LPattern, rough_location: Optional[DataMatrixLocation] = None) -> \
    Optional[PreciseLocation]:
        if rough_location is None:
            return self._fit_from_l_pattern(l_pattern)

        edges = cv.Canny(gray_img, 50, 150)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        edges = cv.dilate(edges, kernel, iterations=1)
        edges = cv.erode(edges, kernel, iterations=1)

        contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if not contours:
            return self._fit_from_l_pattern(l_pattern)

        corner = np.array(l_pattern.corner)

        rx, ry, rw, rh = rough_location.bounding_box
        rough_center = (rx + rw / 2, ry + rh / 2)

        best_contour = None
        best_score = -float('inf')

        for cnt in contours:
            if len(cnt) < 4:
                continue

            area = cv.contourArea(cnt)
            if area < 100:
                continue

            M = cv.moments(cnt)
            if M['m00'] == 0:
                continue

            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']

            dist_to_corner = np.sqrt((cx - corner[0]) ** 2 + (cy - corner[1]) ** 2)
            dist_to_rough = np.sqrt((cx - rough_center[0]) ** 2 + (cy - rough_center[1]) ** 2)

            if dist_to_rough > min(rw, rh) / 2:
                continue

            rect = cv.minAreaRect(cnt)
            w, h = rect[1]
            if w > 0 and h > 0:
                aspect = max(w, h) / min(w, h)
                if aspect > 3:
                    continue

            score = area / (dist_to_corner + 1)

            if score > best_score:
                best_score = score
                best_contour = cnt

        if best_contour is None:
            return self._fit_from_l_pattern(l_pattern)

        rect = cv.minAreaRect(best_contour)
        box = cv.boxPoints(rect)

        vertices = [(float(p[0]), float(p[1])) for p in box]

        return PreciseLocation(
            vertices=vertices,
            center=(rect[0][0], rect[0][1]),
            angle=rect[2],
            size=(float(rect[0][1]), float(rect[1][1]))
        )

    @staticmethod
    def _fit_from_l_pattern(l_pattern: LPattern) -> PreciseLocation:
        corner = np.array(l_pattern.corner)
        v1 = np.array(l_pattern.vertex1)
        v2 = np.array(l_pattern.vertex2)

        dir_h = v1 - corner
        dir_v = v2 - corner

        side = max(l_pattern.len1, l_pattern.len2)
        dir_h = dir_h / (np.linalg.norm(dir_h) + 1e-6) * side
        dir_v = dir_v / (np.linalg.norm(dir_v) + 1e-6) * side

        p1 = corner
        p2 = corner + dir_h
        p3 = corner + dir_h + dir_v
        p4 = corner + dir_v

        vertices = [
            (float(p1[0]), float(p1[1])),
            (float(p2[0]), float(p2[1])),
            (float(p3[0]), float(p3[1])),
            (float(p4[0]), float(p4[1]))
        ]

        center = np.mean([p1, p2, p3, p4], axis=0)

        return PreciseLocation(
            vertices=vertices,
            center=(float(center[0]), float(center[1])),
            angle=0,
            size=(side, side)
        )

    @staticmethod
    def draw_precise_location(image: np.ndarray,
                              location: PreciseLocation,
                              color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        result = image.copy()
        vertices = location.get_ordered_vertices()

        pts = np.array(vertices, dtype=np.int32).reshape((-1, 1, 2))
        cv.polylines(result, [pts], True, color, 2)

        return result