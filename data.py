from dataclasses import dataclass
from typing import Tuple, Optional, List

import cv2 as cv
import numpy as np

@dataclass
class ValidationResult:
    is_valid: bool
    edge_density: float
    aspect_ratio: float
    score: float
    reason: str = ""

@dataclass
class PreciseLocation:
    vertices: List[Tuple[float, float]]
    center: Tuple[float, float]
    angle: float
    size: Tuple[float, float]

    def ordered_vertices(self) -> List[Tuple[int, int]]:
        return [(int(v[0]), int(v[1])) for v in self.vertices]


@dataclass
class LPattern:
    vertex1: Tuple[float, float]
    corner: Tuple[float, float]
    vertex2: Tuple[float, float]
    len1: float
    len2: float
    score: float = 0.0

    def bounding_box(self, padding: int = 0) -> Tuple[int, int, int, int]:
        xs = [self.vertex1[0], self.corner[0], self.vertex2[0]]
        ys = [self.vertex1[1], self.corner[1], self.vertex2[1]]
        fourth_corner_x = self.vertex1[0] + self.vertex2[0] - self.corner[0]
        fourth_corner_y = self.vertex1[1] + self.vertex2[1] - self.corner[1]
        pts = np.array([self.vertex1, self.vertex2, self.corner, (fourth_corner_x, fourth_corner_y)], dtype=np.float32)
        x, y, w, h = cv.boundingRect(pts.astype(np.int32))
        if padding != 0:
            x = x - padding
            y = y - padding
            w = w + padding
            h = h + padding
        x_min, x_max = int(min(xs)), int(max(xs))
        y_min, y_max = int(min(ys)), int(max(ys))
        return x, y, w, h


@dataclass
class DetectionResult:
    candidate_box: Tuple[int, int, int, int]
    precise_location: Optional[PreciseLocation]
    l_patterns: List[LPattern]
    is_valid: bool
    score: float

    def rectify(self, full_frame: np.ndarray, output_size: int = 400) -> Optional[np.ndarray]:
        if not self.precise_location or not self.l_patterns:
            return None

        l_pat = self.l_patterns[0]

        dst_pts = np.array([
            [0, output_size - 1],
            [output_size - 1, output_size - 1],
            [output_size - 1, 0],
            [0, 0]
        ], dtype=np.float32)

        cx, cy, _, _ = self.candidate_box

        src_corner = np.array([l_pat.corner[0] + cx, l_pat.corner[1] + cy])

        arm1 = np.array(l_pat.vertex1) - np.array(l_pat.corner)
        arm2 = np.array(l_pat.vertex2) - np.array(l_pat.corner)
        if abs(arm1[0]) > abs(arm1[1]):
            src_horiz = np.array([l_pat.vertex1[0] + cx, l_pat.vertex1[1] + cy])
            src_vert = np.array([l_pat.vertex2[0] + cx, l_pat.vertex2[1] + cy])
        else:
            src_horiz = np.array([l_pat.vertex2[0] + cx, l_pat.vertex2[1] + cy])
            src_vert = np.array([l_pat.vertex1[0] + cx, l_pat.vertex1[1] + cy])

        vertices = [np.asarray(v, dtype=np.float32) for v in self.precise_location.vertices]
        remaining = list(range(len(vertices)))

        # dst index mapping: corner → 0 (BL), horiz → 1 (BR), vert → 3 (TL), leftover → 2 (TR)
        assignments = [(src_corner, 0), (src_horiz, 1), (src_vert, 3)]

        ordered_src = [None] * 4
        for l_pt, dst_idx in assignments:
            best_i = min(remaining, key=lambda i: np.linalg.norm(vertices[i] - l_pt))
            ordered_src[dst_idx] = vertices[best_i]
            remaining.remove(best_i)

        ordered_src[2] = vertices[remaining[0]]  # leftover → TR
        ordered_src = np.array(ordered_src, dtype=np.float32)

        M = cv.getPerspectiveTransform(ordered_src, dst_pts)
        return cv.warpPerspective(full_frame, M, (output_size, output_size))


@dataclass
class Decoded:
    detection: DetectionResult
    text: str
    codewords: list[str]
    matrix: np.ndarray