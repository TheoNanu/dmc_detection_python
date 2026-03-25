import cv2 as cv
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class LineSegment:
    p1: Tuple[float, float]
    p2: Tuple[float, float]
    marked: bool = False

    @property
    def length(self) -> float:
        return np.sqrt((self.p2[0] - self.p1[0])**2 + (self.p2[1] - self.p1[1])**2)

    @property
    def angle(self) -> float:
        return np.arctan2(self.p2[1] - self.p1[1], self.p2[0] - self.p1[0])

@dataclass
class LPattern:
    vertex1: Tuple[float, float]
    corner: Tuple[float, float]
    vertex2: Tuple[float, float]
    len1: float
    len2: float
    score: float = 0.0

    def get_bounding_box(self, padding: int = 0) -> Tuple[int, int, int, int]:
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

class LFinderDetector:
    def __init__(self,
                 neighborhood_radius: float = 10.0,
                 min_angle: float = 60.0,
                 max_angle: float = 120.0,
                 max_length_ratio: float = 5.0,
                 min_segment_length: float = 20.0):
        self.neighborhood_radius = neighborhood_radius
        self.min_angle = np.radians(min_angle)
        self.max_angle = np.radians(max_angle)
        self.max_length_ratio = max_length_ratio
        self.min_segment_length = min_segment_length
        self.lsd = cv.createLineSegmentDetector(cv.LSD_REFINE_NONE)

    def detect_lines(self, region: np.ndarray) -> List[LineSegment]:
        if len(region.shape) == 3:
            region = cv.cvtColor(region, cv.COLOR_BGR2GRAY)

        blurred = cv.GaussianBlur(region, (3, 3), 0)

        lines, _, _, _ = self.lsd.detect(blurred)

        # Use the larger region dimension as the upper length bound so that
        # full-width L-arm segments are never filtered out.
        max_len = max(region.shape[0], region.shape[1])

        segments = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                segment = LineSegment(
                    p1=(float(x1), float(y1)),
                    p2=(float(x2), float(y2))
                )
                if self.min_segment_length <= segment.length <= max_len:
                    segments.append(segment)

        return segments

    @staticmethod
    def _distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

    @staticmethod
    def _angle_between_segments(seg1: LineSegment, seg2: LineSegment) -> float:
        angle1 = seg1.angle
        angle2 = seg2.angle

        diff = abs(angle1 - angle2)
        if diff > np.pi:
            diff = 2 * np.pi - diff

        return diff

    def _find_connection_point(self, seg1: LineSegment, seg2: LineSegment) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], float]]:
        endpoints = [
            (seg1.p1, seg1.p2, seg2.p1, seg2.p2),
            (seg1.p1, seg1.p2, seg2.p2, seg2.p1),
            (seg1.p2, seg1.p1, seg2.p1, seg2.p2),
            (seg1.p2, seg1.p1, seg2.p2, seg2.p1),
        ]

        best_match = None
        min_dist = float('inf')

        for s1_corner, s1_end, s2_corner, s2_end in endpoints:
            dist = self._distance(s1_corner, s2_corner)
            if dist < self.neighborhood_radius and dist < min_dist:
                min_dist = dist
                corner = ((s1_corner[0] + s2_corner[0]) / 2,
                         (s1_corner[1] + s2_corner[1]) / 2)
                best_match = (s1_end, corner, s2_end, dist)

        return best_match

    def _calculate_score(self, angle: float, length_ratio: float, connection_dist: float) -> float:
        angle_deg = np.degrees(angle)
        angle_score = 1.0 - abs(angle_deg - 90.0) / 30.0
        ratio_score = 1.0 - (length_ratio - 1.0) / 4.0
        dist_score = 1.0 - connection_dist / self.neighborhood_radius
        return max(0, angle_score * 0.4 + ratio_score * 0.3 + dist_score * 0.3)

    @staticmethod
    def _interior_is_high_frequency(gray: np.ndarray, pattern: LPattern,
                                    min_eigenvalue: float = 100.0,
                                    max_isotropy_ratio: float = 10.0) -> bool:
        img_h, img_w = gray.shape[:2]
        lx, ly, lw, lh = pattern.get_bounding_box(padding=5)
        x1, y1 = max(0, lx), max(0, ly)
        x2, y2 = min(img_w, lx + lw), min(img_h, ly + lh)

        if x2 - x1 < 8 or y2 - y1 < 8:
            return False

        roi = gray[y1:y2, x1:x2]

        cv.imshow("l pattern freq test", roi)
        cv.resizeWindow("l pattern freq test", 640, 480)

        roi = gray[y1:y2, x1:x2].astype(np.float32)

        gx = cv.Sobel(roi, cv.CV_32F, 1, 0, ksize=3)
        gy = cv.Sobel(roi, cv.CV_32F, 0, 1, ksize=3)

        cov_xx = float(np.mean(gx * gx))
        cov_yy = float(np.mean(gy * gy))
        cov_xy = float(np.mean(gx * gy))

        # Eigenvalues of the 2x2 covariance matrix
        trace = cov_xx + cov_yy
        det = cov_xx * cov_yy - cov_xy * cov_xy
        disc = max(0.0, (trace / 2) ** 2 - det)
        l1 = trace / 2 + np.sqrt(disc)
        l2 = trace / 2 - np.sqrt(disc)

        print(f"Structure tensor eigenvalues: λ1={l1:.1f}, λ2={l2:.1f}")

        if l2 < min_eigenvalue:
            print(f"  rejected: λ2 too small (not high-frequency enough)")
            return False

        if l1 > max_isotropy_ratio * l2:
            print(f"  rejected: too anisotropic (single edge, not DMC interior)")
            return False

        return True

    def find_l_patterns(self, frame: np.ndarray, gray: np.ndarray, segments: List[LineSegment]) -> List[LPattern]:
        l_patterns = []

        for i, seg_i in enumerate(segments):
            if seg_i.marked:
                continue

            best_pattern = None
            best_score = 0.0
            best_j = -1

            for j, seg_j in enumerate(segments):
                if i >= j or seg_j.marked:
                    continue

                angle = self._angle_between_segments(seg_i, seg_j)
                # print(f"Found angle: {np.rad2deg(angle)} Min: {np.rad2deg(self.min_angle)} Max: {np.rad2deg(self.max_angle)}")
                if not (self.min_angle <= angle <= self.max_angle):
                    continue

                len_i, len_j = seg_i.length, seg_j.length
                ratio = max(len_i, len_j) / min(len_i, len_j)
                # print(f"Found ratio: {ratio} Max: {self.max_length_ratio}")
                if ratio > self.max_length_ratio:
                    continue

                connection = self._find_connection_point(seg_i, seg_j)
                # print(f"Found connection: {connection}")
                if connection is None:
                    continue

                result = self.draw_segments(frame, [seg_i, seg_j])

                vertex1, corner, vertex2, conn_dist = connection
                score = self._calculate_score(angle, ratio, conn_dist)

                # print(f"Score: {score}")

                if score > best_score:
                    # print(f"Best score found: {score} > {best_score}")
                    best_score = score
                    best_j = j
                    best_pattern = LPattern(
                        vertex1=vertex1,
                        corner=corner,
                        vertex2=vertex2,
                        len1=max(len_i, len_j),
                        len2=min(len_i, len_j),
                        score=score
                    )
                else:
                    pass
                    # print(f"Not best score: {score} < {best_score}")

                cv.imshow("result", result)
                cv.waitKey(0)

            if best_pattern and best_score > 0.5:
                if not self._interior_is_high_frequency(gray, best_pattern, min_eigenvalue=1000, max_isotropy_ratio=1.7):
                    continue
                l_patterns.append(best_pattern)
                seg_i.marked = True
                if best_j >= 0:
                    segments[best_j].marked = True

        l_patterns.sort(key=lambda p: p.score, reverse=True)
        return l_patterns

    def detect(self, region: np.ndarray) -> List[LPattern]:
        segments = self.detect_lines(region)
        return self.find_l_patterns(region, segments)

    @staticmethod
    def draw_segments(image: np.ndarray, segments: List[LineSegment],
                      color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        result = image.copy()
        for seg in segments:
            pt1 = (int(seg.p1[0]), int(seg.p1[1]))
            pt2 = (int(seg.p2[0]), int(seg.p2[1]))
            cv.line(result, pt1, pt2, color, 1)
        return result

    @staticmethod
    def draw_l_patterns(image: np.ndarray, patterns: List[LPattern],
                        color: Tuple[int, int, int] = (255, 0, 255)) -> np.ndarray:
        result = image.copy()
        for pattern in patterns:
            v1 = (int(pattern.vertex1[0]), int(pattern.vertex1[1]))
            corner = (int(pattern.corner[0]), int(pattern.corner[1]))
            v2 = (int(pattern.vertex2[0]), int(pattern.vertex2[1]))

            cv.line(result, v1, corner, color, 2)
            cv.line(result, corner, v2, color, 2)
            cv.circle(result, corner, 4, (0, 0, 255), -1)
            cv.circle(result, v1, 3, (255, 255, 0), -1)
            cv.circle(result, v2, 3, (255, 255, 0), -1)

        return result