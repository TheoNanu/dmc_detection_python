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

    def get_bounding_box(self) -> Tuple[int, int, int, int]:
        xs = [self.vertex1[0], self.corner[0], self.vertex2[0]]
        ys = [self.vertex1[1], self.corner[1], self.vertex2[1]]
        x_min, x_max = int(min(xs)), int(max(xs))
        y_min, y_max = int(min(ys)), int(max(ys))
        return x_min, y_min, x_max - x_min, y_max - y_min

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
    def _count_line_transitions(line: np.ndarray, min_amplitude: float = 20.0) -> int:
        if line.size < 2:
            return 0
        lo, hi = float(np.min(line)), float(np.max(line))
        if hi - lo < min_amplitude:
            return 0
        threshold = (lo + hi) / 2.0
        binary = (line > threshold).astype(np.int8)
        return int(np.sum(np.abs(np.diff(binary))))

    @staticmethod
    def _interior_is_high_frequency(gray: np.ndarray, pattern: LPattern,
                                    min_eigenvalue: float = 100.0,
                                    max_isotropy_ratio: float = 10.0,
                                    min_transitions_per_line: int = 4,
                                    num_scan_lines: int = 5) -> bool:
        img_h, img_w = gray.shape[:2]
        lx, ly, lw, lh = pattern.get_bounding_box()
        x1, y1 = max(0, lx), max(0, ly)
        x2, y2 = min(img_w, lx + lw), min(img_h, ly + lh)

        if x2 - x1 < 8 or y2 - y1 < 8:
            return False

        roi = gray[y1:y2, x1:x2]
        roi_f = roi.astype(np.float32)

        gx = cv.Sobel(roi_f, cv.CV_32F, 1, 0, ksize=3)
        gy = cv.Sobel(roi_f, cv.CV_32F, 0, 1, ksize=3)

        cov_xx = float(np.mean(gx * gx))
        cov_yy = float(np.mean(gy * gy))
        cov_xy = float(np.mean(gx * gy))

        trace = cov_xx + cov_yy
        det = cov_xx * cov_yy - cov_xy * cov_xy
        disc = max(0.0, (trace / 2) ** 2 - det)
        l1 = trace / 2 + np.sqrt(disc)
        l2 = trace / 2 - np.sqrt(disc)

        if l2 < min_eigenvalue:
            print(f"[structure tensor] λ1={l1:.1f}, λ2={l2:.1f} -- λ2 below min {min_eigenvalue}")
            return False

        if l1 > max_isotropy_ratio * l2:
            print(f"[structure tensor] λ1={l1:.1f}, λ2={l2:.1f} -- ratio {l1/max(l2,1e-6):.1f} > {max_isotropy_ratio} (too anisotropic)")
            return False

        # Scale-invariant check: a DMC has multiple intensity transitions along
        # any scan line through its interior; a single solid module has at most
        # two (entering and leaving the module).
        rh, rw = roi.shape[:2]
        h_trans = []
        v_trans = []
        for k in range(1, num_scan_lines + 1):
            row = int(rh * k / (num_scan_lines + 1))
            col = int(rw * k / (num_scan_lines + 1))
            h_trans.append(LFinderDetector._count_line_transitions(roi[row, :]))
            v_trans.append(LFinderDetector._count_line_transitions(roi[:, col]))

        min_h = min(h_trans)
        min_v = min(v_trans)
        median_h = int(np.median(h_trans))
        median_v = int(np.median(v_trans))

        if median_h < min_transitions_per_line or median_v < min_transitions_per_line:
            print(f"[transitions] h={h_trans} v={v_trans} -- median(h)={median_h} median(v)={median_v} below min {min_transitions_per_line}")
            return False

        print(f"[structure tensor] λ1={l1:.1f}, λ2={l2:.1f} -- accepted")
        print(f"[transitions] h_median={median_h} v_median={median_v} (min_h={min_h} min_v={min_v}) -- accepted")
        return True

    def find_l_patterns(self, frame: np.ndarray, gray: np.ndarray, segments: List[LineSegment]) -> List[LPattern]:
        l_patterns = []

        seg_vis = self.draw_segments(frame, segments, color=(0, 255, 255))
        # print(f"[L-finder] LSD detected {len(segments)} segments")
        # cv.imshow("lsd segments", seg_vis)
        # cv.waitKey(0)

        for i, seg_i in enumerate(segments):
            if seg_i.marked:
                continue

            best_pattern = None
            best_score = 0.0
            best_j = -1

            pairs_total = 0
            fail_angle = 0
            fail_ratio = 0
            fail_connection = 0
            min_conn_dist = float('inf')

            for j, seg_j in enumerate(segments):
                if i >= j or seg_j.marked:
                    continue
                pairs_total += 1

                angle = self._angle_between_segments(seg_i, seg_j)
                if not (self.min_angle <= angle <= self.max_angle):
                    fail_angle += 1
                    continue

                len_i, len_j = seg_i.length, seg_j.length
                ratio = max(len_i, len_j) / min(len_i, len_j)
                if ratio > self.max_length_ratio:
                    fail_ratio += 1
                    continue

                # Track the closest endpoint distance across ALL endpoint combos,
                # so we can see how far off the corner connection is when it fails.
                for s1_c in (seg_i.p1, seg_i.p2):
                    for s2_c in (seg_j.p1, seg_j.p2):
                        d = self._distance(s1_c, s2_c)
                        if d < min_conn_dist:
                            min_conn_dist = d

                connection = self._find_connection_point(seg_i, seg_j)
                if connection is None:
                    fail_connection += 1
                    continue

                vertex1, corner, vertex2, conn_dist = connection
                score = self._calculate_score(angle, ratio, conn_dist)

                if score > best_score:
                    # print(f"[L-finder] new best score found: {score}, previous best score: {best_score}")
                    best_score = score
                    pattern_img = frame.copy()
                    if best_pattern:
                        pattern_img = self.draw_l_patterns(pattern_img, [best_pattern], (0, 255, 255))
                    best_j = j
                    best_pattern = LPattern(
                        vertex1=vertex1,
                        corner=corner,
                        vertex2=vertex2,
                        len1=max(len_i, len_j),
                        len2=min(len_i, len_j),
                        score=score
                    )
                    pattern_img = self.draw_l_patterns(pattern_img, [best_pattern], (255, 0, 255))

                    # cv.imshow("pattern_img", pattern_img)
                    # cv.waitKey(0)

            if best_pattern is None and seg_i.length >= 50:
                print(f"[L-finder] anchor seg#{i} (len={seg_i.length:.0f}) "
                      f"pairs={pairs_total} fail_angle={fail_angle} fail_ratio={fail_ratio} "
                      f"fail_connection={fail_connection} "
                      f"min_endpoint_dist={min_conn_dist:.1f} "
                      f"(threshold {self.neighborhood_radius})")

            if best_pattern:
                if best_score <= 0.5:
                    reason = f"score {best_score:.2f} <= 0.5"
                    # print(f"[L-finder] candidate rejected: {reason}")
                    # self._show_pattern(frame, best_pattern, f"L-FINDER REJECT: {reason}", accepted=False)
                    continue
                if not self._interior_is_high_frequency(gray, best_pattern):
                    reason = "interior not high-frequency"
                    # print(f"[L-finder] candidate rejected: {reason}")
                    # self._show_pattern(frame, best_pattern, f"L-FINDER REJECT: {reason}", accepted=False)
                    continue
                # print(f"[L-finder] pattern accepted: score={best_score:.2f}")
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
    def _show_pattern(frame: np.ndarray, pattern: LPattern, label: str, accepted: bool,
                      window: str = "l pattern debug"):
        img = frame.copy()
        color = (0, 255, 0) if accepted else (0, 0, 255)
        cv.line(img,
                (int(pattern.vertex1[0]), int(pattern.vertex1[1])),
                (int(pattern.corner[0]), int(pattern.corner[1])), color, 3)
        cv.line(img,
                (int(pattern.vertex2[0]), int(pattern.vertex2[1])),
                (int(pattern.corner[0]), int(pattern.corner[1])), color, 3)
        cv.circle(img, (int(pattern.corner[0]), int(pattern.corner[1])), 5, (0, 0, 255), -1)
        lx, ly, lw, lh = pattern.get_bounding_box()
        cv.rectangle(img, (lx, ly), (lx + lw, ly + lh), color, 1)
        cv.putText(img, label, (5, 18), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv.LINE_AA)
        cv.imshow(window, img)
        cv.waitKey(0)

    @staticmethod
    def draw_segments(image: np.ndarray, segments: List[LineSegment],
                      color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        result = image.copy()
        for seg in segments:
            pt1 = (int(seg.p1[0]), int(seg.p1[1]))
            pt2 = (int(seg.p2[0]), int(seg.p2[1]))
            cv.line(result, pt1, pt2, color, 5)
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