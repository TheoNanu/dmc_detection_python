import cv2 as cv
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from dm_detector.extraction.candidate_extraction import CandidateExtraction
from dm_detector.location.l_finder_detector import LFinderDetector, LPattern
from dm_detector.location.validator import DataMatrixValidator
from dm_detector.location.dashed_border_detector import DashedBorderDetector
from dm_detector.geometry.border_fitter import BorderFitter, PreciseLocation

@dataclass
class DetectionResult:
    candidate_box: Tuple[int, int, int, int]
    precise_location: Optional[PreciseLocation]
    l_patterns: List[LPattern]
    is_valid: bool
    score: float

    def get_rectified_image(self, full_frame: np.ndarray, output_size: int = 400) -> Optional[np.ndarray]:
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
        src_v1 = np.array([l_pat.vertex1[0] + cx, l_pat.vertex1[1] + cy])
        src_v2 = np.array([l_pat.vertex2[0] + cx, l_pat.vertex2[1] + cy])
        src_v3 = src_v1 + src_v2 - src_corner

        ordered_src = np.array([
            src_corner,
            src_v1,
            src_v3,
            src_v2
        ], dtype=np.float32)

        M = cv.getPerspectiveTransform(ordered_src, dst_pts)
        return cv.warpPerspective(full_frame, M, (output_size, output_size))

class DataMatrixPipeline:

    def __init__(self,
                 canny_t1: int = 50,
                 canny_t2: int = 150,
                 min_area: float = 400.0,
                 min_perimeter: float = 80.0,
                 padding: int = 25):

        self.extractor = CandidateExtraction(
            canny_t1=canny_t1,
            canny_t2=canny_t2,
            min_area=min_area,
            min_perimeter=min_perimeter,
            padding=padding
        )

        self.l_finder = LFinderDetector()
        self.validator = DataMatrixValidator()
        self.border_fitter = BorderFitter()
        self.dashed_detector = DashedBorderDetector()

    def process_frame(self, frame: np.ndarray) -> List[DetectionResult]:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        candidates = self.extractor.get_candidates(frame)
        results = []

        for (x, y, w, h) in candidates:
            region = gray[y:y + h, x:x + w]

            segments = self.l_finder.detect_lines(region)
            l_patterns = self.l_finder.find_l_patterns(cv.cvtColor(region, cv.COLOR_GRAY2BGR), segments)

            precise_location = None
            is_valid = False
            score = 0.0

            if len(l_patterns) > 0:
                l_pattern = l_patterns[0]
                validation = self.validator.validate(region, l_pattern)

                l_pattern_frame = self.draw_l_pattern(cv.cvtColor(region, cv.COLOR_GRAY2BGR), l_pattern)
                cv.imshow("detected", l_pattern_frame)
                cv.waitKey(0)

                if validation.is_valid:
                    dashed_result = self.dashed_detector.detect(region, l_pattern)
                    precise_location = self.border_fitter.fit(region, l_pattern, rough_location=dashed_result)

                    l_pattern_frame = self.draw_l_pattern(cv.cvtColor(region, cv.COLOR_GRAY2BGR), l_pattern)
                    cv.imshow("detected", l_pattern_frame)
                    cv.waitKey(0)

                    if precise_location is None and dashed_result is not None:
                        bx, by, bw, bh = dashed_result.bounding_box
                        vertices = [
                            (float(bx), float(by)),
                            (float(bx + bw), float(by)),
                            (float(bx + bw), float(by + bh)),
                            (float(bx), float(by + bh))
                        ]
                        center = (float(bx + bw / 2), float(by + bh / 2))

                        precise_location = PreciseLocation(
                            vertices=vertices,
                            center=center,
                            angle=0.0,
                            size=(float(bw), float(bh))
                        )

                    if precise_location:
                        global_vertices = [(vx + x, vy + y) for vx, vy in precise_location.vertices]
                        precise_location.vertices = global_vertices
                        precise_location.center = (precise_location.center[0] + x, precise_location.center[1] + y)

                    is_valid = True
                    score = validation.score

            results.append(DetectionResult(
                candidate_box=(x, y, w, h),
                precise_location=precise_location,
                l_patterns=l_patterns,
                is_valid=is_valid,
                score=score
            ))

            # region = frame[y:y + h, x:x + w]

            # cv.imshow("region", region)
            # cv.waitKey(0)

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    @staticmethod
    def draw_results(frame: np.ndarray, results: List[DetectionResult],
                     debug_view: bool = False) -> np.ndarray:
        output = frame.copy()

        for result in results:
            x, y, w, h = result.candidate_box

            if result.precise_location and result.is_valid:
                vertices = result.precise_location.get_ordered_vertices()
                pts = np.array(vertices, dtype=np.int32)
                cv.polylines(output, [pts], True, (0, 255, 0), 2)

                if debug_view:
                    cx, cy = int(result.precise_location.center[0]), int(result.precise_location.center[1])
                    cv.circle(output, (cx, cy), 3, (255, 0, 0), -1)

            elif debug_view:
                cv.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 1)

        return output

    @staticmethod
    def draw_l_pattern(frame: np.ndarray, l_pattern: LPattern, color: tuple=(0, 255, 0)):
        result = frame.copy()

        cv.line(result, (int(l_pattern.vertex1[0]), int(l_pattern.vertex1[1])),
                (int(l_pattern.corner[0]), int(l_pattern.corner[1])), color, 3)
        cv.line(result, (int(l_pattern.vertex2[0]), int(l_pattern.vertex2[1])),
                (int(l_pattern.corner[0]), int(l_pattern.corner[1])), color, 3)

        return result