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

    @staticmethod
    def parent_visited(visited: list, current: tuple):
        for v in visited:
            if current[0] >= v[0] and current[1] >= v[1] and current[2] <= v[2] and current[3] <= v[3]:
                return True

        return False

    def process_frame(self, frame: np.ndarray) -> List[DetectionResult]:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        candidates = self.extractor.get_candidates(frame)
        results = []

        visited_candidates = []

        # sort candidates from the highest area to the lowest area to take advantage of the deduplication strategy
        candidates.sort(reverse=True, key=lambda c: c[2] * c[3])

        for (x, y, w, h) in candidates:
            region = np.ascontiguousarray(gray[y:y + h, x:x + w])

            cv.imshow("region", region)
            cv.waitKey(0)

            if self.parent_visited(visited_candidates, (x, y, x + w, y + h)):
                # print(f"[pipeline] region ({x},{y},{w},{h}) skipped: parent already visited")
                continue
            visited_candidates.append((x, y, x + w, y + h))
            # print(f"[pipeline] processing region ({x},{y},{w},{h})")

            segments = self.l_finder.detect_lines(region)
            l_patterns = self.l_finder.find_l_patterns(cv.cvtColor(region, cv.COLOR_GRAY2BGR), region, segments)

            # print(f"[pipeline] l_patterns found: {len(l_patterns)}")

            region_has_valid = False

            for idx, l_pattern in enumerate(l_patterns):
                validation = self.validator.validate(region, l_pattern)

                label = f"#{idx} VALIDATOR {'ACCEPT' if validation.is_valid else 'REJECT'}: {validation.reason}"
                self.l_finder._show_pattern(
                    cv.cvtColor(region, cv.COLOR_GRAY2BGR),
                    l_pattern, label, validation.is_valid, window="detected"
                )

                if not validation.is_valid:
                    continue

                dashed_result, edges = self.dashed_detector.detect(region, l_pattern)
                dashed_label = f"#{idx} DASHED {'ACCEPT' if dashed_result is not None else 'REJECT'}"
                self.l_finder._show_pattern(
                    cv.cvtColor(region, cv.COLOR_GRAY2BGR),
                    l_pattern, dashed_label, dashed_result is not None, window="detected"
                )

                if dashed_result is None:
                    continue

                precise_location = self.border_fitter.fit(region, edges, l_pattern, rough_location=dashed_result)

                if precise_location is None:
                    print(f"[pipeline] precise location not found")
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
                else:
                    print(f"[pipeline] precise location found: {precise_location.vertices}")
                    cv.drawContours(region, [np.int32(precise_location.vertices)], 0, (0, 0, 255), 2)
                    cv.imshow("precise location", region)
                    cv.waitKey(0)

                global_vertices = [(vx + x, vy + y) for vx, vy in precise_location.vertices]
                precise_location.vertices = global_vertices
                precise_location.center = (precise_location.center[0] + x, precise_location.center[1] + y)

                results.append(DetectionResult(
                    candidate_box=(x, y, w, h),
                    precise_location=precise_location,
                    l_patterns=[l_pattern],
                    is_valid=True,
                    score=validation.score
                ))
                region_has_valid = True

            # if not region_has_valid:
            #     results.append(DetectionResult(
            #         candidate_box=(x, y, w, h),
            #         precise_location=None,
            #         l_patterns=l_patterns,
            #         is_valid=False,
            #         score=0.0
            #     ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    @staticmethod
    def draw_results(frame: np.ndarray, results: List[DetectionResult],
                     debug_view: bool = False) -> np.ndarray:
        output = frame.copy()

        for result in results:
            x, y, w, h = result.candidate_box

            if result.precise_location and result.is_valid:
                print("DMC DETECTED")
                vertices = result.precise_location.get_ordered_vertices()
                pts = np.array(vertices, dtype=np.int32)
                cv.polylines(output, [pts], True, (0, 255, 0), 2)

                if debug_view:
                    print("DMC DETECTED, DRAWING CANDIDATE")
                    cx, cy = int(result.precise_location.center[0]), int(result.precise_location.center[1])
                    cv.circle(output, (cx, cy), 10, (255, 0, 0), -1)

            elif debug_view:
                print("DMC NOT DETECTED BUT DRAWING CANDIDATE")
                # cv.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 1)

        return output

    @staticmethod
    def draw_l_pattern(frame: np.ndarray, l_pattern: LPattern, color: tuple=(0, 255, 0)):
        result = frame.copy()

        cv.line(result, (int(l_pattern.vertex1[0]), int(l_pattern.vertex1[1])),
                (int(l_pattern.corner[0]), int(l_pattern.corner[1])), color, 3)
        cv.line(result, (int(l_pattern.vertex2[0]), int(l_pattern.vertex2[1])),
                (int(l_pattern.corner[0]), int(l_pattern.corner[1])), color, 3)

        return result