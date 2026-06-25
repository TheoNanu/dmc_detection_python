import cv2 as cv
import numpy as np
from typing import List

from dmc.config import DetectorConfig
from dmc.data import DetectionResult
from dmc.debug import DebugSink, NullSink
from dmc.detector.extraction.candidate_extraction import CandidateExtraction
from dmc.detector.location.l_finder_detector import LFinderDetector, LPattern
from dmc.detector.location.validator import DataMatrixValidator
from dmc.detector.location.dashed_border_detector import DashedBorderDetector
from dmc.detector.geometry.border_fitter import BorderFitter, PreciseLocation
from dmc.viz import show_pattern


class DetectionPipeline:

    def __init__(self, detector_config: DetectorConfig = DetectorConfig(), debug: DebugSink = NullSink()):

        self.debug = debug
        self.detector_config = detector_config

        self.extractor = CandidateExtraction(
            canny_t1=self.detector_config.extraction_config.canny_t1,
            canny_t2=self.detector_config.extraction_config.canny_t2,
            min_area=self.detector_config.extraction_config.min_area,
            min_perimeter=self.detector_config.extraction_config.min_perimeter,
            padding=self.detector_config.extraction_config.padding,
            debug=self.debug
        )

        self.l_finder = LFinderDetector(
            neighborhood_radius=self.detector_config.l_finder_config.neighborhood_radius,
            min_angle=self.detector_config.l_finder_config.min_angle,
            max_angle=self.detector_config.l_finder_config.max_angle,
            max_length_ratio=self.detector_config.l_finder_config.max_length_ratio,
            min_segment_length=self.detector_config.l_finder_config.min_segment_length,
            debug=self.debug
        )

        self.validator = DataMatrixValidator(
            min_edge_density=self.detector_config.validator_config.min_edge_density,
            max_edge_density=self.detector_config.validator_config.max_edge_density,
            min_aspect_ratio=self.detector_config.validator_config.min_aspect_ratio,
            max_aspect_ratio=self.detector_config.validator_config.max_aspect_ratio,
            min_size=self.detector_config.validator_config.min_size,
            debug=self.debug
        )

        self.border_fitter = BorderFitter(debug=self.debug)

        self.dashed_detector = DashedBorderDetector(
            tau=self.detector_config.dashed_border_config.tau,
            edge_threshold=self.detector_config.dashed_border_config.edge_threshold,
            min_transitions=self.detector_config.dashed_border_config.min_transitions,
            debug=self.debug
        )

    @staticmethod
    def parent_visited(visited: list, current: tuple):
        for v in visited:
            if current[0] >= v[0] and current[1] >= v[1] and current[2] <= v[2] and current[3] <= v[3]:
                return True

        return False

    def run(self, frame: np.ndarray):
        if not isinstance(frame, np.ndarray):
            raise ValueError("Input image should be a numpy array")

        if len(frame.shape) != 3:
            raise ValueError("Input image should have 3 channels")

        return self.process_frame(frame, self.detector_config)

    def process_frame(self, frame: np.ndarray, detector_config: DetectorConfig) -> List[DetectionResult]:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        candidates = self.extractor.get_candidates(frame)
        results = []

        visited_candidates = []

        # sort candidates from the highest area to the lowest area to take advantage of the deduplication strategy
        candidates.sort(reverse=True, key=lambda c: c[2] * c[3])

        for (x, y, w, h) in candidates:
            region = np.ascontiguousarray(gray[y:y + h, x:x + w])

            self.debug.show("region", region)
            self.debug.pause()


            if self.parent_visited(visited_candidates, (x, y, x + w, y + h)):
                continue

            visited_candidates.append((x, y, x + w, y + h))

            segments = self.l_finder.detect_lines(cv.medianBlur(region, detector_config.smoothing),
                                                  multiscale=detector_config.noisy_surface, scales=(0.5, 0.25),
                                                  apply_line_merging=detector_config.noisy_surface)
            l_patterns = self.l_finder.find_l_patterns(cv.cvtColor(region, cv.COLOR_GRAY2BGR), region, segments)

            region_has_valid = False

            for idx, l_pattern in enumerate(l_patterns):
                if self.check_l_pattern_duplicate(l_pattern, [x, y], results):
                    self.debug.log(f"[pipeline] l_pattern #{idx} skipped: duplicate of an already selected pattern")
                    continue

                validation = self.validator.validate(region, l_pattern)

                label = f"#{idx} VALIDATOR {'ACCEPT' if validation.is_valid else 'REJECT'}: {validation.reason}"
                show_pattern(
                    cv.cvtColor(region, cv.COLOR_GRAY2BGR),
                    l_pattern, label, validation.is_valid, window="detected"
                )

                if not validation.is_valid:
                    continue

                dashed_result, edges = self.dashed_detector.detect(region, l_pattern, scan_method="legacy",
                                                                   smoothing=detector_config.smoothing,
                                                                   canny_percentile=detector_config.canny_percentile)
                dashed_label = f"#{idx} DASHED {'ACCEPT' if dashed_result is not None else 'REJECT'}"
                show_pattern(
                    cv.cvtColor(region, cv.COLOR_GRAY2BGR),
                    l_pattern, dashed_label, dashed_result is not None, window="detected"
                )

                if dashed_result is None:
                    continue

                precise_location, inverted = self.border_fitter.fit(region, edges, l_pattern,
                                                          rough_location=dashed_result,
                                                          gaussian_size=detector_config.border_fitter_config.gaussian_size,
                                                          dilate_size=detector_config.border_fitter_config.dilate_size,
                                                          blob_min_area=detector_config.border_fitter_config.blob_min_area,
                                                          win_in=detector_config.border_fitter_config.win_in,
                                                          win_out=detector_config.border_fitter_config.win_out,
                                                          max_pts_outside=detector_config.border_fitter_config.ransac_max_pts_outside,
                                                          inlier_threshold=detector_config.border_fitter_config.ransac_inlier_threshold)

                if precise_location is None:
                    self.debug.log(f"[pipeline] precise location not found")
                    # print(f"[pipeline] precise location not found")
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
                    self.debug.log(f"[pipeline] precise location found: {precise_location.vertices}")
                    # print(f"[pipeline] precise location found: {precise_location.vertices}")
                    cv.drawContours(region, [np.int32(precise_location.vertices)], 0, (0, 0, 255), 2)

                    self.debug.show("precise location", region)
                    self.debug.pause()

                global_vertices = [(vx + x, vy + y) for vx, vy in precise_location.vertices]
                precise_location.vertices = global_vertices
                precise_location.center = (precise_location.center[0] + x, precise_location.center[1] + y)

                results.append(DetectionResult(
                    candidate_box=(x, y, w, h),
                    precise_location=precise_location,
                    l_patterns=[l_pattern],
                    is_valid=True,
                    score=validation.score,
                    is_inverted=inverted
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

    def check_l_pattern_duplicate(self, current_l_pattern: LPattern, current_candidate: list,
                                  selected: List[DetectionResult],
                                  distance_threshold: float = 60.0,
                                  corner_threshold: float = 20.0,
                                  perp_tol: float = 3.0,
                                  min_overlap_ratio: float = 0.8) -> bool:
        """Return True if `current_l_pattern` duplicates an already-selected one:
        either the same L re-detected in an overlapping region (corner and both
        vertices nearby) or a cut/truncated version of it (arms collinear with
        and overlapping the selected arms). Pixel tolerances are in full-frame
        coordinates; `current_candidate` is the (x, y, ...) candidate box."""
        ox, oy = current_candidate[0], current_candidate[1]
        cur_corner = np.array(current_l_pattern.corner, dtype=float) + (ox, oy)
        cur_v1 = np.array(current_l_pattern.vertex1, dtype=float) + (ox, oy)
        cur_v2 = np.array(current_l_pattern.vertex2, dtype=float) + (ox, oy)

        for s in selected:
            sel_pat = s.l_patterns[0]
            sx, sy = s.candidate_box[0], s.candidate_box[1]
            sel_corner = np.array(sel_pat.corner, dtype=float) + (sx, sy)
            sel_v1 = np.array(sel_pat.vertex1, dtype=float) + (sx, sy)
            sel_v2 = np.array(sel_pat.vertex2, dtype=float) + (sx, sy)

            # Both duplicate scenarios share the L corner, so gate on it first.
            corner_dist = float(np.hypot(*(cur_corner - sel_corner)))
            if corner_dist >= corner_threshold:
                continue

            # vertex1/vertex2 labelling is arbitrary: pair each current arm with
            # the selected arm it is most aligned with (more robust than the
            # vertex distance or the chirality sign, which break for cut arms
            # and near-degenerate Ls respectively).
            straight = (self._arm_alignment(cur_corner, cur_v1, sel_corner, sel_v1)
                        + self._arm_alignment(cur_corner, cur_v2, sel_corner, sel_v2))
            swapped = (self._arm_alignment(cur_corner, cur_v1, sel_corner, sel_v2)
                       + self._arm_alignment(cur_corner, cur_v2, sel_corner, sel_v1))
            if straight >= swapped:
                pairs = ((cur_v1, sel_v1), (cur_v2, sel_v2))
            else:
                pairs = ((cur_v1, sel_v2), (cur_v2, sel_v1))

            # Same L re-detected: corner and both vertices nearby.
            vertex_dist_sum = sum(float(np.hypot(*(cv - sv))) for cv, sv in pairs)
            distance_test = corner_dist + vertex_dist_sum < distance_threshold

            # Cut version: each current arm lies along its selected arm with
            # enough mutual overlap.
            containment_test = all(
                self._arms_overlap(cur_corner, cv, sel_corner, sv,
                                   perp_tol, min_overlap_ratio)
                for cv, sv in pairs)

            if distance_test or containment_test:
                return True

        return False

    @staticmethod
    def _arm_alignment(corner_a, vertex_a, corner_b, vertex_b) -> float:
        """Cosine of the angle between two arms (1 = same direction)."""
        dir_a = vertex_a - corner_a
        dir_b = vertex_b - corner_b
        norm = float(np.hypot(*dir_a)) * float(np.hypot(*dir_b))
        if norm < 1e-9:
            return -1.0
        return float(np.dot(dir_a, dir_b)) / norm

    @staticmethod
    def _arms_overlap(cur_corner, cur_vertex, sel_corner, sel_vertex,
                      perp_tol: float, min_overlap_ratio: float) -> bool:
        """True if the current arm lies along the selected arm's line (within
        `perp_tol` pixels) and the two arms overlap along that line by at least
        `min_overlap_ratio` of the shorter arm. Unlike strict segment
        containment, this tolerates sub-pixel jitter and a cut arm that pokes
        slightly past the selected one."""
        sel_dir = sel_vertex - sel_corner
        sel_len = float(np.hypot(*sel_dir))
        if sel_len < 1e-9:
            return False
        sel_unit = sel_dir / sel_len

        # Perpendicular (point-to-line) distance of both current endpoints from
        # the selected arm's line: |cross(sel_unit, p - sel_corner)|.
        for p in (cur_corner, cur_vertex):
            disp = p - sel_corner
            if abs(sel_unit[0] * disp[1] - sel_unit[1] * disp[0]) > perp_tol:
                return False

        # Overlap along the line, as a fraction of the shorter arm.
        proj = sorted((float(np.dot(sel_unit, cur_corner - sel_corner)),
                       float(np.dot(sel_unit, cur_vertex - sel_corner))))
        overlap = min(sel_len, proj[1]) - max(0.0, proj[0])
        shorter = min(sel_len, proj[1] - proj[0])
        if shorter < 1e-9:
            return False
        return overlap / shorter >= min_overlap_ratio
