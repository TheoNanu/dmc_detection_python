import cv2 as cv
import numpy as np
from typing import List

from config import DetectorConfig
from data import DetectionResult
from debug import DebugSink, NullSink
from dm_detector.extraction.candidate_extraction import CandidateExtraction
from dm_detector.location.l_finder_detector import LFinderDetector, LPattern
from dm_detector.location.validator import DataMatrixValidator
from dm_detector.location.dashed_border_detector import DashedBorderDetector
from dm_detector.geometry.border_fitter import BorderFitter, PreciseLocation


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
        return self.process_frame(frame,
                           smoothing=self.detector_config.smoothing,
                           noisy_surface=self.detector_config.noisy_surface,
                           canny_percentile=self.detector_config.canny_percentile,
                           border_fitter_gaussian_size=self.detector_config.border_fitter_config.gaussian_size,
                           fitter_dilate_size=self.detector_config.border_fitter_config.dilate_size,
                           fitter_blob_removal_min_area=self.detector_config.border_fitter_config.blob_min_area,
                           fitter_win_in=self.detector_config.border_fitter_config.win_in,
                           fitter_win_out=self.detector_config.border_fitter_config.win_out,
                           fitter_ransac_max_pts_outside=self.detector_config.border_fitter_config.ransac_max_pts_outside,
                           fitter_ransac_inlier_threshold_dist=self.detector_config.border_fitter_config.ransac_inlier_threshold)

    def detect(self, frame: np.ndarray, smoothing: int = 11, noisy_surface=False, canny_percentile: float=90.0,
               border_fitter_gaussian_size: int = 3, fitter_dilate_size: int = 5, fitter_blob_removal_min_area: int = 0,
               fitter_win_in: int = 20, fitter_win_out: int = 20, fitter_ransac_max_pts_outside: int = 10,
               fitter_ransac_inlier_threshold_dist: float = 0.9) -> List[DetectionResult]:

        valid_size = [10, 12, 14, 16, 18, 20, 22, 24, 26, 32, 40, 44, 48, 52, 64, 72, 80, 88, 96, 104, 120, 132, 144]
        detection_results = []

        if frame is None:
            print(f"error: could not load image")
            return []

        return self.process_frame(frame.copy(),
                                     smoothing=smoothing,
                                     noisy_surface=noisy_surface,
                                     canny_percentile=canny_percentile,
                                     border_fitter_gaussian_size=border_fitter_gaussian_size,
                                     fitter_dilate_size=fitter_dilate_size,
                                     fitter_blob_removal_min_area=fitter_blob_removal_min_area,
                                     fitter_win_in=fitter_win_in,
                                     fitter_win_out=fitter_win_out,
                                     fitter_ransac_max_pts_outside=fitter_ransac_max_pts_outside,
                                     fitter_ransac_inlier_threshold_dist=fitter_ransac_inlier_threshold_dist)

        # output_frame = frame.copy()
        #
        # cv.imshow("frame after process", frame)
        # cv.waitKey(0)
        #
        # if results:
        #     for res in results:
        #         if res.is_valid:
        #             warped_bgr = res.get_rectified_image(frame, output_size=400)
        #
        #             if warped_bgr is not None:
        #                 warp_gray = cv.cvtColor(warped_bgr, cv.COLOR_BGR2GRAY)
        #                 clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        #                 # warp_gray = clahe.apply(warp_gray)
        #
        #                 cv.imshow("2. rectified image (warped) before smooth", warp_gray)
        #                 print(f"Warp gray before smoothing: {warp_gray}")
        #
        #                 # warp_gray = cv.GaussianBlur(warp_gray,(11, 11), 0)
        #                 warp_gray = cv.medianBlur(warp_gray, smoothing)
        #
        #                 cv.imshow("2. rectified image (warped)", warp_gray)
        #                 print(f"Warp gray before smoothing: {warp_gray}")
        #                 # warp_gray = clahe.apply(warp_gray)
        #
        #                 print("\ngrid estimator test:\n")
        #                 estimator = GridEstimator(margin=1)
        #                 h, w = warp_gray.shape
        #
        #                 # Primary: timing-pattern grid (sub-pixel, per-boundary module
        #                 # positions). Falls back to the autocorrelation pitch when the
        #                 # timing borders are too degraded to locate enough transitions.
        #                 grid = estimator.estimate_grid(warp_gray)
        #
        #                 data_matrix = None
        #                 if grid is not None:
        #                     col_centres, row_centres = grid
        #                     nx, ny = len(col_centres), len(row_centres)
        #                     print(f"[timing] estimated matrix size: {nx} cols x {ny} rows")
        #
        #                     bits = estimator.sample_matrix(warp_gray, col_centres, row_centres)
        #                     data_matrix = bits[1:-1, 1:-1]  # strip the 1-module border
        #
        #                     grid_vis = warp_gray.copy()
        #                     estimator.draw_module_grid(grid_vis, col_centres, row_centres)
        #                     grid_vis = estimator.draw_module_numbers(grid_vis, col_centres, row_centres)
        #                     cv.imshow("final grid", grid_vis)
        #                 else:
        #                     pitch, score = estimator.estimate_pitch(warp_gray)
        #                     if pitch is not None:
        #                         print(f"[autocorr] pitch={pitch:.2f} px score={score:.2f}")
        #                         w_eff = w - 2 * estimator.margin
        #                         h_eff = h - 2 * estimator.margin
        #                         nx = int(round(w_eff / pitch))
        #                         ny = int(round(h_eff / pitch))
        #                         data_matrix = estimator.get_matrix_data(warp_gray, w / nx, h / ny, ny, nx)
        #                         print(f"[autocorr] estimated matrix size: {nx} cols x {ny} rows")
        #                         estimator.draw_grid(warp_gray, h / ny, w / nx)
        #                         cv.imshow("final grid", warp_gray)
        #                     else:
        #                         print("could not estimate grid")
        #
        #                 if data_matrix is not None and (
        #                         data_matrix.shape[0] in valid_size and data_matrix.shape[0] == data_matrix.shape[1]):
        #                     codewords = estimator.ecc200_codewords_from_data_modules(data_matrix)
        #                     print(f"codewords: {codewords}")
        #
        #                     data_codewords = []
        #                     for cw in codewords:
        #                         if cw == 129:  # padding, stop
        #                             break
        #                         if 1 <= cw <= 128:
        #                             data_codewords.append(chr(cw - 1))
        #                         elif 130 <= cw <= 229:
        #                             data_codewords.append(f"{cw - 130:02d}")
        #                         # ignore ECC codewords / mode switches for now
        #
        #                     print("".join(data_codewords))
        #                     print(f"data codewords: {data_codewords}")
        #
        #                     detection_results.append(res)
        #
        # cv.waitKey(0)
        # return  detection_results

    def process_frame(self, frame: np.ndarray,
                      smoothing: int = 11,
                      noisy_surface=False,
                      canny_percentile: float=90.0,
                      border_fitter_gaussian_size: int = 3,
                      fitter_dilate_size: int = 5,
                      fitter_blob_removal_min_area: int = 0,
                      fitter_win_in: int = 20,
                      fitter_win_out: int = 20,
                      fitter_ransac_max_pts_outside: int = 3,
                      fitter_ransac_inlier_threshold_dist: float = 1.2) -> List[DetectionResult]:
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

            # cv.imshow("region", region)
            # cv.waitKey(0)

            if self.parent_visited(visited_candidates, (x, y, x + w, y + h)):
                # print(f"[pipeline] region ({x},{y},{w},{h}) skipped: parent already visited")
                continue

            visited_candidates.append((x, y, x + w, y + h))
            # print(f"[pipeline] processing region ({x},{y},{w},{h})")

            segments = self.l_finder.detect_lines(cv.medianBlur(region, smoothing), multiscale=noisy_surface, scales=(0.5, 0.25),
                                                  apply_line_merging=noisy_surface)
            l_patterns = self.l_finder.find_l_patterns(cv.cvtColor(region, cv.COLOR_GRAY2BGR), region, segments)

            # print(f"[pipeline] l_patterns found: {len(l_patterns)}")

            region_has_valid = False

            for idx, l_pattern in enumerate(l_patterns):
                if self.check_l_pattern_duplicate(l_pattern, [x, y], results):
                    self.debug.log(f"[pipeline] l_pattern #{idx} skipped: duplicate of an already selected pattern")
                    continue

                validation = self.validator.validate(region, l_pattern)

                label = f"#{idx} VALIDATOR {'ACCEPT' if validation.is_valid else 'REJECT'}: {validation.reason}"
                self.l_finder._show_pattern(
                    cv.cvtColor(region, cv.COLOR_GRAY2BGR),
                    l_pattern, label, validation.is_valid, window="detected"
                )

                if not validation.is_valid:
                    continue

                dashed_result, edges = self.dashed_detector.detect(region, l_pattern, scan_method="legacy",
                                                                   smoothing=smoothing, canny_percentile=canny_percentile)
                dashed_label = f"#{idx} DASHED {'ACCEPT' if dashed_result is not None else 'REJECT'}"
                self.l_finder._show_pattern(
                    cv.cvtColor(region, cv.COLOR_GRAY2BGR),
                    l_pattern, dashed_label, dashed_result is not None, window="detected"
                )

                if dashed_result is None:
                    continue

                precise_location = self.border_fitter.fit(region, edges, l_pattern,
                                                          rough_location=dashed_result,
                                                          gaussian_size=border_fitter_gaussian_size,
                                                          dilate_size=fitter_dilate_size,
                                                          blob_min_area=fitter_blob_removal_min_area,
                                                          win_in=fitter_win_in,
                                                          win_out=fitter_win_out,
                                                          max_pts_outside=fitter_ransac_max_pts_outside,
                                                          inlier_threshold=fitter_ransac_inlier_threshold_dist)

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

    @staticmethod
    def draw_results(frame: np.ndarray, results: List[DetectionResult],
                     debug_view: bool = False) -> np.ndarray:
        output = frame.copy()

        for result in results:
            x, y, w, h = result.candidate_box

            if result.precise_location and result.is_valid:
                print("DMC DETECTED")
                vertices = result.precise_location.ordered_vertices()
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

    def draw_dmc(self, frame: np.ndarray, location: PreciseLocation, decoded_text: str = ""):
        output = frame.copy()

        vertices = location.get_ordered_vertices()
        pts = np.array(vertices, dtype=np.int32)
        cv.polylines(output, [pts], True, (0, 255, 0), 2)

        cx, cy = int(location.center[0]), int(location.center[1])

        text_size = cv.getTextSize(decoded_text, cv.FONT_HERSHEY_SIMPLEX, 1, 2)

        print(text_size)

        cv.putText(output, decoded_text,
                   (int(cx - (text_size[0][0] / 2)), int(cy - (text_size[0][1] / 2))),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (54, 54, 173), 2)

        return output

    @staticmethod
    def draw_l_pattern(frame: np.ndarray, l_pattern: LPattern, color: tuple = (0, 255, 0)):
        result = frame.copy()

        cv.line(result, (int(l_pattern.vertex1[0]), int(l_pattern.vertex1[1])),
                (int(l_pattern.corner[0]), int(l_pattern.corner[1])), color, 3)
        cv.line(result, (int(l_pattern.vertex2[0]), int(l_pattern.vertex2[1])),
                (int(l_pattern.corner[0]), int(l_pattern.corner[1])), color, 3)

        return result
