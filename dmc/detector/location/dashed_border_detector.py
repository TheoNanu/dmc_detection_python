import cv2 as cv
import numpy as np
from typing import Tuple, Optional

from dmc.data import DataMatrixLocation
from dmc.debug import DebugSink, NullSink
from dmc.utils import auto_canny
from dmc.viz import draw_sampled_border, show_scanned_lines
from .l_finder_detector import LPattern


class DashedBorderDetector:

    def __init__(self, tau: int = 5, edge_threshold: int = 50, min_transitions: int = 6, debug: DebugSink = NullSink()):
        self.tau = tau
        self.edge_threshold = edge_threshold
        self.min_transitions = min_transitions
        self.debug = debug

    def get_detection_regions(self, l_pattern: LPattern,
                              img_shape: Tuple[int, int]) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int],
    Tuple[np.ndarray, np.ndarray], np.ndarray, Tuple[float, float], np.ndarray, Tuple[float, float], np.ndarray]:
        corner = np.array(l_pattern.corner, dtype=float)
        v_a = np.array(l_pattern.vertex1, dtype=float)
        v_b = np.array(l_pattern.vertex2, dtype=float)

        arm_a = v_a - corner
        arm_b = v_b - corner

        # Classify arms by orientation. The "horizontal" arm is the one whose
        # direction vector is more aligned with the x-axis; its far end is the
        # DMC's opposite-to-corner horizontal neighbor. The other arm is
        # vertical. LPattern.vertex1/vertex2 ordering is arbitrary, so we never
        # rely on it — classification is purely geometric.
        if abs(arm_a[0]) > abs(arm_a[1]):
            horiz_vertex, horiz_arm = v_a, arm_a
            vert_vertex, vert_arm = v_b, arm_b
        else:
            horiz_vertex, horiz_arm = v_b, arm_b
            vert_vertex, vert_arm = v_a, arm_a

        # Fourth DMC corner (opposite the L corner)
        v_diag = horiz_vertex + vert_vertex - corner

        horiz_len = float(np.linalg.norm(horiz_arm))
        vert_len = float(np.linalg.norm(vert_arm))
        tau = self.tau
        img_h, img_w = img_shape
        depth_frac = 0.1

        def strip_aabb(p_start: np.ndarray, p_end: np.ndarray,
                       inward_unit: np.ndarray, inward_depth: float) -> Tuple[int, int, int, int]:
            offset = inward_unit * inward_depth
            pts = np.stack([p_start, p_end, p_start + offset, p_end + offset])
            x_min = int(np.floor(pts[:, 0].min())) - tau
            y_min = int(np.floor(pts[:, 1].min())) - tau
            x_max = int(np.ceil(pts[:, 0].max())) + tau
            y_max = int(np.ceil(pts[:, 1].max())) + tau
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(img_w, x_max)
            y_max = min(img_h, y_max)
            return x_min, y_min, max(1, x_max - x_min), max(1, y_max - y_min)

        # Upper dashed border runs from the vertical arm's far end to v_diag,
        # parallel to the horizontal arm. Its inward direction (toward the DMC
        # interior) is -vert_arm / |vert_arm|.
        inward_upper = -vert_arm / max(vert_len, 1e-6)
        upper_region = strip_aabb(vert_vertex, v_diag, inward_upper, depth_frac * vert_len)

        # Right dashed border runs from the horizontal arm's far end to v_diag,
        # parallel to the vertical arm. Inward direction is -horiz_arm / |horiz_arm|.
        inward_right = -horiz_arm / max(horiz_len, 1e-6)
        right_region = strip_aabb(horiz_vertex, v_diag, inward_right, depth_frac * horiz_len)

        return (upper_region, right_region, (inward_right, inward_upper), vert_vertex,
                (horiz_len, depth_frac * vert_len), horiz_vertex, (horiz_len * depth_frac, vert_len), v_diag)

    @staticmethod
    def _box_profiles(img: np.ndarray, origin: np.ndarray, u_hat: np.ndarray,
                      v_hat: np.ndarray, u_len: int, v_len: int,
                      half_width: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample profiles along ``origin + u*u_hat + v*v_hat`` for every
        v in [0, v_len), each averaged over a thin box of +-half_width px
        perpendicular to the scan line. Pores are blobs of a few px while a
        dash spans the whole band height, so the averaging suppresses surface
        noise but keeps dash contrast.

        Returns ``(profiles, rows, cols)``, each of shape (v_len, u_len):
        the box-averaged profiles plus the pixel coordinates of each profile's
        center scan line (for visualization). The whole strip is sampled as
        one grid and the box means come from a cumulative sum, so cost is one
        image gather per (u, v+-half_width) cell instead of one pass per
        (v, offset) pair."""
        us = np.arange(u_len, dtype=float)
        ks = np.arange(-half_width, v_len + half_width, dtype=float)
        pts = (origin[None, None, :]
               + us[:, None, None] * u_hat[None, None, :]
               + ks[None, :, None] * v_hat[None, None, :])
        rows = np.clip(np.round(pts[..., 1]).astype(int), 0, img.shape[0] - 1)
        cols = np.clip(np.round(pts[..., 0]).astype(int), 0, img.shape[1] - 1)
        grid = img[rows, cols].astype(float)            # (u_len, len(ks))

        win = 2 * half_width + 1
        csum = np.cumsum(np.concatenate(
            [np.zeros((u_len, 1)), grid], axis=1), axis=1)
        # window over ks[v] .. ks[v + 2*half_width]  ==  v-hw .. v+hw
        box = (csum[:, win:] - csum[:, :-win]) / win    # (u_len, v_len)
        # center line of the box that produced profile v is grid column
        # v + half_width; return its coords aligned with the profiles, so
        # rows[v]/cols[v] are the pixels of the scan line at depth v
        center = slice(half_width, half_width + v_len)
        return box.T, rows[:, center].T, cols[:, center].T

    @staticmethod
    def _periodicity_score(profile: np.ndarray, min_run_frac: float = 0.4,
                           min_amplitude: float = 20.0) -> Tuple[int, float]:
        """Transition count and periodicity score of a gray profile.

        Binarizes at the min/max midpoint, drops runs far shorter than the
        median run (pore-induced flips -- a real dash can never be that
        narrow), then weights the count by adjacent-run uniformity. Adjacent
        runs are compared (not global std/mean) so perspective foreshortening,
        which changes dash spacing smoothly, is not penalized."""
        lo, hi = float(np.min(profile)), float(np.max(profile))
        if hi - lo < min_amplitude:
            return 0, 0.0
        binary = (profile > (lo + hi) / 2.0).astype(np.int8)
        runs = np.diff(np.flatnonzero(np.diff(binary)))
        if runs.size > 2:
            runs = runs[runs >= min_run_frac * np.median(runs)]
        transitions = int(runs.size) + 1
        if runs.size < 3:
            return transitions, 0.0
        adj = np.abs(np.log(runs[1:] / runs[:-1].astype(float)))
        return transitions, transitions / (1.0 + 3.0 * float(np.median(adj)))

    def scan_edge_along_arms_direction(
            self,
            edge_img: np.ndarray,
            sample_img: np.ndarray,
            u_hat: np.ndarray,
            v_hat: np.ndarray,
            origin: np.ndarray,
            u_len: int,
            v_len: int,
            angle_range_deg: float = 3.0,
            angle_step_deg: float = 0.5,
            box_half_width: int = 4,
            score_threshold_frac: float = 0.6
    ):
        """Locate the dashed (timing) border inside the search strip.

        The scan pivots at ``origin`` (the L-arm tip -- a detected feature the
        border genuinely starts at, regardless of perspective) and searches
        jointly over direction (+-angle_range_deg around u_hat, since under
        perspective the border is not parallel to the opposite L arm) and
        inward offset v. Rows are scored on the box-averaged gray profile, not
        the edge image, so surface pores do not inflate the count.

        Among rows at the best angle, the OUTERMOST row whose score reaches
        score_threshold_frac of the strip's best is chosen: interior data-module
        rows are periodic too, so periodicity alone cannot separate them from
        the border -- but the border is by definition the outermost periodic
        structure.
        """
        angles = np.arange(-angle_range_deg, angle_range_deg + 1e-9, angle_step_deg)

        best = None  # (score, transitions, angle_idx, v, u_rot, v_rot)
        per_angle: list = []
        coords_per_angle: list = []
        for a_idx, deg in enumerate(angles):
            rad = np.radians(deg)
            c, s = np.cos(rad), np.sin(rad)
            u_rot = np.array([c * u_hat[0] - s * u_hat[1], s * u_hat[0] + c * u_hat[1]])
            v_rot = np.array([c * v_hat[0] - s * v_hat[1], s * v_hat[0] + c * v_hat[1]])
            profiles, prof_rows, prof_cols = self._box_profiles(
                sample_img, origin, u_rot, v_rot, u_len, v_len, box_half_width)
            coords_per_angle.append((prof_rows, prof_cols))

            rows = []
            for v in range(v_len):
                transitions, score = self._periodicity_score(profiles[v])
                rows.append((score, transitions))
                if best is None or score > best[0]:
                    best = (score, transitions, a_idx, v, u_rot, v_rot)
            per_angle.append(rows)

        if best is None or best[0] <= 0.0:
            show_scanned_lines(edge_img, coords_per_angle, debug=self.debug)
            return 0, np.array([]), [], 0

        # outermost row above threshold, at the best angle
        threshold = score_threshold_frac * best[0]
        a_idx, u_rot, v_rot = best[2], best[4], best[5]
        chosen_v, transitions = best[3], best[1]
        for v, (score, t) in enumerate(per_angle[a_idx]):
            if score >= threshold:
                chosen_v, transitions = v, t
                break

        self.debug.log(f"[dashed] best angle {angles[a_idx]:+.1f} deg, outermost row v={chosen_v} "
              f"(best v={best[3]}), transitions={transitions}, score={per_angle[a_idx][chosen_v][0]:.1f}")


        show_scanned_lines(edge_img, coords_per_angle,
                                 best_angle_idx=a_idx, chosen_v=chosen_v, debug=self.debug)

        edge_row = np.empty(u_len)
        sampled_coords = []
        for u in range(u_len):
            coords = origin + u * u_rot + chosen_v * v_rot
            row = max(0, min(int(round(coords[1])), sample_img.shape[0] - 1))
            col = max(0, min(int(round(coords[0])), sample_img.shape[1] - 1))
            edge_row[u] = sample_img[row, col]
            sampled_coords.append((col, row))

        return chosen_v, edge_row, sampled_coords, transitions

    def scan_edge_along_arms_direction_legacy(
            self,
            edge_img: np.ndarray,
            sample_img: np.ndarray,
            u_hat: np.ndarray,
            v_hat: np.ndarray,
            origin: np.ndarray,
            u_len: int,
            v_len: int
    ):
        """Original scan: no angle search, count transitions directly on the
        EDGE image, pick the single row with the most transitions (argmax).

        Kept as a baseline / fallback. Works well on clean, fronto-parallel,
        high-contrast symbols, but on noisy surfaces the pore edges inflate
        the count and the argmax can latch onto an interior row -- which is
        what motivated ``scan_edge_along_arms_direction``. Same return
        signature, so ``detect`` can call either interchangeably.
        """
        best_v = 0
        best_transitions = 0
        for v in range(v_len):
            signal = np.array([
                int(edge_img[
                        max(0, min(int(round((origin + u * u_hat + v * v_hat)[1])), edge_img.shape[0] - 1)),
                        max(0, min(int(round((origin + u * u_hat + v * v_hat)[0])), edge_img.shape[1] - 1))
                    ])
                for u in range(u_len)
            ])
            binary = (signal > 0).astype(np.int8)
            transitions = int(np.sum(np.abs(np.diff(binary))))
            if transitions > best_transitions:
                best_transitions = transitions
                best_v = v

        edge_row = np.empty(u_len)
        sampled_coords = []
        for u in range(u_len):
            coords = origin + u * u_hat + best_v * v_hat
            row = max(0, min(int(round(coords[1])), sample_img.shape[0] - 1))
            col = max(0, min(int(round(coords[0])), sample_img.shape[1] - 1))
            edge_row[u] = sample_img[row, col]
            sampled_coords.append((col, row))

        return best_v, edge_row, sampled_coords, best_transitions

    @staticmethod
    def find_outer_border_line(
            edge_img: np.ndarray,
            u_hat: np.ndarray,
            v_hat: np.ndarray,
            origin: np.ndarray,
            u_len: int,
            inward_depth: int,
            margin: int = 10,
            threshold_frac: float = 0.4
    ) -> list:
        outer_origin = origin - margin * v_hat
        total_v = margin + inward_depth

        transitions_per_v = []
        for v in range(total_v):
            signal = np.array([
                int(edge_img[
                    max(0, min(int(round((outer_origin + u * u_hat + v * v_hat)[1])), edge_img.shape[0] - 1)),
                    max(0, min(int(round((outer_origin + u * u_hat + v * v_hat)[0])), edge_img.shape[1] - 1))
                ])
                for u in range(u_len)
            ])
            binary = (signal > 0).astype(np.int8)
            transitions = int(np.sum(np.abs(np.diff(binary))))
            transitions_per_v.append(transitions)

        max_trans = max(transitions_per_v) if transitions_per_v else 0
        threshold = max_trans * threshold_frac

        outer_v = 0
        for v, t in enumerate(transitions_per_v):
            if t >= threshold:
                outer_v = v
                break

        sampled_coords = []
        for u in range(u_len):
            coords = outer_origin + u * u_hat + outer_v * v_hat
            row = max(0, min(int(round(coords[1])), edge_img.shape[0] - 1))
            col = max(0, min(int(round(coords[0])), edge_img.shape[1] - 1))
            sampled_coords.append((col, row))

        return sampled_coords

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

    def count_transitions(self, border: np.ndarray) -> int:
        self.debug.log(f"[dashed-border] border: {border}")

        if len(border) < 2:
            return 0

        b_min, b_max = float(np.min(border)), float(np.max(border))

        if b_max - b_min < 20.0:
            return 0

        threshold = (b_min + b_max) / 2.0
        binary_border = (border > threshold).astype(np.int8)

        transitions = np.sum(np.abs(np.diff(binary_border)))
        return int(transitions)

    def detect(self, gray_img: np.ndarray, l_pattern: LPattern,
               scan_method: str = "angle", smoothing: int = 15, canny_percentile: float = 90.0) -> Tuple[Optional[DataMatrixLocation], np.ndarray]:
        """Locate the dashed borders of the symbol whose L finder pattern is
        ``l_pattern``. ``scan_method`` selects the border scan:

          "angle"  -- angle search + box-averaged gray periodicity scoring
                      (robust on noisy / perspective-distorted surfaces)
          "legacy" -- original edge-image transition argmax (fast, best on
                      clean fronto-parallel symbols)
        """
        gray_img = cv.medianBlur(gray_img, smoothing)
        edges = auto_canny(gray_img, percentile=canny_percentile)

        (upper_region, right_region, (inward_right_unit, inward_upper_unit), vert_vertex, (border_len_upper, upper_inward),
         horiz_vertex, (right_inward, border_len_right), v_diag) = self.get_detection_regions(l_pattern, gray_img.shape)

        self.debug.show("canny dashed border", edges)
        self.debug.pause()

        if scan_method == "legacy":
            scan = self.scan_edge_along_arms_direction_legacy
            scan_kwargs = {}
        elif scan_method == "angle":
            scan = self.scan_edge_along_arms_direction
            scan_kwargs = {"angle_range_deg": 15.0}
        else:
            raise ValueError(f"unknown scan_method {scan_method!r}; expected 'angle' or 'legacy'")

        upper_dashed_row, extracted_arr_upper, upper_coords, t_upper = scan(edges, gray_img, inward_right_unit * -1,
                                                                                    inward_upper_unit, vert_vertex,
                                                                                    int(border_len_upper), int(upper_inward), **scan_kwargs)
        right_dashed_col, extracted_arr_right, right_coords, t_right = scan(edges, gray_img, inward_upper_unit * -1,
                                                                                    inward_right_unit, horiz_vertex,
                                                                                    int(border_len_right), int(right_inward), **scan_kwargs)

        draw_sampled_border(edges, upper_coords, right_coords, debug=self.debug)

        t_upper = self.count_transitions(extracted_arr_upper)
        t_right = self.count_transitions(extracted_arr_right)

        if t_upper < self.min_transitions or t_right < self.min_transitions:
            self.debug.log(f"[dashed] rejected: upper_transitions={t_upper}, right_transitions={t_right}, min={self.min_transitions}")
            return None, edges

        self.debug.log(f"[dashed] accepted: upper_transitions={t_upper}, right_transitions={t_right}")

        upper_outer_coords = self.find_outer_border_line(
            edges, inward_right_unit * -1, inward_upper_unit, vert_vertex,
            int(border_len_upper), int(upper_inward))
        right_outer_coords = self.find_outer_border_line(
            edges, inward_upper_unit * -1, inward_right_unit, horiz_vertex,
            int(border_len_right), int(right_inward))

        corners = np.array([l_pattern.corner, vert_vertex, horiz_vertex, v_diag])
        min_x = int(np.floor(corners[:, 0].min()))
        min_y = int(np.floor(corners[:, 1].min()))
        max_x = int(np.ceil(corners[:, 0].max()))
        max_y = int(np.ceil(corners[:, 1].max()))
        bounding_box = (min_x, min_y, max_x - min_x, max_y - min_y)

        quad_corners = np.array([l_pattern.corner, horiz_vertex, v_diag, vert_vertex])
        quads = tuple((int(p[0]), int(p[1])) for p in quad_corners)

        return DataMatrixLocation(
            l_pattern=l_pattern,
            upper_border=upper_region,
            right_border=right_region,
            bounding_box=bounding_box,
            quads=quads,
            upper_outer_coords=upper_outer_coords,
            right_outer_coords=right_outer_coords
        ), edges