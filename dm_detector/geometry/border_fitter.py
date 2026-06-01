import cv2
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

    METHOD_EXTENT = "extent"
    METHOD_RANSAC = "ransac"
    METHOD_OUTER = "outer"
    METHOD_CLEAN = "clean"

    def __init__(self, method: str = METHOD_CLEAN):
        if method not in (self.METHOD_EXTENT, self.METHOD_RANSAC,
                          self.METHOD_OUTER, self.METHOD_CLEAN):
            raise ValueError(f"Unknown border-fitter method: {method!r}")
        self.method = method

    def fit(self, gray_img: np.ndarray, edge_img: np.ndarray, l_pattern: LPattern,
            rough_location: Optional[DataMatrixLocation] = None) -> Optional[PreciseLocation]:
        if rough_location is None:
            return self._fit_from_l_pattern(l_pattern)

        if self.method == self.METHOD_EXTENT:
            return self._fit_extent(gray_img, rough_location)
        if self.method == self.METHOD_OUTER:
            return self._fit_outer_edges(gray_img, rough_location)
        if self.method == self.METHOD_CLEAN:
            return self._fit_clean_linefit(gray_img, rough_location)
        return self._fit_ransac(edge_img, rough_location)

    @staticmethod
    def _build_precise_location(vertices_list: List[Tuple[float, float]]) -> PreciseLocation:
        pts = np.array(vertices_list)
        ctr = np.mean(pts, axis=0)
        side = pts[1] - pts[0]
        angle = float(np.degrees(np.arctan2(side[1], side[0])))
        w = float(np.linalg.norm(pts[1] - pts[0]))
        h = float(np.linalg.norm(pts[3] - pts[0]))

        return PreciseLocation(
            vertices=vertices_list,
            center=(float(ctr[0]), float(ctr[1])),
            angle=angle,
            size=(w, h)
        )

    def _fit_clean_linefit(self, gray_img: np.ndarray,
                           rough_location: DataMatrixLocation,
                           dilate: int = 5,
                           ransac_iterations: int = 1500) -> Optional[PreciseLocation]:
        """Quad-guided noise removal, then fit one *outer-tangent* line per side
        and intersect.

        The rough quad is used only as a *guide*: everything outside it is
        surrounding texture/noise and is masked away, leaving the DMC isolated.
        Each side is then fit to the cleaned boundary with an outer-tangent
        RANSAC (see ``_ransac_line_outer``) that prefers the outermost supported
        line rather than the one with the most inliers — so the dense first
        interior column/row can't win over the sparse, dashed outer border and
        clip it. No connected-component speck filter is used, because the timing
        dashes are themselves small components; the outer-tangent fit tolerates
        the few surviving specks instead.
        """
        quad = np.array([np.array(q, dtype=float) for q in rough_location.quads])
        center = quad.mean(axis=0)

        # 1. Binarise dark modules -> white. The threshold is Otsu computed over
        #    the DMC *interior* (the rough quad) only, not the whole ROI: on a
        #    noisy grey surface a global Otsu is biased by the large background
        #    and floods the image white, which then makes the dilated mask (not
        #    the DMC) the apparent border and pushes the fitted corners outward.
        #    Estimating the threshold from the DMC's own bimodal pixels adapts
        #    per image with no hardcoded value.
        blurred = cv.GaussianBlur(gray_img, (3, 3), 0)
        quad_mask = np.zeros(blurred.shape[:2], dtype=np.uint8)
        cv.fillConvexPoly(quad_mask, quad.astype(np.int32), 255)
        interior = blurred[quad_mask > 0]
        thr, _ = cv.threshold(interior.reshape(-1, 1), 0, 255,
                              cv.THRESH_BINARY + cv.THRESH_OTSU)
        _, bw = cv.threshold(blurred, thr, 255, cv.THRESH_BINARY_INV)
        cv.imshow("bw", bw)

        # 2. mask away everything outside the quad
        mask = cv.dilate(quad_mask, cv.getStructuringElement(cv.MORPH_ELLIPSE, (dilate, dilate)))
        cleaned = cv.bitwise_and(bw, mask)
        cleaned = cv.morphologyEx(cleaned, cv.MORPH_OPEN, np.ones((2, 2), np.uint8))

        cv.imshow("cleaned", cleaned)
        cv.waitKey(0)

        # 3. fit an outer-tangent line per side
        lines = []
        all_pts = []
        for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
            p_start, p_end = quad[a], quad[b]
            direction = p_end - p_start
            length = float(np.linalg.norm(direction))
            if length < 2.0:
                return None
            u_hat = direction / length
            perp = np.array([-u_hat[1], u_hat[0]])
            if np.dot(perp, (p_start + p_end) / 2 - center) < 0:
                perp = -perp  # outward normal

            pts = self._scan_boundary_inward(cleaned, p_start, p_end, perp)
            if len(pts) < 2:
                return None
            line = self._ransac_line_outer(
                pts, perp, direction, max_iterations=ransac_iterations
            )
            if line is None:
                return None
            lines.append(line)
            all_pts.append(pts)

        # 4. adjacent-side intersections -> corners (quad order: corner,horiz,vdiag,vert)
        corner = self._intersect_lines(lines[3], lines[0])
        horiz = self._intersect_lines(lines[0], lines[1])
        v_diag = self._intersect_lines(lines[1], lines[2])
        vert = self._intersect_lines(lines[2], lines[3])
        if any(v is None for v in (corner, horiz, v_diag, vert)):
            return None

        self._show_fitted_lines(cleaned, lines, all_pts, [corner, horiz, v_diag, vert])

        return self._build_precise_location(
            [(float(p[0]), float(p[1])) for p in (corner, horiz, v_diag, vert)]
        )

    @staticmethod
    def _show_fitted_lines(cleaned: np.ndarray, lines: list, all_pts: list,
                           corners: list, window: str = "border fit") -> None:
        """Show the cleaned DMC with each side's boundary points, the fitted
        outer-tangent lines, and the resulting quad (upscaled for visibility)."""
        vis = cv.cvtColor(cleaned, cv.COLOR_GRAY2BGR)
        colors = [(0, 0, 255), (0, 255, 255), (255, 0, 0), (0, 255, 0)]  # side 0..3
        diag = float(np.hypot(*cleaned.shape[:2]))

        for (normal, c), pts, col in zip(lines, all_pts, colors):
            for p in pts:
                cv.circle(vis, (int(round(p[0])), int(round(p[1]))), 1, col, -1)
            normal = np.asarray(normal, dtype=float)
            d = np.array([-normal[1], normal[0]])   # line direction
            p0 = -c * normal                          # a point on the line (unit normal)
            pa = (p0 - d * diag).astype(int)
            pb = (p0 + d * diag).astype(int)
            cv.line(vis, tuple(pa), tuple(pb), col, 1)

        quad = np.array([[int(round(p[0])), int(round(p[1]))] for p in corners], dtype=np.int32)
        cv.polylines(vis, [quad], True, (255, 0, 255), 1)
        for p in corners:
            cv.circle(vis, (int(round(p[0])), int(round(p[1]))), 2, (255, 0, 255), -1)

        scale = max(1, int(round(600.0 / max(cleaned.shape[:2]))))
        if scale > 1:
            vis = cv.resize(vis, None, fx=scale, fy=scale, interpolation=cv.INTER_NEAREST)
        cv.imshow(window, vis)
        cv.waitKey(0)

    @staticmethod
    def _scan_boundary_inward(cleaned: np.ndarray, p_start: np.ndarray, p_end: np.ndarray,
                              perp: np.ndarray, win_out: float = 8.0, win_in: float = 8.0,
                              step: float = 0.5) -> np.ndarray:
        """Sample along a side and, at each sample, scan perpendicular from
        outside inward (``perp`` points outward) for the first two consecutive
        foreground pixels — the DMC's outer boundary on a noise-free image.
        Two-in-a-row rejects a lone surviving speck; gaps find nothing and are
        skipped. At dashed-border gap positions this returns the interior module
        edge; the outer-tangent fit downstream keeps those points *inside* the
        line rather than on it."""
        direction = p_end - p_start
        length = float(np.linalg.norm(direction))
        if length < 2.0:
            return np.empty((0, 2))
        u_hat = direction / length

        h, w = cleaned.shape[:2]

        def is_fg(q: np.ndarray) -> bool:
            ix, iy = int(round(q[0])), int(round(q[1]))
            return 0 <= iy < h and 0 <= ix < w and cleaned[iy, ix] > 0

        pts = []
        n = max(4, int(length))
        for i in range(n):
            base = p_start + (i + 0.5) * length / n * u_hat
            s = win_out
            while s >= -win_in:
                if is_fg(base + perp * s) and is_fg(base + perp * (s - step)):
                    pts.append(base + perp * s)
                    break
                s -= step

        return np.array(pts) if pts else np.empty((0, 2))

    @staticmethod
    def _ransac_line_outer(points: np.ndarray, perp_out: np.ndarray, expected_dir: np.ndarray,
                           max_iterations: int = 1500, inlier_threshold: float = 1.2,
                           max_outside: int = 2,
                           angle_tol_deg: float = 22.0) -> Optional[Tuple[np.ndarray, float]]:
        """Outer-tangent RANSAC.

        Among candidate lines whose direction is within ``angle_tol_deg`` of
        ``expected_dir``, keep only those with at most ``max_outside`` foreground
        points lying strictly *outward* of the line (i.e. the line is an outer
        boundary, not an interior one), and pick the best-supported such line.
        This makes the sparse dashed outer edge — whose tips define the true
        boundary — win over a denser interior line that would otherwise clip it.
        """
        n = len(points)
        if n < 2:
            return None
        perp_out = perp_out / (np.linalg.norm(perp_out) + 1e-12)
        expected_angle = float(np.arctan2(expected_dir[1], expected_dir[0]))
        tol = np.radians(angle_tol_deg)

        def angle_diff(a: float, b: float) -> float:
            d = abs(a - b) % np.pi
            return min(d, np.pi - d)

        best_inliers = None
        best_key = (-1, -1e18)  # (inlier_count, signed offset c) — more inliers, then more outward
        rng = np.random.default_rng(0)  # deterministic: same warp every run
        for _ in range(max_iterations):
            idx = rng.choice(n, 2, replace=False)
            p1, p2 = points[idx[0]], points[idx[1]]
            d = p2 - p1
            d_len = np.linalg.norm(d)
            if d_len < 1e-6:
                continue
            if angle_diff(float(np.arctan2(d[1], d[0])), expected_angle) > tol:
                continue
            normal = np.array([-d[1], d[0]]) / d_len
            if np.dot(normal, perp_out) < 0:
                normal = -normal  # orient normal outward
            c = -np.dot(normal, p1)
            signed = points @ normal + c          # > 0 means outward of the line
            inliers = np.abs(signed) < inlier_threshold
            outside = int(np.sum(signed > inlier_threshold))
            if outside > max_outside:
                continue
            key = (int(np.sum(inliers)), float(c))
            if key > best_key:
                best_key = key
                best_inliers = inliers

        if best_inliers is None or int(np.sum(best_inliers)) < 2:
            return None

        inlier_pts = points[best_inliers].astype(np.float32)
        vx, vy, x0, y0 = cv.fitLine(inlier_pts, cv.DIST_L2, 0, 0.01, 0.01).flatten()
        normal = np.array([-vy, vx])
        if np.dot(normal, perp_out) < 0:
            normal = -normal
        c = -(normal[0] * x0 + normal[1] * y0)
        return (normal, c)

    def _fit_outer_edges(self, gray_img: np.ndarray,
                         rough_location: DataMatrixLocation,
                         ransac_iterations: int = 500,
                         inlier_threshold: float = 1.5) -> Optional[PreciseLocation]:
        """Fit each of the four DMC borders to the *outer edge* of its outermost
        module, then intersect adjacent borders for the corners.

        Unlike the rough quad (which puts the timing borders on the module
        centerline, parallel to the L-arms), every side is fit independently to
        its own boundary points, so a side that is not parallel to its opposite
        L-arm (perspective) is tracked correctly instead of clipping the
        outermost module row toward the far corner.
        """
        # rough_location.quads = (corner, horiz_vertex, v_diag, vert_vertex)
        quad = np.array([np.array(q, dtype=float) for q in rough_location.quads])
        center = quad.mean(axis=0)

        # Otsu threshold from the quad's bounding box so "dark module vs light
        # background" is decided locally rather than over the whole ROI.
        h_img, w_img = gray_img.shape[:2]
        pad = 6
        x0 = max(0, int(np.floor(quad[:, 0].min())) - pad)
        y0 = max(0, int(np.floor(quad[:, 1].min())) - pad)
        x1 = min(w_img, int(np.ceil(quad[:, 0].max())) + pad)
        y1 = min(h_img, int(np.ceil(quad[:, 1].max())) + pad)
        crop = gray_img[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        thr = float(cv.threshold(crop, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[0])

        # Fit a line per edge (corner->horiz, horiz->vdiag, vdiag->vert, vert->corner)
        edge_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
        lines = []
        for a, b in edge_pairs:
            edge_len = float(np.linalg.norm(quad[b] - quad[a]))
            win_out = max(5.0, 0.18 * edge_len)   # start out in the background
            win_in = max(1.5, 0.05 * edge_len)    # stay below ~1 module inward
            pts = self._scan_outer_edge_points(
                gray_img, quad[a], quad[b], center, thr, win_out, win_in
            )
            if len(pts) < 2:
                return None
            line = self._ransac_fit_line(
                pts, max_iterations=ransac_iterations, inlier_threshold=inlier_threshold
            )
            if line is None:
                return None
            lines.append(line)

        # Each quad corner is the intersection of its two adjacent edge lines.
        corner = self._intersect_lines(lines[3], lines[0])
        horiz = self._intersect_lines(lines[0], lines[1])
        v_diag = self._intersect_lines(lines[1], lines[2])
        vert = self._intersect_lines(lines[2], lines[3])
        if any(v is None for v in (corner, horiz, v_diag, vert)):
            return None

        return self._build_precise_location(
            [(float(p[0]), float(p[1])) for p in (corner, horiz, v_diag, vert)]
        )

    @staticmethod
    def _scan_outer_edge_points(gray: np.ndarray, p_start: np.ndarray, p_end: np.ndarray,
                                center: np.ndarray, thr: float,
                                win_out: float, win_in: float,
                                step: float = 0.4) -> np.ndarray:
        """Sample along a border segment; at each sample march perpendicular
        *inward from outside* and take the first dark pixel as the outer edge of
        the outermost module.

        Marching from the background side makes the outermost module the first
        thing hit, so the dense interior can't be mistaken for the boundary. The
        inward cap (``win_in``, kept below one module) means a dashed-border gap
        finds no dark pixel and is skipped rather than latching onto an interior
        module. RANSAC downstream fits the line through the surviving points."""
        direction = p_end - p_start
        length = float(np.linalg.norm(direction))
        if length < 2.0:
            return np.empty((0, 2))

        u_hat = direction / length
        perp = np.array([-u_hat[1], u_hat[0]])
        if np.dot(perp, (p_start + p_end) / 2 - center) < 0:
            perp = -perp  # ensure perp points outward

        h, w = gray.shape[:2]

        def sample(q: np.ndarray) -> float:
            ix, iy = int(round(q[0])), int(round(q[1]))
            if 0 <= iy < h and 0 <= ix < w:
                return float(gray[iy, ix])
            return 255.0

        pts = []
        n = max(4, int(length))
        for i in range(n):
            base = p_start + (i + 0.5) * length / n * u_hat

            # March inward from outside; the first sustained dark pixel is the
            # outer boundary. Require two consecutive dark samples so isolated
            # background speckle (dot-peen / crack noise) is not mistaken for it.
            edge_s = None
            s = win_out
            while s >= -win_in:
                if sample(base + perp * s) < thr and sample(base + perp * (s - step)) < thr:
                    edge_s = s
                    break
                s -= step
            if edge_s is None:
                continue  # gap position, no outer module here
            pts.append(base + perp * edge_s)

        return np.array(pts) if pts else np.empty((0, 2))

    def _fit_extent(self, gray_img: np.ndarray,
                    rough_location: DataMatrixLocation,
                    extend_factor: float = 1.15,
                    warp_size: int = 400,
                    threshold: float = 0.15) -> Optional[PreciseLocation]:
        quad_pts = np.array([q for q in rough_location.quads], dtype=np.float32)
        center = np.mean(quad_pts, axis=0)

        extended = np.array(
            [center + extend_factor * (q - center) for q in quad_pts],
            dtype=np.float32
        )
        h_img, w_img = gray_img.shape[:2]
        extended = np.clip(extended, [0, 0], [w_img - 1, h_img - 1]).astype(np.float32)

        dst_pts = np.array([
            [0, warp_size - 1],
            [warp_size - 1, warp_size - 1],
            [warp_size - 1, 0],
            [0, 0],
        ], dtype=np.float32)

        M = cv.getPerspectiveTransform(extended, dst_pts)
        warped = cv.warpPerspective(gray_img, M, (warp_size, warp_size))

        blurred = cv.GaussianBlur(warped, (3, 3), 0)
        _, binary = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

        bounds = self._find_dmc_extent(binary, threshold=threshold)
        if bounds is None:
            return None
        x_left, y_top, x_right, y_bottom = bounds

        self._draw_extent_debug(warped, binary, x_left, y_top, x_right, y_bottom)

        warped_corners = np.array([
            [x_left, y_bottom],
            [x_right, y_bottom],
            [x_right, y_top],
            [x_left, y_top],
        ], dtype=np.float32)

        M_inv = np.linalg.inv(M)
        original_corners = cv.perspectiveTransform(
            warped_corners.reshape(-1, 1, 2), M_inv
        ).reshape(-1, 2)

        return self._build_precise_location(
            [(float(p[0]), float(p[1])) for p in original_corners]
        )

    def _fit_ransac(self, edge_img: np.ndarray,
                    rough_location: DataMatrixLocation,
                    scan_range: int = 3,
                    outward_limit: int = 4,
                    ransac_iterations: int = 1000) -> Optional[PreciseLocation]:
        quad_pts = np.array([q for q in rough_location.quads], dtype=np.float32)
        center = np.mean(quad_pts, axis=0)

        borders = [
            (quad_pts[0], quad_pts[1]),
            (quad_pts[1], quad_pts[2]),
            (quad_pts[2], quad_pts[3]),
            (quad_pts[3], quad_pts[0]),
        ]

        fitted_lines = []
        all_border_pts = []
        for p_start, p_end in borders:
            border_pts = self._scan_border_points(
                edge_img, p_start, p_end, center,
                scan_range=scan_range, outward_limit=outward_limit
            )
            all_border_pts.append(border_pts)
            if len(border_pts) < 2:
                return None
            line = self._ransac_fit_line(border_pts, max_iterations=ransac_iterations)
            if line is None:
                return None
            fitted_lines.append(line)

        self._draw_scan_debug(edge_img, borders, all_border_pts, fitted_lines)

        vertices_raw = [
            self._intersect_lines(fitted_lines[3], fitted_lines[0]),
            self._intersect_lines(fitted_lines[0], fitted_lines[1]),
            self._intersect_lines(fitted_lines[1], fitted_lines[2]),
            self._intersect_lines(fitted_lines[2], fitted_lines[3]),
        ]

        if any(v is None for v in vertices_raw):
            return None

        return self._build_precise_location(
            [(float(v[0]), float(v[1])) for v in vertices_raw]
        )

    @staticmethod
    def _find_dmc_extent(binary: np.ndarray,
                         threshold: float = 0.15) -> Optional[Tuple[int, int, int, int]]:
        row_ratio = np.mean(binary > 0, axis=1)
        col_ratio = np.mean(binary > 0, axis=0)

        rows_above = np.where(row_ratio > threshold)[0]
        cols_above = np.where(col_ratio > threshold)[0]

        if len(rows_above) == 0 or len(cols_above) == 0:
            return None

        return (int(cols_above[0]), int(rows_above[0]),
                int(cols_above[-1]), int(rows_above[-1]))

    @staticmethod
    def _draw_extent_debug(warped: np.ndarray, binary: np.ndarray,
                           x_left: int, y_top: int, x_right: int, y_bottom: int) -> None:
        warped_bgr = cv.cvtColor(warped, cv.COLOR_GRAY2BGR)
        binary_bgr = cv.cvtColor(binary, cv.COLOR_GRAY2BGR)

        for img in (warped_bgr, binary_bgr):
            cv.line(img, (0, y_top), (img.shape[1] - 1, y_top), (0, 0, 255), 1)
            cv.line(img, (0, y_bottom), (img.shape[1] - 1, y_bottom), (0, 0, 255), 1)
            cv.line(img, (x_left, 0), (x_left, img.shape[0] - 1), (0, 255, 0), 1)
            cv.line(img, (x_right, 0), (x_right, img.shape[0] - 1), (0, 255, 0), 1)

        cv.imshow("border_fitter: warped + extent", warped_bgr)
        cv.imshow("border_fitter: binary + extent", binary_bgr)
        cv.waitKey(0)

    @staticmethod
    def _scan_border_points(edge_img: np.ndarray, p_start: np.ndarray, p_end: np.ndarray,
                            center: np.ndarray, scan_range: int = 15,
                            outward_limit: int = 4) -> np.ndarray:
        direction = p_end - p_start
        length = float(np.linalg.norm(direction))
        if length < 1:
            return np.array([])

        u_hat = direction / length
        perp = np.array([-u_hat[1], u_hat[0]])
        midpoint = (p_start + p_end) / 2
        if np.dot(perp, midpoint - center) < 0:
            perp = -perp

        usable = length
        skip = 0.0
        num_samples = int(usable)

        border_points = []
        img_h, img_w = edge_img.shape[:2]
        for i in range(num_samples):
            t = skip + (i + 0.5) * usable / num_samples
            base = p_start + t * u_hat

            for d in range(outward_limit, -scan_range - 1, -1):
                pos = base + perp * d
                px = int(round(pos[0]))
                py = int(round(pos[1]))

                if 0 <= px < img_w and 0 <= py < img_h:
                    if edge_img[py, px] > 0:
                        border_points.append([float(px), float(py)])

        return np.array(border_points) if border_points else np.array([])

    @staticmethod
    def _ransac_fit_line(points: np.ndarray, max_iterations: int = 300,
                         inlier_threshold: float = 2.0) -> Optional[Tuple[np.ndarray, float]]:
        if len(points) < 2:
            return None

        best_inliers = None
        best_inlier_count = 0
        n = len(points)

        for _ in range(max_iterations):
            idx = np.random.choice(n, 2, replace=False)
            p1, p2 = points[idx[0]], points[idx[1]]

            d = p2 - p1
            d_len = np.linalg.norm(d)
            if d_len < 1e-6:
                continue

            normal = np.array([-d[1], d[0]]) / d_len
            c = -np.dot(normal, p1)

            distances = np.abs(points @ normal + c)
            inliers = distances < inlier_threshold
            inlier_count = int(np.sum(inliers))

            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_inliers = inliers

        if best_inliers is None or best_inlier_count < 2:
            return None

        inlier_pts = points[best_inliers].astype(np.float32)
        result = cv.fitLine(inlier_pts, cv.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x0, y0 = result.flatten()

        normal = np.array([-vy, vx])
        c = -(normal[0] * x0 + normal[1] * y0)

        return (normal, c)

    @staticmethod
    def _intersect_lines(line1: Tuple[np.ndarray, float],
                         line2: Tuple[np.ndarray, float]) -> Optional[np.ndarray]:
        n1, c1 = line1
        n2, c2 = line2

        A = np.array([n1, n2])
        b = np.array([-c1, -c2])

        det = np.linalg.det(A)
        if abs(det) < 1e-6:
            return None

        return np.linalg.solve(A, b)

    @staticmethod
    def _draw_scan_debug(edge_img: np.ndarray, borders: list, all_border_pts: list,
                         fitted_lines: list) -> None:
        if len(edge_img.shape) == 2:
            debug = cv.cvtColor(edge_img, cv.COLOR_GRAY2BGR)
        else:
            debug = edge_img.copy()

        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
        labels = ["bottom", "right", "top", "left"]

        pts_only = debug.copy()
        for i, pts in enumerate(all_border_pts):
            for pt in pts:
                cv.circle(pts_only, (int(round(pt[0])), int(round(pt[1]))), 1, colors[i], -1)

        with_lines = debug.copy()
        for i, ((p_start, p_end), (normal, c)) in enumerate(zip(borders, fitted_lines)):
            d_start = float(np.dot(normal, p_start) + c)
            d_end = float(np.dot(normal, p_end) + c)
            proj_start = p_start - d_start * normal
            proj_end = p_end - d_end * normal
            cv.line(
                with_lines,
                (int(round(proj_start[0])), int(round(proj_start[1]))),
                (int(round(proj_end[0])), int(round(proj_end[1]))),
                colors[i], 2
            )

        legend_h = 20 * len(labels) + 10
        legend = np.zeros((legend_h, 200, 3), dtype=np.uint8)
        for i, (color, lbl) in enumerate(zip(colors, labels)):
            cv.circle(legend, (15, 15 + i * 20), 5, color, -1)
            cv.putText(legend, lbl, (30, 20 + i * 20), cv.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        target_size = 800
        h, w = pts_only.shape[:2]
        scale = max(1, int(round(target_size / max(h, w))))
        if scale > 1:
            pts_only = cv.resize(pts_only, (w * scale, h * scale), interpolation=cv.INTER_NEAREST)
            with_lines = cv.resize(with_lines, (w * scale, h * scale), interpolation=cv.INTER_NEAREST)

        cv.imshow("border_fitter: sampled points", pts_only)
        cv.imshow("border_fitter: ransac lines", with_lines)
        cv.imshow("border_fitter: legend", legend)
        cv.waitKey(0)

    @staticmethod
    def _fit_from_l_pattern(l_pattern: LPattern) -> PreciseLocation:
        corner = np.array(l_pattern.corner)
        v1 = np.array(l_pattern.vertex1)
        v2 = np.array(l_pattern.vertex2)

        dir_h = v1 - corner
        dir_v = v2 - corner

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
            size=(float(np.linalg.norm(dir_h)), float(np.linalg.norm(dir_v)))
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
