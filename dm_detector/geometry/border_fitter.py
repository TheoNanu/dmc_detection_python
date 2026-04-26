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

    def __init__(self, method: str = METHOD_EXTENT):
        if method not in (self.METHOD_EXTENT, self.METHOD_RANSAC):
            raise ValueError(f"Unknown border-fitter method: {method!r}")
        self.method = method

    def fit(self, gray_img: np.ndarray, edge_img: np.ndarray, l_pattern: LPattern,
            rough_location: Optional[DataMatrixLocation] = None) -> Optional[PreciseLocation]:
        if rough_location is None:
            return self._fit_from_l_pattern(l_pattern)

        if self.method == self.METHOD_EXTENT:
            return self._fit_extent(gray_img, rough_location)
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
