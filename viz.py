from typing import List, Tuple, Optional, Union

import numpy as np
import cv2 as cv

from data import PreciseLocation, LPattern, DetectionResult
from debug import DebugSink, NullSink


def draw_dmc(frame: np.ndarray, location: PreciseLocation, decoded_text: str = ""):
    output = frame.copy()

    vertices = location.ordered_vertices()
    pts = np.array(vertices, dtype=np.int32)
    cv.polylines(output, [pts], True, (0, 255, 0), 2)

    cx, cy = int(location.center[0]), int(location.center[1])

    text_size = cv.getTextSize(decoded_text, cv.FONT_HERSHEY_SIMPLEX, 1, 2)

    print(text_size)

    cv.putText(output, decoded_text,
               (int(cx - (text_size[0][0] / 2)), int(cy - (text_size[0][1] / 2))),
               cv.FONT_HERSHEY_SIMPLEX, 1, (54, 54, 173), 2)

    return output

def draw_l_pattern(frame: np.ndarray, l_pattern: LPattern, color: tuple = (0, 255, 0)):
    result = frame.copy()

    cv.line(result, (int(l_pattern.vertex1[0]), int(l_pattern.vertex1[1])),
            (int(l_pattern.corner[0]), int(l_pattern.corner[1])), color, 3)
    cv.line(result, (int(l_pattern.vertex2[0]), int(l_pattern.vertex2[1])),
            (int(l_pattern.corner[0]), int(l_pattern.corner[1])), color, 3)

    return result

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

def draw_precise_location(image: np.ndarray,
                          location: PreciseLocation,
                          color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    result = image.copy()
    vertices = location.ordered_vertices()

    pts = np.array(vertices, dtype=np.int32).reshape((-1, 1, 2))
    cv.polylines(result, [pts], True, color, 2)

    return result

def _draw_scan_debug(edge_img: np.ndarray, borders: list, all_border_pts: list,
                     fitted_lines: list, deb: DebugSink) -> None:
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

    deb.show("border_fitter: sampled points", pts_only)
    deb.show("border_fitter: ransac lines", with_lines)
    deb.show("border_fitter: legend", legend)
    deb.pause()
    # cv.imshow("border_fitter: sampled points", pts_only)
    # cv.imshow("border_fitter: ransac lines", with_lines)
    # cv.imshow("border_fitter: legend", legend)
    # cv.waitKey(0)

def _draw_extent_debug(warped: np.ndarray, binary: np.ndarray,
                       x_left: int, y_top: int, x_right: int, y_bottom: int, debug: DebugSink) -> None:
    warped_bgr = cv.cvtColor(warped, cv.COLOR_GRAY2BGR)
    binary_bgr = cv.cvtColor(binary, cv.COLOR_GRAY2BGR)

    for img in (warped_bgr, binary_bgr):
        cv.line(img, (0, y_top), (img.shape[1] - 1, y_top), (0, 0, 255), 1)
        cv.line(img, (0, y_bottom), (img.shape[1] - 1, y_bottom), (0, 0, 255), 1)
        cv.line(img, (x_left, 0), (x_left, img.shape[0] - 1), (0, 255, 0), 1)
        cv.line(img, (x_right, 0), (x_right, img.shape[0] - 1), (0, 255, 0), 1)

    debug.show("border_fitter: warped + extent", warped_bgr)
    debug.show("border_fitter: binary + extent", binary_bgr)
    debug.pause()
    # cv.imshow("border_fitter: warped + extent", warped_bgr)
    # cv.imshow("border_fitter: binary + extent", binary_bgr)
    # cv.waitKey(0)

def _show_fitted_lines(cleaned: np.ndarray, lines: list, all_pts: list,
                       corners: list, window: str = "border fit", need_resize: bool = False,
                       debug: DebugSink = NullSink()) -> None:
    """Show the cleaned DMC with each side's boundary points, the fitted
    outer-tangent lines, and the resulting quad (upscaled for visibility)."""
    vis = cv.cvtColor(cleaned, cv.COLOR_GRAY2BGR)
    colors = [(0, 0, 255), (0, 140, 255), (255, 0, 0), (0, 255, 0)]  # side 0..3
    diag = float(np.hypot(*cleaned.shape[:2]))

    for (normal, c), pts, col in zip(lines, all_pts, colors):
        for p in pts:
            cv.circle(vis, (int(round(p[0])), int(round(p[1]))), 1, col, -1)
        normal = np.asarray(normal, dtype=float)
        d = np.array([-normal[1], normal[0]])   # line direction
        p0 = -c * normal                          # a point on the line (unit normal)
        pa = (p0 - d * diag).astype(int)
        pb = (p0 + d * diag).astype(int)
        debug.log(f"[border-fitter] {pts.shape[0]} points for side with color: {col}")
        cv.line(vis, tuple(pa), tuple(pb), col, 1)

    quad = np.array([[int(round(p[0])), int(round(p[1]))] for p in corners], dtype=np.int32)
    cv.polylines(vis, [quad], True, (255, 0, 255), 1)
    for p in corners:
        cv.circle(vis, (int(round(p[0])), int(round(p[1]))), 2, (255, 0, 255), -1)

    scale = max(1, int(round(600.0 / max(cleaned.shape[:2]))))
    if scale > 1:
        vis = cv.resize(vis, None, fx=scale, fy=scale, interpolation=cv.INTER_NEAREST)

    if need_resize:
        vis = cv.resize(vis, None, fx=2.0, fy=2.0, interpolation=cv.INTER_NEAREST)

    debug.show(window, vis)
    debug.pause()
    # cv.imshow(window, vis)
    # cv.waitKey(0)

def _show_sampled_points(cleaned: np.ndarray, pts: np.ndarray, window: str = "sampled points",
                         color: tuple = (0, 0, 255), need_resize: bool = False, debug: DebugSink = NullSink()):
    vis = cv.cvtColor(cleaned, cv.COLOR_GRAY2BGR)

    for p in pts:
        cv.circle(vis, (int(round(p[0])), int(round(p[1]))), 1, color, -1)

    scale = max(1, int(round(600.0 / max(cleaned.shape[:2]))))
    if scale > 1:
        vis = cv.resize(vis, None, fx=scale, fy=scale, interpolation=cv.INTER_NEAREST)

    if need_resize:
        vis = cv.resize(vis, None, fx=2.0, fy=2.0, interpolation=cv.INTER_NEAREST)

    debug.show(window, vis)
    debug.pause()
    # cv.imshow(window, vis)
    # cv.waitKey(0)

def _show_scanned_lines(edge_img: np.ndarray,
                        coords_per_angle: list,
                        best_angle_idx: Optional[int] = None,
                        chosen_v: Optional[int] = None,
                        window: str = "scanned lines",
                        debug: DebugSink = NullSink()):
    """One composite overlay of every line the scan visited: all angles in
    dim red, the best angle's rows in yellow, the chosen line in green."""
    if not coords_per_angle:
        return
    vis = cv.cvtColor(edge_img, cv.COLOR_GRAY2BGR)
    for rows, cols in coords_per_angle:
        vis[rows, cols] = (60, 60, 200)
    if best_angle_idx is not None:
        rows, cols = coords_per_angle[best_angle_idx]
        vis[rows, cols] = (0, 255, 255)
        if chosen_v is not None:
            vis[rows[chosen_v], cols[chosen_v]] = (0, 255, 0)

    debug.show(window, vis)
    debug.pause(0)
    # cv.imshow(window, vis)
    # cv.waitKey(0)

def draw_sampled_border(edge_img: np.ndarray,
                        upper_coords: Union[list | None],
                        right_coords: Union[list | None],
                        debug: DebugSink = NullSink()):
    vis = cv.cvtColor(edge_img, cv.COLOR_GRAY2BGR)
    if upper_coords is not None:
        for (col, row) in upper_coords:
            vis[row, col] = (0, 0, 255)

    if right_coords is not None:
        for (col, row) in right_coords:
            vis[row, col] = (0, 0, 255)

    debug.show("sampled borders", vis)
    debug.pause()
    # cv.imshow("sampled borders", vis)
    # cv.waitKey(0)