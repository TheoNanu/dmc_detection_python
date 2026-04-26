import cv2 as cv
import numpy as np
from dataclasses import dataclass
from .l_finder_detector import LPattern

@dataclass
class ValidationResult:
    is_valid: bool
    edge_density: float
    aspect_ratio: float
    score: float
    reason: str = ""

class DataMatrixValidator:

    def __init__(self,
                 min_edge_density: float = 0.003,
                 max_edge_density: float = 0.75,
                 min_aspect_ratio: float = 0.5,
                 max_aspect_ratio: float = 2.0,
                 min_size: int = 20):
        self.min_edge_density = min_edge_density
        self.max_edge_density = max_edge_density
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_size = min_size

    def validate(self, gray_region: np.ndarray, l_pattern: LPattern) -> ValidationResult:
        h, w = gray_region.shape[:2]

        if h < self.min_size or w < self.min_size:
            reason = f"region too small ({w}x{h})"
            print(f"[validator] rejected: {reason}")
            return ValidationResult(False, 0, 0, 0, reason)

        # Check the L-pattern arm ratio, not the region's aspect ratio —
        # the candidate bounding box can be any shape
        aspect_ratio = max(w, h) / min(w, h)
        len_ratio = l_pattern.len1 / (l_pattern.len2 + 1e-6)
        if len_ratio > 2.5:
            reason = f"L-arm ratio too large ({len_ratio:.2f})"
            print(f"[validator] rejected: {reason}")
            return ValidationResult(False, 0, aspect_ratio, 0, reason)

        # Crop to the L-pattern bounding box so large regions don't dilute density
        lx, ly, lw, lh = l_pattern.get_bounding_box()
        x1, y1 = max(0, lx), max(0, ly)
        x2, y2 = min(w, lx + lw), min(h, ly + lh)
        roi = gray_region[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else gray_region

        # CLAHE before Canny so low-exposure regions still produce edges
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        roi_enhanced = clahe.apply(roi)

        # cv.imshow("roi", roi)
        # cv.imshow("roi_enhanced", roi_enhanced)
        edges = cv.Canny(roi_enhanced, 50, 150)
        # cv.imshow("edges", edges)
        # cv.waitKey(0)

        edge_pixels = np.count_nonzero(edges)
        edge_density = float(edge_pixels) / (roi.shape[0] * roi.shape[1])

        if not (self.min_edge_density <= edge_density <= self.max_edge_density):
            reason = f"edge density {edge_density:.4f} out of [{self.min_edge_density}, {self.max_edge_density}]"
            print(f"[validator] rejected: {reason}")
            return ValidationResult(False, edge_density, aspect_ratio, 0, reason)

        # Edge density depends on module pixel size (~ 2/P^2), so it is not
        # scale-invariant and can't be turned into a useful score on its own.
        # The L-finder's structure-tensor check already validates interior
        # texture, so we gate only on the permissive density range and use
        # the L-score as the validator score.
        l_score = l_pattern.score if hasattr(l_pattern, 'score') else 0.5
        total_score = l_score

        if total_score > 0.4:
            reason = f"score={total_score:.2f} density={edge_density:.4f} l={l_score:.2f}"
            print(f"[validator] accepted: {reason}")
        else:
            reason = f"l_score {l_score:.2f} <= 0.4 (density={edge_density:.4f})"
            print(f"[validator] rejected: {reason}")

        return ValidationResult(
            is_valid=total_score > 0.4,
            edge_density=edge_density,
            aspect_ratio=aspect_ratio,
            score=total_score,
            reason=reason
        )