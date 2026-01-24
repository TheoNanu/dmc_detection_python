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

class DataMatrixValidator:

    def __init__(self,
                 min_edge_density: float = 0.1,
                 max_edge_density: float = 0.6,
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
            return ValidationResult(False, 0, 0, 0)

        aspect_ratio = max(w, h) / min(w, h)
        if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
            return ValidationResult(False, 0, aspect_ratio, 0)

        edges = cv.Canny(gray_region, 50, 150)
        edge_pixels = np.count_nonzero(edges)
        edge_density = float(edge_pixels) / (h * w)

        if not (self.min_edge_density <= edge_density <= self.max_edge_density):
            return ValidationResult(False, edge_density, aspect_ratio, 0)

        len_ratio = l_pattern.len1 / (l_pattern.len2 + 1e-6)
        if len_ratio > 2.5:
            return ValidationResult(False, edge_density, aspect_ratio, 0)

        density_score = 1.0 - abs(edge_density - 0.3) / 0.3
        aspect_score = 1.0 - abs(aspect_ratio - 1.0) / 1.0
        l_score = l_pattern.score if hasattr(l_pattern, 'score') else 0.5

        total_score = density_score * 0.3 + aspect_score * 0.3 + l_score * 0.4

        return ValidationResult(
            is_valid=total_score > 0.4,
            edge_density=edge_density,
            aspect_ratio=aspect_ratio,
            score=total_score
        )