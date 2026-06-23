from dataclasses import dataclass, field


@dataclass
class ExtractionConfig:
    canny_t1: int = 50
    canny_t2: int = 150
    min_area: float = 400.0
    min_perimeter: float = 80.0
    padding: int = 25


@dataclass
class ValidatorConfig:
    min_edge_density: float = 0.003
    max_edge_density: float = 0.75
    min_aspect_ratio: float = 0.5
    max_aspect_ratio: float = 2.0
    min_size: int = 20


@dataclass
class LFinderConfig:
    neighborhood_radius: float = 10.0
    min_angle: float = 60.0
    max_angle: float = 120.0
    max_length_ratio: float = 5.0
    min_segment_length: float = 20.0


@dataclass
class DashedBorderConfig:
    tau: int = 5
    edge_threshold: int = 50
    min_transitions: int = 6


@dataclass
class BorderFitterConfig:
    gaussian_size: int = 3
    dilate_size: int = 5
    blob_min_area: int = 0
    win_in: int = 20
    win_out: int = 20
    ransac_max_pts_outside: int = 10
    ransac_inlier_threshold: float = 0.9


@dataclass
class DetectorConfig:
    smoothing: int = 11
    noisy_surface: bool = False
    canny_percentile: float = 90.0
    extraction_config: ExtractionConfig = field(default_factory=ExtractionConfig)
    border_fitter_config: BorderFitterConfig = field(default_factory=BorderFitterConfig)
    validator_config: ValidatorConfig = field(default_factory=ValidatorConfig)
    l_finder_config: LFinderConfig = field(default_factory=LFinderConfig)
    dashed_border_config: DashedBorderConfig = field(default_factory=DashedBorderConfig)


@dataclass
class DecoderConfig:
    output_size: int = 400
    valid_sizes: tuple[int, ...] = (10, 12, 14, 16, 18, 20, 22, 24, 26, 32, 40,
                                      44, 48, 52, 64, 72, 80, 88, 96, 104, 120, 132, 144)
    smoothing: int = 11
    estimator_margin: int = 1