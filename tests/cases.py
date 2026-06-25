from dataclasses import dataclass, field

from dmc import DetectorConfig, BorderFitterConfig


@dataclass
class Case:
    image: str
    expected: str
    config: DetectorConfig = field(default_factory=DetectorConfig)


REAL = DetectorConfig(
    smoothing=1,
    noisy_surface=False,
    canny_percentile=90.0,
    border_fitter_config=BorderFitterConfig(
        gaussian_size=3,
        dilate_size=5,
        blob_min_area=0,
        win_in=20,
        win_out=20,
        ransac_max_pts_outside=10,
        ransac_inlier_threshold=0.9
    )
)

SYNTHETIC_SMOOTH = DetectorConfig(
    smoothing=11,
    noisy_surface=False,
    canny_percentile=90.0,
    border_fitter_config=BorderFitterConfig(
        gaussian_size=3,
        dilate_size=30,
        blob_min_area=50,
        win_in=50,
        win_out=30,
        ransac_max_pts_outside=40,
        ransac_inlier_threshold=0.9
    )
)

SYNTHETIC_NOISY_TWO = DetectorConfig(
    smoothing=13,
    noisy_surface=True,
    canny_percentile=90.0,
    border_fitter_config=BorderFitterConfig(
        gaussian_size=3,
        dilate_size=5,
        blob_min_area=50,
        win_in=50,
        win_out=30,
        ransac_max_pts_outside=60,
        ransac_inlier_threshold=1.5
    )
)

SYNTHETIC_NOISY_SIX = DetectorConfig(
    smoothing=11,
    noisy_surface=True,
    canny_percentile=90.0,
    border_fitter_config=BorderFitterConfig(
        gaussian_size=3,
        dilate_size=30,
        blob_min_area=50,
        win_in=50,
        win_out=30,
        ransac_max_pts_outside=40,
        ransac_inlier_threshold=0.9
    )
)

SYNTHETIC_NOISY_NINE = DetectorConfig(
    smoothing=11,
    noisy_surface=True,
    canny_percentile=90.0,
    border_fitter_config=BorderFitterConfig(
        gaussian_size=3,
        dilate_size=30,
        blob_min_area=50,
        win_in=50,
        win_out=30,
        ransac_max_pts_outside=40,
        ransac_inlier_threshold=0.9
    )
)

CASES = [
    Case("dmc_on_object_test_image.png", "00003686754401016001--1829020134", REAL),
    Case("dmc_sample.jpg", "Technomark", REAL),
    Case("dmc_sample2.jpeg", "0009999991000037869801053972573810883302001000240210369623\x1d400G10700324", REAL),
    Case("dmc_test_image.png", "Data Matrix Code Test Image Number 3", REAL),
    Case("img.png", "350081004283911975", REAL),
    Case("synthetic_image1.png", "DMC12345", SYNTHETIC_SMOOTH),
    Case("synthetic_image2.png", "DMC12345", SYNTHETIC_NOISY_TWO),
    Case("synthetic_image3.png", "DMC12345", SYNTHETIC_SMOOTH),
    Case("synthetic_image4.png", "DMC12345", SYNTHETIC_SMOOTH),
    Case("synthetic_image5.png", "DMC12345", SYNTHETIC_SMOOTH),
    Case("synthetic_image6.png", "DMC12345", SYNTHETIC_NOISY_SIX),
    Case("synthetic_image7.png", "DMC12345", SYNTHETIC_SMOOTH),
    Case("synthetic_image8.png", "DMC12345", SYNTHETIC_SMOOTH),
    Case("synthetic_image9.png", "DMC12345", SYNTHETIC_NOISY_NINE),
    Case("synthetic_image10.png", "DMC12345", SYNTHETIC_SMOOTH)
]