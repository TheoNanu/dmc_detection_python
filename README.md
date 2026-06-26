# dmc

![CI](https://github.com/TheoNanu/dmc_detection_python/actions/workflows/ci.yml/badge.svg)

A Python library for **Data Matrix Code (DMC) detection and decoding** in images, built on OpenCV and NumPy.

The pipeline locates Data Matrix codes in a frame (candidate extraction → L-finder pattern detection → validation → dashed-border detection → precise border fitting), rectifies each one, then estimates the module grid, samples it, and decodes the ECC200 payload.

## Features

- Detect one or more Data Matrix codes in a BGR image and recover their precise quadrilateral location.
- Decode located codes to their text payload.
- Fully configurable pipeline via typed config objects (no magic kwargs).
- Optional, pluggable debug visualization — silent by default, opt-in OpenCV windows for inspection.
- A single public entry point: `import dmc`.

## Installation

Requires Python 3.10+.

```bash
pip install -e .
```

On headless Linux (e.g. CI), `opencv-python` needs system GL libraries:

```bash
sudo apt-get install -y libgl1 libglib2.0-0
```

## Quick start

```python
import cv2 as cv
import dmc

frame = cv.imread("test_images/dmc_sample.jpg")   # BGR image (np.ndarray)

for decoded in dmc.detect_and_decode(frame):
    print(decoded.text)                            # the payload
    print(decoded.detection.precise_location.ordered_vertices())  # corner points
```

`detect_and_decode` returns a list of `Decoded` objects (empty if nothing was found). The input must be a BGR `numpy.ndarray`; passing `None`, a non-array (e.g. a PIL image), or a single-channel image raises `ValueError`.

## Per-image configuration

Detection parameters that work for one image often differ for another (lighting, noise, module size, surface). Rather than a single global default, pass a `DetectorConfig` tuned for the input:

```python
from dmc import DetectorConfig, BorderFitterConfig

config = DetectorConfig(
    smoothing=11,
    noisy_surface=True,
    canny_percentile=90.0,
    border_fitter_config=BorderFitterConfig(
        dilate_size=30,
        blob_min_area=50,
        win_in=50,
        win_out=30,
        ransac_max_pts_outside=40,
        ransac_inlier_threshold=0.9,
    ),
)

results = dmc.detect_and_decode(frame, detector_config=config)
```

A practical workflow is to keep a small table of `(image kind, config)` pairs — see `tests/cases.py` for a worked example mapping each test image to the config that decodes it.

## Debugging

By default the library is silent (no windows, no stdout). To inspect each stage visually, pass an OpenCV debug sink:

```python
from dmc import CvDebugSink

results = dmc.detect_and_decode(frame, debug=CvDebugSink())  # opens OpenCV windows
```

## Public API

| Function | Returns | Purpose |
|---|---|---|
| `detect(image, config=DetectorConfig(), debug=NullSink())` | `list[DetectionResult]` | Locate codes only |
| `decode(image, detection, config=DecoderConfig(), debug=NullSink())` | `Decoded \| None` | Decode one located code |
| `detect_and_decode(image, detector_config=DetectorConfig(), decoder_config=DecoderConfig(), debug=NullSink())` | `list[Decoded]` | Locate and decode in one call |

### Result types

- **`DetectionResult`** — `candidate_box`, `precise_location`, `l_patterns`, `is_valid`, `score`; `.rectify(frame, output_size)` returns the deskewed crop.
- **`Decoded`** — `detection` (the `DetectionResult`), `text` (payload), `codewords`, `matrix` (the sampled module grid).
- **`PreciseLocation`** — `vertices`, `center`, `angle`, `size`; `.ordered_vertices()` returns integer corner points.

### Configuration objects

`DetectorConfig` groups the per-stage configs: `ExtractionConfig`, `ValidatorConfig`, `LFinderConfig`, `DashedBorderConfig`, `BorderFitterConfig`. `DecoderConfig` controls grid estimation and sampling (`output_size`, `valid_sizes`, `smoothing`, `estimator_margin`). All are plain dataclasses with sensible defaults — override only what you need.

## Limitations

- Detection is sensitive to parameters; a single config will not handle all images (see *Per-image configuration*).

## Development

```bash
pip install -e ".[test]"
pytest
```

The suite covers decode correctness against known payloads (including reflectance-inverted codes), localization across all sample images, input validation, and pipeline invariants. RANSAC is seeded for reproducibility. Any known, unfixed gaps are tracked as strict `xfail` so they stay visible and the suite alerts if a gap is ever closed.

## Project structure

```
dmc/
├── api.py            # public functions: detect, decode, detect_and_decode
├── config.py         # configuration dataclasses
├── data.py           # result/DTO types
├── debug.py          # DebugSink protocol + NullSink / CvDebugSink
├── viz.py            # drawing helpers
├── detector/         # detection pipeline and its stages
└── decoder/          # grid estimation, sampling, decoding
```
