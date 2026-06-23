from pathlib import Path

import cv2 as cv
import pytest
from PIL import Image


@pytest.fixture(scope="session")
def images_dir() -> Path:
    return Path(__file__).parent.parent / "test_images"

@pytest.fixture
def load_image(images_dir):
    def _load(name: str):
        frame = cv.imread(str(images_dir / name))
        assert frame is not None, f"Could not read {name}"
        return frame

    return _load

@pytest.fixture
def load_image_pillow(images_dir):
    def _load(name: str):
        image = Image.open(str(images_dir / name))
        assert image is not None, f"Could not read {name}"
        return image

    return _load

@pytest.fixture(autouse=True)
def _seed():
    import numpy as np, cv2 as cv
    np.random.seed(0)
    cv.setRNGSeed(0)