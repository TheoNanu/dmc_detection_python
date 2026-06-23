import numpy as np
import pytest

import dmc
from dmc import DecoderConfig
from tests.cases import CASES


def test_blank_image_no_false_positive():
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    assert dmc.detect_and_decode(blank) == []

def test_does_not_raise_on_noise():
    noise = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dmc.detect_and_decode(noise)

def test_decoded_matrix_is_valid(load_image):
    case = next(c for c in CASES if c.expected)
    results = dmc.detect_and_decode(load_image(case.image), detector_config=case.config)
    assert results, f"{case.image}: expected at least one decode"
    for d in results:
        assert d.matrix.shape[0] == d.matrix.shape[1]
        assert d.matrix.shape[0] in DecoderConfig().valid_sizes

def test_none_input_image():
    with pytest.raises(ValueError):
        dmc.detect_and_decode(None)

def test_wrong_format_input_image(load_image_pillow):
    with pytest.raises(ValueError):
        dmc.detect_and_decode(load_image_pillow(CASES[0].image), detector_config=CASES[0].config)

def test_wrong_number_of_channels():
    blank = np.zeros((480, 640), dtype=np.uint8)
    with pytest.raises(ValueError):
        dmc.detect_and_decode(blank)