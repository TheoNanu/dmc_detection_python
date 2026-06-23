import pytest

import dmc
from tests.cases import CASES


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.image)
def test_localization(case, load_image):
    dets = dmc.detect(load_image(case.image), config=case.config)
    assert any(d.precise_location is not None for d in dets), f"{case.image}: no code detected"