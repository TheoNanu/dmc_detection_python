import pytest

import dmc
from tests.cases import CASES

# Images that expose a known, unfixed library limitation. Kept as strict xfail so
# they stay tracked and the suite alerts (xpass -> failure) if the gap is ever closed.
KNOWN_UNSUPPORTED: dict[str, str] = {}

DECODE_CASES = [c for c in CASES if c.expected]


def _param(case):
    marks = []
    if case.image in KNOWN_UNSUPPORTED:
        marks.append(pytest.mark.xfail(reason=KNOWN_UNSUPPORTED[case.image], strict=True))
    return pytest.param(case, id=case.image, marks=marks)


@pytest.mark.parametrize("case", [_param(c) for c in DECODE_CASES])
def test_decode_to_known_dmc_payload(case, load_image):
    results = dmc.detect_and_decode(load_image(case.image), detector_config=case.config)

    texts = [r.text for r in results]

    assert case.expected in texts, f"{case.image}: got {texts}, expected {case.expected}"
