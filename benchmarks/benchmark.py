"""Labeled accuracy benchmark: dmc vs pylibdmtx vs zxing-cpp.

Runs all three decoders over the labeled image set and reports, per decoder:
  - decode-rate: fraction of images for which it returned any payload
  - accuracy:    fraction of *labeled* images it decoded to the correct payload
  - avg time per image

Labels and per-image dmc configs come from tests/cases.py. Images whose
expected payload is "" are treated as unlabeled: they count toward decode-rate
but not accuracy.

Note on fairness: dmc is run with each image's tuned config (its intended
usage); pylibdmtx and zxing-cpp receive the raw image (theirs).

Run from the repo root:
    python -m benchmarks.benchmark
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2 as cv

import dmc
from tests.cases import CASES

IMAGES_DIR = Path(__file__).resolve().parent.parent / "test_images"


def decode_dmc(img, config):
    return [d.text for d in dmc.detect_and_decode(img, detector_config=config)]


def decode_pylibdmtx(img):
    from pylibdmtx import pylibdmtx
    return [b.data.decode("utf-8", "replace") for b in pylibdmtx.decode(img, timeout=3000)]


def decode_zxing(img):
    import zxingcpp
    return [b.text for b in zxingcpp.read_barcodes(img)]


DECODER_NAMES = ["dmc", "pylibdmtx", "zxing"]


def _run(fn, img):
    """Return (texts, elapsed_seconds). A decoder error -> empty result."""
    start = time.perf_counter()
    try:
        texts = list(fn(img))
    except Exception as exc:  # one bad image must not abort the whole run
        print(f"    ! {fn.__name__} raised {type(exc).__name__}: {exc}")
        texts = []
    return texts, time.perf_counter() - start


def _sanitize(text: str) -> str:
    return "".join(ch if ch.isprintable() else "·" for ch in text)


def _cell(label: str, texts: list[str]) -> str:
    if not texts:
        return "miss"
    if not label:
        return "decoded"              # unlabeled: payloads shown in detail section
    return "OK" if label in texts else "WRONG"


def main():
    labeled = 0
    correct = {n: 0 for n in DECODER_NAMES}
    decoded = {n: 0 for n in DECODER_NAMES}
    elapsed = {n: 0.0 for n in DECODER_NAMES}
    rows = []
    total = 0

    for case in CASES:
        img = cv.imread(str(IMAGES_DIR / case.image))
        if img is None:
            print(f"skip (unreadable): {case.image}")
            continue
        total += 1
        label = case.expected
        if label:
            labeled += 1

        cells = {}
        raw = {}
        for name in DECODER_NAMES:
            if name == "dmc":
                texts, dt = _run(lambda im: decode_dmc(im, case.config), img)
            elif name == "pylibdmtx":
                texts, dt = _run(decode_pylibdmtx, img)
            else:
                texts, dt = _run(decode_zxing, img)

            elapsed[name] += dt
            if texts:
                decoded[name] += 1
            if label and label in texts:
                correct[name] += 1
            cells[name] = _cell(label, texts)
            raw[name] = texts

        rows.append((case.image, label or "(unlabeled)", cells, raw))

    _print_report(rows, total, labeled, correct, decoded, elapsed)
    _print_payloads(rows)


def _print_report(rows, total, labeled, correct, decoded, elapsed):
    name_w = max(len(r[0]) for r in rows) if rows else 12
    label_w = max(len(r[1]) for r in rows) if rows else 11
    col = 11

    header = f"{'image':<{name_w}}  {'label':<{label_w}}  " + "".join(f"{n:<{col}}" for n in DECODER_NAMES)
    print("\n" + header)
    print("-" * len(header))
    for image, label, cells, _raw in rows:
        line = f"{image:<{name_w}}  {label:<{label_w}}  " + "".join(f"{cells[n]:<{col}}" for n in DECODER_NAMES)
        print(line)

    print("\nSummary")
    print("-" * 40)
    print(f"images: {total}   labeled: {labeled}")
    print(f"{'decoder':<12}{'decode-rate':<14}{'accuracy':<14}{'avg ms':<10}")
    for n in DECODER_NAMES:
        dr = f"{decoded[n]}/{total}"
        acc = f"{correct[n]}/{labeled}" if labeled else "n/a"
        avg_ms = elapsed[n] / total * 1000 if total else 0.0
        print(f"{n:<12}{dr:<14}{acc:<14}{avg_ms:<10.0f}")
    print("\n(dmc uses per-image tuned configs; references use the raw image.)")


def _print_payloads(rows):
    """Per-image raw payloads. Where >=2 decoders agree, that consensus is a
    reliable ground-truth label candidate for an unlabeled image."""
    print("\nDecoded payloads (consensus = likely true label)")
    print("-" * 60)
    for image, label, _cells, raw in rows:
        print(f"{image}  [label: {label}]")
        for n in DECODER_NAMES:
            texts = raw[n]
            shown = ", ".join(_sanitize(t) for t in texts) if texts else "—"
            print(f"    {n:<11}{shown}")


if __name__ == "__main__":
    main()
