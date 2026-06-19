import numpy as np

from config import DetectorConfig, DecoderConfig
from data import DetectionResult, Decoded
from dm_decoder.decoder import Decoder
from dm_detector.pipeline import DetectionPipeline


def detect(image: np.ndarray, config=DetectorConfig()) -> list[DetectionResult]:
    return DetectionPipeline(config).run(image)

def decode(image: np.ndarray, detection: DetectionResult, config=DecoderConfig()) -> Decoded | None:
    return Decoder(config).decode(image, detection)

def detect_and_decode(image: np.ndarray, detector_config=DetectorConfig(), decoder_config=DecoderConfig()) -> list[Decoded]:
    results = []

    for det in detect(image, detector_config):
        d = decode(image, det, decoder_config)
        if d is not None:
            results.append(d)

    return results