import numpy as np

from dmc.debug import NullSink, DebugSink
from dmc.config import DetectorConfig, DecoderConfig
from dmc.data import DetectionResult, Decoded
from dmc.decoder.decoder import Decoder
from dmc.detector.pipeline import DetectionPipeline


def detect(image: np.ndarray, config: DetectorConfig = DetectorConfig(), debug: DebugSink = NullSink()) -> list[DetectionResult]:
    """
    Detect DMCs in numpy BGR images.

    Args:
        image (np.ndarray): Input image.
        config (DetectorConfig): Configuration object for detection pipeline.
        debug (DebugSink): Debug object.

    Returns:
        list[DetectionResult]: List of DetectionResult objects for each DMC found.
    """
    return DetectionPipeline(config, debug).run(image)

def decode(image: np.ndarray, detection: DetectionResult, config: DecoderConfig = DecoderConfig(), debug: DebugSink = NullSink()) -> Decoded | None:
    """
    Decode DMCs in image having its information as DetectionResult object.

    Args:
        image (np.ndarray): Input image.
        detection (DetectionResult): Object returned by detect method.
        config (DecoderConfig): Configuration object for decoding pipeline.
        debug (DebugSink): Debug object.

    Returns:
        Decoded: Object that contains fields regarding DMC's data and location in the image.
    """
    return Decoder(config, debug).decode(image, detection)

def detect_and_decode(image: np.ndarray, detector_config: DetectorConfig = DetectorConfig(), decoder_config: DecoderConfig = DecoderConfig(),
                      debug: DebugSink = NullSink()) -> list[Decoded]:
    """
    Detect and decode DMCs in numpy BGR images.

    Args:
        image (np.ndarray): Input image.
        detector_config (DetectorConfig): Configuration object for detection pipeline.
        decoder_config (DecoderConfig): Configuration object for decoding pipeline.
        debug (DebugSink): Debug object.

    Returns:
        list[Decoded]: List of objects that contain fields regarding DMC's data and location in the image.
    """
    results = []

    for det in detect(image, detector_config, debug):
        d = decode(image, det, decoder_config, debug)
        if d is not None:
            results.append(d)

    return results