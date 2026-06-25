import time

import cv2 as cv

import dmc
from dmc import viz, DetectorConfig, BorderFitterConfig, CvDebugSink, NullSink


def main():
    filename = "synthetic_image5.png"
    image_path = f"test_images/{filename}"
    frame = cv.imread(image_path)

    if frame is None:
        print(f"error: could not load image from {image_path}")
        return

    start = time.time()

    results = dmc.detect_and_decode(frame,
                                    detector_config=DetectorConfig(
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
                                    ),
                                    debug=CvDebugSink())

    print(f"Execution time: {time.time() - start}")

    if not results:
        print("No Data Matrix codes found")
        return

    output = frame.copy()
    for decoded in results:
        rows, cols = decoded.matrix.shape
        print(f"decoded: {decoded.text!r}  ({rows}x{cols} modules)")
        output = viz.draw_dmc(output, decoded.detection.precise_location, decoded.text)

    out_path = f"{filename.rsplit('.', 1)[0]}_result.png"
    cv.imwrite(out_path, output)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
