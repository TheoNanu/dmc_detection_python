import time

import cv2 as cv

import dmc
from dmc import viz, DetectorConfig, BorderFitterConfig, CvDebugSink, NullSink


def main():
    filename = "dmc_on_object_test_image.png"
    image_path = f"test_images/{filename}"
    frame = cv.imread(image_path)

    if frame is None:
        print(f"error: could not load image from {image_path}")
        return

    start = time.time()

    results = dmc.detect_and_decode(frame, detector_config=DetectorConfig(smoothing=1,
                                                                          noisy_surface=False,
                                                                          canny_percentile=90.0,
                                                                          border_fitter_config=BorderFitterConfig(
                                                                              gaussian_size=3,
                                                                              dilate_size=5,
                                                                              blob_min_area=0,
                                                                              win_out=20,
                                                                              win_in=20,
                                                                              ransac_max_pts_outside=10,
                                                                              ransac_inlier_threshold=0.9
                                                                          )),
                                    debug=NullSink())

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
