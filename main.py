import cv2 as cv
from dm_detector.pipeline import DataMatrixPipeline
from dm_decoder.grid_estimation.estimator import GridEstimator

# def main():
#     valid_size = [10, 12, 14, 16, 18, 20, 22, 24, 26, 32, 40, 44, 48, 52, 64, 72, 80, 88, 96, 104, 120, 132, 144]
#     filename = "dmc_on_object_test_image.png"
#     image_path = f"test_images/{filename}"
#     # image_path = r"C:\Users\Marian\Pictures\17804976506535.JPG"
#     frame = cv.imread(image_path)
#
#     if frame is None:
#         print(f"error: could not load image from {image_path}")
#         return
#
#     detector = DataMatrixPipeline()
#     results = detector.process_frame(frame)
#
#     # output_frame = detector.draw_results(frame, results, debug_view=True)
#     # cv.imshow("1. detection", output_frame)
#
#     output_frame = frame.copy()
#
#     if results:
#         for res in results:
#             if res.is_valid:
#                 warped_bgr = res.get_rectified_image(frame, output_size=400)
#
#                 if warped_bgr is not None:
#                     warp_gray = cv.cvtColor(warped_bgr, cv.COLOR_BGR2GRAY)
#                     # CLAHE is only used to make the preview easier to read. It must NOT
#                     # be fed to the grid estimator: on low-resolution / dot-peen codes it
#                     # amplifies the surface grain (~3 px) into a high-frequency signal
#                     # that the pitch detector locks onto instead of the true module
#                     # pitch, breaking grid estimation (e.g. dmc_on_object_test_image.png:
#                     # score 0.14 -> 1.0 once CLAHE is removed from the estimator input).
#                     clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#                     # warp_gray = clahe.apply(warp_gray)
#
#                     # warp_gray = cv.GaussianBlur(warp_gray,(11, 11), 0)
#                     warp_gray = cv.medianBlur(warp_gray, 11)
#
#
#                     cv.imshow("2. rectified image (warped)", warp_gray)
#                     # warp_gray = clahe.apply(warp_gray)
#
#                     print("\ngrid estimator test:\n")
#                     estimator = GridEstimator(margin=1)
#                     h, w = warp_gray.shape
#
#                     # Primary: timing-pattern grid (sub-pixel, per-boundary module
#                     # positions). Falls back to the autocorrelation pitch when the
#                     # timing borders are too degraded to locate enough transitions.
#                     grid = estimator.estimate_grid(warp_gray)
#
#                     data_matrix = None
#                     if grid is not None:
#                         col_centres, row_centres = grid
#                         nx, ny = len(col_centres), len(row_centres)
#                         print(f"[timing] estimated matrix size: {nx} cols x {ny} rows")
#
#                         bits = estimator.sample_matrix(warp_gray, col_centres, row_centres)
#                         data_matrix = bits[1:-1, 1:-1]  # strip the 1-module border
#
#                         grid_vis = warp_gray.copy()
#                         estimator.draw_module_grid(grid_vis, col_centres, row_centres)
#                         grid_vis = estimator.draw_module_numbers(grid_vis, col_centres, row_centres)
#                         cv.imshow("final grid", grid_vis)
#                     else:
#                         pitch, score = estimator.estimate_pitch(warp_gray)
#                         if pitch is not None:
#                             print(f"[autocorr] pitch={pitch:.2f} px score={score:.2f}")
#                             w_eff = w - 2 * estimator.margin
#                             h_eff = h - 2 * estimator.margin
#                             nx = int(round(w_eff / pitch))
#                             ny = int(round(h_eff / pitch))
#                             data_matrix = estimator.get_matrix_data(warp_gray, w / nx, h / ny, ny, nx)
#                             print(f"[autocorr] estimated matrix size: {nx} cols x {ny} rows")
#                             estimator.draw_grid(warp_gray, h / ny, w / nx)
#                             cv.imshow("final grid", warp_gray)
#                         else:
#                             print("could not estimate grid")
#
#                     if data_matrix is not None and (data_matrix.shape[0] in valid_size and data_matrix.shape[0] == data_matrix.shape[1]):
#                         codewords = estimator.ecc200_codewords_from_data_modules(data_matrix)
#                         print(f"codewords: {codewords}")
#
#                         data_codewords = []
#                         for cw in codewords:
#                             if cw == 129:  # padding, stop
#                                 break
#                             if 1 <= cw <= 128:
#                                 data_codewords.append(chr(cw - 1))
#                             elif 130 <= cw <= 229:
#                                 data_codewords.append(f"{cw - 130:02d}")
#                             # ignore ECC codewords / mode switches for now
#
#                         print("".join(data_codewords))
#                         print(f"data codewords: {data_codewords}")
#                         output_frame = detector.draw_dmc(output_frame, res.precise_location, "".join(data_codewords))
#                         cv.imshow("detected dmc", output_frame)
#                         cv.waitKey(0)
#
#     cv.imwrite(f"{filename.split(".")[0]}_result.png", output_frame)
#
#     cv.waitKey(0)
#     cv.destroyAllWindows()

def main():
    valid_size = [10, 12, 14, 16, 18, 20, 22, 24, 26, 32, 40, 44, 48, 52, 64, 72, 80, 88, 96, 104, 120, 132, 144]
    filename = "dmc_sample.jpg"
    image_path = f"test_images/{filename}"
    # image_path = r"C:\Users\Marian\Pictures\17804976506535.JPG"
    frame = cv.imread(image_path)

    detector = DataMatrixPipeline()
    results = detector.detect(frame,
                              smoothing=1,
                              noisy_surface=False,
                              canny_percentile=90.0,
                              border_fitter_gaussian_size = 3,
                              fitter_dilate_size = 5,
                              fitter_blob_removal_min_area = 0,
                              fitter_win_in = 20,
                              fitter_win_out = 20,
                              fitter_ransac_max_pts_outside = 10,
                              fitter_ransac_inlier_threshold_dist = 0.9)

    output_frame = frame.copy()

    if results:
        for res in results:
            print(res)

    # cv.imwrite(f"{filename.split(".")[0]}_result.png", output_frame)
    #
    # cv.waitKey(0)
    # cv.destroyAllWindows()


if __name__ == "__main__":
    main()