import cv2 as cv
from dm_detector.pipeline import DataMatrixPipeline
from dm_decoder.grid_estimation.estimator import GridEstimator

def main():
    image_path = "./test_images/dmc_on_object_test_image.png"
    frame = cv.imread(image_path)

    if frame is None:
        print(f"error: could not load image from {image_path}")
        return

    detector = DataMatrixPipeline()
    results = detector.process_frame(frame)

    output_frame = detector.draw_results(frame, results, debug_view=True)
    cv.imshow("1. detection", output_frame)

    if results and results[0].is_valid:
        warped_bgr = results[0].get_rectified_image(frame, output_size=400)

        if warped_bgr is not None:
            warp_gray = cv.cvtColor(warped_bgr, cv.COLOR_BGR2GRAY)

            # CLAHE is only used to make the preview easier to read. It must NOT
            # be fed to the grid estimator: on low-resolution / dot-peen codes it
            # amplifies the surface grain (~3 px) into a high-frequency signal
            # that the pitch detector locks onto instead of the true module
            # pitch, breaking grid estimation (e.g. dmc_on_object_test_image.png:
            # score 0.14 -> 1.0 once CLAHE is removed from the estimator input).
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cv.imshow("2. rectified image (warped)", clahe.apply(warp_gray))

            print("\ngrid estimator test:\n")
            estimator = GridEstimator(margin=1)
            h, w = warp_gray.shape

            # Primary: timing-pattern grid (sub-pixel, per-boundary module
            # positions). Falls back to the autocorrelation pitch when the
            # timing borders are too degraded to locate enough transitions.
            grid = estimator.estimate_grid(warp_gray)

            data_matrix = None
            if grid is not None:
                col_centres, row_centres = grid
                nx, ny = len(col_centres), len(row_centres)
                print(f"[timing] estimated matrix size: {nx} cols x {ny} rows")

                bits = estimator.sample_matrix(warp_gray, col_centres, row_centres)
                data_matrix = bits[1:-1, 1:-1]  # strip the 1-module border

                grid_vis = warp_gray.copy()
                estimator.draw_module_grid(grid_vis, col_centres, row_centres)
                grid_vis = estimator.draw_module_numbers(grid_vis, col_centres, row_centres)
                cv.imshow("final grid", grid_vis)
            else:
                pitch, score = estimator.estimate_pitch(warp_gray)
                if pitch is not None:
                    print(f"[autocorr] pitch={pitch:.2f} px score={score:.2f}")
                    w_eff = w - 2 * estimator.margin
                    h_eff = h - 2 * estimator.margin
                    nx = int(round(w_eff / pitch))
                    ny = int(round(h_eff / pitch))
                    data_matrix = estimator.get_matrix_data(warp_gray, w / nx, h / ny, ny, nx)
                    print(f"[autocorr] estimated matrix size: {nx} cols x {ny} rows")
                    estimator.draw_grid(warp_gray, h / ny, w / nx)
                    cv.imshow("final grid", warp_gray)
                else:
                    print("could not estimate grid")

            if data_matrix is not None:
                codewords = estimator.ecc200_codewords_from_data_modules(data_matrix)
                print(f"codewords: {codewords}")

                data_codewords = []
                for cw in codewords:
                    if cw == 129:  # padding, stop
                        break
                    if 1 <= cw <= 128:
                        data_codewords.append(chr(cw - 1))
                    elif 130 <= cw <= 229:
                        data_codewords.append(f"{cw - 130:02d}")
                    # ignore ECC codewords / mode switches for now

                print("".join(data_codewords))
                print(f"data codewords: {data_codewords}")
                cv.waitKey(0)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()