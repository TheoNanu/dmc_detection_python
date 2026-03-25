import cv2 as cv
from dm_detector.pipeline import DataMatrixPipeline
from dm_decoder.grid_estimation.estimator import GridEstimator

def main():
    image_path = "./test_images/dmc_test_image.png"
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
            cv.imshow("2. rectified image (warped)", warped_bgr)

            warp_gray = cv.cvtColor(warped_bgr, cv.COLOR_BGR2GRAY)

            print("\ngrid estimator test:\n")
            estimator = GridEstimator(margin=1)

            pitch, score = estimator.estimate_pitch(warp_gray)

            if pitch is not None:
                print(f"estimated module size (pitch): {pitch:.2f} px")
                print(f"alternation validation score: {score:.2f}")

                margin = estimator.margin
                h, w = warp_gray.shape
                w_eff = w - 2 * margin
                h_eff = h - 2 * margin

                nx = int(round(w_eff / pitch))
                ny = int(round(h_eff / pitch))

                data_matrix = estimator.get_matrix_data(warp_gray, w / nx, h / ny, ny, nx)
                print(f"data matrix: {data_matrix}")
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

                estimator.draw_grid(warp_gray, h / ny, w / nx)

                cv.imshow("final grid", warp_gray)
                cv.waitKey(0)
                print(f"estimated matrix size: {nx} cols x {ny} rows")

                if score > 0.8:
                    print("good score")
                elif score < 0.6:
                    print("poor score")
            else:
                print("could not estimate pitch")

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()