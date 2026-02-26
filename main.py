import cv2 as cv
from dm_detector.pipeline import DataMatrixPipeline
from dm_decoder.grid_estimation.estimator import GridEstimator
from dm_decoder.sampling.sampler import ModuleSampler

def snap_to_valid_size(estimated_n: int) -> int:
    valid_sizes = [10, 12, 14, 16, 18, 20, 22, 24, 26, 32, 36, 40, 44, 48, 52, 64, 72, 80, 88, 96, 104, 120, 132, 144]
    return min(valid_sizes, key=lambda x: abs(x - estimated_n))

def main():
    image_path = "./test_images/dmc_sample2.jpeg"
    frame = cv.imread(image_path)

    if frame is None:
        print(f"error: could not load image from {image_path}")
        return

    detector = DataMatrixPipeline()
    results = detector.process_frame(frame)

    output_frame = detector.draw_results(frame, results)
    cv.imshow("1. detection", output_frame)

    if results and results[0].is_valid:
        warped_bgr = results[0].get_rectified_image(frame, output_size=400)

        if warped_bgr is not None:
            cv.imshow("2. rectified image (warped)", warped_bgr)

            warp_gray = cv.cvtColor(warped_bgr, cv.COLOR_BGR2GRAY)

            print("\ngrid estimator test:\n")

            estimator = GridEstimator()

            pitch, score = estimator.estimate_pitch(warp_gray)

            if pitch is not None:
                h, w = warp_gray.shape

                nx = int(round(w / pitch))
                ny = int(round(h / pitch))

                nx_snapped = snap_to_valid_size(nx)
                ny_snapped = snap_to_valid_size(ny)

                final_pitch_x = w / nx_snapped
                final_pitch_y = h / ny_snapped

                print(f"estimated module size (initial pitch): {pitch:.2f} px")
                print(f"corrected pitch after snap: {final_pitch_x:.2f} px")
                print(f"snapped grid size: {nx_snapped} cols x {ny_snapped} rows")
                print(f"alternation validation score: {score:.2f}")

                if score > 0.8:
                    print("good score")
                elif score < 0.6:
                    print("bad score")

                sampler = ModuleSampler()

                roi_color = warped_bgr.copy()
                sampler.draw_grid(roi_color, horizontal_pitch=final_pitch_x, vertical_pitch=final_pitch_y)
                cv.imshow("3. grid visualization", roi_color)

                data_matrix = sampler.get_matrix_data(
                    image=warp_gray,
                    horizontal_pitch=final_pitch_x,
                    vertical_pitch=final_pitch_y,
                    rows=ny_snapped,
                    cols=nx_snapped
                )

                print(f"data matrix size (without borders): {data_matrix.shape[1]}x{data_matrix.shape[0]}")
                print("binary data region preview (0=white, 1=black):\n")

                for row in data_matrix:
                    row_str = "".join(["1 " if val == 1 else "0 " for val in row])
                    print(row_str)
            else:
                print("could not estimate pitch")

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()