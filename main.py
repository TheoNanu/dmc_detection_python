import cv2 as cv
from dm_detector import DataMatrixPipeline

def main():
    # RUN ON IMAGE
    frame = cv.imread("./test_images/dmc_sample2.jpeg")

    cv.imshow("image", frame)
    cv.waitKey(0)

    pipeline = DataMatrixPipeline()

    print("Press 'q' to exit, 'd' to toggle debug view")
    debug_view = True

    results = pipeline.process_frame(frame)
    output = pipeline.draw_results(frame, results, debug_view)

    print(f"Results: {results}")

    if len(results) > 0:
        print(f"Found valid region: {results[0].is_valid}")

    if results and results[0].is_valid and debug_view:
        warped = results[0].get_rectified_image(frame)
        print(f"Warped image: {warped}")
        if warped is not None:
            cv.imshow("Rectified", warped)
            cv.waitKey(0)

    cv.imshow("Data Matrix Detector", output)
    cv.waitKey(0)

    # RUN ON CAMERA
    # cap = cv.VideoCapture(0)
    # cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    #
    # if not cap.isOpened():
    #     print("Error: Cannot access webcam")
    #     return
    #

    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #
    #     results = pipeline.process_frame(frame)
    #     output = pipeline.draw_results(frame, results, debug_view)
    #
    #     if results and results[0].is_valid and debug_view:
    #         warped = results[0].get_rectified_image(frame)
    #         if warped is not None:
    #             cv.imshow("Rectified", warped)
    #
    #     cv.imshow("Data Matrix Detector", output)
    #
    #     key = cv.waitKey(1) & 0xFF
    #     if key == ord('q'):
    #         break
    #     elif key == ord('d'):
    #         debug_view = not debug_view
    #         print(f"Debug view: {'ON' if debug_view else 'OFF'}")

    # cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()