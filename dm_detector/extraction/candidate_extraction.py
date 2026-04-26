import cv2 as cv
import numpy as np
from typing import List, Tuple

class CandidateExtraction:
    def __init__(self,
                 canny_t1: int = 100,
                 canny_t2: int = 200,
                 min_area: float = 300.0,
                 min_perimeter: float = 500.0,
                 padding: int = 10,
                 min_children: int = 5):

        self.canny_t1 = canny_t1
        self.canny_t2 = canny_t2
        self.min_area = min_area
        self.min_perimeter = min_perimeter
        self.padding = padding
        self.min_children = min_children

    @staticmethod
    def _auto_canny(image: np.ndarray, sigma: float = 0.33) -> np.ndarray:
        v = float(np.median(image))
        lower = int(max(0.0, (1.0 - sigma) * v))
        upper = int(min(255.0, (1.0 + sigma) * v))
        print(f"Canny lower: {lower} upper: {upper}")
        return cv.Canny(image, lower, upper)

    def edge_detection(self, image_gray: np.ndarray) -> np.ndarray:
        return self._auto_canny(image_gray)

    @staticmethod
    def morphological_processing(edges: np.ndarray) -> np.ndarray:
        edges_copy = edges.copy()
        edges_copy = cv.bitwise_not(edges_copy)

        # cv.imshow("edges copy", edges_copy)
        # cv.waitKey(0)

        kernel_dilate = cv.getStructuringElement(cv.MORPH_RECT, (4, 4))
        dilated = cv.dilate(edges_copy, kernel_dilate, iterations=1)

        kernel_open = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        processed = cv.morphologyEx(dilated, cv.MORPH_OPEN, kernel_open)

        return processed

    def contour_analysis(self, original_image: np.ndarray, binary_map: np.ndarray, shape: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
        contours, hierarchy = cv.findContours(binary_map, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        candidate_boxes = []
        img_h, img_w = shape

        if hierarchy is None:
            return []

        hierarchy = hierarchy[0]

        img_copy = original_image.copy()

        bgr_img = cv.cvtColor(binary_map, cv.COLOR_GRAY2BGR)

        for i, contour in enumerate(contours):
            cv.drawContours(img_copy, [contour], 0, (0, 255, 0), 2)

            perimeter = cv.arcLength(contour, True)
            area = cv.contourArea(contour)

            child_count = 0
            i_first_child = hierarchy[i][2]

            if i_first_child != -1:
                current = i_first_child
                while current != -1:
                    child_count += 1
                    current = hierarchy[current][0]

            if perimeter > self.min_perimeter and area > self.min_area:
                x, y, w, h = cv.boundingRect(contour)

                x_new = max(0, x - self.padding)
                y_new = max(0, y - self.padding)
                w_new = min(img_w - x_new, w + 2 * self.padding)
                h_new = min(img_h - y_new, h + 2 * self.padding)

                print(f"[extraction] contour {i}: accepted box ({x_new},{y_new},{w_new},{h_new}) perim={perimeter:.0f} area={area:.0f}")
                candidate_boxes.append((x_new, y_new, w_new, h_new))

        return candidate_boxes

    def get_candidates(self, frame: np.ndarray) -> list:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (11, 11), 1.8)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        gray = cv.adaptiveThreshold(
            enhanced, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv.THRESH_BINARY, 65, 4
        )

        preprocess = self.morphological_processing(gray)

        cv.imshow("clahe", preprocess)
        cv.waitKey(0)

        candidates = self.contour_analysis(frame, preprocess, gray.shape)

        return candidates