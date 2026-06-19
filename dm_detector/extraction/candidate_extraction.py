import cv2 as cv
import numpy as np
from typing import List, Tuple

from debug import DebugSink, NullSink


class CandidateExtraction:
    def __init__(self,
                 canny_t1: int = 100,
                 canny_t2: int = 200,
                 min_area: float = 300.0,
                 min_perimeter: float = 500.0,
                 padding: int = 10,
                 min_children: int = 5,
                 debug: DebugSink = NullSink()):

        self.canny_t1 = canny_t1
        self.canny_t2 = canny_t2
        self.min_area = min_area
        self.min_perimeter = min_perimeter
        self.padding = padding
        self.min_children = min_children
        self.debug = debug

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
    def morphological_processing(edges: np.ndarray, dilate_kernel_size: int = 4, open_kernel_size: int = 5) -> np.ndarray:
        edges_copy = edges.copy()
        edges_copy = cv.bitwise_not(edges_copy)

        # cv.imshow("edges copy", edges_copy)
        # cv.waitKey(0)

        kernel_open = cv.getStructuringElement(cv.MORPH_RECT, (open_kernel_size, open_kernel_size))
        processed = cv.morphologyEx(edges_copy, cv.MORPH_OPEN, kernel_open)

        kernel_dilate = cv.getStructuringElement(cv.MORPH_RECT, (dilate_kernel_size, dilate_kernel_size))
        dilated = cv.dilate(processed, kernel_dilate, iterations=1)

        return dilated

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

                self.debug.log(f"[extraction] contour {i}: accepted box ({x_new},{y_new},{w_new},{h_new}) perim={perimeter:.0f} area={area:.0f}")
                candidate_boxes.append((x_new, y_new, w_new, h_new))

        return candidate_boxes

    @staticmethod
    def cluster_high_overlap_candidates(candidates: List[Tuple[int, int, int, int]],
                                        overlap_threshold: float = 0.5, gap: int = 0) -> List[Tuple[int, int, int, int]]:
        """Merge candidate boxes whose overlap coefficient — intersection area
        over the *smaller* box's area — reaches ``overlap_threshold``. Unlike
        IoU, this catches a small box (partially) contained in a bigger one
        regardless of their size difference. Each box absorbs every box it
        overlaps into a growing union; passes repeat until stable, so chains of
        fragments (e.g. a DMC shattered into many small boxes on a noisy
        surface) merge transitively. Boxes that overlap nothing are kept as-is.

        With ``gap > 0``, boxes that overlap *or* sit within ``gap`` pixels of
        each other are merged regardless of the overlap coefficient (proximity
        clustering — what reassembles a fragmented DMC, whose fragment boxes
        are adjacent rather than overlapping). With ``gap = 0`` only the
        overlap-coefficient criterion applies (redundancy removal).
        """
        clustered = [tuple(c) for c in candidates]

        changed = True
        while changed:
            changed = False
            merged = []
            used = [False] * len(clustered)

            for i in range(len(clustered)):
                if used[i]:
                    continue
                x1, y1, w, h = clustered[i]
                x2, y2 = x1 + w, y1 + h

                for j in range(i + 1, len(clustered)):
                    if used[j]:
                        continue
                    bx1, by1, bw, bh = clustered[j]
                    bx2, by2 = bx1 + bw, by1 + bh

                    iw = min(x2, bx2) - max(x1, bx1)
                    ih = min(y2, by2) - max(y1, by1)

                    # proximity criterion: overlapping or within `gap` px
                    near = gap > 0 and iw > -gap and ih > -gap

                    # overlap criterion: intersection covers enough of the
                    # smaller box
                    overlap_ok = False
                    if iw > 0 and ih > 0:
                        smaller_area = min((x2 - x1) * (y2 - y1), bw * bh)
                        overlap_ok = (smaller_area > 0
                                      and iw * ih / smaller_area >= overlap_threshold)

                    if not (near or overlap_ok):
                        continue

                    # absorb box j into the accumulator and keep scanning, so
                    # later boxes are tested against the grown union
                    x1, y1 = min(x1, bx1), min(y1, by1)
                    x2, y2 = max(x2, bx2), max(y2, by2)
                    used[j] = True
                    changed = True

                merged.append((x1, y1, x2 - x1, y2 - y1))

            # every pass that merges strictly shrinks the list, so this terminates
            clustered = merged

        return clustered

    def get_candidates(self, frame: np.ndarray) -> list:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (11, 11), 0.8)
        # clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # enhanced = clahe.apply(gray)
        gray = cv.adaptiveThreshold(
            gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv.THRESH_BINARY, 65, 10
        )

        preprocess = self.morphological_processing(gray, open_kernel_size=4, dilate_kernel_size=5)

        self.debug.show("clahe", preprocess)
        self.debug.pause()

        candidates = self.contour_analysis(frame, preprocess, gray.shape)

        candidates = self.cluster_high_overlap_candidates(candidates, 0.95, 30)

        return candidates