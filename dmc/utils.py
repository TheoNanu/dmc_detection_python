import cv2 as cv
import numpy as np

def auto_canny(image: np.ndarray, percentile: float = 95.0):
    gx = cv.Sobel(image, cv.CV_16S, 1, 0, ksize=3)
    gy = cv.Sobel(image, cv.CV_16S, 0, 1, ksize=3)
    mag = np.abs(gx.astype(np.int32)) + np.abs(gy.astype(np.int32))  # L1, matches Canny's default norm
    upper = max(1.0, float(np.percentile(mag, percentile)))
    lower = 0.5 * upper
    return cv.Canny(image, lower, upper)

def valid_shape(rows: int, cols: int, valid_sizes: tuple[int, ...]):
    if rows != cols:
        return False

    if rows not in valid_sizes:
        return False

    return True