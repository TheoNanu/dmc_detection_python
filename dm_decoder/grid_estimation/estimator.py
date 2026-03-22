import cv2 as cv
import numpy as np
from typing import Tuple, Optional

class GridEstimator:
    def __init__(self,
                 band_thickness: int = 11,
                 margin: int = 5,
                 hp_sigma: int = 9,
                 pitch_range: Tuple[int, int] = (3, 40)):
        self.k = band_thickness
        self.margin = margin
        self.hp_sigma = hp_sigma
        self.pitch_range = pitch_range

    def estimate_pitch(self, warp_gray: np.ndarray, off: int = 4) -> Tuple[Optional[float], float]:
        h, w = warp_gray.shape[:2]
        x0, x1 = self.margin, w - self.margin
        y0, y1 = off, min(h, off + self.k)

        prof = self._median_profile_from_band(warp_gray, y0, y1, x0, x1)
        hp = self._highpass_1d(prof)
        r = self._autocorr(hp)

        Lmin, Lmax = self.pitch_range
        Lmax = min(Lmax, len(r) - 1)

        search = np.abs(r[Lmin:Lmax + 1])
        if search.size == 0:
            return None, 0.0

        lag_abs_peak = int(np.argmax(search) + Lmin)

        candidates = [lag_abs_peak]
        if lag_abs_peak / 2 >= Lmin:
            candidates.append(int(lag_abs_peak / 2.0))

        best_pitch = None
        best_score = -1.0
        for p in candidates:
            score = self._transition_score_from_pitch(prof, p)
            if score > best_score:
                best_score = score
                best_pitch = float(p)

        return best_pitch, best_score

    @staticmethod
    def _median_profile_from_band(img: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        # cv.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 1)
        cv.imshow("original", img)
        print(f"x0: {x0}, y0: {y0}, x1: {x1}, y1: {y1}")
        band = img[y0:y1, x0:x1]
        visualize = cv.resize(band, (band.shape[1] * 4, band.shape[0] * 4))
        print(f"Original band: {band}")
        band = band.astype(np.float32)
        print(f"Casted band: {band}")
        cv.imshow("band", visualize)
        cv.waitKey(0)
        visualize = visualize.astype(np.float32)
        # cv.imshow("cast", visualize)
        # cv.waitKey(0)
        prof = np.median(band, axis=0)
        return prof

    def _highpass_1d(self, prof: np.ndarray) -> np.ndarray:
        prof = prof.astype(np.float32)
        prof -= prof.mean()

        p2 = prof.reshape(1, -1)
        trend = cv.GaussianBlur(p2, ksize=(0, 0), sigmaX=self.hp_sigma, borderType=cv.BORDER_REPLICATE).reshape(-1)
        # sa vad rezultatele
        hp = prof - trend

        s = hp.std()
        if s < 1e-6:
            return hp
        return hp / s

    @staticmethod
    def _autocorr(hp: np.ndarray) -> np.ndarray:
        r = np.correlate(hp, hp, mode='full').astype(np.float32)
        # sa vad rezultatul
        mid = len(r) // 2
        r = r[mid:]

        if r[0] > 1e-6:
            r /= r[0]
        return r

    @staticmethod
    def _transition_score_from_pitch(prof: np.ndarray, pitch_px: float) -> float:
        pitch_px = float(pitch_px)
        if pitch_px < 2:
            return 0.0

        n = len(prof)
        nb = int(n / pitch_px)
        if nb < 6:
            return 0.0

        vals = []
        for i in range(nb):
            a = int(round(i * pitch_px))
            b = int(round((i + 1) * pitch_px))
            if b <= a + 1:
                continue
            vals.append(float(np.mean(prof[a:b])))

        if len(vals) < 6:
            return 0.0

        vals = np.array(vals, dtype=np.float32)
        thr = np.median(vals)
        bits = (vals < thr).astype(np.uint8)

        transitions = np.sum(bits[1:] != bits[:-1])
        return transitions / max(1, (len(bits) - 1))

    def draw_grid(self, image: np.ndarray, horizontal_pitch: float, vertical_pitch: float) -> None:
        h, w = image.shape[:2]

        i = 1
        while True:
            y = int(round(i * horizontal_pitch))
            if y >= h:
                break
            cv.line(image, (0, y), (w, y), 255, 1)
            i += 1

        i = 1
        while True:
            x = int(round(i * vertical_pitch))
            if x >= w:
                break
            cv.line(image, (x, 0), (x, h), 255, 1)
            i += 1


    def get_matrix_data(self, image: np.ndarray, horizontal_pitch: float, vertical_pitch: float, rows: int,  cols: int) -> np.ndarray:
        result = np.zeros(shape=(rows - 2, cols - 2), dtype=np.uint8)

        for y in range(1, rows - 1):
            for x in range(1, cols - 1):
                module = image[int(y * vertical_pitch):int(y * vertical_pitch + vertical_pitch),
                                int(x * horizontal_pitch):int(x * horizontal_pitch + horizontal_pitch)]
                median = np.median(module)
                if median > 150:
                    result[y - 1][x - 1] = 0
                else:
                    result[y - 1][x - 1] = 1

        return result

    def ecc200_codewords_from_data_modules(self, bits):
        """
        bits: 2D array-like (numRows x numCols), data modules only (0/1).
        Returns: list[int] codewords (0..255) in standard ECC200 order.
        """
        bits = (np.asarray(bits) & 1).astype(np.uint8)
        numRows, numCols = bits.shape

        read = np.zeros((numRows, numCols), dtype=bool)

        def module(r, c):
            # Wrap-around rules (ECC200)
            if r < 0:
                r += numRows
                c += 4 - ((numRows + 4) % 8)
            if c < 0:
                c += numCols
                r += 4 - ((numCols + 4) % 8)
            r %= numRows
            c %= numCols
            read[r, c] = True
            return int(bits[r, c])

        def utah(r, c):
            # 8 bits → one codeword
            return [
                module(r - 2, c - 2),
                module(r - 2, c - 1),
                module(r - 1, c - 2),
                module(r - 1, c - 1),
                module(r - 1, c),
                module(r, c - 2),
                module(r, c - 1),
                module(r, c),
            ]

        def corner1():
            return [
                module(numRows - 1, 0),
                module(numRows - 1, 1),
                module(numRows - 1, 2),
                module(0, numCols - 2),
                module(0, numCols - 1),
                module(1, numCols - 1),
                module(2, numCols - 1),
                module(3, numCols - 1),
            ]

        def corner2():
            return [
                module(numRows - 3, 0),
                module(numRows - 2, 0),
                module(numRows - 1, 0),
                module(0, numCols - 4),
                module(0, numCols - 3),
                module(0, numCols - 2),
                module(0, numCols - 1),
                module(1, numCols - 1),
            ]

        def corner3():
            return [
                module(numRows - 3, 0),
                module(numRows - 2, 0),
                module(numRows - 1, 0),
                module(0, numCols - 2),
                module(0, numCols - 1),
                module(1, numCols - 1),
                module(2, numCols - 1),
                module(3, numCols - 1),
            ]

        def corner4():
            return [
                module(numRows - 1, 0),
                module(numRows - 1, numCols - 1),
                module(0, numCols - 3),
                module(0, numCols - 2),
                module(0, numCols - 1),
                module(1, numCols - 3),
                module(1, numCols - 2),
                module(1, numCols - 1),
            ]

        codewords = []

        r, c = 4, 0
        while (r < numRows) or (c < numCols):
            # Corner cases
            if r == numRows and c == 0:
                bits8 = corner1()
                codewords.append(self._bits_to_byte(bits8))
            if r == numRows - 2 and c == 0 and (numCols % 4) != 0:
                bits8 = corner2()
                codewords.append(self._bits_to_byte(bits8))
            if r == numRows - 2 and c == 0 and (numCols % 8) == 4:
                bits8 = corner3()
                codewords.append(self._bits_to_byte(bits8))
            if r == numRows + 4 and c == 2 and (numCols % 8) == 0:
                bits8 = corner4()
                codewords.append(self._bits_to_byte(bits8))

            # Sweep down-left
            while r >= 0 and c < numCols:
                if r < numRows and c >= 0 and not read[r, c]:
                    codewords.append(self._bits_to_byte(utah(r, c)))
                r -= 2
                c += 2
            r += 1
            c += 3

            # Sweep up-right
            while r < numRows and c >= 0:
                if r >= 0 and c < numCols and not read[r, c]:
                    codewords.append(self._bits_to_byte(utah(r, c)))
                r += 2
                c -= 2
            r += 3
            c += 1

        # Final fixed pattern (bottom-right), if not already read
        if not read[numRows - 1, numCols - 1]:
            # These two are the last bits in the stream
            b1 = module(numRows - 1, numCols - 1)
            b2 = module(numRows - 2, numCols - 2)
            # Append them by creating a final byte if you track bitstream;
            # in practice, many implementations have already consumed them via sweeps.
            # So we don't force-add a codeword here.

        return codewords

    def _bits_to_byte(self, bits8):
        v = 0
        for b in bits8:
            v = (v << 1) | (b & 1)
        return v

# sa verific valorile calculelor (debug)