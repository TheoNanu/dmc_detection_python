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
        band = img[y0:y1, x0:x1].astype(np.float32)
        prof = np.median(band, axis=0)
        return prof

    def _highpass_1d(self, prof: np.ndarray) -> np.ndarray:
        prof = prof.astype(np.float32)
        prof -= prof.mean()

        p2 = prof.reshape(1, -1)
        trend = cv.GaussianBlur(p2, ksize=(0, 0), sigmaX=self.hp_sigma, borderType=cv.BORDER_REPLICATE).reshape(-1)
        # sa vad rezultatele pentru debug
        hp = prof - trend

        s = hp.std()
        if s < 1e-6:
            return hp

        return hp / s

    @staticmethod
    def _autocorr(hp: np.ndarray) -> np.ndarray:
        r = np.correlate(hp, hp, mode='full').astype(np.float32)
        # sa vad rezultatul pentru debug
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

# impartire pe grid si desenarea lor pe cod data matrix pentru vizualizare
# extragerea datelor (impartirea modulelor alb si negru in biti 0 si 1)