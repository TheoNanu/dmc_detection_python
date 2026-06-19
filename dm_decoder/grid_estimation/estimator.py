import cv2 as cv
import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt


class GridEstimator:
    valid_size = [10, 12, 14, 16, 18, 20, 22, 24, 26, 32, 40, 44, 48, 52, 64, 72, 80, 88, 96, 104, 120, 132, 144]

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

    # ------------------------------------------------------------------
    # Timing-pattern grid estimation (sub-pixel, per-boundary).
    #
    # Instead of a single integer pitch, locate every module boundary from
    # the two timing borders (the alternating edges opposite the solid L).
    # In the rectified frame produced by get_rectified_image the solid L sits
    # on the BOTTOM and LEFT, so the TOP border gives the column boundaries
    # and the RIGHT border gives the row boundaries. Boundaries are the
    # sub-pixel zero-crossings of the high-passed border profile; module
    # centres are the midpoints between consecutive boundaries. This is
    # robust to a non-uniform/slightly-cut warp and to low module counts,
    # where autocorrelation gives only a coarse integer pitch.
    # ------------------------------------------------------------------
    def estimate_grid(self, warp_gray: np.ndarray, off: int = 4
                      ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return (col_centres, row_centres) sampling positions, or None if the
        timing borders are too degraded to locate enough boundaries."""
        h, w = warp_gray.shape[:2]

        col_bounds, raw_col_bounds = self._timing_boundaries(warp_gray, axis="x", off=off)
        row_bounds, raw_row_bounds = self._timing_boundaries(warp_gray, axis="y", off=off)
        print(f"[estimate-grid] col bounds: {col_bounds} raw col bounds: {raw_col_bounds} row bounds: {row_bounds} raw row bounds: {raw_row_bounds}")
        if col_bounds is None or row_bounds is None:
            return None

        return (self._boundaries_to_centres(col_bounds),
                self._boundaries_to_centres(row_bounds))

    def _timing_boundaries(self, img: np.ndarray, axis: str, off: int = 4
                           ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        h, w = img.shape[:2]
        if axis == "x":  # top border -> column boundaries, profile along x
            band = img[off:off + self.k, self.margin:w - self.margin]
            prof = np.median(band.astype(np.float32), axis=0)
        else:  # right border -> row boundaries, profile along y
            band = img[self.margin:h - self.margin, w - off - self.k:w - off]
            prof = np.median(band.astype(np.float32), axis=1)

        cv.imshow(f"band {axis}", cv.resize(band, dsize=None, fx=2.0, fy=2.0, interpolation=cv.INTER_NEAREST))
        cv.waitKey(0)
        cv.imwrite(f"band{axis}.png", band)
        print(f"Profile: {prof}")
        hp1, cr, pol, raw_cr = self._subpixel_transitions(prof, amp_window=5)
        print(f"[estimator] Initial number of transitions found: {cr.shape[0]}")

        # x = np.arange(0, hp1.shape[0])
        # fig, axs = plt.subplots(2, 1)
        # h, w = band.shape[:2]
        # if h > w:
        #     band_plt = band.T
        # else:
        #     band_plt = band
        # axs[0].imshow(cv.cvtColor(band_plt, cv.COLOR_GRAY2BGR))
        # axs[1].plot(x, hp1)
        # axs[1].set_xlim(0, x.shape[0])
        #
        # for c, r in zip(cr, raw_cr):
        #     axs[1].axvline(x=c, color="red", linestyle="--", linewidth=1.5)
        #     axs[1].axvline(x=r, color="green", linestyle="--", linewidth=1.5)
        #
        # axs[1].axhline(y=0, color="grey", linestyle="--", linewidth=1.5)
        # plt.show()

        cr, pol, raw_cr = self._reject_spurious(cr, pol, raw_cr)
        print(f"[estimator] Number of transitions after first filter: {cr.shape[0]}")
        print(f"[estimator] Transitions: {cr}")
        print(f"[estimator] Polarities: {pol}")

        # Terminal-polarity constraint. The timing pattern's corner modules fix
        # the colour change at each end, so the first and last *real* transition
        # have a known direction (and, since ECC200 sizes are even, the same one
        # at both ends). The top border (columns) ends black->white (+1); the
        # right border (rows) ends white->black (-1). A transition of the wrong
        # polarity at either end is a warp artifact (e.g. a sliver of background
        # left below the solid L) and is trimmed.
        required = +1 if axis == "x" else -1
        lo, hi = 0, len(cr)
        while hi - lo > 2 and pol[lo] != required:
            lo += 1
        while hi - lo > 2 and pol[hi - 1] != required:
            hi -= 1
        cr = cr[lo:hi]
        raw_cr = raw_cr[lo:hi]
        print(f"[estimator] lo: {lo} hi: {hi}")
        print(f"[estimator] Transitions: {cr}")

        print(f"[estimator] Number of transitions after polarity filter: {cr.shape[0]}")

        cr, raw_cr = self._regularize_transitions(cr, raw_cr)

        print(f"[estimator] Number of transitions after second filter: {cr.shape[0]}")

        x = np.arange(0, hp1.shape[0])
        fig, axs = plt.subplots(2, 1)
        h, w = band.shape[:2]
        if h > w:
            band_plt = band.T
        else:
            band_plt = band
        axs[0].imshow(cv.cvtColor(band_plt, cv.COLOR_GRAY2BGR))
        axs[1].plot(x, hp1)
        axs[1].set_xlim(0, x.shape[0])

        for c, r in zip(cr, raw_cr):
            axs[1].axvline(x=c, color="red", linestyle="--", linewidth=1.5)
            axs[1].axvline(x=r, color="green", linestyle="--", linewidth=1.5)

        axs[1].axhline(y=0, color="grey", linestyle="--", linewidth=1.5)
        plt.show()

        if len(cr) < 6:  # need a handful of modules for a meaningful grid
            return None, None
        return cr + self.margin, raw_cr + self.margin

    @staticmethod
    def _regularize_transitions(cr: np.ndarray, raw_cr: np.ndarray, lo_frac: float = 0.7,
                                sum_tol_frac: float = 0.4) -> Tuple[np.ndarray, np.ndarray]:
        """Repair module-split artifacts locally.

        A surface blob/noise in the timing band can leave a transition stuck
        mid-module, splitting one cell into two sub-pitch gaps. Such a pair of
        consecutive gaps is each below ``lo_frac`` of the pitch yet sums to ~one
        pitch — the in-between transition is the artifact and is dropped. The
        scan is local (re-derived each pass) so it tolerates the large gap
        variance of dot-peen codes without the drift of a fixed global grid.
        Clean codes are untouched (no sub-pitch gap pairs exist)."""
        cr = np.asarray(cr, dtype=float)
        raw_cr = np.asarray(raw_cr)
        changed = True
        while changed and len(cr) > 3:
            changed = False
            gaps = np.diff(cr)
            pitch = float(np.median(gaps))
            if pitch <= 0:
                break
            for i in range(len(gaps) - 1):
                # if 2 adjacent gaps(distance between 2 consecutive module boundaries(or zero-crossings)) are less
                # than the permitted fraction of the median distance of all the found gaps and their sum is in the
                # permitted threshold of the median distance then drop the zero-crossing(boundary) between them
                if (gaps[i] < lo_frac * pitch and gaps[i + 1] < lo_frac * pitch
                        and abs(gaps[i] + gaps[i + 1] - pitch) < sum_tol_frac * pitch):
                    cr = np.delete(cr, i + 1)  # drop the mid-module transition
                    raw_cr = np.delete(raw_cr, i + 1)
                    print(f"[estimator] regularize transitions deleted transition at: {i + 1}")
                    changed = True
                    break
        return cr, raw_cr

    @staticmethod
    def _subpixel_transitions(prof: np.ndarray,
                              min_amplitude: float = 0.3,
                              amp_window: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sub-pixel zero-crossings of the high-passed border profile/DoG style filtered border profile, zero-crossings
        represent edges in the original border profile — one per
        module boundary in an alternating timing pattern. Returns (positions,
        polarities) where polarity is +1 for a black->white edge (profile rising)
        and -1 for white->black (profile falling).

        A crossing is only accepted if the band swings to at least
        ``min_amplitude`` (in std units, since the band is normalized) within
        ``amp_window`` samples on *both* sides — a real module boundary produces
        ~±1.5 std lobes around its zero-crossing, while ripples in the flat
        quiet zone stay well below 0.1 std and would otherwise be picked up as
        transitions (and the first of them could never be removed by
        _reject_spurious, which always keeps the earliest crossing)."""
        prof = prof.astype(np.float32)
        prof = prof - prof.mean()
        trend = cv.GaussianBlur(prof.reshape(1, -1), (0, 0), sigmaX=9,
                                borderType=cv.BORDER_REPLICATE).reshape(-1)
        # remove low intensities around module boundaries
        hp = prof - trend
        # smooth the high-pass band, high-pass -> all the low intensities were removed, only high frequencies kept
        hp = cv.GaussianBlur(hp.reshape(1, -1), (0, 0), sigmaX=1.0,
                             borderType=cv.BORDER_REPLICATE).reshape(-1)
        # normalize for comparable amplitudes across different transitions
        s = hp.std()
        if s > 1e-6:
            hp = hp / s

        cr, pol, raw_cr = [], [], []

        print(f"High-pass filter: {hp}")
        for i in range(len(hp) - 1):
            # zero-crossings are found where there is a sign difference between 2 adjacent samples in the high-pass band
            # a zero-value in the current position can be a sign of a zero-crossing but also a sign of a flat surface
            # so the hp[i] != 0 tries to get rid of that ambiguity
            if hp[i] != 0 and hp[i] * hp[i + 1] < 0:
                # amplitude gate: require a real swing on both sides of the
                # crossing, not just a sign flip in near-zero noise
                left_peak = np.max(np.abs(hp[max(0, i - amp_window):i + 1]))
                right_peak = np.max(np.abs(hp[i + 1:i + 2 + amp_window]))
                print(f"[estimator] zero-crossing found at: {i}-{i+1} with amplitude: {left_peak}x{right_peak}")
                if min(left_peak, right_peak) < min_amplitude:
                    print(f"[estimator] Not passing the min amplitude gate")
                    continue
                # linear interpolation of the zero-crossing
                frac = abs(hp[i]) / (abs(hp[i]) + abs(hp[i + 1]))
                # store the sub-pixel position of the zero-crossing as the position where a sign difference has been
                # found + the interpolated fraction of the zero-crossing from the current position
                cr.append(i + frac)
                raw_cr.append(i)
                # hp[i] > 0 means profile falls through zero: white -> black.
                pol.append(-1 if hp[i] > 0 else +1)
        return hp, np.array(cr), np.array(pol, dtype=int), np.array(raw_cr)

    @staticmethod
    def _reject_spurious(cr: np.ndarray, pol: np.ndarray, raw_cr: np.ndarray,
                         min_frac: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Drop transitions closer than ``min_frac`` of the median spacing to
        their predecessor — these are noise crossings, not module boundaries."""
        if len(cr) < 3:
            return cr, pol, raw_cr
        spacing = float(np.median(np.diff(cr)))
        if spacing <= 0:
            return cr, pol, raw_cr
        keep_idx = [0]
        for i in range(1, len(cr)):
            if cr[i] - cr[keep_idx[-1]] >= min_frac * spacing:
                keep_idx.append(i)

        print(f"[estimator] reject spurious transitions kept: {keep_idx}")
        return cr[keep_idx], pol[keep_idx], raw_cr[keep_idx]

    @staticmethod
    def _boundaries_to_centres(b: np.ndarray) -> np.ndarray:
        """N internal boundaries -> N+1 module centres (outer two extrapolated
        with the adjacent spacing)."""
        centres = [b[0] - (b[1] - b[0]) / 2.0]
        for i in range(1, len(b)):
            centres.append((b[i - 1] + b[i]) / 2.0)
        centres.append(b[-1] + (b[-1] - b[-2]) / 2.0)
        return np.array(centres)

    def sample_matrix(self, img: np.ndarray, col_centres: np.ndarray,
                      row_centres: np.ndarray, win: int = 4) -> np.ndarray:
        """Sample a module bit at each (row, col) centre (1 = dark module).
        Threshold is Otsu over the per-module medians."""
        n_rows, n_cols = len(row_centres), len(col_centres)
        h, w = img.shape[:2]
        meds = np.zeros((n_rows, n_cols), dtype=np.float32)
        for r in range(n_rows):
            y = int(round(row_centres[r]))
            for c in range(n_cols):
                x = int(round(col_centres[c]))
                blk = img[max(0, y - win):y + win + 1, max(0, x - win):x + win + 1]
                meds[r, c] = np.median(blk) if blk.size else 255.0
        thr = cv.threshold(meds.astype(np.uint8), 0, 255,
                           cv.THRESH_BINARY + cv.THRESH_OTSU)[0]
        # Otsu's dark class is value <= thr; use <= so a thr of 0 (saturated,
        # heavily bimodal modules) still classifies the 0-valued dark modules.
        return (meds <= thr).astype(np.uint8)

    @staticmethod
    def _centres_to_boundaries(centres: np.ndarray) -> np.ndarray:
        """N module centres -> N+1 cell boundaries (midpoints between adjacent
        centres, with the two outer edges extrapolated)."""
        c = np.asarray(centres, dtype=float)
        inner = (c[:-1] + c[1:]) / 2.0
        first = c[0] - (c[1] - c[0]) / 2.0
        last = c[-1] + (c[-1] - c[-2]) / 2.0
        return np.concatenate([[first], inner, [last]])

    def draw_module_grid(self, image: np.ndarray, col_centres: np.ndarray,
                         row_centres: np.ndarray, color: int = 255) -> None:
        """Draw the grid lines on the module *boundaries* (not the centres), so
        each cell holds exactly one module."""
        h, w = image.shape[:2]
        for x in self._centres_to_boundaries(col_centres):
            cv.line(image, (int(round(x)), 0), (int(round(x)), h), color, 1)
        for y in self._centres_to_boundaries(row_centres):
            cv.line(image, (0, int(round(y))), (w, int(round(y))), color, 1)

    def draw_module_numbers(self, image: np.ndarray, col_centres: np.ndarray,
                            row_centres: np.ndarray, scale: float = 2.0,
                            color: Tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
        """Return a BGR copy of ``image`` (upscaled by ``scale`` for legibility)
        with each module's 1-based, row-major number drawn centred on its
        centre. Returns a new image because the upscale can't be done in place."""
        vis = cv.resize(image, None, fx=scale, fy=scale, interpolation=cv.INTER_NEAREST)
        if vis.ndim == 2:
            vis = cv.cvtColor(vis, cv.COLOR_GRAY2BGR)

        cc = np.asarray(col_centres, dtype=float) * scale
        rc = np.asarray(row_centres, dtype=float) * scale
        n_cols = len(cc)
        font, font_scale, thickness = cv.FONT_HERSHEY_SIMPLEX, 0.3, 1

        for i in range(len(rc)):
            for j in range(n_cols):
                text = str(i * n_cols + j + 1)
                (tw, th), _ = cv.getTextSize(text, font, font_scale, thickness)
                org = (int(round(cc[j] - tw / 2.0)), int(round(rc[i] + th / 2.0)))
                cv.putText(vis, text, org, font, font_scale, color, thickness, cv.LINE_AA)
        return vis

    @staticmethod
    def _median_profile_from_band(img: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        # cv.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 1)
        # cv.imshow("original", img)
        print(f"x0: {x0}, y0: {y0}, x1: {x1}, y1: {y1}")
        band = img[y0:y1, x0:x1]
        visualize = cv.resize(band, (band.shape[1] * 4, band.shape[0] * 4))
        print(f"Original band: {band}")
        band = band.astype(np.float32)
        print(f"Casted band: {band}")
        # cv.imshow("band", visualize)
        # cv.waitKey(0)
        visualize = visualize.astype(np.float32)
        # cv.imshow("cast", visualize)
        # cv.waitKey(0)
        prof = np.median(band, axis=0)
        return prof

    def _highpass_1d(self, prof: np.ndarray) -> np.ndarray:
        prof = prof.astype(np.float32)
        prof -= prof.mean()

        p2 = prof.reshape(1, -1)
        print(f"HIGHPASS 1D P2: {p2}")
        trend = cv.GaussianBlur(p2, ksize=(0, 0), sigmaX=self.hp_sigma, borderType=cv.BORDER_REPLICATE).reshape(-1)
        print(f"HIGHPASS 1D TREND: {trend}")
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

    def get_matrix_data(self, image: np.ndarray, horizontal_pitch: float, vertical_pitch: float, rows: int,
                        cols: int) -> np.ndarray:
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
