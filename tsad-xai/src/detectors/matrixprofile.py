"""Matrix Profile with fit/score separation (D-C5-3).

WHY THIS IS NOT TSB-AD's CLASS
    `TSB_AD/models/MatrixProfile.py` (PyPI 1.5) is 48 lines with a single method:

        def fit(self, X, y=None):
            self.profile = stumpy.stump(X.ravel(), m=self.window)      # :41
            res = np.zeros(len(X)); res.fill(self.profile[:, 0].min())
            res[self.window//2:-self.window//2+1] = self.profile[:, 0]  # :45
            self.decision_scores_ = res

    It is a **self-join** (no `T_B`), and it has **no `decision_function`** — so it can
    neither score a perturbed series against a fixed reference nor score without
    recomputing everything (BRIEF §7-4). The gate's pre-registered definition needs an
    AB-join against the training region, so we call stumpy directly and prove
    equivalence with TSB-AD's path in the self-join setting
    (`tests/test_matrixprofile.py`, rule 1 of D-C5-3).

MODES
    self_join  score(x) = stumpy.stump(x, m, ignore_trivial=True)
               Identical to TSB-AD, including the exclusion zone
               `ceil(m/4)` (stumpy/stump.py:723 with config.STUMPY_EXCL_ZONE_DENOM = 4,
               stumpy/config.py:19). x is never AB-joined against itself.
    ab_join    score(x) = stumpy.stump(x, m, T_B=reference, ignore_trivial=False)
               reference = the training prefix handed to fit(). No exclusion zone
               applies to an AB-join (stumpy/stump.py:726-728): every subsequence of x
               is matched against every subsequence of the reference.

PADDING (identical in both modes, copied from TSB-AD)
    profile[j] describes the window starting at j, i.e. [j, j+m); it is written to
    index j + m//2, and both ends are filled with the profile's minimum. Note
    `-m//2` parses as `(-m)//2`, which is what makes the slice the right length for
    odd m as well — `tests/test_matrixprofile.py` pins this for both parities.

SCORE ORIENTATION
    Higher = more anomalous: the value is the z-normalised Euclidean distance to the
    nearest neighbour, so a subsequence with no close match scores high. The padded
    ends carry the profile minimum (the *least* anomalous value), as upstream.
"""
from __future__ import annotations

import time

import numpy as np

from src.detectors.base import DetectorError, validate_scores

MODES = ("self_join", "ab_join")


def benchmark_window(x: np.ndarray, periodicity: int = 1) -> int:
    """The window TSB-AD's benchmark path uses for MatrixProfile.

    `model_wrapper.run_MatrixProfile` (PyPI 1.5 :78-84) does
    `slidingWindow = find_length_rank(data, rank=periodicity)` with
    `Optimal_Uni_algo_HP_dict['MatrixProfile'] = {'periodicity': 1}`
    (HP_list.py:244). Unlike IForest, MatrixProfile really does use it.
    """
    from TSB_AD.utils.slidingWindows import find_length_rank
    w = int(find_length_rank(np.asarray(x, dtype=np.float64).reshape(-1, 1), rank=int(periodicity)))
    if w < 3:
        raise DetectorError(f"find_length_rank returned {w}; stumpy needs m >= 3")
    return w


def _pad_like_tsb_ad(profile: np.ndarray, n: int, window: int) -> np.ndarray:
    """TSB_AD/models/MatrixProfile.py:43-45, reproduced exactly."""
    res = np.zeros(n, dtype=np.float64)
    res.fill(profile.min())
    res[window // 2:(-window) // 2 + 1] = profile
    return res


class MatrixProfileDetector:
    name = "MatrixProfile"

    def __init__(self, reference: np.ndarray | None, config: dict):
        self._reference = reference
        self.config = config
        self.n_score_calls = 0
        self.last_flags: list[str] = []

    @classmethod
    def fit(cls, x_fit: np.ndarray, *, seed: int = 0, window: int | None = None,
            mode: str = "ab_join", periodicity: int = 1, constant_score: str = "error",
            **extra) -> "MatrixProfileDetector":
        """`fit` stores the reference set; no model is trained (the algorithm is exact).

        mode="ab_join"   x_fit is the reference (the training prefix).
        mode="self_join" x_fit is ignored as a reference; score(x) self-joins x, which
                         is what TSB-AD does and what `fit_on=full` means here.
        `seed` is accepted for interface compatibility and recorded; Matrix Profile is
        deterministic, so it has no effect (the gate runs a single seed).
        """
        if extra:
            raise TypeError(f"unknown MatrixProfile hyper-parameters: {sorted(extra)}")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        x_fit = np.asarray(x_fit, dtype=np.float64).ravel()
        if not np.isfinite(x_fit).all():
            raise DetectorError("fit input contains NaN/inf")
        if window is None:
            window = benchmark_window(x_fit, periodicity)
        window = int(window)
        if window < 3:
            raise DetectorError(f"window {window} < 3 (stumpy minimum)")
        if mode == "ab_join" and x_fit.size < window:
            raise DetectorError(f"reference length {x_fit.size} < window {window}")
        t0 = time.perf_counter()
        reference = x_fit.copy() if mode == "ab_join" else None
        config = {
            "detector": "MatrixProfile", "backend": f"stumpy {_stumpy_version()}",
            "mode": mode, "window": window, "periodicity": int(periodicity),
            "exclusion_zone": int(np.ceil(window / 4)) if mode == "self_join" else 0,
            "normalize": True,            # stumpy z-normalises by definition
            "deterministic": True, "random_state": int(seed),
            "n_fit": int(x_fit.size), "fit_runtime_s": time.perf_counter() - t0,
            "constant_score": constant_score,
        }
        return cls(reference, config)

    def score(self, x: np.ndarray) -> np.ndarray:
        import stumpy
        x = np.asarray(x, dtype=np.float64).ravel()
        if not np.isfinite(x).all():
            raise DetectorError("input contains NaN/inf")
        w = self.config["window"]
        if x.size < w:
            raise DetectorError(f"series length {x.size} < window {w}")
        self.n_score_calls += 1
        try:
            if self.config["mode"] == "self_join":
                profile = stumpy.stump(x, m=w)
            else:
                profile = stumpy.stump(x, m=w, T_B=self._reference, ignore_trivial=False)
        except Exception as e:
            raise DetectorError(f"stumpy.stump failed: {type(e).__name__}: {e}") from e
        dist = np.asarray(profile[:, 0], dtype=np.float64)
        if dist.size != x.size - w + 1:
            raise DetectorError(f"profile length {dist.size} != n - m + 1 = {x.size - w + 1}")
        s, flags = validate_scores(_pad_like_tsb_ad(dist, x.size, w), x.size,
                                   constant_score=self.config["constant_score"])
        self.last_flags = flags
        return s

    def score_patch(self, x: np.ndarray, R: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """Scores of only the windows whose support meets R = [a, b)  (W2 rev1 §4 E0).

        Returns (J, scores): J are score indices in the padded convention above (window start
        + w//2), i.e. rev1 §2 `J(R) = [a + h - w + 1, b + h - 1]` clipped to the valid windows;
        scores[k] is the value `score(x)[J[k]]` would hold.

        stumpy path (1.14.1): `stumpy.stump(T_A = x[lo : hi + w], m = w, T_B = reference,
        ignore_trivial = False)` (stump.py:stump), lo/hi = first/last window start touched by R
        — the same call `score()` makes, on a slice. In an AB-join without an exclusion zone
        (stump.py:726-728) every T_A window is matched independently against T_B, so the slice
        gives the same values up to floating-point rounding: stumpy accumulates the covariance
        along each diagonal (stump.py:_compute_diagonal), and a different T_A start changes the
        accumulation path. rev1 E0 therefore compares against the full score with a tolerance
        and computes resp_before and resp_after both through this method.
        AB-join only: a self-join score of a window depends on the whole series.
        """
        import stumpy
        if self.config["mode"] != "ab_join":
            raise DetectorError("score_patch is defined for ab_join only")
        x = np.asarray(x, dtype=np.float64).ravel()
        w = self.config["window"]
        n = x.size
        a, b = int(R[0]), int(R[1])
        if not (0 <= a < b <= n):
            raise DetectorError(f"bad region [{a}, {b}) for n={n}")
        if n < w:
            raise DetectorError(f"series length {n} < window {w}")
        lo, hi = max(0, a - w + 1), min(b - 1, n - w)       # window starts touched by R
        seg = x[lo:hi + w]
        if not np.isfinite(seg).all():
            raise DetectorError("input contains NaN/inf in the patch")
        self.n_patch_calls = getattr(self, "n_patch_calls", 0) + 1
        try:
            out = stumpy.stump(seg, m=w, T_B=self._reference, ignore_trivial=False)
        except Exception as e:
            raise DetectorError(f"stumpy.stump failed: {type(e).__name__}: {e}") from e
        d = np.asarray(out[:, 0], dtype=np.float64)
        if d.size != hi - lo + 1 or not np.isfinite(d).all():
            raise DetectorError(f"patch profile length {d.size} != {hi - lo + 1} or non-finite")
        return np.arange(lo, hi + 1) + w // 2, d

    @property
    def reference(self) -> np.ndarray | None:
        return None if self._reference is None else self._reference.view()


def _stumpy_version() -> str:
    import stumpy
    return getattr(stumpy, "__version__", "unknown")
