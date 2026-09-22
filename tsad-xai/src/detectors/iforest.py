"""IForest through TSB-AD's class, fit-once / score-many (BRIEF B2-B4).

Source facts this file relies on are in docs/TSBAD_INTERNALS.md with line
numbers; the tests in tests/test_fit_score_consistency.py, test_padding.py and
test_wrapper_determinism.py re-measure them on every run.
"""
from __future__ import annotations

import time

import numpy as np

from src.detectors.base import DetectorError, validate_scores

_PATCHED = False


def _tsb_iforest_class():
    """Import TSB-AD's IForest, applying the sklearn>=1.6 shim if needed."""
    global _PATCHED
    import sklearn
    from TSB_AD.models.IForest import IForest
    major, minor = (int(v) for v in sklearn.__version__.split(".")[:2])
    if (major, minor) >= (1, 6) and not _PATCHED:
        # environment.yml pins <1.6; this is the documented fallback only.
        from src.compat import patch_tsb_ad
        patch_tsb_ad()
        _PATCHED = True
    return IForest


def _as_column(x: np.ndarray) -> np.ndarray:
    """TSB-AD's IForest needs (n, 1): `n_samples, n_features = X.shape` (IForest.py:181, :240)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2 and x.shape[1] == 1:
        return x
    if x.ndim != 1:
        raise DetectorError(f"expected a univariate series, got shape {x.shape}")
    if not np.isfinite(x).all():
        raise DetectorError("input contains NaN/inf")
    return x[:, None]


class IForestDetector:
    name = "IForest"

    def __init__(self, clf, config: dict):
        self._clf = clf
        self.config = config
        self.n_score_calls = 0

    @classmethod
    def fit(cls, x_fit: np.ndarray, *, seed: int, slidingWindow: int = 100, n_estimators: int = 200,
            max_features=1, n_jobs: int = 1, normalize: bool = False, constant_score: str = "error",
            **extra) -> "IForestDetector":
        if extra:
            raise TypeError(f"unknown IForest hyper-parameters: {sorted(extra)}")
        if normalize:
            # fit() z-scores the window matrix, decision_function() does not
            # (PyPI 1.5, IForest.py:185-189 vs :240-245): scores would disagree.
            raise ValueError("normalize=True is forbidden: TSB-AD 1.5 applies it in fit() only "
                             "(docs/TSBAD_INTERNALS.md §4)")
        if not isinstance(seed, (int, np.integer)):
            raise TypeError(f"seed must be an int, got {type(seed).__name__}")
        X = _as_column(x_fit)
        if X.shape[0] < slidingWindow:
            raise DetectorError(f"fit series length {X.shape[0]} < slidingWindow {slidingWindow}")
        IForest = _tsb_iforest_class()
        t0 = time.perf_counter()
        clf = IForest(slidingWindow=int(slidingWindow), n_estimators=int(n_estimators),
                      max_features=max_features, n_jobs=int(n_jobs),
                      random_state=int(seed), normalize=False)
        clf.fit(X)
        fit_s = time.perf_counter() - t0
        # config = everything needed to rebuild this exact model (BRIEF B2)
        config = {
            "detector": "IForest", "tsb_ad": "1.5(PyPI)",
            "slidingWindow": int(slidingWindow), "n_estimators": int(n_estimators),
            "max_features": max_features, "max_features_type": type(max_features).__name__,
            "n_jobs": int(n_jobs), "random_state": int(seed), "normalize": False,
            # class defaults not exposed, recorded so nobody has to guess (IForest.py:139-151)
            "max_samples": "auto", "contamination": 0.1, "bootstrap": False,
            "n_fit": int(X.shape[0]), "fit_runtime_s": fit_s, "constant_score": constant_score,
        }
        det = cls(clf, config)
        # B3 on the training scores too: a silent failure in fit is as bad as one in score
        validate_scores(clf.decision_scores_, X.shape[0], constant_score=constant_score)
        return det

    def score(self, x: np.ndarray) -> np.ndarray:
        """decision_function only — never refits (IForest.py:220-250)."""
        X = _as_column(x)
        if X.shape[0] < self.config["slidingWindow"]:
            raise DetectorError(f"series length {X.shape[0]} < slidingWindow {self.config['slidingWindow']}")
        self.n_score_calls += 1
        try:
            out = self._clf.decision_function(X)
        except Exception as e:
            raise DetectorError(f"decision_function failed: {type(e).__name__}: {e}") from e
        s, flags = validate_scores(out, X.shape[0], constant_score=self.config["constant_score"])
        self.last_flags = flags
        return s

    @property
    def train_scores(self) -> np.ndarray:
        """TSB-AD's own `decision_scores_` on the fit data (what run_IForest returns)."""
        return np.asarray(self._clf.decision_scores_, dtype=np.float64)
