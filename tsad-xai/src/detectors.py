"""Detector wrapper providing a retrain-free scoring function f(X) -> scores.

WHY THIS FILE EXISTS
--------------------
TSB_AD.model_wrapper.run_Unsupervise_AD() constructs a fresh detector and calls
clf.fit(data) on EVERY invocation. Verified by reading TSB_AD/model_wrapper.py:

    def run_IForest(data, slidingWindow=100, ...):
        clf = IForest(...)
        clf.fit(data)
        return clf.decision_scores_.ravel()

Stage 3/5/7 need thousands of scored perturbations. Using run_Unsupervise_AD
would retrain thousands of times. Measured on L=3000, IForest:
    fit                = 0.154 s
    decision_function  = 0.020 s      -> 7.7x, and it grows with model cost.

So we fit once and call decision_function() thereafter.

THREE VERIFIED TRAPS
--------------------
1. run_Unsupervise_AD swallows every exception with a bare `except:` and
   RETURNS A STRING instead of raising. Never use it in a loop.

2. IForest.fit() applies zscore normalisation to the windowed matrix, but
   IForest.decision_function() DOES NOT. With the default normalize=True the
   two disagree (measured max abs difference 0.0368 on our synthetic series).
   We therefore force normalize=False, which makes them identical
   (measured max abs difference 0.0).

3. sklearn>=1.6 breaks decision_function via check_is_fitted. See src/compat.py.

4. KMeansAD HAS NO decision_function. It is a bare
   sklearn BaseEstimator/OutlierMixin, and its scoring entry point is
   predict(). Calling decision_function() raises
       AttributeError: 'KMeansAD' object has no attribute 'decision_function'
   and, because SUPPORTED advertised the name, the failure only appeared when
   a second detector was first used (the R5 screen). predict() is fit-free --
   it calls self.model.predict on the fitted KMeans -- so fit-once/score-many
   still holds; only the method name differs.
   KMeansAD is also NOT subject to trap 2: _preprocess_data() applies the same
   zscore in fit and in predict, so the two agree by construction.
   It does print three lines on EVERY predict() call. Stage 3 makes ~700 calls
   per series, so scoring suppresses stdout.

SCORE CONVENTION (verified)
---------------------------
  * length == len(X); short window scores are edge-padded by TSB-AD itself
        ceil((w-1)/2) copies of the first value at the front
        (w-1)//2      copies of the last value at the back
  * higher == more anomalous (TSB-AD applies invert_order internally)
"""
import os
from contextlib import redirect_stdout

import numpy as np

from .compat import patch_tsb_ad

_PATCHED = False


def _ensure_patched():
    global _PATCHED
    if not _PATCHED:
        patch_tsb_ad()
        _PATCHED = True


class ScoreFn:
    """Fit once; score many times without refitting.

    Parameters
    ----------
    name : {'IForest', 'KMeansAD', 'AutoEncoder'}
    train : (L, C) array. For semi-supervised detectors this must be
            normal-only data. TSB-AD ships train splits that are normal-only.
    """

    SUPPORTED = ("IForest", "KMeansAD", "AutoEncoder")

    def __init__(self, name, train, window=100, **kwargs):
        if name not in self.SUPPORTED:
            raise ValueError(f"{name} not in {self.SUPPORTED}")
        _ensure_patched()
        self.name = name
        self.window = window
        train = np.asarray(train, float)
        if train.ndim == 1:
            train = train[:, None]

        if name == "IForest":
            from TSB_AD.models.IForest import IForest
            # normalize=False is REQUIRED -- see trap 2 in the module docstring.
            self.clf = IForest(slidingWindow=window, normalize=False,
                               n_estimators=kwargs.get("n_estimators", 100),
                               max_features=kwargs.get("max_features", 1),
                               n_jobs=kwargs.get("n_jobs", 1))
        elif name == "KMeansAD":
            from TSB_AD.models.KMeansAD import KMeansAD
            self.clf = KMeansAD(k=kwargs.get("k", 20), window_size=window,
                                stride=kwargs.get("stride", 1),
                                n_jobs=kwargs.get("n_jobs", 1))
        else:
            from TSB_AD.models.AE import AutoEncoder
            self.clf = AutoEncoder(slidingWindow=window,
                                   hidden_neurons=kwargs.get("hidden_neurons", [64, 32]),
                                   batch_size=kwargs.get("batch_size", 128),
                                   epochs=kwargs.get("epochs", 50))
        self.clf.fit(train)
        self._n_calls = 0
        # See trap 4: KMeansAD scores through predict(), not decision_function.
        self._score_attr = "predict" if name == "KMeansAD" else "decision_function"
        self._quiet = name == "KMeansAD"

    def __call__(self, X):
        X = np.asarray(X, float)
        if X.ndim == 1:
            X = X[:, None]
        self._n_calls += 1
        fn = getattr(self.clf, self._score_attr)
        if self._quiet:
            with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
                s = np.asarray(fn(X)).ravel()
        else:
            s = np.asarray(fn(X)).ravel()
        if len(s) != len(X):                      # defensive; should not fire
            s = np.interp(np.linspace(0, 1, len(X)), np.linspace(0, 1, len(s)), s)
        return s

    def aggregate(self, X, idx=None, how="max"):
        """Scalar summary of the score, restricted to idx if given.

        'max' is the default because anomaly decisions are threshold crossings,
        not averages: a single strong spike is what a practitioner acts on.
        """
        s = self(X)
        s = s if idx is None else s[np.asarray(idx, int)]
        return float({"max": np.max, "mean": np.mean, "sum": np.sum}[how](s))

    @property
    def n_calls(self):
        return self._n_calls


def self_check(score_fn, X, tol=1e-9):
    """Assert the two invariants every downstream metric depends on."""
    s = score_fn(X)
    assert len(s) == len(X), f"score length {len(s)} != data length {len(X)}"
    assert np.isfinite(s).all(), "non-finite scores"
    assert np.allclose(s, score_fn(X), atol=tol), "scoring is not deterministic"
    return True
