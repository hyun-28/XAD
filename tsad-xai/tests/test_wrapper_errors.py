"""BRIEF B6: tests/test_wrapper_errors.py — every B3 failure mode raises DetectorError."""
import numpy as np
import pytest

from src.detectors import DetectorError, fit_detector, validate_scores
from src.detectors.iforest import IForestDetector
from tests._synthetic import sine_with_spike


class _Fake:
    """Stands in for TSB-AD's IForest with a scripted decision_function."""
    def __init__(self, out):
        self.out = out
        self.decision_scores_ = np.ones(3)

    def decision_function(self, X):
        return self.out(X) if callable(self.out) else self.out


def _det(out, constant_score="error"):
    cfg = {"slidingWindow": 10, "constant_score": constant_score}
    return IForestDetector(_Fake(out), cfg)


X = np.linspace(0, 1, 500)


def test_tsb_ad_error_string_becomes_DetectorError():
    # model_wrapper.py:16-19 returns a str instead of raising
    with pytest.raises(DetectorError, match="error string"):
        _det("Model function 'run_IForest' is not defined.").score(X)


def test_length_mismatch():
    with pytest.raises(DetectorError, match="length"):
        _det(np.zeros(499)).score(X)


def test_nan_and_inf():
    bad = np.arange(500.0)
    bad[7] = np.nan
    with pytest.raises(DetectorError, match="non-finite"):
        _det(bad).score(X)
    bad[7] = np.inf
    with pytest.raises(DetectorError, match="non-finite"):
        _det(bad).score(X)


def test_constant_output_error_or_flag():
    with pytest.raises(DetectorError, match="constant"):
        _det(np.full(500, 3.0)).score(X)
    d = _det(np.full(500, 3.0), constant_score="flag")
    s = d.score(X)
    assert s.dtype == np.float64 and d.last_flags == ["CONSTANT_SCORE"]


def test_shapes_and_dtype():
    s, flags = validate_scores(np.arange(500, dtype=np.float32)[:, None], 500)
    assert s.shape == (500,) and s.dtype == np.float64 and flags == []
    with pytest.raises(DetectorError, match="1-D"):
        validate_scores(np.zeros((250, 2)), 500)
    with pytest.raises(DetectorError, match="numeric"):
        validate_scores(np.array(["a"] * 500), 500)


def test_exception_inside_decision_function_is_wrapped():
    def boom(X):
        raise RuntimeError("inner failure")
    with pytest.raises(DetectorError, match="inner failure"):
        _det(boom).score(X)


def test_normalize_true_is_rejected():
    with pytest.raises(ValueError, match="normalize=True is forbidden"):
        fit_detector("IForest", sine_with_spike(), seed=0, normalize=True)


def test_unknown_hyperparameter_and_detector():
    with pytest.raises(TypeError, match="unknown IForest"):
        fit_detector("IForest", sine_with_spike(), seed=0, bogus=1)
    with pytest.raises(ValueError, match="unknown detector"):
        fit_detector("KMeansAD", sine_with_spike(), seed=0)


def test_series_shorter_than_window():
    with pytest.raises(DetectorError, match="slidingWindow"):
        fit_detector("IForest", sine_with_spike(n=80), seed=0, slidingWindow=100)
    d = fit_detector("IForest", sine_with_spike(), seed=0)
    with pytest.raises(DetectorError, match="slidingWindow"):
        d.score(sine_with_spike(n=50))
