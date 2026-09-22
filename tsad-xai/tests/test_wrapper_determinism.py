"""BRIEF B6: tests/test_wrapper_determinism.py — seeds and fit-once."""
import numpy as np

from src.detectors import fit_detector
from tests._synthetic import sine_with_spike


def test_same_seed_twice_is_bitwise_identical():
    x = sine_with_spike()
    a = fit_detector("IForest", x, seed=1).score(x)
    b = fit_detector("IForest", x, seed=1).score(x)
    assert np.array_equal(a, b)


def test_different_seeds_differ():
    x = sine_with_spike()
    a = fit_detector("IForest", x, seed=0).score(x)
    b = fit_detector("IForest", x, seed=1).score(x)
    assert not np.array_equal(a, b)


def test_fit_once_then_score_twice_is_identical_and_does_not_refit():
    x = sine_with_spike()
    d = fit_detector("IForest", x, seed=0)
    inner = d._clf.detector_               # the fitted sklearn forest
    s1 = d.score(x)
    s2 = d.score(x)
    assert np.array_equal(s1, s2)
    assert d._clf.detector_ is inner       # same object: no refit happened
    assert d.n_score_calls == 2


def test_scoring_does_not_touch_global_numpy_rng():
    x = sine_with_spike()
    d = fit_detector("IForest", x, seed=0)
    np.random.seed(7)
    before = np.random.rand()
    np.random.seed(7)
    d.score(x)
    fit_detector("IForest", x, seed=3)
    assert np.random.rand() == before


def test_config_is_complete_and_explicit():
    x = sine_with_spike()
    d = fit_detector("IForest", x, seed=2, max_features=1.0)
    for k in ("slidingWindow", "n_estimators", "max_features", "max_features_type", "n_jobs",
              "random_state", "normalize", "max_samples", "contamination", "bootstrap", "n_fit"):
        assert k in d.config
    assert d.config["random_state"] == 2 and d.config["normalize"] is False
    assert d.config["max_features_type"] == "float" and d.config["n_fit"] == len(x)
