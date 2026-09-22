"""BRIEF B6: tests/test_fit_score_consistency.py.

(1) normalize=False: the wrapper's score(x_fit) equals TSB-AD's own path bit
    for bit — an IForest built exactly as model_wrapper.run_IForest builds it
    (slidingWindow=100, n_estimators, max_features=1, n_jobs=1) with the same
    random_state, and its decision_scores_ (what run_IForest returns).
    Tolerance: 0 (np.array_equal). Both paths run the same sklearn forest on
    the same window matrix; any difference would be a wrapper bug.
(2) normalize=True: fit() z-scores the windows, decision_function() does not
    (TSB_AD/models/IForest.py:185-189 vs :240-245, PyPI 1.5) — the trap is
    real and measurable.
"""
import numpy as np
import pytest

from src.detectors import fit_detector
from tests._synthetic import sine_with_spike


@pytest.mark.parametrize("max_features", [1, 1.0])
@pytest.mark.parametrize("seed", [0, 1])
def test_wrapper_matches_tsb_ad_path_bitwise(max_features, seed):
    from TSB_AD.models.IForest import IForest
    x = sine_with_spike()
    # TSB-AD path = model_wrapper.run_IForest's constructor call + normalize=False + the seed
    ref = IForest(slidingWindow=100, n_estimators=200, max_features=max_features, n_jobs=1,
                  random_state=seed, normalize=False)
    ref.fit(x[:, None])
    ref_scores = ref.decision_scores_.ravel()

    d = fit_detector("IForest", x, seed=seed, max_features=max_features)
    assert np.array_equal(d.score(x), ref_scores)
    assert np.array_equal(d.train_scores, ref_scores)


def test_normalize_true_breaks_fit_score_consistency():
    from TSB_AD.models.IForest import IForest
    x = sine_with_spike()[:, None]
    bad = IForest(slidingWindow=100, n_estimators=100, max_features=1, random_state=0, normalize=True).fit(x)
    good = IForest(slidingWindow=100, n_estimators=100, max_features=1, random_state=0, normalize=False).fit(x)
    diff_bad = np.abs(bad.decision_scores_ - bad.decision_function(x)).max()
    diff_good = np.abs(good.decision_scores_ - good.decision_function(x)).max()
    assert diff_good == 0.0
    assert diff_bad > 1e-6, f"the normalize trap did not reproduce (max diff {diff_bad})"


def test_run_IForest_defaults_are_what_the_config_says():
    """D-B2-1: the benchmark path's IForest arguments (max_features int 1, window 100)."""
    import inspect
    from TSB_AD import model_wrapper
    from src.config import load_yaml
    sig = inspect.signature(model_wrapper.run_IForest)
    assert sig.parameters["slidingWindow"].default == 100
    assert sig.parameters["max_features"].default == 1 and isinstance(sig.parameters["max_features"].default, int)
    assert sig.parameters["n_estimators"].default == 100      # benchmark overrides to 200 via HP_list
    from TSB_AD.HP_list import Optimal_Uni_algo_HP_dict
    assert Optimal_Uni_algo_HP_dict["IForest"] == {"n_estimators": 200}
    cfg = load_yaml("detectors")["iforest"]
    assert cfg["slidingWindow"] == 100 and cfg["n_estimators"] == 200
    assert cfg["max_features_primary"] == 1 and isinstance(cfg["max_features_primary"], int)
    assert cfg["max_features_variants"] == [1, 1.0] and cfg["normalize"] is False
