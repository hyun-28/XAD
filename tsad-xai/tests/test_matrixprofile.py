"""D-C5-3: the Matrix Profile wrapper.

Rule 1 of the researcher's pre-registration: in the self-join setting our scores
must equal TSB-AD's `MatrixProfile.fit().decision_scores_` including the padding.
Rule 2: fit_on=train_prefix is an AB-join against the stored reference, never a
self-join of x against itself.
"""
import numpy as np
import pytest

from src.detectors import DetectorError, fit_detector
from src.detectors.matrixprofile import MatrixProfileDetector, _pad_like_tsb_ad, benchmark_window


def _series(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.sin(np.linspace(0, 60 * np.pi, n)) + 0.05 * rng.standard_normal(n)
    x[n // 2:n // 2 + 40] += 3.0
    return x


@pytest.mark.parametrize("w", [100, 51, 7])
def test_self_join_equals_tsb_ad_including_padding(w):
    from TSB_AD.models.MatrixProfile import MatrixProfile
    x = _series()
    ref = MatrixProfile(window=w)
    ref.fit(x.reshape(-1, 1))
    ours = fit_detector("MatrixProfile", x, seed=0, window=w, mode="self_join").score(x)
    assert np.array_equal(ours, ref.decision_scores_)


@pytest.mark.parametrize("w", [4, 5])
def test_padding_alignment_for_both_parities(w):
    n = 40
    profile = np.arange(n - w + 1, dtype=np.float64) + 10.0
    padded = _pad_like_tsb_ad(profile, n, w)
    assert padded.shape == (n,)
    # profile[j] lands at j + w//2; the ends carry the profile minimum
    for j in (0, 3, len(profile) - 1):
        assert padded[j + w // 2] == profile[j]
    assert np.all(padded[:w // 2] == profile.min())
    assert np.all(padded[len(profile) + w // 2:] == profile.min())


def test_ab_join_uses_the_reference_not_the_scored_series():
    """A train-prefix AB-join must not exclude x's own neighbours: the training
    windows match themselves, so their distance is ~0, while a self-join would
    give them a nonzero nearest-neighbour distance."""
    x = _series()
    te = 1200
    d = fit_detector("MatrixProfile", x[:te], seed=0, window=100, mode="ab_join")
    s = d.score(x)
    assert d.config["mode"] == "ab_join" and d.config["exclusion_zone"] == 0
    assert np.allclose(s[:te - 100], 0.0, atol=1e-6)        # self-matches inside the reference
    self_join = fit_detector("MatrixProfile", x, seed=0, window=100, mode="self_join").score(x)
    assert self_join[:te - 100].max() > 1e-3                # the exclusion zone forbids self-matching
    assert not np.array_equal(s, self_join)


def test_ab_join_scores_an_unseen_pattern_higher_than_a_seen_one():
    """The elevated scores sit in the anomaly's WINDOW SUPPORT, not in the anomaly.

    profile[j] describes window [j, j+w) and is written to index j + w//2, so a
    window that overlaps [start, stop) is reported somewhere in
    [start - w + 1 + w//2, stop - 1 + w//2) ~= [start - w/2, stop + w/2).
    For an anomaly shorter than w the maximum therefore lands BEFORE `start`.
    This is why gate 1 keeps buffer = w (D-C2-1) and why the report carries the
    window-support diagnostic (D-C5-5).
    """
    x = _series()
    w, te = 100, 1200
    start, stop = 1500, 1540
    d = fit_detector("MatrixProfile", x[:te], seed=0, window=w, mode="ab_join")
    s = d.score(x)
    support = slice(start - w + 1 + w // 2, stop - 1 + w // 2)
    outside = np.ones(len(s), bool)
    outside[:te] = False
    outside[max(0, start - 2 * w):stop + 2 * w] = False
    assert s[support].max() > 5 * np.percentile(s[outside], 99)
    assert s[support].argmax() + support.start < start      # the peak precedes the GT interval


def test_fit_stores_a_copy_of_the_reference_and_does_not_alias_the_input():
    x = _series(n=1000)
    d = MatrixProfileDetector.fit(x[:500], window=50, mode="ab_join")
    assert d.reference is not None and len(d.reference) == 500
    assert not np.shares_memory(d.reference, x)
    assert MatrixProfileDetector.fit(x, window=50, mode="self_join").reference is None


def test_scoring_is_deterministic_and_does_not_refit():
    x = _series()
    d = fit_detector("MatrixProfile", x[:1200], seed=0, window=100, mode="ab_join")
    a, b = d.score(x), d.score(x)
    assert np.array_equal(a, b) and d.n_score_calls == 2


def test_benchmark_window_matches_find_length_rank():
    from TSB_AD.utils.slidingWindows import find_length_rank
    x = _series()
    assert benchmark_window(x) == int(find_length_rank(x.reshape(-1, 1), rank=1))


def test_bad_inputs_raise():
    x = _series(n=500)
    with pytest.raises(ValueError, match="mode"):
        fit_detector("MatrixProfile", x, seed=0, window=50, mode="cross_join")
    with pytest.raises(TypeError, match="unknown MatrixProfile"):
        fit_detector("MatrixProfile", x, seed=0, window=50, bogus=1)
    with pytest.raises(DetectorError, match="window"):
        fit_detector("MatrixProfile", x, seed=0, window=2, mode="self_join")
    with pytest.raises(DetectorError, match="reference length"):
        fit_detector("MatrixProfile", x[:30], seed=0, window=50, mode="ab_join")
    d = fit_detector("MatrixProfile", x, seed=0, window=50, mode="self_join")
    with pytest.raises(DetectorError, match="series length"):
        d.score(x[:10])
    bad = x.copy(); bad[3] = np.nan
    with pytest.raises(DetectorError, match="NaN"):
        d.score(bad)


def test_config_records_everything_needed_to_reproduce():
    d = fit_detector("MatrixProfile", _series()[:1200], seed=0, window=100, mode="ab_join")
    for k in ("detector", "backend", "mode", "window", "periodicity", "exclusion_zone",
              "normalize", "deterministic", "n_fit"):
        assert k in d.config
    assert d.config["deterministic"] is True and d.config["backend"].startswith("stumpy ")
