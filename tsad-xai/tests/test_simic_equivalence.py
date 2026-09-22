"""BRIEF2 D1-3: our DDS/PES/CMI must equal Šimić et al.'s originals bit for bit.

The original `utils/res_utils.py` imports only os/numpy/matplotlib/seaborn/math,
so it can be imported directly from the pinned clone — no MongoDB, no torch
(measured 2026-09-22). If the clone is absent the whole module skips, so the
suite still runs on a machine that has not fetched it.
"""
import numpy as np
import pytest

from src.config import REPO_ROOT
from src.metrics.faithfulness_simic import (CMI, cmi, compute_dataset_dds, cubic_weights,
                                            decaying_degradation_score, pes)

ORIG_DIR = REPO_ROOT / "third_party" / "simic_cmi"
PINNED_COMMIT = "edc6a870b1a50fce3385bfce6a468583d696ea75"


@pytest.fixture(scope="module")
def orig():
    """The upstream functions, imported from the pinned clone."""
    import subprocess
    import sys
    if not (ORIG_DIR / "utils" / "res_utils.py").is_file():
        pytest.skip(f"pinned clone missing: run scripts/00_fetch_simic.py ({ORIG_DIR})")
    head = subprocess.run(["git", "-C", str(ORIG_DIR), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    assert head == PINNED_COMMIT, f"third_party/simic_cmi is at {head}, expected {PINNED_COMMIT}"
    sys.path.insert(0, str(ORIG_DIR))
    try:
        from utils import res_utils
    finally:
        sys.path.remove(str(ORIG_DIR))
    return res_utils


def _random_pairs(n=1000, seed=0):
    """1,000 random curve pairs, length 2-60, values 0-100 (BRIEF2 D1-3)."""
    rng = np.random.default_rng(seed)
    return [(rng.uniform(0, 100, k), rng.uniform(0, 100, k))
            for k in rng.integers(2, 61, size=n)]


def test_dds_matches_original_on_1000_random_pairs(orig):
    pairs = _random_pairs()
    ours = np.array([decaying_degradation_score(m, l) for m, l in pairs])
    theirs = np.array([orig.decaying_degradation_score(m, l) for m, l in pairs])
    assert np.allclose(ours, theirs, atol=1e-12, rtol=0)
    assert np.abs(ours - theirs).max() == 0.0          # in fact identical, not just close


def test_dataset_dds_pes_and_cmi_match_original(orig):
    pairs = _random_pairs(seed=1)
    morfs = [m for m, _ in pairs]
    lerfs = [l for _, l in pairs]
    ours = compute_dataset_dds(morfs, lerfs)
    theirs = orig.compute_dataset_dds(morfs, lerfs)
    assert np.allclose(ours, theirs, atol=1e-12, rtol=0)
    # PES over many random subsets of the DDS values
    rng = np.random.default_rng(2)
    for _ in range(200):
        sub = ours[rng.integers(0, len(ours), size=rng.integers(1, 50))]
        assert pes(sub) == orig.pes(sub)
        assert cmi(float(sub.mean()), pes(sub)) == orig.CMI(float(sub.mean()), orig.pes(sub))


def test_cmi_matches_original_over_the_whole_sign_grid(orig):
    rng = np.random.default_rng(3)
    vals = np.concatenate([rng.uniform(-1, 1, 400), [0.0, -0.0, 1.0, -1.0, 1e-12, -1e-12]])
    for d in vals:
        for p in vals[::7]:
            assert cmi(d, p) == orig.CMI(d, p)


def test_max_diff_default_is_the_originals_100():
    m, l = [100.0, 60.0, 20.0], [100.0, 95.0, 90.0]
    assert decaying_degradation_score(m, l) == decaying_degradation_score(m, l, max_diff=100)
    # halving max_diff doubles the score (it is a pure divisor)
    assert decaying_degradation_score(m, l, max_diff=50) == pytest.approx(
        2 * decaying_degradation_score(m, l), rel=0, abs=1e-15)


def test_cubic_weights_shape_and_values():
    w = cubic_weights(4)
    assert np.allclose(w, (np.array([4, 3, 2, 1]) / 4) ** 3)
    assert w[0] > w[-1] > 0


# ---- boundary cases named in BRIEF2 D1-3 -------------------------------------------

def test_identical_curves_give_zero_everything():
    c = [100.0, 80.0, 60.0, 40.0]
    d = decaying_degradation_score(c, c)
    assert d == 0.0
    assert pes([d]) == 0.0
    assert cmi(d, pes([d])) == 0.0


def test_opposite_signs_give_cmi_zero():
    assert cmi(0.5, -0.5) == 0.0
    assert cmi(-0.5, 0.5) == 0.0
    assert cmi(0.5, 0.0) == 0.0         # upstream's `<= 0` branch: a zero also yields 0
    assert cmi(0.0, 0.5) == 0.0


def test_fully_reversed_curve_gives_negative_dds_and_positive_cmi():
    morf = [100.0, 95.0, 90.0, 85.0]     # MoRF barely drops
    lerf = [100.0, 40.0, 20.0, 0.0]      # LeRF collapses -> the attribution is anti-faithful
    d = decaying_degradation_score(lerf, morf)   # swap so that morf is the fast-falling one
    assert d > 0
    d_rev = decaying_degradation_score(morf, lerf)
    assert d_rev < 0
    p = pes([d_rev, d_rev, d_rev])
    assert p == -1.0
    assert cmi(d_rev, p) > 0             # same sign -> harmonic mean of magnitudes


def test_perfect_scores_are_bounded_by_one():
    n = 20
    morf = [100.0] + [0.0] * (n - 1)     # everything destroyed immediately
    lerf = [100.0] * n                   # nothing lost
    assert decaying_degradation_score(morf, lerf) == pytest.approx(1.0, abs=1e-15)
    assert decaying_degradation_score(lerf, morf) == pytest.approx(-1.0, abs=1e-15)
    assert cmi(1.0, 1.0) == pytest.approx(1.0)


# ---- our deliberate divergences (docs/SIMIC_PORT.md) --------------------------------

def test_short_or_malformed_input_raises_where_upstream_returns_nan(orig):
    import warnings
    with pytest.raises(ValueError, match="at least 2 points"):
        decaying_degradation_score([100.0], [100.0])
    with warnings.catch_warnings():                    # upstream: nan + RuntimeWarning
        warnings.simplefilter("ignore")
        assert np.isnan(orig.decaying_degradation_score([100.0], [100.0]))
    with pytest.raises(ValueError, match="same length"):
        decaying_degradation_score([1.0, 2.0], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="non-finite"):
        decaying_degradation_score([1.0, np.nan], [1.0, 2.0])
    with pytest.raises(ValueError, match="empty"):
        pes([])
    with pytest.raises(ValueError, match="max_diff"):
        decaying_degradation_score([1.0, 2.0], [1.0, 2.0], max_diff=0)


def test_CMI_alias_is_the_same_function():
    assert CMI is cmi
