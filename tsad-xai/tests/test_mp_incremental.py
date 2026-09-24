"""BRIEF3 P2/P4: Z/R detectors, window classification, incremental == full recomputation."""
import numpy as np
import pytest

from src.detectors import DetectorError, fit_detector
from src.detectors.mp import MPScorer, affected_range, classify_windows
from src.perturb.operators_simic import OPERATORS, Space, erase

ATOL = 1e-8     # BRIEF3 §6 / S4


def _series(n=2500, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return np.sin(2 * np.pi * t / 37) + 0.3 * np.sin(2 * np.pi * t / 11) + 0.05 * rng.standard_normal(n)


def test_z_equals_gate1_detector_padded():
    """Z with T_A = x is the gate-1 AB-join (D-P2-1)."""
    x, te, m = _series(), 900, 30
    gate = fit_detector("MatrixProfile", x[:te], seed=0, window=m, mode="ab_join").score(x)
    z = MPScorer("Z", m, x[:te])
    assert np.array_equal(z.pad(z.profile(x), x.size), gate)


def test_r_is_plain_euclidean_nearest_neighbour():
    x, te, m = _series(600, 1), 250, 8
    r = MPScorer("R", m, x[:te]).profile(x)
    ref = np.lib.stride_tricks.sliding_window_view(x[:te], m)
    for j in (0, 100, 300, 592):
        brute = np.sqrt(((ref - x[j:j + m]) ** 2).sum(axis=1)).min()
        assert abs(r[j] - brute) < 1e-9


@pytest.mark.parametrize("a,r,m,n", [(50, 10, 8, 200), (50, 3, 8, 200), (0, 5, 8, 200),
                                     (195, 5, 8, 200), (100, 8, 8, 200), (7, 1, 3, 20)])
def test_window_classes(a, r, m, n):
    wc = classify_windows(a, r, m, n)
    starts = np.arange(n - m + 1)
    overlap = (starts <= a + r - 1) & (starts + m - 1 >= a)
    contained = (starts >= a) & (starts + m <= a + r)
    assert np.array_equal(wc.interior, starts[contained])
    assert np.array_equal(wc.boundary, starts[overlap & ~contained])
    assert np.array_equal(wc.untouched, starts[~overlap])
    assert len(wc.interior) == max(0, r - m + 1)
    lo, hi = affected_range(a, r, m, n)
    assert np.array_equal(wc.affected, np.arange(lo, hi))


@pytest.mark.parametrize("kind", ["Z", "R"])
@pytest.mark.parametrize("op", list(OPERATORS))
def test_incremental_equals_full(kind, op):
    x, te, m = _series(), 900, 30
    sc = MPScorer(kind, m, x[:te])
    base = sc.profile(x)
    for a, r in [(1200, 15), (1500, 30), (1800, 60), (2000, 120), (x.size - 40, 40)]:
        if op == "NearestNeighborWindow" and a + r > x.size - r:
            pass    # upstream's left-only edge branch; still defined
        new, _ = erase(x, op, a, r, train_end=te, m=m, space=Space(True, "train"), seed=0)
        inc = sc.incremental(new, x, base, a, r)
        full = sc.profile(new)
        np.testing.assert_allclose(inc, full, rtol=0, atol=ATOL)
        wc = classify_windows(a, r, m, x.size)
        assert np.array_equal(inc[wc.untouched], base[wc.untouched])     # exact by construction


def test_z_constant_interior_is_sqrt_m_without_constant_reference_windows():
    x, te, m = _series(), 900, 30
    sc = MPScorer("Z", m, x[:te])
    new = x.copy()
    new[1500:1600] = 0.123
    p = sc.profile(new)
    assert np.all(p[1500:1600 - m + 1] == np.sqrt(m))


def test_incremental_refuses_changes_outside_the_region():
    x, te, m = _series(), 900, 30
    sc = MPScorer("R", m, x[:te])
    base = sc.profile(x)
    new = x.copy()
    new[1500:1510] = 0
    new[10] += 1
    with pytest.raises(DetectorError):
        sc.incremental(new, x, base, 1500, 10)
