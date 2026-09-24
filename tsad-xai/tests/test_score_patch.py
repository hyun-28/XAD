"""W2 rev1 E0: MatrixProfileDetector.score_patch — J(R), agreement with score(), identity bit-equality."""
import numpy as np
import pytest

from src.detectors import DetectorError, fit_detector
from src.perturb.operators import Ctx, apply

TOL = 1e-6      # rev1 §4-3


def _series(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return np.sin(2 * np.pi * t / 41) + 0.4 * np.sin(2 * np.pi * t / 13) + 0.05 * rng.standard_normal(n)


@pytest.mark.parametrize("w", [7, 30, 31])
def test_J_is_rev1_definition(w):
    """J(R) = [a + h - w + 1, b + h - 1], clipped to valid window centres [h, n - w + h]."""
    x, te = _series(), 1000
    det = fit_detector("MatrixProfile", x[:te], seed=0, window=w, mode="ab_join")
    h, n = w // 2, x.size
    for a, b in [(1500, 1501), (1500, 1500 + w), (2000, 2000 + 3 * w), (n - 5, n), (te, te + 4)]:
        J, s = det.score_patch(x, (a, b))
        want = np.arange(max(a + h - w + 1, h), min(b + h - 1, n - w + h) + 1)
        assert np.array_equal(J, want) and s.size == J.size
        # support(j) = [j - h, j - h + w) meets [a, b) exactly for j in J
        for j in (J[0], J[-1]):
            assert (j - h) < b and (j - h + w) > a
        if J[0] > h:
            assert not ((J[0] - 1 - h) < b and (J[0] - 1 - h + w) > a)


@pytest.mark.parametrize("w", [7, 30, 31])
def test_patch_matches_full_score_after_perturbation(w):
    x, te = _series(), 1000
    det = fit_detector("MatrixProfile", x[:te], seed=0, window=w, mode="ab_join")
    rng = np.random.default_rng(1)
    for L in (max(1, round(0.25 * w)), max(1, round(0.5 * w)), w):
        for a in (1200, 1777, 2500, x.size - L):
            xp = x.copy()
            xp[a:a + L] += rng.standard_normal(L)
            J, s = det.score_patch(xp, (a, a + L))
            full = det.score(xp)
            assert np.max(np.abs(s - full[J])) <= TOL


def test_identity_is_bit_identical_and_ai_zero():
    x, te, w = _series(), 1000, 30
    det = fit_detector("MatrixProfile", x[:te], seed=0, window=w, mode="ab_join")
    ctx = Ctx(train_end=te, gt=(2800, 2850), w=w)
    for a, L in [(1300, 8), (1600, 15), (2100, 30)]:
        _, before = det.score_patch(x, (a, a + L))
        _, after = det.score_patch(apply("identity", x, [(a, a + L)], None, ctx), (a, a + L))
        assert np.array_equal(before, after)
        assert (after.max() - before.max()) / 1.0 == 0.0


def test_score_patch_refuses_self_join_and_bad_regions():
    x = _series()
    sj = fit_detector("MatrixProfile", x, seed=0, window=20, mode="self_join")
    with pytest.raises(DetectorError):
        sj.score_patch(x, (100, 110))
    ab = fit_detector("MatrixProfile", x[:1000], seed=0, window=20, mode="ab_join")
    for bad in [(5, 5), (-1, 3), (2990, 3001)]:
        with pytest.raises(DetectorError):
            ab.score_patch(x, bad)
