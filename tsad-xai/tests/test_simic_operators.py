"""BRIEF3 P3: the ported PMs equal Šimić et al.'s classes bit for bit; ContextReconstruct's
definition (ours, no upstream) is checked by unit tests.

The upstream module imports numpy/pandas/tslearn/scipy only, so the real classes are imported
from the pinned clone (measured 2026-09-23 in tsadxai-w1). If the clone is absent the
equivalence tests skip.
"""
import math
import subprocess
import sys

import numpy as np
import pytest

from src.config import REPO_ROOT
from src.perturb.operators_simic import (OPERATORS, UPSTREAM_CLASS, UPSTREAM_COMMIT, OperatorError,
                                         Space, context_reconstruct, context_reconstruct_match, erase)

ORIG_DIR = REPO_ROOT / "third_party" / "simic_cmi"
UPSTREAM_OPS = [k for k, v in UPSTREAM_CLASS.items() if v is not None]


@pytest.fixture(scope="module")
def ssp():
    if not (ORIG_DIR / "utils" / "subsequence_perturbation.py").is_file():
        pytest.skip(f"pinned clone missing: run scripts/00_fetch_simic.py ({ORIG_DIR})")
    head = subprocess.run(["git", "-C", str(ORIG_DIR), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    assert head == UPSTREAM_COMMIT, f"third_party/simic_cmi is at {head}, expected {UPSTREAM_COMMIT}"
    sys.path.insert(0, str(ORIG_DIR))
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)   # scipy.ndimage.filters (:6)
            from utils.subsequence_perturbation import SubSequencePerturber
    finally:
        sys.path.remove(str(ORIG_DIR))
    return SubSequencePerturber


def _upstream(ssp, name, sample, s, e, *, original=None, seed=None):
    """Run one upstream PM once. `original` overrides the statistics source (the adaptation)."""
    p = getattr(ssp, UPSTREAM_CLASS[name])(sample.copy())
    if original is not None:
        p.original = np.array(original, copy=True)
    if seed is not None:
        np.random.seed(seed)
    return np.asarray(p.perturb_subsequence(s, e), dtype=np.float64)


def _regions(n, rng, k=40):
    """Random [s, e) including both edges and the NearestNeighborWindow edge branches."""
    out = [(0, 5), (n - 5, n), (1, 2), (n // 2, n // 2 + 1), (n - 12, n - 6)]
    while len(out) < k:
        s = int(rng.integers(0, n - 1))
        e = int(rng.integers(s + 1, min(n, s + n // 3) + 1))
        out.append((s, e))
    return out


@pytest.mark.parametrize("name", UPSTREAM_OPS)
def test_literal_upstream_usage_bit_exact(ssp, name):
    """ref = canvas = one classification-style sample: our function IS the upstream PM."""
    rng = np.random.default_rng(1)
    for trial in range(25):
        n = int(rng.integers(20, 300))
        sample = rng.standard_normal(n) * rng.uniform(0.1, 3) + rng.uniform(-2, 2)
        for s, e in _regions(n, rng):
            try:
                theirs = _upstream(ssp, name, sample, s, e, seed=trial)
            except ValueError:
                # upstream is undefined here (NearestNeighborWindow near an edge): so are we
                with pytest.raises(OperatorError):
                    OPERATORS[name](sample, s, e, ref=sample, seed=trial)
                continue
            ours = OPERATORS[name](sample, s, e, ref=sample, seed=trial)
            assert np.array_equal(ours, theirs), (name, n, s, e)


@pytest.mark.parametrize("name", UPSTREAM_OPS)
def test_adapted_usage_bit_exact(ssp, name):
    """The adaptation (D-P3-1): positions on the whole standardised series, statistics from its
    training part. Upstream reproduces it when `original` is replaced by the training part —
    except UniformNoise100, whose noise length is len(original) upstream and len(canvas) here;
    its bounds are hard-coded, so there the upstream run keeps original = the canvas."""
    rng = np.random.default_rng(2)
    for trial in range(10):
        n = int(rng.integers(200, 2000))
        te = int(rng.integers(n // 5, n // 2))
        z = rng.standard_normal(n).cumsum()
        z = (z - z[:te].mean()) / z[:te].std()
        for _ in range(20):
            s = int(rng.integers(te, n - 10))
            e = int(rng.integers(s + 1, min(n, s + 200) + 1))
            orig = None if name == "UniformNoise100" else z[:te]
            try:
                theirs = _upstream(ssp, name, z, s, e, original=orig, seed=trial)
            except ValueError:
                with pytest.raises(OperatorError):
                    OPERATORS[name](z, s, e, ref=z[:te], seed=trial)
                continue
            ours = OPERATORS[name](z, s, e, ref=z[:te], seed=trial)
            assert np.array_equal(ours, theirs), (name, n, te, s, e)


def test_upstream_inverse_is_max_minus_x_not_sign_flip(ssp):
    x = np.array([1.0, 5.0, 2.0, 3.0])
    assert np.array_equal(_upstream(ssp, "Inverse", x, 1, 3), [1.0, 0.0, 3.0, 3.0])


def test_nearest_neighbor_window_ignores_the_erased_values():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(200)
    y = x.copy()
    y[80:111] = 1e6
    assert np.array_equal(OPERATORS["NearestNeighborWindow"](x, 80, 111),
                          OPERATORS["NearestNeighborWindow"](y, 80, 111))
    out = OPERATORS["NearestNeighborWindow"](x, 80, 111)          # r = 31: 16 left + 15 right
    assert np.array_equal(out[80:96], x[64:80]) and np.array_equal(out[96:111], x[111:126])


# --- the series-level wrapper ---------------------------------------------------------------

@pytest.mark.parametrize("name", list(OPERATORS))
def test_erase_touches_only_the_region(name):
    rng = np.random.default_rng(4)
    x = rng.standard_normal(3000).cumsum() * 0.3 + 5.0
    a, r = 2000, 57
    new, _ = erase(x, name, a, r, train_end=1500, m=20, space=Space(True, "train"), seed=0)
    assert np.array_equal(new[:a], x[:a]) and np.array_equal(new[a + r:], x[a + r:])
    assert np.isfinite(new).all()


def test_erase_standardised_space_meaning():
    """Zero -> the training mean; UniformNoise100 -> mu + sigma U(-1,1); OOD -> 100 x abs-max."""
    rng = np.random.default_rng(5)
    x = rng.standard_normal(2000) * 3.0 + 7.0
    te, a, r = 800, 1200, 50
    sp = Space(True, "train")
    mu, sd = x[:te].mean(), x[:te].std()
    z_tr = (x[:te] - mu) / sd
    new, _ = erase(x, "Zero", a, r, train_end=te, m=10, space=sp)
    assert np.all(new[a:a + r] == mu)
    new, _ = erase(x, "UniformNoise100", a, r, train_end=te, m=10, space=sp, seed=3)
    u = np.random.RandomState(3).uniform(-1, 1, x.size)[a:a + r]
    assert np.array_equal(new[a:a + r], u * sd + mu)
    new, _ = erase(x, "OutOfDistHigh", a, r, train_end=te, m=10, space=sp)
    assert np.all(new[a:a + r] == max(abs(z_tr.max()), abs(z_tr.min())) * 100 * sd + mu)


def test_erase_rejects_bad_calls():
    x = np.arange(100.0) % 7
    with pytest.raises(OperatorError):
        erase(x, "Nope", 50, 5, train_end=40, m=5, space=Space(True, "train"))
    with pytest.raises(OperatorError):
        erase(x, "UniformNoise100", 50, 5, train_end=40, m=5, space=Space(True, "train"), seed=None)
    with pytest.raises(OperatorError):
        erase(np.r_[np.ones(40), np.arange(60.0)], "Zero", 50, 5, train_end=40, m=5,
              space=Space(True, "train"))                                   # sigma = 0


# --- ContextReconstruct (definition tests; no upstream) -------------------------------------

def _rw(n, seed):
    return np.random.default_rng(seed).standard_normal(n).cumsum()


def test_cr_recovers_an_exact_offset_copy():
    """If the test region holds train[j0 : j0+m+r] + c, the fill is exactly the original."""
    m, r, te = 20, 45, 1000
    tr = _rw(te, 6)
    j0 = 300
    test = _rw(1000, 7)
    a = te + 400
    x = np.r_[tr, test]
    x[a - m:a + r] = tr[j0:j0 + m + r] + 2.5
    out = context_reconstruct(x, a, a + r, m=m, train_end=te)
    assert context_reconstruct_match(x, a, a + r, m=m, train_end=te).j == j0
    np.testing.assert_allclose(out[a:a + r], x[a:a + r], rtol=0, atol=1e-12)


def test_cr_never_reads_the_erased_region():
    m, r, te = 15, 40, 800
    x = _rw(2000, 8)
    a = 1300
    y = x.copy()
    y[a:a + r] = np.random.default_rng(9).standard_normal(r) * 1e3
    assert np.array_equal(context_reconstruct(x, a, a + r, m=m, train_end=te),
                          context_reconstruct(y, a, a + r, m=m, train_end=te))


def test_cr_respects_j_plus_m_plus_r_le_train_end():
    """The only exact match sits too close to train_end; it must not be chosen."""
    m, r, te = 20, 60, 1000
    x = _rw(2500, 10)
    a = 1800
    j_bad = te - m - r + 5                              # j + m + r = te + 5 > te
    x[j_bad:j_bad + m] = x[a - m:a]
    match = context_reconstruct_match(x, a, a + r, m=m, train_end=te)
    assert match.j + m + r <= te and match.j != j_bad
    x2 = x.copy()
    j_ok = te - m - r                                   # j + m + r = te: allowed
    x2[j_ok:j_ok + m] = x2[a - m:a]
    assert context_reconstruct_match(x2, a, a + r, m=m, train_end=te).j == j_ok


def test_cr_left_seam_is_the_training_step():
    m, r, te = 12, 30, 600
    x = _rw(1500, 11)
    a = 900
    j = context_reconstruct_match(x, a, a + r, m=m, train_end=te).j
    out = context_reconstruct(x, a, a + r, m=m, train_end=te)
    assert math.isclose(out[a] - out[a - 1], x[j + m] - x[j + m - 1], abs_tol=1e-12)
    np.testing.assert_allclose(out[a:a + r] - out[a - 1], x[j + m:j + m + r] - x[j + m - 1],
                               rtol=0, atol=1e-12)


def test_cr_is_equivariant_under_standardisation():
    m, r, te = 16, 48, 900
    x = _rw(2200, 12) * 4.0 + 30.0
    a = 1500
    raw, _ = erase(x, "ContextReconstruct", a, r, train_end=te, m=m, space=Space(False, "train"))
    std, info = erase(x, "ContextReconstruct", a, r, train_end=te, m=m, space=Space(True, "train"))
    np.testing.assert_allclose(std, raw, rtol=0, atol=1e-10)
    assert info["cr_j"] == context_reconstruct_match(x, a, a + r, m=m, train_end=te).j


def test_cr_rejects_regions_it_is_not_defined_for():
    x = _rw(1000, 13)
    with pytest.raises(OperatorError):
        context_reconstruct(x, 300, 320, m=10, train_end=400)            # overlaps the training part
    with pytest.raises(OperatorError):
        context_reconstruct(x, 900, 990, m=10, train_end=60)             # train too short for m + r
