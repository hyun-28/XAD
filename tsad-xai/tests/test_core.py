"""Property tests. Run: python -m pytest tests/ -q   (or: python tests/test_core.py)

These are the reproducibility defence line. Every claim the paper makes about
the operators and metrics is asserted here.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.perturbations import PERTURBATIONS
from src.baselines import normalize, n1_random, n2_uniform, n3_zscore, n4_recon_error, o1_oracle, o2_achievable
from src.metrics.localization import localization_precision, lift, relevance_auc
from src.metrics.faithfulness import segment_indices, mean_relevance_per_segment, kendall_w
from src.metrics import fisher_mean

rng = np.random.default_rng(0)
L = 500
X1 = np.sin(np.arange(L) / 20.0) + 0.05 * rng.standard_normal(L)
X2 = np.stack([X1, np.cos(np.arange(L) / 15.0)], axis=1)
LABEL = np.zeros(L, int); LABEL[200:230] = 1
IDX = np.arange(100, 150)


def test_shape_preserved():
    for name, op in PERTURBATIONS.items():
        for X in (X1, X2):
            assert op(X, IDX, rng).shape == X.shape, name


def test_identity_on_empty_mask():
    empty = np.array([], int)
    for name, op in PERTURBATIONS.items():
        for X in (X1, X2):
            assert np.allclose(op(X, empty, rng), X), name


def test_no_mutation():
    for name, op in PERTURBATIONS.items():
        before = X1.copy()
        op(X1, IDX, np.random.default_rng(1))
        assert np.allclose(X1, before), f"{name} mutated its input"


def test_deterministic_given_rng():
    for name, op in PERTURBATIONS.items():
        a = op(X1, IDX, np.random.default_rng(7))
        b = op(X1, IDX, np.random.default_rng(7))
        assert np.allclose(a, b), name


def test_only_masked_region_changes():
    keep = np.setdiff1d(np.arange(L), IDX)
    for name, op in PERTURBATIONS.items():
        out = op(X1, IDX, np.random.default_rng(2))
        assert np.allclose(out[keep], X1[keep]), f"{name} changed unmasked timesteps"


def test_b6_preserves_value_multiset():
    out = PERTURBATIONS["B6_shuffle"](X1, IDX, np.random.default_rng(3))
    assert np.allclose(np.sort(out[IDX]), np.sort(X1[IDX]))


def test_relevance_normalisation():
    for fn in (n1_random, n2_uniform, n3_zscore, n4_recon_error):
        r = fn(X1)
        assert len(r) == L
        assert (r >= 0).all()
        assert np.isclose(r.sum(), 1.0), fn.__name__


def test_oracle_lp_is_one():
    assert np.isclose(localization_precision(o1_oracle(LABEL), LABEL), 1.0)


def test_uniform_lift_is_one():
    assert np.isclose(lift(n2_uniform(X1), LABEL), 1.0)


def test_uniform_lp_equals_anomaly_ratio():
    assert np.isclose(localization_precision(n2_uniform(X1), LABEL), LABEL.mean())


def test_o2_never_exceeds_o1():
    score = rng.random(L)
    assert (localization_precision(o2_achievable(LABEL, score), LABEL)
            <= localization_precision(o1_oracle(LABEL), LABEL) + 1e-9)


def test_relevance_auc_bounds():
    assert np.isclose(relevance_auc(o1_oracle(LABEL), LABEL), 1.0)
    assert 0.0 <= relevance_auc(n1_random(X1), LABEL) <= 1.0


def test_segments_partition_exactly():
    for K in (5, 10, 20):
        segs = segment_indices(L, K)
        assert len(segs) == K
        assert np.array_equal(np.concatenate(segs), np.arange(L))


def test_segment_relevance_sums_to_one_when_equal_width():
    segs = segment_indices(L, 10)
    m = mean_relevance_per_segment(n2_uniform(X1), segs)
    assert np.allclose(m, m[0])


def test_fisher_mean_matches_plain_mean_when_identical():
    assert np.isclose(fisher_mean([0.5, 0.5, 0.5]), 0.5)


def test_kendall_w_perfect_agreement():
    R = np.array([[1, 2, 3, 4]] * 5)
    assert np.isclose(kendall_w(R), 1.0)


def test_kendall_w_detects_disagreement():
    R = np.array([[1, 2, 3, 4], [4, 3, 2, 1], [1, 4, 2, 3], [3, 1, 4, 2]])
    assert kendall_w(R) < 0.5


if __name__ == "__main__":
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); passed += 1; print(f"  PASS  {name}")
            except AssertionError as e:
                failed += 1; print(f"  FAIL  {name}: {e}")
            except Exception as e:
                failed += 1; print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
