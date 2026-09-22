"""BRIEF_A-followup F7: recording groups (F2)."""
import numpy as np
import pytest

from src.data.recording_groups import (SeriesRef, candidate_pairs, compare_pairs, content_groups,
                                       grouping_disagreements, name_group_of, prefix_correlation)


@pytest.mark.parametrize("name,expected", [
    ("DISTORTED1sddb40", "1sddb40"),
    ("NOISEMesoplodonDensirostris", "MesoplodonDensirostris"),
    ("tiltAPB1", "tiltAPB1"),
    ("DISTORTEDNOISEx", "NOISEx"),          # exactly one prefix is stripped
    ("xDISTORTED", "xDISTORTED"),           # only a leading prefix counts
])
def test_prefix_rule(name, expected):
    assert name_group_of(name) == expected


def _ref(num, name, x, domain="ECG", train_end=None):
    x = np.asarray(x, dtype=np.float64)
    return SeriesRef(num=num, name=name, domain=domain, length=len(x),
                     train_end=train_end or len(x) // 2, x=x)


def test_synthetic_same_signal_plus_noise_groups_together():
    rng = np.random.default_rng(0)
    base = np.sin(np.linspace(0, 40 * np.pi, 4000))
    plain = _ref(1, "sig", base)
    noisy = _ref(2, "DISTORTEDsig", base + 0.05 * rng.standard_normal(base.size))
    other = _ref(3, "other", rng.standard_normal(base.size))           # different signal, same domain
    cut = _ref(4, "sigcut", base[:3000])                                # same recording, shorter cut
    refs = [plain, noisy, other, cut]
    pairs = compare_pairs(refs, compare_points=5000, corr_threshold=0.95, cross_length_tol=0.5)
    cg = content_groups(refs, pairs)
    assert cg[1] == cg[2] == cg[4] == "cg001"
    assert cg[3] != cg[1]
    dis = grouping_disagreements(refs, cg)
    assert dis["diff_name_same_content"] == {"cg001": ["sig", "sigcut"]}
    assert dis["same_name_diff_content"] == {}


def test_cross_name_pairs_respect_domain_and_length_tolerance():
    a = _ref(1, "a", np.arange(1000.0), domain="ECG")
    b = _ref(2, "b", np.arange(1000.0), domain="ECG")
    c = _ref(3, "c", np.arange(1000.0), domain="Gait")          # other domain -> never compared
    d = _ref(4, "d", np.arange(1200.0), domain="ECG")           # 20 % longer -> filtered at tol 1 %
    e = _ref(5, "DISTORTEDa", np.arange(1500.0), domain="ECG")  # same name -> always compared
    pairs = candidate_pairs([a, b, c, d, e], cross_length_tol=0.01)
    keys = {(p.num, q.num) for p, q in pairs}
    assert keys == {(1, 5), (1, 2)}


def test_constant_prefix_gives_nan_not_a_match():
    a = _ref(1, "a", np.zeros(100))
    b = _ref(2, "DISTORTEDa", np.zeros(100))
    n, r = prefix_correlation(a, b, 5000)
    assert n == 50 and np.isnan(r)
    pairs = compare_pairs([a, b], compare_points=5000, corr_threshold=0.95, cross_length_tol=0.01)
    assert len(pairs) == 1 and pairs[0].same_recording is False


def test_correlation_uses_min_of_train_ends_and_compare_points():
    x = np.random.default_rng(1).standard_normal(10000)
    a = _ref(1, "a", x, train_end=8000)
    b = _ref(2, "DISTORTEDa", x, train_end=300)
    n, r = prefix_correlation(a, b, 5000)
    assert n == 300 and r == pytest.approx(1.0)
