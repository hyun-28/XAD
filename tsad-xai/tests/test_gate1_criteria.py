"""Task C2 pre-registered criteria (src/gate1.py) on hand-built score vectors."""
import numpy as np
import pytest

from src.gate1 import evaluate, normal_test_mask, p2_archive_rule, p2_brief_rule


def test_normal_test_mask_excludes_anomaly_and_buffer():
    m = normal_test_mask(n=1000, train_end=400, start0=700, stop0=710, buffer=0)
    assert not m[:400].any() and m[400:700].all() and not m[700:710].any() and m[710:].all()
    mw = normal_test_mask(n=1000, train_end=400, start0=700, stop0=710, buffer=50)
    assert mw[400:650].all() and not mw[650:760].any() and mw[760:].all()
    with pytest.raises(ValueError):
        normal_test_mask(1000, 400, 300, 310, 0)          # anomaly inside the training prefix
    with pytest.raises(ValueError, match="empty"):
        normal_test_mask(1000, 400, 500, 510, 1000)       # buffer swallows the test region


def test_p2_archive_rule_is_1based_and_uses_L():
    # anomaly 1-based [1250, 1250] (L=1): correct iff 1150 < P < 1350  -> 0-based 1150 <= p0 < 1349
    s0, e0 = 1249, 1250
    assert p2_archive_rule(1150, s0, e0) and not p2_archive_rule(1149, s0, e0)
    assert p2_archive_rule(1348, s0, e0) and not p2_archive_rule(1349, s0, e0)
    # long anomaly: L dominates the slack. 1-based [1000, 1500], L=501: 499 < P < 2001
    s0, e0 = 999, 1500
    assert p2_archive_rule(499, s0, e0) and not p2_archive_rule(498, s0, e0)
    assert p2_archive_rule(1999, s0, e0) and not p2_archive_rule(2000, s0, e0)


def test_p2_brief_rule():
    assert p2_brief_rule(1149, 1249, 1250) and not p2_brief_rule(1148, 1249, 1250)
    assert p2_brief_rule(1349, 1249, 1250) and not p2_brief_rule(1350, 1249, 1250)


def test_evaluate_detects_a_spike_and_not_a_flat_score():
    n, te, s0, e0, w = 2000, 500, 1500, 1510, 100
    score = np.zeros(n)
    score[s0:e0] = 5.0
    d = evaluate(score, train_end=te, start0=s0, stop0=e0, window=w)
    assert d.p1_b0 and d.p1_bw and d.p2_archive and d.p2_brief and d.argmax0 == s0
    assert d.n_normal_b0 == 1500 - 10 and d.n_normal_bw == 1500 - 10 - 2 * w
    flat = np.ones(n)
    d2 = evaluate(flat, train_end=te, start0=s0, stop0=e0, window=w)
    assert not d2.p1_b0 and not d2.p1_bw           # max == percentile, not greater


def test_buffer_matters_when_scores_leak_around_the_anomaly():
    """Window-centred scores spill w/2 points around the anomaly; buffer=w removes the spill."""
    n, te, s0, e0, w = 4000, 1000, 3000, 3010, 100
    rng = np.random.default_rng(0)
    score = rng.uniform(0, 1, n)
    score[s0 - 50:e0 + 50] = 1.2         # the spill is higher than every normal value
    score[s0:e0] = 1.25
    d = evaluate(score, train_end=te, start0=s0, stop0=e0, window=w)
    assert d.thr_b0 > d.thr_bw           # leaked spill inflates the buffer-0 threshold
    assert d.p1_bw and d.p1_b0
