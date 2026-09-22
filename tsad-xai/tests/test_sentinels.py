"""BRIEF_A-followup F7: −999 sentinels (F3)."""
import hashlib

import numpy as np
import pytest

from src.data import sentinels as sen
from src.data.conventions import IndexConvention, to_half_open


def _series():
    x = np.full(1000, 10.0)
    x[100] = -999.0            # in training prefix
    x[600] = -999.0            # normal test region
    x[700] = -999.0            # inside GT (raw 1-based fields 701..702)
    x[650] = -998.9            # near, not exact
    return x


def test_exact_and_near_detection():
    x = _series()
    x[:500] += np.tile([0.0, 1.0], 250)          # MAD(train) = 0.5 -> tol = 1.5
    res = sen.detect(x, train_end=500, near_mad_factor=3.0)
    assert res.tol == pytest.approx(1.5)
    assert res.exact_idx.tolist() == [100, 600, 700]
    assert res.near_idx.tolist() == [100, 600, 650, 700]


def test_mad_zero_makes_near_equal_exact():
    x = _series()
    res = sen.detect(x, train_end=50, near_mad_factor=3.0)     # constant prefix
    assert res.tol == 0.0
    assert res.near_idx.tolist() == res.exact_idx.tolist()


def test_classification_uses_1based_inclusive_boundaries():
    x = _series()
    res = sen.detect(x, train_end=500)
    # raw fields begin=701, end=702 -> [700, 702) under the archive convention
    start0, stop0 = to_half_open(701, 702, IndexConvention.MATLAB_1BASED_INCLUSIVE)
    sen.classify(res, train_end=500, start0=start0, stop0=stop0)
    assert res.in_train.tolist() == [100]
    assert res.in_test_normal.tolist() == [600]
    assert res.in_gt.tolist() == [700]
    assert res.exclusion_idx.tolist() == [100, 600]
    # a 0-based-inclusive misreading would put the GT at [701, 703) and lose the point
    res2 = sen.detect(x, train_end=500)
    sen.classify(res2, train_end=500, start0=701, stop0=703)
    assert res2.in_gt.size == 0 and 700 in res2.in_test_normal


def test_classification_rejects_gt_inside_training_prefix():
    res = sen.detect(_series(), train_end=500)
    with pytest.raises(ValueError):
        sen.classify(res, train_end=500, start0=100, stop0=101)


def test_twin_length_mismatch_no_guessed_alignment():
    idx = np.array([100, 600, 700])
    twin = np.full(650, 10.0)
    twin[100] = -999.0
    tc = sen.twin_check(idx, twin, twin_num=2, window=51, k=10.0)
    assert (tc.n_confirmed, tc.n_absent, tc.n_length_mismatch) == (1, 1, 1)
    assert tc.status == "partial"
    tc_all = sen.twin_check(np.array([700, 800]), twin, twin_num=2)
    assert tc_all.status == "length_mismatch"
    assert sen.twin_check(np.array([100]), twin, twin_num=2).status == "confirmed"
    assert sen.twin_check(np.array([600]), twin, twin_num=2).status == "absent"


def test_local_outlier_rule():
    x = np.sin(np.linspace(0, 6 * np.pi, 400))
    assert not sen.local_outlier(x, 200)
    x[200] = -999.0
    assert sen.local_outlier(x, 200)
    flat = np.zeros(100)
    assert sen.outlier_ratio(flat, 50) == 0.0
    flat[50] = 1.0
    assert sen.outlier_ratio(flat, 50) == np.inf
    with pytest.raises(ValueError):
        sen.local_outlier(x, 10, window=50)


def test_series_is_not_modified():
    x = _series()
    before = hashlib.sha256(x.tobytes()).hexdigest()
    res = sen.detect(x, train_end=500)
    sen.classify(res, train_end=500, start0=700, stop0=702)
    _ = res.exclusion_idx
    sen.twin_check(res.exact_idx, x, twin_num=1)
    for i in res.exact_idx:
        sen.outlier_ratio(x, int(i))
    assert hashlib.sha256(x.tobytes()).hexdigest() == before
