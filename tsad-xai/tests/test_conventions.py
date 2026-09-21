"""BRIEF B6: tests/test_conventions.py — round trip + one known-series snapshot."""
import numpy as np
import pytest

from src.data.conventions import (IndexConvention, active_convention, from_half_open,
                                  to_half_open, train_prefix_stop)
from src.data.ucr import load_series, parse_filename


@pytest.mark.parametrize("conv", list(IndexConvention))
@pytest.mark.parametrize("begin,end", [(1, 1), (1, 2), (52000, 52620), (168250, 168250), (7, 8)])
def test_round_trip(conv, begin, end):
    if conv is IndexConvention.ZERO_BASED_EXCLUSIVE and begin == end:
        with pytest.raises(ValueError, match="empty"):   # exclusive end cannot express L = 1
            to_half_open(begin, end, conv)
        return
    start0, stop0 = to_half_open(begin, end, conv)
    assert stop0 > start0
    assert from_half_open(start0, stop0, conv) == (begin, end)


def test_matlab_convention_arithmetic():
    c = IndexConvention.MATLAB_1BASED_INCLUSIVE
    assert to_half_open(52000, 52620, c) == (51999, 52620)      # 001_..._52000_52620
    assert to_half_open(168250, 168250, c) == (168249, 168250)  # 079/187: single point
    assert (52620 - 52000 + 1) == (52620 - 51999)               # L = end - begin + 1 (slide 4)


def test_zero_based_variants():
    assert to_half_open(10, 12, IndexConvention.ZERO_BASED_INCLUSIVE) == (10, 13)
    assert to_half_open(10, 12, IndexConvention.ZERO_BASED_EXCLUSIVE) == (10, 12)


def test_empty_interval_raises():
    with pytest.raises(ValueError, match="empty"):
        to_half_open(5, 5, IndexConvention.ZERO_BASED_EXCLUSIVE)
    with pytest.raises(ValueError):
        to_half_open(0, 3, IndexConvention.MATLAB_1BASED_INCLUSIVE)


def test_train_prefix_is_convention_independent():
    assert train_prefix_stop(35000) == 35000
    with pytest.raises(ValueError):
        train_prefix_stop(0)


def test_active_convention_is_valid_enum():
    assert active_convention() in IndexConvention


# --- snapshot on a known series: the -999 sentinel inserted on slide 46 -------

def test_known_series_gt_boundary(official_fulldata_dir):
    fn = "185_UCR_Anomaly_resperation11_58000_110800_110801.txt"
    m = parse_filename(fn)
    x = load_series(official_fulldata_dir / fn)
    start0, stop0 = to_half_open(m.begin_raw, m.end_raw, IndexConvention.MATLAB_1BASED_INCLUSIVE)
    assert (start0, stop0) == (110799, 110801)
    assert x[start0] == -999.0                       # the inserted "missing value" sentinel
    assert x[stop0 - 1] != -999.0                    # ... and the second point is ordinary
    # under the 0-based reading the sentinel would fall outside the interval
    s0, e0 = to_half_open(m.begin_raw, m.end_raw, IndexConvention.ZERO_BASED_INCLUSIVE)
    assert not np.any(x[s0:e0] == -999.0)
    # NOTE (measured 2026-09-21): -999 also occurs naturally elsewhere in this
    # record (83 times, 31 inside the training prefix), so uniqueness of the
    # sentinel is NOT asserted; only its position relative to the fields is.
    assert (x[max(0, start0 - 50):start0] != -999.0).all()
    assert (x[stop0:stop0 + 50] != -999.0).all()
