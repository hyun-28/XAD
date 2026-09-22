"""D-B2-3: src/data/ucr.py::slice_fit is the only place that cuts the fit data."""
import numpy as np
import pytest

from src.data.ucr import parse_filename, slice_fit

META = parse_filename("012_UCR_Anomaly_tiltAPB1_100000_114283_114350.txt")


def test_train_prefix_length_equals_train_end():
    x = np.arange(120_000, dtype=np.float64)
    x_fit, n_fit = slice_fit(x, META, "train_prefix")
    assert n_fit == META.train_end_raw == 100_000 == len(x_fit)
    assert x_fit[0] == 0.0 and x_fit[-1] == 99_999.0
    assert not x_fit.flags.writeable


def test_full_uses_everything():
    x = np.arange(120_000, dtype=np.float64)
    x_fit, n_fit = slice_fit(x, META, "full")
    assert n_fit == 120_000 and np.shares_memory(x_fit, x)


def test_rejects_bad_inputs():
    x = np.arange(120_000, dtype=np.float64)
    with pytest.raises(ValueError, match="fit_on"):
        slice_fit(x, META, "test_only")
    with pytest.raises(ValueError, match="1-D"):
        slice_fit(x[:, None], META, "full")
    with pytest.raises(ValueError, match="train_end"):
        slice_fit(x[:50_000], META, "train_prefix")      # train_end beyond the series
