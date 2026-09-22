"""BRIEF B6: tests/test_padding.py — window→timestep rule as documented (TSBAD_INTERNALS §4).

TSB-AD pads the (n − w + 1) window scores to n with ceil((w−1)/2) copies of the
first score in front and (w−1)//2 copies of the last at the back
(IForest.py:214-216, :247-249). Consequently score[t] for 50 ≤ t < n−49 (w=100)
is the score of the window starting at t−50, i.e. [t−50, t+50).
"""
import math

import numpy as np
import pytest

from src.detectors import fit_detector
from tests._synthetic import sine_with_spike


@pytest.mark.parametrize("w", [100, 51, 10])
def test_edge_padding_counts_and_alignment(w):
    from TSB_AD.models.feature import Window
    x = sine_with_spike()
    n = len(x)
    d = fit_detector("IForest", x, seed=0, slidingWindow=w)
    s = d.score(x)
    assert len(s) == n
    front, back = math.ceil((w - 1) / 2), (w - 1) // 2
    assert front + back == w - 1
    # front/back are copies of the first/last window score
    assert np.all(s[:front + 1] == s[0]) and np.all(s[n - back - 1:] == s[-1])
    # the un-padded core equals the raw sklearn scores on the window matrix
    W = Window(window=w).convert(x[:, None])
    assert W.shape == (n - w + 1, w)
    raw = -d._clf.detector_.decision_function(W)          # invert_order = multiply by -1
    assert np.array_equal(s[front:n - back], raw)
    # alignment: window j starts at sample j and is reported at timestep j + front
    j = 1234
    assert s[j + front] == raw[j]
    assert np.array_equal(W[j], x[j:j + w])


def test_shorter_input_is_padded_to_its_own_length():
    x = sine_with_spike()
    d = fit_detector("IForest", x, seed=0)
    assert len(d.score(x[:777])) == 777
