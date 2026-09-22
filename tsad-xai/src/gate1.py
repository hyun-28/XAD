"""Gate 1 — pre-registered detection criteria (BRIEF Task C, C2). Pure numpy.

All indices are 0-based half-open as produced by src/data/conventions.py; the
archive's own scoring rule is applied in its 1-based frame by converting once,
here, and nowhere else.

P1 (primary)   max(score[start0:stop0]) > percentile(score[normal_test], 99)
               normal_test = test region [train_end, n) minus [start0 - buffer,
               stop0 + buffer). buffer = 0 and buffer = slidingWindow are both
               reported; the GATE uses buffer = slidingWindow (D-C2-1): TSB-AD
               reports window scores at the window centre, so with buffer = 0
               the anomaly's own window scores leak ~w/2 points into
               normal_test and inflate the 99th percentile. Percentile: numpy
               default (linear interpolation).

P2 (secondary) position of the test-region argmax vs the anomaly.
  P2_archive   the archive's rule, UCR_AnomalyDataSets.pptx slide 4 and
               Irrational Exuberance.pptx slide 64 (docs/ucr_supplement_text.md
               line 1495):  min(begin-L, begin-100) < P < max(end+L, end+100)
               with begin/end/P 1-based inclusive and L = end - begin + 1.
  P2_brief     BRIEF C2's stated window: start0 - 100 <= p0 < stop0 + 100.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PERCENTILE = 99.0
P2_SLACK = 100


@dataclass(frozen=True)
class Detection:
    n: int
    train_end: int
    start0: int
    stop0: int
    max_anom: float
    thr_b0: float
    thr_bw: float
    n_normal_b0: int
    n_normal_bw: int
    p1_b0: bool
    p1_bw: bool
    argmax0: int
    p2_archive: bool
    p2_brief: bool

    def as_row(self) -> dict:
        return dict(self.__dict__)


def normal_test_mask(n: int, train_end: int, start0: int, stop0: int, buffer: int) -> np.ndarray:
    """Boolean mask over [0, n): test region minus the anomaly ± buffer."""
    if not (0 < train_end < start0 < stop0 <= n):
        raise ValueError(f"bad layout: n={n} train_end={train_end} [{start0},{stop0})")
    if buffer < 0:
        raise ValueError("buffer must be >= 0")
    m = np.zeros(n, dtype=bool)
    m[train_end:] = True
    m[max(0, start0 - buffer):min(n, stop0 + buffer)] = False
    if not m.any():
        raise ValueError("normal_test is empty: the anomaly ± buffer covers the whole test region")
    return m


def p2_archive_rule(p0: int, start0: int, stop0: int, slack: int = P2_SLACK) -> bool:
    """Archive rule in its own 1-based frame. p0/start0/stop0 are 0-based."""
    begin, end = start0 + 1, stop0          # 1-based inclusive fields
    L = end - begin + 1
    P = p0 + 1
    return min(begin - L, begin - slack) < P < max(end + L, end + slack)


def p2_brief_rule(p0: int, start0: int, stop0: int, slack: int = P2_SLACK) -> bool:
    return (start0 - slack) <= p0 < (stop0 + slack)


def evaluate(score: np.ndarray, *, train_end: int, start0: int, stop0: int, window: int) -> Detection:
    score = np.asarray(score, dtype=np.float64)
    n = score.shape[0]
    if score.ndim != 1:
        raise ValueError("score must be 1-D")
    if not np.isfinite(score).all():
        raise ValueError("score contains non-finite values")
    m0 = normal_test_mask(n, train_end, start0, stop0, 0)
    mw = normal_test_mask(n, train_end, start0, stop0, window)
    max_anom = float(score[start0:stop0].max())
    thr0 = float(np.percentile(score[m0], PERCENTILE))
    thrw = float(np.percentile(score[mw], PERCENTILE))
    # argmax over the test region; ties -> first occurrence (np.argmax)
    argmax0 = int(train_end + np.argmax(score[train_end:]))
    return Detection(
        n=n, train_end=train_end, start0=start0, stop0=stop0, max_anom=max_anom,
        thr_b0=thr0, thr_bw=thrw, n_normal_b0=int(m0.sum()), n_normal_bw=int(mw.sum()),
        p1_b0=bool(max_anom > thr0), p1_bw=bool(max_anom > thrw),
        argmax0=argmax0,
        p2_archive=p2_archive_rule(argmax0, start0, stop0),
        p2_brief=p2_brief_rule(argmax0, start0, stop0),
    )
