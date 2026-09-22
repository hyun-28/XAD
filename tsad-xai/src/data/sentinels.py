"""The −999 missing-value sentinel (BRIEF_A-followup F3).

UCR_AnomalyDataSets.pptx slide 46: "In some legacy systems, missing values
are encoded at -999. Here we randomly inserted such a value" (resperation11).
Task A measured the same value occurring naturally in 23 plain series, often
in the training prefix. Everything here is read-only: no series is modified,
no value is interpolated (that is a research decision, out of scope).

Detection (F3-1)   exact: x == -999.0
                   near : |x + 999| <= tol, tol = near_mad_factor * MAD(x[:train_end])
                          (near is a superset of exact; MAD = 0 -> near == exact)
                   MEASURED 2026-09-22 (reports/data_audit.md §8): the 23 series
                   with exact hits are 12-bit-scale integer signals (range about
                   ±2047, training MAD 60–530), so tol = 3×MAD is 180–1,600 and
                   `near` sweeps in ordinary samples (6.2 M points over 68 series).
                   The near count is therefore reported but NOT used downstream;
                   classification, twin check and the exclusion set use `exact`.
                   DECISIONS.md D-F3-2.
Spike test         each exact index is also tested in its own series with the
                   local-outlier rule below (`spike_n`, `max_k`): is the −999 a
                   spike, or an ordinary sample of a signal whose range covers −999?
Classification     each `exact` index is in_gt / in_train / in_test_normal, using
(F3-3)             the active index convention for the GT interval.
Twin check (F3-2)  for a plain series, every DISTORTED / NOISE twin (same
                   name_group) is tested at the same 0-based exact indices with a
                   local-outlier rule: |x[i] - med_w(i)| > k * MAD_w(i), window w
                   centred on i. An index beyond the twin's length is
                   length_mismatch; no alignment is guessed.
Exclusion set      data/derived/sentinels/<num>.npy = sorted exact indices that
(F3-4)             are NOT in_gt (int64, 0-based). Written by 02_audit.py.
                   DECISION 2026-09-22 (D-F3-4): -999 is treated as an ordinary
                   measurement; the main analysis excludes nothing. The files
                   and the `no_sentinel` subset serve the sensitivity analysis.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SENTINEL_VALUE = -999.0


def mad(v: np.ndarray) -> float:
    v = np.asarray(v, dtype=np.float64)
    if v.size == 0:
        return float("nan")
    m = np.median(v)
    return float(np.median(np.abs(v - m)))


@dataclass
class SentinelResult:
    tol: float
    exact_idx: np.ndarray                       # 0-based, sorted
    near_idx: np.ndarray                        # 0-based, sorted, superset of exact
    in_gt: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    in_train: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    in_test_normal: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))

    @property
    def exclusion_idx(self) -> np.ndarray:
        """exact indices outside the GT interval (what downstream code masks out)."""
        return np.sort(np.concatenate([self.in_train, self.in_test_normal])).astype(np.int64)


def detect(x: np.ndarray, train_end: int, *, near_mad_factor: float = 3.0) -> SentinelResult:
    """F3-1. `x` is not modified (the caller may hash it before and after)."""
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"expected 1-D series, got shape {x.shape}")
    if not (0 < train_end <= len(x)):
        raise ValueError(f"train_end {train_end} outside (0, {len(x)}]")
    tol = near_mad_factor * mad(x[:train_end])
    if not np.isfinite(tol):
        raise ValueError("tolerance is not finite")
    exact = np.flatnonzero(x == SENTINEL_VALUE).astype(np.int64)
    near = np.flatnonzero(np.abs(x - SENTINEL_VALUE) <= tol).astype(np.int64)
    if not np.isin(exact, near).all():
        raise AssertionError("near must contain exact")
    return SentinelResult(tol=float(tol), exact_idx=exact, near_idx=near)


def classify(res: SentinelResult, *, train_end: int, start0: int, stop0: int) -> SentinelResult:
    """F3-3. Partition `exact_idx` into in_gt / in_train / in_test_normal.

    GT interval is 0-based half-open [start0, stop0) from conventions.py.
    Training prefix is x[:train_end] (convention-independent, D-A2-2).
    """
    if not (0 <= start0 < stop0):
        raise ValueError(f"bad GT interval [{start0}, {stop0})")
    if start0 < train_end:
        raise ValueError("GT interval overlaps the training prefix (A3 BEGIN_LE_TRAIN_END)")
    idx = res.exact_idx
    gt = (idx >= start0) & (idx < stop0)
    tr = idx < train_end
    res.in_gt = idx[gt]
    res.in_train = idx[tr]
    res.in_test_normal = idx[~gt & ~tr]
    if len(res.in_gt) + len(res.in_train) + len(res.in_test_normal) != len(idx):
        raise AssertionError("classification is not a partition")
    return res


def outlier_ratio(x: np.ndarray, i: int, *, window: int = 51) -> float:
    """|x[i] − median(x[i−h:i+h+1])| / MAD(same window), h = window // 2.

    The window is clipped at the series ends. MAD = 0 (flat window) gives inf
    for a nonzero deviation and 0.0 for a zero deviation.
    """
    if window < 3 or window % 2 == 0:
        raise ValueError("window must be an odd integer >= 3")
    if not (0 <= i < len(x)):
        raise IndexError(i)
    h = window // 2
    w = x[max(0, i - h):i + h + 1]
    med = np.median(w)
    m = np.median(np.abs(w - med))
    dev = abs(float(x[i]) - med)
    if m == 0.0:
        return float("inf") if dev > 0 else 0.0
    return float(dev / m)


def local_outlier(x: np.ndarray, i: int, *, window: int = 51, k: float = 10.0) -> bool:
    """outlier_ratio(x, i) > k."""
    return outlier_ratio(x, i, window=window) > k


TWIN_STATUSES = ("confirmed", "absent", "length_mismatch", "partial")


@dataclass
class TwinCheck:
    twin_num: int
    n_idx: int
    n_confirmed: int
    n_absent: int
    n_length_mismatch: int

    @property
    def status(self) -> str:
        """confirmed / absent / length_mismatch when unanimous over the in-range
        indices; `partial` when both confirmed and absent occur."""
        if self.n_length_mismatch == self.n_idx:
            return "length_mismatch"
        if self.n_absent == 0:
            return "confirmed"
        if self.n_confirmed == 0:
            return "absent"
        return "partial"

    def summary(self) -> str:
        s = f"{self.twin_num:03d}:{self.status}({self.n_confirmed}/{self.n_absent}/{self.n_length_mismatch})"
        return s


def twin_check(idx: np.ndarray, twin_x: np.ndarray, twin_num: int, *, window: int = 51,
               k: float = 10.0) -> TwinCheck:
    """F3-2 for one twin. `idx` are the plain series' sentinel indices (0-based)."""
    n_conf = n_abs = n_mis = 0
    for i in np.asarray(idx, dtype=np.int64):
        if i >= len(twin_x):
            n_mis += 1
        elif local_outlier(twin_x, int(i), window=window, k=k):
            n_conf += 1
        else:
            n_abs += 1
    return TwinCheck(twin_num=twin_num, n_idx=int(len(idx)), n_confirmed=n_conf,
                     n_absent=n_abs, n_length_mismatch=n_mis)
