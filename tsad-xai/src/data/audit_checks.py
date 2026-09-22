"""A3 loading-integrity checks and A2 empirical index-convention evidence."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.data.ucr import UcrMeta

# Flag codes written to manifest.audit_flags (';'-separated).
# severity: STOP  -> BRIEF §7-5 (GT contradiction in the filename) / §1-1
#           WARN  -> exit code != 0 but not a STOP
#           INFO  -> recorded only
SEVERITY = {
    "NOT_1D_FLOAT64": "STOP",
    "NAN_OR_INF": "STOP",
    "END_GT_LEN": "STOP",           # end beyond the series (1-based: end > len)
    "BEGIN_GT_END": "STOP",
    "BEGIN_LE_TRAIN_END": "STOP",   # anomaly inside the training prefix
    "TRAIN_END_GE_LEN": "STOP",     # no test region
    "CONST_TRAIN_PREFIX": "WARN",   # z-normalisation would divide by zero
    "END_EQ_LEN": "INFO",           # anomaly touches the last point
    "SINGLE_LINE_FORMAT": "INFO",   # slide-7 format variant
    "NOT_IN_DECK": "INFO",
    "DECK_FIELDS_DIFFER": "INFO",
    "DOMAIN_OFFICIAL_NA": "INFO",
    "DOMAIN_DISAGREE": "INFO",
    "HAS_MINUS999": "INFO",         # legacy missing-value sentinel present (slide 46)
    # --- BRIEF_A-followup (2026-09-22) ---
    # All three were WARN/STOP until the researcher's decisions of 2026-09-22
    # (DECISIONS.md D-F2-2, D-F3-4): -999 is an ordinary measurement, not a
    # missing-value marker, and content_group is the unit of analysis. They
    # remain in the manifest as INFO so the facts stay visible.
    "SENTINEL_IN_NORMAL": "INFO",   # F3: exact -999 in the training prefix or the normal test region
    "SENTINEL_IN_GT_UNDOCUMENTED": "INFO",  # F3: exact -999 inside GT, deck silent on -999 (123 ECG4)
    "CONTENT_GROUP_CROSSES_NAME": "INFO",   # F2: content_group spans >1 name_group (18 groups)
}


@dataclass
class A3Result:
    length: int
    flags: list[str] = field(default_factory=list)
    train_std: float = float("nan")
    n_nonfinite: int = 0
    n_minus999: int = 0


def check_series(x: np.ndarray, meta: UcrMeta, layout: dict) -> A3Result:
    """All A3 checks on one series. Collects, never raises (audit mode)."""
    r = A3Result(length=int(x.shape[0]))
    if x.ndim != 1 or x.dtype != np.float64:
        r.flags.append("NOT_1D_FLOAT64")
    r.n_nonfinite = int((~np.isfinite(x)).sum())
    if r.n_nonfinite:
        r.flags.append("NAN_OR_INF")
    n, te, b, e = r.length, meta.train_end_raw, meta.begin_raw, meta.end_raw
    # The raw fields are compared in their own (1-based inclusive) frame, so
    # the checks below do not depend on configs/conventions.yaml. A 1-based
    # end may legitimately equal len(x).
    if e > n:
        r.flags.append("END_GT_LEN")
    elif e == n:
        r.flags.append("END_EQ_LEN")
    if b > e:
        r.flags.append("BEGIN_GT_END")
    if b <= te:
        r.flags.append("BEGIN_LE_TRAIN_END")
    if te >= n:
        r.flags.append("TRAIN_END_GE_LEN")
    prefix = x[:te] if te <= n else x
    if prefix.size:
        r.train_std = float(np.std(prefix))
        if r.train_std == 0.0:
            r.flags.append("CONST_TRAIN_PREFIX")
    if layout["single_line"]:
        r.flags.append("SINGLE_LINE_FORMAT")
    # Slide 46 says a -999 "missing value" was inserted as the anomaly of
    # resperation11; measured 2026-09-21: the same sentinel occurs naturally
    # elsewhere in that record, which matters for any detector run later.
    r.n_minus999 = int((x == -999.0).sum())
    if r.n_minus999:
        r.flags.append("HAS_MINUS999")
    return r


def _local_deviation(x: np.ndarray, i: int, guard: int = 2, reach: int = 12) -> float:
    """|x[i] - median of neighbours| with a +-guard gap around i."""
    left = x[max(0, i - reach):max(0, i - guard)]
    right = x[i + guard + 1:i + reach + 1]
    nb = np.concatenate([left, right])
    if nb.size == 0:
        return float("nan")
    return float(abs(x[i] - np.median(nb)))


def index_evidence(x: np.ndarray, meta: UcrMeta) -> dict:
    """Contrast the two readings of a very short anomaly's filename fields.

    H1 (MATLAB 1-based inclusive) puts the anomaly at 0-based [begin-1, end);
    H0 (0-based inclusive) puts it at [begin, end+1). For each hypothesis the
    score is the mean local deviation (|x[i] - median of its neighbours|,
    neighbours = i-12..i-3 and i+3..i+12) over the hypothesised points.
    ratio = S(H1) / S(H0); >= 2 favours H1, <= 0.5 favours H0, otherwise
    the series is not decisive (e.g. time-warp anomalies have no outlying
    point). Purely descriptive: the decision is D3.
    """
    b, e = meta.begin_raw, meta.end_raw
    h1 = list(range(b - 1, e))
    h0 = list(range(b, e + 1))
    s1 = float(np.mean([_local_deviation(x, i) for i in h1]))
    s0 = float(np.mean([_local_deviation(x, i) for i in h0]))
    ratio = s1 / s0 if s0 > 0 else (float("inf") if s1 > 0 else float("nan"))
    if ratio >= 2:
        verdict = "H1 (1-based)"
    elif ratio <= 0.5:
        verdict = "H0 (0-based)"
    else:
        verdict = "not decisive"
    return {"H1_idx0": f"[{b-1},{e})", "H0_idx0": f"[{b},{e+1})",
            "dev_H1": s1, "dev_H0": s0, "ratio_H1_over_H0": ratio, "verdict": verdict}
