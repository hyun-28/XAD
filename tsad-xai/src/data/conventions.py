"""Single source of truth for the archive's index convention (BRIEF A2).

Every consumer converts the raw filename fields through `to_half_open` and
`train_prefix_stop`; nothing else in the code base may do index arithmetic on
the raw fields. The active convention comes from configs/conventions.yaml,
where the documentary evidence for the default is listed.
"""
from __future__ import annotations

from enum import Enum

from src.config import load_yaml


class ConventionUndetermined(RuntimeError):
    """configs/conventions.yaml has index_convention: null (D3 undecided)."""


class IndexConvention(str, Enum):
    # begin/end are MATLAB indices (1-based, inclusive).  Evidence:
    # UCR_AnomalyDataSets.pptx slide 4 (L = end - begin + 1), slide 5
    # ("From 1 to X is training data", "-ascii"), slides 52/53/90-99
    # (T(start_anomaly:end_anomaly) injection code).
    MATLAB_1BASED_INCLUSIVE = "matlab_1based_inclusive"
    ZERO_BASED_INCLUSIVE = "zero_based_inclusive"
    ZERO_BASED_EXCLUSIVE = "zero_based_exclusive"


def active_convention() -> IndexConvention:
    raw = load_yaml("conventions").get("index_convention")
    if raw is None:
        raise ConventionUndetermined(
            "configs/conventions.yaml: index_convention is null (BRIEF §8 D3 undecided)")
    try:
        return IndexConvention(raw)
    except ValueError as e:
        raise ValueError(f"unknown index_convention {raw!r}; "
                         f"allowed: {[c.value for c in IndexConvention]}") from e


def approval_status() -> str:
    """Researcher sign-off recorded in configs/conventions.yaml (`approved`).

    Returns 'approved <date>' or 'sign-off pending'. Read by 02_audit.py so
    the report's D3 line is generated, never hand-edited (BRIEF §1-4).
    """
    approved = load_yaml("conventions").get("approved")
    return f"approved {approved}" if approved else "sign-off pending"


def to_half_open(begin_raw: int, end_raw: int,
                 convention: IndexConvention) -> tuple[int, int]:
    """Raw filename (begin, end) -> 0-based half-open [start0, stop0)."""
    if begin_raw < 0 or end_raw < 0:
        raise ValueError(f"negative index fields: begin={begin_raw} end={end_raw}")
    if convention is IndexConvention.MATLAB_1BASED_INCLUSIVE:
        if begin_raw < 1:
            raise ValueError(f"1-based begin must be >= 1, got {begin_raw}")
        start0, stop0 = begin_raw - 1, end_raw
    elif convention is IndexConvention.ZERO_BASED_INCLUSIVE:
        start0, stop0 = begin_raw, end_raw + 1
    elif convention is IndexConvention.ZERO_BASED_EXCLUSIVE:
        start0, stop0 = begin_raw, end_raw
    else:  # pragma: no cover - enum is closed
        raise AssertionError(convention)
    if stop0 <= start0:
        raise ValueError(f"empty anomaly interval under {convention.value}: "
                         f"begin={begin_raw} end={end_raw} -> [{start0}, {stop0})")
    return start0, stop0


def from_half_open(start0: int, stop0: int,
                   convention: IndexConvention) -> tuple[int, int]:
    """Inverse of `to_half_open` (used by the round-trip test)."""
    if stop0 <= start0:
        raise ValueError(f"empty interval [{start0}, {stop0})")
    if convention is IndexConvention.MATLAB_1BASED_INCLUSIVE:
        return start0 + 1, stop0
    if convention is IndexConvention.ZERO_BASED_INCLUSIVE:
        return start0, stop0 - 1
    if convention is IndexConvention.ZERO_BASED_EXCLUSIVE:
        return start0, stop0
    raise AssertionError(convention)  # pragma: no cover


def train_prefix_stop(train_end_raw: int) -> int:
    """0-based exclusive stop of the anomaly-free training prefix.

    Independent of the begin/end convention: slide 5 of
    UCR_AnomalyDataSets.pptx says "From 1 to X is training data", i.e. the
    first X points -> x[:X].
    """
    if train_end_raw < 1:
        raise ValueError(f"train_end must be >= 1, got {train_end_raw}")
    return train_end_raw
