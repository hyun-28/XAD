"""Detectors Z / R for W2 and the incremental scorer (BRIEF3 P2, P4).

    Z  stumpy.stump(T_A=x, m, T_B=train, ignore_trivial=False)  z-normalised MP (gate 1, D-C5-3)
    R  stumpy.aamp (T_A=x, m, T_B=train, ignore_trivial=False)  non-normalised MP, p = 2 (Euclidean)

T_A is the WHOLE series x, not x[train_end:]: that is what gate 1 ran
(src/detectors/matrixprofile.py, `score`), so Z's padded score reproduces the stored gate-1
score and its P99 threshold (DECISIONS D-P2-1). In an AB-join without an exclusion zone each
window of T_A is matched independently against T_B, so a test window's value does not depend
on where T_A starts (measured difference 4.5e-13 on #066, from stumpy's floating-point
recurrence).

INDEX CONVENTION (same as gate 1): profile[j] belongs to the window [j, j+m) and is written to
score index j + m//2; both ends are padded with the profile minimum (`_pad_like_tsb_ad`).

CONSTANT WINDOWS (Z only; stumpy 1.14.1, stump.py:201-204): both windows constant ->
pearson = 1 -> D = 0; exactly one constant -> pearson = 0.5 -> D = sqrt(2m(1-0.5)) = sqrt(m).
"constant" is ptp == 0 (core.py:2614). aamp has no such branch.

INCREMENTAL SCORING
    Erasing [a, a+r) changes exactly the windows whose start j satisfies j <= a+r-1 and
    j+m-1 >= a, i.e. j in [a-m+1, a+r-1] clipped to [0, n-m]. Only those are recomputed, from
    T_A = x'[lo : hi+m] with lo, hi the clipped bounds (= x'[a-m+1 : a+r+m-1] away from the
    edges); every other profile value is copied from the unperturbed profile.
    The copy is exact by construction. A full recomputation is NOT bit-exact on the untouched
    windows: stumpy accumulates the covariance along each diagonal, so the erased values enter
    and leave the sum with a rounding residual (measured up to 3.9e-10; DECISIONS D-P4-1).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.detectors.base import DetectorError
from src.detectors.matrixprofile import _pad_like_tsb_ad

KINDS = ("Z", "R")


def _mp(kind: str, t_a: np.ndarray, m: int, t_b: np.ndarray) -> np.ndarray:
    import stumpy
    if kind == "Z":
        out = stumpy.stump(t_a, m, T_B=t_b, ignore_trivial=False)
    elif kind == "R":
        out = stumpy.aamp(t_a, m, T_B=t_b, ignore_trivial=False)
    else:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    d = np.asarray(out[:, 0], dtype=np.float64)
    if d.size != t_a.size - m + 1 or not np.isfinite(d).all():
        raise DetectorError(f"{kind}: profile of length {d.size} (expected {t_a.size - m + 1}) "
                            "or non-finite values")
    return d


@dataclass(frozen=True)
class WindowClasses:
    """Window-start indices, by how they relate to an erased region [a, a+r)."""
    interior: np.ndarray     # [j, j+m) inside [a, a+r)
    boundary: np.ndarray     # overlaps [a, a+r) but not contained in it
    untouched: np.ndarray    # no overlap

    @property
    def affected(self) -> np.ndarray:
        return np.sort(np.concatenate([self.interior, self.boundary]))


def affected_range(a: int, r: int, m: int, n: int) -> tuple[int, int]:
    """Half-open range [lo, hi) of window starts touched by erasing [a, a+r)."""
    if r < 1 or a < 0 or a + r > n or m < 1 or m > n:
        raise ValueError(f"bad region a={a} r={r} m={m} n={n}")
    return max(0, a - m + 1), min(a + r - 1, n - m) + 1


def classify_windows(a: int, r: int, m: int, n: int) -> WindowClasses:
    lo, hi = affected_range(a, r, m, n)
    starts = np.arange(n - m + 1)
    affected = (starts >= lo) & (starts < hi)
    interior = (starts >= a) & (starts + m <= a + r)
    return WindowClasses(interior=starts[interior], boundary=starts[affected & ~interior],
                         untouched=starts[~affected])


class MPScorer:
    """Z or R against a fixed reference (the training prefix)."""

    def __init__(self, kind: str, m: int, reference: np.ndarray):
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        reference = np.asarray(reference, dtype=np.float64).ravel()
        if int(m) < 3:
            raise DetectorError(f"m={m} < 3 (stumpy minimum)")
        if reference.size < m or not np.isfinite(reference).all():
            raise DetectorError(f"reference of length {reference.size} unusable for m={m}")
        self.kind, self.m, self.reference = kind, int(m), reference.copy()
        self.reference.setflags(write=False)

    def profile(self, x: np.ndarray) -> np.ndarray:
        """Full recomputation: one value per window start, length n - m + 1."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 1 or x.size < self.m or not np.isfinite(x).all():
            raise DetectorError(f"input unusable: shape {x.shape}, m={self.m}")
        return _mp(self.kind, x, self.m, self.reference)

    def pad(self, profile: np.ndarray, n: int) -> np.ndarray:
        """Profile -> per-point score with gate 1's index convention."""
        return _pad_like_tsb_ad(profile, n, self.m)

    def incremental(self, x_new: np.ndarray, x_base: np.ndarray, base_profile: np.ndarray,
                    a: int, r: int) -> np.ndarray:
        """Profile of x_new, recomputing only the windows touched by [a, a+r).

        x_new must equal x_base outside [a, a+r) — checked, because the result is only right
        under that condition — and base_profile must be self.profile(x_base).
        """
        x_new = np.asarray(x_new, dtype=np.float64)
        x_base = np.asarray(x_base, dtype=np.float64)
        n, m = x_base.size, self.m
        if x_new.shape != x_base.shape or base_profile.shape != (n - m + 1,):
            raise DetectorError(f"shape mismatch: x_new {x_new.shape}, x_base {x_base.shape}, "
                                f"profile {base_profile.shape}")
        if not (np.array_equal(x_new[:a], x_base[:a]) and np.array_equal(x_new[a + r:], x_base[a + r:])):
            raise DetectorError(f"x_new differs from x_base outside [{a}, {a + r})")
        lo, hi = affected_range(a, r, m, n)
        out = np.array(base_profile, dtype=np.float64, copy=True)
        out[lo:hi] = _mp(self.kind, x_new[lo:hi - 1 + m], m, self.reference)
        return out
