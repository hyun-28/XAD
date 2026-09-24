"""Šimić et al. perturbation methods (PMs), ported for a long series; plus ContextReconstruct.

UPSTREAM
    third_party/simic_cmi/utils/subsequence_perturbation.py at edc6a870 (D-D1-1).
    Every upstream PM is a class holding `self.original` (a copy of ONE classification
    sample) and `self.sample` (the array it mutates), with
    `perturb_subsequence(start, end)` writing `self.sample[start:end]`:

        Zero                   :94       sample[s:e] = 0
        SampleMean             :58       sample[s:e] = mean(original)
        OutOfDistHigh          :121-122  sample[s:e] = max(|original.max()|, |original.min()|) * 100
        Inverse                :100      sample[s:e] = original.max() - sample[s:e]   (NOT a sign flip)
        UniformNoise100        :21-24    noise = np.random.uniform(-1, 1, len(original)) (bounds
                                         hard-coded; unseeded global RNG); sample[s:e] = noise[s:e]
        LinearInterpolation    :135-144  sample[s:e] = nan; an edge NaN becomes mean(original);
                                         pandas Series.interpolate('linear')
        NearestNeighborWindow  :235-263  neighbours chosen by POSITION, not similarity: the
                                         ceil(r/2) points left of s followed by the floor(r/2)
                                         points right of e; right-only at s == 0, left-only when
                                         e > len - r. Never reads sample[s:e] itself.

    The functions below take the two roles apart:
        canvas  the array positions refer to (upstream `self.sample`)
        ref     the statistics source (upstream `self.original`)
    With ref = canvas they are upstream literally; tests/test_simic_operators.py proves that
    against the imported upstream classes, bit for bit.

ADAPTATION TO A LONG SERIES (researcher decision 2026-09-23, DECISIONS D-P3-1)
    Upstream data were standardised with the training set's mean/std (all 125 saved
    `training_conf.json` have "data_normalization_method": "standard";
    utils/utils.py:67-73). To keep each PM's meaning (Zero = mean level, U(-1,1) = ±1 sigma,
    OutOfDist = 100 x abs-max in sigma units):
        1. z = (x - mu) / sigma, mu/sigma = mean/std (ddof 0, as utils.py:71) of x[:train_end]
        2. canvas = z (the whole series: the erased region lies in the test part, so positions
           must refer to the full series), ref = z[:train_end]  (the "sample")
        3. apply the PM to canvas[a:a+r]
        4. x' = x with x'[a:a+r] = z'[a:a+r] * sigma + mu. ONLY the erased region is written
           back: a round trip through (x - mu) / sigma * sigma + mu is not bit-exact, and the
           untouched points must be.
    UniformNoise100's noise vector has the length of the CANVAS (upstream: len(original), which
    is the same thing when the sample is the canvas); it is drawn from
    np.random.RandomState(seed), the generator behind upstream's np.random.uniform after
    np.random.seed(seed).
    Consequence: Zero writes mu and SampleMean writes mean(z_train) * sigma + mu, which is mu up
    to rounding. The pilot checks whether the two are bit-identical (S5); nothing here forces it.

CONTEXT RECONSTRUCT (ours, researcher definition 2026-09-23, docs/PREREG_W2.md appendix)
    query = x[a-m:a]; j = argmin over z-normalised Euclidean distance between the query and
    x_train[j:j+m], subject to j + m + r <= train_end; fill =
    x_train[j+m : j+m+r] + (x[a-1] - x_train[j+m-1]). The erased region is never read.
    z-normalised distance is invariant and the fill equivariant under x -> (x - mu)/sigma, so
    running it in the standardised space gives the same fill up to rounding.

Every function returns a NEW array and never mutates its inputs (upstream mutates in place).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

UPSTREAM_COMMIT = "edc6a870b1a50fce3385bfce6a468583d696ea75"


class OperatorError(ValueError):
    """An operator was called outside the conditions it is defined for."""


def _check(canvas: np.ndarray, s: int, e: int) -> np.ndarray:
    canvas = np.array(canvas, dtype=np.float64, copy=True)
    if canvas.ndim != 1:
        raise OperatorError(f"canvas must be 1-D, got shape {canvas.shape}")
    if not (0 <= s < e <= canvas.size):
        raise OperatorError(f"bad region [{s}, {e}) for length {canvas.size}")
    if not np.isfinite(canvas).all():
        raise OperatorError("canvas contains NaN/inf")
    return canvas


# --- the seven upstream PMs (canvas / ref split; ref = canvas is upstream literally) ---------

def zero(canvas, s, e, *, ref=None, **_):
    out = _check(canvas, s, e)
    out[s:e] = 0
    return out


def sample_mean(canvas, s, e, *, ref, **_):
    out = _check(canvas, s, e)
    out[s:e] = np.mean(ref)
    return out


def out_of_dist_high(canvas, s, e, *, ref, **_):
    out = _check(canvas, s, e)
    ref = np.asarray(ref)
    out[s:e] = max(abs(ref.max()), abs(ref.min())) * 100
    return out


def inverse(canvas, s, e, *, ref, **_):
    out = _check(canvas, s, e)
    out[s:e] = np.asarray(ref).max() - out[s:e]
    return out


def uniform_noise100(canvas, s, e, *, seed, **_):
    out = _check(canvas, s, e)
    if seed is None:
        raise OperatorError("UniformNoise100 needs a fixed seed (upstream used the unseeded global RNG)")
    noise = np.random.RandomState(int(seed)).uniform(-1, 1, out.size)
    out[s:e] = noise[s:e]
    return out


def linear_interpolation(canvas, s, e, *, ref, **_):
    import pandas as pd
    out = _check(canvas, s, e)
    out[s:e] = np.nan
    if np.isnan(out[0]):
        out[0] = np.mean(ref)
    if np.isnan(out[-1]):
        out[-1] = np.mean(ref)
    return pd.Series(out).interpolate(method="linear").to_numpy()


def nearest_neighbor_window(canvas, s, e, **_):
    out = _check(canvas, s, e)
    src = out.copy()                   # upstream reads self.sample before writing it
    w = e - s
    # Where a neighbour would start before 0 or end after len, upstream slices with a
    # negative/overlong index and numpy then fails to broadcast; it never returns a
    # result there. We raise the same failure explicitly.
    if s == 0:
        if e + w > out.size:
            raise OperatorError(f"NearestNeighborWindow: right neighbour [{e}, {e + w}) past {out.size}")
        out[s:e] = src[s + w:e + w]
    elif e > out.size - w:
        if s - w < 0:
            raise OperatorError(f"NearestNeighborWindow: left neighbour starts at {s - w} < 0")
        out[s:e] = src[s - w:e - w]
    else:
        left = w // 2 + (w % 2)        # upstream :248-251
        right = w // 2
        if s - left < 0:
            raise OperatorError(f"NearestNeighborWindow: left half starts at {s - left} < 0")
        out[s:e] = np.concatenate([src[s - left:s], src[e:e + right]])
    return out


# --- ours ------------------------------------------------------------------------------------

@dataclass(frozen=True)
class CRMatch:
    j: int
    dist: float


def context_reconstruct_match(canvas, s, e, *, m: int, train_end: int) -> CRMatch:
    """The reference position ContextReconstruct copies from (see module docstring)."""
    import stumpy
    canvas = np.asarray(canvas, dtype=np.float64)
    r = e - s
    if m < 3:
        raise OperatorError(f"m={m} < 3")
    if s < train_end:
        raise OperatorError(f"erased region starts at {s} < train_end {train_end}: the search "
                            "region would overlap the erased points")
    if s - m < 0:
        raise OperatorError(f"no length-{m} context left of {s}")
    last_j = train_end - m - r                     # j + m + r <= train_end
    if last_j < 0:
        raise OperatorError(f"training region {train_end} too short for m={m}, r={r}")
    query = canvas[s - m:s]
    hay = canvas[:last_j + m]                      # windows start at 0..last_j
    d = np.asarray(stumpy.mass(query, hay), dtype=np.float64)
    if d.size != last_j + 1 or not np.isfinite(d).all():
        raise OperatorError(f"mass returned {d.size} values (expected {last_j + 1}) or non-finite")
    j = int(np.argmin(d))                          # ties -> smallest j
    return CRMatch(j=j, dist=float(d[j]))


def context_reconstruct(canvas, s, e, *, m: int, train_end: int, match: CRMatch | None = None, **_):
    out = _check(canvas, s, e)
    if match is None:
        match = context_reconstruct_match(out, s, e, m=m, train_end=train_end)
    j, r = match.j, e - s
    out[s:e] = out[j + m:j + m + r] + (out[s - 1] - out[j + m - 1])
    return out


# --- registry and the series-level wrapper ---------------------------------------------------

OPERATORS = {
    "Zero": zero,
    "SampleMean": sample_mean,
    "OutOfDistHigh": out_of_dist_high,
    "Inverse": inverse,
    "UniformNoise100": uniform_noise100,
    "LinearInterpolation": linear_interpolation,
    "NearestNeighborWindow": nearest_neighbor_window,
    "ContextReconstruct": context_reconstruct,
}
# Name of the upstream class each operator reproduces (None = not an upstream PM).
UPSTREAM_CLASS = {k: k for k in OPERATORS if k != "ContextReconstruct"} | {"ContextReconstruct": None}
# Points outside [a, a+r) an operator reads, as (left, right) extents for a given r and m.
# Used by the pilot to keep every read inside the certain-normal region.
READ_EXTENT = {
    "LinearInterpolation": lambda r, m: (1, 1),
    "NearestNeighborWindow": lambda r, m: (math.ceil(r / 2), r // 2),
    "ContextReconstruct": lambda r, m: (m, 0),
}


@dataclass(frozen=True)
class Space:
    """How the upstream "sample" maps onto a long series (configs/w2_pilot.yaml `operator_space`)."""
    standardize: bool          # apply in the (x - mu)/sigma space and map back
    reference: str             # "train" -> x[:train_end]; "full" -> x


def erase(x: np.ndarray, op: str, a: int, r: int, *, train_end: int, m: int, space: Space,
          seed: int | None = None) -> tuple[np.ndarray, dict]:
    """x with [a, a+r) replaced by operator `op`. Returns (x', info). Only [a, a+r) changes."""
    if op not in OPERATORS:
        raise OperatorError(f"unknown operator {op!r}; known: {sorted(OPERATORS)}")
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if not (0 < train_end < n):
        raise OperatorError(f"train_end {train_end} outside (0, {n})")
    if r < 1 or not (0 <= a and a + r <= n):
        raise OperatorError(f"bad region a={a} r={r} for n={n}")
    if space.reference not in ("train", "full"):
        raise OperatorError(f"space.reference must be train|full, got {space.reference!r}")
    ref_raw = x[:train_end] if space.reference == "train" else x
    if space.standardize:
        mu, sigma = float(np.mean(ref_raw)), float(np.std(ref_raw))
        if not (sigma > 0 and np.isfinite(sigma)):
            raise OperatorError(f"reference std is {sigma}; cannot standardise")
        canvas = (x - mu) / sigma
    else:
        mu, sigma = 0.0, 1.0
        canvas = x.copy()
    ref = canvas[:train_end] if space.reference == "train" else canvas
    info: dict = {"mu": mu, "sigma": sigma}
    extra: dict = {}
    if op == "ContextReconstruct":
        extra["match"] = context_reconstruct_match(canvas, a, a + r, m=m, train_end=train_end)
        info.update(cr_j=extra["match"].j, cr_dist=extra["match"].dist)
    out = OPERATORS[op](canvas, a, a + r, ref=ref, seed=seed, m=m, train_end=train_end, **extra)
    if out.shape != canvas.shape:
        raise OperatorError(f"{op} changed the shape {canvas.shape} -> {out.shape}")
    if not (np.array_equal(out[:a], canvas[:a]) and np.array_equal(out[a + r:], canvas[a + r:])):
        raise OperatorError(f"{op} changed points outside [{a}, {a + r})")
    new = x.copy()
    new[a:a + r] = out[a:a + r] * sigma + mu if space.standardize else out[a:a + r]
    if not np.isfinite(new).all():
        raise OperatorError(f"{op} produced NaN/inf")
    return new, info
