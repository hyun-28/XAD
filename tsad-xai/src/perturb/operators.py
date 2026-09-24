"""W2 rev1 masking operators (E1, §6): B1-B6 unchanged, recon_test / recon_train (D11), identity.

INTERFACE   apply(name, x, runs, rng, ctx) -> x_perturbed
    runs  list of half-open [a, b), 0-based, pairwise disjoint
    rng   np.random.Generator (required for stochastic operators; unused otherwise)
    ctx   Ctx(train_end, gt=(start0, stop0), w)
    x is never modified; a new float64 array is returned.

B1-B6 are the functions of `src/perturbations.py` (the definitions `experiments/exp_a_artifact.py`
imports: `from src.perturbations import PERTURBATIONS, IDENTITY`), called unchanged with
idx = arange(a, b). For one run this IS the legacy call. For several runs each run's replacement is
computed from the UNPERTURBED x and written into its own [a, b) only (DECISIONS D-E1-2): the
legacy functions take one index array and treat [min(idx), max(idx)] as one span (B3's context,
B4's interpolation), which would overwrite the points between runs.

recon_test / recon_train follow the D11 text (PENDING, brief rev1 §3) literally; see `_recon`.
The metadata in `META` mirrors configs/operators.yaml (checked by tests/test_operators_w2.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.perturbations import IDENTITY, PERTURBATIONS


class OperatorError(ValueError):
    """Operator called outside its definition."""


class NoDonorError(OperatorError):
    """recon_* found no admissible donor (rev1 §11 counts these)."""


@dataclass(frozen=True)
class Ctx:
    train_end: int
    gt: tuple[int, int]          # (start0, stop0), 0-based half-open
    w: int


# canonical_shape: whether the z-normalised output inside a run is fixed regardless of input
# (rev1 §6). "constant": std 0 by construction; "linear": a straight line (B4 — z-normalises to
# the same ramp for every slope, D11 note); "none": depends on the input.
META = {
    "B1_zero":          {"canonical_shape": "constant", "uses_reference_set": False, "stochastic": False},
    "B2_global_mean":   {"canonical_shape": "constant", "uses_reference_set": False, "stochastic": False},
    "B3_local_mean":    {"canonical_shape": "constant", "uses_reference_set": False, "stochastic": False},
    "B4_linear_interp": {"canonical_shape": "linear",   "uses_reference_set": False, "stochastic": False},
    "B5_gaussian":      {"canonical_shape": "none",     "uses_reference_set": False, "stochastic": True},
    "B6_shuffle":       {"canonical_shape": "none",     "uses_reference_set": False, "stochastic": True},
    "recon_test":       {"canonical_shape": "none",     "uses_reference_set": False, "stochastic": False},
    "recon_train":      {"canonical_shape": "none",     "uses_reference_set": True,  "stochastic": False},
    "identity":         {"canonical_shape": "none",     "uses_reference_set": False, "stochastic": False},
}
NAMES = tuple(META)


def influence(a: int, b: int, w: int) -> tuple[int, int]:
    """rev1 §2: I(R) = [a - w + 1, b + w - 1) — the only scores that change when R is perturbed."""
    return a - w + 1, b + w - 1


def exclusion(ctx: Ctx) -> tuple[int, int]:
    """rev1 §2: E = [GT_start - w, GT_stop + w)."""
    return ctx.gt[0] - ctx.w, ctx.gt[1] + ctx.w


def context_len(w: int) -> int:
    """D11: c = max(1, w // 4)."""
    return max(1, w // 4)


def _check_runs(x: np.ndarray, runs) -> list[tuple[int, int]]:
    rs = sorted((int(a), int(b)) for a, b in runs)
    for a, b in rs:
        if not (0 <= a < b <= x.size):
            raise OperatorError(f"bad run [{a}, {b}) for n={x.size}")
    for (a0, b0), (a1, b1) in zip(rs, rs[1:]):
        if a1 < b0:
            raise OperatorError(f"runs overlap: [{a0}, {b0}) and [{a1}, {b1})")
    return rs


def _overlaps(u0, u1, iv) -> bool:
    return u0 < iv[1] and iv[0] < u1


def _sq_dist(x: np.ndarray, starts: np.ndarray, q: np.ndarray, chunk: int = 65536) -> np.ndarray:
    """sum((x[s:s+len(q)] - q)**2) for every s in starts, exactly (no FFT/cumsum shortcuts, so ties
    are decided on the same arithmetic everywhere)."""
    from numpy.lib.stride_tricks import sliding_window_view
    sw = sliding_window_view(x, q.size)
    out = np.empty(starts.size, dtype=np.float64)
    for i in range(0, starts.size, chunk):
        s = starts[i:i + chunk]
        out[i:i + chunk] = ((sw[s] - q) ** 2).sum(axis=1)
    return out


def recon_donor(x: np.ndarray, run: tuple[int, int], runs, ctx: Ctx, source: str) -> tuple[int, float]:
    """D11 donor for one run. Returns (u, squared context distance).

    Candidates u: [u, u+L) inside the source region (test: [train_end, n); train: [0, train_end)),
    not overlapping E, I(R) of the run, I(.) of every other run, or any run's edge-context intervals
    [a-c, a) and [b, b+c); u - c >= 0 and u + L + c <= n so the donor's own context exists.
    Criterion: raw Euclidean distance between x[a-c:a] || x[b:b+c] and x[u-c:u] || x[u+L:u+L+c];
    ties -> smallest u. Donor values come from the unperturbed x.
    """
    n, w = x.size, ctx.w
    a, b = run
    L, c = b - a, context_len(w)
    if a - c < 0 or b + c > n:
        raise NoDonorError(f"run [{a}, {b}): its own context [{a - c}, {b + c}) leaves [0, {n})")
    lo, hi = (ctx.train_end, n) if source == "test" else (0, ctx.train_end)
    forbidden = [exclusion(ctx)]
    for ra, rb in runs:
        forbidden += [influence(ra, rb, w), (ra - c, ra), (rb, rb + c)]
    u = np.arange(max(lo, c), min(hi, n - c) - L + 1)
    if u.size:
        ok = np.ones(u.size, dtype=bool)
        for f0, f1 in forbidden:
            ok &= ~((u < f1) & (f0 < u + L))
        u = u[ok]
    if u.size == 0:
        raise NoDonorError(f"run [{a}, {b}): no admissible {source} donor of length {L}")
    d = _sq_dist(x, u - c, x[a - c:a]) + _sq_dist(x, u + L, x[b:b + c])
    k = int(np.argmin(d))                           # first minimum = smallest u
    return int(u[k]), float(d[k])


def apply(name: str, x: np.ndarray, runs, rng: np.random.Generator | None, ctx: Ctx) -> np.ndarray:
    if name not in META:
        raise OperatorError(f"unknown operator {name!r}; known: {NAMES}")
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise OperatorError("x must be a finite 1-D array")
    rs = _check_runs(x, runs)
    if META[name]["stochastic"] and rng is None:
        raise OperatorError(f"{name} is stochastic: pass a seeded np.random.Generator")
    if name == "identity":
        return IDENTITY(x, np.empty(0, dtype=int))      # src/perturbations.py:124
    out = x.copy()
    if name in ("recon_test", "recon_train"):
        src = name.split("_")[1]
        for run in rs:
            u, _ = recon_donor(x, run, rs, ctx, src)
            out[run[0]:run[1]] = x[u:u + run[1] - run[0]]
        return out
    fn = PERTURBATIONS[name]
    for a, b in rs:
        y = np.asarray(fn(x, np.arange(a, b), rng=rng), dtype=np.float64)
        if y.shape != x.shape:
            raise OperatorError(f"{name} changed the shape")
        out[a:b] = y[a:b]
    return out
