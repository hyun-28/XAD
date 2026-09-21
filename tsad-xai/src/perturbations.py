"""Perturbation operators B1-B6.

Contract for every operator -- enforced by tests/test_perturbations.py:
    op(X, idx, rng=None) -> X'
      * returns a NEW array; never mutates the input
      * preserves shape exactly
      * is the identity when idx is empty
      * is deterministic given rng

X is (L,) or (L, C). idx is an integer index array over the time axis.

B1_zero is the operator used by arXiv:2601.19017 (frequency bands set to zero).
It is included as the incumbent, not as a recommendation.
"""
import numpy as np


def _prep(X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        return X[:, None].copy(), True
    return X.copy(), False


def _out(Xw, was_1d):
    return Xw[:, 0] if was_1d else Xw


def _idx(idx):
    return np.asarray(idx, dtype=int).ravel()


def b1_zero(X, idx, rng=None):
    """Replace the masked region with 0."""
    Xw, f = _prep(X)
    idx = _idx(idx)
    if idx.size:
        Xw[idx, :] = 0.0
    return _out(Xw, f)


def b2_global_mean(X, idx, rng=None):
    """Replace with the per-channel global mean of the series."""
    Xw, f = _prep(X)
    idx = _idx(idx)
    if idx.size:
        Xw[idx, :] = Xw.mean(axis=0)
    return _out(Xw, f)


def b3_local_mean(X, idx, rng=None, pad=50):
    """Replace with the mean of the surrounding context window."""
    Xw, f = _prep(X)
    idx = _idx(idx)
    if not idx.size:
        return _out(Xw, f)
    L = Xw.shape[0]
    lo, hi = int(idx.min()), int(idx.max())
    ctx = np.r_[np.arange(max(0, lo - pad), lo),
                np.arange(hi + 1, min(L, hi + 1 + pad))]
    ctx = np.setdiff1d(ctx, idx)
    Xw[idx, :] = Xw[ctx, :].mean(axis=0) if ctx.size else Xw.mean(axis=0)
    return _out(Xw, f)


def b4_linear_interp(X, idx, rng=None):
    """Linearly interpolate across the masked run using its bracketing points."""
    Xw, f = _prep(X)
    idx = _idx(idx)
    if not idx.size:
        return _out(Xw, f)
    L, C = Xw.shape
    lo, hi = int(idx.min()), int(idx.max())
    if lo > 0:
        a = Xw[lo - 1]
    elif hi + 1 < L:
        a = Xw[hi + 1]
    else:
        a = Xw[lo]
    b = Xw[hi + 1] if hi + 1 < L else a
    n = hi - lo + 1
    for c in range(C):
        Xw[lo:hi + 1, c] = np.linspace(a[c], b[c], n + 2)[1:-1]
    return _out(Xw, f)


def b5_gaussian(X, idx, rng=None, scale=1.0):
    """Replace with Gaussian noise matched to the series mean/std."""
    Xw, f = _prep(X)
    idx = _idx(idx)
    if not idx.size:
        return _out(Xw, f)
    rng = np.random.default_rng(0) if rng is None else rng
    mu = Xw.mean(axis=0)
    sd = np.where(Xw.std(axis=0) == 0, 1e-8, Xw.std(axis=0)) * scale
    Xw[idx, :] = rng.normal(mu, sd, size=(idx.size, Xw.shape[1]))
    return _out(Xw, f)


def b6_shuffle(X, idx, rng=None):
    """Shuffle values within the masked run only.

    Preserves the local value distribution exactly, so any score change is
    attributable to temporal structure rather than to value-range artefacts.
    """
    Xw, f = _prep(X)
    idx = _idx(idx)
    if not idx.size:
        return _out(Xw, f)
    rng = np.random.default_rng(0) if rng is None else rng
    Xw[idx, :] = Xw[rng.permutation(idx), :]
    return _out(Xw, f)


PERTURBATIONS = {
    "B1_zero": b1_zero,
    "B2_global_mean": b2_global_mean,
    "B3_local_mean": b3_local_mean,
    "B4_linear_interp": b4_linear_interp,
    "B5_gaussian": b5_gaussian,
    "B6_shuffle": b6_shuffle,
}

IDENTITY = lambda X, idx, rng=None: np.asarray(X, dtype=float).copy()
