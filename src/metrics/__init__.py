"""Metrics for the three experiment families.

artifact.py       Stage 3  -- does perturbation manufacture anomalies?
faithfulness.py   Stage 5  -- the incumbent protocol (arXiv:2601.19017), ported
localization.py   Stage 6  -- label-based, perturbation-free
"""
import numpy as np


def fisher_mean(rhos):
    """Average correlations through the Fisher z transform.

    Correlation coefficients are not additive; arXiv:2601.19017 uses exactly
    this procedure to aggregate over test samples.
    """
    r = np.asarray([x for x in np.asarray(rhos, float).ravel() if np.isfinite(x)])
    if r.size == 0:
        return np.nan
    r = np.clip(r, -0.999999, 0.999999)
    z = np.arctanh(r).mean()
    return float(np.tanh(z))


def bootstrap_ci(values, stat=np.median, n=1000, alpha=0.05, rng=None):
    rng = np.random.default_rng(0) if rng is None else rng
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (np.nan, np.nan)
    draws = [stat(rng.choice(v, size=v.size, replace=True)) for _ in range(n)]
    return (float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2)))
