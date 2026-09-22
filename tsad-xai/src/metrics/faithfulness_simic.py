"""DDS / PES / CMI — reimplementation of Šimić et al.'s faithfulness metrics.

ORIGIN
    https://github.com/perturbationeffect/cmi-am-validation-for-dl-ts-classifiers
    commit edc6a870b1a50fce3385bfce6a468583d696ea75, file `utils/res_utils.py`
    (`decaying_degradation_score` :51-78, `compute_dataset_dds` :31-49,
     `pes` :81-99, `CMI` :176-191).
    Licensed under the Apache License 2.0; the upstream LICENSE travels with the
    pinned clone in `third_party/simic_cmi/` (git-ignored, see docs/SIMIC_PORT.md).
    This file is an independent reimplementation of those four functions, kept
    numerically identical — `tests/test_simic_equivalence.py` compares it against
    the original source on 1,000 random curve pairs at atol=1e-12.

WHAT THE METRICS MEASURE (classification setting, the original's)
    A perturbation curve is the model's confidence in the predicted class after
    perturbing 0, 1, 2, ... regions of the input. MoRF removes the regions the
    attribution method calls most important first, LeRF the least important
    first. A faithful attribution makes the MoRF curve fall faster, so
    `lerf - morf` is positive.

    DDS  weighted mean of (lerf - morf), cubic decaying weights, normalised by
         the largest value that mean could take -> [-1, 1].
    PES  Kerby's simple difference f - u over a set of per-sample DDS values:
         the share with DDS > 0 minus the share with DDS < 0 -> [-1, 1].
    CMI  harmonic mean of |DDS| and |PES| when they share a sign, else 0 -> [0, 1].

THE NORMALISATION CONSTANT
    The original hard-codes 100 as "the maximum theoretical difference at each
    point" (`res_utils.py:73`), i.e. it assumes the model outputs a probability
    scaled to 0-100. It is exposed here as `max_diff` and defaults to 100, so
    this module reproduces the original exactly unless a caller says otherwise.
    Anomaly scores have no such bound — that is what `faithfulness_ad.py` is
    about, and it is a researcher decision (D7-D10), not something this file
    guesses.

DIFFERENCES FROM THE ORIGINAL (all deliberate, all listed in docs/SIMIC_PORT.md)
    * curves of length < 2 raise instead of returning nan with a RuntimeWarning
      (the original divides by a zero `dds_max`). BRIEF §1-1, fail loud.
    * unequal curve lengths, non-finite values and empty DDS sets raise.
    * the weights are computed once from the length, exactly as upstream.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

MAX_DIFF_DEFAULT = 100.0   # res_utils.py:73 — "the maximum theoretical difference"


def _curve_pair(morf: Sequence[float], lerf: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(morf, dtype=np.float64)
    b = np.asarray(lerf, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError(f"perturbation curves must be 1-D, got {a.shape} and {b.shape}")
    if a.shape != b.shape:
        raise ValueError(f"MoRF and LeRF curves must have the same length, got {a.size} and {b.size}")
    if a.size < 2:
        # upstream: max_diffed[1:] = 100 leaves an all-zero array, so dds_max == 0
        # and the result is nan (RuntimeWarning). We refuse instead.
        raise ValueError("perturbation curves must have at least 2 points "
                         "(the first point is unperturbed, so a length-1 curve has no signal)")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("perturbation curves contain non-finite values")
    return a, b


def cubic_weights(n: int) -> np.ndarray:
    """res_utils.py:67-68 — ((n, n-1, ..., 1) / n) ** 3."""
    if n < 1:
        raise ValueError("n must be >= 1")
    linear = np.arange(n, 0, -1, dtype=np.float64) / n
    return linear ** 3


def decaying_degradation_score(morf: Sequence[float], lerf: Sequence[float], *,
                               max_diff: float = MAX_DIFF_DEFAULT) -> float:
    """DDS of one sample (res_utils.py:51-78). Range [-1, 1] for curves in [0, max_diff]."""
    a, b = _curve_pair(morf, lerf)
    if not np.isfinite(max_diff) or max_diff <= 0:
        raise ValueError(f"max_diff must be a positive finite number, got {max_diff!r}")
    diffed = b - a
    w = cubic_weights(diffed.size)
    dds = float(np.average(diffed, weights=w))
    # the reachable maximum: the first point is unperturbed in both curves
    max_diffed = np.zeros_like(diffed)
    max_diffed[1:] = max_diff
    dds_max = float(np.average(max_diffed, weights=w))
    return dds / dds_max


def compute_dataset_dds(morf_curves: Iterable[Sequence[float]], lerf_curves: Iterable[Sequence[float]],
                        *, max_diff: float = MAX_DIFF_DEFAULT) -> np.ndarray:
    """One DDS per sample (res_utils.py:31-49)."""
    morf_curves, lerf_curves = list(morf_curves), list(lerf_curves)
    if len(morf_curves) != len(lerf_curves):
        raise ValueError("Number of MoRF and LeRF perturbation curves has to be equal")
    if not morf_curves:
        raise ValueError("no perturbation curves given")
    return np.array([decaying_degradation_score(m, l, max_diff=max_diff)
                     for m, l in zip(morf_curves, lerf_curves)], dtype=np.float64)


def pes(dds_values: Sequence[float]) -> float:
    """Perturbation effect size, Kerby's simple difference f - u (res_utils.py:81-99).

    Exact zeros count as neither favourable nor unfavourable, so |PES| < 1 when
    any DDS is exactly 0.
    """
    v = np.asarray(dds_values, dtype=np.float64)
    if v.ndim != 1:
        raise ValueError(f"dds_values must be 1-D, got shape {v.shape}")
    if v.size == 0:
        raise ValueError("dds_values is empty")
    if not np.isfinite(v).all():
        raise ValueError("dds_values contains non-finite values")
    f = float(np.count_nonzero(v > 0)) / v.size
    u = float(np.count_nonzero(v < 0)) / v.size
    return f - u


def cmi(dds: float, pes_value: float) -> float:
    """Consistency-magnitude index (res_utils.py:176-191).

    0 when the two disagree in sign OR either is exactly 0 (upstream branches on
    `pes * dds <= 0`); otherwise the harmonic mean of the magnitudes.
    """
    d, p = float(dds), float(pes_value)
    if not (np.isfinite(d) and np.isfinite(p)):
        raise ValueError(f"dds and pes must be finite, got {dds!r} and {pes_value!r}")
    if p * d <= 0:
        return 0.0
    return 2 / ((1 / abs(d)) + (1 / abs(p)))


# The upstream name is `CMI`; keep it as an alias so ported call sites read the same.
CMI = cmi
