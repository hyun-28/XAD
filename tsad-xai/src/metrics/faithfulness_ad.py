"""DDS / PES / CMI adapted to anomaly detection — INTERFACE ONLY, NOT IMPLEMENTED.

BRIEF2 §D2 is explicit: do not implement this. The original metrics
(src/metrics/faithfulness_simic.py) measure how the predicted class's
probability (0-100) falls as regions are perturbed. An anomaly detector has no
such quantity, and four choices have to be made by the researcher before any
adaptation can be written. They are registered as D7-D10 in docs/DECISIONS.md:

    D7  what scalar the perturbation curve tracks — max score inside the GT
        interval, mean score inside it, max over the whole series, ...
        (classification's "probability of the predicted class" has no AD twin)
    D8  normalisation — the original divides by 100. IForest scores are
        unbounded and are not probabilities, so DDS would leave [-1, 1].
        Candidates: min-max over the unperturbed score, a training-region
        quantile, the score's own IQR, ...
    D9  direction — is a falling score under MoRF "good"? Zero-masking a NORMAL
        region already lowers IForest's score (measured in the W1 dry run,
        docs/R5-SCREEN.md), so the sign convention is not obvious.
    D10 whether normal series regions enter the curve at all. The original uses
        every test sample; UCR has exactly one anomaly per series.

Everything here raises NotImplementedError on purpose.
"""
from __future__ import annotations

from typing import Callable, Literal, Sequence

import numpy as np

TrackedQuantity = Literal["gt_max", "gt_mean", "series_max"]           # D7
Normalisation = Literal["minmax_unperturbed", "train_quantile", "none"]  # D8

_DECISION_MSG = ("faithfulness_ad is an interface only (BRIEF2 §D2). Researcher decisions "
                 "D7-D10 in docs/DECISIONS.md must be made first.")


def perturbation_curve(score_fn: Callable[[np.ndarray], np.ndarray], x: np.ndarray,
                       order: Sequence[Sequence[int]], *, tracked: TrackedQuantity = "gt_max",
                       gt_slice: slice | None = None) -> np.ndarray:
    """The AD analogue of a MoRF/LeRF curve: one scalar per perturbation step.

    Parameters
    ----------
    score_fn : a fitted detector's `score` (src/detectors/base.py) — no refit.
    x        : the unperturbed series.
    order    : region index groups, most-important-first (MoRF) or least (LeRF);
               step k perturbs the union of the first k groups.
    tracked  : D7 — which scalar is read off each perturbed score vector.
    gt_slice : the labelled anomaly interval, 0-based half-open, for the
               GT-relative choices of `tracked`.

    Returns
    -------
    (len(order) + 1,) array; element 0 is the unperturbed value.
    """
    raise NotImplementedError(_DECISION_MSG + " (D7 tracked quantity, D10 which regions)")


def normalise_curve(curve: np.ndarray, *, method: Normalisation = "minmax_unperturbed",
                    reference: np.ndarray | None = None) -> np.ndarray:
    """D8: bring an unbounded AD perturbation curve onto the scale DDS assumes."""
    raise NotImplementedError(_DECISION_MSG + " (D8 normalisation)")


def dds_ad(morf: np.ndarray, lerf: np.ndarray, *, sign: Literal["drop_is_faithful",
                                                                "rise_is_faithful"] = "drop_is_faithful",
           max_diff: float | None = None) -> float:
    """DDS for anomaly detection: `faithfulness_simic.decaying_degradation_score`
    with the AD normalisation (D8) and sign convention (D9) applied first."""
    raise NotImplementedError(_DECISION_MSG + " (D8 normalisation, D9 direction)")
