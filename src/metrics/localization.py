"""Stage 6 -- label-based localization. No perturbation anywhere in this file.

This is what breaks the circularity: the ground truth comes from the dataset
labels, not from the same masking mechanism that Occlusion/KernelSHAP use to
build their attributions.

TERMINOLOGY DISCIPLINE
----------------------
Localization measures PLAUSIBILITY, not faithfulness. arXiv:2601.19017 draws
that distinction explicitly and it must be preserved in the paper. The claim is
never "localization is a better faithfulness metric". The claim is that TSAD
lets us measure both axes independently, and that they disagree.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def localization_precision(relevance, label):
    """Fraction of the explanation's mass that lands on labelled anomalies."""
    r = np.asarray(relevance, float).ravel()
    y = np.asarray(label, int).ravel()
    s = r.sum()
    r = r / s if s > 0 else np.full_like(r, 1.0 / len(r))
    return float(r[y == 1].sum())


def lift(relevance, label):
    """LP normalised by the anomaly ratio. 1.0 == indistinguishable from uniform."""
    y = np.asarray(label, int).ravel()
    ar = y.mean()
    if ar == 0:
        return np.nan
    return localization_precision(relevance, y) / ar


def relevance_auc(relevance, label, kind="roc"):
    """Threshold-free: treat relevance as a score for the anomaly label."""
    y = np.asarray(label, int).ravel()
    r = np.asarray(relevance, float).ravel()
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    fn = roc_auc_score if kind == "roc" else average_precision_score
    return float(fn(y, r))


def normalized_lp(relevance, label, ceiling_relevance, floor_relevance=None):
    """(LP - LP_floor) / (LP_ceiling - LP_floor).

    0 == no better than the uniform floor.
    1 == as good as this detector allows (O2).
    """
    y = np.asarray(label, int).ravel()
    lp = localization_precision(relevance, y)
    lo = (localization_precision(floor_relevance, y)
          if floor_relevance is not None else y.mean())
    hi = localization_precision(ceiling_relevance, y)
    if hi - lo <= 1e-12:
        return np.nan
    return float((lp - lo) / (hi - lo))


def evaluate_all(relevance, label, ceiling_relevance=None):
    out = {"LP": localization_precision(relevance, label),
           "Lift": lift(relevance, label),
           "AUROC": relevance_auc(relevance, label, "roc"),
           "AUPRC": relevance_auc(relevance, label, "pr")}
    if ceiling_relevance is not None:
        out["LP_norm"] = normalized_lp(relevance, label, ceiling_relevance)
    return out


def localization_precision_tolerant(relevance, label, tol=100):
    """LP with the UCR archive's official tolerance.

    Wu & Keogh's protocol counts a prediction as correct if it lies within
    `tol` points of any point of the anomaly (tol=100 in the archive). Reporting
    both the strict and tolerant variants pre-empts the reviewer question about
    boundary sensitivity of the labels.
    """
    y = np.asarray(label, int).ravel()
    if y.sum() == 0:
        return np.nan
    idx = np.flatnonzero(y)
    lo, hi = max(0, idx.min() - tol), min(len(y), idx.max() + 1 + tol)
    widened = np.zeros_like(y); widened[lo:hi] = 1
    return localization_precision(relevance, widened)
