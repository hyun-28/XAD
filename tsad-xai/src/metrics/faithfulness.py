"""Stage 5 -- the incumbent perturbation-based faithfulness protocol, ported
from arXiv:2601.19017 (frequency bands -> time segments).

Their procedure, reproduced exactly:
    1. split the input into K bands/segments
    2. mean relevance per segment            R_k
    3. mask segment k, measure |f(X~) - f(X)|   dPred_k
    4. Spearman rho between R and dPred, per sample
    5. aggregate across samples via Fisher z

OUR ONE DEVIATION -- and it is a deliberate contribution:
    they use |dPred|. Removing a band can IMPROVE detection (their Table 3:
    removing band 5 lifts mean AUC from 73.5% to 76.8%), so the absolute value
    conflates "harmful to remove" with "helpful to remove". We compute BOTH
    signed and absolute variants and report the gap.

`gt_perturb` is exposed as an argument because Experiment D (Stage 7) varies it
to measure how much of the resulting ranking is an artefact of choosing a
ground-truth perturbation that matches one explainer's own mechanism.
"""
import numpy as np
from scipy.stats import spearmanr

from . import fisher_mean


def segment_indices(L, K):
    edges = np.linspace(0, L, K + 1).astype(int)
    return [np.arange(edges[k], edges[k + 1]) for k in range(K)]


def mean_relevance_per_segment(relevance, segments):
    r = np.abs(np.asarray(relevance, float)).ravel()
    s = r.sum()
    r = r / s if s > 0 else np.full_like(r, 1.0 / len(r))
    return np.array([r[seg].mean() if len(seg) else 0.0 for seg in segments])


def delta_pred_per_segment(score_fn, X, segments, gt_perturb, rng=None, how="max"):
    """Score change per masked segment. Returns (signed, absolute)."""
    rng = np.random.default_rng(0) if rng is None else rng
    agg = {"max": np.max, "mean": np.mean}[how]
    base = agg(score_fn(X))
    signed = []
    for seg in segments:
        Xp = gt_perturb(X, seg, rng)
        signed.append(agg(score_fn(Xp)) - base)
    signed = np.asarray(signed, float)
    return signed, np.abs(signed)


def faithfulness_sample(relevance, score_fn, X, gt_perturb, K=10, rng=None, how="max"):
    """Spearman rho for one sample, in both signed and absolute variants."""
    segs = segment_indices(len(X), K)
    R = mean_relevance_per_segment(relevance, segs)
    d_signed, d_abs = delta_pred_per_segment(score_fn, X, segs, gt_perturb, rng, how)
    out = {}
    for name, d in (("signed", d_signed), ("abs", d_abs)):
        if np.std(R) < 1e-12 or np.std(d) < 1e-12:
            out[name] = np.nan
        else:
            out[name] = float(spearmanr(R, d).statistic)
    return out


def faithfulness_aggregate(per_sample):
    """Fisher-z mean over samples, for each variant."""
    keys = per_sample[0].keys() if per_sample else []
    return {k: fisher_mean([d[k] for d in per_sample]) for k in keys}


def kendall_w(rank_matrix):
    """Kendall's W over (n_conditions, n_items) rankings.

    Stage 7: rank explainers under each ground-truth perturbation.
    W near 1 -> the choice of ground-truth perturbation does not matter.
    W low    -> the "most faithful explainer" is an artefact of that choice.
    """
    R = np.asarray(rank_matrix, float)
    m, n = R.shape                      # m raters (perturbations), n items
    Rj = R.sum(axis=0)
    S = ((Rj - Rj.mean()) ** 2).sum()
    denom = m ** 2 * (n ** 3 - n) / 12.0
    return float(S / denom) if denom > 0 else np.nan
