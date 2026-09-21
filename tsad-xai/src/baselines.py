"""Baseline explanations: the floor (N1-N4) and the ceiling (O1-O2).

Without these, no localization or faithfulness number is interpretable.
Every function returns a non-negative relevance vector of length L that sums
to 1, matching the normalisation convention of arXiv:2601.19017.

N4 (reconstruction error) is the PRIMARY baseline. Reconstruction error is what
a large part of the TSAD literature already uses as an explanation, and it is
free. Any XAI method that cannot beat N4 has no practical justification.

NOTE ON N4: TSB-AD's AutoEncoder.decision_function returns a PER-WINDOW
Euclidean distance (pairwise_distances_no_broadcast), not a per-timestep error,
and it runs under torch.no_grad(). Per-timestep error therefore has to be
computed here rather than read off the detector.
"""
import numpy as np


def normalize(r, eps=1e-12):
    """Non-negative, sums to 1."""
    r = np.abs(np.asarray(r, float)).ravel()
    s = r.sum()
    return np.full_like(r, 1.0 / len(r)) if s < eps else r / s


# ---------------------------------------------------------------- floor (N)

def n1_random(X, score_fn=None, rng=None):
    """Absolute floor: uniform random relevance."""
    rng = np.random.default_rng(0) if rng is None else rng
    return normalize(rng.random(len(X)))


def n2_uniform(X, score_fn=None, rng=None):
    """Equal relevance everywhere. Defines Lift == 1."""
    return np.full(len(X), 1.0 / len(X))


def n3_zscore(X, score_fn=None, rng=None):
    """|x - mu| / sigma. A statistical explanation that never looks at the model."""
    A = np.asarray(X, float)
    A = A[:, None] if A.ndim == 1 else A
    sd = np.where(A.std(axis=0) == 0, 1e-8, A.std(axis=0))
    return normalize(np.abs((A - A.mean(axis=0)) / sd).sum(axis=1))


def n4_recon_error(X, model=None, window=100, score_fn=None, rng=None):
    """Per-timestep reconstruction error, folded back from window space.

    If `model` is a fitted TSB-AD AutoEncoder, its inner torch module is used.
    Otherwise a cheap moving-average residual stands in, which keeps the
    baseline available even for non-reconstruction detectors.
    """
    A = np.asarray(X, float)
    A = A[:, None] if A.ndim == 1 else A
    L = len(A)
    if model is None:
        k = max(3, window // 10)
        ker = np.ones(k) / k
        smooth = np.stack([np.convolve(A[:, c], ker, mode="same")
                           for c in range(A.shape[1])], axis=1)
        return normalize(((A - smooth) ** 2).sum(axis=1))

    import torch
    net = model.model
    net.eval()
    from TSB_AD.models.feature import Window
    from sklearn.preprocessing import MinMaxScaler
    W = Window(window=window).convert(A[:, 0] if A.shape[1] == 1 else A)
    W = np.asarray(W, float)
    W = MinMaxScaler(feature_range=(0, 1)).fit_transform(W.T).T
    if getattr(model, "preprocessing", False):
        W = (W - model.mean) / model.std
    with torch.no_grad():
        rec = net(torch.tensor(W, dtype=torch.float32)).cpu().numpy()
    err = (W - rec) ** 2                                  # (n_windows, window)
    # fold window-local errors back onto the time axis
    acc = np.zeros(L)
    cnt = np.zeros(L)
    off = int(np.ceil((window - 1) / 2))
    for i in range(err.shape[0]):
        for j in range(err.shape[1]):
            t = i + j - off + off  # window i covers timesteps i..i+window-1
            t = min(max(i + j, 0), L - 1)
            acc[t] += err[i, j]
            cnt[t] += 1
    return normalize(acc / np.where(cnt == 0, 1, cnt))


FLOOR = {"N1_random": n1_random, "N2_uniform": n2_uniform,
         "N3_zscore": n3_zscore, "N4_recon_error": n4_recon_error}


# -------------------------------------------------------------- ceiling (O)

def o1_oracle(label):
    """Relevance == the label. LP == 1 by construction."""
    y = np.asarray(label, int).ravel()
    return normalize(y.astype(float)) if y.sum() else np.full(len(y), 1.0 / len(y))


def o2_achievable(label, score, top_frac=None):
    """Ceiling limited by what the detector actually found.

    An explainer cannot point at an anomaly the detector never reacted to, so
    O1 is an unreachable ceiling. O2 restricts the oracle to labelled anomalies
    that fall inside the detector's own top-scoring region.
    """
    y = np.asarray(label, int).ravel()
    s = np.asarray(score, float).ravel()
    frac = (y.mean() if top_frac is None else top_frac) or 0.01
    thr = np.quantile(s, 1.0 - frac)
    detected = ((y == 1) & (s >= thr)).astype(float)
    return normalize(detected) if detected.sum() else o1_oracle(y)
