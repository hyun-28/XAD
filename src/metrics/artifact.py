"""Stage 3 -- Artifact Index.

The question: when we perturb a region that is LABELLED NORMAL, does the
anomaly score rise anyway? If it does, deletion-based faithfulness metrics are
partly measuring the perturbation, not the explanation.

The control is the same perturbation applied to LABELLED ANOMALOUS regions.
Without that contrast there is no baseline and the numbers mean nothing.
"""
import numpy as np


def sample_windows(label, width, n, rng, region="normal", margin=0, min_purity=1.0):
    """Sample window start indices for the requested region.

    region='normal'  : windows lying entirely inside labelled-normal stretches.
    region='anomaly' : windows whose masked span is at least `min_purity` of the
                       way anomalous.

    NOTE: anomalous segments are usually SHORTER than `width`, so requiring
    100% purity would return an empty set and silently delete the control arm.
    For region='anomaly' the caller should either shrink `width` or lower
    `min_purity`; `effective_width` below does the former automatically.
    """
    y = np.asarray(label, int).ravel()
    L = len(y)
    want = 0 if region == "normal" else 1
    purity = 1.0 if region == "normal" else min_purity
    ok = []
    for s in range(0, max(1, L - width)):
        seg = y[s:s + width]
        if len(seg) < width:
            continue
        frac = (seg == want).mean()
        if frac >= purity:
            if margin and (s < margin or s + width > L - margin):
                continue
            ok.append(s)
    if not ok:
        return np.array([], int)
    ok = np.array(ok, int)
    return rng.choice(ok, size=min(n, len(ok)), replace=False)


def effective_width(label, width, region):
    """Shrink the mask width so anomalous segments can actually host a window."""
    if region != "anomaly":
        return width
    y = np.asarray(label, int).ravel()
    runs, cur = [], 0
    for v in y:
        if v == 1:
            cur += 1
        elif cur:
            runs.append(cur); cur = 0
    if cur:
        runs.append(cur)
    if not runs:
        return width
    return int(max(5, min(width, min(runs))))


def artifact_index(score_fn, X, label, perturb, width=50, n=100,
                   region="normal", rng=None, how="max"):
    """Standardised score shift caused by perturbing `region`.

    AI = median_w [ (f(X~_w) - f(X)) / sigma_f ]  evaluated on the masked span.

    AI significantly above 0 on normal regions == manufactured anomalies.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    base = score_fn(X)
    sigma = base.std()
    sigma = 1e-8 if sigma < 1e-8 else sigma
    width = effective_width(label, width, region)
    starts = sample_windows(label, width, n, rng, region=region, min_purity=0.9)
    if starts.size == 0:
        return {"AI": np.nan, "VR": np.nan, "n": 0, "width": width,
                "deltas": np.array([])}

    thr95 = np.quantile(base, 0.95)
    deltas, violations = [], 0
    agg = {"max": np.max, "mean": np.mean}[how]
    for s in starts:
        idx = np.arange(s, s + width)
        Xp = perturb(X, idx, rng)
        sp = score_fn(Xp)
        deltas.append((agg(sp[idx]) - agg(base[idx])) / sigma)
        if agg(sp[idx]) > thr95:
            violations += 1
    deltas = np.asarray(deltas, float)
    return {"AI": float(np.median(deltas)),
            "VR": violations / len(starts),
            "n": len(starts),
            "width": width,
            "deltas": deltas}
