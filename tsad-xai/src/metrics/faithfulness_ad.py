"""Faithfulness for anomaly detection — W2 rev1 D7-D9, D13 (approved 2026-09-24, commit f29eddf).

This replaces the BRIEF2 interface stub (whose D7-D10 labels meant different open questions; the
approved rev1 decisions supersede it).

    f(x)        = resp(x, R_anom) = max over J(R_anom) of score_patch(x, R_anom)          (D7, D13)
    Ω           = I(R_anom) ∩ [train_end, n);  f is exactly independent of x outside Ω      (D13)
    segments    length g = max(1, w // 4); if that gives K > 64 segments, g is raised to
                ceil(|Ω| / 64) so K <= 64 (the last segment may be shorter)                  (D13)
    masking     masked segments grouped into maximal runs; the operator is applied per run,
                each run from the unmasked x (D-E1-2, approved)
    AMs         Random (uniform, seeded), FeatureAblation f(x) - f(x with k masked),
                KernelSHAP (numpy port of captum 0.9.0 KernelShap, see `kernel_shap`),
                MPNative (per-point (ẑ_q - ẑ_nn)^2 of the window attaining resp, summed per segment)
    curves      MoRF / LeRF at 10 %, 20 %, ..., 100 % of the segments; element 0 = f(x)
    DDS         faithfulness_simic.decaying_degradation_score with max_diff = scale(x)  (D8, D13);
                sign as upstream (a falling MoRF curve is faithful); values are not clipped (D9)
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np

from src.metrics.faithfulness_simic import decaying_degradation_score

MAX_SEGMENTS = 64
STEPS = 10


def omega(train_end: int, gt: tuple[int, int], w: int, n: int) -> tuple[int, int]:
    """D13: Ω = I(R_anom) ∩ test region = [max(GT_start - w + 1, train_end), min(GT_stop + w - 1, n))."""
    lo, hi = max(gt[0] - w + 1, train_end), min(gt[1] + w - 1, n)
    if hi <= lo:
        raise ValueError(f"empty Ω: [{lo}, {hi})")
    return lo, hi


def segments(om: tuple[int, int], w: int) -> tuple[list[tuple[int, int]], int]:
    """D13 segmentation of Ω. Returns (segments, g)."""
    lo, hi = om
    size = hi - lo
    g = max(1, w // 4)
    if math.ceil(size / g) > MAX_SEGMENTS:
        g = math.ceil(size / MAX_SEGMENTS)
    segs = [(s, min(s + g, hi)) for s in range(lo, hi, g)]
    if len(segs) > MAX_SEGMENTS:
        raise AssertionError("segmentation exceeded 64")
    return segs, g


def runs_of(segs: Sequence[tuple[int, int]], masked: Sequence[int]) -> list[tuple[int, int]]:
    """Maximal runs of adjacent masked segments (D13)."""
    out: list[tuple[int, int]] = []
    for k in sorted(set(int(v) for v in masked)):
        a, b = segs[k]
        if out and out[-1][1] == a:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


def random_attribution(K: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(0.0, 1.0, K)


def feature_ablation(f0: float, f_masked: Callable[[Sequence[int]], float], K: int) -> np.ndarray:
    return np.array([f0 - f_masked([k]) for k in range(K)])


def kernel_shap(f_present: Callable[[np.ndarray], float], K: int, n_samples: int,
                rng: np.random.Generator) -> np.ndarray:
    """numpy port of captum 0.9.0 `KernelShap` (captum/attr/_core/kernel_shap.py:268-365):
      * samples: all-present, all-absent, then n_samples - 2 draws with k ~ p(k) ∝ (K-1)/(k(K-k)),
        k in 1..K-1, and a uniformly random subset of size k (captum: top-k of a normal draw);
      * weights 1e6 for the all-present / all-absent samples, 1 otherwise (:301-323);
      * weighted least squares with intercept (captum: sklearn LinearRegression, sample_weight);
        coefficients = attributions. Rank-deficient systems get numpy's minimum-norm solution.
    `f_present(z)` evaluates f with z_k = 1 kept and z_k = 0 masked.
    """
    if K == 1:
        z = np.array([[1], [0]])
        y = np.array([f_present(z[0]), f_present(z[1])])
        return np.array([y[0] - y[1]])
    ks = np.arange(1, K)
    p = (K - 1) / (ks * (K - ks))
    p = p / p.sum()
    Z = [np.ones(K, dtype=int), np.zeros(K, dtype=int)]
    for _ in range(max(0, n_samples - 2)):
        k = int(rng.choice(ks, p=p))
        z = np.zeros(K, dtype=int)
        z[rng.choice(K, size=k, replace=False)] = 1
        Z.append(z)
    Z = np.array(Z)
    y = np.array([f_present(z) for z in Z])
    wts = np.ones(len(Z))
    wts[:2] = 1e6
    X = np.c_[np.ones(len(Z)), Z]
    sw = np.sqrt(wts)
    coef, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    return coef[1:]


def mp_native(x: np.ndarray, reference: np.ndarray, J: np.ndarray, scores: np.ndarray, w: int,
              segs: Sequence[tuple[int, int]]) -> tuple[np.ndarray, dict]:
    """MPNative (D13): window j_max attaining resp; its nearest training neighbour (stumpy.mass, first
    minimum); per-point (ẑ_q,i - ẑ_nn,i)^2 summed per segment; 0 outside j_max's support.
    A constant window z-normalises to zeros (stumpy's constant convention gives D^2 = m, which is
    what the zero vector reproduces against a non-constant neighbour)."""
    import stumpy
    k = int(np.argmax(scores))
    start = int(J[k]) - w // 2
    q = np.asarray(x[start:start + w], dtype=np.float64)
    d = np.asarray(stumpy.mass(q, reference), dtype=np.float64)
    u = int(np.argmin(d))
    nn = np.asarray(reference[u:u + w], dtype=np.float64)

    def z(v):
        s = v.std()
        return np.zeros_like(v) if np.ptp(v) == 0 else (v - v.mean()) / s

    contrib = (z(q) - z(nn)) ** 2
    attr = np.zeros(len(segs))
    for i, (a, b) in enumerate(segs):
        lo, hi = max(a, start), min(b, start + w)
        if hi > lo:
            attr[i] = contrib[lo - start:hi - start].sum()
    return attr, {"j_max": int(J[k]), "nn": u, "sum_contrib": float(contrib.sum()),
                  "score_sq": float(scores[k] ** 2)}


def tie_rank(K: int, rng: np.random.Generator) -> np.ndarray:
    """D-E3-5: a seeded random rank per segment, used only to break ties (seed = series, AM)."""
    perm = rng.permutation(K)
    rank = np.empty(K, dtype=np.int64)
    rank[perm] = np.arange(K)
    return rank


def order_d35(attr: np.ndarray, ties: np.ndarray, direction: str, key: str = "signed") -> np.ndarray:
    """D-E3-5 (approved b8b783c): MoRF = descending `key` value, ties broken by the seeded random rank;
    LeRF = the exact reverse of MoRF (Šimić interpret_model_regions.py:369, np.flip).
    key "signed" (main) or "abs" (|R| sensitivity, FeatureAblation / KernelSHAP only)."""
    v = np.asarray(attr, dtype=np.float64)
    if key == "abs":
        v = np.abs(v)
    elif key != "signed":
        raise ValueError(key)
    if ties.shape != v.shape:
        raise ValueError("tie ranks do not match the attribution")
    morf = np.lexsort((ties, -v))
    if direction == "MoRF":
        return morf
    if direction == "LeRF":
        return morf[::-1].copy()
    raise ValueError(direction)


def curve_from_order(f0: float, f_masked: Callable[[Sequence[int], int], float], o: np.ndarray) -> np.ndarray:
    """Same steps as `curve`, for an explicit order."""
    K = o.size
    vals = [f0]
    for t in range(1, STEPS + 1):
        vals.append(f_masked(o[:math.ceil(t * K / STEPS)].tolist(), t))
    return np.array(vals)


def order(attr: np.ndarray, direction: str) -> np.ndarray:
    """INDEX-TIE VERSION (E3 run of 2026-09-25, kept to reproduce it; superseded by `order_d35`, D-E3-5).
    MoRF = descending attribution, LeRF = ascending; ties -> smaller segment index first."""
    idx = np.arange(attr.size)
    if direction == "MoRF":
        return np.lexsort((idx, -attr))
    if direction == "LeRF":
        return np.lexsort((idx, attr))
    raise ValueError(direction)


def curve(f0: float, f_masked: Callable[[Sequence[int], int], float], attr: np.ndarray, direction: str) -> np.ndarray:
    """[f(x), f(mask 10 %), ..., f(mask 100 %)]; step t masks the first ceil(t K / 10) of the order.
    `f_masked(masked, t)` receives the step so a stochastic operator can be seeded per step."""
    K = attr.size
    o = order(attr, direction)
    vals = [f0]
    for t in range(1, STEPS + 1):
        m = math.ceil(t * K / STEPS)
        vals.append(f_masked(o[:m].tolist(), t))
    return np.array(vals)


def dds(morf: np.ndarray, lerf: np.ndarray, scale: float) -> float:
    """D8/D13: max_diff = scale(x)."""
    return decaying_degradation_score(morf, lerf, max_diff=scale)
