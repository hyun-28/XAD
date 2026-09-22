"""Detector wrapper interface (BRIEF Task B, B2/B3).

Why this exists: TSB-AD's run_Unsupervise_AD refits on every call and turns
every exception into a returned string (docs/TSBAD_INTERNALS.md §1-2). The
Artifact Index needs "same fitted model, many score() calls", so the wrapper
fits once, scores through decision_function only, and validates every score
vector before anyone downstream sees it.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from src.config import load_yaml


SUPPORTED = ("IForest", "MatrixProfile")


class DetectorError(RuntimeError):
    """A detector returned something that must not be used as a score."""


@runtime_checkable
class FittedDetector(Protocol):
    name: str
    config: dict           # every hyper-parameter needed to reproduce, explicitly

    def score(self, x: np.ndarray) -> np.ndarray: ...   # no refit, ever


def validate_scores(out, n: int, *, constant_score: str = "error") -> tuple[np.ndarray, list[str]]:
    """BRIEF B3 checks on one score() output. Returns (float64 scores, flags).

    constant_score: "error" -> DetectorError on zero variance; "flag" -> record it.
    """
    if isinstance(out, str):
        # TSB-AD's silent failure path (model_wrapper.py:16-19) returns a str
        raise DetectorError(f"detector returned an error string: {out!r}")
    try:
        s = np.asarray(out)
    except Exception as e:  # pragma: no cover - defensive
        raise DetectorError(f"score output is not array-like: {e}") from e
    if s.ndim == 2 and s.shape[1] == 1:
        s = s[:, 0]
    if s.ndim != 1:
        raise DetectorError(f"score output must be 1-D or (n,1), got shape {s.shape}")
    if s.shape[0] != n:
        raise DetectorError(f"score length {s.shape[0]} != input length {n} (padding rule violated)")
    if s.dtype.kind not in "fiu":
        raise DetectorError(f"score dtype {s.dtype} is not numeric")
    s = s.astype(np.float64, copy=False)
    if not np.isfinite(s).all():
        raise DetectorError(f"{int((~np.isfinite(s)).sum())} non-finite score values")
    flags = []
    if s.size and s.var() == 0.0:
        if constant_score == "error":
            raise DetectorError("constant score vector (variance 0): the Artifact Index is undefined")
        if constant_score != "flag":
            raise ValueError(f"constant_score must be 'error' or 'flag', got {constant_score!r}")
        flags.append("CONSTANT_SCORE")
    return s, flags


def fit_detector(name: str, x_fit: np.ndarray, *, seed: int, **hp) -> FittedDetector:
    """Fit once. `x_fit` is whatever src/data/ucr.py::slice_fit returned."""
    if name == "IForest":
        from src.detectors.iforest import IForestDetector
        return IForestDetector.fit(x_fit, seed=seed, **hp)
    if name == "MatrixProfile":
        from src.detectors.matrixprofile import MatrixProfileDetector
        return MatrixProfileDetector.fit(x_fit, seed=seed, **hp)
    raise ValueError(f"unknown detector {name!r}; supported: {sorted(SUPPORTED)}")


def detector_config(name: str) -> dict:
    """Defaults from configs/detectors.yaml for one detector (a copy)."""
    cfg = load_yaml("detectors")
    key = name.lower()
    if key not in cfg:
        raise KeyError(f"configs/detectors.yaml has no section {key!r}")
    return dict(cfg[key])
