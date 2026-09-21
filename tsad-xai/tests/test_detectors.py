"""Detector-contract tests. REQUIRES TSB-AD; tests/test_core.py must not.

tests/test_core.py is deliberately pure-numpy so the metric logic can be
checked without the heavy stack. That is why nothing there caught the bug this
file exists for: ScoreFn.SUPPORTED advertised three detectors, but only
IForest had ever been instantiated, and KMeansAD has no decision_function at
all. The break surfaced only when the R5 screen first asked for a second
detector -- exactly the "works until you actually use it" pattern the project
already documents for sklearn>=1.6.

The contract every detector must satisfy, because every downstream metric
assumes it:
    len(f(X)) == len(X)      -- artifact.py indexes scores by timestep
    finite                   -- medians and quantiles propagate NaN otherwise
    deterministic            -- perturbation deltas are differences of scores
    fit-free scoring         -- Stage 3 makes ~700 calls per series

Run:  python tests/test_detectors.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detectors import ScoreFn                      # noqa: E402

PASS, FAIL = [], []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  PASS  {name}")
    except Exception as e:
        FAIL.append((name, e))
        print(f"  FAIL  {name}: {type(e).__name__}: {e}")


def synthetic(L=1500, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(L)
    x = np.sin(t / 20.0) + 0.05 * rng.standard_normal(L)
    x[700:760] += 4.0
    return x[:, None]


def test_every_supported_detector_scores():
    """The bug: SUPPORTED listed detectors that could not actually score.

    Iterating SUPPORTED rather than naming IForest is the whole point -- adding
    a detector to that tuple without a working scoring path must fail here.
    """
    X = synthetic()
    for name in ScoreFn.SUPPORTED:
        if name == "AutoEncoder":
            # Trains a torch net; too slow for a contract test. Covered by
            # test_autoencoder_scoring_path_exists below.
            continue
        sf = ScoreFn(name, X[:800], window=100)
        s = sf(X)
        assert len(s) == len(X), f"{name}: score length {len(s)} != {len(X)}"
        assert np.isfinite(s).all(), f"{name}: non-finite scores"


def test_autoencoder_scoring_path_exists():
    """Cheap guard for the detector too slow to fit here."""
    from TSB_AD.models.AE import AutoEncoder
    assert hasattr(AutoEncoder, "decision_function"), \
        "AutoEncoder lost decision_function; ScoreFn._score_attr needs updating"


def test_kmeansad_has_no_decision_function():
    """Pin the upstream fact that motivated the dispatch in ScoreFn.

    If a future TSB-AD adds decision_function to KMeansAD this fails, which is
    the point: the special case in ScoreFn should then be re-examined rather
    than silently kept.
    """
    from TSB_AD.models.KMeansAD import KMeansAD
    assert not hasattr(KMeansAD, "decision_function"), \
        "KMeansAD now has decision_function -- revisit ScoreFn._score_attr"
    assert hasattr(KMeansAD, "predict")


def test_scoring_is_deterministic():
    """Perturbation deltas are differences of scores; drift would be measured
    as an effect."""
    X = synthetic()
    for name in ("IForest", "KMeansAD"):
        sf = ScoreFn(name, X[:800], window=100)
        assert np.array_equal(sf(X), sf(X)), f"{name}: scoring not deterministic"


def test_scoring_does_not_refit():
    """ScoreFn exists to avoid TSB-AD's refit-per-call. Scoring must leave the
    fitted model untouched, or Stage 3's cost model is wrong."""
    X = synthetic()
    for name in ("IForest", "KMeansAD"):
        sf = ScoreFn(name, X[:800], window=100)
        before = sf(X)
        for _ in range(3):
            sf(X)
        assert np.array_equal(before, sf(X)), f"{name}: scoring mutated the model"
        assert sf.n_calls == 5, f"{name}: n_calls miscounted ({sf.n_calls})"


def test_kmeansad_scoring_is_quiet():
    """KMeansAD prints 3 lines per predict(). At ~700 calls per series that
    buries every other message, so ScoreFn suppresses it."""
    import io
    from contextlib import redirect_stdout
    X = synthetic()
    sf = ScoreFn("KMeansAD", X[:800], window=100)   # fit may print; that is fine
    buf = io.StringIO()
    with redirect_stdout(buf):
        sf(X)
    assert buf.getvalue() == "", f"KMeansAD scoring printed: {buf.getvalue()!r}"


def test_iforest_normalize_is_false():
    """Trap 2: IForest.fit z-scores the window matrix and decision_function
    does not, so the default normalize=True makes them disagree."""
    X = synthetic()
    sf = ScoreFn("IForest", X[:800], window=100)
    assert sf.clf.normalize is False, "IForest must be built with normalize=False"


if __name__ == "__main__":
    for nm, fn in list(globals().items()):
        if nm.startswith("test_") and callable(fn):
            check(nm, fn)
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    sys.exit(1 if FAIL else 0)
