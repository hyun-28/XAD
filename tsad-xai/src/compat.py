"""scikit-learn >= 1.6 compatibility shim for TSB-AD.

VERIFIED FAILURE (scikit-learn 1.8.0, TSB_AD 1.5):
    TSB_AD.models.base.BaseDetector does not implement __sklearn_tags__.
    sklearn>=1.6's check_is_fitted() calls get_tags(estimator), which raises
        AttributeError: 'IForest' object has no attribute '__sklearn_tags__'
    fit() is unaffected. Only decision_function() breaks -- which is exactly
    the method this project depends on for perturbation scoring.

PREFERRED FIX: pin scikit-learn<1.6 in the environment (see environment.yml).
This shim is the fallback when pinning is not possible.
"""
import importlib

_DEFAULT_MODULES = (
    "TSB_AD.models.IForest",
    "TSB_AD.models.KMeansAD",
    "TSB_AD.models.AE",
    "TSB_AD.models.LOF",
    "TSB_AD.models.PCA",
    "TSB_AD.models.OCSVM",
)


def _check_is_fitted(est, attrs=None, **kwargs):
    attrs = attrs or []
    missing = [a for a in attrs if not hasattr(est, a)]
    if missing:
        raise RuntimeError(f"Detector is not fitted; missing attributes {missing}")
    return True


def patch_tsb_ad(module_names=_DEFAULT_MODULES, verbose=False):
    """Replace check_is_fitted inside TSB-AD model modules. Idempotent."""
    patched = []
    for name in module_names:
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        if hasattr(mod, "check_is_fitted"):
            mod.check_is_fitted = _check_is_fitted
            patched.append(name)
    if verbose:
        print(f"[compat] patched check_is_fitted in {len(patched)} module(s)")
    return patched
