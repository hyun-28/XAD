"""Detectors: BRIEF Task B interface (base.py, iforest.py) + the pre-Task-B
`ScoreFn` (legacy.py), re-exported so experiments/exp_a_artifact.py and
scripts/screen_difficulty.py keep working unchanged (DECISIONS D-B2-4)."""
from src.detectors.base import (SUPPORTED, DetectorError, FittedDetector, detector_config,
                                fit_detector, validate_scores)
from src.detectors.legacy import ScoreFn, self_check

__all__ = ["SUPPORTED", "DetectorError", "FittedDetector", "detector_config", "fit_detector", "validate_scores",
           "ScoreFn", "self_check"]
