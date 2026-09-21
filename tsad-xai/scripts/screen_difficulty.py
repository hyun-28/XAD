"""R5 -- the difficulty screen the selection rules deferred.

WHY R5 EXISTS
-------------
build_dataset_config.py applies R2/R3/R4/R7/R8 from filenames and lengths alone.
R5 -- "the series is neither trivial nor impossible" -- cannot be decided that
way because it depends on a fitted detector, so it lives here.

Both ends of the range destroy the study, for opposite reasons:

  IMPOSSIBLE.  If the detector never reacts to the anomaly, every explainer is
  explaining noise. Localization precision then measures nothing about the
  explainer, and the O2 ceiling in src/baselines.py collapses onto O1.

  TRIVIAL.  If a statistic that never looks at the model already points at the
  anomaly, every explainer scores near-perfectly and the methods cannot be
  told apart. N3_zscore is exactly that statistic, and it is already the
  project's floor baseline -- so "trivial" is operationalised as "the floor
  already solves it".

THE CRITERION IS THE ARCHIVE'S OWN
----------------------------------
Wu & Keogh score a prediction correct if it lies within 100 points of the
anomaly. We reuse that rather than inventing a threshold: the single
highest-scoring timestep in the test portion must fall within `--tol` points
of the labelled anomaly interval. One prediction per series, which is what the
archive's one-anomaly-per-series property affords.

SEMI-SUPERVISED SETUP
---------------------
UCR gives an anomaly-free training prefix per series (`train_end`), so the
detector is FIT ON THE TRAIN PREFIX ONLY and scored on the whole series. This
is the setup the archive was designed for. Fitting on the full series would
leak the anomaly into the model that is supposed to find it.

Usage:
    python scripts/screen_difficulty.py                     # reads configs/datasets.yaml
    python scripts/screen_difficulty.py --detector IForest --tol 100
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.baselines import n3_zscore          # noqa: E402
from src.data_ucr import load_file           # noqa: E402
from src.detectors import ScoreFn            # noqa: E402


def hits(score, y, start, tol):
    """Archive criterion: does the top-scoring timestep land near the anomaly?

    `score` is scored on the test portion; `start` is that portion's offset
    into the full series, so the returned distance is in full-series
    coordinates and comparable across detectors.
    """
    idx = np.flatnonzero(y)
    if idx.size == 0 or score.size == 0:
        return False, np.nan
    peak = start + int(np.argmax(score))
    dist = 0 if idx.min() <= peak <= idx.max() else int(
        min(abs(peak - idx.min()), abs(peak - idx.max())))
    return dist <= tol, dist


def screen_one(entry, root, detector, window, tol):
    path = os.path.join(root, entry["file"])
    x, y, meta = load_file(path)
    k = int(meta["train_end"])

    # The archive guarantees the prefix is anomaly-free. Verify rather than
    # trust it: a violation would silently invalidate the semi-supervised setup.
    train_contaminated = bool(y[:k].sum())

    sf = ScoreFn(detector, x[:k], window=window)
    s_test = sf(x[k:])
    det_hit, det_dist = hits(s_test, y, k, tol)

    # The floor baseline, computed on the same test portion. n3_zscore never
    # sees the model, so if it wins the series is trivial by construction.
    triv = n3_zscore(x[k:])
    tri_hit, tri_dist = hits(triv, y, k, tol)

    yt = y[k:]
    auroc = np.nan
    if 0 < yt.sum() < len(yt):
        from sklearn.metrics import roc_auc_score
        auroc = float(roc_auc_score(yt, s_test))

    if not det_hit:
        verdict = "impossible"
    elif tri_hit:
        verdict = "trivial"
    else:
        verdict = "keep"

    return {
        "file": entry["file"],
        "distorted": entry.get("distorted"),
        "length": len(x),
        "train_end": k,
        "anomaly_ratio": float(y.mean()),
        "detector_hit": det_hit,
        "detector_dist": det_dist,
        "trivial_hit": tri_hit,
        "trivial_dist": tri_dist,
        "AUROC": auroc,
        "verdict": verdict,
        "train_contaminated": train_contaminated,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/datasets.yaml")
    ap.add_argument("--detector", default="IForest")
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--tol", type=int, default=100,
                    help="UCR archive tolerance in points (default 100)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="results/difficulty_screen.csv")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    if cfg.get("format") != "ucr":
        raise SystemExit(f"{a.config} is format={cfg.get('format')!r};"
                         " R5 is implemented for the UCR archive only.")
    files = cfg["files"][:a.limit] if a.limit else cfg["files"]
    root = cfg["root"]

    rows, t0 = [], time.time()
    for i, e in enumerate(files, 1):
        try:
            r = screen_one(e, root, a.detector, a.window, a.tol)
        except Exception as exc:                       # keep the sweep alive
            r = {"file": e["file"], "verdict": "error",
                 "error": f"{type(exc).__name__}: {exc}"}
        rows.append(r)
        print(f"[{i:3d}/{len(files)}] {r['verdict']:<10} "
              f"dist={r.get('detector_dist')} "
              f"trivial={r.get('trivial_hit')} {e['file']}", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)

    print(f"\n--- R5 screen ({a.detector}, tol={a.tol}, "
          f"{time.time() - t0:.0f}s) ---")
    print(df.verdict.value_counts().to_string())
    if "train_contaminated" in df and df.train_contaminated.any():
        bad = df[df.train_contaminated].file.tolist()
        print(f"\nWARNING: {len(bad)} series have anomalous points inside the"
              f" supposedly anomaly-free training prefix: {bad}")
    keep = df[df.verdict == "keep"]
    print(f"\n{len(keep)} of {len(df)} survive R5.")
    if len(keep) < len(df):
        print("R5 is a SCREEN, not a filter applied here -- removing series"
              " changes the sample, so edit the config deliberately and record"
              " it as an amendment. This file only reports.")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
