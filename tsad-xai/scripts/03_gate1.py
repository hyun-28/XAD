"""03_gate1.py -- BRIEF Task C: run a detector on the 250 official UCR series.

--detector IForest (default): fit_on {train_prefix, full} x max_features {1, 1.0}
  x seeds {0, 1, 2} = 12 conditions per series.
--detector MatrixProfile (gate-1 replacement, D-C5-3): fit_on {train_prefix, full}
  x one seed = 2 conditions; the window is find_length_rank(x, rank=1) per series
  (logged), fit_on=full -> stumpy self-join, train_prefix -> AB-join against the
  training prefix. Deterministic, so seeds are not repeated.

For every condition:
  slice_fit -> fit_detector (once) -> score(full series) -> save raw scores ->
  pre-registered criteria (src/gate1.py), with P1's buffer = that detector's window.

Writes
  results/gate1/<detector>/scores/<num>_<cond>.npy   raw scores (git-ignored)
  results/gate1/<detector>/detection.csv             series x condition x P1/P2
  results/gate1/<detector>/summary.json              run id, grid, failure rate, timing
  runs/<run_id>/calls.jsonl, failures.jsonl, env.txt        BRIEF B5 call log

Failures are recorded, never skipped silently; the loop continues and the
exit code is 1 if the failure rate exceeds --max-failure-rate (default 0).
Judgement (gate pass/STOP) is done by scripts/04_report.py.

Usage:  python scripts/03_gate1.py [--detector MatrixProfile] [--limit 5] [--workers 4]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import REPO_ROOT, load_yaml, resolve  # noqa: E402
from src.data.ucr import load_series, parse_filename, slice_fit  # noqa: E402
from src.detectors import DetectorError, fit_detector  # noqa: E402
from src.gate1 import evaluate  # noqa: E402
from src.runlog import CallLog, env_header  # noqa: E402

GATE_ROOT = REPO_ROOT / "results" / "gate1"
DETECTION_COLUMNS = [
    "num", "name", "variant", "domain_goswami", "is_medical", "content_group", "name_group",
    "detector", "fit_on", "max_features", "seed", "n_fit", "window", "n_estimators", "mode",
    "n", "train_end", "start0", "stop0", "max_anom", "thr_b0", "thr_bw", "n_normal_b0", "n_normal_bw",
    "p1_b0", "p1_bw", "argmax0", "p2_archive", "p2_brief",
    "max_support", "p1_support_bw", "train_max", "train_p99", "train_selfmatch_max", "n_selfmatch",
    "n_const_train",
    "fit_s", "score_s", "score_file",
]


def conditions(det_cfg: dict, detector: str) -> list[dict]:
    """One dict per condition; `tag` names the score file."""
    if detector == "IForest":
        ifo = det_cfg["iforest"]
        return [{"fit_on": fo, "max_features": mf, "seed": int(sd),
                 "tag": f"{fo}_{mf_tag(mf)}_seed{sd}"}
                for fo in det_cfg["fit_on_variants"]
                for mf in ifo["max_features_variants"]
                for sd in det_cfg["seeds"]]
    if detector == "MatrixProfile":
        mp = det_cfg["matrixprofile"]
        return [{"fit_on": fo, "mode": mp["mode_for_fit_on"][fo], "seed": int(sd),
                 "tag": f"{fo}_{mp['mode_for_fit_on'][fo]}_seed{sd}"}
                for fo in det_cfg["fit_on_variants"]
                for sd in mp["seeds"]]
    raise ValueError(f"unknown detector {detector!r}")


def mf_tag(mf) -> str:
    return f"mf{mf}" if isinstance(mf, int) else f"mf{mf:.1f}"


def results_dir(detector: str) -> Path:
    return GATE_ROOT / detector


def score_path(detector: str, num: int, tag: str) -> Path:
    return results_dir(detector) / "scores" / f"{num:03d}_{tag}.npy"


def _hyperparams(detector: str, det_cfg: dict, cond: dict, x: np.ndarray) -> dict:
    """The per-condition detector kwargs, plus `window` for the report and P1's buffer."""
    if detector == "IForest":
        ifo = det_cfg["iforest"]
        return {"slidingWindow": int(ifo["slidingWindow"]), "n_estimators": int(ifo["n_estimators"]),
                "n_jobs": int(ifo["n_jobs"]), "max_features": cond["max_features"],
                "constant_score": det_cfg["constant_score"]}
    mp = det_cfg["matrixprofile"]
    from src.detectors.matrixprofile import benchmark_window
    # find_length_rank is computed on the FULL series in the benchmark path
    # (model_wrapper.run_MatrixProfile:80 gets the whole `data`), so the window does
    # not change with fit_on; it is logged per series (D-C5-3 rule 3).
    return {"window": benchmark_window(x, int(mp["periodicity"])), "mode": cond["mode"],
            "periodicity": int(mp["periodicity"]), "constant_score": mp["constant_score"]}


def _window_of(detector: str, hp: dict) -> int:
    return int(hp["slidingWindow"] if detector == "IForest" else hp["window"])


def _count_constant_windows(x: np.ndarray, w: int) -> int:
    """Number of length-w windows with zero variance (D12-a).

    Uses the rolling min/max rather than a rolling std: exact, and it cannot report a
    false positive from floating-point cancellation.
    """
    n = len(x)
    if n < w:
        return 0
    from numpy.lib.stride_tricks import sliding_window_view
    v = sliding_window_view(np.asarray(x, dtype=np.float64), w)
    return int(np.count_nonzero(v.max(axis=1) == v.min(axis=1)))


def run_series(row: dict, fulldata_dir: str, det_cfg: dict, detector: str) -> dict:
    """All conditions for one series. Runs in a worker process; returns records only."""
    meta = parse_filename(row["filename"])
    x = load_series(Path(fulldata_dir) / row["filename"])
    x.setflags(write=False)
    calls, rows = [], []
    for cond in conditions(det_cfg, detector):
        fit_on, seed = cond["fit_on"], cond["seed"]
        base = {"series_num": row["num"], "detector": detector, "seed": seed, "fit_on": fit_on}
        x_fit, n_fit = slice_fit(x, meta, fit_on)
        hp = _hyperparams(detector, det_cfg, cond, x)
        w = _window_of(detector, hp)
        # --- fit ---
        t0 = time.perf_counter()
        try:
            det = fit_detector(detector, x_fit, seed=seed, **hp)
            fit_s = time.perf_counter() - t0
            calls.append({**base, "config": det.config, "phase": "fit", "n": n_fit, "runtime_s": fit_s,
                          "ok": True, "error": None, "train_end": meta.train_end_raw, "n_fit": n_fit})
        except Exception as e:
            calls.append({**base, "config": {**hp, "random_state": seed}, "phase": "fit", "n": n_fit,
                          "runtime_s": time.perf_counter() - t0, "ok": False,
                          "error": f"{type(e).__name__}: {e}", "train_end": meta.train_end_raw, "n_fit": n_fit})
            continue
        # --- score the full series ---
        t0 = time.perf_counter()
        try:
            s = det.score(x)
            score_s = time.perf_counter() - t0
            calls.append({**base, "config": det.config, "phase": "score", "n": len(x), "runtime_s": score_s,
                          "ok": True, "error": None, "train_end": meta.train_end_raw, "n_fit": n_fit})
        except Exception as e:
            calls.append({**base, "config": det.config, "phase": "score", "n": len(x),
                          "runtime_s": time.perf_counter() - t0, "ok": False,
                          "error": f"{type(e).__name__}: {e}", "train_end": meta.train_end_raw, "n_fit": n_fit})
            continue
        p = score_path(detector, row["num"], cond["tag"])
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, s)
        start0, stop0 = int(row["start0"]), int(row["stop0"])
        d = evaluate(s, train_end=meta.train_end_raw, start0=start0, stop0=stop0, window=w)
        # Diagnostics, not criteria (D-C5-5): a window score is reported at its
        # start + w//2, so an anomaly's elevated scores occupy the window support
        # [start - w + 1 + w//2, stop - 1 + w//2). For anomalies shorter than w the
        # peak can fall outside [start0, stop0) that P1 reads.
        sup_lo = max(0, start0 - w + 1 + w // 2)
        sup_hi = min(len(s), stop0 - 1 + w // 2 + 1)
        max_support = float(s[sup_lo:sup_hi].max()) if sup_hi > sup_lo else float("nan")
        # Sanity check pre-registered in D-C5-4: in an AB-join every window that lies
        # WHOLLY inside the reference matches itself, so its score must be ~0. Two
        # parts of s[:train_end] are not such windows and are excluded here:
        #   * the padded head s[:w//2] (carries the profile minimum, not a window score)
        #   * windows straddling train_end (start > train_end - w): they reach into the
        #     test region, which the reference does not contain.
        # Measured on 001: max over the contained part is 2.6e-06 while the straddling
        # part reaches 4.1 -- which is why the raw train_max is reported separately.
        train_scores = s[:meta.train_end_raw]
        sm_lo, sm_hi = w // 2, meta.train_end_raw - w + w // 2 + 1
        selfmatch = s[sm_lo:sm_hi] if sm_hi > sm_lo else np.empty(0)
        # D12-a: constant subsequences (sigma == 0 over a length-w window) in the
        # region an AB-join scores against. stumpy gives D^2 = 0 when both query and
        # reference window are constant and D^2 = m when exactly one is, so these
        # windows behave differently from every other one under masking (W2).
        n_const_train = _count_constant_windows(x[:meta.train_end_raw], w)
        rows.append({
            "num": row["num"], "name": row["name"], "variant": row["variant"],
            "domain_goswami": row["domain_goswami"], "is_medical": row["is_medical"],
            "content_group": row["content_group"], "name_group": row["name_group"],
            "detector": detector, "fit_on": fit_on, "max_features": cond.get("max_features", "NA"),
            "seed": seed, "n_fit": n_fit, "window": w,
            "n_estimators": hp.get("n_estimators", "NA"), "mode": cond.get("mode", "NA"),
            **d.as_row(),
            "max_support": max_support, "p1_support_bw": bool(max_support > d.thr_bw),
            "train_max": float(train_scores.max()), "train_p99": float(np.percentile(train_scores, 99)),
            "train_selfmatch_max": float(selfmatch.max()) if selfmatch.size else float("nan"),
            "n_selfmatch": int(selfmatch.size), "n_const_train": n_const_train,
            "fit_s": round(fit_s, 4), "score_s": round(score_s, 4),
            "score_file": str(p.relative_to(REPO_ROOT)),
        })
    return {"num": row["num"], "calls": calls, "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default="IForest", choices=["IForest", "MatrixProfile"])
    ap.add_argument("--limit", type=int, default=None, help="first N series only (smoke test)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--max-failure-rate", type=float, default=0.0)
    args = ap.parse_args()
    if args.run_id is None:
        args.run_id = datetime.now(timezone.utc).strftime(f"gate1_{args.detector}_%Y%m%dT%H%M%SZ")
    header = env_header()
    print(header)

    paths = load_yaml("paths")
    det_cfg = load_yaml("detectors")
    a1 = json.loads(resolve(paths["ucr"]["a1_result"]).read_text())
    if a1["stop"]:
        raise RuntimeError("01_download.py recorded STOP")
    fulldata_dir = str(Path(a1["anchors"]["official"]) / paths["ucr"]["fulldata_subdir"])

    with resolve(paths["manifest"]).open() as f:
        manifest = list(csv.DictReader(f))
    if len(manifest) != int(paths["ucr"]["expected_n_series"]):
        raise RuntimeError(f"manifest has {len(manifest)} rows, expected {paths['ucr']['expected_n_series']}")
    for r in manifest:
        r["num"] = int(r["num"])
        if r["start0"] == "NA":
            raise RuntimeError(f"{r['filename']}: start0 is NA (index convention undetermined)")
    if args.limit:
        manifest = manifest[:args.limit]
    conds = conditions(det_cfg, args.detector)
    print(f"detector={args.detector} series={len(manifest)} conditions={len(conds)} -> "
          f"{len(manifest)*len(conds)} fits; workers={args.workers} run_id={args.run_id}")

    RESULTS = results_dir(args.detector)
    RESULTS.mkdir(parents=True, exist_ok=True)
    run_dir = REPO_ROOT / "runs" / args.run_id
    t_start = time.perf_counter()
    det_rows: list[dict] = []
    with CallLog(run_dir) as log:
        def _absorb(res):
            for c in res["calls"]:
                log.record(**c)
            det_rows.extend(res["rows"])
            n_fail = sum(1 for c in res["calls"] if not c["ok"])
            print(f"  {res['num']:03d}: {len(res['rows'])}/{len(conds)} conditions scored"
                  + (f", {n_fail} FAILED" if n_fail else ""), flush=True)

        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futs = [ex.submit(run_series, r, fulldata_dir, det_cfg, args.detector) for r in manifest]
                for fut in as_completed(futs):
                    _absorb(fut.result())
        else:
            for r in manifest:
                _absorb(run_series(r, fulldata_dir, det_cfg, args.detector))
        summary = log.summary()
    elapsed = time.perf_counter() - t_start

    det_rows.sort(key=lambda r: (r["num"], r["fit_on"], str(r["max_features"]), r["seed"]))
    windows = [r["window"] for r in det_rows]
    with (RESULTS / "detection.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DETECTION_COLUMNS)
        w.writeheader()
        w.writerows(det_rows)
    out = {
        "run_id": args.run_id, "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "env_header": header, "n_series": len(manifest), "limit": args.limit,
        "detector_name": args.detector, "conditions": conds,
        "window_summary": ({"min": int(min(windows)), "median": float(np.median(windows)),
                            "max": int(max(windows))} if windows else None),
        "detector": det_cfg, "elapsed_s": elapsed, **summary,
        "n_detection_rows": len(det_rows), "expected_rows": len(manifest) * len(conds),
    }
    (RESULTS / "summary.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {RESULTS/'detection.csv'} ({len(det_rows)} rows), summary.json; "
          f"elapsed {elapsed:.0f}s; calls={summary['n_calls']} failed={summary['n_failed']} "
          f"rate={summary['failure_rate']:.3%}")
    if summary["failure_rate"] > args.max_failure_rate:
        print(f"FAIL: failure rate {summary['failure_rate']:.3%} > tolerance {args.max_failure_rate:.3%} "
              f"(see {run_dir/'failures.jsonl'})")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
