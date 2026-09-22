"""03_gate1.py -- BRIEF Task C: run IForest on the 250 official UCR series.

Grid (configs/detectors.yaml): fit_on {train_prefix, full} x max_features {1, 1.0}
x seeds {0, 1, 2} = 12 conditions per series. For every condition:
  slice_fit -> fit_detector (once) -> score(full series) -> save raw scores ->
  pre-registered criteria (src/gate1.py).

Writes
  results/gate1/scores/<num>_<fit_on>_mf<mf>_seed<k>.npy   raw scores (git-ignored)
  results/gate1/detection.csv                               series x condition x P1/P2
  results/gate1/summary.json                                run id, grid, failure rate, timing
  runs/<run_id>/calls.jsonl, failures.jsonl, env.txt        BRIEF B5 call log

Failures are recorded, never skipped silently; the loop continues and the
exit code is 1 if the failure rate exceeds --max-failure-rate (default 0).
Judgement (gate pass/STOP) is done by scripts/04_report.py.

Usage:  python scripts/03_gate1.py [--limit 5] [--workers 4] [--run-id ID]
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

RESULTS = REPO_ROOT / "results" / "gate1"
DETECTION_COLUMNS = [
    "num", "name", "variant", "domain_goswami", "is_medical", "content_group", "name_group",
    "fit_on", "max_features", "seed", "n_fit", "window", "n_estimators",
    "n", "train_end", "start0", "stop0", "max_anom", "thr_b0", "thr_bw", "n_normal_b0", "n_normal_bw",
    "p1_b0", "p1_bw", "argmax0", "p2_archive", "p2_brief", "fit_s", "score_s", "score_file",
]


def conditions(det_cfg: dict) -> list[tuple[str, object, int]]:
    ifo = det_cfg["iforest"]
    return [(fit_on, mf, int(seed))
            for fit_on in det_cfg["fit_on_variants"]
            for mf in ifo["max_features_variants"]
            for seed in det_cfg["seeds"]]


def mf_tag(mf) -> str:
    return f"mf{mf}" if isinstance(mf, int) else f"mf{mf:.1f}"


def score_path(num: int, fit_on: str, mf, seed: int) -> Path:
    return RESULTS / "scores" / f"{num:03d}_{fit_on}_{mf_tag(mf)}_seed{seed}.npy"


def run_series(row: dict, fulldata_dir: str, det_cfg: dict) -> dict:
    """All conditions for one series. Runs in a worker process; returns records only."""
    meta = parse_filename(row["filename"])
    x = load_series(Path(fulldata_dir) / row["filename"])
    x.setflags(write=False)
    ifo = det_cfg["iforest"]
    hp_base = {"slidingWindow": int(ifo["slidingWindow"]), "n_estimators": int(ifo["n_estimators"]),
               "n_jobs": int(ifo["n_jobs"]), "constant_score": det_cfg["constant_score"]}
    calls, rows = [], []
    for fit_on, mf, seed in conditions(det_cfg):
        base = {"series_num": row["num"], "detector": "IForest", "seed": seed, "fit_on": fit_on}
        x_fit, n_fit = slice_fit(x, meta, fit_on)
        hp = {**hp_base, "max_features": mf}
        # --- fit ---
        t0 = time.perf_counter()
        try:
            det = fit_detector("IForest", x_fit, seed=seed, **hp)
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
        p = score_path(row["num"], fit_on, mf, seed)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, s)
        d = evaluate(s, train_end=meta.train_end_raw, start0=int(row["start0"]), stop0=int(row["stop0"]),
                     window=hp["slidingWindow"])
        rows.append({
            "num": row["num"], "name": row["name"], "variant": row["variant"],
            "domain_goswami": row["domain_goswami"], "is_medical": row["is_medical"],
            "content_group": row["content_group"], "name_group": row["name_group"],
            "fit_on": fit_on, "max_features": mf, "seed": seed, "n_fit": n_fit,
            "window": hp["slidingWindow"], "n_estimators": hp["n_estimators"],
            **d.as_row(), "fit_s": round(fit_s, 4), "score_s": round(score_s, 4),
            "score_file": str(p.relative_to(REPO_ROOT)),
        })
    return {"num": row["num"], "calls": calls, "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="first N series only (smoke test)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("gate1_%Y%m%dT%H%M%SZ"))
    ap.add_argument("--max-failure-rate", type=float, default=0.0)
    args = ap.parse_args()
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
    conds = conditions(det_cfg)
    print(f"series={len(manifest)} conditions={len(conds)} -> {len(manifest)*len(conds)} fits; "
          f"workers={args.workers} run_id={args.run_id}")

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
                futs = [ex.submit(run_series, r, fulldata_dir, det_cfg) for r in manifest]
                for fut in as_completed(futs):
                    _absorb(fut.result())
        else:
            for r in manifest:
                _absorb(run_series(r, fulldata_dir, det_cfg))
        summary = log.summary()
    elapsed = time.perf_counter() - t_start

    det_rows.sort(key=lambda r: (r["num"], r["fit_on"], str(r["max_features"]), r["seed"]))
    with (RESULTS / "detection.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DETECTION_COLUMNS)
        w.writeheader()
        w.writerows(det_rows)
    out = {
        "run_id": args.run_id, "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "env_header": header, "n_series": len(manifest), "limit": args.limit,
        "conditions": [{"fit_on": a, "max_features": b, "seed": c} for a, b, c in conds],
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
