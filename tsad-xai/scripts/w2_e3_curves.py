"""w2_e3_curves.py -- E3 curves recomputed under D-E3-5 (random seeded ties; decision commit b8b783c).

Attributions are NOT recomputed: they are read from results/w2/e3_attr.csv.gz (E3 run, 6c83779).
Only the MoRF/LeRF curves are recomputed, with
    order      D-E3-5: signed descending, ties by a seeded random rank with seed = (series, AM) only;
               LeRF = exact reverse of MoRF
    conditions main     4 AMs x (B1-B6 + recon_train), attributions explained with recon_test
               self     FeatureAblation / KernelSHAP explained with p, evaluated under p (p in B1-B6)
               abs      |R| order, FeatureAblation / KernelSHAP (main attributions) under B1-B6
    rng keys   stochastic operators use exactly E3's keys (main/abs: tag 0, self: tag 1), so only the
               order differs from the index-tie version.
Check (fail loud): for attributions without ties (Random, KernelSHAP in this sample) the D-E3-5 order equals
the index-tie order, so their main curves must equal the stored E3 curves exactly.

Writes results/w2/faithfulness_v2.csv.gz, results/w2/e3v2_series.csv, results/w2/e3v2_run.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import REPO_ROOT, load_yaml  # noqa: E402

OUT = REPO_ROOT / "results" / "w2"
DECISION_COMMIT_2 = "b8b783cc027e8cc3bf44f49219410ac5a69d35e9"
TIE_TAG = 5


def run_series(e: dict, cfg: dict, attrs: dict, v1: dict) -> dict:
    from w2_e3 import AMS, EVAL_OPS, EXPL_OP, SEPARATE_OPS, Series, op_index
    from src.metrics import faithfulness_ad as fa
    t0 = time.perf_counter()
    seed = int(cfg["e3"]["seed"])
    sr = Series(e, seed)
    ties = {am: fa.tie_rank(sr.K, np.random.default_rng([seed, sr.num, AMS.index(am), TIE_TAG])) for am in AMS}
    rows, tie_stats = [], []

    def curves(attr, am, op, tag, key):
        oi, ai = op_index(op), AMS.index(am)
        fm = lambda m, t: sr.f_masked(m, op, (tag, oi, ai, t))  # noqa: E731 (E3's keys)
        morf = fa.curve_from_order(sr.f0, fm, fa.order_d35(attr, ties[am], "MoRF", key))
        lerf = fa.curve_from_order(sr.f0, fm, fa.order_d35(attr, ties[am], "LeRF", key))
        return morf, lerf, fa.dds(morf, lerf, sr.scale)

    def add(cond, am, expl, op, morf, lerf, d):
        rows.append({"series_id": sr.num, "condition": cond, "am": am, "expl_op": expl, "eval_op": op, "dds": d,
                     "morf": json.dumps(morf.tolist()), "lerf": json.dumps(lerf.tolist())})

    for am in AMS:
        a = attrs[(sr.num, "main", am, EXPL_OP)]
        if a.size != sr.K:
            raise RuntimeError(f"{sr.num} {am}: stored attribution has {a.size} segments, K = {sr.K}")
        _, cnt = np.unique(a, return_counts=True)
        tie_stats.append({"series_id": sr.num, "condition": "main", "am": am, "expl_op": EXPL_OP, "K": sr.K,
                          "tied_segments": int(cnt[cnt > 1].sum())})
        for op in EVAL_OPS + SEPARATE_OPS:
            morf, lerf, d = curves(a, am, op, 0, "signed")
            add("main" if op in EVAL_OPS else "separate", am, EXPL_OP, op, morf, lerf, d)
            if cnt.max() == 1:                                   # no ties: must equal the stored E3 curve
                old = v1[(sr.num, "main" if op in EVAL_OPS else "separate", am, op)]
                if not (np.array_equal(morf, old[0]) and np.array_equal(lerf, old[1])):
                    raise RuntimeError(f"{sr.num} {am} {op}: tie-free curve differs from the stored E3 curve")
    for op in EVAL_OPS:
        for am in ("FeatureAblation", "KernelSHAP"):
            a = attrs[(sr.num, "sensitivity", am, op)]
            _, cnt = np.unique(a, return_counts=True)
            tie_stats.append({"series_id": sr.num, "condition": "self", "am": am, "expl_op": op, "K": sr.K,
                              "tied_segments": int(cnt[cnt > 1].sum())})
            add("self", am, op, op, *curves(a, am, op, 1, "signed"))
            add("abs", am, EXPL_OP, op, *curves(attrs[(sr.num, "main", am, EXPL_OP)], am, op, 0, "abs"))
    return {"rows": rows, "ties": tie_stats,
            "info": {"series_id": sr.num, "K": sr.K, "n_evals": sr.n_evals, "b3_runs": sr.b3_runs,
                     "b3_overlap": sr.b3_overlap, "seconds": time.perf_counter() - t0}}


def main() -> int:
    cfg = load_yaml("w2")
    os.environ["NUMBA_NUM_THREADS"] = str(cfg["execution"]["numba_threads_per_worker"])
    import subprocess
    from src.runlog import env_header
    from w2_e2 import check_decisions
    header = env_header()
    print(header, flush=True)
    dec = check_decisions()
    if subprocess.run(["git", "-C", str(REPO_ROOT), "merge-base", "--is-ancestor", DECISION_COMMIT_2, "HEAD"]).returncode:
        raise RuntimeError("D-E3-5 decision commit b8b783c is not an ancestor of HEAD")
    sample = yaml.safe_load((REPO_ROOT / "configs" / "w2_sample.yaml").read_text())["series"]
    at = pd.read_csv(OUT / "e3_attr.csv.gz")
    v1f = pd.read_csv(OUT / "faithfulness.csv.gz")
    per = {}
    for e in sample:
        num = int(e["num"])
        a = at[at.series_id == num]
        attrs = {(num, r.condition, r.am, r.expl_op): np.array(json.loads(r.attr)) for r in a.itertuples()}
        f = v1f[(v1f.series_id == num) & v1f.condition.isin(["main", "separate"])]
        v1 = {(num, r.condition, r.am, r.eval_op): (np.array(json.loads(r.morf)), np.array(json.loads(r.lerf)))
              for r in f.itertuples()}
        per[num] = (attrs, v1)
    import multiprocessing as mp
    t0 = time.perf_counter()
    order = sorted(sample, key=lambda e: -e["n"])
    with ProcessPoolExecutor(max_workers=int(cfg["execution"]["workers"]), mp_context=mp.get_context("spawn")) as ex:
        res = list(ex.map(run_series, order, [cfg] * len(order), [per[int(e["num"])][0] for e in order],
                          [per[int(e["num"])][1] for e in order]))
    rows = pd.DataFrame([r for x in res for r in x["rows"]])
    ties = pd.DataFrame([r for x in res for r in x["ties"]])
    info = pd.DataFrame([x["info"] for x in res]).sort_values("series_id")
    rows.to_csv(OUT / "faithfulness_v2.csv.gz", index=False)
    ties.to_csv(OUT / "e3v2_ties.csv", index=False)
    info.to_csv(OUT / "e3v2_series.csv", index=False)
    (OUT / "e3v2_run.json").write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "env_header": header, **dec,
        "decision_commit_2": DECISION_COMMIT_2, "elapsed_s": time.perf_counter() - t0, "n_series": len(info),
        "n_rows": len(rows), "tie_tag": TIE_TAG,
        "tie_free_check": "passed (Random/KernelSHAP main curves identical to the index-tie E3 curves)"}, indent=2))
    print(f"E3 v2 curves: {len(info)} series, {len(rows)} rows, {time.perf_counter() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
