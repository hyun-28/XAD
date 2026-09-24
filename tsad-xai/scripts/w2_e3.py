"""w2_e3.py -- W2 rev1 §9 E3: attribution-method ranking under different evaluation operators (C2).

Definitions: D13, D-E3-1(rev) as approved (commit f29eddf), src/metrics/faithfulness_ad.py.
    explanation operator   recon_test (fixed)
    evaluation operators   B1-B6 (main); recon_train (reported separately); identity excluded
    AMs                    Random, FeatureAblation, KernelSHAP (S samples), MPNative
    sensitivity            per evaluation operator p: FeatureAblation and KernelSHAP recomputed with
                           explanation operator = p, evaluated under p
    curves                 MoRF / LeRF, 10 steps; DDS with max_diff = scale(x)

Modes
    --budget   rev1 §9: time the pipeline's evaluation kinds on 3 series (train_end quantiles 0/.5/1 of
               the sample) and extrapolate main + sensitivity to all 89 series; writes
               results/w2/e3_budget.json. No attribution or faithfulness value is stored.
    (default)  run E3 on all 89 series; writes results/w2/faithfulness.csv.gz (one row per series x
               condition x AM x evaluation operator: DDS and both curves), results/w2/e3_series.csv
               (K, g, eval counts, B3 context-overlap counts, exclusions), results/w2/e3_attr.csv.gz.

S (KernelSHAP samples) is read from configs/w2.yaml `e3.kernel_shap_samples` (rule D-E3-2).
"""
from __future__ import annotations

import argparse
import json
import math
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
from src.config import REPO_ROOT, load_yaml  # noqa: E402

OUT = REPO_ROOT / "results" / "w2"
EVAL_OPS = ["B1_zero", "B2_global_mean", "B3_local_mean", "B4_linear_interp", "B5_gaussian", "B6_shuffle"]
EXPL_OP = "recon_test"
SEPARATE_OPS = ["recon_train"]
AMS = ["Random", "FeatureAblation", "KernelSHAP", "MPNative"]
B3_PAD = 50       # src/perturbations.py:51 b3_local_mean(pad=50)


def s_rule(K: int, rule: str) -> int:
    if rule == "shap_auto":
        return 2 * K + 2048          # shap KernelExplainer nsamples="auto"
    if rule == "captum_default":
        return 25                    # captum 0.9.0 KernelShap.attribute(n_samples=25)
    raise ValueError(rule)


class Series:
    """One series set up for E3: detector, Ω, segments, masked-f evaluator with counters."""

    def __init__(self, e: dict, seed: int):
        from src import w2
        from src.detectors.matrixprofile import MatrixProfileDetector
        from src.metrics import faithfulness_ad as fa
        from src.perturb.operators import Ctx, ReconCache
        man, g1 = w2.manifest(), w2.gate1_primary()
        self.num = int(e["num"])
        self.lay = lay = w2.layout(self.num, man, g1)
        self.x = w2.load_checked(self.num, man)
        self.scale = float(e["scale"])
        self.det = MatrixProfileDetector.fit(self.x[:lay.train_end], window=lay.w, mode="ab_join",
                                             constant_score="error")
        self.R = (lay.start0, lay.stop0)
        self.ctx = Ctx(train_end=lay.train_end, gt=self.R, w=lay.w)
        self.cache = ReconCache(self.x, self.ctx)
        self.J0, self.s0 = self.det.score_patch(self.x, self.R)
        self.f0 = float(self.s0.max())
        self.om = fa.omega(lay.train_end, self.R, lay.w, lay.n)
        self.segs, self.g = fa.segments(self.om, lay.w)
        self.K = len(self.segs)
        self.seed = seed
        self.n_evals = 0
        self.b3_runs = 0
        self.b3_overlap = 0

    def f_masked(self, masked, op: str, rng_key=None) -> float:
        from src.metrics.faithfulness_ad import runs_of
        from src.perturb.operators import META, apply_cached
        runs = runs_of(self.segs, masked)
        if not runs:
            return self.f0
        if op == "B3_local_mean":
            for a, b in runs:
                self.b3_runs += 1
                ctx_iv = [(a - B3_PAD, a), (b, b + B3_PAD)]
                if any(ra < c1 and c0 < rb for ra, rb in runs if (ra, rb) != (a, b) for c0, c1 in ctx_iv):
                    self.b3_overlap += 1
        rng = np.random.default_rng([self.seed, self.num, *rng_key]) if META[op]["stochastic"] else None
        xp = apply_cached(op, self.x, runs, rng, self.ctx, self.cache)
        self.n_evals += 1
        return float(self.det.score_patch(xp, self.R)[1].max())


def op_index(op: str) -> int:
    return (EVAL_OPS + SEPARATE_OPS + [EXPL_OP]).index(op)


def explain(sr: Series, op: str, S: int, tag: int) -> dict:
    """FeatureAblation and KernelSHAP with explanation operator `op`."""
    from src.metrics import faithfulness_ad as fa
    oi = op_index(op)
    fa_attr = fa.feature_ablation(sr.f0, lambda m: sr.f_masked(m, op, (tag, oi, 1, int(m[0]))), sr.K)
    counter = iter(range(10 ** 9))

    def f_present(z):
        masked = np.flatnonzero(z == 0).tolist()
        return sr.f_masked(masked, op, (tag, oi, 2, next(counter)))

    ks_attr = fa.kernel_shap(f_present, sr.K, S, np.random.default_rng([sr.seed, sr.num, tag, oi, 3]))
    return {"FeatureAblation": fa_attr, "KernelSHAP": ks_attr}


def curves_for(sr: Series, attr: np.ndarray, am_i: int, op: str, tag: int):
    from src.metrics import faithfulness_ad as fa
    oi = op_index(op)
    fm = lambda m, t: sr.f_masked(m, op, (tag, oi, am_i, t))  # noqa: E731 (same key for MoRF and LeRF)
    morf = fa.curve(sr.f0, fm, attr, "MoRF")
    lerf = fa.curve(sr.f0, fm, attr, "LeRF")
    return morf, lerf, fa.dds(morf, lerf, sr.scale)


def run_series(e: dict, cfg: dict) -> dict:
    from src.metrics import faithfulness_ad as fa
    from src.perturb.operators import NoDonorError
    t_start = time.perf_counter()
    c3 = cfg["e3"]
    sr = Series(e, int(c3["seed"]))
    S = s_rule(sr.K, c3["kernel_shap_samples"])
    info = {"series_id": sr.num, "content_group": e["content_group"], "domain": e["domain"],
            "detected": bool(e["detected"]), "K": sr.K, "g": sr.g, "omega_len": sr.om[1] - sr.om[0], "S": S,
            "excluded": "", "scale": sr.scale}
    rows, attrs = [], []
    try:
        main = explain(sr, EXPL_OP, S, tag=0)
    except NoDonorError as ex:
        info["excluded"] = f"no donor for recon_test explanation: {ex}"
        info["seconds"] = time.perf_counter() - t_start
        return {"info": info, "rows": rows, "attrs": attrs}
    am_attr = {"Random": fa.random_attribution(sr.K, np.random.default_rng([sr.seed, sr.num, 0])),
               **main}
    mp_attr, mp_info = fa.mp_native(sr.x, sr.x[:sr.lay.train_end], sr.J0, sr.s0, sr.lay.w, sr.segs)
    am_attr["MPNative"] = mp_attr
    info.update({f"mpnative_{k}": v for k, v in mp_info.items()})
    for am in AMS:
        attrs.append({"series_id": sr.num, "condition": "main", "am": am, "expl_op": EXPL_OP,
                      "attr": json.dumps([float(v) for v in am_attr[am]])})
    for op in EVAL_OPS + SEPARATE_OPS:
        cond = "main" if op in EVAL_OPS else "separate"
        for ai_, am in enumerate(AMS):
            try:
                morf, lerf, d = curves_for(sr, am_attr[am], ai_, op, tag=0)
            except NoDonorError:
                rows.append({"series_id": sr.num, "condition": cond, "am": am, "expl_op": EXPL_OP, "eval_op": op,
                             "dds": np.nan, "morf": "", "lerf": "", "no_donor": True})
                continue
            rows.append({"series_id": sr.num, "condition": cond, "am": am, "expl_op": EXPL_OP, "eval_op": op,
                         "dds": d, "morf": json.dumps(morf.tolist()), "lerf": json.dumps(lerf.tolist()),
                         "no_donor": False})
    # sensitivity: explanation operator = evaluation operator (FA, KS only)
    for op in EVAL_OPS:
        sens = explain(sr, op, S, tag=1)
        for am in ("FeatureAblation", "KernelSHAP"):
            attrs.append({"series_id": sr.num, "condition": "sensitivity", "am": am, "expl_op": op,
                          "attr": json.dumps([float(v) for v in sens[am]])})
            morf, lerf, d = curves_for(sr, sens[am], AMS.index(am), op, tag=1)
            rows.append({"series_id": sr.num, "condition": "sensitivity", "am": am, "expl_op": op, "eval_op": op,
                         "dds": d, "morf": json.dumps(morf.tolist()), "lerf": json.dumps(lerf.tolist()),
                         "no_donor": False})
    info.update({"n_evals": sr.n_evals, "b3_runs": sr.b3_runs, "b3_overlap": sr.b3_overlap,
                 "seconds": time.perf_counter() - t_start})
    return {"info": info, "rows": rows, "attrs": attrs}


# --------------------------------------------------------------------------- budget

def budget_series(e: dict, cfg: dict) -> dict:
    """Time each evaluation kind on one series; keep times and counts only (no f, attribution or DDS)."""
    c3 = cfg["e3"]
    sr = Series(e, int(c3["seed"]))
    rng = np.random.default_rng([int(c3["seed"]), sr.num, 99])
    reps = int(c3["budget_reps"])
    out = {"num": sr.num, "K": sr.K, "n": sr.lay.n, "n_B": sr.lay.train_end - sr.lay.w + 1,
           "n_A": int(sr.J0.size), "w": sr.lay.w}

    def timed(kind, masks, op):
        ts = []
        for i, m in enumerate(masks):
            t0 = time.perf_counter()
            sr.f_masked(m, op, (7, op_index(op), 0, i))
            ts.append(time.perf_counter() - t0)
        out[kind] = float(np.mean(ts))

    singles = [[int(k)] for k in rng.choice(sr.K, size=min(reps, sr.K), replace=False)]
    randoms = [sorted(rng.choice(sr.K, size=int(rng.integers(1, sr.K)), replace=False).tolist()) if sr.K > 1 else [0]
               for _ in range(reps)]
    prefixes = [list(range(math.ceil(t * sr.K / 10))) for t in range(1, 11)]
    timed("t_fa_recon", singles, EXPL_OP)          # cold cache: FA's single-segment runs
    timed("t_ks_recon", randoms, EXPL_OP)          # random coalitions, mostly new runs
    for op in EVAL_OPS + SEPARATE_OPS:
        timed(f"t_curve_{op}", prefixes, op)
        timed(f"t_rand_{op}", randoms, op)
    return out


def predict(e: dict, per_eval: dict, S_rule: str) -> float:
    """Seconds for one series = counts x per-eval seconds (kinds as in budget_series)."""
    K = max(1, math.ceil(min(e["gt_len"] + 2 * e["w"] - 2, e["n"]) / max(1, e["w"] // 4)))
    K = min(K, 64)
    S = s_rule(K, S_rule)
    t = K * per_eval["t_fa_recon"] + S * per_eval["t_ks_recon"]                      # main FA, KS
    for op in EVAL_OPS + SEPARATE_OPS:
        t += len(AMS) * 2 * 10 * per_eval[f"t_curve_{op}"]                               # main curves
    for op in EVAL_OPS:
        t += K * per_eval[f"t_rand_{op}"] + S * per_eval[f"t_rand_{op}"]                # sensitivity FA, KS
        t += 2 * 2 * 10 * per_eval[f"t_curve_{op}"]                                     # sensitivity curves
    return t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml("w2")
    os.environ["NUMBA_NUM_THREADS"] = str(cfg["execution"]["numba_threads_per_worker"])
    from src.runlog import env_header
    header = env_header()
    print(header, flush=True)
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from w2_e2 import check_decisions
    dec = check_decisions()
    sample = yaml.safe_load((REPO_ROOT / "configs" / "w2_sample.yaml").read_text())["series"]
    W = int(cfg["execution"]["workers"])
    import multiprocessing as mp
    ctxm = mp.get_context("spawn")
    OUT.mkdir(parents=True, exist_ok=True)
    if args.budget:
        ent = sorted(sample, key=lambda e: (e["train_end"], e["num"]))
        pick = [ent[int(round(q * (len(ent) - 1)))] for q in (0.0, 0.5, 1.0)]
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=W, mp_context=ctxm) as ex:
            meas = list(ex.map(budget_series, pick, [cfg] * len(pick)))
        # per-eval model per kind: t = c0 + c1 * n_A * n_B + c2 * n  (NNLS over the 3 series)
        from scipy.optimize import nnls
        kinds = [k for k in meas[0] if k.startswith("t_")]
        X = np.array([[1.0, m["n_A"] * m["n_B"], m["n"]] for m in meas])
        coefs = {k: nnls(X, np.array([m[k] for m in meas]))[0].tolist() for k in kinds}
        res = {}
        for rule in ("shap_auto", "captum_default"):
            per = []
            for e in sample:
                nA = e["gt_len"] + e["w"] - 1
                nB = e["train_end"] - e["w"] + 1
                pe = {k: float(np.dot(coefs[k], [1.0, nA * nB, e["n"]])) for k in kinds}
                per.append(predict(e, pe, rule))
            load = [0.0] * W
            for t in sorted(per, reverse=True):
                load[int(np.argmin(load))] += t
            res[rule] = {"cpu_h": sum(per) / 3600, "wall_h": max(load) / 3600, "max_series_h": max(per) / 3600}
        doc = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "env_header": header,
               **dec, "elapsed_s": time.perf_counter() - t0, "series": [m["num"] for m in meas],
               "measurements": meas, "coefs": coefs, "estimate": res,
               "budget_hours": float(cfg["e3"]["budget_hours"])}
        (OUT / "e3_budget.json").write_text(json.dumps(doc, indent=2))
        for rule, v in res.items():
            print(f"E3 budget [{rule}]: CPU {v['cpu_h']:.2f} h, wall {v['wall_h']:.2f} h (4 workers), "
                  f"largest series {v['max_series_h']:.2f} h", flush=True)
        chosen = res[cfg["e3"]["kernel_shap_samples"]]
        if chosen["wall_h"] > float(cfg["e3"]["budget_hours"]):
            print("STOP (rev1 §9/§11): E3 predicted time exceeds the budget")
            return 2
        return 0
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=W, mp_context=ctxm) as ex:
        res = list(ex.map(run_series, sorted(sample, key=lambda e: -e["n"]), [cfg] * len(sample)))
    rows = pd.DataFrame([r for x in res for r in x["rows"]])
    info = pd.DataFrame([x["info"] for x in res]).sort_values("series_id")
    attrs = pd.DataFrame([r for x in res for r in x["attrs"]])
    rows.to_csv(OUT / "faithfulness.csv.gz", index=False)
    attrs.to_csv(OUT / "e3_attr.csv.gz", index=False)
    info.to_csv(OUT / "e3_series.csv", index=False)
    (OUT / "e3_run.json").write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "env_header": header, **dec,
        "elapsed_s": time.perf_counter() - t0, "kernel_shap_samples": cfg["e3"]["kernel_shap_samples"],
        "n_series": len(info), "n_excluded": int((info.excluded != "").sum())}, indent=2))
    print(f"E3: {len(info)} series, {len(rows)} curve rows, excluded {int((info.excluded != '').sum())}, "
          f"{time.perf_counter() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
