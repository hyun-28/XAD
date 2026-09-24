"""w2_gate2.py -- W2 rev1 §7 gate 2: time the E2 grid on 5 series, extrapolate; write reports/w2_gate2.md.

Gate 2 (rev1 §7):
  1. E0 accuracy passed (max|Δ| <= 1e-6)            <- results/w2/e0/summary.json (scripts/w2_e0.py)
  2. identity AI exactly 0.0 in every case          <- same
  3. predicted E2 wall time <= 2 h on the Mac, 4 workers, for the sample
     E2 grid per series = 9 operators x (20 normal positions x 3 lengths + 20 SAR positions of length
     |GT| + 1 anomaly region); per region: 1 score_patch before + per operator 1 application and
     1 score_patch after.

TIMING (D-G2-1): the 5 series of the 89-group sample at train_end quantiles 0/.25/.5/.75/1 run the
FULL E2 grid in 4 worker processes (2 numba threads each). Only wall times are kept. Every score
computed here is discarded immediately: no AI, resp or score value is stored, printed or summarised
(rev1 §12). Per-call models fitted by NNLS:
    score_patch  t = c0 + c1 n_B + c2 n_A n_B      n_A = windows in J(R), n_B = train_end - w + 1
    operator op  t = c0 + c1 n + c2 n c            c = max(1, w // 4)
The wall time for a sample = makespan of a longest-processing-time-first assignment of the
predicted per-series times to 4 workers.

The stumpy trace (rev1 §4-5) is checked against the installed source at run time: every cited line
must contain the quoted text, else the script fails.

Exit: 0 gate passed, 2 STOP (any gate condition failed), 1 error.
"""
from __future__ import annotations

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

REPORT = REPO_ROOT / "reports" / "w2_gate2.md"
TIMING_CSV = REPO_ROOT / "results" / "w2" / "gate2_timing.csv"

# (file, line, text that must appear on that line, function) — stumpy 1.14.1
TRACE = {
    "a": [("core.py", 2590, "def _rolling_isconstant(a, w):", "_rolling_isconstant"),
          ("core.py", 2614, "out[i] = np.ptp(a[i : i + w])", "_rolling_isconstant"),
          ("core.py", 2616, "return out == 0", "_rolling_isconstant"),
          ("core.py", 2650, "a_subseq_isconstant = _rolling_isconstant", "rolling_isconstant"),
          ("core.py", 4365, "T_subseq_isconstant = rolling_isconstant(T, m, T_subseq_isconstant)", "process_isconstant"),
          ("core.py", 2331, "T_subseq_isconstant = process_isconstant(T, m, T_subseq_isconstant)", "preprocess_diagonal"),
          ("stump.py", 695, "core.preprocess_diagonal(T_A, m, T_subseq_isconstant=T_A_subseq_isconstant)", "stump"),
          ("stump.py", 704, "core.preprocess_diagonal(T_B, m, T_subseq_isconstant=T_B_subseq_isconstant)", "stump"),
          ("core.py", 359, "def z_norm(a, axis=0, threshold=config.STUMPY_STDDEV_THRESHOLD):", "z_norm"),
          ("config.py", 14, '"STUMPY_STDDEV_THRESHOLD": 1e-7,', "_STUMPY_DEFAULTS")],
    "b": [("stump.py", 207, "pearson = min(1.0, pearson)", "_compute_diagonal"),
          ("stump.py", 482, "p_norm = np.abs(2 * m * (1 - ρ[0, :, ::-1]))", "_stump"),
          ("stump.py", 500, "np.sqrt(p_norm),", "_stump"),
          ("config.py", 21, '"STUMPY_FASTMATH_FLAGS": {"nsz", "arcp", "contract", "afn", "reassoc"},', "_STUMPY_DEFAULTS"),
          ("core.py", 3526, "def _check_P(P, threshold=1e-6):", "_check_P")],
    "c": [("stump.py", 201, "if T_B_subseq_isconstant[uint64_j] and T_A_subseq_isconstant[uint64_i]:", "_compute_diagonal"),
          ("stump.py", 202, "pearson = 1.0", "_compute_diagonal"),
          ("stump.py", 203, "elif T_B_subseq_isconstant[uint64_j] or T_A_subseq_isconstant[uint64_i]:", "_compute_diagonal"),
          ("stump.py", 204, "pearson = 0.5", "_compute_diagonal"),
          ("core.py", 1156, "D_squared = 0", "_calculate_squared_distance"),
          ("core.py", 1158, "D_squared = m", "_calculate_squared_distance")],
}


def check_trace() -> dict:
    import stumpy
    if stumpy.__version__ != "1.14.1":
        raise RuntimeError(f"stumpy {stumpy.__version__}: the trace was written for 1.14.1")
    root = Path(stumpy.__file__).parent
    for items in TRACE.values():
        for fname, line, text, _ in items:
            src = (root / fname).read_text(encoding="utf-8").splitlines()
            if text not in src[line - 1]:
                raise RuntimeError(f"stumpy trace drifted: {fname}:{line} does not contain {text!r}")
    return {"version": stumpy.__version__, "root": str(root)}


# --------------------------------------------------------------------------- timing worker

def e2_regions(lay, cfg, rng):
    from src import w2
    g = cfg["e2_grid"]
    out = []
    for f in g["lengths_w"]:
        L = w2.length_for(float(f), lay.w)
        for R in w2.draw_disjoint(rng, lay.admissible_starts(L), L, int(g["n_normal_positions"])):
            out.append(("normal", f"{f}w", R))
    L = lay.stop0 - lay.start0
    for R in w2.draw_disjoint(rng, lay.admissible_starts(L), L, int(g["n_sar_positions"])):
        out.append(("sar_normal", "GT", R))
    out.append(("anomaly", "GT", (lay.start0, lay.stop0)))
    return out


def time_series(num: int, cfg: dict) -> dict:
    from src import w2
    from src.detectors.matrixprofile import MatrixProfileDetector
    from src.perturb.operators import Ctx, NoDonorError, apply
    man, g1 = w2.manifest(), w2.gate1_primary()
    lay = w2.layout(num, man, g1)
    x = w2.load_checked(num, man)
    det = MatrixProfileDetector.fit(x[:lay.train_end], window=lay.w, mode="ab_join", constant_score="error")
    g = cfg["e2_grid"]
    regions = e2_regions(lay, cfg, np.random.default_rng([int(g["position_seed"]), num]))
    op_rng = np.random.default_rng([int(g["operator_seed"]), num])
    ctx = Ctx(train_end=lay.train_end, gt=(lay.start0, lay.stop0), w=lay.w)
    n_B = lay.train_end - lay.w + 1
    rows = []
    t_series = time.perf_counter()
    for rid, (kind, lab, R) in enumerate(regions):
        t0 = time.perf_counter()
        J, _ = det.score_patch(x, R)          # value discarded (rev1 §12)
        rows.append({"num": num, "rid": rid, "train_end": lay.train_end, "w": lay.w, "kind": kind, "length": lab, "L": R[1] - R[0], "op": "(before)",
                     "n_A": int(J.size), "n_B": n_B, "n": lay.n, "c": max(1, lay.w // 4),
                     "t_op": 0.0, "t_patch": time.perf_counter() - t0, "no_donor": False})
        for op in g["operators"]:
            t0 = time.perf_counter()
            try:
                xp = apply(op, x, [R], op_rng, ctx)
                no_donor = False
            except NoDonorError:
                xp, no_donor = None, True
            t_op = time.perf_counter() - t0
            t_patch = 0.0
            if xp is not None:
                t0 = time.perf_counter()
                det.score_patch(xp, R)            # value discarded (rev1 §12)
                t_patch = time.perf_counter() - t0
            del xp
            rows.append({"num": num, "rid": rid, "train_end": lay.train_end, "w": lay.w, "kind": kind, "length": lab, "L": R[1] - R[0], "op": op,
                         "n_A": int(J.size), "n_B": n_B, "n": lay.n, "c": max(1, lay.w // 4),
                         "t_op": t_op, "t_patch": t_patch, "no_donor": no_donor})
    return {"num": num, "rows": rows, "wall_s": time.perf_counter() - t_series,
            "counts": pd.Series([k for k, _, _ in regions]).value_counts().to_dict()}


# --------------------------------------------------------------------------- models

def nnls(X, y):
    from scipy.optimize import nnls as _nnls
    coef, _ = _nnls(X, y)
    return coef


def lpt_makespan(times, workers):
    load = [0.0] * workers
    for t in sorted(times, reverse=True):
        i = int(np.argmin(load))
        load[i] += t
    return max(load)


def predict_series(e: dict, models: dict, ops: list[str], cfg: dict) -> float:
    g = cfg["e2_grid"]
    w, n, te = e["w"], e["n"], e["train_end"]
    n_B, c = te - w + 1, max(1, w // 4)
    regs = []
    for f in g["lengths_w"]:
        p = e["positions"][f"{f}w"]
        regs += [p["L"]] * min(int(g["n_normal_positions"]), p["max_disjoint"])
    p = e["positions"]["GT"]
    regs += [p["L"]] * min(int(g["n_sar_positions"]), p["max_disjoint"]) + [e["gt_len"]]
    cp = models["patch"]
    total = 0.0
    for L in regs:
        n_A = L + w - 1
        t_patch = cp[0] + cp[1] * n_B + cp[2] * n_A * n_B
        total += t_patch * (1 + len(ops))
        for op in ops:
            co = models["op"][op]
            total += co[0] + co[1] * n + co[2] * n * c
    return total


# --------------------------------------------------------------------------- main

def main() -> int:
    cfg = load_yaml("w2")
    os.environ["NUMBA_NUM_THREADS"] = str(cfg["execution"]["numba_threads_per_worker"])
    from src.runlog import env_header
    header = env_header()
    print(header, flush=True)
    trace = check_trace()
    e0 = json.loads((REPO_ROOT / "results" / "w2" / "e0" / "summary.json").read_text())
    sample = yaml.safe_load((REPO_ROOT / "configs" / "w2_sample.yaml").read_text())
    ops_cfg = yaml.safe_load((REPO_ROOT / "configs" / "operators.yaml").read_text())["operators"]
    ops = cfg["e2_grid"]["operators"]
    if list(ops) != list(ops_cfg):
        raise RuntimeError("e2_grid.operators != configs/operators.yaml")
    ent = sorted(sample["series"], key=lambda e: (e["train_end"], e["num"]))
    idx = sorted({int(round(q * (len(ent) - 1))) for q in cfg["gate2"]["timing_train_end_quantiles"]})
    tnums = [ent[i]["num"] for i in idx]
    if len(tnums) != 5:
        raise RuntimeError(f"timing series not distinct: {tnums}")
    print(f"gate 2 timing series: {tnums}", flush=True)
    import multiprocessing as mp
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=int(cfg["execution"]["workers"]), mp_context=mp.get_context("spawn")) as ex:
        res = list(ex.map(time_series, tnums, [cfg] * len(tnums)))
    elapsed = time.perf_counter() - t0
    tim = pd.DataFrame([r for x in res for r in x["rows"]])
    TIMING_CSV.parent.mkdir(parents=True, exist_ok=True)
    tim.to_csv(TIMING_CSV, index=False)

    pt = tim[tim.t_patch > 0]
    models = {"patch": nnls(np.c_[np.ones(len(pt)), pt.n_B, pt.n_A * pt.n_B], pt.t_patch.to_numpy()), "op": {}}
    for op in ops:
        t = tim[tim.op == op]
        models["op"][op] = nnls(np.c_[np.ones(len(t)), t.n, t.n * t.c], t.t_op.to_numpy())
    by_num = {e["num"]: e for e in sample["series"]}
    val = []
    for x in res:
        pred = predict_series(by_num[x["num"]], models, ops, cfg)
        meas = float(tim[tim.num == x["num"]][["t_op", "t_patch"]].sum().sum())
        val.append({"num": x["num"], "measured_s": meas, "predicted_s": pred, "wall_s": x["wall_s"],
                    "counts": x["counts"]})
    pred = {e["num"]: predict_series(e, models, ops, cfg) for e in sample["series"]}
    s60 = [pred[e["num"]] for e in sample["series"] if e["in_sample_60"]]
    s89 = list(pred.values())
    W = int(cfg["execution"]["workers"])
    est = {"60": {"cpu_h": sum(s60) / 3600, "wall_h": lpt_makespan(s60, W) / 3600, "max_series_h": max(s60) / 3600},
           "89": {"cpu_h": sum(s89) / 3600, "wall_h": lpt_makespan(s89, W) / 3600, "max_series_h": max(s89) / 3600}}
    budget = float(cfg["gate2"]["budget_hours"])
    nd = tim[tim.op.isin(["recon_test", "recon_train"])].groupby("op").no_donor.agg(["sum", "count"])
    gate = {"1_e0_accuracy": bool(e0["accuracy_pass"]),
            "2_identity_ai_zero": bool(e0["identity_ai_zero_all"] and e0["identity_bitexact_all"]),
            "3_time_60": bool(est["60"]["wall_h"] <= budget), "3_time_89": bool(est["89"]["wall_h"] <= budget)}
    write_report(cfg, header, trace, e0, sample, ops_cfg, tim, models, val, est, gate, nd, tnums, elapsed)
    print(f"gate 2: {gate}; E2 wall estimate 60 = {est['60']['wall_h']:.2f} h, 89 = {est['89']['wall_h']:.2f} h")
    ok = gate["1_e0_accuracy"] and gate["2_identity_ai_zero"] and gate["3_time_60"]
    if not ok:
        print("STOP (rev1 §7/§11): a gate-2 condition failed")
        return 2
    return 0


# --------------------------------------------------------------------------- report

def md_table(header, rows):
    out = ["| " + " | ".join(map(str, header)) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def git_head() -> str:
    import subprocess
    return subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                          text=True).stdout.strip()


def write_report(cfg, header, trace, e0, sample, ops_cfg, tim, models, val, est, gate, nd, tnums, elapsed):
    L = []
    A = L.append
    yes = lambda b: "**PASS**" if b else "**FAIL → STOP**"  # noqa: E731
    A("# W2 rev1 — gate 2 (session 1)\n")
    A(f"GENERATED by scripts/w2_gate2.py at {datetime.now(timezone.utc).isoformat(timespec='seconds')}; "
      f"code HEAD `{git_head()}`. Do not hand-edit. No AI or faithfulness value appears in this report "
      "(rev1 §12): the timing run discards every score it computes.\n")
    A(f"`{header}`\n")
    A("## Gate 2 verdict (rev1 §7)\n")
    A(md_table(["#", "condition", "observed", "verdict"], [
        [1, "E0 accuracy: max|Δ| ≤ 1e-6", f"{e0['max_abs_delta']:.3g}", yes(gate["1_e0_accuracy"])],
        [2, "identity AI == 0.0 in every case", f"{e0['identity_ai_cases']}/{e0['identity_ai_cases']} exact; "
                                                 f"patch bit-identical {e0['n_regions']}/{e0['n_regions']}",
         yes(gate["2_identity_ai_zero"])],
        [3, f"E2 predicted wall time ≤ {cfg['gate2']['budget_hours']:g} h (4 workers), 60 groups",
         f"{est['60']['wall_h']:.2f} h", yes(gate["3_time_60"])],
        ["3′", "same, 89 groups (decides 60 → 89, rev1 §5)", f"{est['89']['wall_h']:.2f} h",
         "within budget" if gate["3_time_89"] else "over budget (stay at 60)"],
    ]))
    A("")

    # E0
    A("## 1. E0 — score_patch accuracy (rev1 §4-3)\n")
    A(f"{e0['n_regions']} regions on {len(e0['series'])} random series ({e0['series']}), "
      f"{e0['n_windows_compared']:,} window scores compared with a full recomputation of the perturbed "
      f"series (additive Gaussian noise; DECISIONS D-E0-1). Run with {e0['workers']} workers × "
      f"{e0['numba_threads_per_worker']} numba threads, {e0['elapsed_s']:.0f} s.\n")
    A(md_table(["statistic", "value"], [
        ["max |Δ|", f"{e0['max_abs_delta']:.3g}"],
        *[[f"quantile {k}", f"{v:.3g}"] for k, v in e0["quantiles"].items()],
        ["exactly 0", f"{e0['n_exact_zero']:,} ({100 * e0['n_exact_zero'] / e0['n_windows_compared']:.1f}%)"],
        ["regions over tol", len(e0["regions_over_tol"])],
        ["diagnostic: unperturbed patch vs stored gate-1 score, max |Δ|", f"{e0['unperturbed_vs_gate1_max_abs']:.3g}"],
    ]))
    A("\nDistribution of |Δ| (count per decade):\n")
    edges, counts = e0["hist_edges"], e0["hist_counts"]
    A(md_table(["|Δ| in", "count"], [[f"[{edges[i]}, {edges[i + 1]})", c] for i, c in enumerate(counts) if c]))
    A("\n**Identity (rev1 §4-4):** `score_patch(identity(x), R)` is bit-identical to `score_patch(x, R)` "
      f"in {e0['n_regions']}/{e0['n_regions']} regions, and AI = (resp_after − resp_before)/scale(x) is exactly "
      f"0.0 in {e0['identity_ai_cases']}/{e0['identity_ai_cases']} cases.\n")
    reg = pd.read_csv(REPO_ROOT / "results" / "w2" / "e0" / "regions.csv")
    reg["w_bin"] = pd.cut(reg.w, [0, 50, 100, 150, 300], right=True)
    reg["n_bin"] = pd.cut(reg.n, [0, 20_000, 50_000, 100_000, 1_000_000], right=True)
    fr = [float(f) for f in cfg["e0"]["length_fracs"]]
    reg["nominal"] = reg.L_over_w.map(lambda v: f"{min(fr, key=lambda f: abs(f - v))}w")   # |R| = round(f·w)
    A("**Speed (rev1 §4-6):** mean `score_patch` wall time per R (E0 run, perturbed series):\n")
    for col, lab in (("w_bin", "w"), ("nominal", "|R| (nominal)"), ("n_bin", "series length n")):
        g = reg.groupby(col, observed=True).t_patch_s.agg(["mean", "count"])
        A(md_table([lab, "mean s per R", "R"], [[k, f"{v['mean']:.4f}", int(v["count"])] for k, v in g.iterrows()]))
        A("")

    # stumpy trace
    A("## 2. stumpy source trace (rev1 §4-5) — stumpy " + trace["version"] + "\n")
    A("Every line below was read from the installed package at run time and must contain the quoted "
      f"text (`{trace['root']}`).\n")
    names = {"a": "(a) constant-subsequence criterion", "b": "(b) correlation near 1",
             "c": "(c) the D² = m branch"}
    for k, items in TRACE.items():
        A(f"**{names[k]}**\n")
        A(md_table(["file:line", "function", "code"], [[f"`stumpy/{f}:{ln}`", f"`{fn}`", f"`{t}`"]
                                                        for f, ln, t, fn in items]))
        A("")
    A("- **(a)** A subsequence is constant iff `ptp == 0` exactly (`_rolling_isconstant`); no tolerance. "
      "`stump` gets the flags through `preprocess_diagonal → process_isconstant → rolling_isconstant`. "
      "`STUMPY_STDDEV_THRESHOLD = 1e-7` is used by `core.z_norm` only, not on this path.\n"
      "- **(b)** No threshold snaps a near-1 correlation to distance 0: the only treatment is the clamp "
      "`min(1.0, pearson)` and `sqrt(|2m(1 − ρ)|)`. The covariance is accumulated along each diagonal "
      "in a function compiled with `fastmath` (incl. `reassoc`), so an exact match (true ρ = 1) returns "
      "a small positive distance. `_check_P` only warns when many distances are < 1e-6.\n"
      "- **(c)** One constant side → `pearson = 0.5` → D² = 2m(1 − 0.5) = m, D = √m; both constant → "
      "`pearson = 1.0` → D = 0. The assignment is exact, so D = `sqrt(m)` to the last bit (m ≤ 2^53).\n")
    g1 = pd.read_csv(REPO_ROOT / "results" / "gate1" / "MatrixProfile" / "detection.csv")
    g1 = g1[(g1.fit_on == "train_prefix") & (g1["mode"] == "ab_join")]
    sm = g1.train_selfmatch_max
    worst = g1.loc[sm.idxmax()]
    A("**D12 tol — proposal (the researcher decides; D12 is PENDING):**\n")
    A(md_table(["prediction", "why", "proposed tol"], [
        ["D12-a (i) s(j*)_after = √w; (ii) resp_after ≥ √w − tol",
         "branch (c) assigns ρ = 0.5 exactly → D = sqrt(w) bit-exact", "0 (exact); 1e-12 if a margin is wanted"],
        ["D12-a′ s(j*)_after = 0 (n_const_train > 0)", "branch (c) assigns ρ = 1.0 exactly → D = 0 bit-exact",
         "0 (exact)"],
        ["D12-b recon_train, |R| = w: s(j*)_after ≈ 0",
         "true ρ = 1 but computed through (b): residual depends on the series. Gate-1 exact self-matches "
         f"(training windows vs themselves, all 250): max {sm.max():.2g} (#{int(worst.num):03d}, w = "
         f"{int(worst.window)}), p99 {sm.quantile(0.99):.2g}, median {sm.median():.2g}",
         "1e-3 (≈ 3× the largest observed exact-match residual)"],
    ]))
    A("")

    # timing
    A("## 3. Gate-2 timing and the E2 estimate (rev1 §7-3)\n")
    A(f"Timing series (89-group sample at train_end quantiles 0/.25/.5/.75/1): {tnums}. Each ran the full "
      f"E2 grid ({len(cfg['e2_grid']['operators'])} operators; per region 1 score_patch before + 9 × "
      f"(operator + score_patch)) in {cfg['execution']['workers']} workers × "
      f"{cfg['execution']['numba_threads_per_worker']} numba threads; {elapsed:.0f} s wall in total. "
      f"Raw timings: `{TIMING_CSV.relative_to(REPO_ROOT)}` (times and sizes only).\n")
    per_region = (tim.t_op + tim.t_patch).groupby([tim.num, tim.rid]).sum()
    A(md_table(["num", "n", "train_end", "w", "regions (normal / SAR / anomaly)", "mean s per R (all 9 ops)",
                "measured s", "model s", "wall s"],
               [[v["num"], int(tim[tim.num == v["num"]].n.iloc[0]), int(tim[tim.num == v["num"]].train_end.iloc[0]),
                 int(tim[tim.num == v["num"]].w.iloc[0]),
                 f"{v['counts'].get('normal', 0)} / {v['counts'].get('sar_normal', 0)} / {v['counts'].get('anomaly', 0)}",
                 f"{per_region.loc[v['num']].mean():.3f}", f"{v['measured_s']:.1f}", f"{v['predicted_s']:.1f}",
                 f"{v['wall_s']:.1f}"] for v in val]))
    A("\nPer-call means over the timing run:\n")
    A(md_table(["operator", "mean operator s", "mean score_patch after s", "no donor"],
               [[op, f"{g.t_op.mean():.4f}", f"{g[g.t_patch > 0].t_patch.mean():.4f}" if (g.t_patch > 0).any() else "—",
                 int(g.no_donor.sum())] for op, g in tim.groupby("op", sort=False)]))
    cp = models["patch"]
    A(f"\nModel `score_patch`: t = {cp[0]:.3g} + {cp[1]:.3g}·n_B + {cp[2]:.3g}·n_A·n_B.\n")
    A(md_table(["sample", "predicted CPU h (sum)", "predicted wall h (LPT, 4 workers)", "largest series h",
                f"≤ {cfg['gate2']['budget_hours']:g} h"],
               [[k, f"{v['cpu_h']:.2f}", f"{v['wall_h']:.2f}", f"{v['max_series_h']:.2f}",
                 "yes" if v["wall_h"] <= cfg["gate2"]["budget_hours"] else "**no**"] for k, v in est.items()]))
    A("")
    if len(nd):
        A("recon donors not found in the timing run: " + ", ".join(
            f"{op} {int(r['sum'])}/{int(r['count'])}" for op, r in nd.iterrows())
          + " (rev1 §11 threshold 5% applies to E2).\n")

    # sample
    A("## 4. Sample (rev1 §5; `configs/w2_sample.yaml`, DECISIONS D-S-1)\n")
    df = pd.DataFrame(sample["series"])
    rows = []
    for d in sorted(sample["domain_groups_89"]):
        s89, s60 = df[df.domain == d], df[(df.domain == d) & df.in_sample_60]
        rows.append([d, sample["domain_groups_89"][d], len(s60), int(s60.detected.sum()), int(s89.detected.sum()),
                     int((s60.n_const_train > 0).sum()), int((s89.n_const_train > 0).sum())])
    rows.append(["**total**", len(df), int(df.in_sample_60.sum()), int(df[df.in_sample_60].detected.sum()),
                 int(df.detected.sum()), int((df[df.in_sample_60].n_const_train > 0).sum()),
                 int((df.n_const_train > 0).sum())])
    A(md_table(["domain", "groups (89)", "groups (60)", "detected (60)", "detected (89)",
                "n_const_train > 0 (60)", "n_const_train > 0 (89)"], rows))
    few = [(int(e["num"]), k, v["max_disjoint"]) for e in sample["series"] for k, v in e["positions"].items()
           if k != "GT" and v["max_disjoint"] < cfg["e2_grid"]["n_normal_positions"]]
    sar_few = sum(e["positions"]["GT"]["max_disjoint"] < 10 for e in sample["series"])
    A(f"\nscale(x) invalid: {int((~df.scale_ok).sum())}/89. Series with < 20 disjoint normal positions at "
      f"some length: {few if few else 'none'}. SAR flag `sar_few_positions` (< 10 positions of length |GT|): "
      f"{sar_few}/89.\n")

    # operators
    A("## 5. Operators (rev1 §6; `configs/operators.yaml`)\n")
    A(md_table(["operator", "definition", "canonical_shape", "uses_reference_set", "stochastic"],
               [[k, v["definition"], v["canonical_shape"], v["uses_reference_set"], v["stochastic"]]
                for k, v in ops_cfg.items()]))
    A("")
    A("## 6. Open items\n")
    A("- D10–D13, D-E3-1(rev): PENDING in DECISIONS; E2/E3 not run.\n"
      "- D12 tol: proposal in §2.\n"
      "- Multi-run semantics for E3 (DECISIONS D-E1-2).\n"
      "- 60 → 89 groups: the 89-group estimate is within budget (§3); rev1 §5 says to move to 89 in that case.\n"
      f"- Series with fewer than 20 disjoint normal positions at some length (E2a will place fewer): {few or 'none'}.\n"
      "- B2/B5 take their statistics over the whole series, including the training part and the anomaly "
      "(legacy definition, kept unchanged per rev1 §6).\n"
      "- The timing includes numba JIT compilation on each worker's first call (conservative).\n")
    REPORT.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
