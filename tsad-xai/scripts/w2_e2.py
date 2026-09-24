"""w2_e2.py -- W2 rev1 §8 E2: Artifact Index (E2a) and SAR (E2b) on the 89-group sample.

Decisions: DECISIONS "W2 rev1 결정 — 승인" (commit f29eddf) and D-E2-* (implementation).
Refuses to run unless that decision commit is an ancestor of HEAD and DECISIONS.md is unmodified.

Per series (configs/w2_sample.yaml, all 89):
  E2a  20 admissible normal positions (seeded, disjoint within a length) x |R| in {0.25, 0.5, 1.0} w
       (round half up, min 1); D-E2a-1: >= 10 placed -> use them (positions_short if < 20);
       < 10 -> that length is dropped for the series.
  E2b  R_anom = [GT_start, GT_stop) + 20 admissible normal positions of length L = |GT|.
  For each region: resp_before = max score_patch(x, R) (once), then per operator (9):
       resp_after = max score_patch(op(x), R), AI = (resp_after - resp_before) / scale(x)  (D7-D9).
       |R| = w: s(j*)_after = score_patch value at j* = a + w//2 (D12).
SAR (D10): removal = max(0, -AI_anom); reversal = AI_anom > 0; den = median |AI_normal_L|;
       SAR = removal / den, SAR_abs = |AI_anom| / den; den < 1e-6 -> NaN + sar_den_zero;
       < 10 SAR positions -> sar_few_positions; < 5 -> excluded from SAR only.
STOP (rev1 §11): recon_test no-donor > 5% of regions; scale exclusions > 10%; any D12 prediction
       with > 1% mismatches. The outputs are written first, then exit 2.

Writes results/w2/ai.csv.gz, results/w2/sar.csv.gz (parquet engine not in the reference env,
D-E2-1), results/w2/e2_summary.json. The report is scripts/w2_e2_report.py.
"""
from __future__ import annotations

import json
import os
import subprocess
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

DECISION_COMMIT = "f29eddfc4e9d16326af80211050db874332f2198"
OUT = REPO_ROOT / "results" / "w2"
TOL_A, TOL_B = 1e-12, 1e-3          # D12 (approved)
SAR_DEN_MIN = 1e-6                  # D10


def git(*a) -> str:
    return subprocess.run(["git", "-C", str(REPO_ROOT), *a], capture_output=True, text=True).stdout.strip()


def check_decisions() -> dict:
    anc = subprocess.run(["git", "-C", str(REPO_ROOT), "merge-base", "--is-ancestor", DECISION_COMMIT, "HEAD"])
    if anc.returncode != 0:
        raise RuntimeError(f"decision commit {DECISION_COMMIT[:7]} is not an ancestor of HEAD (rev1 §3)")
    if git("status", "--porcelain", "--", "docs/DECISIONS.md"):
        raise RuntimeError("docs/DECISIONS.md has uncommitted changes")
    return {"decision_commit": DECISION_COMMIT, "decision_time": git("show", "-s", "--format=%cI", DECISION_COMMIT),
            "head": git("rev-parse", "HEAD")}


def regions_for(lay, e, cfg, rng):
    from src import w2
    g = cfg["e2_grid"]
    out, info = [], {}
    for f in g["lengths_w"]:
        L = w2.length_for(float(f), lay.w)
        placed = w2.draw_disjoint(rng, lay.admissible_starts(L), L, int(g["n_normal_positions"]))
        info[f"{f}w"] = {"L": L, "placed": len(placed), "max_disjoint": lay.max_disjoint(L)}
        if len(placed) >= 10:                                   # D-E2a-1
            out += [("normal", str(f), i, R) for i, R in enumerate(placed)]
    L = lay.stop0 - lay.start0
    placed = w2.draw_disjoint(rng, lay.admissible_starts(L), L, int(g["n_sar_positions"]))
    info["GT"] = {"L": L, "placed": len(placed), "max_disjoint": lay.max_disjoint(L)}
    out += [("sar_normal", "GT", i, R) for i, R in enumerate(placed)]
    out.append(("anomaly", "GT", 0, (lay.start0, lay.stop0)))
    return out, info


def run_series(e: dict, cfg: dict) -> dict:
    from src import w2
    from src.detectors.matrixprofile import MatrixProfileDetector
    from src.perturb.operators import META, Ctx, NoDonorError, apply
    man, g1 = w2.manifest(), w2.gate1_primary()
    num = int(e["num"])
    lay = w2.layout(num, man, g1)
    x = w2.load_checked(num, man)
    sc = w2.scale_of(num, lay)
    if abs(sc["scale"] - e["scale"]) > 1e-9 or sc["scale_ok"] != e["scale_ok"]:
        raise RuntimeError(f"{num}: scale differs from configs/w2_sample.yaml")
    g = cfg["e2_grid"]
    det = MatrixProfileDetector.fit(x[:lay.train_end], window=lay.w, mode="ab_join", constant_score="error")
    regions, info = regions_for(lay, e, cfg, np.random.default_rng([int(g["position_seed"]), num]))
    ctx = Ctx(train_end=lay.train_end, gt=(lay.start0, lay.stop0), w=lay.w)
    h, w = lay.w // 2, lay.w
    base = {"series_id": num, "content_group": e["content_group"], "domain": e["domain"],
            "detected": bool(e["detected"]), "n_const_train": int(e["n_const_train"]), "scale": sc["scale"],
            "w": w}
    rows = []
    t0 = time.perf_counter()
    for rid, (region, L_rel, pos_idx, (a, b)) in enumerate(regions):
        J, s0 = det.score_patch(x, (a, b))
        resp_before = float(s0.max())
        kj = (a + h) - int(J[0]) if b - a == w else None
        if kj is not None and not (0 <= kj < J.size and J[kj] == a + h):
            raise RuntimeError(f"{num}: j* = {a + h} not in J(R)")
        for k, op in enumerate(g["operators"]):
            rng = np.random.default_rng([int(g["operator_seed"]), num, rid, k])
            row = {**base, "op": op, "canonical_shape": META[op]["canonical_shape"], "region": region,
                   "L_rel": L_rel, "pos_idx": pos_idx, "a": a, "b": b, "L": b - a, "resp_before": resp_before}
            try:
                xp = apply(op, x, [(a, b)], rng, ctx)
            except NoDonorError:
                rows.append({**row, "resp_after": np.nan, "s_jstar_after": np.nan, "AI": np.nan, "no_donor": True})
                continue
            _, s1 = det.score_patch(xp, (a, b))
            ra = float(s1.max())
            rows.append({**row, "resp_after": ra, "s_jstar_after": float(s1[kj]) if kj is not None else np.nan,
                         "AI": (ra - resp_before) / sc["scale"] if sc["scale_ok"] else np.nan, "no_donor": False})
    return {"num": num, "rows": rows, "info": info, "scale_ok": sc["scale_ok"], "t": time.perf_counter() - t0}


def sar_table(ai: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (num, op), g in ai.groupby(["series_id", "op"], sort=True):
        an = g[g.region == "anomaly"]
        nr = g[(g.region == "sar_normal") & ~g.no_donor]
        n_pos = int((g.region == "sar_normal").sum())
        ai_anom = float(an.AI.iloc[0]) if len(an) and not an.no_donor.iloc[0] else np.nan
        den = float(np.median(np.abs(nr.AI))) if len(nr) else np.nan
        den_zero = bool(np.isfinite(den) and den < SAR_DEN_MIN)
        excluded = n_pos < 5
        ok = np.isfinite(ai_anom) and np.isfinite(den) and not den_zero and not excluded
        removal = max(0.0, -ai_anom) if np.isfinite(ai_anom) else np.nan
        out.append({"series_id": num, "content_group": g.content_group.iloc[0], "domain": g.domain.iloc[0],
                    "detected": bool(g.detected.iloc[0]), "op": op, "AI_anom": ai_anom, "removal": removal,
                    "reversal": bool(np.isfinite(ai_anom) and ai_anom > 0), "den": den,
                    "n_sar_positions": n_pos, "sar_few_positions": n_pos < 10, "sar_excluded": excluded,
                    "sar_den_zero": den_zero, "SAR": removal / den if ok else np.nan,
                    "SAR_abs": abs(ai_anom) / den if ok else np.nan})
    return pd.DataFrame(out)


def d12(ai: pd.DataFrame) -> dict:
    eq = ai[(ai.L == ai.w) & ~ai.no_donor]
    res = {}
    c = eq[(eq.canonical_shape == "constant") & (eq.n_const_train == 0)]
    root = np.sqrt(c.w.astype(float))
    res["D12-a(i)"] = c.assign(ok=(c.s_jstar_after - root).abs() <= TOL_A, observed=c.s_jstar_after, expected=root)
    res["D12-a(ii)"] = c.assign(ok=c.resp_after >= root - TOL_A, observed=c.resp_after, expected=root)
    c2 = eq[(eq.canonical_shape == "constant") & (eq.n_const_train > 0)]
    res["D12-a′"] = c2.assign(ok=c2.s_jstar_after.abs() <= TOL_A, observed=c2.s_jstar_after, expected=0.0)
    c3 = eq[eq.op == "recon_train"]
    res["D12-b"] = c3.assign(ok=c3.s_jstar_after <= TOL_B, observed=c3.s_jstar_after, expected=0.0)
    return res


def main() -> int:
    cfg = load_yaml("w2")
    os.environ["NUMBA_NUM_THREADS"] = str(cfg["execution"]["numba_threads_per_worker"])
    from src.runlog import env_header
    header = env_header()
    print(header, flush=True)
    dec = check_decisions()
    sample = yaml.safe_load((REPO_ROOT / "configs" / "w2_sample.yaml").read_text())
    series = sample["series"]
    if len(series) != 89:
        raise RuntimeError("expected the 89-group sample")
    ops_cfg = yaml.safe_load((REPO_ROOT / "configs" / "operators.yaml").read_text())["operators"]
    if list(cfg["e2_grid"]["operators"]) != list(ops_cfg):
        raise RuntimeError("operator list mismatch")
    import multiprocessing as mp
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=int(cfg["execution"]["workers"]), mp_context=mp.get_context("spawn")) as ex:
        res = list(ex.map(run_series, series, [cfg] * len(series)))
    elapsed = time.perf_counter() - t0
    ai = pd.DataFrame([r for x in res for r in x["rows"]])
    OUT.mkdir(parents=True, exist_ok=True)
    ai.to_csv(OUT / "ai.csv.gz", index=False)
    sar = sar_table(ai)
    sar.to_csv(OUT / "sar.csv.gz", index=False)
    checks = d12(ai)
    d12_summary = {k: {"n": int(len(v)), "n_ok": int(v.ok.sum()),
                       "mismatch_rate": float(1 - v.ok.mean()) if len(v) else None,
                       "stop": bool(len(v) and (1 - v.ok.mean()) > 0.01)} for k, v in checks.items()}
    positions = {int(x["num"]): x["info"] for x in res}
    nd = ai[ai.op == "recon_test"]
    nregions = ai[ai.op == "recon_test"].shape[0]
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "env_header": header,
        **dec, "elapsed_s": elapsed, "n_series": len(series), "n_rows": int(len(ai)),
        "series_seconds": {str(x["num"]): x["t"] for x in res}, "positions": positions,
        "scale_excluded": [int(x["num"]) for x in res if not x["scale_ok"]],
        "no_donor": {op: {"n": int(g.no_donor.sum()), "of": int(len(g))}
                     for op, g in ai[ai.op.isin(["recon_test", "recon_train"])].groupby("op")},
        "d12": d12_summary,
    }
    stops = []
    if nregions and nd.no_donor.sum() / nregions > 0.05:
        stops.append(f"recon_test no-donor {nd.no_donor.sum()}/{nregions} > 5%")
    if len(summary["scale_excluded"]) > 0.10 * len(series):
        stops.append("scale exclusions > 10%")
    stops += [f"{k}: {v['n'] - v['n_ok']}/{v['n']} mismatches > 1%" for k, v in d12_summary.items() if v["stop"]]
    summary["stops"] = stops
    (OUT / "e2_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"E2: {len(ai)} rows over {len(series)} series in {elapsed:.0f} s; D12 "
          + "; ".join(f"{k} {v['n_ok']}/{v['n']}" for k, v in d12_summary.items())
          + f"; recon_test no-donor {int(nd.no_donor.sum())}/{nregions}", flush=True)
    if stops:
        print("STOP (rev1 §11): " + " | ".join(stops))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
