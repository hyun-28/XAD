"""w2_tie_null.py -- W2 follow-up #8: the spread that arbitrary choices (tie order, Random draws) put on C2.

Decision D-F8-1 (commit 2948e9f, before any replicate curve). The main result (D-E3-5) is not changed; its
files are only read. Condition = E3 main: explanation operator recon_test, evaluation operators B1–B6, 4 AMs,
stored attributions (results/w2/e3_attr.csv.gz).

Replicate k (1…20):
    ties     `fa.tie_rank(K, default_rng([20260929, k, series, AMS.index(am)]))`, the same permutation for every
             operator (D-E3-5 principle). AMS = Random, FeatureAblation, KernelSHAP, MPNative (w2_e3.AMS).
    Random   `fa.random_attribution(K, default_rng([20260930, k, series]))` (uniform [0, 1)).
    curves   FeatureAblation, MPNative, Random recomputed with `fa.order_d35` (signed) + `fa.curve_from_order`
             and the E3 masked-f evaluator (`w2_e3.Series.f_masked`) with E3's operator rng keys (tag 0), exactly
             as scripts/w2_e3_curves.py; so only the order (and Random's attribution) changes.
             KernelSHAP has no ties in this sample (e3v2_ties.csv) -> its stored main curves are used.
Replicate 0: the main seeds (ties [20260924, series, AMS.index(am), 5], stored Random attribution). Its curves
must equal results/w2/faithfulness_v2.csv.gz bit for bit (DDS at the CSV's ~16-digit precision) and W must be
0.233 at 3 decimals, else STOP (exit 2).

`run`    computes replicates in order, one checkpoint per replicate (results/w2/tie_null/rep_KK.csv.gz: per series
         DDS and curves); existing checkpoints are skipped (resume). Stops starting new replicates after 8 h of
         accumulated run time (results/w2/tie_null/state.json).
`report` aggregates the finished replicates -> results/w2/tie_null.csv, reports/w2_tie_null.md.
No test statistic. No interpretation.
"""
from __future__ import annotations

import json
import multiprocessing as mp
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import REPO_ROOT, load_yaml  # noqa: E402

RES = REPO_ROOT / "results" / "w2"
CK = RES / "tie_null"
OUT_CSV = RES / "tie_null.csv"
REPORT = REPO_ROOT / "reports" / "w2_tie_null.md"
DECISION_COMMIT = "2948e9f"
N_REP = 20
TIE_SEED, RAND_SEED = 20260929, 20260930
MAIN_SEED, MAIN_TIE_TAG = 20260924, 5          # w2.yaml e3.seed, w2_e3_curves.TIE_TAG
BUDGET_S = 8 * 3600
RECOMPUTED = ("Random", "FeatureAblation", "MPNative")


def rep_series(e: dict, k: int, attrs: dict) -> list[dict]:
    from w2_e3 import AMS, EVAL_OPS, Series, op_index
    from src.metrics import faithfulness_ad as fa
    sr = Series(e, MAIN_SEED)
    rows = []
    for am in RECOMPUTED:
        ai = AMS.index(am)
        if k == 0:
            ties = fa.tie_rank(sr.K, np.random.default_rng([MAIN_SEED, sr.num, ai, MAIN_TIE_TAG]))
            a = attrs[am]
        else:
            ties = fa.tie_rank(sr.K, np.random.default_rng([TIE_SEED, k, sr.num, ai]))
            a = (fa.random_attribution(sr.K, np.random.default_rng([RAND_SEED, k, sr.num])) if am == "Random"
                 else attrs[am])
        if a.size != sr.K:
            raise RuntimeError(f"{sr.num} {am}: attribution size {a.size} != K {sr.K}")
        for op in EVAL_OPS:
            oi = op_index(op)
            fm = lambda m, t: sr.f_masked(m, op, (0, oi, ai, t))  # noqa: E731 (E3's keys, tag 0 = main)
            morf = fa.curve_from_order(sr.f0, fm, fa.order_d35(a, ties, "MoRF", "signed"))
            lerf = fa.curve_from_order(sr.f0, fm, fa.order_d35(a, ties, "LeRF", "signed"))
            rows.append({"k": k, "series_id": sr.num, "am": am, "eval_op": op, "dds": fa.dds(morf, lerf, sr.scale),
                         "n_tied": int(a.size - np.unique(a).size), "morf": json.dumps(morf.tolist()),
                         "lerf": json.dumps(lerf.tolist())})
    return rows


def load_attrs() -> dict:
    at = pd.read_csv(RES / "e3_attr.csv.gz")
    at = at[(at.condition == "main") & (at.expl_op == "recon_test")]
    return {(int(r.series_id), r.am): np.array(json.loads(r.attr)) for r in at.itertuples()}


def check_k0(rows: pd.DataFrame) -> list[str]:
    v2 = pd.read_csv(RES / "faithfulness_v2.csv.gz")
    v2 = v2[(v2.condition == "main") & v2.am.isin(RECOMPUTED)].set_index(["series_id", "am", "eval_op"])
    bad = []
    for r in rows.itertuples():
        o = v2.loc[(r.series_id, r.am, r.eval_op)]
        # curves: bit-identical (same JSON text); dds: the v2 CSV stores ~16 significant digits, so compare at
        # that precision (the dds is a deterministic function of the identical curves)
        if r.morf != o.morf or r.lerf != o.lerf or not np.isclose(r.dds, o.dds, rtol=1e-14, atol=1e-15):
            bad.append(f"{r.series_id} {r.am} {r.eval_op}")
    if len(rows) != len(v2):
        bad.append(f"row count {len(rows)} != {len(v2)}")
    return bad


def run() -> int:
    cfg = load_yaml("w2")
    os.environ["NUMBA_NUM_THREADS"] = str(cfg["execution"]["numba_threads_per_worker"])
    from src.runlog import env_header
    if subprocess.run(["git", "-C", str(REPO_ROOT), "merge-base", "--is-ancestor", DECISION_COMMIT, "HEAD"]).returncode:
        raise RuntimeError(f"D-F8-1 decision commit {DECISION_COMMIT} is not an ancestor of HEAD")
    CK.mkdir(parents=True, exist_ok=True)
    state_p = CK / "state.json"
    state = json.loads(state_p.read_text()) if state_p.exists() else {"elapsed_s": 0.0, "reps": {}}
    header = env_header()
    print(header, flush=True)
    sample = sorted(yaml.safe_load((REPO_ROOT / "configs" / "w2_sample.yaml").read_text())["series"], key=lambda e: -e["n"])
    attrs = load_attrs()
    per = [{am: attrs[(int(e["num"]), am)] for am in RECOMPUTED} for e in sample]
    with ProcessPoolExecutor(max_workers=int(cfg["execution"]["workers"]), mp_context=mp.get_context("spawn")) as ex:
        for k in range(0, N_REP + 1):
            p = CK / f"rep_{k:02d}.csv.gz"
            if p.exists():
                continue
            if state["elapsed_s"] >= BUDGET_S:
                print(f"budget of {BUDGET_S / 3600:.0f} h reached before replicate {k}; stopping", flush=True)
                break
            t0 = time.perf_counter()
            res = list(ex.map(rep_series, sample, [k] * len(sample), per))
            rows = pd.DataFrame([r for x in res for r in x])
            if k == 0:
                bad = check_k0(rows)
                if bad:
                    (CK / "STOP_k0.txt").write_text("\n".join(bad))
                    print(f"STOP: replicate 0 differs from faithfulness_v2 in {len(bad)} rows (see {CK / 'STOP_k0.txt'})")
                    return 2
            tmp = p.with_suffix(".tmp")
            rows.to_csv(tmp, index=False, compression="gzip")
            tmp.rename(p)
            dt = time.perf_counter() - t0
            state["elapsed_s"] += dt
            state["reps"][str(k)] = {"seconds": dt, "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                     "head": subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                                                            capture_output=True, text=True).stdout.strip(),
                                     "env_header": header}
            state_p.write_text(json.dumps(state, indent=2))
            print(f"replicate {k}/{N_REP} done in {dt:.0f} s (total {state['elapsed_s'] / 3600:.2f} h)", flush=True)
    return 0


# ---------------------------------------------------------------------------------------------- report

def per_rep(d: pd.DataFrame, rev: pd.DataFrame) -> dict:
    from scipy.stats import rankdata
    from src.metrics.faithfulness_simic import cmi, pes
    from w2_e3_report import AMS, EVAL_OPS, kendall_w, pair_rhos
    cells, diffs = [], []
    for op in EVAL_OPS:
        for am in AMS:
            v = d[(d.eval_op == op) & (d.am == am)].set_index("series_id").dds
            if v.size != 89 or v.isna().any():
                raise ValueError(f"{op}/{am}: {v.size} series or NaN")
            md, pe = float(v.mean()), pes(v.to_numpy())
            st = "0 포함" if md == 0 or pe == 0 else ("(+,+)" if md > 0 and pe > 0 else ("(−,−)" if md < 0 and pe < 0 else "충돌"))
            r = rev[rev.op == op].set_index("series_id").reversal.reindex(v.index)
            neg = v < 0
            p1, p0 = float(neg[r].mean()), float(neg[~r].mean())
            cells.append({"eval_op": op, "am": am, "mean_dds": md, "pes": pe, "cmi": cmi(md, pe), "sign_type": st,
                          "p_neg_rev": p1, "p_neg_norev": p0, "diff": p1 - p0})
    C = pd.DataFrame(cells)
    c = C.pivot(index="eval_op", columns="am", values="cmi").loc[EVAL_OPS, AMS]
    rk = c.apply(lambda r: pd.Series(rankdata(-r.to_numpy()), index=r.index), axis=1)
    C["rank"] = [rk.loc[r.eval_op, r.am] for r in C.itertuples()]
    return {"cells": C, "W": kendall_w(rk.to_numpy()), "rhos": pair_rhos(rk)}


def report() -> int:
    from w2_e2_report import f, md_table
    from w2_e3_report import AMS, EVAL_OPS
    state = json.loads((CK / "state.json").read_text())
    done = sorted(int(p.stem.split("_")[1].split(".")[0]) for p in CK.glob("rep_*.csv.gz"))
    if 0 not in done:
        raise RuntimeError("replicate 0 (reproduction) missing")
    v2 = pd.read_csv(RES / "faithfulness_v2.csv.gz")
    ks = v2[(v2.condition == "main") & (v2.am == "KernelSHAP")][["series_id", "am", "eval_op", "dds"]]
    ai = pd.read_csv(RES / "ai.csv.gz", dtype={"L_rel": str})
    rev = ai[ai.region == "anomaly"][["series_id", "op", "AI"]].assign(reversal=lambda d: d.AI > 0)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    raw, cell_rows, rep_rows, rho_rows = [], [], [], []
    R = {}
    for k in done:
        d = pd.read_csv(CK / f"rep_{k:02d}.csv.gz")[["series_id", "am", "eval_op", "dds"]]
        d = pd.concat([d, ks], ignore_index=True)
        R[k] = per_rep(d, rev)
        raw.append(d.assign(k=k, level="series"))
        cell_rows.append(R[k]["cells"].assign(k=k, level="cell"))
        rep_rows.append({"k": k, "level": "replicate", "W": R[k]["W"],
                         "n_negneg": int((R[k]["cells"].sign_type == "(−,−)").sum())})
        rho_rows += [{"k": k, "level": "pair", "op_a": a, "op_b": b, "rho": v} for (a, b), v in R[k]["rhos"].items()]
    k0_ok = f"{R[0]['W']:.3f}" == "0.233"
    reps = [k for k in done if k >= 1]
    CE, RP = pd.concat(cell_rows, ignore_index=True), pd.DataFrame(rep_rows)
    out = pd.concat([pd.concat(raw, ignore_index=True), CE, RP, pd.DataFrame(rho_rows)], ignore_index=True)
    out.to_csv(OUT_CSV, index=False, float_format="%.10g")

    L = []
    A = L.append
    A("# W2 후속 #8 — 임의 선택(동률 순서·Random 난수)이 만드는 C2 잡음의 폭\n")
    A(f"GENERATED by scripts/w2_tie_null.py report at {datetime.now(timezone.utc).isoformat(timespec='seconds')}, HEAD "
      f"`{head[:7]}`. Do not hand-edit. 결정: D-F8-1 (`{DECISION_COMMIT}`, 복제 계산 전). 주 결과 파일은 읽기만 했다. "
      "검정 통계량 없음. 해석 없음.\n")
    A(md_table(["항목", "내용"], [
        ["조건", "E3 주 조건: 설명용 recon_test, 평가용 B1–B6, 설명기 4종, 저장된 기여도 (`e3_attr.csv.gz`)"],
        ["복제 k ≥ 1", f"동률 순열 `default_rng([{TIE_SEED}, k, 시리즈, AMS.index(설명기)])` (AMS = {', '.join(AMS)}; "
                      f"연산자 간 같은 순열), Random 기여도 `uniform(0, 1, K)` with `default_rng([{RAND_SEED}, k, 시리즈])`"],
        ["곡선", "FeatureAblation, MPNative, Random 재계산 (`order_d35` signed, `curve_from_order`, E3의 연산자 난수 키 그대로) — "
                "연산자 B5·B6의 난수는 복제 간 고정. KernelSHAP은 동률이 없어(e3v2_ties.csv) 저장 곡선"],
        ["k = 0", f"주 결과 시드 ([{MAIN_SEED}, 시리즈, 설명기, {MAIN_TIE_TAG}], 저장된 Random 기여도)"],
        ["완료 복제", f"k = {', '.join(map(str, done))} (k ≥ 1: {len(reps)}개), 누적 실행 {state['elapsed_s'] / 3600:.2f} h"],
        ["C1 연결 차이", "#4 [2]와 같음: P(DDS < 0 \\| reversal) − P(DDS < 0 \\| 비reversal), reversal = E2 이상 구간 AI > 0"],
    ]))
    A("")
    A("## 1. k = 0 재현 확인\n")
    A(f"- k = 0의 FeatureAblation·MPNative·Random 곡선 {6 * 3 * 89}개 쌍이 `faithfulness_v2.csv.gz`와 비트 일치, DDS는 CSV 저장 자릿수(rtol 1e-14)에서 일치 "
      f"(실행 중 확인; 불일치면 실행이 멈춘다).\n- W = {R[0]['W']:.6f} → 표시 {R[0]['W']:.3f} (주 결과 0.233): **{'일치' if k0_ok else '불일치'}**.\n")
    if not reps:
        REPORT.write_text("\n".join(L), encoding="utf-8")
        return 0
    W = RP[RP.k >= 1].W.to_numpy()
    A(f"## 2. W의 분포 (k = 1…{max(reps)}, n = {len(reps)})\n")
    A(md_table(["중앙값", "최소", "최대", "2.5%", "97.5%", "k = 0 (주 결과)"],
               [[f(np.median(W)), f(W.min()), f(W.max()), f(np.percentile(W, 2.5)), f(np.percentile(W, 97.5)), f(R[0]["W"])]]))
    A("\n복제별 W: " + ", ".join(f"k{k}={R[k]['W']:.3f}" for k in reps) + "\n")
    nn = RP[RP.k >= 1].n_negneg.to_numpy()
    A("## 3. (−,−) 칸 수의 분포 (24칸 중)\n")
    A(md_table(["중앙값", "최소", "최대", "k = 0"], [[f(np.median(nn)), int(nn.min()), int(nn.max()),
                                                     int(RP[RP.k == 0].n_negneg.iloc[0])]]))
    vc = pd.Series(nn).value_counts().sort_index()
    A("\n값별 복제 수: " + ", ".join(f"{int(v)}칸: {int(c)}회" for v, c in vc.items()) + "\n")
    A(f"## 4. 설명기별 1위 횟수 (연산자 6 × 복제 {len(reps)} = {6 * len(reps)}번 중; 동률 1위는 평균 순위라 1.0이 아니면 세지 않음)\n")
    c1 = CE[(CE.k >= 1)]
    A(md_table(["설명기", "1위 횟수", *[f"{op}" for op in EVAL_OPS]],
               [[am, int(((c1.am == am) & (c1["rank"] == 1)).sum()),
                 *[int(((c1.am == am) & (c1.eval_op == op) & (c1["rank"] == 1)).sum()) for op in EVAL_OPS]] for am in AMS]))
    A(f"\n동률 1위(순위 < 1.5이지만 1.0이 아님) 사례 수: {int(((c1['rank'] > 1) & (c1['rank'] < 1.5)).sum())}.\n")
    A("## 5. #4 [2] 차이 P(DDS<0 | reversal) − P(DDS<0 | 비reversal) — 복제 분포와 D-F8-1 표기\n")
    A("D-F8-1: 차이가 **같은 연산자의 Random 복제 20회 분포의 최댓값**보다 크면 \"잡음을 넘는다\". D-F8-1은 비교할 설명기 "
      "값이 무엇인지(주 결과 한 값인지 복제 분포의 어느 값인지) 정하지 않았으므로, 아래 세 가지 읽기를 모두 표시한다 "
      "(새 결정 아님): (i) k = 0(주 결과) 값 > Random 최댓값, (ii) 복제 중앙값 > Random 최댓값, "
      "(iii) 복제 최솟값 > Random 최댓값(모든 복제가 넘음). KernelSHAP은 복제 간 값이 같다.\n")
    rows = []
    marks = []
    for op in EVAL_OPS:
        rnd = c1[(c1.eval_op == op) & (c1.am == "Random")]["diff"].to_numpy()
        rmax = rnd.max()
        rows.append([op, "Random", f"{f(rnd.min())} / {f(np.median(rnd))} / {f(rmax)}", f(CE[(CE.k == 0) & (CE.eval_op == op) & (CE.am == 'Random')]['diff'].iloc[0]), "—", "—", "—"])
        for am in ("FeatureAblation", "KernelSHAP", "MPNative"):
            v = c1[(c1.eval_op == op) & (c1.am == am)]["diff"].to_numpy()
            v0 = float(CE[(CE.k == 0) & (CE.eval_op == op) & (CE.am == am)]["diff"].iloc[0])
            m = (v0 > rmax, float(np.median(v)) > rmax, float(v.min()) > rmax)
            rows.append([op, am, f"{f(v.min())} / {f(np.median(v))} / {f(v.max())}", f(v0),
                         *["넘는다" if x else "" for x in m]])
            if any(m):
                marks.append((op, am, v0, float(np.median(v)), float(v.min()), rmax, m))
    A(md_table(["연산자", "설명기", "복제 최소 / 중앙값 / 최대", "k = 0", "(i) k = 0", "(ii) 중앙값", "(iii) 최솟값"], rows))
    A("\n### \"잡음을 넘는다\" 칸 (어느 읽기로든)\n")
    A(md_table(["연산자", "설명기", "k = 0", "복제 중앙값", "복제 최솟값", "Random 최댓값", "(i)", "(ii)", "(iii)"],
               [[op, am, f(a), f(b), f(c), f(rm), *["예" if x else "아니오" for x in m]] for op, am, a, b, c, rm, m in marks])
      if marks else "없음.")
    A("")
    A("## 6. 칸별 CMI와 순위의 복제 분포 (최소 / 중앙값 / 최대; k = 0)\n")
    A(md_table(["연산자", "설명기", "CMI", "CMI k = 0", "순위", "순위 k = 0", "(−,−) 복제 수"],
               [[op, am, " / ".join(f(x) for x in (s.cmi.min(), s.cmi.median(), s.cmi.max())),
                 f(float(CE[(CE.k == 0) & (CE.eval_op == op) & (CE.am == am)].cmi.iloc[0])),
                 " / ".join(f(x, 2) for x in (s["rank"].min(), s["rank"].median(), s["rank"].max())),
                 f(float(CE[(CE.k == 0) & (CE.eval_op == op) & (CE.am == am)]["rank"].iloc[0]), 2),
                 int((s.sign_type == "(−,−)").sum())]
                for op in EVAL_OPS for am in AMS for s in [c1[(c1.eval_op == op) & (c1.am == am)]]]))
    A("\n## 7. 연산자 쌍 Spearman ρ의 복제 분포 (최소 / 중앙값 / 최대)\n")
    RH = pd.DataFrame(rho_rows)
    A(md_table(["쌍", "k ≥ 1: 최소 / 중앙값 / 최대", "NaN 복제 수", "k = 0"],
               [[f"{a} – {b}", " / ".join(f(x) for x in (g.rho.min(), g.rho.median(), g.rho.max())),
                 int(g.rho.isna().sum()), f(float(RH[(RH.k == 0) & (RH.op_a == a) & (RH.op_b == b)].rho.iloc[0]))]
                for (a, b), g in RH[RH.k >= 1].groupby(["op_a", "op_b"], sort=False)]))
    A("")
    REPORT.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {REPORT} and {OUT_CSV}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("run", "report"):
        raise SystemExit("usage: w2_tie_null.py run|report")
    sys.exit(run() if sys.argv[1] == "run" else report())
