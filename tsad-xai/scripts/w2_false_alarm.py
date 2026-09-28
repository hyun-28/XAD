"""w2_false_alarm.py -- W2 follow-up #5: false-alarm rate on the rev1 E2a normal positions.

No new perturbation. Inputs: results/w2/ai.csv.gz (E2: per position resp_before / resp_after / AI), the gate-1
stored Z scores, and the unperturbed Z profile recomputed here (the only new computation).

[0] Threshold and alarm = the E5 implementation (D-E5-9, D-E5-35 item 2), i.e. the pilot code:
      threshold  scripts/w2_pilot.py:201-202 `p99_threshold`: np.percentile(padded unperturbed profile
                 [normal_test_mask(n, train_end, start0, stop0, buffer)], PERCENTILE), buffer = w
                 (src/gate1.py:54-65; PERCENTILE from src/gate1.py)
      alarm      scripts/w2_pilot.py:239-240: `max over affected windows > threshold` (strict), before and after
    Detector Z = `MPScorer("Z", m = w, reference = x[:train_end])` (src/detectors/mp.py), padded with
    `MPScorer.pad` (gate-1 convention). The padded profile must be bit-identical (np.array_equal) to the stored
    gate-1 score, else STOP (exit 2) before anything is aggregated.
    before = max of the unperturbed profile over the affected windows (mp.affected_range = every window
             meeting R), as the pilot's `b[aff].max()`.
    after  = resp_after of ai.csv.gz: the max over the same windows (rev1 J(R)) of rev1 `score_patch` on x′
             (E0: |score_patch − full| ≤ 1e-6). Not recomputed.
[1] Position: false alarm = before ≤ threshold and after > threshold. Positions with before > threshold are
    removed from numerator and denominator (D-E5-10 rule) and counted.
[2] Per (series, operator, |R|): rate = false alarms / remaining positions; cells with < 10 remaining positions
    are excluded and counted. Across series: mean and median; series bootstrap (B = 2000,
    np.random.default_rng(20260928) afresh per subset, shared indices), percentile 95 % CI. All 89 and
    flagged-removed (w2_e2_report.flags).
[3] AI of false-alarm vs other counted positions, per operator (median, IQR).
Writes results/w2/false_alarm.csv and reports/w2_false_alarm.md. No test statistic, no interpretation.
Env FA_CACHE=<path.pkl> caches the per-series computation (scratch use only).
"""
from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import REPO_ROOT, load_yaml  # noqa: E402

RES = REPO_ROOT / "results" / "w2"
OUT_CSV = RES / "false_alarm.csv"
REPORT = REPO_ROOT / "reports" / "w2_false_alarm.md"
OPS = ["B1_zero", "B2_global_mean", "B3_local_mean", "B4_linear_interp", "B5_gaussian", "B6_shuffle",
       "recon_test", "recon_train", "identity"]
LENS = ["0.25", "0.5", "1.0"]
B = 2000
SEED = 20260928
MIN_POS = 10


def series_job(num: int, regions: list[tuple[int, int]]) -> dict:
    from src import w2
    from src.detectors.mp import MPScorer, affected_range
    from w2_pilot import p99_threshold
    man, g1 = w2.manifest(), w2.gate1_primary()
    lay = w2.layout(num, man, g1)
    x = w2.load_checked(num, man)
    n, te, w = lay.n, lay.train_end, lay.w
    t0 = time.perf_counter()
    sc = MPScorer("Z", w, x[:te])
    prof = sc.profile(x)
    z = sc.pad(prof, n)
    stored = w2.gate1_score(num)
    equal = bool(z.shape == stored.shape and np.array_equal(z, stored))
    thr = p99_threshold(z, n, te, lay.start0, lay.stop0, w)       # buffer = w (D-E5-9)
    before = {}
    for a, b in regions:
        lo, hi = affected_range(a, b - a, w, n)
        before[(a, b)] = float(prof[lo:hi].max())
    return {"num": num, "equal": equal, "max_abs_diff": float(np.abs(z - stored).max()) if z.shape == stored.shape
            else float("inf"), "threshold": thr, "threshold_from_stored": p99_threshold(stored, n, te, lay.start0,
                                                                                      lay.stop0, w),
            "w": w, "n": n, "before": before, "t_s": time.perf_counter() - t0}


def compute(ai: pd.DataFrame) -> list[dict]:
    cache = os.environ.get("FA_CACHE")
    if cache and Path(cache).exists():
        return pickle.loads(Path(cache).read_bytes())
    cfg = load_yaml("w2")
    os.environ["NUMBA_NUM_THREADS"] = str(cfg["execution"]["numba_threads_per_worker"])
    reg = ai.drop_duplicates(["series_id", "a", "b"]).groupby("series_id")[["a", "b"]].apply(
        lambda d: list(map(tuple, d.to_numpy().tolist())))
    out = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=int(cfg["execution"]["workers"]), mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(series_job, int(num), r): num for num, r in reg.items()}
        for k, fu in enumerate(as_completed(futs), 1):
            out.append(fu.result())
            if k % 9 == 0 or k == len(futs):
                print(f"{k}/{len(futs)} series, {time.perf_counter() - t0:.0f} s", flush=True)
    if cache:
        Path(cache).write_bytes(pickle.dumps(out))
    return out


def boot_ci(M: np.ndarray, rng_idx: list[np.ndarray]) -> tuple[np.ndarray, ...]:
    ok = ~np.all(np.isnan(M), axis=0)
    mean_b = np.full((B, M.shape[1]), np.nan)
    med_b = np.full((B, M.shape[1]), np.nan)
    for i, ii in enumerate(rng_idx):
        Xb = M[ii][:, ok]
        if np.all(np.isnan(Xb), axis=0).any():
            raise ValueError("a bootstrap replicate has a cell with no series")
        mean_b[i, ok], med_b[i, ok] = np.nanmean(Xb, axis=0), np.nanmedian(Xb, axis=0)
    def q(a):
        lo, hi = np.full(M.shape[1], np.nan), np.full(M.shape[1], np.nan)
        lo[ok], hi[ok] = np.percentile(a[:, ok], [2.5, 97.5], axis=0)
        return lo, hi
    return q(mean_b), q(med_b)


def main() -> int:
    from src.runlog import env_header
    from w2_e2_report import f, flags, md_table
    ai = pd.read_csv(RES / "ai.csv.gz", dtype={"L_rel": str})
    sar = pd.read_csv(RES / "sar.csv.gz")
    import json
    summ = json.loads((RES / "e2_summary.json").read_text())
    nrm = ai[(ai.region == "normal") & ~ai.no_donor].copy()
    if nrm.duplicated(["series_id", "op", "a", "b"]).any():
        raise ValueError("duplicate normal positions")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()
    res = compute(nrm)
    per = pd.DataFrame([{k: v for k, v in r.items() if k != "before"} for r in res]).sort_values("num")
    if len(per) != 89:
        raise ValueError(f"{len(per)} series computed")
    if not per.equal.all():
        bad = per[~per.equal]
        REPORT.write_text("# W2 후속 #5 — STOP: 무교란 Z가 게이트 1 저장 점수와 비트 일치하지 않음\n\n"
                          + md_table(["series", "max |Δ|"], [[int(r.num), f(r.max_abs_diff, 4)] for r in bad.itertuples()])
                          + "\n", encoding="utf-8")
        print(f"STOP: {len(bad)} series not bit-identical; see {REPORT}")
        return 2
    if not (per.threshold == per.threshold_from_stored).all():
        raise ValueError("threshold differs between recomputed and stored scores despite bit equality")
    thr = per.set_index("num").threshold
    bef = {(r["num"], a, b): v for r in res for (a, b), v in r["before"].items()}
    nrm["thr"] = nrm.series_id.map(thr)
    nrm["before"] = [bef[(s, a, b)] for s, a, b in zip(nrm.series_id, nrm.a, nrm.b)]
    nrm["pre_alarm"] = nrm.before > nrm.thr
    nrm["after_alarm"] = nrm.resp_after > nrm.thr
    nrm["fa"] = ~nrm.pre_alarm & nrm.after_alarm
    # consistency of the two "before" values (full profile vs rev1 score_patch), reported
    d_before = (nrm.before - nrm.resp_before).abs()
    flip_before = int(((nrm.resp_before > nrm.thr) != nrm.pre_alarm).sum())
    near = int(((nrm.resp_after - nrm.thr).abs() <= 1e-6).sum())

    cell = nrm.groupby(["series_id", "op", "L_rel"]).agg(n_pos=("fa", "size"), pre=("pre_alarm", "sum"),
                                                          fa=("fa", "sum")).reset_index()
    cell["remaining"] = cell.n_pos - cell.pre
    cell["included"] = cell.remaining >= MIN_POS
    cell["rate"] = np.where(cell.included, cell.fa / cell.remaining.where(cell.remaining > 0, 1), np.nan)
    fl = flags(ai, sar, summ)
    flagged = set(fl[fl.flagged].series_id)
    if len(flagged) != 10:
        raise ValueError(f"{len(flagged)} flagged series, expected 10")
    ids_all = np.array(sorted(nrm.series_id.unique()))
    subsets = {"all": ids_all, "no_flagged": np.array([i for i in ids_all if i not in flagged])}
    cols = [(op, lr) for op in OPS for lr in LENS]
    R = cell.pivot_table(index="series_id", columns=["op", "L_rel"], values="rate", dropna=False)
    R = R.reindex(index=ids_all, columns=pd.MultiIndex.from_tuples(cols))
    agg_rows = []
    for sub, ids in subsets.items():
        M = R.loc[ids].to_numpy(dtype=float)
        rng = np.random.default_rng(SEED)
        idx = [rng.integers(0, ids.size, ids.size) for _ in range(B)]
        (mlo, mhi), (dlo, dhi) = boot_ci(M, idx)
        csub = cell[cell.series_id.isin(ids)]
        for j, (op, lr) in enumerate(cols):
            v = M[:, j]
            cc = csub[(csub.op == op) & (csub.L_rel == lr)]
            agg_rows.append({"part": "rate", "subset": sub, "op": op, "L_rel": lr, "n_series": int(np.isfinite(v).sum()),
                             "n_cells_excluded_lt10": int((~cc.included).sum()),
                             "positions": int(cc.n_pos.sum()), "pre_alarm_removed": int(cc.pre.sum()),
                             "false_alarms": int(cc.fa.sum()),
                             "mean": float(np.nanmean(v)) if np.isfinite(v).any() else np.nan,
                             "mean_ci_lo": mlo[j], "mean_ci_hi": mhi[j],
                             "median": float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan,
                             "median_ci_lo": dlo[j], "median_ci_hi": dhi[j], "B": B, "seed": SEED})
    A_ = pd.DataFrame(agg_rows)
    # [3] AI of counted positions (included cells, not pre-alarmed)
    cnt = nrm.merge(cell[["series_id", "op", "L_rel", "included"]], on=["series_id", "op", "L_rel"])
    cnt = cnt[cnt.included & ~cnt.pre_alarm]
    ai_rows = []
    for sub, ids in subsets.items():
        c = cnt[cnt.series_id.isin(ids)]
        for op in OPS:
            for lr in ["all", *LENS]:
                d = c[(c.op == op) & ((c.L_rel == lr) if lr != "all" else True)]
                r = {"part": "ai_by_fa", "subset": sub, "op": op, "L_rel": lr}
                for tag, s in (("fa", d[d.fa].AI), ("nofa", d[~d.fa].AI)):
                    r[f"{tag}_n"] = len(s)
                    r[f"{tag}_median"], r[f"{tag}_q25"], r[f"{tag}_q75"] = (
                        (float(s.median()), float(s.quantile(.25)), float(s.quantile(.75))) if len(s) else (np.nan,) * 3)
                ai_rows.append(r)
    AIt = pd.DataFrame(ai_rows)
    thr_rows = per.assign(part="threshold").rename(columns={"num": "series_id"})[
        ["part", "series_id", "n", "w", "threshold", "equal", "max_abs_diff", "t_s"]]
    pd.concat([thr_rows, A_, AIt], ignore_index=True).to_csv(OUT_CSV, index=False, float_format="%.10g")

    L = []
    A = L.append
    A("# W2 후속 #5 — rev1 E2 정상 구간 위치의 오경보율\n")
    A(f"GENERATED by scripts/w2_false_alarm.py at {datetime.now(timezone.utc).isoformat(timespec='seconds')}, HEAD "
      f"`{head[:7]}`. Do not hand-edit. 새 섭동 없음: 지운 뒤 값은 `results/w2/ai.csv.gz`(w2_e2.py run "
      f"{summ['generated_utc']})의 `resp_after`. 새로 계산한 것은 무교란 Z 프로파일뿐. 검정 통계량 없음. 해석 없음.\n")
    A(f"`{env_header()}`\n")
    A("## 0. 임계값과 경보 (E5와 같은 구현: D-E5-9, D-E5-35 2번)\n")
    A(md_table(["항목", "정의 (코드)"], [
        ["탐지기", "Z = `MPScorer(\"Z\", m = w, reference = x[:train_end])` (`src/detectors/mp.py`), rev1 탐지기와 같은 "
                "AB-join. `MPScorer.pad`로 게이트 1 규약 패딩"],
        ["임계값", "`scripts/w2_pilot.py:201-202` `p99_threshold`: `np.percentile(score[normal_test_mask(n, te, s0, s1, m)], "
                "PERCENTILE)` — score = 패딩한 무교란 Z, 마스크 = 테스트 구간 − [start0 − w, stop0 + w) "
                "(`src/gate1.py:54-65`), buffer = w, PERCENTILE = 99 (`src/gate1.py`)"],
        ["경보", "`scripts/w2_pilot.py:239-240`: 영향 창 점수의 최댓값 **>** 임계값 (지우기 전·후 각각)"],
        ["영향 창", "R과 겹치는 창 전부 = `mp.affected_range(a, r, w, n)` (창 시작 [a − w + 1, b − 1]) = rev1 J(R)"],
        ["지우기 전 값", "무교란 전체 프로파일의 영향 창 최댓값 (파일럿 `b[aff].max()`)"],
        ["지운 뒤 값", "`ai.csv.gz` `resp_after` = rev1 `score_patch`의 J(R) 최댓값 (E0: 전체 재계산과 차 ≤ 1e-6). 다시 계산하지 않음"],
        ["오경보", "지우기 전 ≤ 임계값 이고 지운 뒤 > 임계값. 지우기 전 > 임계값인 위치는 분자·분모에서 제외 (D-E5-10 규칙)"],
        ["집계", f"(시리즈, 연산자, \\|R\\|)마다 오경보 / 남은 위치; 남은 위치 < {MIN_POS}이면 그 칸 제외 → 시리즈 간 평균·중앙값; "
               f"시리즈 부트스트랩 B = {B}, 시드 {SEED} (부분집합마다 새로 시작, 인덱스 공유), 백분위 95% CI"],
        ["위치", "`ai.csv.gz` region == normal (rev1 E2a), no_donor 행 없음"],
    ]))
    A("")
    A("## 1. 비트 일치와 임계값\n")
    A(f"- 패딩한 무교란 Z가 게이트 1 저장 점수(`results/gate1/MatrixProfile/scores/*_train_prefix_ab_join_seed0.npy`)와 "
      f"**비트 일치: {int(per.equal.sum())}/{len(per)} 시리즈** (np.array_equal). 임계값은 저장 점수로 계산한 값과 같음.\n"
      f"- 무교란 Z 계산 시간: 합계 {per.t_s.sum():.0f} s (시리즈 최대 {per.t_s.max():.0f} s).\n"
      f"- 지우기 전 값 두 가지(전체 프로파일 대 rev1 `score_patch`의 `resp_before`)의 차: 최대 {f(float(d_before.max()), 3)}; "
      f"사전 경보 판정이 달라지는 위치 {flip_before}곳.\n"
      f"- \\|resp_after − 임계값\\| ≤ 1e-6인 위치 (E0 허용오차 안에서 판정이 경계에 있는 곳): {near}곳 (연산자 행 기준).\n")
    A("## 2. 오경보율 — 시리즈 간 평균 [95% CI], 중앙값 [95% CI]\n")
    for sub, title in (("all", "89 시리즈"), ("no_flagged", "플래그 10개 제외 (79 시리즈)")):
        t = A_[A_.subset == sub].set_index(["op", "L_rel"])
        A(f"### {title}\n")
        A(md_table(["연산자", "\\|R\\|", "시리즈 수", "평균 [95% CI]", "중앙값 [95% CI]", "위치", "사전 경보 제외",
                    "오경보 위치", "제외된 칸 (< 10)"],
                   [[op, f"{lr}w", int(r.n_series), f"{f(r['mean'])} [{f(r.mean_ci_lo)}, {f(r.mean_ci_hi)}]",
                     f"{f(r['median'])} [{f(r.median_ci_lo)}, {f(r.median_ci_hi)}]", int(r.positions),
                     int(r.pre_alarm_removed), int(r.false_alarms), int(r.n_cells_excluded_lt10)]
                    for (op, lr), r in t.iterrows()]))
        A("")
    A("## 3. 사전 경보로 빠진 위치\n")
    pre = nrm.drop_duplicates(["series_id", "a", "b"])
    A(f"사전 경보는 연산자와 무관하다 (지우기 전 값). 고유 위치 {len(pre)}곳 중 **{int(pre.pre_alarm.sum())}곳** "
      f"(시리즈 {int(pre[pre.pre_alarm].series_id.nunique())}개). 길이별:\n")
    A(md_table(["\\|R\\|", "고유 위치", "사전 경보", "비율", "시리즈 수 (사전 경보 ≥ 1)"],
               [[f"{lr}w", int((pre.L_rel == lr).sum()), int(pre[pre.L_rel == lr].pre_alarm.sum()),
                 f"{100 * pre[pre.L_rel == lr].pre_alarm.mean():.1f}%",
                 int(pre[(pre.L_rel == lr) & pre.pre_alarm].series_id.nunique())] for lr in LENS]))
    A("")
    A("## 4. 오경보 위치와 비오경보 위치의 AI (기술 통계; 집계에 들어간 위치)\n")
    for sub, title in (("all", "89 시리즈"), ("no_flagged", "플래그 제외")):
        t = AIt[(AIt.subset == sub) & (AIt.L_rel == "all")]
        A(f"### {title} — 길이 합산\n")
        A(md_table(["연산자", "오경보: n", "AI 중앙값 [IQR]", "비오경보: n", "AI 중앙값 [IQR]"],
                   [[r.op, int(r.fa_n), f"{f(r.fa_median)} [{f(r.fa_q25)}, {f(r.fa_q75)}]", int(r.nofa_n),
                     f"{f(r.nofa_median)} [{f(r.nofa_q25)}, {f(r.nofa_q75)}]"] for r in t.itertuples()]))
        A("")
    A("길이별 값은 `results/w2/false_alarm.csv` (part = ai_by_fa).\n")
    REPORT.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {REPORT} and {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
