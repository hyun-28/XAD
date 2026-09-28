"""w2_detected_split.py -- W2 follow-up #7 (EXPLORATORY, post hoc): detected vs undetected series.

Aggregation of stored files only; nothing is perturbed or re-scored. Every table is labelled 탐색적(사후).
Group = `detected` of results/w2/ai.csv.gz (gate-1 detection; 65 / 24 series).

[1] scale(x), q99(s_normal), median(s_normal) per group: `src.w2.scale_of` (D8) on the stored gate-1 score;
    scale is checked equal to configs/w2_sample.yaml.
[2] Per operator (B1–B6, recon_test) × |R| (0.25, 0.5, 1.0 w), per group:
    a. normal-region AI: per-series median -> median across series (w2_e2_report.per_series_median; W2_REPORT §5.1)
    b. false-alarm rate: the #5 definition (scripts/w2_false_alarm.py): threshold `w2_pilot.p99_threshold` on the
       stored gate-1 score (bit-identical to the recomputed unperturbed Z, #5: 89/89), before = max of the profile
       over the affected windows (= stored score at window start + w//2, `_pad_like_tsb_ad`), after =
       `resp_after`; pre-alarmed positions removed; cells with < 10 remaining positions excluded; mean across series.
    Bootstrap: series resampled with replacement WITHIN each group (stratified), B = 2000,
    np.random.default_rng(20260928), percentile 95 % CI, for each group's value and for detected − undetected.
    The ratio detected / undetected of a is given as a point value only.
[3] Reproduction (STOP on mismatch): a pooled over both groups vs docs/W2_REPORT.md §5.1 at the printed
    precision; b pooled vs results/w2/false_alarm.csv (subset all; rtol 1e-9, the CSV keeps 10 significant digits).
Writes results/w2/detected_split.csv and reports/w2_detected_split.md. No test statistic, no interpretation.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import REPO_ROOT  # noqa: E402

RES = REPO_ROOT / "results" / "w2"
OUT_CSV = RES / "detected_split.csv"
REPORT = REPO_ROOT / "reports" / "w2_detected_split.md"
OPS = ["B1_zero", "B2_global_mean", "B3_local_mean", "B4_linear_interp", "B5_gaussian", "B6_shuffle", "recon_test"]
LENS = ["0.25", "0.5", "1.0"]
B = 2000
SEED = 20260928
MIN_POS = 10
TAG = "탐색적(사후)"


def main() -> int:
    from src import w2
    from w2_c1_ci import matches, parse_w2_report
    from w2_e2_report import f, md_table, per_series_median
    from w2_pilot import p99_threshold
    ai = pd.read_csv(RES / "ai.csv.gz", dtype={"L_rel": str})
    summ = json.loads((RES / "e2_summary.json").read_text())
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()
    det = ai.drop_duplicates("series_id").set_index("series_id").detected.astype(bool)
    ids = np.array(sorted(det.index))
    groups = {"detected": ids[det.loc[ids].to_numpy()], "undetected": ids[~det.loc[ids].to_numpy()]}
    if (groups["detected"].size, groups["undetected"].size) != (65, 24):
        raise ValueError(f"group sizes {groups['detected'].size}/{groups['undetected'].size}, expected 65/24")

    # ---- per-series quantities
    man, g1 = w2.manifest(), w2.gate1_primary()
    samp = {s["num"]: s for s in yaml.safe_load((REPO_ROOT / "configs" / "w2_sample.yaml").read_text())["series"]}
    fa_csv = pd.read_csv(RES / "false_alarm.csv")
    thr_csv = fa_csv[fa_csv.part == "threshold"].set_index("series_id").threshold
    nrm = ai[(ai.region == "normal") & ~ai.no_donor].copy()
    sc_rows, thr, before = [], {}, {}
    for num in ids:
        lay = w2.layout(int(num), man, g1)
        sc = w2.scale_of(int(num), lay)
        if sc["scale"] != samp[int(num)]["scale"] and not np.isclose(sc["scale"], samp[int(num)]["scale"], rtol=1e-11):
            raise ValueError(f"{num}: scale {sc['scale']} != w2_sample.yaml {samp[int(num)]['scale']}")
        sc_rows.append({"series_id": int(num), "detected": bool(det[num]), **{k: sc[k] for k in ("scale", "q99", "median")}})
        s = w2.gate1_score(int(num))
        t = p99_threshold(s, lay.n, lay.train_end, lay.start0, lay.stop0, lay.w)
        if t != thr_csv[num] and not np.isclose(t, thr_csv[num], rtol=1e-9):
            raise ValueError(f"{num}: threshold {t} != false_alarm.csv {thr_csv[num]}")
        thr[num] = t
        h, w, n = lay.w // 2, lay.w, lay.n
        for a, b in nrm[nrm.series_id == num][["a", "b"]].drop_duplicates().itertuples(index=False):
            lo, hi = max(0, a - w + 1), min(b - 1, n - w) + 1          # mp.affected_range
            before[(num, a, b)] = float(s[lo + h:hi + h].max())
    SC = pd.DataFrame(sc_rows)
    nrm["thr"] = nrm.series_id.map(thr)
    nrm["pre"] = [before[(s_, a, b)] > t for s_, a, b, t in zip(nrm.series_id, nrm.a, nrm.b, nrm.thr)]
    nrm["fa"] = ~nrm.pre & (nrm.resp_after > nrm.thr)
    cell = nrm.groupby(["series_id", "op", "L_rel"]).agg(n=("fa", "size"), pre=("pre", "sum"), fa=("fa", "sum")).reset_index()
    cell["rem"] = cell.n - cell.pre
    cell["rate"] = np.where(cell.rem >= MIN_POS, cell.fa / cell.rem.clip(lower=1), np.nan)
    cols = pd.MultiIndex.from_tuples([(op, lr) for op in OPS for lr in LENS])
    RATE = cell.pivot_table(index="series_id", columns=["op", "L_rel"], values="rate", dropna=False).reindex(index=ids, columns=cols)
    AIM = per_series_median(nrm, by=("op", "L_rel")).pivot_table(index="series_id", columns=["op", "L_rel"],
                                                                 values="AI").reindex(index=ids, columns=cols)

    # ---- [3] reproduction
    pr = parse_w2_report()
    rep, bad = [], 0
    for op in OPS:
        for lr in LENS:
            v = float(np.nanmedian(AIM[(op, lr)]))
            ok = matches(pr.loc[op, lr], v)
            bad += not ok
            rep.append(["a", op, f"{lr}w", pr.loc[op, lr], f(v, 6), "일치" if ok else "**불일치**"])
    fr = fa_csv[(fa_csv.part == "rate") & (fa_csv.subset == "all")].set_index(["op", "L_rel"])
    fr.index = fr.index.set_levels([fr.index.levels[0], [str(x) for x in fr.index.levels[1]]])
    for op in OPS:
        for lr in LENS:
            v = float(np.nanmean(RATE[(op, lr)]))
            ref = fr.loc[(op, lr)]
            ok = bool(np.isclose(v, ref["mean"], rtol=1e-9, atol=0) and int(np.isfinite(RATE[(op, lr)]).sum()) == ref.n_series)
            bad += not ok
            rep.append(["b", op, f"{lr}w", f(float(ref["mean"]), 6), f(v, 6), "일치" if ok else "**불일치**"])
    if bad:
        REPORT.write_text(f"# W2 후속 #7 — STOP: 재현 불일치 ({TAG})\n\n"
                          + md_table(["대상", "연산자", "\\|R\\|", "기준", "재계산", "판정"], rep) + "\n", encoding="utf-8")
        print(f"STOP: {bad} mismatches; see {REPORT}")
        return 2

    # ---- [2] stratified bootstrap
    rng = np.random.default_rng(SEED)
    nd, nu = groups["detected"].size, groups["undetected"].size
    idx = [(rng.integers(0, nd, nd), rng.integers(0, nu, nu)) for _ in range(B)]
    out = []
    for tgt, M, stat in (("ai_median", AIM, np.nanmedian), ("fa_mean", RATE, np.nanmean)):
        Xd, Xu = M.loc[groups["detected"]].to_numpy(float), M.loc[groups["undetected"]].to_numpy(float)
        bd, bu = np.empty((B, Xd.shape[1])), np.empty((B, Xu.shape[1]))
        for i, (id_, iu) in enumerate(idx):
            Yd, Yu = Xd[id_], Xu[iu]
            if np.all(np.isnan(Yd), axis=0).any() or np.all(np.isnan(Yu), axis=0).any():
                raise ValueError(f"{tgt}: a replicate has a cell with no series")
            bd[i], bu[i] = stat(Yd, axis=0), stat(Yu, axis=0)
        pd_, pu = stat(Xd, axis=0), stat(Xu, axis=0)
        q = lambda a: np.percentile(a, [2.5, 97.5], axis=0)  # noqa: E731
        (dlo, dhi), (ulo, uhi), (xlo, xhi) = q(bd), q(bu), q(bd - bu)
        for j, (op, lr) in enumerate(M.columns):
            out.append({"part": tgt, "op": op, "L_rel": lr,
                        "det_n": int(np.isfinite(Xd[:, j]).sum()), "det": pd_[j], "det_lo": dlo[j], "det_hi": dhi[j],
                        "und_n": int(np.isfinite(Xu[:, j]).sum()), "und": pu[j], "und_lo": ulo[j], "und_hi": uhi[j],
                        "diff": pd_[j] - pu[j], "diff_lo": xlo[j], "diff_hi": xhi[j],
                        "diff_ci_contains_0": bool(xlo[j] <= 0 <= xhi[j]),
                        "ratio": pd_[j] / pu[j] if tgt == "ai_median" and pu[j] != 0 else np.nan, "B": B, "seed": SEED})
    T = pd.DataFrame(out)
    scale_rows = []
    for k in ("scale", "q99", "median"):
        for g in ("detected", "undetected"):
            v = SC[SC.detected == (g == "detected")][k]
            scale_rows.append({"part": f"dist_{k}", "group": g, "n": len(v), "median": v.median(), "q25": v.quantile(.25),
                               "q75": v.quantile(.75), "min": v.min(), "max": v.max()})
    S = pd.DataFrame(scale_rows)
    pd.concat([S, T], ignore_index=True).to_csv(OUT_CSV, index=False, float_format="%.10g")

    L = []
    A = L.append
    A(f"# W2 후속 #7 — 탐지 성공·실패 집단의 차이 ({TAG})\n")
    A(f"GENERATED by scripts/w2_detected_split.py at {datetime.now(timezone.utc).isoformat(timespec='seconds')}, HEAD "
      f"`{head[:7]}`. Do not hand-edit. **{TAG} 분석**: 결과를 본 뒤 세운 질문이다. 기존 파일 집계만 "
      f"(`results/w2/ai.csv.gz` — w2_e2.py run {summ['generated_utc']}; 게이트 1 저장 점수; `results/w2/false_alarm.csv`). "
      "섭동·점수 재계산 없음. 검정 통계량 없음. 해석 없음.\n")
    A(f"## 0. 정의 ({TAG})\n")
    A(md_table(["항목", "정의"], [
        ["집단", "`ai.csv.gz` `detected` (게이트 1 탐지 여부): detected 65, undetected 24"],
        ["scale, q99, median", "`src.w2.scale_of` (D8): 게이트 1 저장 점수의 s_normal에서 q99, median, scale = q99 − median "
                               "(`configs/w2_sample.yaml`의 scale과 일치 확인)"],
        ["a. AI", "정상 구간(region normal, no_donor 없음) → 시리즈 안 중앙값 → 집단 안 시리즈 간 중앙값 (W2_REPORT §5.1과 같음)"],
        ["b. 오경보율", "#5 정의 (`scripts/w2_false_alarm.py`): 임계값 `w2_pilot.p99_threshold`(buffer = w, 저장 점수 = 무교란 Z와 "
                      "비트 일치, #5), 지우기 전 = 영향 창의 무교란 점수 최댓값, 지운 뒤 = `resp_after`, 사전 경보 위치 제외, "
                      f"남은 위치 < {MIN_POS}인 칸 제외 → 집단 안 시리즈 간 평균. 임계값은 `false_alarm.csv`와 일치 확인"],
        ["부트스트랩", f"집단 안에서 시리즈 복원 추출(층화), B = {B}, `np.random.default_rng({SEED})`, 백분위 95% CI; "
                    "차이 CI는 같은 재추출에서 (detected − undetected)"],
        ["비", "a의 detected / undetected, 점추정만"],
    ]))
    A("")
    A(f"## 1. 재현 확인 ({TAG})\n")
    A(f"두 집단을 합친 값 {len(rep)}칸 모두 일치 (a: W2_REPORT §5.1 표시 자릿수, b: `false_alarm.csv` rtol 1e-9와 시리즈 수).\n")
    A(md_table(["대상", "연산자", "\\|R\\|", "기준", "재계산", "판정"], rep))
    A("")
    A(f"## 2. scale 분포 ({TAG})\n")
    A(md_table(["값", "집단", "n", "중앙값", "IQR [q25, q75]", "최솟값", "최댓값"],
               [[r.part.replace("dist_", ""), r.group, r.n, f(r["median"]), f"{f(r.q75 - r.q25)} [{f(r.q25)}, {f(r.q75)}]",
                 f(r["min"]), f(r["max"])] for _, r in S.iterrows()]))
    A("")
    ci = lambda v, lo, hi: f"{f(v)} [{f(lo)}, {f(hi)}]"  # noqa: E731
    A(f"## 3. AI 대 오경보율 ({TAG})\n")
    for lr in LENS:
        A(f"### \\|R\\| = {lr}w ({TAG})\n")
        a_, b_ = T[(T.part == "ai_median") & (T.L_rel == lr)].set_index("op"), T[(T.part == "fa_mean") & (T.L_rel == lr)].set_index("op")
        A(md_table(["연산자", "AI detected", "AI undetected", "AI 차 [CI]", "AI 비", "오경보율 detected", "오경보율 undetected",
                    "오경보율 차 [CI]"],
                   [[op, ci(a_.loc[op].det, a_.loc[op].det_lo, a_.loc[op].det_hi),
                     ci(a_.loc[op].und, a_.loc[op].und_lo, a_.loc[op].und_hi),
                     ci(a_.loc[op]["diff"], a_.loc[op].diff_lo, a_.loc[op].diff_hi), f(a_.loc[op].ratio),
                     ci(b_.loc[op].det, b_.loc[op].det_lo, b_.loc[op].det_hi),
                     ci(b_.loc[op].und, b_.loc[op].und_lo, b_.loc[op].und_hi),
                     ci(b_.loc[op]["diff"], b_.loc[op].diff_lo, b_.loc[op].diff_hi)] for op in OPS]))
        n_ = T[(T.L_rel == lr) & (T.part == "fa_mean")]
        A(f"\n시리즈 수 (오경보율): detected {int(n_.det_n.min())}–{int(n_.det_n.max())}, "
          f"undetected {int(n_.und_n.min())}–{int(n_.und_n.max())}.\n")
    A(f"## 4. 목록 ({TAG})\n")
    for tgt, name in (("fa_mean", "오경보율"), ("ai_median", "AI")):
        z = T[(T.part == tgt) & T.diff_ci_contains_0]
        A(f"**{name} 차이(detected − undetected)의 CI가 0을 포함하는 칸:** "
          + ("; ".join(f"{r.op} {r.L_rel}w" for r in z.itertuples()) or "없음") + "\n")
    REPORT.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {REPORT} and {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
