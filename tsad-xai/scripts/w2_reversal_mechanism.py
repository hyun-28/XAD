"""w2_reversal_mechanism.py -- W2 follow-up #4 (EXPLORATORY, post hoc): where reversal comes from.

Aggregation of stored results only; no perturbation or curve is recomputed. The hypothesis was formed after
the results were seen, so every table is labelled 탐색적 (exploratory).

[1] Anomaly-region rows of results/w2/ai.csv.gz (region == "anomaly"; one per series × operator):
    |GT| = L (= stop0 − start0), w, resp_before, resp_after, AI, n_const_train. reversal = AI > 0 (the
    definition of sar.csv.gz `reversal`; checked equal here). Cells: |GT| ≥ w × resp_before < √w, for series
    with n_const_train == 0; series with n_const_train > 0 in separate rows (same split).
    Deterministic check (constant erasers B1–B3; canonical_shape == "constant" checked):
        |GT| ≥ w and n_const_train == 0  ->  resp_after ≥ √w − 1e-12.
    Every violating case is listed.
[2] E3 main result (results/w2/faithfulness_v2.csv.gz, condition main, random ties D-E3-5), eval ops B1–B6 ×
    4 AMs: per series, reversal under the same operator (E2 anomaly AI > 0) vs sign of that series' DDS
    (> 0 / < 0 / == 0). 2 × 3 counts and P(DDS < 0 | reversal), P(DDS < 0 | no reversal). No test statistic.
Writes results/w2/reversal_mechanism.csv and reports/w2_reversal_mechanism.md. No interpretation.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import REPO_ROOT  # noqa: E402
from w2_e2_report import f, md_table  # noqa: E402

RES = REPO_ROOT / "results" / "w2"
OUT_CSV = RES / "reversal_mechanism.csv"
REPORT = REPO_ROOT / "reports" / "w2_reversal_mechanism.md"
CONST = ["B1_zero", "B2_global_mean", "B3_local_mean"]
NONCONST = ["B4_linear_interp", "B5_gaussian", "B6_shuffle"]
AMS = ["Random", "FeatureAblation", "KernelSHAP", "MPNative"]
TOL = 1e-12
EXPL = "탐색적"


def load() -> pd.DataFrame:
    ai = pd.read_csv(RES / "ai.csv.gz", dtype={"L_rel": str})
    an = ai[ai.region == "anomaly"].copy()
    if an.duplicated(["series_id", "op"]).any() or an.no_donor.any():
        raise ValueError("anomaly rows: duplicate (series, op) or no-donor rows")
    if (an.groupby("series_id").resp_before.nunique() != 1).any():
        raise ValueError("resp_before differs across operators of a series")
    sar = pd.read_csv(RES / "sar.csv.gz")
    m = an.merge(sar[["series_id", "op", "reversal"]], on=["series_id", "op"], how="left", validate="1:1")
    an["reversal"] = an.AI > 0
    if not (m.reversal.to_numpy() == an.reversal.to_numpy()).all():
        raise ValueError("AI > 0 disagrees with sar.csv.gz reversal")
    if not (an[an.op.isin(CONST)].canonical_shape == "constant").all():
        raise ValueError("B1–B3 are not all canonical_shape constant")
    an["root_w"] = np.sqrt(an.w.astype(float))
    an["gt_ge_w"] = an.L >= an.w
    an["before_lt_root"] = an.resp_before < an.root_w
    an["const_train"] = an.n_const_train > 0
    return an


def four_cells(d: pd.DataFrame, part: str, ops_tag: str) -> list[dict]:
    rows = []
    for ct in (False, True):
        for ge in (True, False):
            for lt in (True, False):
                c = d[(d.const_train == ct) & (d.gt_ge_w == ge) & (d.before_lt_root == lt)]
                n, k = len(c), int(c.reversal.sum())
                rows.append({"part": part, "ops": ops_tag, "n_const_train_gt0": ct, "gt_ge_w": ge, "before_lt_root_w": lt,
                             "n": n, "n_series": int(c.series_id.nunique()), "reversal": k,
                             "reversal_rate": k / n if n else np.nan})
    return rows


def main() -> int:
    an = load()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()
    run = json.loads((RES / "e3v2_run.json").read_text())
    summ = json.loads((RES / "e2_summary.json").read_text())
    rows = []
    # [1]
    rows += four_cells(an[an.op.isin(CONST)], "1_cells", "B1–B3")
    for op in CONST + NONCONST:
        rows += four_cells(an[an.op == op], "1_cells", op)
    rows += four_cells(an[an.op.isin(NONCONST)], "1_cells", "B4–B6")
    chk = an[an.op.isin(CONST) & an.gt_ge_w & ~an.const_train].copy()
    chk["ok"] = chk.resp_after >= chk.root_w - TOL
    viol = chk[~chk.ok]
    ser = an.drop_duplicates("series_id")
    n_lt = int(ser.before_lt_root.sum())
    # [2]
    fz = pd.read_csv(RES / "faithfulness_v2.csv.gz")
    fz = fz[(fz.condition == "main") & fz.eval_op.isin(CONST + NONCONST)]
    j = fz.merge(an[["series_id", "op", "reversal"]].rename(columns={"op": "eval_op"}),
                 on=["series_id", "eval_op"], how="left", validate="m:1")
    if j.reversal.isna().any() or j.dds.isna().any():
        raise ValueError("missing reversal or DDS after the join")
    j["sign"] = np.select([j.dds > 0, j.dds < 0], ["+", "−"], "0")
    xt = []
    for op in CONST + NONCONST:
        for am in AMS:
            d = j[(j.eval_op == op) & (j.am == am)]
            if len(d) != 89:
                raise ValueError(f"{op}/{am}: {len(d)} series")
            r = {"part": "2_crosstab", "ops": op, "am": am}
            for rv, tag in ((True, "rev"), (False, "norev")):
                s = d[d.reversal == rv]
                for sg, st in (("+", "pos"), ("−", "neg"), ("0", "zero")):
                    r[f"{tag}_{st}"] = int((s.sign == sg).sum())
                r[f"{tag}_n"] = len(s)
                r[f"p_neg_{tag}"] = r[f"{tag}_neg"] / len(s) if len(s) else np.nan
            r["diff_p_neg"] = r["p_neg_rev"] - r["p_neg_norev"]
            xt.append(r)
    X = pd.DataFrame(xt)
    out = pd.concat([pd.DataFrame(rows), X,
                     viol.assign(part="1_violations")[["part", "series_id", "op", "L", "w", "resp_after", "root_w"]]],
                    ignore_index=True)
    out.to_csv(OUT_CSV, index=False, float_format="%.10g")

    L = []
    A = L.append
    A(f"# W2 후속 #4 — reversal의 원인 확인 ({EXPL} 분석, 사후)\n")
    A(f"GENERATED by scripts/w2_reversal_mechanism.py at {datetime.now(timezone.utc).isoformat(timespec='seconds')}, "
      f"HEAD `{head[:7]}`. Do not hand-edit. **{EXPL}(사후) 분석**: 결과를 본 뒤 세운 가설의 확인이다. 기존 파일 집계만 "
      f"(`results/w2/ai.csv.gz`, `sar.csv.gz` — w2_e2.py run {summ['generated_utc']}; `faithfulness_v2.csv.gz` — "
      f"w2_e3_curves.py run {run['generated_utc']}). 새 섭동·곡선 없음. 검정 통계량 없음. 해석 없음.\n")
    A("## 0. 정의\n")
    A(md_table(["항목", "정의"], [
        ["행", "`ai.csv.gz`의 region == anomaly 행 (시리즈 × 연산자당 1행). 89 시리즈"],
        ["\\|GT\\|, w", "이상 구간 길이 L (= stop0 − start0), 게이트 1 창 길이 w"],
        ["reversal", "이상 구간 AI > 0 (= `sar.csv.gz` `reversal`; 이 스크립트에서 일치 확인)"],
        ["resp_before < √w", "이상 구간의 원래 반응(창 최대 점수)이 √w보다 작음. resp_before는 연산자와 무관(확인)"],
        ["n_const_train > 0", f"학습 구간에 상수 창이 있는 시리즈 ({int(ser.const_train.sum())}개) — 별도 행"],
        ["검산", f"B1–B3 (canonical_shape = constant 확인), \\|GT\\| ≥ w, n_const_train = 0 → resp_after ≥ √w − {TOL:g}"],
        ["DDS 부호", "E3 주 결과(`faithfulness_v2.csv.gz` condition main, 무작위 동률)의 시리즈별 DDS: > 0 / < 0 / == 0"],
    ]))
    A("")
    T1 = pd.DataFrame(rows)

    def cell_table(tag):
        t = T1[T1.ops == tag]
        return md_table(["n_const_train", "\\|GT\\| ≥ w", "resp_before < √w", "(시리즈, 연산자) 수", "시리즈 수",
                         "reversal 수", "reversal 비율"],
                        [["> 0" if r.n_const_train_gt0 else "= 0", "예" if r.gt_ge_w else "아니오",
                          "예" if r.before_lt_root_w else "아니오", r.n, r.n_series, r.reversal,
                          f"{100 * r.reversal_rate:.1f}%" if r.n else "—"] for r in t.itertuples()])
    A(f"## 1. [{EXPL}] 네 칸 표 — 상수 지우개 B1–B3\n")
    A(f"### B1–B3 합산 ({EXPL}; 셈 단위 = (시리즈, 연산자))\n")
    A(cell_table("B1–B3"))
    for op in CONST:
        A(f"\n### {op} ({EXPL})\n")
        A(cell_table(op))
    A(f"\n## 2. [{EXPL}] 결정론적 검산 — B1–B3, \\|GT\\| ≥ w, n_const_train = 0\n")
    A(f"대상 (시리즈, 연산자) {len(chk)}건 (시리즈 {chk.series_id.nunique()}개), resp_after ≥ √w − {TOL:g} 성립 "
      f"{int(chk.ok.sum())}건, **어긋난 사례 {len(viol)}건**. 여유 (resp_after − √w)의 최솟값 "
      f"{f(float((chk.resp_after - chk.root_w).min()), 4) if len(chk) else '—'}.\n")
    if len(viol):
        A(md_table(["series", "operator", "\\|GT\\|", "w", "resp_after", "√w", "resp_after − √w"],
                   [[int(r.series_id), r.op, int(r.L), int(r.w), f(r.resp_after, 8), f(r.root_w, 8),
                     f(r.resp_after - r.root_w, 4)] for r in viol.itertuples()]))
        A("")
    A(f"## 3. [{EXPL}] 참고 — 비상수 지우개 B4–B6 (기술 통계)\n")
    A(f"### B4–B6 합산 ({EXPL})\n")
    A(cell_table("B4–B6"))
    for op in NONCONST:
        A(f"\n### {op} ({EXPL})\n")
        A(cell_table(op))
    A(f"\n## 4. [{EXPL}] resp_before < √w인 시리즈\n")
    A(f"89 시리즈 중 **{n_lt}개** (n_const_train = 0: {int((ser.before_lt_root & ~ser.const_train).sum())}, "
      f"> 0: {int((ser.before_lt_root & ser.const_train).sum())}). 그중 \\|GT\\| ≥ w: "
      f"{int((ser.before_lt_root & ser.gt_ge_w).sum())}개.\n")
    A(f"## 5. [{EXPL}] reversal과 DDS 부호 (E3 주 결과, 평가용 B1–B6 × 설명기 4종)\n")
    A("rev = 같은 연산자로 GT를 지웠을 때 reversal 난 시리즈, norev = 나지 않은 시리즈. 칸 = 시리즈 수. "
      "P(−) = 그 행에서 DDS < 0 비율. 차이 = P(− | rev) − P(− | norev).\n")
    A(md_table(["평가용", "설명기", "rev: + / − / 0 (n)", "norev: + / − / 0 (n)", "P(− \\| rev)", "P(− \\| norev)", "차이"],
               [[r.ops, r.am, f"{r.rev_pos} / {r.rev_neg} / {r.rev_zero} ({r.rev_n})",
                 f"{r.norev_pos} / {r.norev_neg} / {r.norev_zero} ({r.norev_n})",
                 f(r.p_neg_rev), f(r.p_neg_norev), f(r.diff_p_neg)] for r in X.itertuples()]))
    A(f"\n### 차이의 절댓값 상위 5 ({EXPL})\n")
    top = X.reindex(X.diff_p_neg.abs().sort_values(ascending=False).index).head(5)
    A(md_table(["평가용", "설명기", "P(− \\| rev)", "P(− \\| norev)", "차이"],
               [[r.ops, r.am, f(r.p_neg_rev), f(r.p_neg_norev), f(r.diff_p_neg)] for r in top.itertuples()]))
    A(f"\n### Random (대조군, {EXPL})\n")
    rd = X[X.am == "Random"]
    A(md_table(["평가용", "P(− \\| rev)", "P(− \\| norev)", "차이"],
               [[r.ops, f(r.p_neg_rev), f(r.p_neg_norev), f(r.diff_p_neg)] for r in rd.itertuples()]))
    A("")
    REPORT.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {REPORT} and {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
