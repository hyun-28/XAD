"""w2_c2_signs.py -- W2 follow-up #1: DDS / PES sign table of the C2 cells (aggregation of stored results only).

Reads the stored per-series DDS (already divided by the D8 scale, `faithfulness_ad.dds`) and recomputes nothing
upstream of them: no curve, no attribution. Per cell (evaluation operator, AM):
    mean DDS, PES, f = share DDS > 0, u = share DDS < 0, share DDS == 0, per-series median and IQR,
    CMI = `faithfulness_simic.cmi(mean DDS, PES)` (the implemented function the E3 reports use),
    sign type of (mean DDS, PES): (+,+) / (−,−) / conflict (opposite, both non-zero) / zero (either exactly 0),
    current rank (rank 1 = highest CMI within the operator; ties -> average rank, D-E3-3).

Tables (each separate):
    main      faithfulness_v2.csv.gz condition main  (random ties, D-E3-5; expl = recon_test; B1–B6 × 4 AMs)
    a         faithfulness_v2.csv.gz condition separate  (recon_train evaluation; one row)
    b         faithfulness_v2.csv.gz condition self  (expl = eval) for FeatureAblation, KernelSHAP
    c         faithfulness_v2.csv.gz condition abs   (|R| ordering) for FeatureAblation, KernelSHAP
    d         faithfulness.csv.gz condition main  (index-tie version, appendix A)
In b and c, Random and MPNative come from the main condition, as `w2_e3v2_report.rank_with` does; their
cells are identical to main and marked source = main.

Post-hoc exploratory (only for tables that contain a (−,−) cell): those cells' CMI set to 0, and to −|CMI|;
ranks per operator and Kendall's W (`w2_e3_report.kendall_w`, same as the E3 reports) recomputed, next to
the original. Does not change the main result.

Writes results/w2/c2_signs.csv and reports/w2_c2_signs.md. No interpretation.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import REPO_ROOT  # noqa: E402
from src.metrics.faithfulness_simic import cmi, pes  # noqa: E402
from w2_e3_report import AMS, EVAL_OPS, f, kendall_w, md_table  # noqa: E402

RES = REPO_ROOT / "results" / "w2"
OUT_CSV = RES / "c2_signs.csv"
REPORT = REPO_ROOT / "reports" / "w2_c2_signs.md"
SENS_AMS = ("FeatureAblation", "KernelSHAP")
TABLES = {
    "main": "주 결과 — 무작위 동률(D-E3-5), 설명용 recon_test, 평가용 B1–B6",
    "a": "a. recon_train 평가 (별도 보고분)",
    "b": "b. 자기 일치 민감도 (설명용 = 평가용; FeatureAblation·KernelSHAP)",
    "c": "c. |R| 정렬 민감도 (FeatureAblation·KernelSHAP)",
    "d": "d. 인덱스 동률 버전 (부록 A, 규칙 확정 전 버전)",
}


def sgn(v: float) -> str:
    return "+" if v > 0 else ("−" if v < 0 else "0")


def sign_type(d: float, p: float) -> str:
    if d == 0 or p == 0:
        return "0 포함"
    if d > 0 and p > 0:
        return "(+,+)"
    if d < 0 and p < 0:
        return "(−,−)"
    return "충돌"


def cell(v: np.ndarray) -> dict:
    if v.size == 0 or not np.isfinite(v).all():
        raise ValueError("empty or non-finite DDS column")
    md, pe = float(v.mean()), pes(v)
    q25, q50, q75 = (float(np.percentile(v, q)) for q in (25, 50, 75))
    return {"n": int(v.size), "mean_dds": md, "pes": pe, "f": float(np.mean(v > 0)), "u": float(np.mean(v < 0)),
            "zero": float(np.mean(v == 0)), "median": q50, "q25": q25, "q75": q75, "iqr": q75 - q25,
            "cmi": cmi(md, pe), "sign_dds": sgn(md), "sign_pes": sgn(pe), "sign_type": sign_type(md, pe)}


def col(df: pd.DataFrame, cond: str, op: str, am: str) -> np.ndarray:
    s = df[(df.condition == cond) & (df.eval_op == op) & (df.am == am)]
    if s.series_id.duplicated().any():
        raise ValueError(f"duplicate series in {cond}/{op}/{am}")
    return s.dds.dropna().to_numpy(dtype=np.float64)


def build(v2: pd.DataFrame, v1: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for op in EVAL_OPS:
        for am in AMS:
            rows.append({"table": "main", "eval_op": op, "am": am, "source": "main", **cell(col(v2, "main", op, am))})
            rows.append({"table": "d", "eval_op": op, "am": am, "source": "v1 main", **cell(col(v1, "main", op, am))})
            for t, cond in (("b", "self"), ("c", "abs")):
                src = cond if am in SENS_AMS else "main"
                rows.append({"table": t, "eval_op": op, "am": am, "source": src, **cell(col(v2, src, op, am))})
    for am in AMS:
        rows.append({"table": "a", "eval_op": "recon_train", "am": am, "source": "separate",
                     **cell(col(v2, "separate", "recon_train", am))})
    out = pd.DataFrame(rows)
    for t in TABLES:
        m = out.table == t
        # pandas average rank of descending CMI == rankdata(-cmi) (w2_e3_report.ranks)
        out.loc[m, "rank"] = out[m].groupby("eval_op").cmi.rank(ascending=False, method="average")
    return out


def rank_w(tab: pd.DataFrame, cmi_col: str) -> tuple[pd.DataFrame, float]:
    c = tab.pivot(index="eval_op", columns="am", values=cmi_col)[AMS]
    rk = c.apply(lambda r: pd.Series(rankdata(-r.to_numpy()), index=r.index), axis=1)
    W = kendall_w(rk.to_numpy()) if rk.shape[0] > 1 else float("nan")
    return rk, W


def main() -> int:
    v2 = pd.read_csv(RES / "faithfulness_v2.csv.gz")
    v1 = pd.read_csv(RES / "faithfulness.csv.gz")
    for name, d in (("v2", v2), ("v1", v1)):
        if d.dds.isna().any():
            raise ValueError(f"{name}: NaN DDS present; the table would silently drop series")
    run = json.loads((RES / "e3v2_run.json").read_text())
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()
    T = build(v2, v1)
    T.to_csv(OUT_CSV, index=False, float_format="%.10g")

    L = []
    A = L.append
    A("# W2 후속 #1 — C2 칸의 DDS·PES 부호표\n")
    A(f"GENERATED by scripts/w2_c2_signs.py at {datetime.now(timezone.utc).isoformat(timespec='seconds')}, HEAD "
      f"`{head[:7]}`. Do not hand-edit. 기존 파일 집계만: `results/w2/faithfulness_v2.csv.gz` "
      f"(w2_e3_curves.py run {run['generated_utc']}), `results/w2/faithfulness.csv.gz` (인덱스 동률 버전). "
      "곡선·기여도는 다시 계산하지 않았다. 해석 없음.\n")
    A(f"numpy {np.__version__}, pandas {pd.__version__}, scipy {scipy.__version__}\n")
    A("## 0. 정의\n")
    A(md_table(["항목", "정의"], [
        ["DDS", "저장된 시리즈별 DDS (D8 척도로 정규화된 값, `faithfulness_ad.dds` → `decaying_degradation_score(max_diff = scale)`)"],
        ["평균 DDS, PES", "칸의 시리즈별 DDS 평균; PES = f − u (`faithfulness_simic.pes`)"],
        ["f, u, 0 비율", "DDS > 0, DDS < 0, DDS == 0 인 시리즈의 비율"],
        ["중앙값, IQR", "시리즈별 DDS의 np.percentile 50, 75 − 25 (linear)"],
        ["CMI", "`faithfulness_simic.cmi(평균 DDS, PES)` — E3 보고서가 쓰는 구현 그대로 "
                "(`src/metrics/faithfulness_simic.py:122-133`: `p * d <= 0`이면 0, 아니면 2 / (1/\\|d\\| + 1/\\|p\\|))"],
        ["부호 유형", "(평균 DDS, PES)의 부호: (+,+) / (−,−) / 충돌(반대 부호, 둘 다 0 아님) / 0 포함(어느 하나가 정확히 0)"],
        ["순위", "연산자 안에서 CMI 내림차순, 1 = 최고, 동률은 평균 순위 (D-E3-3)"],
        ["W", "Kendall's W, 동률 보정 없음 (`w2_e3_report.kendall_w`, E3 보고서와 같은 함수)"],
        ["b, c의 Random·MPNative", "주 조건 값을 그대로 쓴다 (`w2_e3v2_report.rank_with`와 같음). 표의 source = main"],
    ]))
    A("")

    counts = []
    for t, title in TABLES.items():
        tab = T[T.table == t]
        A(f"## {title}\n")
        A(md_table(["평가용", "AM", "source", "n", "평균 DDS", "PES", "f", "u", "0 비율", "중앙값", "IQR [q25, q75]",
                    "CMI", "부호 유형", "순위"],
                   [[r.eval_op, r.am, r.source, r.n, f(r.mean_dds), f(r.pes), f(r.f), f(r.u), f(r.zero), f(r["median"]),
                     f"{f(r.iqr)} [{f(r.q25)}, {f(r.q75)}]", f(r.cmi), r.sign_type, f(r["rank"], 2)]
                    for _, r in tab.iterrows()]))
        vc = tab.sign_type.value_counts()
        cnt = {k: int(vc.get(k, 0)) for k in ("(+,+)", "(−,−)", "충돌", "0 포함")}
        counts.append([t, len(tab), *cnt.values()])
        _, W = rank_w(tab, "cmi")
        A(f"\n부호 유형 개수: " + ", ".join(f"{k} {v}" for k, v in cnt.items()) + f" (칸 {len(tab)}개). "
          f"Kendall's W (이 표의 순위): {f(W)}\n")
    A("## 요약 — 표별 부호 유형 개수\n")
    A(md_table(["표", "칸 수", "(+,+)", "(−,−)", "충돌", "0 포함"], counts))
    A("")

    A("## [3] 구현 확인 — (−,−) 칸의 CMI (표에서 읽기)\n")
    nn = T[T.sign_type == "(−,−)"].sort_values("table", key=lambda s: s.map({t: i for i, t in enumerate(TABLES)}), kind="stable")
    if nn.empty:
        A("해당 없음 — 어느 표에도 (−,−) 칸이 없다.\n")
    else:
        A(md_table(["표", "평가용", "AM", "평균 DDS", "PES", "CMI", "CMI > 0", "순위"],
                   [[r.table, r.eval_op, r.am, f(r.mean_dds), f(r.pes), f(r.cmi), str(bool(r.cmi > 0)), f(r["rank"], 2)]
                    for _, r in nn.iterrows()]))
        A("")

    A("## [4] 탐색적 민감도 — 사후 분석 (주 결과를 바꾸지 않음)\n")
    A("(−,−) 칸이 있는 표만. 변형 1: 그 칸의 CMI = 0. 변형 2: 그 칸의 CMI = −|CMI| (부호를 살린 CMI). "
      "다른 칸은 그대로. 순위·W는 같은 함수로 다시 계산.\n")
    any_t = False
    for t, title in TABLES.items():
        tab = T[T.table == t].copy()
        neg = tab.sign_type == "(−,−)"
        if not neg.any():
            continue
        any_t = True
        tab["cmi_zero"] = np.where(neg, 0.0, tab.cmi)
        tab["cmi_signed"] = np.where(neg, -np.abs(tab.cmi), tab.cmi)
        (r0, W0), (r1, W1), (r2, W2) = (rank_w(tab, c) for c in ("cmi", "cmi_zero", "cmi_signed"))
        A(f"### {title} — 사후 분석\n")
        ops = [op for op in (EVAL_OPS if t != "a" else ["recon_train"])]
        A(md_table(["평가용", *[f"{a} 순위 (원래 / =0 / −\\|CMI\\|)" for a in AMS]],
                   [[op, *[f"{f(r0.loc[op, a], 2)} / {f(r1.loc[op, a], 2)} / {f(r2.loc[op, a], 2)}" for a in AMS]]
                    for op in ops]))
        A(f"\nKendall's W: 원래 {f(W0)}, CMI = 0 {f(W1)}, CMI = −|CMI| {f(W2)}"
          + (" (행이 1개라 W 없음)" if t == "a" else "") + "\n")
    if not any_t:
        A("해당 없음 — (−,−) 칸이 없어 수행하지 않았다.\n")

    A("## 주 결과 대비 부호 유형이 달라진 칸 (a–d)\n")
    base = T[T.table == "main"].set_index(["eval_op", "am"])
    diff = []
    for t in ("b", "c", "d"):
        for _, r in T[T.table == t].iterrows():
            b0 = base.loc[(r.eval_op, r.am)]
            if r.sign_type != b0.sign_type:
                diff.append([t, r.eval_op, r.am, r.source, b0.sign_type, r.sign_type])
    A(md_table(["표", "평가용", "AM", "source", "주 결과 유형", "이 표의 유형"], diff) if diff else "없음 (b–d).")
    A("\n표 a(recon_train)는 주 결과에 같은 평가용 연산자가 없어 칸 대 칸 비교를 하지 않는다. 유형은 위 표 a에 있다.\n")
    REPORT.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {REPORT} and {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
