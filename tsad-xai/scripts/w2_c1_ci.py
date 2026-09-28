"""w2_c1_ci.py -- W2 follow-up #3: uncertainty intervals for the C1 numbers (aggregation of stored E2 results only).

Reads results/w2/ai.csv.gz, sar.csv.gz, e2_summary.json (scripts/w2_e2.py). No perturbation is recomputed and
the AI / SAR definitions and the aggregation are those of scripts/w2_e2_report.py (imported, not re-written):
    normal-region AI  : rows region == "normal" & ~no_donor -> per-series median per (op, |R|) -> median across series
    reversal          : share of series with AI_anom > 0 (sar.csv.gz `reversal`), per operator
    SAR, SAR_abs      : median across series (NaN = SAR-excluded / undefined, dropped as the report does)

[0] Reproduction: the point estimates are compared with the table in docs/W2_REPORT.md §5.1 at the precision
    printed there (|value − printed| ≤ half a unit of the last printed digit; a printed "0" must be exactly 0).
    Any mismatch -> exit 2, no intervals are written.
[1] Bootstrap: unit = series (= content_group, one per group); resample series with replacement, B = 2000,
    np.random.default_rng(20260928) started afresh for every subset; percentile 95 % CI (np.percentile 2.5 /
    97.5, linear). Wilson 95 % interval for the reversal share (z = 1.959963984540054).
    d. paired contrast per series: (per-series median AI of op) − (per-series median AI of recon_test) at the
    same |R|, i.e. the difference of the two series-level values the main table aggregates; median across
    series and CI. Series missing either value at that |R| are dropped for that cell.
[2] Subsets (descriptive): all 89, flagged series removed (w2_e2_report.flags), ECG vs non-ECG (all other
    domains) for a and b, each with and without flagged series.
No test statistic, no p-value. Writes results/w2/c1_ci.csv and reports/w2_c1_ci.md. No interpretation.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import REPO_ROOT  # noqa: E402
from w2_e2_report import LENS, OPS, f, flags, md_table, per_series_median  # noqa: E402

RES = REPO_ROOT / "results" / "w2"
OUT_CSV = RES / "c1_ci.csv"
REPORT = REPO_ROOT / "reports" / "w2_c1_ci.md"
W2_REPORT = REPO_ROOT / "docs" / "W2_REPORT.md"
B = 2000
SEED = 20260928
Z = 1.959963984540054
CONTRAST_OPS = ["B1_zero", "B2_global_mean", "B3_local_mean", "B4_linear_interp", "B5_gaussian", "B6_shuffle"]
N_FLAGGED_EXPECTED = 10   # W2_REPORT §5.1 "플래그 시리즈 10개"


# ------------------------------------------------------------------ series-level matrices

def series_level(ai: pd.DataFrame, sar: pd.DataFrame, ids: np.ndarray) -> dict[str, pd.DataFrame]:
    """One row per series (index = ids). Columns = cells. These are exactly the values the report medians."""
    nrm = ai[(ai.region == "normal") & ~ai.no_donor]
    ps = per_series_median(nrm, by=("op", "L_rel"))
    A = ps.pivot_table(index="series_id", columns=["op", "L_rel"], values="AI").reindex(ids)
    A = A.reindex(columns=pd.MultiIndex.from_product([OPS, LENS]))
    D = pd.DataFrame({(op, lr): A[(op, lr)] - A[("recon_test", lr)] for op in CONTRAST_OPS for lr in LENS}, index=ids)
    s = sar.set_index(["series_id", "op"])
    if s.index.duplicated().any():
        raise ValueError("sar.csv.gz has duplicate (series, op)")
    rev = s.reversal.unstack().reindex(index=ids, columns=OPS).astype(float)
    SAR = s.SAR.unstack().reindex(index=ids, columns=OPS)
    SARa = s.SAR_abs.unstack().reindex(index=ids, columns=OPS)
    if rev.isna().any().any():
        raise ValueError("reversal missing for some (series, op)")
    return {"ai": A, "diff": D, "reversal": rev, "sar": SAR, "sar_abs": SARa}


def point(M: pd.DataFrame, target: str) -> pd.Series:
    if target == "reversal":
        return M.mean()
    return M.apply(lambda c: c.dropna().median() if c.notna().any() else np.nan)


def wilson(k: int, n: int) -> tuple[float, float]:
    p = k / n
    den = 1 + Z ** 2 / n
    c = (p + Z ** 2 / (2 * n)) / den
    h = Z * np.sqrt(p * (1 - p) / n + Z ** 2 / (4 * n ** 2)) / den
    return max(0.0, c - h), min(1.0, c + h)


def run_subset(mats: dict, subset: str, ids: np.ndarray, targets: list[str]) -> list[dict]:
    rows = []
    rng = np.random.default_rng(SEED)   # afresh per subset; one resample index stream shared by its targets
    idx = [rng.integers(0, ids.size, ids.size) for _ in range(B)]
    for target in targets:
        M = mats[target].loc[ids]
        pt = point(M, target)
        X = M.to_numpy(dtype=float)
        ok = ~np.all(np.isnan(X), axis=0)
        bs = np.full((B, X.shape[1]), np.nan)
        for b, ii in enumerate(idx):
            Xb = X[ii][:, ok]
            if target == "reversal":
                bs[b, ok] = Xb.mean(axis=0)
            else:
                if np.all(np.isnan(Xb), axis=0).any():
                    raise ValueError(f"{subset}/{target}: a replicate has a cell with no series")
                bs[b, ok] = np.nanmedian(Xb, axis=0)
        lo, hi = (np.full(X.shape[1], np.nan) for _ in range(2))
        lo[ok], hi[ok] = np.percentile(bs[:, ok], [2.5, 97.5], axis=0)
        for j, col in enumerate(M.columns):
            op, lr = (col if isinstance(col, tuple) else (col, ""))
            r = {"subset": subset, "target": target, "op": op, "L_rel": lr, "n_series": int(M[col].notna().sum()),
                 "point": float(pt[col]), "ci_lo": float(lo[j]), "ci_hi": float(hi[j]),
                 "wilson_lo": np.nan, "wilson_hi": np.nan, "B": B, "seed": SEED}
            if target == "reversal":
                k = int(M[col].sum())
                r["wilson_lo"], r["wilson_hi"] = wilson(k, int(M[col].size))
                r["k"] = k
            r["ci_contains_0"] = bool(ok[j] and lo[j] <= 0 <= hi[j])
            rows.append(r)
    return rows


# ------------------------------------------------------------------ [0] reproduction

def parse_w2_report() -> pd.DataFrame:
    text = W2_REPORT.read_text(encoding="utf-8").split("\n")
    i0 = next(i for i, l in enumerate(text) if l.startswith("### 5.1"))
    rows = []
    for l in text[i0:]:
        m = re.match(r"^\| (\S+) \| (.+) \|$", l)
        if m and m.group(1) in OPS:
            cells = [c.strip() for c in l.strip("|").split("|")]
            rows.append(cells)
        if l.startswith("### 5.2"):
            break
    if len(rows) != len(OPS):
        raise ValueError(f"W2_REPORT §5.1: parsed {len(rows)} operator rows, expected {len(OPS)}")
    return pd.DataFrame(rows, columns=["op", "0.25", "0.5", "1.0", "reversal", "sar", "sar_abs"]).set_index("op")


def matches(printed: str, value: float, pct: bool = False) -> bool:
    s = printed.strip()
    if s == "—":
        return not np.isfinite(value)
    if pct:
        s = s.rstrip("%")
        value = 100 * value
    dec = len(s.split(".")[1]) if "." in s else 0
    v = float(s)
    if dec == 0:
        return value == v
    return abs(value - v) <= 0.5 * 10 ** (-dec) + 1e-12


def main() -> int:
    ai = pd.read_csv(RES / "ai.csv.gz", dtype={"L_rel": str})
    sar = pd.read_csv(RES / "sar.csv.gz")
    summ = json.loads((RES / "e2_summary.json").read_text())
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()
    fl = flags(ai, sar, summ)
    flagged = set(fl[fl.flagged].series_id)
    if len(flagged) != N_FLAGGED_EXPECTED:
        raise ValueError(f"{len(flagged)} flagged series, W2_REPORT says {N_FLAGGED_EXPECTED}")
    dom = sar.drop_duplicates("series_id").set_index("series_id").domain
    all_ids = np.array(sorted(dom.index))
    if all_ids.size != 89:
        raise ValueError(f"{all_ids.size} series, expected 89")
    mats = series_level(ai, sar, all_ids)

    # [0]
    pr = parse_w2_report()
    pa, prv = point(mats["ai"], "ai"), point(mats["reversal"], "reversal")
    ps_, psa = point(mats["sar"], "sar"), point(mats["sar_abs"], "sar_abs")
    rep_rows, bad = [], 0
    for op in OPS:
        vals = [(lr, pa[(op, lr)], False) for lr in LENS] + [("reversal", prv[op], True), ("sar", ps_[op], False),
                                                               ("sar_abs", psa[op], False)]
        for col, v, pct in vals:
            ok = matches(pr.loc[op, col], float(v), pct)
            bad += not ok
            rep_rows.append([op, col, pr.loc[op, col], f"{100 * v:.4f}%" if pct else f(float(v), 6), "일치" if ok else "**불일치**"])
    if bad:
        REPORT.write_text("# W2 후속 #3 — STOP: 재현 불일치\n\n" + md_table(
            ["연산자", "열", "W2_REPORT", "재계산", "판정"], rep_rows) + "\n", encoding="utf-8")
        print(f"STOP: {bad} mismatches; see {REPORT}")
        return 2

    # [1], [2]
    no_fl = np.array([i for i in all_ids if i not in flagged])
    subsets = {
        "all": (all_ids, ["ai", "reversal", "sar", "sar_abs", "diff"]),
        "no_flagged": (no_fl, ["ai", "reversal", "sar", "sar_abs", "diff"]),
        "ECG": (all_ids[(dom.loc[all_ids] == "ECG").to_numpy()], ["ai", "reversal"]),
        "nonECG": (all_ids[(dom.loc[all_ids] != "ECG").to_numpy()], ["ai", "reversal"]),
        "ECG_no_flagged": (no_fl[(dom.loc[no_fl] == "ECG").to_numpy()], ["ai", "reversal"]),
        "nonECG_no_flagged": (no_fl[(dom.loc[no_fl] != "ECG").to_numpy()], ["ai", "reversal"]),
    }
    rows = []
    for name, (ids, targets) in subsets.items():
        rows += run_subset(mats, name, ids, targets)
    T = pd.DataFrame(rows)
    T.to_csv(OUT_CSV, index=False, float_format="%.10g")

    def ci(r):
        return f"{f(r.point)} [{f(r.ci_lo)}, {f(r.ci_hi)}]"

    def g(subset, target):
        return T[(T.subset == subset) & (T.target == target)]

    L = []
    A = L.append
    A("# W2 후속 #3 — C1 수치의 신뢰구간\n")
    A(f"GENERATED by scripts/w2_c1_ci.py at {datetime.now(timezone.utc).isoformat(timespec='seconds')}, HEAD "
      f"`{head[:7]}`. Do not hand-edit. 기존 파일 집계만: `results/w2/ai.csv.gz`, `sar.csv.gz`, `e2_summary.json` "
      f"(scripts/w2_e2.py run {summ['generated_utc']}). 새 섭동 계산 없음. 검정 통계량·p값 없음. 해석 없음.\n")
    A(f"numpy {np.__version__}, pandas {pd.__version__}\n")
    A("## 0. 방법\n")
    A(md_table(["항목", "내용"], [
        ["AI, SAR 정의·집계", "`scripts/w2_e2_report.py`와 같음 (그 모듈의 `per_series_median`, `flags`를 import). "
                          "정상 구간 AI = region normal, no_donor 제외 → 시리즈 안 중앙값 → 시리즈 간 중앙값"],
        ["reversal", "AI_anom > 0인 시리즈 비율 (`sar.csv.gz` `reversal`)"],
        ["SAR, SAR_abs", "시리즈 간 중앙값, 정의되지 않은 시리즈(NaN)는 제외 (보고서와 같음)"],
        ["부트스트랩", f"시리즈(= content_group) 복원 추출, B = {B}, `np.random.default_rng({SEED})`를 부분집합마다 새로 시작, "
                    "한 부분집합 안의 모든 대상은 같은 재추출 인덱스를 공유, 백분위 95% CI (np.percentile 2.5, 97.5)"],
        ["Wilson", "reversal 비율의 Wilson 95% 구간 (z = 1.95996)"],
        ["짝지은 대비 d", "시리즈별 (연산자의 시리즈 중앙값 AI − recon_test의 시리즈 중앙값 AI), 같은 \\|R\\|; 한쪽 값이 없는 "
                      "시리즈는 그 칸에서 제외 → 시리즈 간 중앙값과 CI"],
        ["CI가 0을 포함", "ci_lo ≤ 0 ≤ ci_hi"],
        ["플래그", f"`w2_e2_report.flags` (D-E2-2) — {len(flagged)}개: {', '.join(map(str, sorted(flagged)))}"],
        ["ECG / 비ECG", f"domain == ECG ({int((dom == 'ECG').sum())}개) / 그 외 전부 ({int((dom != 'ECG').sum())}개)"],
    ]))
    A("")
    A("## 1. [0] 재현 확인 — `docs/W2_REPORT.md` §5.1 표\n")
    A(f"{len(rep_rows)}칸 (연산자 {len(OPS)} × 열 6) 모두 표시 자릿수에서 일치 (불일치 0).\n")
    A(md_table(["연산자", "열", "W2_REPORT", "재계산", "판정"], rep_rows))
    A("")
    for sub, title in (("all", "89 시리즈"), ("no_flagged", f"플래그 {len(flagged)}개 제외 ({no_fl.size} 시리즈)")):
        A(f"## 2. [1] 부트스트랩 — {title}\n")
        A("### a. 정상 구간 AI 중앙값 [95% CI]\n")
        t = g(sub, "ai").set_index(["op", "L_rel"])
        A(md_table(["연산자", *[f"{lr}w" for lr in LENS], "n (0.25w / 0.5w / 1.0w)"],
                   [[op, *[ci(t.loc[(op, lr)]) for lr in LENS], " / ".join(str(t.loc[(op, lr)].n_series) for lr in LENS)]
                    for op in OPS]))
        A("\n### b. reversal 비율 — 부트스트랩 [95% CI], Wilson [95%]\n")
        t = g(sub, "reversal").set_index("op")
        A(md_table(["연산자", "k / n", "비율", "부트스트랩 CI", "Wilson"],
                   [[op, f"{int(t.loc[op].k)} / {t.loc[op].n_series}", f"{100 * t.loc[op].point:.1f}%",
                     f"[{100 * t.loc[op].ci_lo:.1f}%, {100 * t.loc[op].ci_hi:.1f}%]",
                     f"[{100 * t.loc[op].wilson_lo:.1f}%, {100 * t.loc[op].wilson_hi:.1f}%]"] for op in OPS]))
        A("\n### c. SAR, SAR_abs 중앙값 [95% CI]\n")
        t1, t2 = g(sub, "sar").set_index("op"), g(sub, "sar_abs").set_index("op")
        A(md_table(["연산자", "n", "SAR", "SAR_abs"],
                   [[op, t1.loc[op].n_series, ci(t1.loc[op]), ci(t2.loc[op])] for op in OPS]))
        A("\n### d. 짝지은 대비: 연산자 AI − recon_test AI (시리즈별) 중앙값 [95% CI]\n")
        t = g(sub, "diff").set_index(["op", "L_rel"])
        A(md_table(["연산자", *[f"{lr}w" for lr in LENS], "n (0.25w / 0.5w / 1.0w)"],
                   [[op, *[ci(t.loc[(op, lr)]) for lr in LENS], " / ".join(str(t.loc[(op, lr)].n_series) for lr in LENS)]
                    for op in CONTRAST_OPS]))
        A("")
    A("## 3. [2] ECG 대 비ECG (a, b; 기술 통계)\n")
    for tag, e, ne in (("89 시리즈", "ECG", "nonECG"), ("플래그 제외", "ECG_no_flagged", "nonECG_no_flagged")):
        te_, tn_ = g(e, "ai").set_index(["op", "L_rel"]), g(ne, "ai").set_index(["op", "L_rel"])
        ne_n, nn_n = T[T.subset == e].n_series.max(), T[T.subset == ne].n_series.max()
        A(f"### {tag} — a. 정상 구간 AI 중앙값 [95% CI] (ECG n ≤ {ne_n}, 비ECG n ≤ {nn_n})\n")
        A(md_table(["연산자", "\\|R\\|", "ECG", "비ECG", "점추정 부호 (ECG / 비ECG)"],
                   [[op, f"{lr}w", ci(te_.loc[(op, lr)]), ci(tn_.loc[(op, lr)]),
                     f"{np.sign(te_.loc[(op, lr)].point):+.0f} / {np.sign(tn_.loc[(op, lr)].point):+.0f}"]
                    for op in OPS for lr in LENS]))
        te_, tn_ = g(e, "reversal").set_index("op"), g(ne, "reversal").set_index("op")
        A(f"\n### {tag} — b. reversal 비율 (부트스트랩 CI; Wilson)\n")
        A(md_table(["연산자", "ECG k/n", "ECG", "비ECG k/n", "비ECG"],
                   [[op, f"{int(te_.loc[op].k)}/{te_.loc[op].n_series}",
                     f"{100 * te_.loc[op].point:.1f}% [{100 * te_.loc[op].ci_lo:.1f}, {100 * te_.loc[op].ci_hi:.1f}]; "
                     f"W [{100 * te_.loc[op].wilson_lo:.1f}, {100 * te_.loc[op].wilson_hi:.1f}]",
                     f"{int(tn_.loc[op].k)}/{tn_.loc[op].n_series}",
                     f"{100 * tn_.loc[op].point:.1f}% [{100 * tn_.loc[op].ci_lo:.1f}, {100 * tn_.loc[op].ci_hi:.1f}]; "
                     f"W [{100 * tn_.loc[op].wilson_lo:.1f}, {100 * tn_.loc[op].wilson_hi:.1f}]"] for op in OPS]))
        A("")
    A("## 4. 목록\n")
    for sub in subsets:
        z = T[(T.subset == sub) & T.ci_contains_0]
        A(f"**CI가 0을 포함하는 칸 — {sub}:** "
          + ("; ".join(f"{r.target} {r.op}{' ' + r.L_rel + 'w' if r.L_rel else ''}" for r in z.itertuples()) or "없음") + "\n")
    for tag, e, ne in (("89 시리즈", "ECG", "nonECG"), ("플래그 제외", "ECG_no_flagged", "nonECG_no_flagged")):
        a1, a2 = g(e, "ai").set_index(["op", "L_rel"]).point, g(ne, "ai").set_index(["op", "L_rel"]).point
        dif = [f"{op} {lr}w (ECG {f(a1[(op, lr)])}, 비ECG {f(a2[(op, lr)])})" for op in OPS for lr in LENS
               if np.sign(a1[(op, lr)]) != np.sign(a2[(op, lr)])]
        A(f"**a의 점추정 부호가 ECG와 비ECG에서 다른 칸 — {tag}:** " + ("; ".join(dif) or "없음") + "\n")
    REPORT.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {REPORT} and {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
