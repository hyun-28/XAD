"""w2_e4.py -- W2 rev1 §10 E4 (C3, reduced): plausibility of the E3 attributions against the GT interval.

Decisions: D-E4-1/2/3 and D-E3-5 (commit b8b783c). Called "plausibility" throughout, distinct from faithfulness.
No hypothesis test (rev1 §10, D-E4-3.6).

Per series (89) and AM (4), attributions = E3 main condition (explanation operator recon_test,
results/w2/e3_attr.csv.gz), on the K segments of Ω (D13):
  projection  segment value -> ReLU (main) or |R| (sensitivity) at segment level, then R_k / |seg_k| to every point
              of seg_k (mass preserving)                                                         (D-E4-2.1)
  RMA         mass inside GT / mass over Ω; NaN if the mass is 0                                 (D-E4-2.2, Arras §3.5 Eq. 1-2)
  RRA         K = |GT| points; share of the top-K points lying in GT; ties by a seeded random rank,
              seed = (series, AM) as D-E3-5                                                     (D-E4-2.3, Arras §3.5 Eq. 3-4)
  chance      |GT| / |Ω| (points); oracle = RMA / RRA of the explanation o_k = |seg_k ∩ GT| / |seg_k|
  normalised  (value - chance) / (oracle - chance); NaN if the denominator < 1e-6              (D-E4-2.4)
Ranking (D-E4-3): primary = normalised RRA (ReLU); sensitivity = normalised RMA (ReLU) and normalised RRA (|R|)
(the |R| sensitivity is applied to the primary metric only, not crossed with RMA). content_group (= series) mean,
NaN excluded (counted) -> AM ranking (descending; ties -> average rank). Kendall τ (tau-b) against the E3 main
ranking (D-E3-5 random-tie version) of each evaluation operator B1-B6; group-level τ distribution.

Writes results/w2/plausibility.csv and reports/w2_plausibility.md.
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
from scipy.stats import kendalltau, rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import REPO_ROOT, load_yaml  # noqa: E402
from src.metrics import faithfulness_ad as fa  # noqa: E402
from w2_e3_report import AMS, EVAL_OPS, f, md_table  # noqa: E402
from w2_e3v2_report import analysis  # noqa: E402

RES = REPO_ROOT / "results" / "w2"
REPORT = REPO_ROOT / "reports" / "w2_plausibility.md"
DECISION_COMMIT_2 = "b8b783cc027e8cc3bf44f49219410ac5a69d35e9"
TIE_TAG = 5                 # same seed tuple as D-E3-5: (e3.seed, series, AM index, 5)
ORACLE_INDEX = len(AMS)     # the oracle explanation gets its own seed slot
DEN_MIN = 1e-6


def point_values(seg_vals: np.ndarray, segs, lo: int, hi: int) -> np.ndarray:
    v = np.zeros(hi - lo)
    for r, (a, b) in zip(seg_vals, segs):
        v[a - lo:b - lo] = r / (b - a)
    return v


def rma(v: np.ndarray, gt_mask: np.ndarray) -> float:
    tot = v.sum()
    return float(v[gt_mask].sum() / tot) if tot > 0 else np.nan


def rra(v: np.ndarray, gt_mask: np.ndarray, rng: np.random.Generator) -> float:
    k = int(gt_mask.sum())
    tie = rng.permutation(v.size)
    top = np.lexsort((tie, -v))[:k]
    return float(gt_mask[top].sum() / k)


def main() -> int:
    from src.runlog import env_header
    from w2_e2 import check_decisions
    header = env_header()
    print(header)
    dec = check_decisions()
    if subprocess.run(["git", "-C", str(REPO_ROOT), "merge-base", "--is-ancestor", DECISION_COMMIT_2, "HEAD"]).returncode:
        raise RuntimeError("decision commit b8b783c is not an ancestor of HEAD")
    seed = int(load_yaml("w2")["e3"]["seed"])
    sample = {int(e["num"]): e for e in yaml.safe_load((REPO_ROOT / "configs" / "w2_sample.yaml").read_text())["series"]}
    at = pd.read_csv(RES / "e3_attr.csv.gz")
    at = at[(at.condition == "main")]
    rows = []
    for num, e in sorted(sample.items()):
        lo, hi = fa.omega(e["train_end"], tuple(e["gt"]), e["w"], e["n"])
        segs, g = fa.segments((lo, hi), e["w"])
        s0, s1 = e["gt"]
        if not (lo <= s0 < s1 <= hi):
            raise RuntimeError(f"{num}: GT not inside Ω")
        gt = np.zeros(hi - lo, dtype=bool)
        gt[s0 - lo:s1 - lo] = True
        chance = gt.sum() / gt.size
        orc = np.array([max(0, min(b, s1) - max(a, s0)) / (b - a) for a, b in segs])
        vo = point_values(orc, segs, lo, hi)
        o_rma = rma(vo, gt)
        o_rra = rra(vo, gt, np.random.default_rng([seed, num, ORACLE_INDEX, TIE_TAG]))
        for ai, am in enumerate(AMS):
            r = at[(at.series_id == num) & (at.am == am)]
            if len(r) != 1:
                raise RuntimeError(f"{num} {am}: {len(r)} attribution rows")
            R = np.array(json.loads(r.attr.iloc[0]))
            if R.size != len(segs):
                raise RuntimeError(f"{num} {am}: {R.size} segments vs K = {len(segs)}")
            neg = float(np.abs(R[R < 0]).sum() / np.abs(R).sum()) if np.abs(R).sum() > 0 else np.nan
            row = {"series_id": num, "content_group": e["content_group"], "domain": e["domain"], "am": am,
                   "K": len(segs), "g": g, "omega_len": hi - lo, "gt_len": s1 - s0, "gt_shorter_than_g": (s1 - s0) < g,
                   "chance": chance, "oracle_rma": o_rma, "oracle_rra": o_rra, "neg_mass_ratio": neg}
            for tag, vals in (("relu", np.maximum(R, 0.0)), ("abs", np.abs(R))):
                v = point_values(vals, segs, lo, hi)
                m = rma(v, gt)
                k = rra(v, gt, np.random.default_rng([seed, num, ai, TIE_TAG]))
                dm, dk = o_rma - chance, o_rra - chance
                row.update({f"rma_{tag}": m, f"rra_{tag}": k,
                            f"nrma_{tag}": (m - chance) / dm if (np.isfinite(m) and dm >= DEN_MIN) else np.nan,
                            f"nrra_{tag}": (k - chance) / dk if dk >= DEN_MIN else np.nan})
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "plausibility.csv", index=False)
    write_report(df, header, dec)
    print(f"E4: {df.series_id.nunique()} series x {len(AMS)} AMs; wrote {REPORT}")
    return 0


def e4_rank(df: pd.DataFrame, col: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    means = df.groupby("am")[col].mean().reindex(AMS)            # NaN excluded by mean()
    n_nan = df.groupby("am")[col].apply(lambda s: int(s.isna().sum())).reindex(AMS)
    return means, pd.Series(rankdata(-means.to_numpy()), index=AMS), n_nan


def write_report(df: pd.DataFrame, header: str, dec: dict) -> None:
    v2 = pd.read_csv(RES / "faithfulness_v2.csv.gz")
    A3 = analysis(v2)
    rk3 = A3["rk"]
    L = []
    A = L.append
    A("# W2 E4 — plausibility of the E3 attributions against the GT interval (C3, reduced)\n")
    A(f"GENERATED by scripts/w2_e4.py at {datetime.now(timezone.utc).isoformat(timespec='seconds')}; HEAD "
      f"`{dec['head'][:7]}`. Do not hand-edit. This axis is **plausibility** (agreement with the labelled interval), "
      "not faithfulness. Descriptive statistics only; no hypothesis test (rev1 §10). No interpretation is given.\n")
    A(f"`{header}`\n")
    A(md_table(["item", "value"], [
        ["decisions", "D-E4-1/2/3, D-E3-5 (`b8b783c`); metrics: Arras et al. arXiv:2003.07258 §3.5"],
        ["attributions", "E3 main condition (explanation operator recon_test), `results/w2/e3_attr.csv.gz`"],
        ["projection", "segment value → ReLU (main) / |R| (sensitivity) → R_k/|seg_k| on each point of Ω"],
        ["ties (RRA)", "seeded random rank, seed = (series, AM) as D-E3-5; oracle has its own seed slot"],
        ["E3 comparison", "E3 main ranking, random-tie version (`reports/w2_ranking.md`)"],
        ["series × AMs", f"{df.series_id.nunique()} × {len(AMS)}"],
    ]))
    A("")
    A("## 1. Raw values per AM (mean across series; chance and oracle per series averaged)\n")
    rows = []
    for am in AMS:
        d = df[df.am == am]
        rows.append([am, f(d.rma_relu.mean()), f(d.rra_relu.mean()), f(d.rma_abs.mean()), f(d.rra_abs.mean()),
                     int(d.rma_relu.isna().sum()), int(d.rma_abs.isna().sum())])
    A(md_table(["AM", "RMA (ReLU)", "RRA (ReLU)", "RMA (|R|)", "RRA (|R|)", "RMA NaN (ReLU)", "RMA NaN (|R|)"], rows))
    d0 = df.drop_duplicates("series_id")
    A(f"\nPer series (same for every AM): chance |GT|/|Ω| mean {f(d0.chance.mean())} (median {f(d0.chance.median())}); "
      f"oracle RMA mean {f(d0.oracle_rma.mean())}, oracle RRA mean {f(d0.oracle_rra.mean())}.\n")
    A("## 2. Normalised values and the plausibility ranking (D-E4-3)\n")
    cols = [("nrra_relu", "normalised RRA, ReLU (primary)"), ("nrma_relu", "normalised RMA, ReLU (sensitivity)"),
            ("nrra_abs", "normalised RRA, |R| (sensitivity)")]
    rows, ranks = [], {}
    for col, lab in cols:
        m, r, n = e4_rank(df, col)
        ranks[col] = r
        for am in AMS:
            s = df[df.am == am][col]
            rows.append([lab, am, f(m[am]), f(s.median()), f"[{f(s.quantile(.25))}, {f(s.quantile(.75))}]",
                         int(n[am]), f(r[am], 2)])
    A(md_table(["metric", "AM", "mean", "median", "IQR", "NaN excluded", "rank"], rows))
    A("")
    A("## 3. Plausibility (E4) vs faithfulness (E3) — AM ranks side by side\n")
    A(md_table(["AM", "E4 primary", "E4 nRMA", "E4 nRRA |R|", *[f"E3 {op}" for op in EVAL_OPS]],
               [[am, f(ranks["nrra_relu"][am], 2), f(ranks["nrma_relu"][am], 2), f(ranks["nrra_abs"][am], 2),
                 *[f(rk3.loc[op, am], 2) for op in EVAL_OPS]] for am in AMS]))
    A("\n**Kendall τ (tau-b) between the E4 ranking and each E3 operator's ranking:**\n")
    rows = []
    for col, lab in cols:
        r = [lab]
        for op in EVAL_OPS:
            t = kendalltau(ranks[col].to_numpy(), rk3.loc[op, AMS].to_numpy()).statistic
            r.append(f(float(t)))
        rows.append(r)
    A(md_table(["E4 metric", *EVAL_OPS], rows))
    A("\n**Resolution limit:** with 4 AMs Kendall τ takes only 7 values without ties (−1, −2/3, −1/3, 0, 1/3, 2/3, 1).\n")
    # group-level τ
    main = v2[v2.condition == "main"]
    taus = []
    for num, g in main.groupby("series_id"):
        e4 = df[df.series_id == num].set_index("am").nrra_relu.reindex(AMS)
        if e4.isna().any():
            continue
        r4 = rankdata(-e4.to_numpy())
        t = g.pivot_table(index="eval_op", columns="am", values="dds").reindex(index=EVAL_OPS, columns=AMS)
        for op in EVAL_OPS:
            r3 = rankdata(-t.loc[op].to_numpy())
            taus.append({"series_id": num, "eval_op": op,
                         "tau": float(kendalltau(r4, r3).statistic) if np.ptp(r4) > 0 and np.ptp(r3) > 0 else np.nan})
    tg = pd.DataFrame(taus)
    rows = []
    for op in EVAL_OPS:
        s = tg[tg.eval_op == op].tau.dropna()
        rows.append([op, len(s), f(s.median()), f"[{f(s.quantile(.25))}, {f(s.quantile(.75))}]", f(s.mean())])
    A("**Group level** (per series: E3 AMs ranked by DDS under each operator, E4 AMs by normalised RRA):\n")
    A(md_table(["E3 operator", "series", "median τ", "IQR", "mean τ"], rows))
    A("")
    A("## 4. Diagnostics\n")
    rs = df[df.am == "Random"].nrra_relu.dropna()
    A(f"- **Sanity (D-E4-3.7): Random normalised RRA** — median {f(rs.median())}, IQR [{f(rs.quantile(.25))}, "
      f"{f(rs.quantile(.75))}], n = {len(rs)}.\n")
    rows = []
    for am in AMS:
        s = df[df.am == am].neg_mass_ratio
        rows.append([am, f(s.median()), f"[{f(s.quantile(.25))}, {f(s.quantile(.75))}]", f(s.max())])
    A("- **Negative mass ratio Σ|R⁻| / Σ|R| per AM** (diagnostic):\n")
    A(md_table(["AM", "median", "IQR", "max"], rows))
    short = d0[d0.gt_shorter_than_g]
    A(f"\n- **GT shorter than the segment length g:** {len(short)} of {len(d0)} series"
      + (f"; their oracle RMA median {f(short.oracle_rma.median())} (range {f(short.oracle_rma.min())}–"
         f"{f(short.oracle_rma.max())}), oracle RRA median {f(short.oracle_rra.median())} (range "
         f"{f(short.oracle_rra.min())}–{f(short.oracle_rra.max())})" if len(short) else "") + ".\n")
    den = d0[(d0.oracle_rra - d0.chance) < DEN_MIN]
    A(f"- Series with oracle − chance < 1e-6 (normalised RRA NaN for every AM): {len(den)}.\n")
    A("- Mass outside GT but inside Ω is not corrected (D-E4-2.6).\n")
    REPORT.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
