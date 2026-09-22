"""BRIEF_A-followup F2/F3/F4 orchestration for scripts/02_audit.py.

Takes the per-series rows the audit loop built (plus the loaded arrays),
computes recording groups, sentinel statistics and subset sizes, adds the new
manifest columns in place, and returns everything the report needs. Pure
computation: writing files is left to the script, except the derived
sentinel index files whose location is a config value.
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.config import load_yaml, resolve
from src.data import sentinels as sen
from src.data.recording_groups import (SeriesRef, compare_pairs, content_groups,
                                       group_sizes, grouping_disagreements, name_group_of)
from src.data.subsets import n_groups, select, summarise

NA = "NA"

FOLLOWUP_COLUMNS = [
    "name_group", "content_group", "name_group_size", "content_group_size",
    "sentinel_exact_n", "sentinel_near_n", "sentinel_spike_n", "sentinel_max_k",
    "sentinel_in_gt_n", "sentinel_in_train_n", "sentinel_in_test_normal_n", "sentinel_twin_status",
]


# ---------------------------------------------------------------- F2
def run_recording_groups(rows: list[dict], series: dict[int, np.ndarray]) -> dict:
    cfg = load_yaml("stats")["recording_groups"]
    refs = [SeriesRef(num=r["num"], name=r["name"], domain=r["domain_goswami"],
                      length=r["length"], train_end=r["train_end_raw"], x=series[r["num"]])
            for r in rows]
    pairs = compare_pairs(refs, compare_points=int(cfg["compare_points"]),
                          corr_threshold=float(cfg["corr_threshold"]),
                          cross_length_tol=float(cfg["cross_length_tol"]))
    cg = content_groups(refs, pairs)
    ng = {r["num"]: name_group_of(r["name"]) for r in rows}
    cg_sizes, ng_sizes = group_sizes(cg), group_sizes(ng)
    dis = grouping_disagreements(refs, cg)
    crossing = set(dis["diff_name_same_content"])
    for r in rows:
        r["name_group"] = ng[r["num"]]
        r["content_group"] = cg[r["num"]]
        r["name_group_size"] = ng_sizes[ng[r["num"]]]
        r["content_group_size"] = cg_sizes[cg[r["num"]]]
        if cg[r["num"]] in crossing:
            r["_flags"].append("CONTENT_GROUP_CROSSES_NAME")
    return {"cfg": cfg, "pairs": pairs, "content": cg, "name": ng,
            "cg_sizes": cg_sizes, "ng_sizes": ng_sizes, "disagreements": dis}


def write_pairs_csv(pairs, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["num_a", "num_b", "name_a", "name_b", "same_name_group", "domain", "length_a",
            "length_b", "train_end_a", "train_end_b", "n_compared", "corr", "same_recording"]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for p in pairs:
            w.writerow([getattr(p, c) if c != "corr" else f"{p.corr:.6f}" for c in cols])


# ---------------------------------------------------------------- F3
def run_sentinels(rows: list[dict], series: dict[int, np.ndarray], deck_text: dict[int, str]) -> dict:
    """deck_text: num -> concatenated slide text for that series (STOP-3 check)."""
    cfg = load_yaml("sentinels")
    if float(cfg["value"]) != sen.SENTINEL_VALUE:
        raise ValueError("configs/sentinels.yaml value must be -999.0 (module constant)")
    factor, w, k = float(cfg["near_mad_factor"]), int(cfg["twin_window"]), float(cfg["twin_k"])
    derived = resolve(cfg["derived_dir"])
    derived.mkdir(parents=True, exist_ok=True)

    by_num = {r["num"]: r for r in rows}
    by_name_group = defaultdict(list)
    for r in rows:
        by_name_group[r["name_group"]].append(r)

    results: dict[int, sen.SentinelResult] = {}
    for r in rows:
        x = series[r["num"]]
        res = sen.detect(x, r["train_end_raw"], near_mad_factor=factor)
        if r["start0"] == NA:
            raise RuntimeError(f"{r['filename']}: start0 undetermined; sentinels need the convention")
        sen.classify(res, train_end=r["train_end_raw"], start0=int(r["start0"]), stop0=int(r["stop0"]))
        results[r["num"]] = res
        r["sentinel_exact_n"] = int(len(res.exact_idx))
        r["sentinel_near_n"] = int(len(res.near_idx))
        ratios = [sen.outlier_ratio(x, int(i), window=w) for i in res.exact_idx]
        r["sentinel_spike_n"] = int(sum(1 for q in ratios if q > k))
        r["sentinel_max_k"] = f"{max(ratios):.2f}" if ratios else NA
        r["sentinel_in_gt_n"] = int(len(res.in_gt))
        r["sentinel_in_train_n"] = int(len(res.in_train))
        r["sentinel_in_test_normal_n"] = int(len(res.in_test_normal))
        r["_sentinel_tol"] = res.tol
        if len(res.in_train) or len(res.in_test_normal):
            r["_flags"].append("SENTINEL_IN_NORMAL")
        if len(res.in_gt):
            txt = deck_text.get(r["num"], "")
            if "-999" not in txt and "999" not in txt:
                r["_flags"].append("SENTINEL_IN_GT_UNDOCUMENTED")
        np.save(derived / f"{r['num']:03d}.npy", res.exclusion_idx)

    # twin check: plain series with any sentinel vs its DISTORTED/NOISE twins
    twin_checks: dict[int, list[sen.TwinCheck]] = {}
    for r in rows:
        res = results[r["num"]]
        if r["variant"] != "plain" or len(res.exact_idx) == 0:
            r["sentinel_twin_status"] = NA
            continue
        twins = [t for t in by_name_group[r["name_group"]] if t["variant"] != "plain"]
        if not twins:
            r["sentinel_twin_status"] = NA
            continue
        checks = [sen.twin_check(res.exact_idx, series[t["num"]], t["num"], window=w, k=k)
                  for t in sorted(twins, key=lambda t: t["num"])]
        twin_checks[r["num"]] = checks
        r["sentinel_twin_status"] = ";".join(c.summary() for c in checks)

    return {"cfg": cfg, "results": results, "twin_checks": twin_checks, "derived_dir": derived}


def sentinel_figures(rows: list[dict], series: dict[int, np.ndarray], sent: dict) -> list[Path]:
    """Top-N plain series (most sentinels) with >= 1 twin that is not length_mismatch."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = sent["cfg"]
    out_dir = resolve(cfg["figures_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.png"):     # the directory is script output; no stale files
        stale.unlink()
    by_num = {r["num"]: r for r in rows}
    cands = []
    for num, checks in sent["twin_checks"].items():
        drawable = [c.twin_num for c in checks if c.status != "length_mismatch"]
        if drawable:
            cands.append((len(sent["results"][num].exact_idx), num, drawable))
    cands.sort(key=lambda c: (-c[0], c[1]))   # most sentinels first, ties by num
    written = []
    for n_s, num, twins in cands[:int(cfg["figures_top_n"])]:
        r = by_num[num]
        res = sent["results"][num]
        idx = res.exact_idx
        # zoom on the first sentinel outside the GT (or the first one at all)
        focus = int(res.exclusion_idx[0]) if len(res.exclusion_idx) else int(idx[0])
        lo, hi = max(0, focus - 300), focus + 300
        nums = [num] + twins
        fig, axes = plt.subplots(len(nums), 2, figsize=(13, 2.4 * len(nums)), squeeze=False)
        for row_i, n in enumerate(nums):
            x = series[n]
            t = by_num[n]
            a_full, a_zoom = axes[row_i]
            a_full.plot(x, lw=0.3, color="k")
            for i in idx:
                if i < len(x):
                    a_full.axvline(i, color="tab:red", lw=0.4, alpha=0.6)
            a_full.axvspan(int(t["start0"]), int(t["stop0"]), color="tab:green", alpha=0.3)
            a_full.axvline(t["train_end_raw"], color="tab:blue", lw=0.8, ls="--")
            a_full.set_title(f"{t['filename']}  (n sentinel idx = {int((idx < len(x)).sum())} of {len(idx)})",
                             fontsize=8)
            seg = x[lo:min(hi, len(x))]
            a_zoom.plot(range(lo, lo + len(seg)), seg, "k.-", lw=0.6, ms=2)
            for i in idx:
                if lo <= i < min(hi, len(x)):
                    a_zoom.axvline(i, color="tab:red", lw=0.8, alpha=0.7)
            a_zoom.set_title(f"zoom [{lo}, {hi}) around sentinel idx {focus}", fontsize=8)
        fig.suptitle(f"{r['filename']}: −999 sentinels (red) vs twins, same 0-based indices; "
                     "green = GT, blue dashed = train_end", fontsize=9)
        fig.tight_layout()
        p = out_dir / f"{num:03d}_{r['name']}.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        written.append(p)
    return written


# ---------------------------------------------------------------- F4
def run_subsets(rows: list[dict]) -> list[dict]:
    cfg = load_yaml("subsets")
    combos = [[name] for name in cfg["subsets"]] + [list(c) for c in cfg.get("report_combinations", [])]
    out = []
    for combo in combos:
        d = summarise(rows, combo, subsets_cfg=cfg["subsets"])
        # sensitivity column: the same subset counted in the sensitivity unit
        d["n_sensitivity_groups"] = n_groups(select(rows, combo, cfg["subsets"]),
                                             load_yaml("stats")["sensitivity_unit"])
        out.append(d)
    return out
