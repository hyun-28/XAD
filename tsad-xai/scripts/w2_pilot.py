"""w2_pilot.py -- BRIEF3 W2-0: one-series pilot (P1 selection, P5 run, P6 timing, P7 figure F2).

    P1  pick the series by the coded rule in configs/w2_pilot.yaml (never by eye)
    P5  3 positions x 4 lengths x 8 erasers x 2 detectors (Z = z-norm MP, R = raw MP);
        every condition is scored twice: incremental (W2's path) and full recomputation
    S1-S5 sanity checks, judged by the criteria fixed BEFORE the run (DECISIONS D-P5-1)
    P6  timing on the pilot + calibration series, extrapolated to W2-1 over all 250 series
    P7  figure F2 (zoom + full view), generated here, never hand-edited

The pilot's numbers are not research results (BRIEF3 §0).

Writes
    reports/w2_pilot.md                    the report (GENERATED, do not hand-edit)
    reports/w2_pilot.csv                   one row per (position, r, operator, detector)
    reports/w2_pilot_timing.csv            calibration timings (P6)
    reports/figures/w2_pilot/F2_{zoom,full}.{png,pdf}

Exit code: 0 all STOP-class checks (S1-S4) pass; 2 at least one failed (BRIEF3 §7 STOP);
1 on any error.

Usage:  python scripts/w2_pilot.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import REPO_ROOT, load_yaml, resolve  # noqa: E402
from src.data.ucr import load_series, parse_filename  # noqa: E402
from src.detectors import mp as mpmod  # noqa: E402
from src.detectors.matrixprofile import benchmark_window  # noqa: E402
from src.detectors.mp import MPScorer, classify_windows  # noqa: E402
from src.gate1 import PERCENTILE, normal_test_mask  # noqa: E402
from src.perturb.operators_simic import (OPERATORS, READ_EXTENT, UPSTREAM_COMMIT, Space,  # noqa: E402
                                         context_reconstruct_match, erase)
from src.runlog import env_header  # noqa: E402

GATE_DIR = REPO_ROOT / "results" / "gate1" / "MatrixProfile"
REPORT = REPO_ROOT / "reports" / "w2_pilot.md"
CSV_OUT = REPO_ROOT / "reports" / "w2_pilot.csv"
TIMING_OUT = REPO_ROOT / "reports" / "w2_pilot_timing.csv"
FIG_DIR = REPO_ROOT / "reports" / "figures" / "w2_pilot"
PREREG = "tsad-xai/docs/PREREG_W2.md"
CONST_OPS = ("Zero", "SampleMean", "OutOfDistHigh")
PROVENANCE_FILES = ("scripts/w2_pilot.py", "src/perturb/operators_simic.py", "src/detectors/mp.py",
                    "src/detectors/matrixprofile.py", "src/gate1.py", "configs/w2_pilot.yaml")
CSV_COLUMNS = [
    "series_id", "detector", "operator", "r", "r_over_m", "position", "a",
    "n_interior", "n_boundary", "n_untouched",
    "dS_interior_mean", "dS_boundary_mean", "max_before", "max_after", "p99_normal",
    "false_alarm", "false_alarm_before",
    "untouched_max_abs_delta", "untouched_frac_bitexact",
    "interior_min", "interior_max", "interior_n_unique",
    "inc_full_max_abs_diff", "inc_full_max_abs_diff_sq", "inc_full_D_at_worst",
    "t_incremental_s", "t_full_s", "t_operator_s",
    "fill_min", "fill_max", "cr_j", "cr_dist",
]


# --------------------------------------------------------------------------- helpers

def git(*args) -> str:
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True,
                          check=True).stdout.strip()


def prereg_commit() -> dict:
    """P0 must be committed and unmodified before the pilot runs."""
    log = git("log", "-1", "--format=%H %cI", "--", "docs/PREREG_W2.md")
    if not log:
        raise RuntimeError("docs/PREREG_W2.md is not committed (BRIEF3 P0 must precede the pilot)")
    if git("status", "--porcelain", "--", "docs/PREREG_W2.md"):
        raise RuntimeError("docs/PREREG_W2.md has uncommitted changes")
    h, t = log.split()
    return {"hash": h, "time": t, "path": PREREG}


def file_hashes() -> dict:
    return {p: hashlib.sha256((REPO_ROOT / p).read_bytes()).hexdigest()[:12] for p in PROVENANCE_FILES}


def md_table(header, rows):
    out = ["| " + " | ".join(map(str, header)) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def fmt(v, p=4):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    a = abs(v)
    return f"{v:.{p}g}" if (a != 0 and (a < 1e-3 or a >= 1e5)) else f"{v:.{p}f}"


def fulldata_dir() -> Path:
    paths = load_yaml("paths")
    a1 = json.loads(resolve(paths["ucr"]["a1_result"]).read_text())
    if a1["stop"]:
        raise RuntimeError("01_download.py recorded STOP")
    return Path(a1["anchors"]["official"]) / paths["ucr"]["fulldata_subdir"]


def load_checked(row: dict, ddir: Path) -> np.ndarray:
    p = ddir / row["filename"]
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    if digest != row["sha256"]:
        raise RuntimeError(f"{p.name}: sha256 {digest} != manifest {row['sha256']}")
    x = load_series(p)
    x.setflags(write=False)
    if x.size != int(row["length"]):
        raise RuntimeError(f"{p.name}: length {x.size} != manifest {row['length']}")
    return x


# --------------------------------------------------------------------------- P1

def normal_pieces(n, te, s0, s1, m, variant="B"):
    """Certain-normal test pieces, half-open. Variant B is the rule (D-P1-1); A starts at te."""
    lo = te + m if variant == "B" else te
    return [(lo, s0 - m), (s1 + m, n)]


def select_series(cfg: dict, manifest: pd.DataFrame, det: pd.DataFrame) -> tuple[pd.Series, dict]:
    sel = cfg["selection"]
    d = det[(det.fit_on == sel["gate_fit_on"]) & (det["mode"] == sel["gate_mode"])].copy()
    if len(d) != 250:
        raise RuntimeError(f"gate-1 MP {sel['gate_fit_on']}/{sel['gate_mode']}: {len(d)} rows, expected 250")
    steps = [("gate-1 MP primary condition", len(d))]
    d = d[d.p1_bw.astype(bool)]
    steps.append(("1. P1 (buffer = m) passed", len(d)))
    d = d[d.domain_goswami.isin(sel["domains"])]
    steps.append((f"2. domain ∈ {{{', '.join(sel['domains'])}}}", len(d)))
    k = int(sel["min_normal_len_windows"])

    def piece_stats(r, variant):
        ps = [max(0, b - a) for a, b in normal_pieces(r.n, r.train_end, r.start0, r.stop0, r.window, variant)]
        return max(ps), sum(ps)

    for v in ("A", "B"):
        st = d.apply(lambda r: piece_stats(r, v), axis=1, result_type="expand")
        d[f"longest_{v}"], d[f"total_{v}"] = st[0], st[1]
    ok = d[d.longest_B >= k * d.window]
    steps.append((f"3. longest certain-normal piece ≥ {k}m", len(ok)))
    if ok.empty:
        raise RuntimeError("no series satisfies the selection rule (BRIEF3 P1: STOP)")
    ok = ok.sort_values(["n", "num"])
    chosen = ok.iloc[0]
    steps.append(("4. shortest n, then smallest num", 1))
    robust = []
    for label, col in [("longest piece, from train_end + m (rule)", "longest_B"),
                       ("longest piece, from train_end", "longest_A"),
                       ("total length, from train_end + m", "total_B"),
                       ("total length, from train_end", "total_A")]:
        alt = d[d[col] >= k * d.window].sort_values(["n", "num"])
        robust.append((label, int(alt.iloc[0].num) if len(alt) else None, len(alt)))
    return chosen, {"steps": steps, "robust": robust}


# --------------------------------------------------------------------------- positions

def draw_positions(cfg, pieces, m, r_list):
    r_max = max(r_list)
    left = max([m - 1, 1] + [f(r_max, m)[0] for f in READ_EXTENT.values()])
    right = max([m - 1, 1] + [f(r_max, m)[1] for f in READ_EXTENT.values()])
    valid = np.concatenate([np.arange(p0 + left, p1 - r_max - right + 1) for p0, p1 in pieces
                            if p1 - r_max - right + 1 > p0 + left] or [np.empty(0, int)])
    if valid.size == 0:
        raise RuntimeError("no position fits the footprint in any certain-normal piece")
    rng = np.random.default_rng(int(cfg["position_seed"]))
    chosen = []
    for _ in range(int(cfg["position_max_draws"])):
        a = int(valid[rng.integers(valid.size)])
        fp = (a - left, a + r_max + right)
        if all(fp[1] <= c[0] or fp[0] >= c[1] for _, c in chosen):
            chosen.append((a, fp))
            if len(chosen) == int(cfg["n_positions"]):
                break
    else:
        raise RuntimeError(f"could not place {cfg['n_positions']} disjoint positions")
    chosen.sort()
    return chosen, (left, right)


# --------------------------------------------------------------------------- P5

def p99_threshold(score, n, te, s0, s1, m):
    return float(np.percentile(score[normal_test_mask(n, te, s0, s1, m)], PERCENTILE))


def run_grid(cfg, x, te, s0, s1, m, positions, scorers, base, p99, series_id):
    space = Space(bool(cfg["operator_space"]["standardize"]), cfg["operator_space"]["reference"])
    n = x.size
    r_list = [int(math.floor(f * m)) for f in cfg["r_over_m"]]
    rows, keep = [], {}
    for pi, (a, _) in enumerate(positions):
        for f, r in zip(cfg["r_over_m"], r_list):
            wc = classify_windows(a, r, m, n)
            aff = wc.affected
            for op in cfg["operators"]:
                t0 = time.perf_counter()
                x_new, info = erase(x, op, a, r, train_end=te, m=m, space=space, seed=int(cfg["noise_seed"]))
                t_op = time.perf_counter() - t0
                for k in cfg["detectors"]:
                    sc = scorers[k]
                    t0 = time.perf_counter()
                    inc = sc.incremental(x_new, x, base[k], a, r)
                    t_inc = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    full = sc.profile(x_new)
                    t_full = time.perf_counter() - t0
                    b = base[k]
                    diff = np.abs(inc - full)
                    un_d = np.abs(full[wc.untouched] - b[wc.untouched])
                    inter = inc[wc.interior]
                    rows.append({
                        "series_id": series_id, "detector": k, "operator": op, "r": r, "r_over_m": f,
                        "position": pi, "a": a,
                        "n_interior": wc.interior.size, "n_boundary": wc.boundary.size,
                        "n_untouched": wc.untouched.size,
                        "dS_interior_mean": float(np.mean(inter - b[wc.interior])) if inter.size else float("nan"),
                        "dS_boundary_mean": float(np.mean(inc[wc.boundary] - b[wc.boundary])),
                        "max_before": float(b[aff].max()), "max_after": float(inc[aff].max()),
                        "p99_normal": p99[k],
                        "false_alarm": bool(inc[aff].max() > p99[k]),
                        "false_alarm_before": bool(b[aff].max() > p99[k]),
                        "untouched_max_abs_delta": float(un_d.max()),
                        "untouched_frac_bitexact": float(np.mean(un_d == 0)),
                        "interior_min": float(inter.min()) if inter.size else float("nan"),
                        "interior_max": float(inter.max()) if inter.size else float("nan"),
                        "interior_n_unique": int(np.unique(inter).size),
                        "inc_full_max_abs_diff": float(diff.max()),
                        "inc_full_D_at_worst": float(full[int(np.argmax(diff))]),
                        "inc_full_max_abs_diff_sq": float(np.abs(inc ** 2 - full ** 2).max()),
                        "t_incremental_s": t_inc, "t_full_s": t_full, "t_operator_s": t_op,
                        "fill_min": float(x_new[a:a + r].min()), "fill_max": float(x_new[a:a + r].max()),
                        "cr_j": info.get("cr_j", ""), "cr_dist": info.get("cr_dist", ""),
                    })
                    keep[(pi, r, op, k)] = (inc, full)
                keep[(pi, r, op, "x")] = x_new
    return rows, keep, r_list


def sanity(cfg, rows, keep, m, r_list, n_pos):
    df = pd.DataFrame(rows)
    out = {}
    # S1: untouched windows under full recomputation (researcher 2026-09-23: atol, D-P4-1)
    s1 = df.untouched_max_abs_delta.max()
    out["S1"] = {"pass": bool(s1 <= cfg["s1_atol"]),
                 "detail": {k: (float(g.untouched_max_abs_delta.max()), float(g.untouched_frac_bitexact.mean()),
                                float(g.untouched_frac_bitexact.min()))
                            for k, g in df.groupby("detector")}}
    # S4: incremental == full
    worst = df.loc[df.inc_full_max_abs_diff.idxmax()]
    out["S4"] = {"pass": bool(df.inc_full_max_abs_diff.max() <= cfg["s4_atol"]),
                 "n_fail": int((df.inc_full_max_abs_diff > cfg["s4_atol"]).sum()),
                 "fails": df[df.inc_full_max_abs_diff > cfg["s4_atol"]][
                     ["detector", "operator", "r", "position", "inc_full_max_abs_diff",
                      "inc_full_max_abs_diff_sq", "inc_full_D_at_worst"]].to_dict("records"),
                 "worst": worst.to_dict(), "max_sq": float(df.inc_full_max_abs_diff_sq.max())}
    # S2 (Z) / S3 (R) / S5 (Zero ≡ SampleMean), r >= m only
    s2, s3, s5 = [], [], []
    for pi in range(n_pos):
        for r in r_list:
            if r >= m:
                ivals = {}
                for k in ("Z", "R"):
                    for op in CONST_OPS:
                        inc, _ = keep[(pi, r, op, k)]
                        a = int(df[(df.position == pi)].a.iloc[0])
                        ivals[(k, op)] = inc[a:a + r - m + 1]
                z_all = np.concatenate([ivals[("Z", op)] for op in CONST_OPS])
                zu = np.unique(z_all)
                s2.append({"position": pi, "r": r, "n_unique": zu.size, "value": float(zu[0]),
                           "equals_sqrt_m": bool(zu.size == 1 and zu[0] == np.sqrt(m)),
                           "pass": bool(zu.size == 1)})
                pairs = []
                for p, q in (("Zero", "OutOfDistHigh"), ("SampleMean", "OutOfDistHigh")):
                    u, v = ivals[("R", p)], ivals[("R", q)]
                    pairs.append((p, q, bool(not np.array_equal(u, v)), float(np.min(np.abs(u - v)))))
                s3.append({"position": pi, "r": r, "pairs": pairs, "pass": all(pp[2] for pp in pairs)})
            xs = keep[(pi, r, "Zero", "x")], keep[(pi, r, "SampleMean", "x")]
            x_eq = bool(np.array_equal(*xs))
            prof_eq = {k: bool(np.array_equal(keep[(pi, r, "Zero", k)][0], keep[(pi, r, "SampleMean", k)][0])
                               and np.array_equal(keep[(pi, r, "Zero", k)][1], keep[(pi, r, "SampleMean", k)][1]))
                       for k in ("Z", "R")}
            s5.append({"position": pi, "r": r, "x_equal": x_eq, **prof_eq,
                       "max_fill_diff": float(np.abs(xs[0] - xs[1]).max())})
    out["S2"] = {"pass": all(s["pass"] for s in s2), "rows": s2}
    out["S3"] = {"pass": all(s["pass"] for s in s3), "rows": s3}
    out["S5"] = {"pass": all(s["x_equal"] and s["Z"] and s["R"] for s in s5), "rows": s5}
    return out


# --------------------------------------------------------------------------- P6

def median_time(fn, repeats):
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def calibrate(cfg, manifest, det_rows, ddir, pilot_num, rf):
    """Time the incremental kernel (one AB-join of the r + 2m - 2 affected points against the
    training prefix) and ContextReconstruct's search on real series of very different
    training lengths. Content does not change stumpy's cost; lengths do."""
    tc = cfg["timing"]
    d = det_rows.set_index("num")
    by_n = d.sort_values(["n", "train_end"], ascending=False)
    top = list(by_n.index[:int(tc["n_top"])])
    tes = d.train_end.sort_values()
    quant = []
    for q in tc["quantile_series"]:
        target = tes.quantile(q)
        for num in (tes - target).abs().sort_values().index:
            if num not in top and num not in quant and num != pilot_num:
                quant.append(int(num))
                break
    series = [(int(pilot_num), "pilot")] + [(int(s), "top") for s in top] + [(q, "quantile") for q in quant]
    rows = []
    for num, role in series:
        row = manifest.loc[num]
        x = load_checked(row, ddir)
        te, m = int(d.loc[num].train_end), int(d.loc[num].window)
        train = np.ascontiguousarray(x[:te])
        a = te + (x.size - te) // 2
        for f in rf:
            r = int(math.floor(f * m))
            if a - m + 1 < te or a + r + m > x.size:
                raise RuntimeError(f"{num}: timing position does not fit")
            seg = np.ascontiguousarray(x[a - m + 1:a + r + m - 1])
            for k in ("Z", "R"):
                t = median_time(lambda: mpmod._mp(k, seg, m, train), int(tc["repeats"]))
                rows.append({"num": num, "role": role, "n": x.size, "train_end": te, "m": m, "r": r,
                             "kind": k, "what": "incremental", "n_A": seg.size - m + 1, "n_B": te - m + 1,
                             "seconds": t})
            t = median_time(lambda: context_reconstruct_match(x, a, a + r, m=m, train_end=te), int(tc["repeats"]))
            rows.append({"num": num, "role": role, "n": x.size, "train_end": te, "m": m, "r": r,
                         "kind": "-", "what": "cr_search", "n_A": m, "n_B": te - m - r + 1, "seconds": t})
    return pd.DataFrame(rows)


def fit_nnls(X, y):
    from scipy.optimize import nnls
    coef, _ = nnls(X, y)
    pred = X @ coef
    return coef, pred


def extrapolate(cfg, timing, pilot_rows, det_rows, pilot_num):
    """W2-1 wall time for all 250 series, single process (stumpy parallelises internally)."""
    tc = cfg["timing"]
    K, n_ops, rf = int(tc["K"]), len(cfg["operators"]), cfg["r_over_m"]
    models, fit_quality = {}, {}
    for k in ("Z", "R"):
        t = timing[(timing.what == "incremental") & (timing.kind == k)]
        X = np.c_[np.ones(len(t)), t.n_B, t.n_A * t.n_B]
        coef, pred = fit_nnls(X, t.seconds.to_numpy())
        models[k] = coef
        fit_quality[k] = float(np.max(np.abs(pred - t.seconds) / t.seconds))
    t = timing[timing.what == "cr_search"]
    X = np.c_[np.ones(len(t)), t.n_B]
    models["cr"], pred = fit_nnls(X, t.seconds.to_numpy())
    fit_quality["cr"] = float(np.max(np.abs(pred - t.seconds) / t.seconds))
    pr = pd.DataFrame(pilot_rows)
    t_other_op = float(pr[pr.operator != "ContextReconstruct"].groupby(["position", "r", "operator"])
                       .t_operator_s.first().mean())
    # full scoring of the unperturbed series, once per detector per series: gate-1 measured Z
    # (score_s, T_A = x, 4 workers in parallel); R scaled by the pilot's R/Z full-time ratio
    rz = float(pr[pr.detector == "R"].t_full_s.mean() / pr[pr.detector == "Z"].t_full_s.mean())
    out = []
    for num, row in det_rows.set_index("num").iterrows():
        te, m, n = int(row.train_end), int(row.window), int(row.n)
        nB = te - m + 1
        t_inc = 0.0
        t_ops = 0.0
        for f in rf:
            r = int(math.floor(f * m))
            nA = r + m - 1
            for k in ("Z", "R"):
                c = models[k]
                t_inc += K * n_ops * (c[0] + c[1] * nB + c[2] * nA * nB)
            c = models["cr"]
            t_ops += K * (c[0] + c[1] * (te - m - r + 1)) + K * (n_ops - 1) * t_other_op
        t_base = float(row.score_s) * (1 + rz)
        out.append({"num": int(num), "n": n, "train_end": te, "m": m, "t_incremental_h": t_inc / 3600,
                    "t_operators_h": t_ops / 3600, "t_baseline_h": t_base / 3600,
                    "t_total_h": (t_inc + t_ops + t_base) / 3600})
    est = pd.DataFrame(out)
    return est, models, fit_quality, rz, t_other_op


# --------------------------------------------------------------------------- P7

def figure(cfg, x, te, s0, s1, m, positions, keep, scorers, base, p99, name, r_list):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fc = cfg["figure"]
    pi = int(fc["position_index"])
    r = int(math.floor(fc["r_over_m"] * m))
    if r not in r_list:
        raise RuntimeError(f"figure r={r} is not a pilot length")
    a = positions[pi][0]
    ops = fc["operators"]
    n = x.size
    # Okabe-Ito vermillion / blue (CVD-safe) + distinct dash patterns (BRIEF3 §9)
    style = {"orig": dict(color="#000000", ls="-", lw=1.4, label="original"),
             ops[0]: dict(color="#D55E00", ls="--", lw=1.6, label=f"after {ops[0]}"),
             ops[1]: dict(color="#0072B2", ls="-.", lw=1.6, label=f"after {ops[1]}")}
    plt.rcParams.update({"font.size": fc["font_pt"], "axes.titlesize": fc["font_pt"] + 1,
                         "legend.fontsize": fc["font_pt"] - 1, "pdf.fonttype": 42})
    written = []
    for view in ("zoom", "full"):
        fig, axes = plt.subplots(3, 1, figsize=tuple(fc["size_in"]), sharex=True, constrained_layout=True)
        if view == "zoom":
            c = a + r / 2
            lo, hi = int(max(0, c - fc["zoom_half_width_m"] * m)), int(min(n, c + fc["zoom_half_width_m"] * m))
        else:
            lo, hi = 0, n
        idx = np.arange(lo, hi)
        ax = axes[0]
        ax.plot(idx, x[lo:hi], **style["orig"])
        seg = np.arange(max(0, a - 1), min(n, a + r + 1))
        for op in ops:
            ax.plot(seg, keep[(pi, r, op, "x")][seg], **style[op])
        ax.set_ylabel("value")
        for axx in axes:
            axx.axvspan(0, te, color="0.6", alpha=0.25, lw=0)
            axx.axvspan(s0, s1, color="#D55E00", alpha=0.25, lw=0)
            axx.axvspan(a, a + r, color="#F0E442", alpha=0.45, lw=0)
            axx.set_xlim(lo, hi - 1)
            axx.grid(axis="y", color="0.9", lw=0.6)
            for sp in ("top", "right"):
                axx.spines[sp].set_visible(False)
        for axx, k, lab in ((axes[1], "Z", "detector Z score\n(z-normalised MP)"),
                            (axes[2], "R", "detector R score\n(non-normalised MP)")):
            sc = scorers[k]
            axx.plot(idx, sc.pad(base[k], n)[lo:hi], **style["orig"])
            for op in ops:
                axx.plot(idx, sc.pad(keep[(pi, r, op, k)][0], n)[lo:hi], **style[op])
            axx.axhline(p99[k], color="0.35", ls=":", lw=1.3, label="P99 of normal test score")
            axx.set_ylabel(lab)
        axes[2].set_xlabel("time index (score at window start + m//2, as in gate 1)")
        from matplotlib.patches import Patch
        h0, l0 = axes[0].get_legend_handles_labels()
        h0 += axes[1].get_legend_handles_labels()[0][-1:]
        l0 += axes[1].get_legend_handles_labels()[1][-1:]
        # shading entries only for the spans visible in this view
        for (s_lo, s_hi), patch, lab in (((0, te), Patch(color="0.6", alpha=0.25), "training region"),
                                         ((s0, s1), Patch(color="#D55E00", alpha=0.25), "labelled anomaly"),
                                         ((a, a + r), Patch(color="#F0E442", alpha=0.45), "erased region")):
            if s_hi > lo and s_lo < hi:
                h0.append(patch)
                l0.append(lab)
        fig.legend(h0, l0, ncol=len(l0), loc="outside lower center", frameon=False)
        fig.suptitle(f"F2 — erasing a normal segment: {name}, m = {m}, r = 2m = {r}, "
                     f"erased [{a}, {a + r})" + ("" if view == "zoom" else " (full series)"))
        for ext in ("png", "pdf"):
            p = FIG_DIR / f"F2_{view}.{ext}"
            fig.savefig(p, dpi=fc["dpi"])
            written.append(p)
        plt.close(fig)
    return written


# --------------------------------------------------------------------------- main

def main() -> int:
    started = datetime.now(timezone.utc)
    header = env_header()
    print(header)
    cfg = load_yaml("w2_pilot")
    pre = prereg_commit()
    if datetime.fromisoformat(pre["time"]) >= started:
        raise RuntimeError("PREREG commit is not earlier than this run")
    import numba
    import stumpy
    runinfo = {"started": started.isoformat(timespec="seconds"), "head": git("rev-parse", "HEAD"),
               "dirty": [ln for ln in git("status", "--porcelain", "--", *PROVENANCE_FILES).splitlines()],
               "hashes": file_hashes(), "stumpy": stumpy.__version__,
               "numba_threads": int(numba.config.NUMBA_NUM_THREADS)}

    manifest = pd.read_csv(resolve(load_yaml("paths")["manifest"]), dtype={"sha256": str}).set_index("num",
                                                                                                    drop=False)
    det = pd.read_csv(GATE_DIR / "detection.csv")
    chosen, selinfo = select_series(cfg, manifest, det)
    num = int(chosen.num)
    row = manifest.loc[num]
    meta = parse_filename(row["filename"])
    ddir = fulldata_dir()
    x = load_checked(row, ddir)
    n, te, s0, s1, m = x.size, int(meta.train_end_raw), int(row.start0), int(row.stop0), int(chosen.window)
    if (te, s0, s1, n) != (int(chosen.train_end), int(chosen.start0), int(chosen.stop0), int(chosen.n)):
        raise RuntimeError("manifest and gate-1 detection.csv disagree on the layout")
    if benchmark_window(x, int(load_yaml("detectors")["matrixprofile"]["periodicity"])) != m:
        raise RuntimeError("recomputed benchmark window != gate-1 window")
    print(f"P1: series {num:03d} {row['name']} n={n} train_end={te} anomaly=[{s0},{s1}) m={m}")

    pieces = normal_pieces(n, te, s0, s1, m)
    r_list = [int(math.floor(f * m)) for f in cfg["r_over_m"]]
    positions, (ext_l, ext_r) = draw_positions(cfg, pieces, m, r_list)
    print(f"positions: {[p[0] for p in positions]}")

    scorers = {k: MPScorer(k, m, x[:te]) for k in cfg["detectors"]}
    # warm-up (numba JIT), excluded from all timings
    for k in scorers:
        sc = MPScorer(k, m, x[:te][:500])
        sc.profile(x[:600])
    context_reconstruct_match(x, positions[0][0], positions[0][0] + r_list[0], m=m, train_end=te)
    base, p99, t_base = {}, {}, {}
    for k, sc in scorers.items():
        t0 = time.perf_counter()
        base[k] = sc.profile(x)
        t_base[k] = time.perf_counter() - t0
        p99[k] = p99_threshold(sc.pad(base[k], n), n, te, s0, s1, m)
    stored = np.load(GATE_DIR / "scores" / f"{num:03d}_train_prefix_ab_join_seed0.npy")
    z_pad = scorers["Z"].pad(base["Z"], n)
    gate_check = {"stored_equal": bool(np.array_equal(z_pad, stored)),
                  "stored_max_abs_diff": float(np.abs(z_pad - stored).max()),
                  "thr_equal": bool(p99["Z"] == float(chosen.thr_bw)), "thr_gate": float(chosen.thr_bw)}
    if gate_check["stored_max_abs_diff"] > cfg["s4_atol"] or abs(p99["Z"] - gate_check["thr_gate"]) > cfg["s4_atol"]:
        raise RuntimeError(f"Z does not reproduce the gate-1 score/threshold: {gate_check}")

    print("P5: running grid ...", flush=True)
    rows, keep, _ = run_grid(cfg, x, te, s0, s1, m, positions, scorers, base, p99, num)
    checks = sanity(cfg, rows, keep, m, r_list, len(positions))
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {CSV_OUT} ({len(rows)} rows)")

    print("P7: figure ...", flush=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figs = figure(cfg, x, te, s0, s1, m, positions, keep, scorers, base, p99, row["name"], r_list)

    print("P6: timing calibration ...", flush=True)
    d_primary = det[(det.fit_on == cfg["selection"]["gate_fit_on"]) & (det["mode"] == cfg["selection"]["gate_mode"])]
    timing = calibrate(cfg, manifest, d_primary, ddir, num, cfg["r_over_m"])
    timing.to_csv(TIMING_OUT, index=False)
    est, models, fitq, rz, t_other = extrapolate(cfg, timing, rows, d_primary, num)

    write_report(cfg, header, runinfo, pre, selinfo, chosen, row, x, te, s0, s1, m, pieces, positions,
                 (ext_l, ext_r), r_list, scorers, base, p99, t_base, gate_check, rows, checks, timing,
                 est, models, fitq, rz, t_other, figs)
    stop = [s for s in ("S1", "S2", "S3", "S4") if not checks[s]["pass"]]
    print(f"wrote {REPORT}")
    if stop:
        print(f"STOP: sanity check(s) failed: {stop} (BRIEF3 §7)")
        return 2
    return 0


# --------------------------------------------------------------------------- report

def write_report(cfg, header, runinfo, pre, selinfo, chosen, row, x, te, s0, s1, m, pieces, positions,
                 ext, r_list, scorers, base, p99, t_base, gate_check, rows, checks, timing, est, models,
                 fitq, rz, t_other, figs):
    df = pd.DataFrame(rows)
    rel = lambda p: str(Path(p).relative_to(REPO_ROOT))  # noqa: E731
    L = []
    A = L.append
    A("# W2-0 single-series pilot (BRIEF3)\n")
    A(f"GENERATED by scripts/w2_pilot.py at {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
      "Do not hand-edit. **Pilot numbers are not research results and are not used for any claim "
      "(BRIEF3 §0).**\n")
    A(f"`{header}`\n")
    A(md_table(["item", "value"], [
        ["PREREG (P0) commit", f"`{pre['hash']}` at {pre['time']} (`{pre['path']}`)"],
        ["pilot run started (UTC)", runinfo["started"]],
        ["code HEAD at run", f"`{runinfo['head']}`" + (f"; uncommitted: {', '.join(runinfo['dirty'])}"
                                                       if runinfo["dirty"] else "")],
        ["source SHA256 (12)", "<br>".join(f"`{k}` {v}" for k, v in runinfo["hashes"].items())],
        ["stumpy / numba threads", f"{runinfo['stumpy']} / {runinfo['numba_threads']}"],
        ["upstream PMs", f"Šimić et al. `utils/subsequence_perturbation.py` @ `{UPSTREAM_COMMIT[:8]}`"],
    ]))
    A("")

    # 1
    A("## 1. Series selection (P1)\n")
    A("Rule, coded in `select_series` before the series was known to the code "
      "(`configs/w2_pilot.yaml` `selection`, DECISIONS D-P1-1). Certain-normal test region = "
      "`[train_end + m, start0 − m) ∪ [stop0 + m, n)`.\n")
    A(md_table(["step", "series left"], selinfo["steps"]))
    A("")
    A(md_table(["item", "value"], [
        ["selected", f"**{int(chosen.num):03d} {row['name']}**"],
        ["domain / content_group", f"{chosen.domain_goswami} / {chosen.content_group}"],
        ["n, train_end, anomaly [start0, stop0)", f"{len(x)}, {te}, [{s0}, {s1})"],
        ["m (gate-1 window, find_length_rank)", m],
        ["gate-1 P1 (buffer = m)", f"max_anom {fmt(chosen.max_anom)} > thr_bw {fmt(chosen.thr_bw)}"],
        ["certain-normal pieces", ", ".join(f"[{a}, {b}) = {b - a} pts = {(b - a) / m:.1f}m" for a, b in pieces)],
        ["training windows that are constant (D12-a)", int(chosen.n_const_train)],
    ]))
    A("\nRobustness of the pick to the reading of rule 3 (BRIEF3 does not say which length):\n")
    A(md_table(["reading of rule 3", "selected num", "series passing"],
               [[a, f"{b:03d}" if b else "—", c] for a, b, c in selinfo["robust"]]))
    A("")

    # 2
    A("## 2. Setup\n")
    mu, sd = float(np.mean(x[:te])), float(np.std(x[:te]))
    z_tr = (x[:te] - mu) / sd
    A(md_table(["item", "value"], [
        ["r = ⌊f·m⌋", ", ".join(f"{f}m → {r}" for f, r in zip(cfg["r_over_m"], r_list))],
        ["positions (a, sorted; seed " + str(cfg["position_seed"]) + ")",
         ", ".join(f"#{i}: a = {a} (footprint [{fp[0]}, {fp[1]}))" for i, (a, fp) in enumerate(positions))],
        ["footprint", f"[a − {ext[0]}, a + r_max + {ext[1]}): affected windows plus every point an "
                      "operator reads, for r_max; must lie in one certain-normal piece; footprints disjoint"],
        ["operator space (D-P3-1)", f"standardised by x[:train_end]: μ = {mu:.6g}, σ = {sd:.6g}"],
        ["detectors", "Z = `stumpy.stump`, R = `stumpy.aamp`; T_A = x, T_B = x[:train_end], "
                      "ignore_trivial=False (D-P2-1)"],
        ["P99 of normal test score (buffer = m)", f"Z {p99['Z']:.6g}, R {p99['R']:.6g}"],
        ["Z reproduces gate 1", f"padded score == stored gate-1 file: {fmt(gate_check['stored_equal'])} "
                                f"(max |Δ| {gate_check['stored_max_abs_diff']:.3g}); P99 == thr_bw "
                                f"{gate_check['thr_gate']:.6g}: {fmt(gate_check['thr_equal'])}"],
        ["full scoring of the unperturbed series", f"Z {t_base['Z']:.3f} s, R {t_base['R']:.3f} s"],
    ]))
    A("\nFill values in raw units (this series):\n")
    A(md_table(["operator", "fill"], [
        ["Zero", f"μ = {mu:.6g} (0 in standardised units)"],
        ["SampleMean", f"mean(z_train)·σ + μ; mean(z_train) = {np.mean(z_tr):.3g}"],
        ["OutOfDistHigh", f"{max(abs(z_tr.max()), abs(z_tr.min())) * 100:.6g}σ + μ = "
                          f"{max(abs(z_tr.max()), abs(z_tr.min())) * 100 * sd + mu:.6g}"],
        ["Inverse", f"(max(z_train) − z)·σ + μ, max(z_train) = {z_tr.max():.6g}"],
        ["UniformNoise100", f"μ + σ·U(−1, 1), RandomState({cfg['noise_seed']}) over the whole series, sliced"],
        ["LinearInterpolation", "straight line between x[a−1] and x[a+r]"],
        ["NearestNeighborWindow (이웃 붙이기)", "x[a−⌈r/2⌉ : a] followed by x[a+r : a+r+⌊r/2⌋]"],
        ["ContextReconstruct", "offset-aligned continuation of the nearest training match of x[a−m:a]"],
    ]))
    cr = df[(df.operator == "ContextReconstruct") & (df.detector == "Z")]
    A("\nContextReconstruct matches:\n")
    A(md_table(["position", "r", "j", "z-norm distance of the context match"],
               [[int(t.position), int(t.r), int(t.cr_j), fmt(float(t.cr_dist))] for t in cr.itertuples()]))
    A("")

    # 3
    A("## 3. Sanity checks (criteria fixed before the run, DECISIONS D-P5-1)\n")
    s1, s2, s3, s4, s5 = (checks[k] for k in ("S1", "S2", "S3", "S4", "S5"))
    verdict = lambda ok: "**PASS**" if ok else "**FAIL → STOP**"  # noqa: E731
    s1d = "; ".join(f"{k}: max {v[0]:.3g}, bit-exact windows {100 * v[1]:.2f}% mean / {100 * v[2]:.2f}% min"
                    for k, v in s1["detail"].items())
    s2vals = sorted({(s["value"], s["equals_sqrt_m"]) for s in s2["rows"]})
    s3min = min(pp[3] for s in s3["rows"] for pp in s["pairs"]) if s3["rows"] else float("nan")
    A(md_table(["ID", "check", "criterion", "observed", "verdict"], [
        ["S1", "untouched windows, full recomputation (Z, R)", f"max |Δ| ≤ {cfg['s1_atol']:g}", s1d,
         verdict(s1["pass"])],
        ["S2", "Z interior, Zero / SampleMean / OutOfDistHigh, r ≥ m", "one value across the three",
         "; ".join(f"value {v:.10g}, = √m exactly: {fmt(e)}" for v, e in s2vals)
         + f" (√m = {np.sqrt(m):.10g}; {sum(s['pass'] for s in s2['rows'])}/{len(s2['rows'])} cells)",
         verdict(s2["pass"])],
        ["S3", "R interior, pairs with different fills: (Zero, OOD), (SampleMean, OOD), r ≥ m",
         "arrays differ", f"{sum(s['pass'] for s in s3['rows'])}/{len(s3['rows'])} cells; "
                          f"min |Δ| over windows {s3min:.4g}", verdict(s3["pass"])],
        ["S4", "incremental == full, all 192 runs", f"max |Δ| ≤ {cfg['s4_atol']:g}",
         f"max |Δ| {s4['worst']['inc_full_max_abs_diff']:.3g} ({s4['n_fail']} runs over); "
         f"on D²: max {s4['max_sq']:.3g}", verdict(s4["pass"])],
        ["S5", "Zero ≡ SampleMean (researcher 2026-09-23)", "x′ and Z, R profiles bit-identical",
         f"{sum(s['x_equal'] and s['Z'] and s['R'] for s in s5['rows'])}/{len(s5['rows'])} cells; "
         f"max fill diff {max(s['max_fill_diff'] for s in s5['rows']):.3g}",
         "PASS" if s5["pass"] else "FAIL (reported; not a BRIEF3 STOP)"],
    ]))
    A("")
    if not s4["pass"]:
        A("### S4 failures\n")
        A("Runs where incremental and full recomputation differ by more than the tolerance. "
          "`|ΔD²|` is the same difference on the squared distance 2m(1 − ρ); `D there` is the full "
          "recomputation's value at the window with the largest |ΔD|.\n")
        A(md_table(["detector", "operator", "r", "position", "max |ΔD|", "max |ΔD²|", "D there"],
                   [[f["detector"], f["operator"], f["r"], f["position"], f"{f['inc_full_max_abs_diff']:.3g}",
                     f"{f['inc_full_max_abs_diff_sq']:.3g}", f"{f['inc_full_D_at_worst']:.3g}"]
                    for f in s4["fails"]]))
        A("")

    # 4
    A("## 4. Per-condition summary (mean over the 3 positions)\n")
    A("`false alarm` = affected-window max after erasure > P99 (count of 3 positions); "
      "`before` = the same on the unperturbed series. ΔS = mean change over interior / boundary windows.\n")
    for k in cfg["detectors"]:
        A(f"### Detector {k} (P99 = {p99[k]:.4g})\n")
        tab = []
        for op in cfg["operators"]:
            for r in r_list:
                g = df[(df.detector == k) & (df.operator == op) & (df.r == r)]
                tab.append([cfg["operator_labels"].get(op, op), r, f"{int(g.false_alarm.sum())}/3",
                            f"{int(g.false_alarm_before.sum())}/3", fmt(g.dS_interior_mean.mean()),
                            fmt(g.dS_boundary_mean.mean()), fmt(g.max_after.mean()),
                            f"{fmt(g.interior_min.min())} – {fmt(g.interior_max.max())}"])
        A(md_table(["operator", "r", "false alarm", "before", "ΔS interior", "ΔS boundary",
                    "max after", "interior range"], tab))
        A("")

    # 5
    A("## 5. Timing and the W2-1 estimate (P6)\n")
    pr = df
    A(md_table(["detector", "incremental, mean per call (s)", "full recomputation, mean per call (s)",
                "ratio"],
               [[k, f"{pr[pr.detector == k].t_incremental_s.mean():.4f}", f"{pr[pr.detector == k].t_full_s.mean():.4f}",
                 f"{pr[pr.detector == k].t_full_s.mean() / pr[pr.detector == k].t_incremental_s.mean():.1f}×"]
                for k in cfg["detectors"]]))
    A("\nBy r (pilot):\n")
    A(md_table(["detector", "r", "incremental (s)", "full (s)"],
               [[k, r, f"{g.t_incremental_s.mean():.4f}", f"{g.t_full_s.mean():.4f}"]
                for (k, r), g in pr.groupby(["detector", "r"])]))
    tc = cfg["timing"]
    n_runs = 250 * int(tc["K"]) * len(r_list) * len(cfg["operators"]) * len(cfg["detectors"])
    A(f"\n**Extrapolation method.** W2-1 = 250 series × K {tc['K']} × {len(r_list)} lengths × "
      f"{len(cfg['operators'])} erasers × {len(cfg['detectors'])} detectors = **{n_runs:,} incremental runs** "
      "(BRIEF3 §8 counted 7 erasers → 140,000; ContextReconstruct makes it 8). "
      "Assumed cost of one incremental run = one stumpy AB-join of the n_A = r + m − 1 affected windows "
      "against the n_B = train_end − m + 1 reference windows: "
      "`t = c0 + c1·n_B + c2·n_A·n_B` (c1: per-call preprocessing of T_B, c2: the diagonal traversal), "
      "fitted by non-negative least squares on direct timings of the pilot series, the "
      f"{tc['n_top']} longest series and {len(tc['quantile_series'])} series at training-length quantiles "
      f"(`{rel(TIMING_OUT)}`, median of {tc['repeats']} calls each). ContextReconstruct's search: "
      "`t = c0 + c1·n_B` (stumpy.mass). Other erasers: pilot mean per call "
      f"({t_other * 1e3:.3f} ms). Plus one full scoring per detector per series for the unperturbed profile: "
      "gate-1 measured Z (`score_s`, 4 workers in parallel) and R = Z × the pilot's R/Z full-time ratio "
      f"({rz:.2f}). Single process; stumpy parallelises internally on {runinfo['numba_threads']} numba threads.\n")
    A(md_table(["model", "c0 (s)", "c1 (s per n_B)", "c2 (s per n_A·n_B)", "max relative residual"],
               [[k, f"{models[k][0]:.3g}", f"{models[k][1]:.3g}", f"{models[k][2]:.3g}" if len(models[k]) > 2 else "—",
                 f"{100 * fitq[k]:.1f}%"] for k in ("Z", "R", "cr")]))
    tot = est.t_total_h.sum()
    A("\n" + md_table(["component", "hours"], [
        ["incremental scoring (Z + R)", f"{est.t_incremental_h.sum():.1f}"],
        ["erasers (incl. ContextReconstruct search)", f"{est.t_operators_h.sum():.1f}"],
        ["unperturbed full scoring (Z + R)", f"{est.t_baseline_h.sum():.1f}"],
        ["**total**", f"**{tot:.1f}**"],
    ]))
    A("\nFive series with the largest predicted time:\n")
    top = est.sort_values("t_total_h", ascending=False).head(5)
    A(md_table(["num", "n", "train_end", "m", "incremental h", "erasers h", "baseline h", "total h"],
               [[f"{int(t.num):03d}", int(t.n), int(t.train_end), int(t.m), f"{t.t_incremental_h:.2f}",
                 f"{t.t_operators_h:.2f}", f"{t.t_baseline_h:.2f}", f"{t.t_total_h:.2f}"] for t in top.itertuples()]))
    A(f"\nTop five together: {top.t_total_h.sum():.1f} h of {tot:.1f} h "
      f"({100 * top.t_total_h.sum() / tot:.0f}%).\n")
    budget = float(tc["budget_hours"])
    if tot > budget:
        A(f"**⚠ Over the {budget:g} h Mac-local budget.** Options (not applied; researcher decision): "
          "(a) run W2-1 on the server (`docs/SETUP.md`), (b) reduce K, (c) cache the per-call T_B "
          "preprocessing (c1 term), (d) split the longest series to the server only.\n")
    else:
        A(f"Within the {budget:g} h Mac-local budget.\n")

    # 6
    A("## 6. Figure F2 (P7)\n")
    A(f"Position #{cfg['figure']['position_index']}, r = 2m, erasers {' and '.join(cfg['figure']['operators'])}.\n")
    for p in figs:
        A(f"- `{rel(p)}`")
    A("")

    # 7
    A("## 7. Open items (undecided)\n")
    A("- **S4 tolerance near zero distance** — see §3; decided by the researcher if S4 failed.\n"
      "- **D7–D10** (AD faithfulness adaptation) remain undecided (D-D2-1); not touched here.\n"
      "- **Final eraser list for W2-1** — BRIEF3 §1: the pilot list is not final.\n"
      "- **W2-1 compute location / K** — see §5.\n")
    REPORT.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
