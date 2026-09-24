"""w2_e0.py -- W2 rev1 §4 E0: score_patch accuracy on real series, identity bit-equality, speed.

    20 random series (seeded, from all 250) x 20 random R each, |R| in {0.25, 0.5, 1.0} w
    (round half up, min 1), R inside the test region [train_end, n). Each R gets additive Gaussian
    noise (sd = noise_sd_over_train_std x std(x[:train_end])). The 20 R of a series are drawn with
    pairwise-disjoint influence ranges I(R), so ONE full recomputation of x' serves all 20: a
    window's score depends only on its own points and T_B (D-E0-1).

    accuracy   max |score_patch(x', R) - score(x')[J(R)]| <= tol (1e-6)   -> else STOP (rev1 §11)
    identity   score_patch(identity(x), R) bit-identical to score_patch(x, R), and
               AI = (resp_after - resp_before) / scale(x) == 0.0 exactly    (identity only; §12)
    speed      score_patch wall time per R, by w, |R|, n

Writes results/w2/e0/regions.csv (one row per R; no AI values), results/w2/e0/summary.json.
Exit: 0 pass, 2 STOP, 1 error.
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import REPO_ROOT, load_yaml  # noqa: E402

OUT = REPO_ROOT / "results" / "w2" / "e0"


def run_series(num: int, cfg: dict) -> dict:
    from src.detectors.matrixprofile import MatrixProfileDetector
    from src.perturb.operators import Ctx, apply, influence
    from src import w2
    e0 = cfg["e0"]
    man, g1 = w2.manifest(), w2.gate1_primary()
    lay = w2.layout(num, man, g1)
    x = w2.load_checked(num, man)
    te, w, n = lay.train_end, lay.w, lay.n
    rng = np.random.default_rng([int(e0["seed"]), num])
    regions, blocks = [], []
    for _ in range(int(e0["max_draws"])):
        L = w2.length_for(float(rng.choice(e0["length_fracs"])), w)
        a = int(rng.integers(te, n - L + 1))
        i0, i1 = influence(a, a + L, w)
        if all(i1 <= b0 or i0 >= b1 for b0, b1 in blocks):
            regions.append((a, a + L))
            blocks.append((i0, i1))
            if len(regions) == int(e0["n_regions"]):
                break
    if len(regions) < int(e0["n_regions"]):
        raise RuntimeError(f"{num}: placed only {len(regions)} regions with disjoint I(R)")
    sd = float(e0["noise_sd_over_train_std"]) * float(np.std(x[:te]))
    xp = np.array(x, copy=True)
    for a, b in regions:
        xp[a:b] = xp[a:b] + sd * rng.standard_normal(b - a)
    det = MatrixProfileDetector.fit(x[:te], window=w, mode="ab_join", constant_score="error")
    t0 = time.perf_counter()
    full = det.score(xp)
    t_full = time.perf_counter() - t0
    stored = w2.gate1_score(num)
    sc = w2.scale_of(num, lay)
    ctx = Ctx(train_end=te, gt=(lay.start0, lay.stop0), w=w)
    rows, deltas = [], []
    for a, b in regions:
        t0 = time.perf_counter()
        J, s = det.score_patch(xp, (a, b))
        t_patch = time.perf_counter() - t0
        d = np.abs(s - full[J])
        deltas.append(d)
        J0, s0 = det.score_patch(x, (a, b))
        J1, s1 = det.score_patch(apply("identity", x, [(a, b)], None, ctx), (a, b))
        # AI is undefined when scale(x) is invalid (D8: the series is excluded); recorded as None
        ai_identity_zero = (bool((s1.max() - s0.max()) / sc["scale"] == 0.0) if sc["scale_ok"] else None)
        rows.append({"num": num, "n": n, "train_end": te, "w": w, "a": a, "b": b, "L": b - a,
                     "L_over_w": (b - a) / w, "n_windows": int(J.size), "max_abs_delta": float(d.max()),
                     "t_patch_s": t_patch, "identity_bitexact": bool(np.array_equal(s0, s1) and np.array_equal(J0, J1)),
                     "identity_ai_zero": ai_identity_zero, "scale_ok": sc["scale_ok"],
                     "unperturbed_vs_gate1_max_abs": float(np.abs(s0 - stored[J0]).max())})
    return {"num": num, "rows": rows, "deltas": np.concatenate(deltas), "t_full_s": t_full}


def main() -> int:
    cfg = load_yaml("w2")
    os.environ["NUMBA_NUM_THREADS"] = str(cfg["execution"]["numba_threads_per_worker"])
    from src.runlog import env_header
    from src import w2
    header = env_header()
    print(header, flush=True)
    e0 = cfg["e0"]
    rng = np.random.default_rng(int(e0["seed"]))
    nums = sorted(int(v) for v in rng.choice(np.arange(1, 251), size=int(e0["n_series"]), replace=False))
    g1 = w2.gate1_primary()
    print(f"E0: series {nums}; expected full-score time from gate 1 (10 threads): "
          f"{g1.loc[nums].score_s.sum():.0f} s", flush=True)
    import multiprocessing as mp
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=int(cfg["execution"]["workers"]), mp_context=mp.get_context("spawn")) as ex:
        res = list(ex.map(run_series, nums, [cfg] * len(nums)))
    elapsed = time.perf_counter() - t0
    rows = [r for x in res for r in x["rows"]]
    deltas = np.concatenate([x["deltas"] for x in res])
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "regions.csv", index=False)
    tol = float(e0["tol"])
    q = [0.5, 0.9, 0.99, 0.999, 1.0]
    edges = [0] + [10.0 ** k for k in range(-16, 0)]
    hist = np.histogram(deltas, bins=edges + [np.inf])[0]
    over = df[df.max_abs_delta > tol]
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "env_header": header,
        "numba_threads_per_worker": cfg["execution"]["numba_threads_per_worker"],
        "workers": cfg["execution"]["workers"], "elapsed_s": elapsed,
        "series": nums, "n_regions": len(df), "n_windows_compared": int(deltas.size),
        "tol": tol, "max_abs_delta": float(deltas.max()),
        "quantiles": {str(k): float(np.quantile(deltas, k)) for k in q},
        "n_exact_zero": int((deltas == 0).sum()),
        "hist_edges": [str(e) for e in edges] + ["inf"], "hist_counts": hist.tolist(),
        "regions_over_tol": over[["num", "a", "b", "L", "w", "max_abs_delta"]].to_dict("records"),
        "accuracy_pass": bool(over.empty),
        "identity_bitexact_all": bool(df.identity_bitexact.all()),
        "identity_ai_zero_all": bool(df.loc[df.scale_ok, "identity_ai_zero"].astype(bool).all()),
        "identity_ai_cases": int(df.scale_ok.sum()),
        "n_scale_not_ok": int((~df.drop_duplicates("num").scale_ok).sum()),
        "t_full_s": {str(x["num"]): x["t_full_s"] for x in res},
        "unperturbed_vs_gate1_max_abs": float(df.unperturbed_vs_gate1_max_abs.max()),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"E0: {len(df)} regions, {deltas.size} windows; max|Δ| = {deltas.max():.3g} (tol {tol:g}); "
          f"identity bit-exact {summary['identity_bitexact_all']}, AI==0 {summary['identity_ai_zero_all']}; "
          f"{elapsed:.0f} s", flush=True)
    if not summary["accuracy_pass"] or not summary["identity_bitexact_all"] or not summary["identity_ai_zero_all"]:
        print("STOP (rev1 §11): E0 criterion failed; see results/w2/e0/summary.json")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
