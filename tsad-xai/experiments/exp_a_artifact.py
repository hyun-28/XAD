"""Experiment A -- does perturbation manufacture anomalies?

This is the W3 GO/NO-GO gate. It uses NO explainer at all, which is why it runs
first and cheap: if the artifact effect does not exist, contribution C1 dies
here and nothing downstream is worth building.

Usage
-----
    python experiments/exp_a_artifact.py --synthetic          # no data needed
    python experiments/exp_a_artifact.py --csv path/to.csv    # TSB-AD file

Reads a TSB-AD CSV: last column 'Label', all preceding columns are values.

Output
------
results/exp_a.csv with one row per (series, perturbation, region):
    AI  = median standardised score shift
    VR  = fraction of perturbed normal windows pushed above the 95th percentile
Plus the normal-vs-anomalous contrast, which is the control.
"""
import argparse, os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.perturbations import PERTURBATIONS, IDENTITY
from src.metrics.artifact import artifact_index
from src.metrics import bootstrap_ci


def make_synthetic(L=4000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(L)
    x = (np.sin(t / 20.0) + 0.4 * np.sin(t / 7.0) + 0.05 * rng.standard_normal(L))
    label = np.zeros(L, int)
    for s in (900, 2100, 3300):
        x[s:s + 40] += 2.5
        label[s:s + 40] = 1
    return x[:, None], label


def load_csv(path):
    df = pd.read_csv(path).dropna()
    data = df.iloc[:, :-1].values.astype(float)
    label = df["Label"].astype(int).to_numpy()
    return data, label


def run(score_fn, X, label, width=50, n=80, seed=0):
    rows = []
    for pname, op in list(PERTURBATIONS.items()) + [("IDENTITY", IDENTITY)]:
        for region in ("normal", "anomaly"):
            r = artifact_index(score_fn, X, label, op, width=width, n=n,
                               region=region, rng=np.random.default_rng(seed))
            lo, hi = bootstrap_ci(r["deltas"]) if r["n"] else (np.nan, np.nan)
            rows.append({"perturbation": pname, "region": region,
                         "AI": r["AI"], "VR": r["VR"], "n": r["n"],
                         "width": r.get("width", width),
                         "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def signal_to_artifact(df):
    """SAR = |AI_anomaly| / |AI_normal|.

    How much bigger is the real signal (masking a true anomaly) than the
    artifact (masking a normal region)? SAR near 1 means the perturbation is
    mostly measuring itself. Discovered empirically -- it was NOT in the
    original design, and it reverses the design's prediction about B6.
    """
    piv = df.pivot_table(index="perturbation", columns="region", values="AI")
    piv["SAR"] = (piv.get("anomaly").abs() / piv.get("normal").abs().replace(0, np.nan))
    return piv.sort_values("SAR", ascending=False)


def verdict(df):
    """W3 gate decision."""
    norm = df[(df.region == "normal") & (df.perturbation != "IDENTITY")]
    fires = norm[(norm.ci_lo > 0) | (norm.ci_hi < 0)]
    return {"gate": "GO" if len(fires) else "NO-GO",
            "n_perturbations_with_significant_artifact": int(len(fires)),
            "worst": (None if norm.empty else
                      str(norm.loc[norm.AI.abs().idxmax(), "perturbation"])),
            "max_abs_AI": (None if norm.empty else float(norm.AI.abs().max()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv"); ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--detector", default="IForest")
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--width", type=int, default=50)
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--out", default="results/exp_a.csv")
    a = ap.parse_args()

    X, label = make_synthetic() if a.synthetic or not a.csv else load_csv(a.csv)
    print(f"data {X.shape}  anomaly ratio {label.mean():.4f}")

    from src.detectors import ScoreFn, self_check
    sf = ScoreFn(a.detector, X, window=a.window)
    self_check(sf, X)
    print(f"detector {a.detector} fitted; score invariants OK")

    df = run(sf, X, label, width=a.width, n=a.n)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)
    print(df.to_string(index=False))
    print("\n--- signal-to-artifact ratio (higher is safer) ---")
    print(signal_to_artifact(df).to_string())
    v = verdict(df)
    print("\nW3 GATE:", json.dumps(v, indent=2))
    print(f"score_fn calls: {sf.n_calls}")


if __name__ == "__main__":
    main()
