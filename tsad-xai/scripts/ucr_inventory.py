"""Inventory the UCR Anomaly Archive and apply UCR-appropriate selection rules.

Replaces scripts/build_dataset_config.py for UCR. That script parses a domain
field out of TSB-AD filenames; UCR filenames have no domain field, so its rule
R6 (domain-balanced sampling) cannot run here.

RULES APPLIED
  U1  parseable filename
  U2  anom_len >= 10        mask width must fit inside the anomaly (Experiment A)
  U3  6,000 <= length <= 900,000
  U4  train_end / length <= 0.7   leaves enough test region to sample from
  U5  stratified by `distorted` (synthetic vs natural), fixed seed
      -- replaces the impossible domain stratification

Usage:
    python scripts/ucr_inventory.py --root path/to/UCR_Anomaly_FullData --n 40
"""
import argparse, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_ucr import parse_filename


def scan(root, read_values=True):
    if not os.path.isdir(root):
        sys.exit(
            f"\n--root not found: {root}\n"
            f"'path/to/...' is a placeholder -- replace it with the real path.\n"
            f"Find it with:\n"
            f'  find ~ -type d -name "UCR_Anomaly_FullData" 2>/dev/null\n'
        )
    rows = []
    for fn in sorted(os.listdir(root)):
        m = parse_filename(fn)
        if m is None:
            rows.append({"file": fn, "parse_ok": False}); continue
        r = dict(m, file=fn, parse_ok=True)
        if read_values:
            try:
                x = np.loadtxt(os.path.join(root, fn))
                r["length"] = int(len(x))
                r["anomaly_ratio"] = r["anom_len"] / len(x)
                r["train_frac"] = r["train_end"] / len(x)
            except Exception as e:
                r["error"] = str(e)
        rows.append(r)
    return pd.DataFrame(rows)


def select(df, n=40, seed=0):
    d = df[df.parse_ok & df.length.notna()].copy()
    d["U2"] = d.anom_len >= 10
    d["U3"] = d.length.between(6000, 900000)
    d["U4"] = d.train_frac <= 0.7
    keep = d[d.U2 & d.U3 & d.U4]
    rng = np.random.default_rng(seed)
    picked = []
    for _, g in keep.groupby("distorted"):
        take = min(max(1, n // 2), len(g))
        picked.append(g.iloc[rng.permutation(len(g))[:take]])
    out = pd.concat(picked) if picked else keep
    return d, out.sort_values("idx")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="path to UCR_Anomaly_FullData (250 .txt files)")
    ap.add_argument("--no-values", action="store_true",
                    help="skip reading file contents; much faster, but "
                         "length/anomaly_ratio/train_frac will be unavailable")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="configs/ucr_selection.csv")
    a = ap.parse_args()

    df = scan(a.root, read_values=not a.no_values)
    print(f"scanned {len(df)} files | parsed {int(df.parse_ok.sum())}")
    if "length" in df:
        print(f"length      min={df.length.min():.0f} med={df.length.median():.0f} max={df.length.max():.0f}")
        print(f"anom_len    min={df.anom_len.min():.0f} med={df.anom_len.median():.0f} max={df.anom_len.max():.0f}")
        print(f"anom_ratio  min={df.anomaly_ratio.min():.5f} med={df.anomaly_ratio.median():.5f} max={df.anomaly_ratio.max():.5f}")
        print(f"distorted   {int(df.distorted.sum())} / {len(df)}")
    d, keep = select(df, a.n, a.seed)
    for r in ("U2", "U3", "U4"):
        print(f"  {r}: {int(d[r].sum())} pass")
    print(f"selected {len(keep)}  (distorted {int(keep.distorted.sum())} / natural {int((~keep.distorted).sum())})")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    keep.to_csv(a.out, index=False)
    df.to_csv("results/ucr_inventory_full.csv", index=False)
    print(f"wrote {a.out} -- commit this")
