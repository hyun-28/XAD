"""Apply the pre-registered selection rules and freeze the result.

Run this ONCE, commit configs/datasets.yaml, and never hand-edit it. If the
selection changes after results are seen, that is cherry-picking and a reviewer
will ask about it. The rules live here so the answer is "here is the script".

RULES (pre-registered)
  R1  file appears in the official TSB-AD Eval split
  R2  0.4% <= anomaly ratio <= 10%
      lower bound set just below UCR's ~0.6% so the primary archive survives
      *** THE STATED RATIONALE IS FACTUALLY WRONG -- measured 2026-09-07 on the
      real 250-file archive, UCR's anomaly ratio has median 0.34% (min 0.0005%,
      max 4.9%). The 0.4% bound therefore sits ABOVE the median and rejects
      147/250 series; only 100 survive all rules. The rule is left AS
      PRE-REGISTERED here on purpose -- relaxing a threshold after seeing the
      data is the cherry-picking this script exists to prevent. If the bound is
      to change, change it as a deliberate, documented amendment BEFORE results
      are seen. For reference: a 0.001 bound would admit 194, 0.0005 -> 211.
  R3  6,000 <= length <= 900,000  (UCR spans 6,684 to 900,000)
      NOTE: an earlier draft capped this at 60,000 on the theory that KernelSHAP
      cost scales with series length. It does not -- we analyse sampled WINDOWS,
      so cost scales with the window count `n`, not with L. The 60,000 cap would
      have silently excluded most of the UCR archive (mean length ~67,800),
      i.e. gutted the primary dataset. Compute is bounded by `n` instead.
  R4  anomalies are not confined to the final 5% of the series
      (excludes run-to-failure bias, one of Wu & Keogh's four flaws)
  R5  the series is neither trivial nor impossible -- deferred to
      scripts/screen_difficulty.py, which needs a fitted detector
  R6  domain-balanced sample with a fixed seed
      NOT APPLICABLE TO UCR. The archive has no domain field (src/data_ucr.py
      documents this: the nine-domain split comes from Goswami et al. and is
      not in the filenames). For --format ucr the sample is balanced on
      `distorted` instead -- 92 injected vs 158 natural anomalies -- which
      data_ucr.py argues matters more for this project anyway.

Usage:
    python scripts/build_dataset_config.py --root data/TSB-AD-U --n 40
    python scripts/build_dataset_config.py --format ucr --n 40 \
        --root AnomalyDatasets_2021/.../FilesAreInHere/UCR_Anomaly_FullData
"""
import argparse, os, re, json, sys
import numpy as np
import pandas as pd

# `python scripts/build_dataset_config.py` puts scripts/ on sys.path, not the
# repo root, so `from src...` would fail. Fix it here rather than requiring
# callers to set PYTHONPATH.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# TSB-AD filenames look like:
#   001_NAB_id_1_Facility_tr_1007_1st_2014.csv
#    ^   ^          ^      ^        ^
#    |   source     |      domain   train/test boundary index
#    index          series id
FNAME = re.compile(r"^(?P<idx>\d+)_(?P<source>[A-Za-z0-9]+)_id_(?P<sid>\d+)_"
                   r"(?P<domain>[A-Za-z0-9]+)_tr_(?P<tr>\d+)_(?P<rest>.*)\.csv$")


def parse_name(fn):
    m = FNAME.match(fn)
    return m.groupdict() if m else None


def scan(root):
    rows = []
    for fn in sorted(os.listdir(root)):
        if not fn.endswith(".csv"):
            continue
        meta = parse_name(fn)
        try:
            df = pd.read_csv(os.path.join(root, fn)).dropna()
            y = df["Label"].astype(int).to_numpy()
        except Exception as e:
            rows.append({"file": fn, "error": str(e)}); continue
        L = len(y)
        anom = np.flatnonzero(y == 1)
        rows.append({
            "file": fn,
            "source": (meta or {}).get("source"),
            "domain": (meta or {}).get("domain"),
            "tr": int((meta or {}).get("tr", 0) or 0),
            "length": L,
            "n_channels": df.shape[1] - 1,
            "anomaly_ratio": float(y.mean()),
            "n_anomaly_segments": int(np.sum(np.diff(np.r_[0, y, 0]) == 1)),
            "last_anom_frac": float(anom.max() / L) if anom.size else np.nan,
            "min_seg_len": int(min((len(g) for g in _runs(y)), default=0)),
        })
    return pd.DataFrame(rows)


def scan_ucr(root):
    """Inventory the raw UCR archive (.txt, label encoded in the filename).

    Deliberately does NOT load values as floats. Every quantity the rules need
    is either in the filename or is the series length, and length is just the
    number of whitespace-separated tokens -- so this is one read per file
    instead of a full np.loadtxt of 374MB.

    WATCH OUT: 9 of the 250 files put the whole series on ONE line. Counting
    newlines gives length=1 for those and produces anomaly_ratio > 1. Count
    tokens, not lines.
    """
    from src.data_ucr import parse_filename
    rows = []
    for fn in sorted(os.listdir(root)):
        meta = parse_filename(fn)
        if meta is None:
            continue
        with open(os.path.join(root, fn)) as fh:
            L = len(fh.read().split())
        if L == 0:
            rows.append({"file": fn, "error": "empty"}); continue
        rows.append({
            "file": fn,
            "source": "UCR",
            # UCR has exactly one anomaly per series, so segment count is 1 and
            # min_seg_len is just its length. Both are known without loading.
            "domain": None,
            "distorted": bool(meta["distorted"]),
            "name": meta["base"],
            "tr": meta["train_end"],
            "anom_start": meta["anom_start"],
            "anom_end": meta["anom_end"],
            "length": L,
            "n_channels": 1,
            "anomaly_ratio": meta["anom_len"] / L,
            "n_anomaly_segments": 1,
            "last_anom_frac": meta["anom_end"] / L,
            "min_seg_len": meta["anom_len"],
        })
    return pd.DataFrame(rows)


def _runs(y):
    out, cur = [], []
    for i, v in enumerate(y):
        if v == 1:
            cur.append(i)
        elif cur:
            out.append(cur); cur = []
    if cur:
        out.append(cur)
    return out


# UCR ships the same underlying recording several times, differing only in
# preprocessing: FOO, DISTORTEDFOO and NOISEFOO share one signal AND one
# anomaly interval. They are not independent samples, and any bootstrap that
# treats them as such understates its own CI width (pseudo-replication).
# Balancing on `distorted` actively PULLS IN such pairs, so this must be
# reported every run. Not auto-removed: dropping series is a change to the
# pre-registered selection and belongs in a documented amendment, not here.
UCR_BASE = re.compile(r"^\d+_UCR_Anomaly_(?:DISTORTED|NOISE)?(.+?)_"
                      r"(\d+)_(\d+)_(\d+)\.txt$")


def duplicate_groups(files):
    """Group selected UCR files by (base recording, anomaly interval)."""
    g = {}
    for fn in files:
        m = UCR_BASE.match(fn)
        if m:
            g.setdefault((m.group(1), m.group(3), m.group(4)), []).append(fn)
    return {k: v for k, v in g.items() if len(v) > 1}


def apply_rules(df, n=40, seed=0, strat="domain"):
    d = df.dropna(subset=["length"]).copy()
    d["R2"] = d.anomaly_ratio.between(0.004, 0.10)
    d["R3"] = d.length.between(6000, 900000)
    # R7: the mask width must fit inside the shortest anomalous segment,
    # otherwise Experiment A's anomaly control arm is empty (see artifact.py).
    d["R7"] = d.min_seg_len >= 10
    d["R4"] = ~((d.last_anom_frac > 0.95) & (d.n_anomaly_segments <= 1))
    keep = d[d.R2 & d.R3 & d.R4 & d.R7].copy()
    rng = np.random.default_rng(seed)
    picked = []
    if not keep.empty:
        per = max(1, n // max(1, keep[strat].nunique()))
        for _, g in keep.groupby(strat, dropna=False):
            take = min(per, len(g))
            picked.append(g.iloc[rng.permutation(len(g))[:take]])
        keep = pd.concat(picked)
        if len(keep) > n:
            keep = keep.iloc[rng.permutation(len(keep))[:n]]
    return d, keep.sort_values("file")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/TSB-AD-U")
    ap.add_argument("--format", choices=("tsb", "ucr"), default="tsb",
                    help="tsb: TSB-AD .csv with a Label column. "
                         "ucr: raw UCR .txt, label encoded in the filename.")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="configs/datasets.yaml")
    a = ap.parse_args()

    ucr = a.format == "ucr"
    strat = "distorted" if ucr else "domain"
    df = scan_ucr(a.root) if ucr else scan(a.root)
    if df.empty:
        raise SystemExit(f"no matching files under {a.root!r} for "
                         f"--format {a.format}. Wrong --format or wrong root?")
    inv = f"results/dataset_inventory_{a.format}.csv"
    os.makedirs("results", exist_ok=True)
    df.to_csv(inv, index=False)
    allrows, keep = apply_rules(df, a.n, a.seed, strat=strat)

    print(f"scanned {len(df)} files  (format={a.format}, stratify on {strat})")
    for r in ("R2", "R3", "R4", "R7"):
        if r in allrows:
            print(f"  {r}: {int(allrows[r].sum())} pass")
    print(f"  ALL rules: {int((allrows.R2 & allrows.R3 & allrows.R4 & allrows.R7).sum())} pass")
    print(f"selected {len(keep)} of the {a.n} requested")
    if len(keep) < a.n:
        print(f"  WARNING: only {len(keep)} series survive the pre-registered"
              f" rules; the study is under-powered at n={a.n}.")
    cols = (["file", "distorted", "length", "anomaly_ratio", "min_seg_len"]
            if ucr else ["file", "domain", "length", "anomaly_ratio",
                         "n_anomaly_segments", "min_seg_len"])
    print(keep[cols].to_string(index=False))

    if ucr:
        dups = duplicate_groups(keep.file.tolist())
        uniq = len(keep) - sum(len(v) - 1 for v in dups.values())
        print(f"\nunique underlying recordings: {uniq} of {len(keep)} selected")
        if dups:
            print(f"  WARNING: {len(dups)} group(s) are the SAME recording with"
                  f" the SAME anomaly interval, differing only in preprocessing"
                  f" (DISTORTED/NOISE). Treating them as independent samples is"
                  f" pseudo-replication and will understate bootstrap CI width.")
            for (base, st, en), v in sorted(dups.items()):
                print(f"    {base} [{st},{en}]: " + ", ".join(
                    f.split("_UCR_Anomaly_")[0] for f in v))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        f.write("# GENERATED by scripts/build_dataset_config.py -- do not hand-edit.\n")
        f.write(f"# rules: R2 ar in [0.004,0.10]; R3 len in [6000,900000];"
                f" R4 not run-to-failure; R7 min_seg_len>=10;"
                f" R6 balanced on {strat} seed={a.seed}\n")
        f.write(f"# scanned {len(df)} files under {a.root}\n")
        f.write(f"format: {a.format}\nroot: {a.root}\nseed: {a.seed}\nfiles:\n")
        for _, r in keep.iterrows():
            f.write(f"  - file: {r.file}\n")
            if ucr:
                # anom_start/anom_end ARE the localization ground truth --
                # freeze them here so Stage 6 never re-derives them.
                f.write(f"    distorted: {bool(r.distorted)}\n"
                        f"    train_end: {int(r.tr)}\n"
                        f"    anom_start: {int(r.anom_start)}\n"
                        f"    anom_end: {int(r.anom_end)}\n")
            else:
                f.write(f"    domain: {r.domain}\n")
            f.write(f"    length: {int(r.length)}\n"
                    f"    anomaly_ratio: {r.anomaly_ratio:.5f}\n"
                    f"    min_seg_len: {int(r.min_seg_len)}\n")
    print(f"\nwrote {a.out}  (commit this file)")
    print("NOTE: R1 (official Eval split) and R5 (difficulty screen) are not"
          " applied here -- see the docstring.")


if __name__ == "__main__":
    main()
