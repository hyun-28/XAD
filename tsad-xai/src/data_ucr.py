"""UCR Anomaly Archive loader.

VERIFIED PROPERTIES (Wu & Keogh 2021; Baldán & García-Gil, Expert Systems 2025)
------------------------------------------------------------------------------
  * 250 univariate series
  * EACH SERIES CONTAINS EXACTLY ONE ANOMALY, and its start/end are known
  * the train/test split point is specified per series
  * the training portion is anomaly-free
  * lengths range from 6,684 to 900,000 points
  * official scoring: a prediction counts as correct if it lies within
    100 points of any point of the anomaly

The one-anomaly-per-series property is what this project depends on: the
localization ground truth is unambiguous, with no anomaly-density confound.

FILENAME FORMAT  (confirmed by the user against the real archive, 250 files)
---------------
    <idx>_UCR_Anomaly_<name>_<train_end>_<anom_start>_<anom_end>.txt
e.g.
    001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt
    138_UCR_Anomaly_InternalBleeding16_1200_4187_4197.txt
The label vector is reconstructed from the filename; no separate label file.

THERE IS NO DOMAIN FIELD. The nine-domain split (Acceleration, Air Temperature,
ABP, EPG, ECG, Gait, NASA, Power Demand, RESP) used by many papers comes from
Goswami et al. (arXiv:2210.01078) and is NOT present in the archive. Do not try
to parse it out of filenames.

USE `distorted` INSTEAD. The DISTORTED prefix marks series whose anomaly was
injected/synthesised rather than naturally occurring. That flag is free, it is
in the filename, and for THIS project it matters more than domain does: whether
explanation-evaluation protocols behave differently on synthetic vs natural
anomalies is a question about the protocols, which is what we study.

TWO WAYS TO GET THE DATA
------------------------
1. aeon (easiest, no manual unzip):
       pip install aeon
       from aeon.datasets import load_anomaly_detection
       X, y = load_anomaly_detection(("KDD-TSAD", "001_UCR_Anomaly_DISTORTED1sddb40"))
   Downloads from the TimeEval archive.
2. Raw archive:
       https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCR_TimeSeriesAnomalyDatasets2021.zip
   then use the FilesAreInHere/UCR_Anomaly_FullData/ directory with load_dir().

NOT VERIFIED HERE: neither download host was reachable from the sandbox this
module was written in. The parser is tested against the documented format only.
"""
import os
import re
import numpy as np

FNAME = re.compile(
    r"^(?P<idx>\d+)_UCR_Anomaly_(?P<name>.+?)_"
    r"(?P<train_end>\d+)_(?P<start>\d+)_(?P<end>\d+)\.(txt|csv)$"
)


def parse_filename(fn):
    m = FNAME.match(os.path.basename(fn))
    if not m:
        return None
    d = m.groupdict()
    name = d["name"]
    distorted = name.startswith("DISTORTED")
    return {"idx": int(d["idx"]), "name": name,
            "base": name[len("DISTORTED"):] if distorted else name,
            "distorted": distorted,
            "train_end": int(d["train_end"]),
            "anom_start": int(d["start"]), "anom_end": int(d["end"]),
            "anom_len": int(d["end"]) - int(d["start"]) + 1}


def load_file(path):
    """Return (values, label, meta). Label is 1 on [anom_start, anom_end]."""
    meta = parse_filename(path)
    if meta is None:
        raise ValueError(f"filename does not match the UCR pattern: {path}")
    x = np.loadtxt(path).astype(float).ravel()
    y = np.zeros(len(x), dtype=int)
    lo = max(0, meta["anom_start"] - 1)          # archive indices are 1-based
    hi = min(len(x), meta["anom_end"])
    y[lo:hi] = 1
    meta = dict(meta, length=len(x), anomaly_ratio=float(y.mean()),
                n_segments=int((np.diff(np.r_[0, y, 0]) == 1).sum()))
    return x[:, None], y, meta


def split_train_test(x, y, meta):
    """Anomaly-free training prefix, then the test portion."""
    k = meta["train_end"]
    return x[:k], x[k:], y[k:]


def load_dir(root, limit=None):
    out = []
    for fn in sorted(os.listdir(root)):
        if parse_filename(fn) is None:
            continue
        try:
            out.append(load_file(os.path.join(root, fn)))
        except Exception as e:
            print(f"  skip {fn}: {e}")
        if limit and len(out) >= limit:
            break
    return out


def verify_one_anomaly(items):
    """The property the whole localization design rests on."""
    bad = [m["name"] for _, _, m in items if m["n_segments"] != 1]
    return {"n": len(items), "violations": len(bad), "examples": bad[:5]}


if __name__ == "__main__":
    # parser self-test against the documented format (no download required)
    cases = ["001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt",
             "138_UCR_Anomaly_InternalBleeding16_1200_4187_4197.txt",
             "250_UCR_Anomaly_weallwalk_2951_5580_5730.txt"]
    for c in cases:
        m = parse_filename(c)
        assert m is not None, c
        print(f"{m['idx']:>3}  {m['name']:<24} train_end={m['train_end']:<7}"
              f" anomaly=[{m['anom_start']},{m['anom_end']}]"
              f" len={m['anom_end']-m['anom_start']+1}")
    assert parse_filename("001_NAB_id_1_Facility_tr_1007_1st_2014.csv") is None
    print("parser OK")
