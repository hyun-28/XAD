"""BRIEF_A-followup F5: the legacy loader's GT interval equals the conventions.py one on all 250 files.

The legacy formula (src/data_ucr.py before 2026-09-22) was
    lo = max(0, anom_start - 1);  hi = min(len(x), anom_end)
with the comment "archive indices are 1-based" and no evidence. It is
reproduced here literally and compared with the rewired loader and with the
audited manifest. A single mismatch is a STOP (follow-up brief STOP-1).
"""
import csv

import numpy as np
import pytest

from src.config import load_yaml, resolve
from src.data.conventions import active_convention, to_half_open
from src.data.ucr import parse_filename, scan_archive
from src.data_ucr import load_file


def _legacy_interval(begin, end, n):
    return max(0, begin - 1), min(n, end)


def test_legacy_formula_equals_convention_on_all_250(official_fulldata_dir):
    metas = scan_archive(official_fulldata_dir, 250)
    conv = active_convention()
    manifest = {int(r["num"]): r for r in csv.DictReader(open(resolve(load_yaml("paths")["manifest"])))}
    assert len(manifest) == 250
    mismatches = []
    for m in metas:
        x, y, meta = load_file(str(official_fulldata_dir / m.filename))
        n = len(x)
        legacy = _legacy_interval(m.begin_raw, m.end_raw, n)
        new = to_half_open(m.begin_raw, m.end_raw, conv)
        nz = np.flatnonzero(y)
        loader = (int(nz[0]), int(nz[-1]) + 1)
        man = (int(manifest[m.num]["start0"]), int(manifest[m.num]["stop0"]))
        if not (legacy == new == loader == man):
            mismatches.append((m.filename, legacy, new, loader, man))
        assert meta["n_segments"] == 1
    assert mismatches == [], f"STOP (follow-up STOP-1): {len(mismatches)} mismatches: {mismatches[:5]}"


def test_legacy_parser_still_reads_the_documented_names():
    from src.data_ucr import parse_filename as legacy_parse
    for fn in ["001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt",
               "250_UCR_Anomaly_weallwalk_2951_5580_5730.txt"]:
        a, b = legacy_parse(fn), parse_filename(fn)
        assert (a["train_end"], a["anom_start"], a["anom_end"]) == (b.train_end_raw, b.begin_raw, b.end_raw)
