"""Analysis subsets over the manifest (BRIEF_A-followup F4; configs/subsets.yaml).

A subset is a mapping of filters ANDed together. Filter keys:
    <column>_in      value list, membership
    <column>_not_in  value list, non-membership
    <column>         scalar, equality
Values are compared as strings so the same code works on manifest rows read
back from CSV and on rows built in memory. Unknown columns raise.
"""
from __future__ import annotations

from typing import Iterable

from src.config import load_yaml


def _as_str(v) -> str:
    return "NA" if v is None else str(v)


def _filter_fn(key: str, spec):
    if key.endswith("_in") and not key.endswith("_not_in"):
        col, vals = key[:-3], {_as_str(v) for v in spec}
        return col, (lambda r: _as_str(r[col]) in vals)
    if key.endswith("_not_in"):
        col, vals = key[:-7], {_as_str(v) for v in spec}
        return col, (lambda r: _as_str(r[col]) not in vals)
    if isinstance(spec, (list, tuple, set, dict)):
        raise ValueError(f"filter {key!r}: scalar expected for equality, got {type(spec).__name__}")
    col, val = key, _as_str(spec)
    return col, (lambda r: _as_str(r[col]) == val)


def compile_subset(spec: dict) -> list:
    """[(column, predicate)] for one subset definition."""
    return [_filter_fn(k, v) for k, v in (spec or {}).items()]


def select(rows: list[dict], names: Iterable[str], subsets_cfg: dict | None = None) -> list[dict]:
    """Rows matching every named subset (intersection)."""
    cfg = subsets_cfg if subsets_cfg is not None else load_yaml("subsets")["subsets"]
    preds = []
    for n in names:
        if n not in cfg:
            raise KeyError(f"unknown subset {n!r}; defined: {sorted(cfg)}")
        preds.extend(compile_subset(cfg[n]))
    for col, _ in preds:
        if rows and col not in rows[0]:
            raise KeyError(f"subset filter column {col!r} not in manifest")
    return [r for r in rows if all(p(r) for _, p in preds)]


def n_groups(rows: list[dict], unit: str | None = None) -> int:
    """Distinct values of the unit-of-analysis column (configs/stats.yaml)."""
    unit = unit or load_yaml("stats")["unit_of_analysis"]
    if rows and unit not in rows[0]:
        raise KeyError(f"unit_of_analysis column {unit!r} not in manifest")
    return len({_as_str(r[unit]) for r in rows})


def summarise(rows: list[dict], names: Iterable[str], *, unit: str | None = None,
              min_groups: int | None = None, subsets_cfg: dict | None = None) -> dict:
    names = list(names)
    sel = select(rows, names, subsets_cfg)
    stats = load_yaml("stats")
    unit = unit or stats["unit_of_analysis"]
    min_groups = min_groups if min_groups is not None else int(stats["min_groups_for_hypothesis_test"])
    ng = n_groups(sel, unit)
    return {"subset": " ∩ ".join(names), "n_series": len(sel), "n_groups": ng,
            "unit": unit, "descriptive_only": ng < min_groups}
