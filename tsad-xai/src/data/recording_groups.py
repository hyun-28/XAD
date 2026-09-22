"""Recording groups: which archive series come from the same recording (F2).

Two groupings, both written to the manifest:

  name_group     <name> with the DISTORTED / NOISE prefix removed
                 (UcrMeta.base_name). Cheap, conservative (may over-merge:
                 the nine STAFFIIIDatabase files share a name but not a
                 length). Primary unit of analysis (configs/stats.yaml).
  content_group  connected components of "these two series are the same
                 signal": Pearson correlation of the z-normalised first
                 `compare_points` samples of the training prefix >= threshold.
                 Verifies name_group; sensitivity analysis.

Pairs compared (configs/stats.yaml: recording_groups):
  * every pair inside one name_group;
  * every cross-name pair inside one domain_goswami whose lengths differ by
    at most cross_length_tol (relative to the longer one).

Nothing here modifies a series: the comparison slices a read-only view.
"""
from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from src.data.ucr import VARIANT_PREFIXES


def name_group_of(name: str) -> str:
    """Strip one leading DISTORTED / NOISE prefix (slide 100); nothing else."""
    for p in VARIANT_PREFIXES:
        if name.startswith(p):
            return name[len(p):]
    return name


@dataclass(frozen=True)
class SeriesRef:
    num: int
    name: str
    domain: str
    length: int
    train_end: int
    x: np.ndarray            # the full series (read-only view is enough)


@dataclass
class PairResult:
    num_a: int
    num_b: int
    name_a: str
    name_b: str
    same_name_group: bool
    domain: str
    length_a: int
    length_b: int
    train_end_a: int
    train_end_b: int
    n_compared: int
    corr: float
    same_recording: bool


def _znorm(v: np.ndarray) -> np.ndarray | None:
    v = np.asarray(v, dtype=np.float64)
    sd = v.std()
    if not np.isfinite(sd) or sd == 0.0:
        return None
    return (v - v.mean()) / sd


def prefix_correlation(a: SeriesRef, b: SeriesRef, compare_points: int) -> tuple[int, float]:
    """Pearson r of the z-normalised leading samples of both training prefixes.

    n = min(compare_points, train_end_a, train_end_b). A constant prefix has no
    defined correlation -> nan (never treated as a match).
    """
    n = int(min(compare_points, a.train_end, b.train_end))
    if n < 2:
        return n, float("nan")
    za, zb = _znorm(a.x[:n]), _znorm(b.x[:n])
    if za is None or zb is None:
        return n, float("nan")
    return n, float(np.dot(za, zb) / n)


def candidate_pairs(series: list[SeriesRef], cross_length_tol: float) -> list[tuple[SeriesRef, SeriesRef]]:
    """Same-name pairs (always) + cross-name pairs in one domain within the length tolerance."""
    by_name = defaultdict(list)
    for s in series:
        by_name[name_group_of(s.name)].append(s)
    pairs = []
    for group in by_name.values():
        pairs.extend(itertools.combinations(sorted(group, key=lambda s: s.num), 2))
    by_domain = defaultdict(list)
    for s in series:
        by_domain[s.domain].append(s)
    for members in by_domain.values():
        for a, b in itertools.combinations(sorted(members, key=lambda s: s.num), 2):
            if name_group_of(a.name) == name_group_of(b.name):
                continue
            if abs(a.length - b.length) <= cross_length_tol * max(a.length, b.length):
                pairs.append((a, b))
    return pairs


def compare_pairs(series: list[SeriesRef], *, compare_points: int, corr_threshold: float,
                  cross_length_tol: float) -> list[PairResult]:
    out = []
    for a, b in candidate_pairs(series, cross_length_tol):
        n, r = prefix_correlation(a, b, compare_points)
        out.append(PairResult(
            num_a=a.num, num_b=b.num, name_a=a.name, name_b=b.name,
            same_name_group=name_group_of(a.name) == name_group_of(b.name),
            domain=a.domain if a.domain == b.domain else f"{a.domain}|{b.domain}",
            length_a=a.length, length_b=b.length,
            train_end_a=a.train_end, train_end_b=b.train_end,
            n_compared=n, corr=r,
            same_recording=bool(np.isfinite(r) and r >= corr_threshold),
        ))
    return out


class _UnionFind:
    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # keep the smallest num as the root so labels are deterministic
            if rb < ra:
                ra, rb = rb, ra
            self.parent[rb] = ra


def content_groups(series: list[SeriesRef], pairs: list[PairResult]) -> dict[int, str]:
    """num -> content_group label. Label = 'cg<smallest num in the component>'."""
    uf = _UnionFind([s.num for s in series])
    for p in pairs:
        if p.same_recording:
            uf.union(p.num_a, p.num_b)
    return {s.num: f"cg{uf.find(s.num):03d}" for s in series}


def group_sizes(labels: dict[int, str]) -> dict[str, int]:
    c = defaultdict(int)
    for lab in labels.values():
        c[lab] += 1
    return dict(c)


def grouping_disagreements(series: list[SeriesRef], content: dict[int, str]) -> dict:
    """Where the two groupings disagree.

    same_name_diff_content: name_groups whose members fall into >1 content_group
    diff_name_same_content: content_groups spanning >1 name_group  (STOP-2 of
                            the follow-up brief: report and wait)
    """
    name_of = {s.num: name_group_of(s.name) for s in series}
    by_name = defaultdict(set)
    by_content = defaultdict(set)
    for s in series:
        by_name[name_of[s.num]].add(content[s.num])
        by_content[content[s.num]].add(name_of[s.num])
    return {
        "same_name_diff_content": {k: sorted(v) for k, v in by_name.items() if len(v) > 1},
        "diff_name_same_content": {k: sorted(v) for k, v in by_content.items() if len(v) > 1},
    }
