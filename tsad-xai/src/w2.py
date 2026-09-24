"""Shared W2 rev1 helpers: series loading with checks, layout, scale(x), admissible positions.

Definitions follow brief rev1 §2:
    test region  [train_end, n)
    E            [GT_start - w, GT_stop + w)                       (buffer = w, D-C2-1)
    I(R)         [a - w + 1, b + w - 1)
    admissible   I(R) ⊂ test region and I(R) ∩ E = ∅               ("정상 섭동 허용 조건")
    s_normal     gate-1 padded score on the test region minus E    (= src/gate1.normal_test_mask(buffer=w))
    scale(x)     q99(s_normal) - median(s_normal)                  (D8; numpy default percentile, as gate 1)
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import REPO_ROOT, load_yaml, resolve
from src.data.ucr import load_series, parse_filename
from src.gate1 import normal_test_mask

GATE_DIR = REPO_ROOT / "results" / "gate1" / "MatrixProfile"


def round_half_up(v: float) -> int:
    return int(math.floor(v + 0.5))


def length_for(frac: float, w: int) -> int:
    """rev1 §8 E2a: round(frac * w), half up, minimum 1."""
    return max(1, round_half_up(frac * w))


def fulldata_dir() -> Path:
    paths = load_yaml("paths")
    a1 = json.loads(resolve(paths["ucr"]["a1_result"]).read_text())
    if a1["stop"]:
        raise RuntimeError("01_download.py recorded STOP")
    return Path(a1["anchors"]["official"]) / paths["ucr"]["fulldata_subdir"]


def manifest() -> pd.DataFrame:
    m = pd.read_csv(resolve(load_yaml("paths")["manifest"]), dtype={"sha256": str})
    if len(m) != 250:
        raise RuntimeError(f"manifest has {len(m)} rows")
    return m.set_index("num", drop=False)


def gate1_primary() -> pd.DataFrame:
    """Gate-1 MatrixProfile rows of the primary condition (train_prefix, AB-join), by num."""
    d = pd.read_csv(GATE_DIR / "detection.csv")
    d = d[(d.fit_on == "train_prefix") & (d["mode"] == "ab_join")]
    if len(d) != 250:
        raise RuntimeError(f"gate-1 primary rows: {len(d)}")
    return d.set_index("num", drop=False)


@dataclass(frozen=True)
class Layout:
    num: int
    n: int
    train_end: int
    start0: int
    stop0: int
    w: int

    @property
    def E(self) -> tuple[int, int]:
        return self.start0 - self.w, self.stop0 + self.w

    def pieces(self) -> list[tuple[int, int]]:
        """Test region minus E, as half-open pieces (possibly empty)."""
        e0, e1 = self.E
        return [(self.train_end, min(e0, self.n)), (max(e1, self.train_end), self.n)]

    def admissible_starts(self, L: int) -> list[tuple[int, int]]:
        """Per piece [p0, p1): inclusive range [lo, hi] of starts a with I([a, a+L)) ⊂ piece."""
        out = []
        for p0, p1 in self.pieces():
            lo, hi = p0 + self.w - 1, p1 - L - self.w + 1
            if hi >= lo:
                out.append((lo, hi))
        return out

    def n_admissible(self, L: int) -> int:
        return sum(hi - lo + 1 for lo, hi in self.admissible_starts(L))

    def max_disjoint(self, L: int) -> int:
        return sum((hi - lo) // L + 1 for lo, hi in self.admissible_starts(L))


def layout(num: int, man: pd.DataFrame, g1: pd.DataFrame) -> Layout:
    row, g = man.loc[num], g1.loc[num]
    meta = parse_filename(row["filename"])
    lay = Layout(num=int(num), n=int(row["length"]), train_end=int(meta.train_end_raw),
                 start0=int(row["start0"]), stop0=int(row["stop0"]), w=int(g["window"]))
    if (lay.train_end, lay.start0, lay.stop0, lay.n) != (int(g.train_end), int(g.start0), int(g.stop0), int(g.n)):
        raise RuntimeError(f"{num}: manifest and gate-1 disagree on the layout")
    return lay


def load_checked(num: int, man: pd.DataFrame, ddir: Path | None = None) -> np.ndarray:
    row = man.loc[num]
    p = (ddir or fulldata_dir()) / row["filename"]
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    if digest != row["sha256"]:
        raise RuntimeError(f"{p.name}: sha256 {digest} != manifest {row['sha256']}")
    x = load_series(p)
    if x.size != int(row["length"]):
        raise RuntimeError(f"{p.name}: length {x.size} != manifest {row['length']}")
    x.setflags(write=False)
    return x


def gate1_score(num: int) -> np.ndarray:
    return np.load(GATE_DIR / "scores" / f"{num:03d}_train_prefix_ab_join_seed0.npy")


def scale_of(num: int, lay: Layout) -> dict:
    """D8 scale(x) from the stored gate-1 score. Returns value and validity (not the scores)."""
    s = gate1_score(num)
    if s.size != lay.n:
        raise RuntimeError(f"{num}: stored score length {s.size} != n {lay.n}")
    sn = s[normal_test_mask(lay.n, lay.train_end, lay.start0, lay.stop0, lay.w)]
    q99, med = float(np.percentile(sn, 99)), float(np.median(sn))
    v = q99 - med
    ok = bool(np.isfinite(v) and v > 0)
    return {"scale": v, "q99": q99, "median": med, "n_normal": int(sn.size), "scale_ok": ok,
            "scale_reason": "" if ok else ("non-finite" if not np.isfinite(v) else f"scale {v:.3g} <= 0")}


def draw_disjoint(rng: np.random.Generator, starts: list[tuple[int, int]], L: int, k: int,
                  taken: list[tuple[int, int]] | None = None, max_draws: int = 100000,
                  key=lambda a, L: (a, a + L)) -> list[tuple[int, int]]:
    """Up to k regions [a, a+L) with a from the admissible ranges, pairwise disjoint under `key`
    (default: the regions themselves). Deterministic given rng. Returns what it could place."""
    cand = np.concatenate([np.arange(lo, hi + 1) for lo, hi in starts]) if starts else np.empty(0, int)
    out, blocks = [], list(taken or [])
    if cand.size == 0:
        return out
    for _ in range(max_draws):
        a = int(cand[rng.integers(cand.size)])
        k0, k1 = key(a, L)
        if all(k1 <= b0 or k0 >= b1 for b0, b1 in blocks):
            out.append((a, a + L))
            blocks.append((k0, k1))
            if len(out) == k:
                break
    return sorted(out)
