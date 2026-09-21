"""UCR Anomaly Archive 2021: filename parser and series loader (BRIEF A2/A3).

Filename format (BRIEF §3 A2; UCR_AnomalyDataSets.pptx slide 5):
    <num>_UCR_Anomaly_<name>_<train_end>_<begin>_<end>.txt
The regex peels the three trailing integers off the end so a <name> that
itself contains underscores cannot break parsing.

File format (slide 5: "Files are '-ascii' format"): one float per line with
CRLF line endings for 241 files; nine files (204-208, 225, 226, 242, 243) are a
single line with tab/space separators -- slide 7 "Known issues" lists exactly
these nine, and scripts/02_audit.py re-measures it. `load_series` therefore
tokenises on any whitespace rather than on lines.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

UCR_RE = re.compile(
    r"^(?P<num>\d{3})_UCR_Anomaly_(?P<name>.+)_(?P<train_end>\d+)_(?P<begin>\d+)_(?P<end>\d+)\.txt$"
)
VARIANT_PREFIXES = ("DISTORTED", "NOISE")  # slide 100 of UCR_AnomalyDataSets.pptx


class UcrFilenameError(ValueError):
    """Filename does not match the archive pattern."""


@dataclass(frozen=True)
class UcrMeta:
    num: int
    name: str            # full <name> field, e.g. DISTORTED1sddb40
    train_end_raw: int
    begin_raw: int
    end_raw: int
    filename: str

    @property
    def variant(self) -> str:
        for p in VARIANT_PREFIXES:
            if self.name.startswith(p):
                return p
        return "plain"

    @property
    def base_name(self) -> str:
        """<name> with the DISTORTED/NOISE prefix removed."""
        for p in VARIANT_PREFIXES:
            if self.name.startswith(p):
                return self.name[len(p):]
        return self.name

    @property
    def is_distorted(self) -> bool:
        return self.name.startswith("DISTORTED")

    @property
    def entity(self) -> str:
        """Filename stem without the three numeric fields (Goswami's key)."""
        return f"{self.num:03d}_UCR_Anomaly_{self.name}"


def parse_filename(filename: str) -> UcrMeta:
    m = UCR_RE.match(Path(filename).name)
    if not m:
        raise UcrFilenameError(f"does not match UCR pattern: {filename}")
    d = m.groupdict()
    return UcrMeta(num=int(d["num"]), name=d["name"],
                   train_end_raw=int(d["train_end"]), begin_raw=int(d["begin"]),
                   end_raw=int(d["end"]), filename=Path(filename).name)


def load_series(path: Path) -> np.ndarray:
    """Read one archive file into a 1-D float64 array. Raises on anything odd."""
    text = Path(path).read_text(encoding="ascii")
    tokens = text.split()
    if not tokens:
        raise ValueError(f"empty file: {path}")
    try:
        x = np.asarray(tokens, dtype=np.float64)
    except ValueError as e:
        raise ValueError(f"non-numeric token in {path}: {e}") from e
    if x.ndim != 1:
        raise ValueError(f"expected 1-D, got shape {x.shape}: {path}")
    return x


def file_layout(path: Path) -> dict:
    """Cheap format facts for the audit: line count, token count, separators."""
    raw = Path(path).read_bytes()
    n_lines = raw.count(b"\n")
    return {
        "n_lines": n_lines,
        "n_tokens": len(raw.split()),
        "has_crlf": b"\r\n" in raw,
        "has_tab": b"\t" in raw,
        "single_line": n_lines <= 1,
        "bytes": len(raw),
    }


def scan_archive(fulldata_dir: Path, expected_n: int) -> list[UcrMeta]:
    """Parse every .txt in the directory; assert count, pattern, numbering."""
    fulldata_dir = Path(fulldata_dir)
    if not fulldata_dir.is_dir():
        raise FileNotFoundError(fulldata_dir)
    files = sorted(p for p in fulldata_dir.iterdir() if p.suffix == ".txt")
    if len(files) != expected_n:
        raise RuntimeError(f"{fulldata_dir}: {len(files)} .txt files, expected {expected_n} "
                           f"(200 would be the SIGKDD-2021 contest edition; BRIEF §7-1)")
    metas, bad = [], []
    for p in files:
        try:
            metas.append(parse_filename(p.name))
        except UcrFilenameError:
            bad.append(p.name)
    if bad:
        raise RuntimeError(f"{len(bad)} filenames do not parse (BRIEF §7-1): {bad}")
    nums = [m.num for m in metas]
    if nums != list(range(1, expected_n + 1)):
        dup = sorted({n for n in nums if nums.count(n) > 1})
        missing = sorted(set(range(1, expected_n + 1)) - set(nums))
        raise RuntimeError(f"numbering not 001..{expected_n:03d}: duplicates={dup} missing={missing}")
    return metas
