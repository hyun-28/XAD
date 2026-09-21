"""Per-series provenance from the archive's own slide deck (BRIEF A1/A5-0).

UCR_AnomalyDataSets.pptx (122 slides) names one or more files per slide and
states where the data came from and how the anomaly was made. This module
maps every archive series to the slide(s) that mention it and pulls out the
source statement verbatim. Matching is by <name> with the DISTORTED/NOISE
prefix removed, underscores stripped and case folded, because the deck
spells some names differently from the files (e.g. deck
`mit14046long_term_ecg` vs file `mit14046longtermecg`) and some deck
filenames carry different numeric fields than the shipped files (reported,
not silently accepted).
"""
from __future__ import annotations

import re
from pathlib import Path

from src.data.supplement import pptx_slides
from src.data.ucr import VARIANT_PREFIXES

DECK_FILE_RE = re.compile(
    r"(?:(?P<num>\d{3})_)?UCR_Anomaly_(?P<name>.+?)_(?P<train_end>\d+)_(?P<begin>\d+)_(?P<end>\d+)\.txt")

# Sentences that state where the data came from. Order matters only for the
# `source_primary` field; all matches are kept in `source_lines`.
SOURCE_PATTERNS = [
    r"^Selected input:.*",
    r".*\bcomes? from\b.*",
    r".*This is a real dataset from.*",
    r".*This is an? [A-Za-z ]*dataset.*",
    r".*This dataset (?:was prepared|contains|has|is built|comes).*",
    r".*The dataset is built.*",
    r".*We extracted an? .* data.*",
    r".*Database\b.*",
    r".*\bdatabase\b.*",
    r".*For these files we considered.*",
    r".*Fantasia.*|.*STAFFIII.*|.*CHARIS.*|.*WeAllWalk.*|.*Italian power demand.*",
]


def norm_key(name: str) -> str:
    for p in VARIANT_PREFIXES:
        if name.startswith(p):
            name = name[len(p):]
    return re.sub(r"[_\s]", "", name).lower()


def parse_deck(pptx_path: Path) -> list[dict]:
    """[{slide, files:[{num,name,key,train_end,begin,end}], text, source_lines}]"""
    out = []
    for sl in pptx_slides(pptx_path):
        text = "\n".join(sl["texts"])
        files = []
        for m in DECK_FILE_RE.finditer(text):
            d = m.groupdict()
            # A filename cited inside another series' description ("...from
            # UCR_Anomaly_ECG3...", "similar to ...", "subtle version of ...")
            # is a cross-reference, not that slide's subject.
            lead = text[max(0, m.start() - 40):m.start()]
            is_ref = bool(re.search(r"(from|similar to|version of|same way as|shifted by .* from|\(This)\s*\(?\s*$",
                                    lead, re.IGNORECASE))
            files.append({"num": int(d["num"]) if d["num"] else None, "name": d["name"],
                          "key": norm_key(d["name"]), "train_end": int(d["train_end"]),
                          "begin": int(d["begin"]), "end": int(d["end"]),
                          "cross_reference": is_ref})
        source_lines = []
        for line in text.splitlines():
            s = line.strip()
            if not s or DECK_FILE_RE.search(s) or re.fullmatch(r"[-\d.]+", s):
                continue
            if any(re.match(p, s, re.IGNORECASE) for p in SOURCE_PATTERNS):
                source_lines.append(s)
        out.append({"slide": sl["n"], "files": files, "text": text,
                    "source_lines": source_lines})
    return out


# Slides that describe a family of files by prose rather than by listing every
# filename; the archive names they cover are matched by these substrings.
FAMILY_SLIDES = {
    "mit14046longtermecg": ["MIT-BIH Long-term ECG"],
    "mit14134longtermecg": ["MIT-BIH Long-term ECG"],
    "mit14157longtermecg": ["MIT-BIH Long-term ECG"],
}


def match_series(deck: list[dict], name: str) -> dict:
    """Provenance record for one archive <name> field."""
    key = norm_key(name)
    hits = [(s, f) for s in deck for f in s["files"] if f["key"] == key and not f["cross_reference"]]
    if not hits:
        hits = [(s, f) for s in deck for f in s["files"] if f["key"] == key]
    if not hits:
        # deck text may mention the base name without a full filename
        base = norm_key(name)
        hits = [(s, None) for s in deck if base and base in re.sub(r"[_\s]", "", s["text"]).lower()]
    slides = sorted({s["slide"] for s, _ in hits})
    source_lines: list[str] = []
    for s, _ in hits:
        for line in s["source_lines"]:
            if line not in source_lines:
                source_lines.append(line)
    # Family-level slide (e.g. the MIT-BIH long-term ECG intro on slide 103)
    for fam, needles in FAMILY_SLIDES.items():
        if fam in name:
            for s in deck:
                if any(n in s["text"] for n in needles) and s["slide"] not in slides:
                    slides.append(s["slide"])
                    for line in s["source_lines"]:
                        if line not in source_lines:
                            source_lines.append(line)
    slides.sort()
    deck_fields = [f for _, f in hits if f]
    slide_text = "\n".join(s["text"] for s in deck if s["slide"] in slides)
    return {
        "slide_text": slide_text,
        "deck_slides": slides,
        "deck_found": bool(slides),
        "source_lines": source_lines,
        "source_primary": source_lines[0] if source_lines else None,
        "deck_fields": deck_fields,          # numeric fields as printed in the deck
    }
