"""Domain labels (BRIEF A5): Goswami mapping, official-source category, flags."""
from __future__ import annotations

import json
import re
from pathlib import Path

from src.config import REPO_ROOT, load_yaml

GOSWAMI_JSON = REPO_ROOT / "configs" / "goswami_entity_to_family.json"


def load_goswami() -> tuple[dict[str, str], dict]:
    """entity -> short label, plus the provenance block."""
    if not GOSWAMI_JSON.is_file():
        raise FileNotFoundError(f"{GOSWAMI_JSON} missing; run scripts/00_fetch_goswami_mapping.py")
    blob = json.loads(GOSWAMI_JSON.read_text())
    short = load_yaml("domains")["goswami_short_labels"]
    mapping = {}
    for entity, family in blob["entity_to_family"].items():
        if family not in short:
            raise KeyError(f"no short label for Goswami family {family!r} (configs/domains.yaml)")
        mapping[entity] = short[family]
    return mapping, blob["_provenance"]


def official_domain(source_lines: list[str], fallback_text: str = "") -> tuple[str | None, str]:
    """(label, basis) from the deck's statements.

    basis is 'source_line' when a stated source sentence matched, 'slide_text'
    when only the slide's other prose matched (e.g. the anomaly description
    names the signal), or 'none'.
    """
    rules = load_yaml("domains")["official_source_rules"]
    for line in source_lines:
        for r in rules:
            if re.search(r["pattern"], line, re.IGNORECASE):
                return r["label"], "source_line"
    if fallback_text:
        for r in rules:
            if re.search(r["pattern"], fallback_text, re.IGNORECASE):
                return r["label"], "slide_text"
    return None, "none"


def is_medical(domain_goswami: str | None) -> bool | None:
    if domain_goswami is None:
        return None
    return domain_goswami in set(load_yaml("domains")["medical_domains"])
