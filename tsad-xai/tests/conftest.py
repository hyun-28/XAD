import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_yaml, resolve  # noqa: E402


@pytest.fixture(scope="session")
def official_fulldata_dir() -> Path:
    """The official archive extracted by scripts/01_download.py, or skip."""
    cfg = load_yaml("paths")["ucr"]
    a1 = resolve(cfg["a1_result"])
    if not a1.is_file():
        pytest.skip("run scripts/01_download.py first (official archive not present)")
    anchor = Path(json.loads(a1.read_text())["anchors"]["official"])
    d = anchor / cfg["fulldata_subdir"]
    if not d.is_dir():
        pytest.skip(f"official archive dir missing: {d}")
    return d
