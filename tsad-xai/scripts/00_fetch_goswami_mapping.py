"""00_fetch_goswami_mapping.py -- BRIEF Task A, step A5 (priority 1).

Vendor the UCR entity -> data-family mapping from the official code of
Goswami et al. (ICLR 2023, arXiv:2210.01078), repository
https://github.com/mononitogoswami/tsad-model-selection (Apache-2.0), into
configs/goswami_entity_to_family.json together with the commit hash, file
path and line number it was taken from.

The JSON is committed, so the audit (02_audit.py) never needs network access;
this script exists to make the provenance reproducible and to detect drift
(re-run it and diff the JSON).

Usage:
  python scripts/00_fetch_goswami_mapping.py [--commit <sha>] [--workdir DIR]
"""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import REPO_ROOT  # noqa: E402
from src.runlog import env_header  # noqa: E402

REPO_URL = "https://github.com/mononitogoswami/tsad-model-selection"
SRC_FILE = "src/tsadams/model_trainer/entities.py"
DICT_NAME = "ANOMALY_ARCHIVE_ENTITY_TO_DATA_FAMILY"
# Pinned on 2026-09-21 (HEAD of the default branch at that date).
DEFAULT_COMMIT = "cf01d10af7b6d6ed9825fbd7aae2058af7c1d26b"
OUT = REPO_ROOT / "configs" / "goswami_entity_to_family.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", default=DEFAULT_COMMIT)
    ap.add_argument("--workdir", default=None, help="reuse an existing clone")
    args = ap.parse_args()
    print(env_header())

    if args.workdir:
        clone = Path(args.workdir)
    else:
        clone = Path(tempfile.mkdtemp(prefix="goswami_")) / "tsad-model-selection"
        subprocess.run(["git", "clone", "-q", REPO_URL, str(clone)], check=True)
    subprocess.run(["git", "-C", str(clone), "checkout", "-q", args.commit], check=True)
    head = subprocess.run(["git", "-C", str(clone), "rev-parse", "HEAD"],
                          check=True, capture_output=True, text=True).stdout.strip()
    if head != args.commit:
        raise RuntimeError(f"checked out {head}, expected {args.commit}")

    src_path = clone / SRC_FILE
    tree = ast.parse(src_path.read_text())
    node = next((n for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == DICT_NAME for t in n.targets)), None)
    if node is None:
        raise RuntimeError(f"{DICT_NAME} not found in {SRC_FILE}")
    mapping = ast.literal_eval(node.value)
    if not isinstance(mapping, dict) or not mapping:
        raise RuntimeError(f"{DICT_NAME} did not evaluate to a non-empty dict")

    families = sorted(set(mapping.values()))
    out = {
        "_provenance": {
            "repo": REPO_URL, "commit": head, "file": SRC_FILE,
            "symbol": DICT_NAME, "lines": f"{node.lineno}-{node.end_lineno}",
            "license": "Apache-2.0 (repository LICENSE)",
            "fetched_utc": datetime.now(timezone.utc).isoformat(),
            "n_entities": len(mapping), "families": families,
        },
        "entity_to_family": dict(sorted(mapping.items())),
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"{DICT_NAME}: {len(mapping)} entities, {len(families)} families "
          f"({SRC_FILE}:{node.lineno}-{node.end_lineno} @ {head[:12]})")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
