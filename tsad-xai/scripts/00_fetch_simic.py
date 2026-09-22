"""00_fetch_simic.py -- restore third_party/simic_cmi at the pinned commit (BRIEF2 D1).

The upstream repository is 537 MB (trained models + UCR classification data), so it is
git-ignored here and fetched on demand. Idempotent: if the clone exists it is only verified.

Usage:  python scripts/00_fetch_simic.py [--shallow]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import REPO_ROOT  # noqa: E402

URL = "https://github.com/perturbationeffect/cmi-am-validation-for-dl-ts-classifiers"
COMMIT = "edc6a870b1a50fce3385bfce6a468583d696ea75"
DEST = REPO_ROOT / "third_party" / "simic_cmi"


def git(*args, cwd=None) -> str:
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shallow", action="store_true",
                    help="fetch only the pinned commit (no history); enough for the metric port")
    args = ap.parse_args()
    if (DEST / ".git").is_dir():
        head = git("-C", str(DEST), "rev-parse", "HEAD")
        if head != COMMIT:
            raise RuntimeError(f"{DEST} is at {head}, expected {COMMIT}. "
                               "Check out the pinned commit or delete the directory and re-run.")
        print(f"already present at {COMMIT}: {DEST}")
        return 0
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if args.shallow:
        git("init", str(DEST))
        git("-C", str(DEST), "remote", "add", "origin", URL)
        git("-C", str(DEST), "fetch", "--depth", "1", "origin", COMMIT)
        git("-C", str(DEST), "checkout", "FETCH_HEAD")
    else:
        git("clone", URL, str(DEST))
        git("-C", str(DEST), "checkout", COMMIT)
    head = git("-C", str(DEST), "rev-parse", "HEAD")
    if head != COMMIT:
        raise RuntimeError(f"checked out {head}, expected {COMMIT}")
    print(f"fetched {URL} @ {COMMIT} -> {DEST} (read-only; never modify it)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
