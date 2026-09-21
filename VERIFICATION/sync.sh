#!/usr/bin/env bash
# Re-copy the Task A verification artefacts from tsad-xai/ into this folder.
# Run after `python scripts/01_download.py && python scripts/02_audit.py` in tsad-xai/.
# Everything here is a COPY; the sources of truth are the files under tsad-xai/.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
src="$here/../tsad-xai"
rm -rf "$here/figures"
mkdir -p "$here/figures"
cp "$src/reports/data_audit.md"                    "$here/01_data_audit.md"
cp "$src/data/manifest/ucr_manifest.csv"           "$here/02_ucr_manifest.csv"
cp "$src/data/raw/ucr/CHECKSUMS.sha256"            "$here/03_CHECKSUMS.sha256"
cp "$src/data/raw/ucr/audit_a1.json"               "$here/04_audit_a1.json"
cp "$src/reports/index_convention_evidence.csv"    "$here/05_index_convention_evidence.csv"
cp "$src/reports/domain_unmatched.csv"             "$here/06_domain_unmatched.csv"
cp "$src/docs/DECISIONS.md"                        "$here/07_DECISIONS.md"
cp "$src/docs/DATA_LICENSE.md"                     "$here/08_DATA_LICENSE.md"
cp "$src/configs/conventions.yaml"                 "$here/09_conventions.yaml"
cp "$src/reports/figures/index_convention/"*.png   "$here/figures/"
# PYTHON defaults to the tsadxai env interpreter; override with PYTHON=... if needed.
PYTHON="${PYTHON:-$(command -v conda >/dev/null 2>&1 && conda run -n tsadxai which python 2>/dev/null || echo python)}"
( cd "$src" && "$PYTHON" -m pytest tests -q 2>&1 | tail -1 ) > "$here/10_pytest_result.txt" || true
echo "python: $PYTHON" >> "$here/10_pytest_result.txt"
date -u +"synced %Y-%m-%dT%H:%M:%SZ" >> "$here/10_pytest_result.txt"
echo "synced into $here"
