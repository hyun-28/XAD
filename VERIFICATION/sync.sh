#!/usr/bin/env bash
# Re-copy the Task A (+ follow-up F1-F7) verification artefacts from tsad-xai/ into this folder.
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
# --- follow-up brief (2026-09-22) ---
cp "$src/reports/recording_groups.csv"             "$here/11_recording_groups.csv"
cp "$src/configs/stats.yaml"                       "$here/12_stats.yaml"
cp "$src/configs/subsets.yaml"                     "$here/13_subsets.yaml"
cp "$src/configs/sentinels.yaml"                   "$here/14_sentinels.yaml"
cp "$src/docs/briefs/BRIEF_A-followup.md"          "$here/15_BRIEF_A-followup.md"
# --- Task B / C (2026-09-22) ---
cp "$src/docs/TSBAD_INTERNALS.md"                  "$here/16_TSBAD_INTERNALS.md"
cp "$src/configs/detectors.yaml"                   "$here/17_detectors.yaml"
cp "$src/reports/gate1.md"                         "$here/18_gate1.md"
cp "$src/results/gate1/detection.csv"              "$here/19_gate1_detection.csv"
cp "$src/docs/SETUP.md"                            "$here/20_SETUP.md"
cp "$src/docs/SIMIC_PORT.md"                       "$here/21_SIMIC_PORT.md"
mkdir -p "$here/figures/sentinels"
cp "$src/reports/figures/sentinels/"*.png          "$here/figures/sentinels/"
# PYTHON defaults to the tsadxai env interpreter; override with PYTHON=... if needed.
# Prefer the env built from environment.yml (tsadxai-w1, 2026-09-22), then the older tsadxai.
PYTHON="${PYTHON:-$( { [ -x /opt/miniconda3/envs/tsadxai-w1/bin/python ] && echo /opt/miniconda3/envs/tsadxai-w1/bin/python; } || { command -v conda >/dev/null 2>&1 && conda run -n tsadxai which python 2>/dev/null; } || echo python)}"
( cd "$src" && "$PYTHON" -m pytest tests -q 2>&1 | tail -1 ) > "$here/10_pytest_result.txt" || true
echo "python: $PYTHON" >> "$here/10_pytest_result.txt"
date -u +"synced %Y-%m-%dT%H:%M:%SZ" >> "$here/10_pytest_result.txt"
echo "synced into $here"
