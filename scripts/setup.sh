#!/usr/bin/env bash
# One-shot environment setup. Run from the repo root on the WSL2 machine.
#   bash scripts/setup.sh
#
# Everything except the dataset download was verified in a sandbox on
# 2026-09-07. The dataset host (www.thedatum.org) was not reachable from that
# sandbox, so step 3 is the one step that has NOT been executed end to end.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

echo "==> [1/5] conda environment"
if conda env list | grep -q '^tsadxai '; then
  echo "    tsadxai already exists; updating"
  conda env update -n tsadxai -f environment.yml --prune
else
  conda env create -f environment.yml
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tsadxai

echo "==> [2/5] verifying the load-bearing pins"
python - <<'PY'
import sys, numpy, sklearn
ok = True
if not numpy.__version__.startswith("1."):
    print(f"  FAIL numpy {numpy.__version__} -- TSB-AD requires <2.0"); ok = False
else:
    print(f"  ok   numpy {numpy.__version__}")
maj, minor = (int(x) for x in sklearn.__version__.split(".")[:2])
if (maj, minor) >= (1, 6):
    print(f"  FAIL scikit-learn {sklearn.__version__} -- decision_function will"
          " raise AttributeError. Pin <1.6 or call src.compat.patch_tsb_ad().")
    ok = False
else:
    print(f"  ok   scikit-learn {sklearn.__version__}")
try:
    import torch
    print(f"  ok   torch {torch.__version__} cuda={torch.cuda.is_available()}")
except ImportError:
    print("  FAIL torch missing -- note TSB_AD/utils/utility.py imports torch,"
          " so even the statistical detectors will not import without it.")
    ok = False
sys.exit(0 if ok else 1)
PY

echo "==> [3/5] datasets  (NOT verified in sandbox -- host unreachable there)"
mkdir -p data && cd data
for NAME in TSB-AD-U TSB-AD-M; do
  if [ -d "$NAME" ]; then
    echo "    $NAME present, skipping"
  else
    echo "    downloading $NAME"
    wget -c "https://www.thedatum.org/datasets/${NAME}.zip"
    unzip -q "${NAME}.zip"
  fi
done
cd "$REPO"

echo "==> [4/5] official Tuning/Eval split lists"
if [ ! -d Datasets/File_List ]; then
  git clone --depth 1 --filter=blob:none --sparse \
      https://github.com/TheDatumOrg/TSB-AD.git .tsbad_tmp
  ( cd .tsbad_tmp && git sparse-checkout set Datasets/File_List )
  mkdir -p Datasets && cp -r .tsbad_tmp/Datasets/File_List Datasets/
  rm -rf .tsbad_tmp
fi
ls Datasets/File_List | head

echo "==> [5/5] tests + synthetic smoke run"
python tests/test_core.py
python experiments/exp_a_artifact.py --synthetic --n 40

cat <<'EOF'

Setup complete.

Next, in order:
  python scripts/build_dataset_config.py      # applies the 6 selection rules
  python experiments/exp_a_artifact.py --csv data/TSB-AD-U/<file>.csv

If step 3 failed, download TSB-AD-U.zip / TSB-AD-M.zip manually from
https://www.thedatum.org/datasets/ and unzip into ./data/
EOF
