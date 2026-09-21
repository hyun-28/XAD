#!/usr/bin/env bash
# macOS setup. Run from the repo root:
#   bash scripts/setup_mac.sh
#
# WHY A SEPARATE SCRIPT: environment.yml pins pytorch-cuda=12.1 for the RTX
# A6000. That package does not exist for macOS and the env will fail to solve.
#
# INSTALL ORDER MATTERS. TSB-AD 1.5 declares scikit-learn>=1.3.2 with NO upper
# bound, so installing it pulls the latest sklearn. On sklearn>=1.6,
# check_is_fitted() raises inside every detector's decision_function() --
# verified failing on 1.8.0:
#     AttributeError: 'IForest' object has no attribute '__sklearn_tags__'
# fit() still works, so the break only shows up once you start scoring
# perturbations. We therefore install TSB-AD FIRST and pin sklearn back AFTER.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

ENV_NAME="${ENV_NAME:-tsadxai}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Install Miniforge first:"
  echo "  brew install --cask miniforge"
  exit 1
fi

echo "==> [1/6] create env"
conda env list | grep -q "^${ENV_NAME} " || conda create -y -n "$ENV_NAME" python=3.11
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "==> [2/6] core scientific stack  (enough to run ucr_inventory.py)"
pip install "numpy>=1.24.3,<2.0" "pandas>=2.0.3,<3.0" "scipy>=1.10" \
            "matplotlib>=3.7.5" seaborn statsmodels tqdm pytest

echo "==> [3/6] PyTorch for macOS (CPU / Apple MPS -- no CUDA)"
pip install torch torchvision torchaudio

echo "==> [4/6] TSB-AD and its dependencies"
pip install "TSB-AD==1.5" "hurst>=0.0.5" "arch>=5.3.1" einops torchinfo

echo "==> [5/6] pin scikit-learn back  (MUST come after TSB-AD)"
pip install "scikit-learn>=1.3.2,<1.6"

echo "==> [6/6] explainers"
# shap>=0.50 requires numpy>=2 -- incompatible with TSB-AD's numpy<2.0, and pip
# will install it without an error. Same class of trap as the sklearn pin.
pip install captum "shap<=0.49.1"

echo
echo "==> verifying the load-bearing pins"
python - <<'PY'
import sys
ok = True
import numpy, pandas, scipy, sklearn
print(f"  numpy        {numpy.__version__}")
if not numpy.__version__.startswith("1."):
    print("    FAIL  TSB-AD requires numpy<2.0"); ok = False
print(f"  pandas       {pandas.__version__}")
print(f"  scikit-learn {sklearn.__version__}")
maj, minor = (int(x) for x in sklearn.__version__.split(".")[:2])
if (maj, minor) >= (1, 6):
    print("    FAIL  decision_function() will raise AttributeError.")
    print("    fix:  pip install 'scikit-learn<1.6'")
    ok = False
try:
    import torch
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  torch        {torch.__version__}  device={dev}")
except ImportError:
    print("  FAIL torch missing -- TSB_AD/utils/utility.py imports torch.nn,"
          " so even IForest will not import."); ok = False
try:
    import TSB_AD
    print(f"  TSB_AD       importable")
except Exception as e:
    print(f"  FAIL TSB_AD import: {type(e).__name__}: {e}"); ok = False
sys.exit(0 if ok else 1)
PY

echo
echo "==> tests"
python tests/test_core.py

cat <<'EOF'

Done. Activate with:  conda activate tsadxai

Next:
  1) find the archive
       find ~ -type d -name "UCR_Anomaly_FullData" 2>/dev/null
  2) inventory it
       python scripts/ucr_inventory.py --root "<that path>" --n 40

Apple Silicon note: torch will use the MPS backend. TSB-AD's AutoEncoder does
not request a device explicitly, so it will run on CPU. That is fine at this
scale -- do not spend time on MPS until Stage 4.
EOF
