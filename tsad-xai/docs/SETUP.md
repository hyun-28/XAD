# Setup — Mac (development) and Linux GPU server (execution)

BRIEF §2. One code base, two environment files that differ only in the CUDA build of torch.
No OS branches in code; paths come from `configs/*.yaml`.

## Mac (Apple Silicon)

```bash
conda env create -f environment.yml          # name: tsadxai
conda activate tsadxai
python -m pytest tests -q                    # 114 passed on 2026-09-22
```

Verified 2026-09-22 by building a fresh env from this file (`conda env create -f environment.yml
-n tsadxai-w1`, conda 26.1.1 / libmamba): python 3.11.16, numpy 1.26.4, **scikit-learn 1.5.2**,
**torch 2.3.0**, TSB_AD 1.5 — all 114 tests pass (DECISIONS D-B4-1). Exact resolutions:
`environment.lock.osx-arm64.txt` (conda explicit spec) and `requirements.lock.txt` (pip freeze).

> The pre-existing `tsadxai` env on the researcher's Mac carries torch 2.14.0 (installed by hand,
> not from the yml). It still passes the tests, but it is **not** the reference environment; the
> reference is the yml-built one. To replace it: `conda env remove -n tsadxai && conda env create
> -f environment.yml`.

## Linux GPU server (RTX A6000, CUDA 12.1)

```bash
conda env create -f environment.server.yml   # adds the nvidia channel + pytorch-cuda=12.1
conda activate tsadxai
python -m pytest tests -q
```

Not yet executed on the server (BRIEF §9 asks for one clean run there before Task C).
IForest needs no GPU; torch is only imported because `TSB_AD/utils/utility.py:19` does so at
module level.

## Load-bearing pins (do not relax without re-running the tests)

| pin | why | where verified |
|---|---|---|
| `scikit-learn<1.6` | TSB-AD's `BaseDetector` has no `__sklearn_tags__`; `check_is_fitted` inside `IForest.decision_function` raises on ≥ 1.6 | `docs/TSBAD_INTERNALS.md` §4, fallback `src/compat.py` |
| `numpy<2` | TSB-AD 1.5 declares `numpy>=1.24.3,<2.0` | dist-info METADATA |
| `torch==2.3.0` | version TSB-AD's own README installs; needed for import only | `environment.yml` |
| `TSB-AD==1.5` (PyPI) | no git tag exists for it; source hashes in `docs/TSBAD_INTERNALS.md` and every run-log header | `src/runlog.py` |

## Data

`configs/paths.yaml` points at the researcher's read-only local copy of the UCR 2021 archive and
at `data/raw/ucr/official/` where `scripts/01_download.py` puts the freshly downloaded official
zip and verifies it (`data/raw/ucr/CHECKSUMS.sha256`). Raw series are never committed.

## Run-log header

Every script prints `src/runlog.py::env_header()` first:
`python / numpy / sklearn / TSB-AD (+ 12-hex SHA256 prefix of four source files) / OS / hostname`.
