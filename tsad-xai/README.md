# TSAD-XAI: perturbation-based faithfulness evaluation is confounded in anomaly detection

Working repository for the study described in `TSAD-XAI_파이프라인_설계서.md`.

## Task A — UCR data-lineage audit (BRIEF §3) — DONE 2026-09-21; follow-up F1–F7 2026-09-22

Reproduce (needs network once, for the 184 MB official zip and the Goswami repo):

```bash
conda env create -f environment.yml && conda activate tsadxai
python scripts/01_download.py && python scripts/02_audit.py
python -m pytest tests -q
```

**Exit codes of `02_audit.py`:** 0 clean · 1 WARN · 2 STOP. On the unmodified archive it exits
**0** since the researcher decisions of 2026-09-22 (`docs/DECISIONS.md` D-F2-2, D-F3-4): the
follow-up brief's STOP-2 (recording re-use across names, §7d) and STOP-3 (123 ECG4, §8) are
resolved and the related flags are INFO.

## Task B — TSB-AD safe wrapper (BRIEF §4) — DONE 2026-09-22

| artefact | what |
|---|---|
| `docs/TSBAD_INTERNALS.md` | B1: the IForest path in TSB-AD 1.5 traced with line numbers + measurements; benchmark HP trace |
| `src/detectors/` | `base.py` (`DetectorError`, `FittedDetector`, `fit_detector`, B3 `validate_scores`), `iforest.py` (fit once / `decision_function` many, `normalize=False` forced, seed via `random_state`), `legacy.py` (pre-Task-B `ScoreFn`, re-exported) |
| `src/data/ucr.py::slice_fit` | the single place that applies `fit_on` (`train_prefix` / `full`) |
| `configs/detectors.yaml` | benchmark-traced defaults (window 100, 200 trees, `max_features` int 1), the two researcher variants (`max_features` 1 vs 1.0, `fit_on`), seeds, `constant_score` |
| `src/runlog.py` | env header with TSB-AD source hashes; `CallLog` → `runs/<run_id>/calls.jsonl`, `failures.jsonl` |
| `environment.yml`, `environment.server.yml`, `*.lock.*` | Mac/CPU vs CUDA server; lock files from a fresh build (`docs/SETUP.md`) |
| `tests/` | B6: `test_wrapper_errors`, `test_wrapper_determinism`, `test_fit_score_consistency` (bitwise equality with TSB-AD's own path, and the `normalize=True` mismatch reproduced), `test_padding`, `test_slice_fit`, `test_runlog` — 114 tests pass in the yml-built env |

## Task C — Gate 1 (BRIEF §5) — RUN 2026-09-22, **NOT MET → STOP (§7-6)**

```bash
python scripts/03_gate1.py --workers 4      # 250 × (fit_on 2 × max_features 2 × seeds 3) = 3,000 fits, ~12 min on the Mac
python scripts/04_report.py                 # reports/gate1.md; exit 3 = gate not met
```

| artefact | what |
|---|---|
| `src/gate1.py` | pre-registered criteria P1 (99th-percentile of normal test scores, buffer 0 / w) and P2 (archive rule + BRIEF rule) |
| `results/gate1/detection.csv` (committed) / `scores/*.npy` (1.7 GB, ignored) / `summary.json` | per series × condition results; raw scores kept for the Artifact Index |
| `reports/gate1.md` | judgement, per-condition table, domain / variant / subset breakdowns, seed variance, paired comparisons, runtime, failures |
| `runs/<run_id>/` (ignored) | `calls.jsonl` (6,000 calls, 0 failures), `failures.jsonl`, `env.txt` |

Headline: **IForest failed the gate** — primary condition (`train_prefix`, `max_features=1`,
buffer = w) P1 = **28.4 %** (seed median), and every condition is below 50 % (best:
`full` / `max_features=1.0`, 39.2 %). Per the pre-registered rule the detector was replaced
(D-C5-1/2).

## Gate 1, second detector — MatrixProfile — **PASS** (2026-09-22)

```bash
python scripts/03_gate1.py --detector MatrixProfile --workers 4   # 250 × 2 conditions, ~85 min
python scripts/04_report.py --detector MatrixProfile              # reports/gate1_MatrixProfile.md
```

TSB-AD's `MatrixProfile` is self-join only and has no `decision_function`, so the AB-join the
gate needs is built on stumpy directly and proved identical to TSB-AD's path in the self-join
setting (`max|Δ| = 0.0`; D-C5-3, `src/detectors/matrixprofile.py`).

| condition | P1 (buffer = w) | P2 archive | window-support diagnostic |
|---|---|---|---|
| `train_prefix` (AB-join vs the training prefix) — **primary** | **79.2 %** | 54.0 % | 83.2 % |
| `full` (self-join) | 82.8 % | 58.8 % | 86.8 % |

Window = `find_length_rank(rank=1)` per series (min 6, median 125, max 276), logged per row.
Deterministic: the run was repeated and all 35 shared columns matched. The pre-registered
training-region sanity check holds (max self-match score 3.3e-04, D-C5-4).

Data terms: the archive carries no licence; raw series are never committed — download the official
zip and let `01_download.py` verify it against `data/raw/ucr/CHECKSUMS.sha256` (`docs/DATA_LICENSE.md`).

| artefact | what |
|---|---|
| `reports/data_audit.md` | the audit report (A1 checksum table, A3 findings, cross-tabs, GT and index-convention evidence) — script output, never hand-edited |
| `data/manifest/ucr_manifest.csv` | 250 rows, BRIEF A6 columns (+ provenance columns; A4 columns are `NA`) |
| `data/raw/ucr/CHECKSUMS.sha256` | zip + per-file SHA256 of the official archive |
| `docs/ucr_supplement_text.md` | text of every slide deck and MATLAB file shipped in the zip |
| `docs/DECISIONS.md` | every choice made, with revert instructions |
| `docs/DATA_LICENSE.md` | what the archive does (not) say about its licence |
| `configs/conventions.yaml` | the index convention in force (D3) and its evidence |
| `configs/goswami_entity_to_family.json` | Goswami's own UCR domain mapping, with commit/file/line |
| `reports/figures/index_convention/` | plots of the 20 shortest anomalies under both index readings |
| `reports/recording_groups.csv` | F2: every compared pair (same-name + cross-name within domain/±1 % length), r and verdict |
| `configs/stats.yaml`, `configs/subsets.yaml`, `configs/sentinels.yaml` | F2/F4/F3 parameters: unit of analysis (`name_group`), subset filters, sentinel rules |
| `data/derived/sentinels/<num>.npy` | F3: exact −999 indices outside the GT, per series (0-based; empty for 227 series) |
| `reports/figures/sentinels/` | F3: −999 positions in a plain series vs its DISTORTED/NOISE twins |
| `docs/briefs/BRIEF_A-followup.md` | the follow-up brief (F1–F7, STOP-1..3) |

Headline results: local copy ≡ official archive (266/266 files identical); 250 series parse,
load, and pass every A3 check (0 STOP, 0 WARN); the filename fields are documented by the
archive's own slides as MATLAB 1-based inclusive indices (`[begin-1, end)` in 0-based
half-open form) — set as the default in `configs/conventions.yaml`, **researcher sign-off
pending (D3)** → approved 2026-09-22 (D-A2-3); non-medical subset = 100 / 59 without DISTORTED.

Follow-up headline (2026-09-22, all numbers from `reports/data_audit.md` §7–§9): 100 `name_group`s
vs 89 `content_group`s — 18 content groups span several names because the archive re-used one
recording with different injected anomalies → **unit of analysis = `content_group`** (D-F2-2),
`name_group` for sensitivity; the 23 series containing −999 are ±2047-scale signals in which −999
is an ordinary sample (0 of 823 points are local spikes at k=10) → **treated as normal data**
(D-F3-4), `no_sentinel` kept as a sensitivity subset; `physical` = 42 series / 7 content groups,
`non_medical` = 100 / 22 (both descriptive only); legacy loader equivalence holds on 250/250 (F5).

## Task D — Šimić et al. metric port (BRIEF2) — D1/D2 DONE 2026-09-22

```bash
python scripts/00_fetch_simic.py            # pinned clone into third_party/ (537 MB, git-ignored)
python -m pytest tests/test_simic_equivalence.py -q
```

| artefact | what |
|---|---|
| `src/metrics/faithfulness_simic.py` | DDS / PES / CMI reimplemented from `utils/res_utils.py` @ `edc6a870`; `max_diff` exposed, default 100 as upstream |
| `tests/test_simic_equivalence.py` | 1,000 random curve pairs — max abs difference **exactly 0.0**; CMI sign grid; the brief's boundary cases |
| `docs/SIMIC_PORT.md` | function map, code-internal discrepancies (C1–C5), **paper Eq. 1–6 vs code** (DOI 10.1038/s41598-025-09538-2): Eq. 6's branch is `≥0` where the code has `<=0` — opposite conditions, same numbers (D-D1-4) |
| `src/metrics/faithfulness_ad.py` | AD adaptation: signatures + `NotImplementedError`; D7–D10 registered undecided |

D3 (Wafer model reproduction, `reports/wafer_zero_class.csv`) is not done yet.

---

**Status: environment verified, code tested, experiments not yet run on real data.**
Everything here except the dataset download has been executed and passes.

---

## What was verified (2026-09-07)

TSB_AD 1.5 source (PyPI sdist) was read line by line and exercised on synthetic
data. Six things were confirmed; four of them are traps that would have cost
days if hit during W3.

### 1. `run_Unsupervise_AD` refits on every call — CONFIRMED

```python
def run_IForest(data, slidingWindow=100, ...):
    clf = IForest(...)
    clf.fit(data)                 # <-- every single call
    return clf.decision_scores_.ravel()
```

Measured on L=3000, IForest: `fit` 0.154 s vs `decision_function` 0.020 s, a
**7.7x** gap that grows with model cost. Stage 3 makes ~500 scoring calls per
series; over 40 series this is the difference between hours and days.
`src/detectors.ScoreFn` fits once and calls `decision_function` thereafter.

Worse: `run_Unsupervise_AD` wraps everything in a bare `except:` and **returns
an error string instead of raising**. In a loop that produces silent garbage.
Never call it from experiment code.

### 2. Score length and sign — CONFIRMED

`len(score) == len(data)`. TSB-AD edge-pads short window scores itself:
`ceil((w-1)/2)` copies of the first value at the front, `(w-1)//2` at the back.
Higher means more anomalous (`invert_order` is applied internally).

### 3. `IForest.fit` and `IForest.decision_function` disagree — CONFIRMED BUG

`fit()` z-score-normalises the windowed matrix. `decision_function()` does not.

| setting | `fit` == `decision_function` | max abs diff |
|---|---|---|
| `normalize=True` (default) | **No** | 0.0368 |
| `normalize=False` | Yes | 0.0 |

`ScoreFn` forces `normalize=False`. Without it every perturbation measurement
would be contaminated by a preprocessing mismatch.

### 4. scikit-learn >= 1.6 breaks `decision_function` — CONFIRMED BLOCKER

```
AttributeError: 'IForest' object has no attribute '__sklearn_tags__'
```

`fit()` still works, so this only surfaces once you start scoring perturbations.
Fix: pin `scikit-learn<1.6` (`environment.yml`). Fallback:
`src.compat.patch_tsb_ad()`.

### 5. Even statistical detectors need torch — CONFIRMED

`TSB_AD/utils/utility.py` does `import torch.nn as nn` at module level, and
`models/base.py` imports from it. IForest will not import without torch
installed. Also: TSB-AD's own `setup.py` declares `numpy>=1.24.3,<2.0`.

### 6. AutoEncoder cannot be used with Captum as shipped — CONFIRMED

`AutoEncoder.decision_function` runs under `torch.no_grad()`, so no gradients
flow. It also refits a `MinMaxScaler` on the input *inside* `decision_function`,
which means perturbing the input silently changes preprocessing globally rather
than only in the masked region — a confound for every perturbation experiment.
And it returns a **per-window** Euclidean distance
(`pairwise_distances_no_broadcast`), not a per-timestep error.

Consequences, all handled in code:
* gradient-based explainers need a custom differentiable score built on
  `model.model` (the inner `nn.Sequential`), not on `decision_function`
* baseline **N4** (per-timestep reconstruction error) is implemented in
  `src/baselines.py`, since the detector does not expose it
* the MinMaxScaler refit must be reported as a limitation or frozen

---

## Layout

```
src/
  compat.py          sklearn>=1.6 shim
  detectors.py       ScoreFn: fit once, score many; encodes traps 1-4
  perturbations.py   B1-B6 + IDENTITY control
  baselines.py       floor N1-N4, ceiling O1-O2
  metrics/
    artifact.py      Stage 3: Artifact Index, VR, window sampling
    faithfulness.py  Stage 5: incumbent protocol + signed variant + Kendall's W
    localization.py  Stage 6: LP, Lift, AUROC/AUPRC, normalised LP
experiments/
  exp_a_artifact.py  Stage 3, the W3 gate. Runs with --synthetic, no data needed.
scripts/
  setup.sh                    env + data + smoke test
  build_dataset_config.py     pre-registered selection rules -> configs/datasets.yaml
tests/test_core.py            17 property tests, all passing
```

## Quick start

```bash
bash scripts/setup.sh                                # env, data, tests
python tests/test_core.py                            # 17 passed
python experiments/exp_a_artifact.py --synthetic     # W3 gate, no data
python scripts/build_dataset_config.py --root data/TSB-AD-U --n 40
```

---

## Synthetic dry run — and one prediction already falsified

`exp_a_artifact.py --synthetic` (L=4000, AR=3%, IForest, n=40 windows):

| perturbation | AI (normal) | AI (anomaly) | **SAR** |
|---|---|---|---|
| B3_local_mean | −0.496 | −4.827 | **9.73** |
| B4_linear_interp | −0.247 | −2.385 | **9.65** |
| B1_zero | −0.516 | −4.916 | **9.53** |
| B5_gaussian | **+0.596** | −4.132 | 6.93 |
| B6_shuffle | +0.064 | +0.105 | **1.64** |
| IDENTITY | 0.000 | 0.000 | — |

All six perturbations shift the score significantly on **labelled-normal**
regions (bootstrap CI excludes 0). Sanity checks hold: IDENTITY gives exactly
0, and its VR is 0.05 on normal regions, which is what the 95th-percentile
threshold implies by construction.

**Two things the design document got wrong.**

*Direction.* The design predicted masking would push scores **up**. For
IForest, B1–B4 push them **down** (AI ≈ −0.25 to −0.52): flattening a region
makes it *smoother* than the oscillating normal signal, so it looks *more*
normal. This matters, because the incumbent protocol uses `|ΔPred|` and would
erase the sign entirely. Both variants are now computed in
`metrics/faithfulness.py`.

*B6.* The design predicted `B6_shuffle` would be the safest operator because it
preserves the local value distribution. It is the **worst**: SAR 1.64, because
shuffling a spike keeps the spike, so it fails to remove the anomaly at all.
Low artifact is worthless without signal.

This produced a metric that was not in the design:

> **SAR = |AI_anomaly| / |AI_normal|** — how much larger the real signal is than
> the artifact. SAR near 1 means the perturbation is mostly measuring itself.

These are synthetic-data numbers with one detector. They are a smoke test, not
a result. But they show the pipeline discriminates between operators, which is
the only thing W3 needs to establish.

---

## Prior-work check (2026-09-07) -- the claim is narrower than first stated

The original framing was "nobody uses anomaly labels as explanation ground
truth". That framing is WRONG and must not appear in the paper.

ALREADY DONE by others:
  * Toward Faithful Explanations in Acoustic Anomaly Detection (arXiv:2601.12660,
    Jan 2026) compares AE vs MAE with error maps, saliency, SmoothGrad, IG,
    GradSHAP and Grad-CAM, uses F-score against the true anomaly region as a
    localization measure, AND proposes a perturbation metric that replaces the
    highlighted region with its own RECONSTRUCTION to simulate normal input --
    a smarter artifact-avoiding operator than any of B1-B6 here.
    Their numbers: MAE saliency 0.63 at the 98th percentile, AE error map 0.55.
    Note the error map is essentially our N4 baseline, and it is competitive.
  * The visual-AD survey (arXiv:2302.06670) states that IoU / pixel-AUROC
    overlap with ground-truth anomaly masks is "frequently used", and that doing
    so "conflates localization performance with explanation faithfulness".
    The field already knows about the conflation and has not resolved it.
  * Exathlon (PVLDB 14) ships root-cause intervals + extended effect intervals
    for explanation-discovery evaluation.
  * Fun-TSG (arXiv:2604.14221, 2026) generates variable- and timestamp-level
    ground truth specifically because "TSB-AD alone does not suffice for
    meaningful attribution evaluation" -- their objection is about variable-level
    granularity, which does not apply to univariate series.
  * Conditional Attribution for RCA in TSAD (arXiv:2604.17616, 2026) scores
    explanations against ground-truth root-cause sensors.
  * A comprehensive analysis of perturbation methods for TS attribution
    validation (Sci Rep 2025, s41598-025-09538-2) systematically compares
    perturbation operators for time series classifiers, and taxonomises
    faithfulness evaluation into sanity / localization / retraining /
    perturbation families. Overlaps Experiment A -- read before W3.

STILL OPEN, as far as this search reached:
  * C2. Nobody varies the ground-truth perturbation and shows that the ranking
    of "most faithful explainer" changes. This is the sharpest contribution and
    it survives intact.
  * The disagreement analysis. arXiv:2601.12660 reports both a localization
    measure and a perturbation measure but never asks whether they rank the
    explainers differently.
  * C1 for TSAD specifically. The Sci Rep 2025 study covers classifiers.

REVISED POSITIONING: this is a bridge paper between two subfields that both
evaluate explanations and never compare their criteria. It is not the discovery
of a new ground truth.

## The Figure-1 hypothesis is already visible in published data (2601.12660)

Full text read 2026-09-07. Elrashid et al., Mila / Concordia / Université Laval.
Wood-planer acoustic AD, 46 expert-annotated recordings, 6 attribution methods,
AE vs MAE. Code and annotations public:
github.com/Maab-Nimir/Faithful-Explanations-in-Acoustic-Anomaly-Detection

They compute BOTH axes:
  * F-score  -- peaks vs expert-annotated 1-second intervals (plausibility)
  * FF       -- replace detected regions with the MODEL'S OWN RECONSTRUCTION,
                FF = max(1 - Error(X2_hat, X2)/Error(X1_hat, X1), 0)

THE RANKINGS INVERT, in their own summary sentence:
    "Saliency Map and SmoothGrad provide strong event localization, while
     Integrated Gradients and GradSHAP yield more faithful but less precise
     attributions."
Also: the top F-score is 0.63 from MAE *saliency* at the 98th percentile, while
the most faithful method is the MAE *error map*. Different winners per axis.

They report this as an aside. They do not quantify it, do not rank-correlate the
two axes, and draw no conclusion about protocol validity. That gap is ours.

AND THE CIRCULARITY REPEATS, in a new form:
  FF is defined through reconstruction error. The error-map explanation is
  DERIVED from reconstruction error. Replacing a region with the model's own
  output drives reconstruction error down almost by construction. So the error
  map is structurally advantaged on FF for the same reason Occlusion is
  structurally advantaged in arXiv:2601.19017 -- the ground truth shares a
  mechanism with one of the candidates.
  Two independent papers, one month apart, both circular in the same way,
  neither noticing it in the other. That is a pattern, not an anecdote, and it
  is the strongest version of contribution C2.

Caveats: single domain, 46 annotated clips (2 broken / 10 stuck / 34 uneven),
AE 0.885+-0.032 vs MAE 0.864+-0.048 AUC (difference within noise). Their
faithfulness reference is the Sci Rep 2025 perturbation study (s41598-025-09538-2).

## UCR properties -- CONFIRMED

From Baldán & García-Gil (Expert Systems 42:e13767, 2025):
  * each series contains exactly ONE anomaly, start and end known
  * train/test split point specified per series; training portion anomaly-free
  * lengths span 6,684 to 900,000 points
  * official scoring: correct if within 100 points of any anomaly point

Consequences applied to the code:
  * R3's upper bound raised to 900,000 (250,000 would still have cut UCR)
  * `src/data_ucr.py` reconstructs labels from the filename
    `<idx>_UCR_Anomaly_<name>_<train_end>_<start>_<end>.txt`; parser tested
  * `localization_precision_tolerant(..., tol=100)` added to match the archive's
    own protocol, so boundary sensitivity is reported rather than assumed away

## Corrected after the fact

The first draft of `scripts/build_dataset_config.py` capped series length at
60,000 (rule R3) on the theory that KernelSHAP cost scales with series length.

It does not. This project scores **sampled windows**, so cost scales with the
window count `n`, not with L. The cap was the wrong axis, and it would have
silently excluded most of the UCR Anomaly Archive -- mean length ~67,800 --
which is the primary dataset. The synthetic test file that "verified" the rules
was 12,000 long, so the conflict never surfaced.

Fixed: R3 is now [5,000, 250,000]; R2's floor moved from 0.5% to 0.4% because
UCR sits at ~0.6% and was hugging the boundary. Compute is bounded by `n`.

Added R7: `min_seg_len >= 10`. If the shortest anomalous segment is shorter than
the mask width, Experiment A's anomaly control arm comes back empty. That
failure already happened once during the dry run and produced a table of NaNs
with no error.

## Not verified

* **Dataset download.** `www.thedatum.org` was unreachable from the sandbox, so
  `scripts/setup.sh` step 3 has never been executed. Everything downstream of it
  is unrun on real data.
* **Filename field `tr_NNNN`.** Parsed as the train/test boundary index from
  filename structure. Not confirmed against TSB-AD documentation. `R1` (official
  Eval split membership) is therefore not yet enforced.
* **AutoEncoder + Captum.** Reasoned from source, not executed — the sandbox ran
  out of disk installing torch. This is the largest remaining risk: if the
  custom differentiable score cannot be built, gradient-based explainers drop
  out and the comparison with arXiv:2601.19017 loses three of its four methods.
* **KMeansAD.** `ScoreFn` supports it but it was never instantiated.
* **N4 with a real AutoEncoder.** Only the moving-average fallback path ran.
