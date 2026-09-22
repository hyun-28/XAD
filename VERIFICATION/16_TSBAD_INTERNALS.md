# TSB-AD internals — the IForest path, traced line by line (BRIEF Task B, B1)

Read 2026-09-22 against the installed package **TSB-AD 1.5 (PyPI sdist)** in the `tsadxai`
env (`/opt/miniconda3/envs/tsadxai/lib/python3.11/site-packages/TSB_AD/`). The GitHub
repository `TheDatumOrg/TSB-AD` carries no release tags (`git ls-remote --tags` is empty,
2026-09-22), so there is no commit hash for 1.5; the reproducibility key is the SHA256 of the
four source files this document cites (DECISIONS D-A0-2, D-B1-1):

| file | sha256 |
|---|---|
| `TSB_AD/models/IForest.py` | `b59a8c7660b9bfa585462818e4709c621b56d0995c6d6019914da4138638fa5a` |
| `TSB_AD/model_wrapper.py` | `1702e9bd7d19f5667a5c126de0dffd6f991342d73658d0d3441814e857e1004f` |
| `TSB_AD/models/feature.py` | `e8a9858261c130d0c315977b805197c4921ed1a0bffd7425efcc130df9f45e2a` |
| `TSB_AD/utils/utility.py` | `58d5906ce8b5f2ccd350d9a04eeb3c7b636c473e64967ca36945f76e70eb5f03` |

Everything marked *measured* was run in this env (python 3.11.16, numpy 1.26.4,
scikit-learn 1.5.2) on a synthetic (3000, 1) series; the same checks are
`tests/test_fit_score_consistency.py`, `tests/test_wrapper_determinism.py`, `tests/test_padding.py`.

## 1. Entry point and the refit-on-every-call problem

`model_wrapper.py:48-53`
```python
def run_IForest(data, slidingWindow=100, n_estimators=100, max_features=1, n_jobs=1):
    from .models.IForest import IForest
    clf = IForest(slidingWindow=slidingWindow, n_estimators=n_estimators, max_features=max_features, n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()
```
- A fresh `IForest` is built and fitted on **every** call; the returned score is the
  training-set score (`decision_scores_`), i.e. `fit_on="full"` on whatever is passed.
- `random_state` is **not** forwarded; the constructor default `random_state=0`
  (`models/IForest.py:149`) applies. Seeds therefore have to go through the constructor —
  the wrapper does that.
- `normalize` is not forwarded either → default `True` (`IForest.py:151`). See §4.
- There is a second entry, `run_Sub_IForest` (`model_wrapper.py:40-46`), which sets
  `slidingWindow = find_length_rank(data, rank=periodicity)` (auto period from the ACF of the
  first 20,000 points, `utils/slidingWindows.py:7-…`). BRIEF C1 says "IForest", read as the
  fixed-window `run_IForest` path; the window is a config value (`configs/detectors.yaml`).

### 1a. What the benchmark actually passes (D-B2-1)

`benchmark_exp/Run_Detector_U.py` at GitHub `6beac72e` (2026-09-07; the installed `main.py`
is equivalent): `:46` `Optimal_Det_HP = Optimal_Uni_algo_HP_dict['IForest']` =
`{'n_estimators': 200}` (`HP_list.py:293`, installed `:240`) → `:71`
`run_Unsupervise_AD('IForest', data, **Optimal_Det_HP)` → `run_IForest(data, slidingWindow=100,
n_estimators=200, max_features=1, n_jobs=1)`. `:62` computes `find_length_rank` but hands it only
to `get_metrics` (`:87`). So the benchmark IForest = **window 100, 200 trees, `max_features`
int 1, fitted on the whole series, `normalize=True`, `random_state=0`.**

**Upstream change after 1.5:** at GitHub HEAD `6beac72e`, `IForest.decision_function` z-scores
the window matrix when `normalize` is set (IForest.py:243-247), i.e. the fit/score mismatch
described in §4 is fixed there. The PyPI 1.5 sdist installed here does not contain the fix
(the two `model_wrapper.py` files also differ). `normalize=False` behaves identically in both.

## 2. The silent-failure path

`model_wrapper.py:10-19`
```python
def run_Unsupervise_AD(model_name, data, **kwargs):
    try:
        function_name = f'run_{model_name}'
        function_to_call = globals()[function_name]
        results = function_to_call(data, **kwargs)
        return results
    except:
        error_message = f"Model function '{function_name}' is not defined."
        print(error_message)
        return error_message
```
- Bare `except:` — **any** exception inside the detector (bad shape, window larger than the
  series, NaN, MemoryError…) is swallowed, printed, and **a `str` is returned** with a
  misleading message. This is the exact code location BRIEF B3 row 1 refers to.
- `run_Semisupervise_AD` (`model_wrapper.py:22-30`) has the same structure.
- The wrapper never calls `run_Unsupervise_AD`; it constructs `IForest` directly (§6) and
  still validates every returned score as if it could be a string, so a future TSB-AD change
  cannot reintroduce the trap.

## 3. Constructor arguments and defaults (`models/IForest.py:139-163`)

| argument | default | forwarded to sklearn `IsolationForest`? | note |
|---|---|---|---|
| `slidingWindow` | 100 | no (used by `Window`) | window length in samples |
| `n_estimators` | 100 | yes | TSB-AD's "optimal" univariate HP is 200 (`HP_list.py:240`); `run_IForest` default is 100 |
| `sub` | True | no | **unused** anywhere in the class |
| `max_samples` | "auto" | yes | sklearn: min(256, n_windows) |
| `contamination` | 0.1 | yes | only affects `threshold_`/`labels_` (`base.py:421-…`), not the scores |
| `max_features` | 1. | yes | `run_IForest` passes `1` (int → 1 feature per tree!) — see §3a |
| `bootstrap` | False | yes | |
| `n_jobs` | 1 | yes | |
| `behaviour` | 'old' | **no** (accepted, never used) | legacy pyod arg |
| `random_state` | 0 | yes (`IForest.py:205`) | the only stochastic element |
| `verbose` | 0 | yes | |
| `normalize` | True | no | z-score of the window matrix in `fit` only (§4) |

### 3a. `max_features=1` (int) vs `1.` (float)
`run_IForest`'s signature has `max_features=1` (int), the class default is `1.` (float).
sklearn's `IsolationForest`: int → *that many* features per tree (1 of the 100 window
columns); float → that fraction (1.0 = all 100 columns). The TSB-AD benchmark therefore runs
IForest with **one random window position per tree**. The wrapper exposes `max_features` and
defaults to `run_IForest`'s value (int 1) so scores match the benchmark path; the difference
is recorded in `configs/detectors.yaml` and DECISIONS D-B2-2 (researcher decision; gate 1 runs both).

## 4. `fit` (`IForest.py:165-218`) vs `decision_function` (`IForest.py:220-250`)

| step | `fit` | `decision_function` |
|---|---|---|
| input shape | `n_samples, n_features = X.shape` — **must be 2-D** (`(n, 1)`); a 1-D array raises `ValueError: not enough values to unpack` (*measured*) | same |
| windowing | `Window(window=w).convert(X)` → `(n − w + 1, w)` via `sliding_window_view` (`feature.py:26-29`); `w > n` raises `ValueError: window shape cannot be larger than input array shape` (*measured*) | same |
| normalisation | `if self.normalize: zscore(X, axis=0, ddof=0)` for univariate (`IForest.py:185-187`; `utility.py:24-35`, `nan_to_num` on the result) | **none** (`IForest.py:240-245`) |
| model | new `IsolationForest(..., random_state=self.random_state)` (`:199-206`), `fit` (`:208`) | reuses `self.detector_` |
| score | `invert_order(detector_.decision_function(X))` = `−score` (`utility.py:474-…`, method 'multiplication') → **higher = more anomalous** | same |
| padding | if `len < n_samples`: `ceil((w−1)/2)` copies of the first value in front, `(w−1)//2` copies of the last value at the back (`:214-216`) → `len == n_samples` (*measured*: 3000 → 3000) | identical rule (`:247-249`) |
| post | `_process_decision_scores()` → `threshold_`, `labels_` from `contamination` (`base.py:421-…`) | nothing |

**The `normalize` trap (BRIEF B2):** with `normalize=True` the forest is trained on z-scored
windows but scores raw windows at `decision_function` time. *Measured*: max |fit score −
decision_function score| = **0.0167** with `normalize=True`, **0.0** with `normalize=False`
(earlier measurement 2026-09-07 on another series: 0.0368). The wrapper forces
`normalize=False` and raises if a caller passes `True`.

**Padding is "edge" padding, not zeros**: the first window's score is repeated
`ceil((w−1)/2)` = 50 times for w = 100, the last window's score 49 times. Window i covers
samples `[i, i+w)`; after padding, index t (50 ≤ t ≤ n−50) carries the score of the window
that *starts* at t−50, i.e. the window centred on t (for even w, centred between t−1 and t).
This alignment feeds both the Artifact Index (which region a score belongs to) and
localization metrics; `tests/test_padding.py` pins it.

**No MinMax scaling inside the detector.** TSB-AD's benchmark driver applies
`MinMaxScaler(feature_range=(0,1))` to the score *before* computing metrics
(`main.py:55`), not inside `IForest`. The wrapper returns raw scores; any scaling is a
downstream, explicit step.

## 5. Randomness and determinism (*measured*)

- The only RNG is sklearn's `IsolationForest(random_state=…)`; `Window`, `zscore`,
  `invert_order`, padding are deterministic. No `np.random.*` calls in `IForest.py`,
  `feature.py`, `base.py`, `utility.py`, `slidingWindows.py`, `model_wrapper.py` (grep).
- `fit` does **not** consume the global `np.random` state (state before/after identical).
- Same `random_state` twice → bitwise-identical `decision_scores_`; different seed → different.
- `decision_function` after one `fit` is deterministic and does not refit
  (`fit` ≈ 0.15 s vs `decision_function` ≈ 0.02 s on n = 3000, 2026-09-07 measurement).

## 6. What the wrapper does with all this (`src/detectors/`)

- Builds `IForest(slidingWindow, n_estimators, max_features, n_jobs, random_state=seed, normalize=False)`
  directly — never `run_Unsupervise_AD`.
- `fit_detector(name, x_fit, seed=…, **hp)` fits once on `x_fit` reshaped to `(n, 1)` and
  returns a `FittedDetector` whose `score(x)` calls `decision_function` only.
- `fit_on`: `"train_prefix"` (default, `x[:train_end]`) or `"full"` (TSB-AD behaviour).
- Every `score()` output goes through the B3 checks (string → `DetectorError`, shape, length ==
  len(x), finite, variance > 0, float64).
- sklearn ≥ 1.6 would break `check_is_fitted` inside `decision_function` (`IForest.py:238`);
  the env pins `scikit-learn<1.6` and `src/compat.py::patch_tsb_ad` is the documented fallback.

## 7. Dependency pins (from the installed metadata, `TSB_AD-1.5.dist-info/METADATA`)

- `scikit-learn >=1.3.2` (no upper bound) → no conflict with the project's `<1.6` pin
  (BRIEF §7-3 not triggered).
- `numpy <2.0,>=1.24.3`; `torch >=1.8.0` is a hard requirement even for IForest
  (`utils/utility.py` imports `torch.nn` at module level).
