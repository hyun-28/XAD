# Šimić et al. metric port — function map and code/paper differences (BRIEF2 §D1)

Upstream: `https://github.com/perturbationeffect/cmi-am-validation-for-dl-ts-classifiers`
@ `edc6a870b1a50fce3385bfce6a468583d696ea75` (verified as the remote HEAD on 2026-09-22),
Apache-2.0. Fetched by `scripts/00_fetch_simic.py` into `third_party/simic_cmi/`
(git-ignored, 537 MB; DECISIONS D-D1-1) and never modified.

Our port: `src/metrics/faithfulness_simic.py`. Equivalence: `tests/test_simic_equivalence.py`.

## 1. Function map

| ours (`faithfulness_simic.py`) | upstream (`utils/res_utils.py`) | note |
|---|---|---|
| `decaying_degradation_score(morf, lerf, *, max_diff=100)` | `decaying_degradation_score` :51-78 | `max_diff` exposes the hard-coded `100` (:73); the default reproduces upstream |
| `compute_dataset_dds(morf_curves, lerf_curves, *, max_diff=100)` | `compute_dataset_dds` :31-49 | returns `np.ndarray` of per-sample DDS |
| `pes(dds_values)` | `pes` :81-99 | Kerby's simple difference `f − u` |
| `cmi(dds, pes_value)`, alias `CMI` | `CMI` :176-191 | harmonic mean of magnitudes, else 0 |
| `cubic_weights(n)` | inline in `decaying_degradation_score` :67-68 | factored out so the weighting is testable on its own |

**Not ported** (out of BRIEF2's scope, listed so nobody assumes they are missing by accident):
`degradation_score` :7-29 (the non-decaying variant), `rank_biserial` :101-115,
`combined_mean` :117-141, `combined_stddev` :143-174, `get_row_id`, `create_heatmap`,
`save_results_df`, `plot_barh_on_axis` (plotting/reporting helpers).

## 2. What the upstream code actually computes

```python
diffed         = lerf - morf                       # :65
linear_weights = arange(n, 0, -1) / n              # :67
cubic_weights  = linear_weights ** 3               # :68
dds            = average(diffed, weights=cubic)    # :70
max_diffed     = zeros_like(diffed); max_diffed[1:] = 100   # :72-73
dds_max        = average(max_diffed, weights=cubic)         # :74
return dds / dds_max                               # :76
```
- The **first curve point is unperturbed in both curves**, so it cannot contribute a
  difference; `max_diffed[0] = 0` encodes exactly that, and it is also why a length-1 curve
  has `dds_max == 0`.
- `100` is the normalisation constant: the code assumes the model emits the predicted class'
  probability scaled to 0-100 (comment on :73 and :24).
- PES counts strict inequalities, so an exact `DDS == 0` counts as neither favourable nor
  unfavourable and `|PES| < 1` whenever any DDS is 0.
- CMI branches on `pes * dds <= 0` (:187).

## 3. Code-internal discrepancies (checkable without the paper)

| # | where | what |
|---|---|---|
| C1 | `CMI` docstring (:178) vs branch (:187) | the docstring says "harmonic mean … if they have the same sign. Otherwise 0", but the branch is `pes * dds <= 0`, so an exact **zero** in either argument also returns 0 — "same sign" and "product > 0" differ exactly at 0. We follow the **code**. |
| C2 | `decaying_degradation_score` :72 | `np.zeros_like(diffed)` inherits `diffed`'s dtype; with integer-valued curves passed as Python ints, `max_diffed` is an int array. Harmless here (the ratio is float), but it means the constant is silently truncated if someone passes `max_diff=0.5` upstream. Our port forces float64 first. |
| C3 | `pes` :95-96 | `len(dds_vals[dds_vals > 0])` requires an array; a plain list is converted on :92-93 only if `isinstance(x, list)` — a tuple or generator raises. Our port uses `np.asarray` unconditionally. |
| C4 | `compute_dataset_dds` :43 | the length check is an `assert`, i.e. it disappears under `python -O`. Our port raises `ValueError`. |
| C5 | length-1 curves | upstream divides by `dds_max == 0` → `nan` + `RuntimeWarning`, silently propagating into dataset means. Our port raises (BRIEF §1-1). |

## 4. Paper vs code — Eq. 1-6 (DOI 10.1038/s41598-025-09538-2)

Šimić, I., Veas, E. & Sabol, V. "A comprehensive analysis of perturbation methods in explainable
AI feature attribution validation for neural time series classifiers." *Scientific Reports*
**15**:26607 (2025). Open access; PDF in `Šimić et al/` at the monorepo root (fetched 2026-09-22
from nature.com). Equation text below is from pp. 6-7 of that PDF.

| # | paper | code (`res_utils.py`) | verdict |
|---|---|---|---|
| Eq. 1-2 | `DDS = Σ_{i=1..n} (L_i − M_i)·((n−i+1)/n)³`, normalised by `DDS_max = Σ_{i=2..n} ((n−i+1)/n)³` | `np.average(diffed, weights=cubic) / np.average(max_diffed, weights=cubic)` with `max_diffed[1:] = 100` | **equivalent**, see P1 |
| Eq. 3-5 | `f = #(DDS>0)/#samples`, `u = #(DDS<0)/#samples`, `PES = f − u` | identical, zeros counted in neither | **identical** |
| Eq. 6 | `CMI = 2/(\|DDS\|⁻¹+\|PES\|⁻¹)` **if `DDS·PES ≥ 0`**, else 0 | `if pes*dds <= 0: return 0` else the harmonic mean | **branch conditions are opposite**, see P2 |

**P1 — the sum/average and the 100 cancel; the normalised DDS is the same number.**
The paper is a weighted *sum* over probabilities in [0, 1] divided by a weight sum; the code is a
weighted *average* over curve values on a 0-100 scale divided by another weighted average. Both
reduce to `Σ wᵢ dᵢ / (c · Σ_{i≥2} wᵢ)` with `c = 1` (paper) or `c = 100` (code), so the hard-coded
100 is a **unit conversion for probability×100 inputs, not a different formula**. Verified
numerically: on 2,000 random pairs, `paper(p ∈ [0,1])` vs `code(100·p, max_diff=100)` differ by at
most **8.9e-16**, and `paper(p)` vs `code(p, max_diff=1)` by at most **4.4e-16** (float noise).
This is what `max_diff` in our port is for: set it to 1 to read the paper's equations literally.

**P2 — Eq. 6 says `≥ 0`, the code says `<= 0` returns 0: opposite statements, same numbers.**
The disagreement is exactly the set `DDS·PES = 0`. The paper puts it in the harmonic-mean branch,
where `|DDS|⁻¹ → ∞` makes the expression tend to 0; the code sends it to the `0` branch directly.
Verified over the sign grid including `(0, 0.5)`, `(0.5, 0)` and `(0, 0)`: both give 0.0 exactly.
The code's guard is the **safer implementation** — Eq. 6 taken literally in Python raises
`ZeroDivisionError` on a float zero. BRIEF2 says to implement the code, and we do; the port keeps
`p * d <= 0`, and this row is the documented difference.

**Also noted:** the paper credits the non-decaying Degradation Score to Schulz et al. (ref. 46);
`degradation_score` :7-29 is that metric and is not used by the CMI pipeline (not ported, §1).
No further difference was found between Eq. 1-6 and the four ported functions.

## 5. Equivalence evidence (`tests/test_simic_equivalence.py`, 11 tests)

- 1,000 random curve pairs, length 2-60, values 0-100 (BRIEF2's specification): max absolute
  difference against the upstream functions is **exactly 0.0** — the assertion is
  `np.allclose(atol=1e-12, rtol=0)` *and* `max|Δ| == 0`.
- `compute_dataset_dds` over the same 1,000 pairs; `pes` over 200 random subsets;
  `CMI` over a 400 × 58 grid covering both signs, exact zeros, ±1 and ±1e-12.
- Boundary cases required by the brief: MoRF = LeRF → DDS = 0, PES = 0, CMI = 0; opposite
  signs → CMI = 0; fully reversed curves → DDS < 0 with CMI > 0; saturated curves → DDS = ±1.
- The upstream functions are imported from the pinned clone directly — they need only
  numpy/matplotlib/seaborn/math, so **MongoDB was never installed** (D-D1-2). The module skips
  if the clone is absent.

## 6. Anomaly-detection adaptation

Not implemented, by instruction (BRIEF2 §D2). `src/metrics/faithfulness_ad.py` holds the
signatures and raises `NotImplementedError`; the four blocking choices are registered as
**D7-D10** in `docs/DECISIONS.md` (tracked quantity, normalisation, direction, coverage).
