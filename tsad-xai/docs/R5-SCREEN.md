# R5 difficulty screen — first run on the real UCR archive

**Date:** 2026-09-07
**Script:** `scripts/screen_difficulty.py`
**Sample:** the 40 series in `configs/datasets.yaml` (rule set `v2`, AMENDMENT-001)
**Criterion:** the archive's own — the top-scoring timestep in the test portion
must fall within 100 points of the labelled anomaly (Wu & Keogh)
**Outputs:** `results/difficulty_screen.csv`, `results/difficulty_screen_kmeansad.csv`

---

## Headline

**The detector, not the dataset, decides whether this study is possible.**

| verdict | IForest | KMeansAD |
|---|---|---|
| `keep` (detector finds it, floor baseline does not) | **0** | **16** |
| `trivial` (N3_zscore finds it too) | 1 | 1 |
| `impossible` (detector misses) | 39 | 23 |

With IForest — the detector every result in this repo so far was produced with —
**not one of the 40 series is usable.**

## This is not a bug, and not a property of UCR

Four checks separate "the detector is weak" from "the harness is broken":

1. **The pipeline can land exactly on target.** `084_…s20101mML2` scores
   `detector_dist=6` under IForest, and KMeansAD scores `dist=0` on 13 series.
   Score/label alignment and TSB-AD's edge padding are therefore correct.
2. **Misses are not clustered at index 0**, which is what a padding artefact
   would produce. Distances range from 6 to 109,714.
3. **The training prefix is genuinely anomaly-free** for all 40 series under
   both detectors (`train_contaminated` is 0 everywhere), so the semi-supervised
   setup holds and the archive's guarantee is verified rather than assumed.
4. **The same 40 series, same code, different detector, gives 16 keeps.**

## The two detectors are not close

| | IForest | KMeansAD |
|---|---|---|
| median AUROC (test portion) | 0.619 | **0.912** |
| AUROC > 0.8 | 8 / 40 | **26 / 40** |
| AUROC < 0.5 (worse than chance) | **11 / 40** | 6 / 40 |

Tolerance sensitivity is the sharper diagnostic:

| tolerance | IForest hits | KMeansAD hits |
|---|---|---|
| 100 (archive default) | 1 | 17 |
| 500 | 6 | 18 |
| 1,000 | 11 | 18 |
| 5,000 | 20 | 26 |

KMeansAD is flat from 100 to 1,000 — it either finds the anomaly precisely or
misses it entirely, which is what real localization looks like. IForest's count
grows monotonically with the tolerance, meaning its peaks are scattered and any
"hit" is bought by widening the window. Reporting IForest at a loose tolerance
would manufacture a result.

## The verdicts nest perfectly

```
                  KMeansAD
IForest      impossible  keep  trivial
impossible           23    16        0
trivial               0     0        1
```

Every series KMeansAD can use is one IForest calls impossible, and there is no
series IForest handles that KMeansAD does not. IForest is strictly dominated
here; there is no complementarity argument for keeping it.

## What this costs the existing results

`results/exp_a.csv`, the W3 gate, and the SAR ranking were all produced with
**IForest on a synthetic sine series**. That was appropriate as a smoke test and
is still valid as one. But the screen shows IForest cannot detect anomalies in
the archive the study is actually about, so those numbers cannot be carried
forward as detector-relevant findings. Experiment A has to be re-run on a
detector that passes R5 before any of it means anything on real data.

This also puts a floor under the sample: **16 series, not 40.** Combined with
the two prior findings — SAR point estimates moving 30–45 % between window
sample sizes, and the pseudo-replication AMENDMENT-001 removed — the direction
is consistent. Every correction so far has widened the uncertainty, never
narrowed it.

## Decisions this raises (not taken here)

`screen_difficulty.py` **reports and does not filter**, by design: dropping
series changes the sample and belongs in a recorded amendment, not a script's
side effect. Three things now need deciding, and all three should be settled
before any real-data result is generated:

1. **Which detector.** KMeansAD is the only screened candidate that works.
   AutoEncoder is untested here (it is slow to fit and, per the README, cannot
   be used with Captum as shipped). If the study needs a gradient-based
   explainer, that constraint and this one have to be resolved together.
2. **Whether R5 filters.** Screening to 16 series is defensible and is what R5
   was written for, but it means either accepting n=16 or drawing more
   candidates. Rule set `v2` leaves 211 candidates, so raising `--n` and
   re-screening is available.
3. **Whether the screen is detector-conditional.** A series that is impossible
   for one detector and easy for another is not "hard" in any absolute sense.
   If more than one detector is studied, R5 has to be defined against the set —
   for example, keep a series if any studied detector passes it — or the sample
   silently becomes detector-specific.

## Reproducing

```bash
python scripts/screen_difficulty.py --detector IForest
python scripts/screen_difficulty.py --detector KMeansAD \
    --out results/difficulty_screen_kmeansad.csv
```
