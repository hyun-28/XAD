# AMENDMENT-001 — dataset selection rules R2 and R8

**Date:** 2026-09-07
**Applies to:** `scripts/build_dataset_config.py`, rule set `v1` → `v2`
**Status:** adopted; `v2` is the default, `v1` remains reproducible via `--ruleset v1`

---

## Why this document exists

`build_dataset_config.py` opens by saying the selection rules are pre-registered
so that the answer to "why these series?" is "here is the script", and it warns
that changing the selection after results are seen is cherry-picking.

Two defects in those rules were found while pointing the script at the real UCR
archive for the first time. Fixing them changes which series are studied, so the
fix cannot be a silent edit. This is the record.

### The timing claim, and how to check it

**No experimental results on real data existed when this amendment was made.**
That is verifiable, not asserted:

- `results/exp_a.csv` is the synthetic smoke test — `experiments/exp_a_artifact.py
  --synthetic`, a generated sine series, no archive file involved.
- Nothing else in `results/` predates this amendment except dataset inventories,
  which are counts of files, not outcomes.
- Before this work, `configs/datasets.yaml` pointed at `root: /tmp/fakeU` and
  listed 3 fabricated TSB-AD filenames. The archive had never been read.
- `git log` — the baseline commit `efc7b61` contains that placeholder config.

Both defects were also found from **archive composition alone** (filenames and
series lengths). Neither required fitting a detector or looking at any outcome.

---

## Defect 1 — R2's lower bound contradicts its own stated purpose

### The rule as pre-registered

```
R2  0.4% <= anomaly ratio <= 10%
    lower bound set just below UCR's ~0.6% so the primary archive survives
```

The intent is explicit: the bound exists to **let UCR through**.

### What the archive actually contains

Measured over all 250 files (anomaly length from the filename, series length by
token count):

| statistic | value |
|---|---|
| median anomaly ratio | **0.34 %** |
| min | 0.0005 % |
| max | 4.9 % |

The premise "UCR's ~0.6%" is wrong. A 0.4 % bound sits **above the median**, so
the rule does the opposite of what it was written to do.

| R2 lower bound | pass R2 | pass all rules |
|---|---|---|
| **0.004 (v1)** | 103 / 250 | **100 / 250** |
| 0.002 | — | 165 / 250 |
| 0.001 | — | 194 / 250 |
| **0.0005 (v2)** | — | **211 / 250** |
| 0.0001 | — | 216 / 250 |

### Decision

**Lower bound 0.004 → 0.0005.**

This is a correction to a clerical error, not a threshold tuned toward a result.
The stated intent of the rule is unchanged and is now actually served. `0.0005`
is chosen because it is the smallest bound that still admits essentially the
whole archive while remaining a real constraint — going to `0.0001` buys only 5
more series (216 vs 211) and stops excluding anything.

The upper bound (10 %) is untouched; nothing in UCR approaches it.

### What protects against this being cherry-picking

The anomaly ratio distribution is a property of the archive that is fixed before
any modelling. It cannot be moved by a result, and no result existed. The
alternative — leaving a bound that was written on a false premise and silently
discards 60 % of the primary dataset — is the worse scientific outcome.

---

## Defect 2 — the sample contained pseudo-replicates

### What was found

Under `v1`, the 40 selected series were only **33 unique recordings**. UCR ships
the same recording several times under `FOO`, `DISTORTEDFOO` and `NOISEFOO`,
differing only in preprocessing. Seven such pairs were selected, each sharing
**the same signal and the same anomaly interval**:

```
1sddb40 [52000,52620]                093, 109
BIDMC1 [5400,5600]                   004, 094
GP711MarkerLFM5z2 [7175,7388]        020, 128
InternalBleeding6 [3474,3629]        099, 142
MesoplodonDensirostris [19280,19440] 043, 151
TkeepFifthMARS [5988,6085]           048, 156
gait1 [38500,38800]                  059, 167
```

### Why it matters

Every interval in this study is a bootstrap over series. Bootstrapping treats its
inputs as independent draws; these are not. Fourteen of forty slots carried seven
series' worth of information, so the effective sample size was overstated and
every confidence interval was **narrower than the evidence supports**.

This compounds a separate finding: SAR point estimates already move 30–45 %
between window sample sizes (n=40 vs 80 vs 160), and only the extremes of the
SAR ranking are stable. Both point the same way — uncertainty in this pipeline
is being systematically understated. An amendment that widens intervals is
correcting in the direction the evidence demands.

The selection procedure was also **causing** the problem: R6 balances on
`distorted`, and reaching for both strata pulls in the DISTORTED and undistorted
copies of the same recording.

### Decision

**New rule R8: one series per (recording, anomaly interval).**

- Applied **before** the balanced draw, so deduplication cannot bias which
  stratum loses members.
- The surviving representative is drawn **at random** from each group, not fixed
  to the undistorted one. DISTORTED and NOISE variants are the archive's own
  difficulty ladder; always keeping the plain version would bias the sample
  toward easy series.
- Implemented as `_dedup_recordings()`; grouping key is the filename with the
  `DISTORTED`/`NOISE` prefix stripped, plus the anomaly start and end.

Result: `v2` selects 40 series that are **40 unique recordings**.

### Retained safeguard

`main()` reports duplicate groups on every run regardless of rule set, so if a
future change reintroduces them it is visible in the output rather than buried.

---

## Effect of the amendment

| | v1 | v2 |
|---|---|---|
| candidates passing all rules | 100 / 250 | 211 / 250 |
| selected | 40 | 40 |
| unique recordings among the selected | **33** | **40** |
| DISTORTED / natural balance | 20 / 20 | 20 / 20 |

Both rule sets remain executable:

```bash
python scripts/build_dataset_config.py --format ucr --n 40 --ruleset v1 --root <UCR>
python scripts/build_dataset_config.py --format ucr --n 40 --ruleset v2 --root <UCR>
```

---

## Still not applied

- **R1** (membership of the official TSB-AD Eval split) — not applicable to the
  raw UCR archive, which has no such split.
- **R5** (difficulty screen) — implemented separately in
  `scripts/screen_difficulty.py`, which **reports** and does not filter. Acting
  on its verdict would remove series and therefore requires its own amendment.

  It has now been run: see `docs/R5-SCREEN.md`. Under IForest, **0 of these 40
  series are usable**; under KMeansAD, 16 are. That result does not change this
  amendment — R5 was deferred by design and the v2 sample is what it screened —
  but it does mean the sample this document fixes is an upper bound of 40, and
  the working sample is 16 until a detector decision is recorded.
