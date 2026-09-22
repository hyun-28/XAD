# Decision log

Format (BRIEF §1.5): what / why / alternatives / how to revert. Numbers quoted
here are copied from script output (`reports/data_audit.md`,
`data/raw/ucr/audit_a1.json`) — they are not the source of truth, the scripts are.

---

## D-A0-1 — Existing `tsadxai` conda env reused; only `python-pptx`, `pytest`, `pyyaml` added (2026-09-21)

- **What:** Task A runs in the pre-existing `tsadxai` env (python 3.11.16, numpy 1.26.4,
  scikit-learn 1.5.2, TSB-AD 1.5 from PyPI). Three pip packages were added.
- **Why:** the env already satisfies every BRIEF §2.2 pin; rebuilding it on the Mac would
  mean re-installing torch for a task that does not use it. The TSB-AD 1.5 dependency
  metadata declares `scikit-learn >=1.3.2` with no upper bound, so the `<1.6` pin does not
  conflict (BRIEF §7-3 checked, not triggered).
- **Alternatives:** fresh env from `environment.yml` (deferred to the server run in Task B/C).
- **Revert:** `pip uninstall python-pptx pytest pyyaml`.
- **Deferred:** `environment.yml` restructuring (drop `pytorch-cuda` on Mac), `docs/SETUP.md`
  — agreed 2026-09-21 as Task B scope. `requirements.lock.txt` was regenerated from this env.

## D-A0-2 — TSB-AD identified by PyPI version, not git hash

- **What:** run-log header prints `TSB-AD=1.5(PyPI)`.
- **Why:** BRIEF §2.2 asks for the git commit hash; the package was installed from the PyPI
  sdist, which carries no hash.
- **Alternatives:** install from GitHub at a pinned commit (Task B decision).
- **Revert:** n/a.

## D-A0-3 — New code lives in `src/data/`; legacy `src/data_ucr.py` untouched

- **What:** parser/loader/conventions/domains are new modules under `src/data/` (BRIEF §6).
  `src/data_ucr.py`, `scripts/ucr_inventory.py`, `scripts/build_dataset_config.py` are left as
  they were.
- **Why:** `experiments/exp_a_artifact.py` and the R5 screen depend on the legacy module;
  touching them is Task B/C scope (agreed 2026-09-21).
- **Known issue to carry into Task B:** `src/data_ucr.py:80` hard-codes
  `lo = anom_start - 1  # archive indices are 1-based` without evidence. The evidence found
  in Task A (D-A2-1) happens to agree with it, so the dry-run results in `docs/R5-SCREEN.md`
  and `configs/datasets.yaml` are **not** invalidated — but the module must be rerouted through
  `src/data/conventions.py` before any new result is produced.
- **Revert:** delete `src/data/`.

## D-A1-1 — Scope of "one file differs → STOP" (2026-09-21, agreed with researcher)

- **What:** `scripts/01_download.py` returns STOP (exit 1) only when a series `.txt` differs,
  is missing, or the count ≠ 250. Differences confined to supplementary documents are reported
  and do not stop.
- **Why:** BRIEF A1 step 4 is about the data; the official zip (184,066,400 B) is twice the
  size of the local zip (92,026,813 B), so a document-level difference was expected.
- **Outcome:** no difference at all — 266/266 files identical, including the zip itself
  (the official zip wraps the local zip plus an extracted copy). See D-A1-2.
- **Alternatives:** treat any difference as STOP.
- **Revert:** in `01_download.py`, add `mismatch or local_only or official_only` to `stop_reasons`.

## D-A1-2 — The extracted tree inside the official zip is the reference copy

- **What:** the official zip contains `AnomalyDatasets_2021/UCR_TimeSeriesAnomalyDatasets2021.zip`
  (inner zip, sha256 `5922fa21…` = the researcher's local zip) **and** an extracted tree. 263 of
  265 inner-zip members are identical to the extracted tree; two `.pptx` files differ by a few
  bytes and carry later dates (`UCR_AnomalyDataSets.pptx` 2021-09-22 → later;
  `Introducing MERLIN_3.0.pptx` 2021-08-13 → 2021-10-14). All 250 series are identical in
  all three copies (local extracted, official extracted, inner zip).
- **Why:** the extracted tree is the newer packaging (zip Last-Modified 2021-10-19).
- **Alternatives:** use the inner zip's documents.
- **Revert:** point `01_download.py`'s `find_anchor` at the inner zip's extraction instead.

## D-A1-3 — Supplement text extraction covers `.pptx` and `.m`; PDFs are listed, not parsed

- **What:** `src/data/supplement.py` dumps slide text (+ notes, tables, grouped shapes) and
  MATLAB source; PDFs are inventoried only.
- **Why:** the PDFs are published papers (Wu & Keogh), secondary to the slides that describe
  the datasets; the slides contained everything A2/A5 needed.
- **Alternatives:** parse PDFs with `pdfplumber`.
- **Revert:** extend `extract_all` — it is one branch.

## D-A2-1 — Index convention default = `matlab_1based_inclusive` (researcher sign-off pending, D3)

- **What:** `configs/conventions.yaml: index_convention: matlab_1based_inclusive`, i.e.
  filename `begin`/`end` → 0-based half-open `[begin-1, end)`; training prefix = `x[:train_end]`.
- **Why (documentary, `docs/ucr_supplement_text.md`, `UCR_AnomalyDataSets.pptx`):**
  slide 4 `L = end – begin + 1` (inclusive end); slide 5 "From 1 to X is training data",
  "Files are '-ascii' format" (MATLAB); slides 52, 53, 90–99 injection code
  `T(start_anomaly:end_anomaly)`, `randn(end_anomaly-start_anomaly+1, 1)`; slide 6
  `a(200:280)` "starts at 200". Structural: three files have `begin == end`
  (`_168250_168250`), which an exclusive end could not express.
- **Why (empirical, `reports/index_convention_evidence.csv`):** of the 20 series with 1–2-point
  anomalies, 6 favour the 1-based reading (ratio ≥ 2, incl. the −999 sentinel of
  resperation11 at 0-based 110799 = begin−1, and the tiltAPB4 dropout), 13 are not decisive
  (time-warp anomalies), 1 NOISE variant favours 0-based while both of its twins are not
  decisive — a noise draw.
- **BRIEF A2 says "확정 근거가 없으면 STOP":** evidence was found, so this is not a STOP;
  it is set as the default and flagged "sign-off pending" in the report (BRIEF §8 D3).
- **Alternatives:** `zero_based_inclusive`, `zero_based_exclusive`, or `null` (manifest
  `start0/stop0 = NA`).
- **Revert:** edit the yaml value, re-run `scripts/02_audit.py`.

## D-A2-2 — `train_end` is convention-independent

- **What:** `train_prefix_stop(train_end) == train_end` regardless of `index_convention`.
- **Why:** slide 5 defines it as a count ("from 1 to X"); `x[:X]` is the first X points in
  every reading.
- **Revert:** add a branch in `src/data/conventions.py::train_prefix_stop`.

## D-A3-1 — A3 checks are done on the raw fields in their own frame

- **What:** `END_GT_LEN` fires for `end > len(x)` (not `end >= len(x)` as a 0-based reading
  would need); `end == len(x)` is INFO `END_EQ_LEN`.
- **Why:** under the documented 1-based reading `end == len(x)` is legal. Keeping the checks
  in the raw frame means they do not silently change when D3 changes.
- **Outcome:** 0 STOP, 0 WARN findings; `END_EQ_LEN` did not occur.
- **Revert:** change the comparison in `src/data/audit_checks.py::check_series`.

## D-A3-2 — Severity classes for audit flags

- **What:** STOP (`NOT_1D_FLOAT64`, `NAN_OR_INF`, `END_GT_LEN`, `BEGIN_GT_END`,
  `BEGIN_LE_TRAIN_END`, `TRAIN_END_GE_LEN`) → exit 2; WARN (`CONST_TRAIN_PREFIX`) → exit 1;
  INFO (format variant, deck provenance, domain, −999 sentinel) → exit unaffected.
- **Why:** BRIEF §7-5 names the GT contradictions; a constant training prefix breaks
  z-normalisation but is not a GT error; the rest is documentation.
- **Revert:** edit `SEVERITY` in `src/data/audit_checks.py`.

## D-A4-1 — A4 (TSB-AD-U cross-check) not run this session

- **What:** manifest columns `in_tsbad … gt_agree` are `NA`; report sections 1/2c/5 say so.
- **Why:** BRIEF A4 is optional ("게이트 아님"); agreed 2026-09-21 to skip it in this session.
- **Revert:** implement `src/data/tsbad.py` + extend `02_audit.py`; the columns are already
  in place.

## D-A5-1 — `domain_goswami` vendored verbatim from the authors' repository

- **What:** `configs/goswami_entity_to_family.json` = `ANOMALY_ARCHIVE_ENTITY_TO_DATA_FAMILY`
  from `mononitogoswami/tsad-model-selection` @ `cf01d10af7b6d6ed9825fbd7aae2058af7c1d26b`,
  `src/tsadams/model_trainer/entities.py:233-543`, 250 entities, Apache-2.0. Fetched by
  `scripts/00_fetch_goswami_mapping.py`. Long family names are shortened via
  `configs/domains.yaml: goswami_short_labels`.
- **Why:** BRIEF A5 priority 1 succeeded, so no reconstruction was needed; the column keeps the
  name `domain_goswami` (not `_reconstructed`) and the paper may say "following Goswami et al."
- **Revert:** re-run the fetch script at another commit; diff the JSON.

## D-A5-2 — `domain_official` = our category of the deck's verbatim source statement

- **What:** `source_official` holds the first source sentence from the series' slide(s);
  `domain_official` is assigned by ordered regex rules in `configs/domains.yaml`
  (`official_source_rules`), applied line by line, most specific first.
  Labels outside Goswami's nine are used on purpose where the deck's source is not one of
  those signals: `PPG (bidmc)` (BIDMC1, a plethysmograph — Goswami says ECG) and
  `CHARIS (ECG/ABP/ICP unspecified)` (Goswami says ECG). No source statement → `NA`
  (CIMIS44AirTemperature1 ×2: not in the deck; taichidb ×3: slide 110 gives no source).
- **Why:** the deck is the archive's own provenance (BRIEF A5 priority 0), but it does not
  assign categories; the rule table is ours and is disclosed as such in the report.
- **Outcome:** official == Goswami for 234, differs for 11, NA for 5
  (`reports/domain_unmatched.csv`).
- **Alternatives:** force every series into the nine labels (hides the two disagreements).
- **Revert:** edit the rule table; re-run `02_audit.py`.

## D-A5-3 — `is_medical` = `domain_goswami ∈ {ABP, ECG, RESP}`; Gait and EPG are non-medical

- **What:** `configs/domains.yaml: medical_domains: [ABP, ECG, RESP]`.
- **Why:** BRIEF A5(iv) fixes the three; EPG is insect feeding behaviour; Gait recordings come
  from clinical databases (gaitndd, GaitPhase) but measure locomotion, not a physiological
  signal. Using `domain_goswami` (not `domain_official`) keeps the column defined for all 250.
- **Effect:** non-medical subset = 100 (all variants) / 59 (DISTORTED excluded) /
  50 (plain only) — above the BRIEF's 30 threshold.
- **Revert:** edit the list; re-run.

## D-A6-1 — Manifest carries extra columns beyond BRIEF A6

- **What:** `train_stop0`, `index_convention`, `variant`, `is_noise`, `domain_official`,
  `domain_official_basis`, `domain_agree`, `source_official`, `deck_slides` were added.
- **Why:** NOISE is a third variant class (slide 100) that `is_distorted` alone hides;
  the convention in force must travel with the derived indices; A5 priority 0 produced data
  that had nowhere else to go.
- **Revert:** drop from `MANIFEST_COLUMNS` in `scripts/02_audit.py`.

## D-A6-2 — Deck-vs-filename field differences are INFO, filename wins

- **What:** 21 series have numeric fields in the deck that differ from the shipped filename
  (`DECK_FIELDS_DIFFER`; report §5). The filename is the GT (BRIEF D2).
- **Why:** slide 116 documents that files 202/203 were relabelled after the deck's examples were
  written; several other rows are the deck citing a twin's fields.
- **Revert:** n/a (reporting only).

---

# Task A follow-up (docs/briefs/BRIEF_A-followup.md, 2026-09-22)

## D-A2-3 — D3 index convention APPROVED (2026-09-22, researcher)

- **What:** `matlab_1based_inclusive` is final. `configs/conventions.yaml` now carries
  `approved: 2026-09-22`; `src/data/conventions.py::approval_status()` reads it and
  `scripts/02_audit.py` prints it in `reports/data_audit.md` §6 (no hand edit).
- **Why:** documentary evidence (slides 4, 5, 6, 52–53, 90–99, see D-A2-1) plus four decisive
  empirical cases: 077/185 resperation11 (the −999 spike) and 092/200 tiltAPB4 (the dropout) —
  under the 0-based reading the spike falls outside the labelled interval.
- **Noted:** 108 NOISEresperation2's "0-based" verdict is treated as not decisive — both candidate
  points lie inside the added noise; its twins 079 and 187 are not decisive either.
- **Alternatives:** the other two enum values (D-A2-1).
- **Revert:** change `index_convention`, delete `approved`, re-run `scripts/02_audit.py`.

## D-A1-4 — Data licence settled (2026-09-22, researcher)

- **What:** the archive is treated as public research data with no explicit licence.
  Raw series are never redistributed (`.gitignore` unchanged); README points to the official URL
  and the SHA256 procedure (`scripts/01_download.py`, `data/raw/ucr/CHECKSUMS.sha256`).
  Derived artefacts (manifest, scores, figures) are published.
- **Why:** the researcher re-checked `https://www.cs.ucr.edu/~eamonn/time_series_data_2018/`:
  no licence or terms-of-use wording. The Dau et al. (2019) citation on that page is for the
  classification archive and is not used. The CC BY-NC 4.0 notice mentioned in BRIEF A1 is the
  footer of a personal website (wu.renjie.im), not an archive licence — the item is dropped.
- **Citations to use:** (1) the request in `Start_Here_This_is_a_Read_me.pptx`, verbatim —
  Keogh, E., Dutta Roy, T., Naik, U. & Agrawal, A. (2021). Multi-dataset Time-Series Anomaly
  Detection Competition, SIGKDD 2021; (2) Wu, R. & Keogh, E., IEEE TKDE 35(3):2421–2429, 2023.
- **Alternatives:** contact the authors for explicit terms before publication.
- **Revert:** n/a (documentation); `docs/DATA_LICENSE.md` is the record.

## D-A5-4 — Subset definitions separated: `non_medical` ≠ `physical` (2026-09-22, researcher)

- **What:** `is_medical` (D-A5-3) is unchanged. A new subset `physical` =
  `domain_goswami ∈ {Acceleration, Air Temperature, NASA, Power Demand}` is defined in
  `configs/subsets.yaml` (F4). "Non-medical" (100 series) and "mechanical/industrial", the notion
  the researcher's advisor prefers, are different sets; both are reported side by side.
- **Why:** `non_medical` also contains Gait and EPG (biological signals); `physical` does not.
- **Alternatives:** redefine `is_medical` to include Gait/EPG (rejected: BRIEF A5(iv) fixes it).
- **Revert:** edit `configs/subsets.yaml`.

## D-F2-1 — `content_group` verdict = correlation only; threshold 0.95 kept (2026-09-22)

- **What:** `configs/stats.yaml: recording_groups` — Pearson r of the z-normalised first 5,000
  points of both training prefixes (fewer if a `train_end` is smaller); r ≥ 0.95 ⇒ same recording.
  Length and `train_end` are recorded per pair (`reports/recording_groups.csv`) but do not enter
  the verdict, so a shorter cut of the same recording is still the same recording
  (e.g. 015 DISTORTEDECG4 L=200,000 vs 016 L=30,000 are different content, but a 200,000 vs
  30,000 pair that shares its first 5,000 points would be merged). Cross-name pairs are compared
  only within one `domain_goswami` and within ±1 % length (772 pairs; the unfiltered count would
  be 5,864 — the filter was kept because the brief allows it and it is recorded here).
- **Why the threshold is reasonable (measured, `reports/data_audit.md` §7b):** the distribution
  is bimodal — 265 same-name pairs: 110 below 0.5, 153 at ≥ 0.95, only 2 in [0.8, 0.95)
  (213–215 STAFFIIIDatabase r=0.948, 239–241 taichidbS0715Master r=0.852); 772 cross-name
  pairs: 320 below 0.8, 452 at ≥ 0.95, none in [0.8, 0.95). DISTORTED/NOISE twins land at
  0.95–1.0 as slide 100's construction predicts (added noise ≈ 0.2 × std and 0.1 × std).
- **Outcome:** 100 name_groups vs 89 content_groups. 12 name_groups split (same name, different
  recordings: CHARIS*, STAFFIIIDatabase, mit*longtermecg, ECG4 …) and **18 content_groups span
  several name_groups** (CIMIS44AirTemperature1–6, GP711MarkerLFM5z1–5, Tkeep*MARS, insectEPG1–5,
  PowerDemand1–4 + Italianpowerdemand, gait1–3, resperation*, tiltAPB1–4 …): the archive re-used
  one recording with different injected anomalies. → **follow-up STOP-2, reported, waiting.**
  Flag `CONTENT_GROUP_CROSSES_NAME` (STOP) marks the 146 series involved.
- **Alternatives:** require equal length too (would split the ECG4 200,000/30,000 family
  further, not merge anything); lower threshold to 0.85 (would add only the taichidb pair).
- **Revert:** edit `configs/stats.yaml`, re-run `scripts/02_audit.py`.

## D-F3-1 — Sentinel flags: new `SENTINEL_IN_NORMAL` (WARN) instead of re-grading `HAS_MINUS999`

- **What:** `HAS_MINUS999` stays INFO (any exact −999); `SENTINEL_IN_NORMAL` (WARN) fires when
  `sentinel_in_train_n > 0 or sentinel_in_test_normal_n > 0`; `SENTINEL_IN_GT_UNDOCUMENTED`
  (STOP) is follow-up STOP-3; `CONTENT_GROUP_CROSSES_NAME` (STOP) is STOP-2.
- **Why:** severities are a static table keyed by flag (`src/data/audit_checks.py::SEVERITY`,
  D-A3-2); a flag whose grade depends on the row would break the exit-code rule. Two flags carry
  the same information as the brief's conditional grade.
- **Consequence:** `scripts/02_audit.py` now exits 1 on the unmodified archive (23 WARN series;
  exit 2 while the STOPs stand). README says so; the `&&` chain in the reproduction commands
  stops after `02_audit.py` by design.
- **Revert:** edit `SEVERITY`.

## D-F3-2 — `near` detection reported but not used; exact indices drive everything downstream

- **What:** the brief's `near` rule (`|x + 999| ≤ 3 × MAD(train prefix)`) is computed and stored
  (`sentinel_near_n`) but classification (`in_gt/in_train/in_test_normal`), the twin check and
  the exclusion files `data/derived/sentinels/<num>.npy` use **exact** hits only. An extra pair of
  columns, `sentinel_spike_n` / `sentinel_max_k`, applies the brief's own local-outlier rule
  (w=51, k=10) to the exact points *in their own series*.
- **Why (measured, `reports/data_audit.md` §8):** the 23 series containing −999 are 12-bit-scale
  signals (range ≈ ±2047; 8 of them fully integer-valued) with training-prefix MAD 60–530, so
  tol = 180–1,600 covers a large part of the signal range: `near` returned 6.2 M points in 68
  series, 45 of which have no exact hit at all. The rule presupposes an O(1) signal and does not
  hold here. Using it downstream would have excluded most of the normal region.
- **Finding the researcher must weigh (not decided here):** of the 823 exact −999 points, **0**
  are local spikes at k = 10; the largest ratio is 6.99 — the one injected point of 185
  resperation11 (slide 46), all others ≤ 4.53. The twin check accordingly returns `absent` for 25
  of 26 twins and `partial(1/82/0)` for 077 (the injected point). In other words −999 is an
  ordinary sample value in these signals, not a missing-value marker; the premise of F3
  ("−999 spikes contaminate the normal region") is not supported by the data, and the k = 10
  default would not even flag the documented sentinel. Whether any exclusion is needed, and with
  which k, is a **researcher decision** — the exclusion files are written (exact, non-GT indices)
  so either choice is one config change away.
- **STOP-3 hit:** 123 ECG4 has an exact −999 at 0-based 16832 inside GT [16799, 17100) and its
  deck slides do not mention −999. Measured: the point sits on a smooth ramp
  (…, −965, **−999**, −1034, …), ratio 3.23 — an ordinary sample, not a sentinel. Reported, not
  resolved.
- **Alternatives:** scale-aware near rule (e.g. tol relative to the integer quantum); drop the
  near column.
- **Revert:** `near_mad_factor` in `configs/sentinels.yaml`; switch `exact_idx` → `near_idx` in
  `src/data/sentinels.py::classify` (one line) — not recommended.

## D-F3-3 — Twin-status format and figure selection

- **What:** `sentinel_twin_status` lists every twin as `num:status(confirmed/absent/length_mismatch)`
  joined by `;`; `status` ∈ {confirmed, absent, length_mismatch, partial}, `partial` when a twin
  has both confirmed and absent indices (only 077). Plain series whose name_group has no
  DISTORTED/NOISE member (STAFFIIIDatabase ×6, CHARISfive) get `NA`. Figures show the top-3 plain
  series by exact count **among those with at least one twin that is not length_mismatch**
  (123, 184, 185); 123's short twins are drawn as far as their length allows.
- **Why:** the brief's three-value status cannot express a mixed result, and a figure of a twin
  with no overlapping indices would be empty.
- **Revert:** `figures_top_n` in `configs/sentinels.yaml`; status logic in `sentinels.py::TwinCheck`.

## D-F4-1 — Subsets carry both group counts

- **What:** §9 prints `n_groups (name)` (unit of analysis) and `n_groups (content)` (sensitivity)
  for every subset and combination in `configs/subsets.yaml`. `physical` = 42 series / 19 name
  groups (< 30 ⇒ descriptive only), `non_medical` = 100 / 44.
- **Revert:** drop the column in `scripts/02_audit.py`.

## D-F5-1 — Legacy loader rewired; its parser's NOISE blind spot left as is

- **What:** `src/data_ucr.py::load_file` now calls `to_half_open(…, active_convention())`.
  `tests/test_legacy_equivalence.py` reproduces the old literal formula
  (`max(0, begin−1)`, `min(len, end)`) and shows it equals the convention, the rewired loader and
  the manifest on all 250 files → **STOP-1 not triggered**; `docs/R5-SCREEN.md` and
  `experiments/exp_a_artifact.py` results stay valid.
- **Not changed:** `data_ucr.parse_filename` strips only `DISTORTED` into `base`, so a NOISE series
  keeps its prefix there. Out of F5's scope; new code uses `src/data/ucr.py`.
- **Revert:** restore the two lines (the test will then still pass — that is the point).

## D-F2-2 — Unit of analysis = `content_group` (2026-09-22, researcher; resolves follow-up STOP-2)

- **What:** `configs/stats.yaml: unit_of_analysis: content_group` (89 groups over the full 250),
  `bootstrap_resample_unit: content_group`, `sensitivity_unit: name_group` (100). The
  hypothesis-test population is the full 250-series set; `non_medical` and `physical` are
  descriptive only. `CONTENT_GROUP_CROSSES_NAME` downgraded STOP → INFO (fact kept in the manifest).
- **Why:** 18 content groups span several names because one recording was re-used with different
  injected anomalies (§7d); their normal regions are bit-identical, so `name_group` cannot claim
  independence. `content_group` is the conservative unit.
- **Alternatives:** `name_group` primary (rejected: pseudoreplication on identical normal regions).
- **Revert:** swap the two yaml values, re-run `scripts/02_audit.py`.

## D-F3-4 — −999 is an ordinary measurement; nothing excluded (2026-09-22, researcher; resolves STOP-3)

- **What:** the exact −999 points are treated as normal data in the main analysis. `no_sentinel`
  stays in `configs/subsets.yaml` as a sensitivity subset; `data/derived/sentinels/<num>.npy` are
  still written for it. 123 ECG4's −999 inside the GT: no action. `SENTINEL_IN_NORMAL` and
  `SENTINEL_IN_GT_UNDOCUMENTED` downgraded to INFO, so `02_audit.py` exits 0 on the archive.
- **Why:** D-F3-2 measurements — 0 of 823 points are local spikes; the signals span ±2047.
- **Note:** the researcher asked for this entry to be numbered D-F3-3; that id was already used
  above for the twin-status format, so it is D-F3-4 here.
- **Revert:** re-grade the two flags in `src/data/audit_checks.py`; apply the exclusion files.

---

# Task B — TSB-AD safe wrapper (2026-09-22)

## D-B1-1 — TSB-AD identified by PyPI version + source SHA256 (supersedes the "git hash" part of D-A0-2)

- **What:** `TheDatumOrg/TSB-AD` has no release tags (`git ls-remote --tags` empty, 2026-09-22),
  so PyPI 1.5 has no commit. `src/runlog.py::env_header()` prints the SHA256 (12-hex prefix) of
  `models/IForest.py`, `model_wrapper.py`, `models/feature.py`, `utils/utility.py`; the full hashes
  are in `docs/TSBAD_INTERNALS.md`.
- **Found while tracing:** GitHub HEAD `6beac72e` (2026-09-07) **fixes the normalize trap** —
  `decision_function` now z-scores the windows like `fit` (IForest.py:243-247 at HEAD). The PyPI
  1.5 sdist does not have that fix. We stay on PyPI 1.5 with `normalize=False`, which is consistent
  in both versions; installing from GitHub would change the reference and is a researcher call.
- **Revert:** n/a.

## D-B2-1 — Default IForest hyper-parameters = what TSB-AD's benchmark driver passes

- **Trace (GitHub `6beac72e11d1155ade40870492c00d0d1cfdcaaf`, 2026-09-07; PyPI 1.5 line numbers in
  brackets):** `benchmark_exp/Run_Detector_U.py:46` reads `Optimal_Uni_algo_HP_dict['IForest']` =
  `{'n_estimators': 200}` (`TSB_AD/HP_list.py:293` [installed `:240`]); `:71` calls
  `run_Unsupervise_AD('IForest', data, n_estimators=200)`; `model_wrapper.py:10-19` dispatches to
  `run_IForest(data, slidingWindow=100, n_estimators=200, max_features=1, n_jobs=1)`
  (`model_wrapper.py:59-64` [`:48-53`]). `Run_Detector_U.py:62` computes
  `find_length_rank(...)` but passes it only to `get_metrics` (`:87`), **never to IForest** — the
  benchmark IForest uses a fixed window of 100. The installed package's `main.py:35-56` does the same.
- **What:** `configs/detectors.yaml: iforest = {slidingWindow: 100, n_estimators: 200, n_jobs: 1,
  max_features_primary: 1 (int), normalize: false}`. `run_IForest`'s own default `n_estimators=100`
  is recorded as the alternative. No `find_length_rank` (nothing to log per series).
- **Verified:** `tests/test_fit_score_consistency.py` asserts the signature defaults and the HP
  dict, and that the wrapper's scores equal that path's `decision_scores_` bit for bit.
- **Revert:** edit the yaml.

## D-B2-2 — `max_features`: int 1 (benchmark) vs float 1.0 — researcher decision, both run in gate 1

- **What:** `run_IForest` passes `max_features=1` (**int** → one of the 100 window columns per
  tree: value-based isolation). The class default `1.` (float → all columns: shape-based isolation)
  is a different model. Gate 1 runs `max_features_variants: [1, 1.0]` × seeds × fit_on; the primary
  setting is chosen on the pre-registered detection criterion only, never on AI or later metrics.
  `max_features_primary: 1` is provisional (= benchmark default) until gate 1.
- **Why the benchmark default is int 1:** `model_wrapper.py:59` signature (`max_features=1`) and
  `HP_list.py:293` (no override for the univariate case; the multivariate optimum `:113` is 0.8).
- **Revert:** edit `configs/detectors.yaml`.

## D-B2-3 — `fit_on` is applied in exactly one place: `src/data/ucr.py::slice_fit`

- **What:** `slice_fit(x, meta, fit_on) -> (x_fit, n_fit)`; `train_prefix` = `x[:train_end]`
  (via `conventions.train_prefix_stop`, D-A2-2), `full` = whole series. `fit_detector` takes the
  slice; the caller records `fit_on`, `train_end`, `n_fit` in the run log. `tests/test_slice_fit.py`.
- **Revert:** n/a.

## D-B2-4 — `src/detectors.py` became the package `src/detectors/`; legacy `ScoreFn` re-exported

- **What:** `git mv src/detectors.py src/detectors/legacy.py`; `base.py` (interface, B3 checks),
  `iforest.py`; `__init__` re-exports `ScoreFn`/`self_check`, so `experiments/exp_a_artifact.py`,
  `scripts/screen_difficulty.py`, `tests/test_detectors.py` are unchanged and still pass
  (`exp_a_artifact.py --synthetic` re-run 2026-09-22: 765 calls, OK).
- **Revert:** `git mv` back; delete the package.

## D-B3-1 — Constant score vector is an error by default (`constant_score: error`)

- **What:** `validate_scores` raises `DetectorError` on variance 0; `configs/detectors.yaml:
  constant_score: flag` makes it a logged flag instead. Also applied to the training scores at
  fit time.
- **Why:** the Artifact Index divides by score spread; a constant vector is undefined, not "no anomaly".
- **Revert:** the yaml value.

## D-B4-1 — Environment rebuilt from `environment.yml`; torch 2.3.0 confirmed (no STOP)

- **What:** `environment.yml` (Mac/CPU) and `environment.server.yml` (+ nvidia channel,
  `pytorch-cuda=12.1`) differ by three lines. A fresh env `tsadxai-w1` was created from
  `environment.yml` on 2026-09-22 (conda 26.1.1, libmamba): python 3.11.16, numpy 1.26.4,
  scikit-learn 1.5.2, torch 2.3.0, TSB_AD 1.5; **114/114 tests pass** in it. Lock files regenerated
  from it: `requirements.lock.txt`, `environment.lock.osx-arm64.txt`.
- **Not done:** the researcher's existing `tsadxai` env (torch 2.14.0, hand-installed) was left in
  place — deleting an env is the researcher's call; `docs/SETUP.md` gives the one-liner. The server
  build has not been run yet (BRIEF §9).
- **Revert:** `conda env remove -n tsadxai-w1`.

---

# Task C — Gate 1 (2026-09-22)

## D-C1-1 — Gate 1 executed on the Mac; TSB-AD subset (C1-b) not run

- **What:** BRIEF C1 says "실행 위치: 서버". Measured on the Mac (M-series, 17 GB): fit 100k
  = 2.5 s, score 900k = 5.2 s, full fit 900k = 11.7 s; the 5-series smoke test took 26 s for 60
  fits, so the whole grid (250 × 12 = 3,000 fits) fits in well under an hour with 4 workers.
  It was run here (`runs/gate1_full_20260922`); the server run remains to be done for BRIEF §9's
  "clean env on the server" item. Grid: fit_on {train_prefix, full} × max_features {1, 1.0}
  (D-B2-2) × seeds {0, 1, 2}.
- **C1-b (TSB-AD UCR subset):** not run — A4 was skipped (D-A4-1), so there is no mapping to
  the TSB-AD files. Reported as NA in `reports/gate1.md`.
- **Revert:** re-run `scripts/03_gate1.py` on the server; outputs are deterministic per seed.

## D-C2-1 — Gate uses P1 with buffer = window; buffer = 0 reported alongside

- **What:** BRIEF C2 asks for both buffers and C3 does not say which one judges the gate.
  Pre-registered before looking at results: **buffer = slidingWindow (100)**.
- **Why:** TSB-AD reports each window's score at the window centre (TSBAD_INTERNALS §4), so
  with buffer = 0 the anomaly's own window scores spill ~50 points on each side into
  `normal_test` and inflate the 99th percentile — the anomaly would then be compared against
  itself. `tests/test_gate1_criteria.py::test_buffer_matters_when_scores_leak_around_the_anomaly`.
- **P2:** the archive's rule was found (slide 4 / Irrational Exuberance slide 64:
  `min(begin−L, begin−100) < P < max(end+L, end+100)`, 1-based) and is `p2_archive`; BRIEF's
  `[start0−100, stop0+100)` is `p2_brief`. Both reported; neither is a gate.
- **Percentile:** `np.percentile` default (linear interpolation). Ties in argmax → first index.
- **Revert:** swap `p1_bw` → `p1_b0` in `scripts/04_report.py` (one constant).

## D-C4-1 — Score file names carry the max_features tag

- **What:** BRIEF C4 names `<num>_<fit_on>_seed<k>.npy`; with two `max_features` variants the
  files are `<num>_<fit_on>_mf<1|1.0>_seed<k>.npy` (`results/gate1/scores/`, git-ignored;
  `detection.csv` stores the path per row).
- **Revert:** n/a.

## D-C4-2 — Report keys `max_features` as text ("1" / "1.0"), not as a number

- **What:** `scripts/04_report.py` keeps the CSV text for `max_features`. The first draft parsed it
  to int/float and Python's `1 == 1.0` silently merged the two variants into one condition (the
  first report showed 2 conditions and a "1 vs 1.0" comparison with zero differences). Caught by
  checking that the score files differ (`np.abs(a − b).max() = 0.115` on 001). Fixed and
  regenerated before any number was read.
- **Revert:** n/a.

## D-C5-1 — Gate 1 failed with IForest → detector replaced (2026-09-22, researcher)

- **What:** the pre-registered gate (P1, `fit_on=train_prefix`, `max_features=1`, buffer = w,
  median over seeds) came out at **28.4 % < 50 %**. Per the pre-registered rule the detector is
  replaced. The two alternatives were considered and **rejected**: keeping IForest with
  `max_features=1.0` (39.2 %, still below the gate, and switching the primary after seeing the
  numbers would be threshold-shopping) and redesigning the study around the detected subset
  (71 series, sample composition biased by IForest).
- **Kept:** all IForest results stay — `results/gate1/detection.csv`, the 1.7 GB of raw scores,
  and the four-condition table plus the paired comparison (1.0-only 102 vs 1-only 28) remain in
  `reports/gate1.md`. The new detector is reported alongside, never instead.
- **Revert:** n/a (a record of a decision).

## D-C5-2 — Candidate order pre-registered before any new result was looked at (2026-09-22, researcher)

- **Order:** 1) MatrixProfile — adopted if ≥ 50 %, in which case the lower-ranked candidates are
  **not run**; 2) Sub_PCA; 3) KMeansAD, each run only if the one above it fails. All three below
  50 % ⇒ STOP.
- **MatrixProfile gate conditions (researcher, pre-registered):**
  - score = AB-join against the training region (each subsequence's nearest-neighbour distance to
    the reference set); `fit` stores the reference set, `score` is separate.
  - TSB-AD's MatrixProfile implementation must be source-traced: AB-join support, window-length
    rule (`find_length_rank`?), backend (stumpy?). **If TSB-AD supports self-join only → STOP and
    report**, because only `fit_on=full` would be possible and that is a researcher decision.
  - window length: follow whatever the TSB-AD benchmark path uses; log the per-series w.
  - deterministic algorithm ⇒ no seed repetition; `fit_on` train_prefix (primary) and full (reference).
  - P1 unchanged, buffer = that detector's w (D-C2-1 rule stands).
  - reporting in the same format as IForest (condition table, per-domain, content_group aggregation).
- **Revert:** n/a.
