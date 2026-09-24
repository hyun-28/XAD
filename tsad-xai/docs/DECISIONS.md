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

---

# Task D — Šimić et al. metric port (BRIEF2, 2026-09-22)

## D-D1-1 — `third_party/simic_cmi` is a pinned clone, git-ignored, fetched by a script

- **What:** `https://github.com/perturbationeffect/cmi-am-validation-for-dl-ts-classifiers`
  at commit `edc6a870b1a50fce3385bfce6a468583d696ea75` (verified against the remote's HEAD on
  2026-09-22), cloned to `third_party/simic_cmi/`, never modified. Apache-2.0; the upstream
  LICENSE travels with the clone.
- **Why a clone and not a submodule:** the repository is **537 MB** (469 MB of trained models,
  221 MB of UCR classification data). A submodule would make every `git clone` of this monorepo
  pull it. `third_party/` is git-ignored and `scripts/00_fetch_simic.py` restores it at the
  pinned commit; the tests skip cleanly when it is absent.
- **Revert:** delete the directory, or convert to `git submodule add` at the same commit.

## D-D1-2 — The original imports cleanly; no source extraction needed

- **What:** BRIEF2 D1-3 allows extracting the function source if importing fails on
  MongoDB/torch. Measured 2026-09-22: `utils/res_utils.py` imports only
  os/numpy/matplotlib/seaborn/math, so `tests/test_simic_equivalence.py` imports the real
  upstream functions. **MongoDB was not installed.** Only the pipeline scripts
  (`identify_zero_class.py`, `results_analysis.py`) need pymongo/torch.
- **Revert:** n/a.

## D-D1-3 — Ported functions and deliberate divergences

- **Ported** (`src/metrics/faithfulness_simic.py`): `decaying_degradation_score`,
  `compute_dataset_dds`, `pes`, `CMI` (+ `cubic_weights` factored out). **Not ported:**
  `degradation_score` (the non-decaying variant, unused by the paper's figures),
  `rank_biserial`, `combined_mean/stddev`, plotting helpers.
- **Equivalence:** 1,000 random curve pairs (length 2-60, values 0-100) — max |ours - theirs|
  is exactly **0.0**, not merely within 1e-12; plus a 400 x 58 sign grid for CMI and 200
  random subsets for PES. `tests/test_simic_equivalence.py`.
- **Divergences (fail loud, BRIEF §1-1):** curves shorter than 2 points raise instead of
  returning nan with a RuntimeWarning (upstream's `dds_max` is 0 there); unequal lengths,
  non-finite values, empty DDS sets and `max_diff <= 0` raise. Every one is covered by a test
  that also asserts the upstream behaviour it replaces.
- **`max_diff`:** exposed, default 100 = upstream's hard-coded constant (`res_utils.py:73`).
- **Revert:** n/a.

## D-D2-1 — D7-D10 registered as undecided; `faithfulness_ad.py` is a stub

- **D7 tracked quantity:** what scalar the AD perturbation curve reads off each score vector —
  max inside the GT interval / mean inside it / max over the whole series / other. No AD twin of
  "predicted-class probability" exists.
- **D8 normalisation:** the original divides by 100 (a probability bound). IForest scores are
  unbounded and not probabilities, so DDS leaves [-1, 1]. Candidates: min-max of the unperturbed
  score, a training-region quantile, the score's IQR.
- **D9 direction:** whether a falling score under MoRF counts as faithful. Zero-masking a
  *normal* region already lowers IForest's score (W1 dry run, `docs/R5-SCREEN.md`), so the sign
  convention is not self-evident.
- **D10 coverage:** whether normal-series regions enter the curve at all; the original uses all
  test samples, UCR has one anomaly per series.
- **Status:** UNDECIDED. `src/metrics/faithfulness_ad.py` carries the signatures and docstrings
  and raises `NotImplementedError`; nothing downstream may call it.
- **Revert:** n/a.

## D-D1-4 — Paper-vs-code comparison DONE (2026-09-22; the PDF was fetched)

- **What:** BRIEF2 D1-3 asks for the differences between the paper's Eq. 1-6 and the code.
  The paper is Šimić, Veas & Sabol, *Scientific Reports* 15:26607 (2025),
  DOI 10.1038/s41598-025-09538-2, open access — downloaded 2026-09-22 to
  `Šimić et al/Simic_2025_SciRep_s41598-025-09538-2.pdf`. Full comparison in
  `docs/SIMIC_PORT.md` §4.
- **Findings:** (P1) Eq. 1/2 are a weighted *sum* over probabilities in [0, 1]; the code is a
  weighted *average* over a 0-100 scale. They reduce to the same normalised quantity — the
  hard-coded 100 is a unit conversion, verified to 8.9e-16 on 2,000 random pairs; our `max_diff=1`
  reproduces the paper's equations literally. (P2) **Eq. 6 branches on `DDS·PES ≥ 0`, the code on
  `pes*dds <= 0` — the conditions are opposite**, differing exactly on the set `DDS·PES = 0`.
  Both yield 0 there (the paper's harmonic branch tends to 0 as `|DDS|⁻¹ → ∞`), so the numbers
  agree, but Eq. 6 taken literally raises `ZeroDivisionError` in Python. Per BRIEF2 we implement
  the **code**. (P3) Eq. 3-5 (PES) match the code exactly, zeros in neither count.
- **Revert:** n/a.
## D-C5-6 — P1 alignment handling, fixed BEFORE the MatrixProfile verdict was read (2026-09-22T13:02:58Z, researcher)

- **Recorded at:** 2026-09-22T13:02:58Z (UTC). At this moment the MatrixProfile gate run
  (`runs/gate1_MatrixProfile_20260922`) was still in progress — 222 of 250 series scored — and
  **no success rate had been computed or looked at**: `scripts/04_report.py --detector MatrixProfile`
  had not been run. The rule below is therefore pre-registered with respect to the verdict.
- **Rule:**
  1. The verdict is the pre-registered P1 (`max(score[start0:stop0])`), unchanged.
  2. `p1_support_bw` (D-C5-5) is reported as a **diagnostic** only.
  3. If P1 and `p1_support_bw` fall on the same side of the 50 % gate, the verdict stands as
     computed.
  4. If **P1 < 50 % ≤ p1_support_bw**, that is a **STOP**: the IForest results are recomputed
     under the same window-support criterion so both detectors can be read on one scale, and the
     verdict is the researcher's, not Claude Code's.
- **Why:** the alignment effect (a window score is written at `start + w//2`) can push an
  anomaly's peak outside the interval P1 reads. Deciding in advance what that would mean prevents
  the criterion from being chosen after seeing which one is kinder.
- **Revert:** n/a (a record of a decision).

## D12-a — Constant-subsequence flag added to the gate report (W2 prediction item)

- **What:** per series, count the constant subsequences (σ = 0 over a length-w window) in the
  **training region excluding the self-matching zone** — the region an AB-join scores against —
  and report the counts in the gate report. Column `n_const_train` in `detection.csv`.
- **Why (D12, traced in stumpy `core.py:1152-1167`):** both subsequences constant → D² = 0;
  exactly one constant → D² = m, a fixed value independent of the data. A perturbation that makes
  a window constant therefore either collapses the score to 0 (if the reference holds a constant
  window) or snaps it to √m. Which of the two happens for a given series depends on whether its
  reference region contains a constant window — hence the flag, registered now so W2's Artifact
  Index predictions can be checked against it.
- **This gate reports counts only.** No interpretation, no decision.
- **Revert:** drop the column.

## D-D3-1 — `primefac` removed from the reference env; a separate `tsadxai-simic` env for BRIEF2 D3 (2026-09-22T13:04:38Z, researcher)

- **What happened:** `utils/vit/vit.py` in the pinned clone imports `primefac` at module level,
  so the whole `utils.networks` registry (needed for all five Wafer architectures) fails without
  it. It was installed with `pip install primefac` into **`tsadxai-w1`** — the reference env built
  from `environment.yml` — which is exactly the env that must stay equal to what the yml resolves
  to. `pypdf` had been added there too, to read the Šimić PDF.
- **Fix:** both are removed from `tsadxai-w1`, which is restored to the `environment.yml`
  resolution (verified by re-running the test suite and diffing `pip freeze` against
  `requirements.lock.txt`). Upstream reproduction now runs in a separate env defined by
  **`environment_simic.yml`** (`tsadxai-simic`): python 3.11, torch 2.3.0, primefac, captum.
  `scripts/05_wafer_zero_class.py` is run with that interpreter.
- **Why separate rather than adding it to environment.yml:** nothing in this project imports
  `primefac` or the upstream networks; the gate and the metric port must not depend on the
  upstream repository being installable.
- **Revert:** `conda env remove -n tsadxai-simic`; delete `environment_simic.yml`.

## D-C5-7 — MatrixProfile gate: PASS; D-C5-6 branch 3 applied (2026-09-22T15:26:01Z)

- **Result:** primary condition (`fit_on=train_prefix`, AB-join, buffer = per-series w)
  **P1 = 79.2 %** ≥ 50 % → **PASS**. The diagnostic `p1_support_bw` = 83.2 % is on the same side
  of the gate, so D-C5-6 branch 3 applies: the verdict stands as computed and no recomputation of
  the IForest results under the support criterion is required (branch 4 not triggered).
- **Detector adopted:** MatrixProfile. Per D-C5-2 the lower-ranked candidates (Sub_PCA, KMeansAD)
  are **not run**.
- **IForest results retained** (D-C5-1): `reports/gate1_IForest.md`,
  `results/gate1/IForest/detection.csv`, and the raw scores.
- **Determinism verified:** the gate was run twice (the second time to add `n_const_train`);
  all 35 shared columns of `detection.csv` are identical across the two runs.
- **Runtime note:** 250 series x 2 conditions = 500 scoring calls in 5,012 s wall (4 workers);
  the longest single call was 1,823 s (series 240/241, n = 900,000, w = 125). W2's ~500 calls per
  series make the incremental scorer (W2-1) a prerequisite, not an optimisation.
- **Revert:** `configs/detectors.yaml`; re-run `scripts/03_gate1.py --detector MatrixProfile`.

---

# W2-0 pilot (BRIEF3, 2026-09-23)

## D-P0-1 — PREREG committed before the pilot; pre-commit observations disclosed

- **What:** `docs/PREREG_W2.md` committed as `0d041d7` at 2026-09-23T13:55:43+09:00, before
  `scripts/w2_pilot.py` existed. Prediction ② names ContextReconstruct (researcher, 2026-09-23).
  The scratch measurements made while planning (S1 residuals, constant-fill interior values on
  #066, gate-1 `thr_bw`) are listed in the PREREG's own disclosure section.
- **Enforced:** the pilot refuses to run if the PREREG is uncommitted, modified, or committed
  later than the run's start (`prereg_commit()`).
- **Revert:** n/a.

## D-P1-1 — Series-selection rule 3 read as "longest certain-normal piece ≥ 8m"

- **What:** BRIEF3 P1-3 does not say whether "length" is contiguous or total, nor where the
  "windows straddling `train_end`" exclusion ends. Coded reading: certain-normal =
  `[train_end + m, start0 − m) ∪ [stop0 + m, n)`, and the **longest contiguous piece** must be
  ≥ 8m (erasures are contiguous, so a total over two pieces does not guarantee room).
- **Why it does not matter here:** the four readings (longest/total × start at `train_end` or
  `train_end + m`) all select **#066 DISTORTEDinsectEPG2**. The report regenerates that table.
- **Alternatives:** total length; start at `train_end`.
- **Revert:** `normal_pieces()` / `select_series()` in `scripts/w2_pilot.py`.

## D-P2-1 — Detectors use T_A = x (the whole series), as gate 1 did

- **What (researcher approved, 2026-09-23, plan item D-1):** BRIEF3 P2 writes
  `T_A = test`; the gate-1 implementation it asks to reuse (`D-C5-3`) uses `T_A = x`. We reuse
  gate 1, so Z's padded score equals the stored gate-1 file and its P99 equals `thr_bw` — the
  pilot asserts both. The test-region window values are the same either way (4.5e-13 on #066).
  R = `stumpy.aamp` with the same arguments (p = 2).
- **Constant windows (D12 re-check, stumpy 1.14.1):** `stump.py:201-204` — both constant →
  pearson 1 → D = 0; one constant → pearson 0.5 → D = √m. Constant = `ptp == 0`
  (`core.py:2614`). The W1 record (D12-a) holds for the installed version.
- **Revert:** `src/detectors/mp.py`.

## D-P3-1 — Šimić PMs applied in the training-standardised space (researcher decision, 2026-09-23)

- **Decision (STOP-2 option c):** z = (x − μ)/σ with μ, σ = mean/std (ddof 0) of `x[:train_end]`;
  the upstream "sample" (`self.original`, the statistics source) is `z[:train_end]`; the PM is
  applied; the erased region is mapped back (·σ + μ). Why: the upstream data were standardised
  (all 125 saved `training_conf.json`: `"standard"`), so the PMs keep their meaning
  (Zero = mean level, U(−1,1) = ±1σ, OutOfDistHigh = 100 × abs-max in σ units).
- **Consequences that follow from the decision (not separate choices):**
  - positions refer to the whole series (the "canvas"): the erased region lies in the test part,
    outside the statistics source. UniformNoise100's noise vector therefore has the canvas
    length (upstream: `len(original)`, identical when sample = canvas); its bounds are
    hard-coded, so its statistics source is irrelevant. Drawn from `np.random.RandomState(seed)`
    (= upstream's `np.random.uniform` after `np.random.seed(seed)`), seed in the config.
  - only `[a, a+r)` is written back; the round trip is not bit-exact on untouched points.
  - Zero writes μ; SampleMean writes mean(z_train)·σ + μ. Whether the two are bit-identical is
    checked in the pilot (S5), as the researcher asked; nothing forces it.
- **Where upstream is undefined** (NearestNeighborWindow whose neighbour would start before 0
  or run past the end): upstream fails with a numpy broadcast error; we raise `OperatorError`.
  The equivalence test asserts both fail on the same inputs.
- **Upstream mutates `self.sample` cumulatively (MoRF steps).** Each pilot condition erases a
  fresh copy once, which is upstream's first step.
- **Equivalence:** `tests/test_simic_operators.py` imports the upstream classes from the pinned
  clone and compares bit for bit, both literally (sample = canvas) and in the adapted setting
  (upstream `original` replaced by the training part).
- **Exposed as config:** `operator_space: {standardize, reference}` in `configs/w2_pilot.yaml`.
- **Revert:** set `standardize: false` and/or `reference: full`.

## D-P3-2 — ContextReconstruct implementation details (definition: researcher, 2026-09-23)

- **Definition** as in `docs/PREREG_W2.md` appendix. Implementation choices:
  z-normalised distance by `stumpy.mass(query, x[:train_end − r])` (windows j = 0 …
  train_end − m − r, so j + m + r ≤ train_end); ties → smallest j (`np.argmin`); refuses
  `a < train_end` (the search region would overlap the erased points) and `a < m`.
- Runs through the same standardise/map-back wrapper as the PMs; distance is invariant and the
  fill equivariant under the affine map, so the result equals the raw-space fill to 1e-10
  (`test_cr_is_equivariant_under_standardisation`).
- NearestNeighborWindow is kept exactly as upstream; reports label it "이웃 붙이기".
- **Revert:** `context_reconstruct*` in `src/perturb/operators_simic.py`.

## D-P4-1 — S1 judged with atol = 1e-8 (researcher decision, 2026-09-23, STOP-1 option a)

- **What:** BRIEF3 S1 says "exactly 0". Measured before the run: a full stumpy recomputation
  leaves rounding residuals on untouched windows (1e-14 … 3.9e-10), because stumpy accumulates
  the covariance along each diagonal and the erased values enter and leave the sum. S1 is
  therefore judged at `s1_atol = 1e-8`, and the share of bit-exactly unchanged windows is
  reported alongside.
- **Scope:** the residual exists only on the pilot's full-recomputation path. W2 uses the
  incremental scorer, which copies untouched windows and is exact there by construction.
- **Revert:** `s1_atol` in `configs/w2_pilot.yaml`.

## D-P5-1 — Pilot design and check definitions, fixed before the run

- **Positions:** seed `position_seed`; one start `a` per position shared by all four r
  (nested design). Footprint `[a − L, a + r_max + R)` with L, R the largest reach of the affected
  windows (m − 1) and of every operator's reads (NNW ⌈r/2⌉ / ⌊r/2⌋, ContextReconstruct m,
  LinearInterpolation 1) at r_max; it must lie in one certain-normal piece and the three
  footprints must be disjoint. Positions are sorted by `a`; "first" (F2) = smallest `a`.
- **Metrics** are read from the incremental profile (W2's path); the full recomputation is used
  only for S1/S4. `p99_normal` per detector = gate-1 definition (buffer = m, numpy percentile)
  on that detector's unperturbed padded score. `false_alarm_before` (the same test on the
  unperturbed series) is added so that a pre-existing exceedance is visible.
- **S2** (Z, r ≥ m): the interior values of Zero, SampleMean and OutOfDistHigh form ONE value
  (exact equality); whether it equals √m exactly is recorded.
- **S3** (R, r ≥ m): because D-P3-1 makes Zero and SampleMean write the same fill (to rounding),
  "the three differ" is checked on the pairs with different fills — (Zero, OutOfDistHigh) and
  (SampleMean, OutOfDistHigh): interior arrays not equal. Zero vs SampleMean is **S5**.
- **S5** (researcher request): Zero ≡ SampleMean — x′ and both detectors' profiles
  (incremental and full) bit-identical. Reported; not a BRIEF3 STOP.
- **S4:** max |incremental − full| ≤ 1e-8 on D (BRIEF3 §6), over all 192 runs. `|ΔD²|` is
  reported as a diagnostic, not a criterion.
- **Exit code:** 2 if any of S1–S4 fails (STOP), after the report and CSV are written.
- **Revert:** `configs/w2_pilot.yaml`, `sanity()` in `scripts/w2_pilot.py`.

## D-P6-1 — Timing model for the W2-1 extrapolation

- **What:** cost of one incremental run = one AB-join of n_A = r + m − 1 windows against
  n_B = train_end − m + 1 reference windows, `t = c0 + c1·n_B + c2·n_A·n_B`, fitted by NNLS on
  direct timings (median of 3) on the pilot series, the 5 longest series and 5 series at
  training-length quantiles (10/30/50/70/90 %). ContextReconstruct search `t = c0 + c1·n_B`.
  Other erasers: pilot mean. Unperturbed full scoring: gate-1 `score_s` for Z (measured with 4
  workers in parallel, so pessimistic) and R = Z × pilot R/Z ratio. W2-1 run count with 8 erasers
  = 160,000 (BRIEF3's 140,000 assumed 7).
- **Why direct timings:** the train lengths span two orders of magnitude; a single-series
  constant would not separate the per-call T_B preprocessing from the join itself.
- **Revert:** `calibrate()` / `extrapolate()` in `scripts/w2_pilot.py`.

## D-P7-1 — F2 styling

- Original black solid, Zero Okabe–Ito vermillion `#D55E00` dashed, ContextReconstruct Okabe–Ito
  blue `#0072B2` dash-dot, P99 grey dotted; training grey, anomaly vermillion, erased yellow
  `#F0E442` shading. BRIEF3's red/blue mapped to the CVD-safe Okabe–Ito pair; the line styles
  carry identity without colour. (The skill's palette validator needs node, which is not
  installed; not run.)
- **Revert:** `figure()` in `scripts/w2_pilot.py`.

---

# W2 rev1 session 1 — implementation decisions (2026-09-24)

These are implementation choices made while carrying out brief rev1 §4–§7. None of them is a
research decision; D10–D13 and D-E3-1(rev) stay in the PENDING section below.

## D-E0-1 — `score_patch` and the E0 accuracy test design

- **What:** `MatrixProfileDetector.score_patch(x, R) -> (J, scores)` calls
  `stumpy.stump(T_A = x[lo : hi + w], m = w, T_B = reference, ignore_trivial = False)` (stumpy
  1.14.1, `stump.py:stump`) with lo/hi the first/last window start touched by R; J are score
  indices in the gate-1 convention (start + w//2) and equal rev1 §2 `J(R)` clipped to valid
  windows (`tests/test_score_patch.py::test_J_is_rev1_definition`). AB-join only.
- **E0 test (rev1 §4-3):** 20 series drawn from all 250 (seed `e0.seed`), 20 R per series,
  |R| ∈ {0.25, 0.5, 1.0} w rounded half up (min 1), R inside [train_end, n). Perturbation =
  additive Gaussian noise, sd = std(x[:train_end]). The 20 R of a series are drawn with pairwise
  disjoint I(R), so **one** full recomputation of x′ serves all 20 (a window's value depends only on
  its own points and T_B); 20 full recomputations instead of 400.
- **Alternatives:** one full recomputation per R (400; the 900k series make it hours).
- **Revert:** `configs/w2.yaml` `e0`; `scripts/w2_e0.py`.

## D-E0-2 — Execution profile: 4 workers × 2 numba threads

- **What:** E0, the gate-2 timing and the planned E2 run as 4 worker processes (spawn), each with
  `NUMBA_NUM_THREADS = 2` (8 of 10 cores). Gate 1 ran 4 workers × 10 threads (oversubscribed).
- **Why:** rev1 §7-3 fixes "Mac 4워커"; the thread count per worker was not specified.
- **Revert:** `configs/w2.yaml` `execution`.

## D-E0-3 — The W2-0 pilot test tolerance aligned to rev1 E0 (1e-6)

- **What:** `tests/test_mp_incremental.py` `ATOL` 1e-8 → **1e-6**, the criterion rev1 §0-1/§4-3 sets
  for incremental vs full scoring. With it the pilot's failing case
  (`test_incremental_equals_full[ContextReconstruct-Z]`, max 4.0e-7) passes. The researcher asked
  for this test to be handled when E0 starts (2026-09-24).
- **Not changed:** `reports/w2_pilot.md` keeps its S4 FAIL — it was judged under BRIEF3's 1e-8.
- **Revert:** set `ATOL = 1e-8`.

## D-E1-1 — B1–B6 wrapped unchanged; metadata assignments

- **What:** `src/perturb/operators.py` calls `src/perturbations.py` (what `exp_a_artifact.py`
  imports) with idx = arange(a, b); for one run the output is bit-identical to the legacy call
  (`tests/test_operators_w2.py::test_single_run_equals_legacy_call`). Legacy traits kept as they
  are: B2/B5 statistics over the whole series (incl. training part and anomaly), B3's fixed
  50-point context, B5/B6 need a generator (legacy defaults to `default_rng(0)` when none is given;
  `apply` requires an explicit seeded generator).
- **canonical_shape:** B1/B2/B3 `constant`, B4 `linear`, others `none`. "std 0" is tested as
  `ptp == 0` (stumpy's own criterion, `core.py:_rolling_isconstant`): `np.std` of 140 identical
  values returned 4.4e-16 from rounding its mean.
- **Revert:** `src/perturb/operators.py` `META`, `configs/operators.yaml`.

## D-E1-2 — Several runs: each run is filled from the unperturbed x

- **What:** rev1 D13 says each run gets the operator. The legacy functions take one index array
  and treat [min, max] as one span (B3 context, B4 interpolation would overwrite the gap between
  runs). `apply` therefore computes each run's replacement from the **unperturbed** x and writes
  only that run; stochastic operators draw from one generator in run order.
- **Open:** whether runs should instead be applied sequentially (later runs seeing earlier
  replacements) matters only for E3's multi-segment masking — listed for the researcher.
- **Revert:** the loop at the end of `apply`.

## D-E1-3 — recon_* details not fixed by the D11 text

- The donor's own contexts must exist (u − c ≥ 0, u + L + c ≤ n); they may lie outside the source
  region. A run whose own context leaves [0, n) has no donor (`NoDonorError`, counted for §11).
- Distances are computed directly (chunked), not by FFT/cumsum, so ties are decided on exact
  per-window sums; the squared distance is minimised (same argmin as the distance).
- The donor is copied raw (no offset alignment; D11 specifies none).
- **Revert:** `recon_donor` in `src/perturb/operators.py`.

## D-S-1 — Sample rule (rev1 §5)

- One series per content_group for all 89 groups (seeded choice among members sorted by num);
  the 60-group sample is a stratified subset of these (nested). Allocation: floor(60·share), zeros
  raised to 1, remaining seats by largest fractional part among the non-raised domains
  (→ ECG 34, ABP 10, Gait 7, EPG 3, Power Demand 2, Acceleration 1, RESP 1, Air Temperature 1,
  NASA 1). Every group has exactly one domain (checked); no "other" group arises.
- Metadata from the manifest, gate-1 detection.csv and the stored unperturbed gate-1 scores
  (for scale(x)); no perturbation. Written to `configs/w2_sample.yaml` and committed before E2.
- **Revert:** `scripts/w2_sample.py`, `configs/w2.yaml` `sample`.

## D-G2-1 — Gate-2 timing: 5 series run the full E2 grid; scores discarded

- **What:** the 5 series of the 89-group sample at train_end quantiles 0/.25/.5/.75/1 (nearest rank)
  run the complete E2 grid (9 operators × (20 × 3 normal + 20 SAR + 1 anomaly) regions; per region
  1 score_patch before + 9 × (operator + score_patch)) under the D-E0-2 profile. Only wall times and
  sizes are kept (`results/w2/gate2_timing.csv`); every score is discarded (rev1 §12). Per-series
  times for the whole sample come from NNLS models (score_patch: c0 + c1·n_B + c2·n_A·n_B;
  operator: c0 + c1·n + c2·n·c); wall = LPT makespan over 4 workers. Region counts per series use
  min(20, maximum disjoint positions).
- **Why:** rev1 §7-3 asks for an extrapolation from 5 measured series; measuring the full grid
  on them removes the per-region sampling assumption for those 5.
- **Revert:** `scripts/w2_gate2.py`, `configs/w2.yaml` `gate2`.

---

# PENDING — W2 rev1 결정 (연구자 승인 대기)

> **상태: 2026-09-24 연구자 승인됨 → 아래 "W2 rev1 결정 — 승인" 섹션.** (아래 문구는 PENDING 당시 원문 그대로 둔다.)
> **당시 상태: PENDING. 승인되지 않았다.** 아래는 `docs/briefs/BRIEF_W2_perturbation-experiments_rev1.md`
> §3 "승인 대기 — PENDING"의 D10–D13, D-E3-1(rev) 문구를 **그대로** 옮긴 것이다(브리프 62–130행).
> 옮긴 날짜: 2026-09-24 (W2 rev1 세션 1). 승인 문구와 결정 전용 커밋은 세션 2에서 만든다(브리프 §3, §13).
> 이 섹션이 PENDING인 동안 E2·E3는 실행하지 않는다(브리프 §11).

**D10 — SAR 정의 (rev0 D9의 SAR 부분을 대체)**

- 이상 구간 `R_anom = [GT_start, GT_stop)` (GT 전체). `L = |GT|`.
- `AI_anom = AI(x, R_anom, op)`, 부호 유지.
- 제거 반응 `removal = max(0, −AI_anom)`. `AI_anom > 0`이면 `reversal = True`로 표기한다. 연산자가 이상 구간의 점수를 오히려 올린 경우다.
- 분모: 정상 섭동 허용 조건을 만족하는 **길이 L** 구간 20개(서로 겹치지 않음, seed 고정)의 `median |AI_normal_L|`.
- `SAR = removal / median|AI_normal_L|`.
- 분모 `< 1e-6`이면 `SAR = NaN`, `sar_den_zero = True`, 건수를 보고한다.
- 길이 L 구간이 10개 미만만 들어가면 `sar_few_positions = True`로 표기한다. 5개 미만이면 그 시리즈는 SAR에서만 제외한다 (E2a에서는 제외하지 않는다).
- **보조 지표** `SAR_abs = |AI_anom| / median|AI_normal_L|`(rev0 정의)를 나란히 보고한다. 정의 선택이 결론을 바꾸는지 보이기 위해서다.

**D11 — 재구성형 연산자의 donor 출처**

- **`recon_test` (주 연산자)**
  - donor 후보: 테스트 구간 안이면서 아래 영역과 겹치지 않는 길이 `|R|` 구간.
    - `E`
    - `I(R)`
    - (다중 구간 섭동이면) 다른 모든 구간의 `I(·)`
    - 가장자리 문맥 구간
  - 가장자리 문맥 길이 `c = max(1, w // 4)`.
  - 선택 기준: `x[a−c:a] ‖ x[b:b+c]`와 `x[u−c:u] ‖ x[u+L:u+L+c]`의 **원시값 유클리드 거리**가 최소인 `u`. 동률이면 가장 작은 `u`. 결정적이다.
  - donor는 **섭동 전 원본 x**에서 가져온다.
- **`recon_train` (대조, 순환성 시연)**
  - 같은 기준으로 donor를 **학습 구간**에서 가져온다.
  - 주 분석의 결론에는 쓰지 않는다. D12-b 검증과 "donor 출처가 AI를 결정한다"는 대조에만 쓴다.
- rev0의 "선형 보간" 선택지는 채택하지 않는다. 직선은 z-정규화하면 기울기와 무관하게 같은 모양이 되기 때문이다. B1–B6 안에 선형 보간이 있으면 메타데이터로만 표기한다 (§6).

**D12 — 사전 등록 예측 (rev0 §6의 D12-a를 대체)**

예측 대상은 `resp`가 아니라 **`j*` 윈도우의 점수**다. 허용오차 `tol`은 §4의 stumpy 소스 추적 결과로 정해 보고서에 적는다.

- **D12-a.** 연산자 메타데이터 `canonical_shape = constant`, `|R| = w`, `n_const_train = 0`이면:
  - (i) `s(j*)_after = √w` (등식, `|Δ| ≤ tol`)
  - (ii) `resp_after ≥ √w − tol` (부등식)
  - 근거: stumpy는 한쪽만 상수인 쌍을 `D² = m`으로 둔다(W1 D12, 소스 확인). z-정규화 거리 `D² = 2m(1 − ρ)`에서 이는 `ρ = 0.5`에 해당한다.
  - 주의: (i)은 신호 진폭과 무관하지만 AI는 무관하지 않다. `resp_before`와 `scale(x)`가 시리즈마다 다르기 때문이다. 보고서에서 이 둘을 섞지 않는다.
- **D12-a′.** 같은 조건에서 `n_const_train > 0`이면 `s(j*)_after = 0` (`|Δ| ≤ tol`).
- **D12-b.** `recon_train`, `|R| = w`이면 `s(j*)_after ≈ 0` (`≤ tol`). donor 윈도우가 참조 집합에 그대로 있기 때문이다.
- **판정**: 각 예측에서 조건을 만족하는 사례의 **1% 초과**가 어긋나면 "체계적 불일치"로 STOP한다 (§11).
- 위 조합 외에는 예측하지 않는다. 탐색적으로만 보고한다.

**D13 — E3의 설명 대상과 영역**

- 설명 대상 출력: `f(x) = resp(x, R_anom)`.
- attribution 영역: `Ω = I(R_anom) ∩ 테스트 구간`. `f`는 Ω 밖 입력에 **정확히 무관**하다(AB-join 구조). 따라서 Ω 밖 attribution은 정의하지 않는다.
- 세그먼트: 길이 `g = max(1, w // 4)`로 Ω를 분할한다. 세그먼트 수 `K > 64`이면 `g`를 키워 `K ≤ 64`로 맞추고, 시리즈별 `g`를 기록한다.
- 다중 세그먼트 마스킹: 가려진 세그먼트를 **최대 연속 구간(run)**으로 묶고, 각 run에 연산자를 적용한다.
- **AM별 정의**
  - `RandomAttribution`: 세그먼트별 균등 난수, seed 고정.
  - `FeatureAblation`: `f(x) − f(x with segment k masked)`, 설명용 연산자 사용.
  - `KernelSHAP`: 세그먼트 단위, 설명용 연산자로 마스킹. 표본 수 `S`는 E3 예산 측정 후 고정하고 기록한다.
  - `MPNative`: `resp`를 달성한 윈도우 `j_max`와 그 최근접 학습 이웃에 대해, 시점별 기여 `(ẑ_q,i − ẑ_nn,i)²`를 세그먼트로 합산한다. `j_max`의 support 밖은 0. **새로움을 주장하지 않는다.** 선행연구 확인은 연구자 측에서 한다 (§14).
- **충실성 곡선**
  - MoRF/LeRF를 세그먼트 비율 10단계(10%, 20%, …, 100%)로 계산한다.
  - `max_diff = scale(x)` (D8).
  - DDS/PES/CMI는 `faithfulness_simic.py`를 재사용한다.
- **AM 순위에 쓰는 지표**: Šimić et al. 저장소(commit `edc6a870`)에서 AM 순위 산출에 쓰인 지표를 **소스 추적해 동일하게** 쓴다. 추적 결과(파일:함수)를 기록한다. 임의로 고르지 않는다.
- **집계**
  - (주) 연산자별로 AM 지표를 `content_group` 평균 → AM 순위 → 연산자 쌍 간 Spearman ρ, 전체 Kendall's W.
  - `content_group` 단위 부트스트랩 B = 1000으로 신뢰구간을 낸다.
  - (보조) 그룹별 순위의 W 분포.
  - (보조) 연산자별 AM 쌍 비교는 Wilcoxon signed-rank + Holm.
  - AM이 4개라 순위 쌍의 ρ는 11개 값만 가진다. 보고서에 이 해상도 한계를 명시한다.

**D-E3-1 (rev) — 설명용/평가용 연산자 분리**

- 설명용 연산자는 `recon_test`로 고정한다.
- 주 분석의 평가용 연산자에서 `recon_test`는 **제외**한다 (자기 일치 칸 제거). `recon_train`도 같은 계열이므로 주 분석에서 제외하고 별도로 보고한다.
- **민감도 분석**: 평가용 연산자마다 "설명용 = 평가용"으로 FeatureAblation·KernelSHAP를 다시 계산한다. 주 조건 대비 순위 변화로 자기 일치 효과의 크기를 보고한다.

---

# W2 rev1 결정 — 승인 (2026-09-24, 연구자)

이 섹션은 결정만 담은 커밋으로 기록된다(브리프 rev1 §3). 이 커밋 이후에만 E2·E3를 실행한다.

## D10, D11, D13, D-E3-1(rev) — 승인 (rev1 §3 문구 그대로)

- 위 PENDING 섹션의 D10, D11, D13, D-E3-1(rev) 문구를 **수정 없이** 승인한다.
- **추가(연구자):** `identity`는 E3 평가용 연산자에서 제외한다. **E3 주 분석의 평가용 연산자는 B1–B6**이다.
  (`recon_test`는 설명용, `recon_train`은 별도 보고 — D-E3-1(rev) 그대로.)

## D12 — 승인, tol 확정

- D12-a, D12-a′, D12-b 문구 그대로 승인.
- **tol:** D12-a (i)(ii), D12-a′ = **1e-12**; D12-b = **1e-3**.
- **근거:**
  - stumpy 1.14.1 `stump.py:201-204` (`_compute_diagonal`): 양쪽 상수 → `pearson = 1.0`, 한쪽 상수 →
    `pearson = 0.5`를 **정확히 대입**한다. 따라서 D = √m, D = 0이 부동소수 연산 없이 결정된다(1e-12는 여유).
  - `stump.py:207` (`pearson = min(1.0, pearson)`): 상관이 1에 가까울 때 거리를 0으로 맞추는 임계값이 없다.
    참 거리 0(정확 일치)도 누적 오차만큼 양수로 나온다.
  - 게이트 1 자기일치 잔차(학습 구간 창 vs 자기 자신, 250개): 최대 **3.3e-4 (#249, w = 24)**. D12-b의 1e-3은 이 값의 약 3배.
- **보고 의무:** E2 보고서에 D12-b 사례의 실제 `s(j*)_after` 분포(최댓값, 99% 분위)를 포함한다.

## 표본 — 89그룹 전수

- E2·E3는 **89 content_group 전수(그룹당 1개, `configs/w2_sample.yaml`의 `series` 전체)**로 한다.
- 근거: rev1 §5 "게이트 2에서 89그룹 전체의 예상 시간이 예산 안이면 89그룹으로 올린다" — 게이트 2 예상 0.20 h ≤ 2 h
  (`reports/w2_gate2.md`, 커밋 `af12c15`).
- 60그룹 결과는 따로 내지 않는다.

## D-E1-2 — 승인

- 다중 run은 **각 run을 원본 x 기준으로 독립 처리**(현재 구현)한다.
- **보고 의무:** E3 보고서에 B3의 50점 문맥이 다른 run과 겹친 빈도를 한계로 보고한다.

## B2 / B5 — legacy 정의 유지

- B2(전체 시리즈 평균), B5(전체 시리즈 평균·표준편차의 정규분포)의 legacy 정의를 유지한다.
- 근거(연구자): 완전히 덮인 윈도우에서는 z-정규화 때문에 채움값의 수준·척도가 점수에 영향을 주지 못한다.

## D-E2a-1 — E2a 위치 부족 규칙 (신규)

- 길이별로 확보한 정상 위치가 **10개 이상**이면 확보한 만큼 쓰고 `positions_short` 플래그를 단다(20개 미만일 때).
- **10개 미만**이면 그 시리즈를 **그 길이에서만** 제외한다.

## D-PRE-1 — 사전 노출 기록 (신규)

- W2-0 파일럿(BRIEF3, 커밋 `b96ee3a`)이 존재했다. 시리즈 #066 1개에서 **Šimić et al. 지우개**
  (Zero, SampleMean, OutOfDistHigh, Inverse, UniformNoise100, LinearInterpolation, NearestNeighborWindow)와
  ContextReconstruct를 썼다. rev1의 B1–B6, recon_test, recon_train과는 다른 연산자다.
- 파일럿의 섭동 효과 값(ΔS, 가짜 경보 등, `reports/w2_pilot.*`)을 **Claude Code가 봤다.**
- rev1의 결정들(D10–D13, D-E3-1(rev))은 파일럿 결과를 **참조하지 않고** 작성됐다.
- 연구자의 파일럿 결과 열람 여부: **[보지 않음]**.

## D-C5-4 — 학습 구간 자기일치 sanity check (W1에서 결정, 사후 이관)

- **W1에서 결정, 사후 이관.** 근거 커밋 `6b5d7c1` (2026-09-23, 게이트 1 MatrixProfile PASS):
  `scripts/03_gate1.py`(주석 "Sanity check pre-registered in D-C5-4", `train_selfmatch_max`),
  `scripts/04_report.py` §5a, `reports/gate1_MatrixProfile.md` §5a.
- **내용:** AB-join에서 참조(학습) 구간 안에 완전히 들어가는 창은 자기 자신과 일치하므로 점수 ≈ 0이어야 한다.
  패딩된 앞부분과 `train_end`에 걸치는 창은 제외한다. 결과: 최대 3.3e-4 < 1e-3 → 예측 성립.
  따라서 학습 구간은 Artifact Index의 정상 구간에서 제외한다(s_normal은 테스트 구간 기준).
- 이 문서에 항목이 없었던 것은 누락이며, 내용은 위 커밋의 코드·보고서 그대로다.

## D-C5-5 — 윈도우 support 기준 P1은 진단용 (W1에서 결정, 사후 이관)

- **W1에서 결정, 사후 이관.** 근거 커밋 `6b5d7c1`: `scripts/03_gate1.py`(주석 "Diagnostics, not criteria
  (D-C5-5)", `max_support`, `p1_support_bw`), `scripts/04_report.py`, `reports/gate1_MatrixProfile.md` §2.
  D-C5-6(이 문서)도 이 항목을 참조한다.
- **내용:** 창 점수는 `start + w//2`에 기록되므로 이상 구간의 높은 점수는 support
  `[start − w + 1 + w//2, stop − 1 + w//2)`에 걸친다. w보다 짧은 이상에서는 최고점이 P1이 읽는
  `[start0, stop0)` 밖에 놓일 수 있다. `p1_support_bw`는 **진단**으로만 보고하고 판정 기준이 아니다.
  rev1 §2의 `support(j)` 정의가 이것을 따른다.


---

# W2 E2/E3 — implementation decisions (2026-09-24, after the decision commit f29eddf)

Implementation choices only; the research decisions are in the approved section above.

## D-E2-1 — Outputs are `.csv.gz`, not `.parquet`

- **What:** rev1 §8 names `results/w2/ai.parquet`, `sar.parquet`. The reference env (`tsadxai-w1`,
  `environment.yml`) has no parquet engine (pyarrow / fastparquet absent, checked 2026-09-24), and
  adding one to the reference env is what D-D3-1 forbids. Same columns, written as
  `results/w2/ai.csv.gz`, `sar.csv.gz` (and `faithfulness.csv.gz` for E3).
- **Revert:** add pyarrow to `environment.yml`, rebuild, re-lock, switch `to_csv` → `to_parquet`.

## D-E2-2 — E2 details not fixed by the brief

- Normal positions are pairwise disjoint **within** a length; the three lengths are drawn
  independently (seed `e2_grid.position_seed` per series). Stochastic operators draw from
  `default_rng([operator_seed, series, region, operator])`.
- D12 cases: every region with |R| = w (normal 1.0w, and SAR/anomaly regions when |GT| = w).
- Report primary statistic: median across series of per-series medians (content_group = series);
  pooled region medians alongside. "Flagged" series (for with/without tables): positions_short,
  a dropped length, sar_few_positions, SAR-excluded, sar_den_zero for a non-identity operator, or any
  no-donor region.
- **Revert:** `scripts/w2_e2.py`, `scripts/w2_e2_report.py`.

## D-E3-1 — KernelSHAP = numpy port of captum 0.9.0 KernelShap

- captum is not in the reference env; Šimić et al. called captum `KernelShap` with defaults
  (`interpretability_methods.py:127-143`). The port (`faithfulness_ad.kernel_shap`) follows
  `captum/attr/_core/kernel_shap.py:268-365` (0.9.0, read in `tsadxai-simic`): all-present and
  all-absent samples first with weight 1e6, then k ~ p(k) ∝ (K−1)/(k(K−k)) and a uniform subset of size
  k, weight 1; weighted least squares with intercept; coefficients = attributions. Not tested for
  bit-equality with captum (different RNG and solver).

## D-E3-2 — KernelSHAP sample count S (rule fixed before the budget run)

- **Rule:** S = 2K + 2048 (shap `KernelExplainer` nsamples="auto"). The captum default (25, what
  Šimić et al. ran with per-time-point features) is reported in the budget as the alternative; with
  K up to 64 segments 25 samples leave the regression underdetermined.
- If the predicted E3 time (main + sensitivity) exceeds 8 h → STOP with the S / time table (rev1 §9).
- Budget model: 3 series (train_end quantiles 0/.5/1 of the sample); per evaluation kind the mean
  time over 10 evaluations; per-kind model c0 + c1·n_A·n_B + c2·n (NNLS; 3 points); counts per series
  from K and S; wall = LPT over 4 workers.

## D-E3-3 — Ranking aggregation details

- Per (evaluation operator, AM): DDS per series (= content_group) → mean → CMI(mean DDS, PES of the
  DDS list) — the traced Šimić metric (`rank_ams_by_cmi`, `rank_by_metric = 'cmi-mean'`).
  Ties in CMI (e.g. several AMs at CMI = 0) → average ranks (upstream's stable `sorted` would order
  ties by list position).
- Kendall's W without tie correction; bootstrap percentile CI, B = 1000, seed 20260924.
- Per-group ranking (secondary) uses DDS directly (CMI needs more than one sample).
- Curves: steps mask ceil(t·K/10) segments; MoRF/LeRF of the same step share the operator seed so
  the 100 % points coincide for stochastic operators.

## D-E3-4 — ReconCache (speed only)

- `src/perturb/operators.ReconCache` returns exactly the donor `recon_donor` returns (sorted
  (distance, u) per run, then the first candidate clearing the other runs' constraints);
  `tests/test_operators_w2.py::test_recon_cache_equals_apply`.
