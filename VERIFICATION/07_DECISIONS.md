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
