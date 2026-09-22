# Data licence — UCR Time Series Anomaly Archive 2021

Checked 2026-09-21 (BRIEF Task A, A1); settled 2026-09-22 by researcher decision
(`docs/DECISIONS.md` D-A1-4).

## Decision

The archive is **public research data with no explicit licence**. It is used under the
authors' stated terms, which are a citation request (below).

- Raw series are **not redistributed**: `data/raw/` stays git-ignored; only
  `data/raw/ucr/CHECKSUMS.sha256` is committed. Anyone reproducing this work downloads the
  official zip from the URL below and verifies it with `scripts/01_download.py`
  (per-file SHA256 comparison; see README).
- Derived artefacts — `data/manifest/ucr_manifest.csv`, detector scores, figures, reports —
  are published with this repository.

## Citations to use

1. The request in `Start_Here_This_is_a_Read_me.pptx` (archive root, slide 1), verbatim:

   > To cite this resource: Keogh, E., Dutta Roy, T., Naik, U. & Agrawal, A (2021).
   > Multi-dataset Time-Series Anomaly Detection Competition, SIGKDD 2021.
   > https://compete.hexagon-ml.com/practice/competition/39/

2. Wu, R. & Keogh, E. "Current Time Series Anomaly Detection Benchmarks are Flawed and are
   Creating the Illusion of Progress." *IEEE TKDE* 35(3):2421–2429, 2023 (arXiv:2009.13807).

## What the archive itself says

- No `LICENSE`, `README.txt` or terms-of-use file exists in the zip (full inventory in
  `reports/data_audit.md`, "Supplementary documents preserved").
- A keyword pass over every slide and MATLAB file (`licence`, `copyright`,
  `creative commons`, `CC BY`, `terms`, `permission`, `redistribut…`) found no licence
  statement; see the `licence` section of the keyword index in `docs/ucr_supplement_text.md`
  (the hits there are false positives on the word "terms").
- The only usage statement is the citation request quoted above.

## What the download page says

- `https://www.cs.ucr.edu/~eamonn/time_series_data_2018/` (fetched 2026-09-21; re-checked by
  the researcher 2026-09-22) links the zip directly and carries no licence or terms-of-use
  wording. The citation it gives (Dau et al. 2019) is for the UCR *classification* archive
  and is not used for this data.
- The CC BY-NC 4.0 notice mentioned in BRIEF A1 is the footer of a personal website
  (wu.renjie.im), not a licence attached to the archive. It is not applied.

## Upstream sources named in the deck

Many series are derived from PhysioNet records (sddb, afpdb, apnea-ecg, ltstdb, qtdb, mimicdb,
bidmc, gaitndd), CIMIS weather data, an internal-bleeding dataset
(`http://mathieu.guillame-bert.com`), the DLR activity-recognition benchmark, the GaitPhase
database, NASA telemetry (KDD 2018), Italian power demand, CHARIS, Fantasia and STAFF III.
Each carries its own terms; the per-series statement is in the manifest column
`source_official`. Because raw series are not redistributed, those terms bind only the
original download, not this repository.
