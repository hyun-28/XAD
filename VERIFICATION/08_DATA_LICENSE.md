# Data licence — UCR Time Series Anomaly Archive 2021

Checked 2026-09-21 as part of BRIEF Task A (A1).

## What the archive itself says

- No `LICENSE`, `README.txt` or terms-of-use file exists in the zip (full inventory in
  `reports/data_audit.md`, "Supplementary documents preserved").
- A keyword pass over every slide and MATLAB file (`licence`, `copyright`,
  `creative commons`, `CC BY`, `terms`, `permission`, `redistribut…`) found no licence
  statement; see the `licence` section of the keyword index in `docs/ucr_supplement_text.md`
  (the hits there are false positives on the word "terms").
- The only usage statement is the citation request in `Start_Here_This_is_a_Read_me.pptx`,
  slide 1:

  > To cite this resource: Keogh, E., Dutta Roy, T., Naik, U. & Agrawal, A (2021).
  > Multi-dataset Time-Series Anomaly Detection Competition, SIGKDD 2021.
  > https://compete.hexagon-ml.com/practice/competition/39/

## What the download page says

- `https://www.cs.ucr.edu/~eamonn/time_series_data_2018/` (fetched 2026-09-21) links the zip
  directly. The page HTML contained no Creative Commons markup or licence wording at fetch
  time. The CC BY-NC 4.0 notice mentioned in BRIEF A1 was **not found** on this page; it may
  refer to a different page or an earlier revision. This is unresolved.

## Upstream sources named in the deck

Many series are derived from PhysioNet records (sddb, afpdb, apnea-ecg, ltstdb, qtdb, mimicdb,
bidmc, gaitndd), CIMIS weather data, an internal-bleeding dataset
(`http://mathieu.guillame-bert.com`), the DLR activity-recognition benchmark, the GaitPhase
database, NASA telemetry (KDD 2018), Italian power demand, CHARIS, Fantasia and STAFF III.
Each carries its own terms; the per-series statement is in the manifest column
`source_official`.

## Researcher action

Decide, before publication, which terms govern redistribution of derived artefacts
(manifest, figures). Raw series are not committed to this repository (`.gitignore`), only
their SHA256 checksums.
