"""02_audit.py -- BRIEF Task A: A1 (report), A2, A3, A5, A6.

Reads the OFFICIAL copy produced by scripts/01_download.py (never the
researcher's local copy), audits all 250 series, and writes:

  docs/ucr_supplement_text.md            slide/.m text + keyword index (A1)
  data/manifest/ucr_manifest.csv         one row per series (A6)
  reports/figures/index_convention/*.png empirical convention evidence (A2)
  reports/index_convention_evidence.csv  the numbers behind those figures
  reports/domain_unmatched.csv           series with no official domain
  reports/data_audit.md                  the report (A6, six sections + A1)

Exit code: 2 if any STOP-severity finding (BRIEF §7-1/§7-5), 1 if any
WARN-severity finding, else 0. Nothing is skipped silently: every finding
is in the manifest's audit_flags and in the report.

Usage:  python scripts/02_audit.py [--no-figures]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import REPO_ROOT, load_yaml, resolve  # noqa: E402
from src.data.audit_checks import SEVERITY, check_series, index_evidence  # noqa: E402
from src.data.conventions import (ConventionUndetermined, active_convention,  # noqa: E402
                                  to_half_open, train_prefix_stop)
from src.data.domains import is_medical, load_goswami, official_domain  # noqa: E402
from src.data.provenance import match_series, parse_deck  # noqa: E402
from src.data.supplement import extract_all  # noqa: E402
from src.data.ucr import file_layout, load_series, scan_archive  # noqa: E402
from src.runlog import env_header  # noqa: E402

MANIFEST_COLUMNS = [
    "num", "name", "filename", "sha256", "length", "train_end_raw", "begin_raw", "end_raw",
    "start0", "stop0", "train_stop0", "index_convention",
    "anom_len", "variant", "is_distorted", "is_noise",
    "domain_goswami", "is_medical",
    "domain_official", "domain_official_basis", "domain_agree", "source_official", "deck_slides",
    "in_tsbad", "tsbad_filename", "tsbad_domain", "match_method",
    "tsbad_label_n_segments", "tsbad_label_start0", "tsbad_label_stop0", "gt_agree",
    "audit_flags",
]
NA = "NA"


def md_table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(map(str, header)) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    header = env_header()
    print(header)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    paths = load_yaml("paths")
    ucr_cfg = paths["ucr"]
    n_expected = int(ucr_cfg["expected_n_series"])
    reports_dir = resolve(paths["reports_dir"])
    docs_dir = resolve(paths["docs_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    # ---- A1: the download/checksum step must have run and passed -----------
    a1_path = resolve(ucr_cfg["a1_result"])
    if not a1_path.is_file():
        raise FileNotFoundError(f"{a1_path} missing: run scripts/01_download.py first")
    a1 = json.loads(a1_path.read_text())
    if a1["stop"]:
        print("STOP recorded by 01_download.py:", a1["stop_reasons"])
        return 2
    official_anchor = Path(a1["anchors"]["official"])
    archive_root = official_anchor / Path(ucr_cfg["fulldata_subdir"]).parts[0]
    fulldata_dir = official_anchor / ucr_cfg["fulldata_subdir"]

    # ---- A1: supplementary documents ---------------------------------------
    supp = extract_all(archive_root, docs_dir / "ucr_supplement_text.md")
    print(f"supplement: {len(supp['documents'])} documents, {len(supp['hits'])} keyword hits")

    # ---- A2: filenames -------------------------------------------------------
    metas = scan_archive(fulldata_dir, n_expected)   # raises on STOP §7-1
    print(f"parsed {len(metas)} filenames, numbering 001..{n_expected:03d} OK")

    conv_err = None
    try:
        conv = active_convention()
    except ConventionUndetermined as e:
        conv, conv_err = None, str(e)
    print(f"index convention: {conv.value if conv else 'UNDETERMINED'}")

    # ---- A5 inputs -------------------------------------------------------------
    goswami, goswami_prov = load_goswami()
    deck = parse_deck(archive_root / "FilesAreInHere" / "UCR_AnomalyDataSets.pptx")

    # ---- A3 + manifest rows ----------------------------------------------------
    rows, evidence, figures = [], [], []
    series_cache = {}
    for m in metas:
        p = fulldata_dir / m.filename
        x = load_series(p)
        layout = file_layout(p)
        a3 = check_series(x, m, layout)
        flags = list(a3.flags)

        if conv is not None and "BEGIN_GT_END" not in flags:
            start0, stop0 = to_half_open(m.begin_raw, m.end_raw, conv)
            train_stop0 = train_prefix_stop(m.train_end_raw)
        else:
            start0 = stop0 = train_stop0 = NA
        anom_len = m.end_raw - m.begin_raw + 1  # inclusive count, slide 4 "L = end - begin + 1"

        prov = match_series(deck, m.name)
        if not prov["deck_found"]:
            flags.append("NOT_IN_DECK")
        deck_fields = prov["deck_fields"]
        if deck_fields and not any(f["train_end"] == m.train_end_raw and f["begin"] == m.begin_raw
                                   and f["end"] == m.end_raw for f in deck_fields):
            flags.append("DECK_FIELDS_DIFFER")
        dom_off, basis = official_domain(prov["source_lines"], prov["slide_text"])
        if dom_off is None:
            flags.append("DOMAIN_OFFICIAL_NA")
        dom_gos = goswami.get(m.entity)
        if dom_gos is None:
            raise KeyError(f"{m.entity} missing from Goswami mapping")
        if dom_off is not None and dom_off != dom_gos:
            flags.append("DOMAIN_DISAGREE")

        if anom_len <= 2:
            ev = index_evidence(x, m)
            evidence.append({"num": m.num, "filename": m.filename, "anom_len_incl": anom_len, **ev})
            if not args.no_figures:
                figures.append((m, x))

        rows.append({
            "num": m.num, "name": m.name, "filename": m.filename, "sha256": sha256_file(p),
            "length": a3.length, "train_end_raw": m.train_end_raw, "begin_raw": m.begin_raw,
            "end_raw": m.end_raw, "start0": start0, "stop0": stop0, "train_stop0": train_stop0,
            "index_convention": conv.value if conv else NA,
            "anom_len": anom_len, "variant": m.variant, "is_distorted": m.is_distorted,
            "is_noise": m.variant == "NOISE",
            "domain_goswami": dom_gos, "is_medical": is_medical(dom_gos),
            "domain_official": dom_off if dom_off is not None else NA,
            "domain_official_basis": basis,
            "domain_agree": (dom_off == dom_gos) if dom_off is not None else NA,
            "source_official": prov["source_primary"] or NA,
            "deck_slides": ";".join(map(str, prov["deck_slides"])) or NA,
            # A4 not run this session (optional step; DECISIONS.md D-A4-1)
            "in_tsbad": NA, "tsbad_filename": NA, "tsbad_domain": NA, "match_method": NA,
            "tsbad_label_n_segments": NA, "tsbad_label_start0": NA, "tsbad_label_stop0": NA,
            "gt_agree": NA,
            "audit_flags": ";".join(flags) or "none",
            "_train_std": a3.train_std, "_layout": layout, "_flags": flags,
            "_n_minus999": a3.n_minus999,
        })
    print(f"audited {len(rows)} series")

    # ---- manifest -----------------------------------------------------------------
    manifest_path = resolve(paths["manifest"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {manifest_path}")

    # ---- figures + evidence csv -------------------------------------------------------
    ev_path = reports_dir / "index_convention_evidence.csv"
    if evidence:
        with ev_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(evidence[0].keys()))
            w.writeheader()
            w.writerows(evidence)
    fig_dir = reports_dir / "figures" / "index_convention"
    if figures:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig_dir.mkdir(parents=True, exist_ok=True)
        for m, x in figures:
            b, e = m.begin_raw, m.end_raw
            lo, hi = max(0, b - 31), min(len(x), e + 30)
            fig, ax = plt.subplots(figsize=(9, 3.2))
            ax.plot(range(lo, hi), x[lo:hi], "k.-", lw=0.8, ms=3)
            ax.axvspan(b - 1 - 0.5, e - 1 + 0.5, color="tab:green", alpha=0.25,
                       label=f"MATLAB 1-based incl. -> 0-based [{b-1},{e})")
            ax.axvspan(b - 0.5, e + 0.5, color="tab:red", alpha=0.18,
                       label=f"0-based incl. -> [{b},{e+1})")
            ax.set_title(m.filename, fontsize=9)
            ax.set_xlabel("0-based index")
            ax.legend(fontsize=7, loc="best")
            fig.tight_layout()
            fig.savefig(fig_dir / f"{m.num:03d}.png", dpi=110)
            plt.close(fig)
        print(f"wrote {len(figures)} figures to {fig_dir}")

    # ---- domain_unmatched.csv ---------------------------------------------------------
    unmatched = [r for r in rows if r["domain_official"] == NA or r["domain_agree"] is False]
    with (reports_dir / "domain_unmatched.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["num", "name", "domain_official", "domain_goswami", "deck_slides", "source_official"])
        for r in unmatched:
            w.writerow([r["num"], r["name"], r["domain_official"], r["domain_goswami"],
                        r["deck_slides"], r["source_official"]])

    # ---- report ------------------------------------------------------------------------
    flag_counter = Counter(fl for r in rows for fl in r["_flags"])
    sev_counter = Counter(SEVERITY[fl] for fl in flag_counter.elements())
    stop_rows = [r for r in rows if any(SEVERITY[fl] == "STOP" for fl in r["_flags"])]
    warn_rows = [r for r in rows if any(SEVERITY[fl] == "WARN" for fl in r["_flags"])]

    def xtab(rs, key):
        t = defaultdict(lambda: Counter())
        for r in rs:
            t[r[key]][r["variant"]] += 1
        doms = sorted(t, key=lambda d: (str(d)))
        hdr = ["domain", "plain", "DISTORTED", "NOISE", "total"]
        body = [[d, t[d]["plain"], t[d]["DISTORTED"], t[d]["NOISE"], sum(t[d].values())] for d in doms]
        tot = Counter(r["variant"] for r in rs)
        body.append(["**total**", tot["plain"], tot["DISTORTED"], tot["NOISE"], len(rs)])
        return md_table(hdr, body)

    L = []
    L += [f"# UCR Anomaly Archive 2021 — data lineage audit (BRIEF Task A)", "",
          f"GENERATED by scripts/02_audit.py at {now}. Do not hand-edit.", "",
          f"`{header}`", ""]

    # A1
    c = a1["compare"]
    L += ["## A1. Official archive vs local copy", "",
          md_table(["", "official (downloaded)", "local (researcher's copy)"], [
              ["zip", f"`{Path(a1['official']['zip_path']).name}` {a1['official']['zip_bytes']:,} B",
               f"`{Path(a1['local']['zip_path']).name}` {a1['local']['zip_bytes']:,} B"],
              ["zip sha256", a1["official"]["zip_sha256"], a1["local"]["zip_sha256"]],
              ["files compared", a1["official"]["n_files"], a1["local"]["n_files"]],
              ["series .txt", a1["official"]["n_series_txt"], a1["local"]["n_series_txt"]],
          ]), "",
          md_table(["result", "count"], [
              ["match (identical sha256)", len(c["match"])],
              ["mismatch", len(c["mismatch"])],
              ["local only", len(c["local_only"])],
              ["official only", len(c["official_only"])],
              ["**series .txt mismatch / local-only / official-only**",
               f"**{len(c['series_mismatch'])} / {len(c['series_local_only'])} / {len(c['series_official_only'])}**"],
          ]), "",
          f"Source URL: `{a1['official']['url']}`  ", f"Fetched: {a1['generated_utc']}  ",
          f"Expected series count: {a1['expected_n_series']} — observed {a1['official']['n_series_txt']}.", ""]
    for k in c["mismatch"] + c["local_only"] + c["official_only"]:
        L.append(f"- differs: `{k}`")
    inner = a1.get("inner_zip_vs_extracted", {})
    if inner.get("present"):
        L += ["", f"The official zip contains the archive twice: an inner zip "
              f"(`{Path(inner['inner_zip']).name}`, identical to the local zip) and an extracted tree. "
              f"Inner-zip members identical to the extracted tree: {inner['n_same']}; different: "
              f"{len(inner['differences'])}. The audit uses the extracted tree (newer packaging)."]
        for d in inner["differences"]:
            L.append(f"  - `{d['member']}` inner {d.get('inner_bytes')} B (dated {d.get('inner_date')}) "
                     f"vs extracted {d.get('extracted_bytes')} B")
    L += ["", "### Supplementary documents preserved (zip contents other than the 250 series)", "",
          md_table(["path", "bytes", "parsed"], [[f"`{d['path']}`", f"{d['bytes']:,}",
                                                 "text" if d["kind"] in (".pptx", ".m") else "listed only"]
                                                for d in supp["documents"]]),
          "", "Text of the .pptx and .m files: `docs/ucr_supplement_text.md`.", ""]

    # 1. summary
    L += ["## 1. Summary", "",
          md_table(["item", "value"], [
              ["original series (official archive)", len(rows)],
              ["TSB-AD UCR subset", "NA — A4 not run (optional; DECISIONS D-A4-1)"],
              ["mapped to TSB-AD", "NA"], ["missing from TSB-AD", "NA"], ["GT disagreements (TSB-AD)", "NA"],
              ["A3 findings: STOP-severity", sev_counter["STOP"]],
              ["A3 findings: WARN-severity", sev_counter["WARN"]],
              ["A3 findings: INFO", sev_counter["INFO"]],
              ["series with any STOP flag", len(stop_rows)],
              ["series with any WARN flag", len(warn_rows)],
          ]), "",
          "Flag counts:", "",
          md_table(["flag", "severity", "series"], [[k, SEVERITY[k], v] for k, v in sorted(flag_counter.items())]),
          ""]
    if stop_rows:
        L += ["**STOP (BRIEF §7-5): filename GT contradictions**", ""]
        L += [f"- {r['filename']}: {r['audit_flags']}" for r in stop_rows]
        L.append("")
    if warn_rows:
        L += ["WARN:", ""] + [f"- {r['filename']}: {r['audit_flags']}" for r in warn_rows] + [""]
    single = [r for r in rows if r["_layout"]["single_line"]]
    single_ids = ", ".join("%03d" % r["num"] for r in single)
    L += ["### A3 format facts", "",
          f"- single-line files (slide 7 'Known issues' lists 204-208, 225, 226, 242, 243): "
          f"measured {len(single)} → {single_ids}",
          f"- of those, separator contains TAB: {sum(r['_layout']['has_tab'] for r in single)}; "
          f"CRLF line endings anywhere: {sum(r['_layout']['has_crlf'] for r in rows)} files",
          f"- all files parse to 1-D float64 with no NaN/inf: "
          f"{'yes' if not flag_counter['NAN_OR_INF'] and not flag_counter['NOT_1D_FLOAT64'] else 'NO'}",
          f"- length min / median / max: {min(r['length'] for r in rows):,} / "
          f"{int(np.median([r['length'] for r in rows])):,} / {max(r['length'] for r in rows):,}",
          f"- training prefix constant (std = 0): {flag_counter['CONST_TRAIN_PREFIX']} series",
          "- series containing the legacy sentinel −999 (slide 46 inserts one as an anomaly): "
          + (", ".join(f"{r['num']:03d} ({r['_n_minus999']}×, {int((load_series(fulldata_dir / r['filename'])[:r['train_end_raw']] == -999.0).sum())} in training prefix)"
                       for r in rows if r["_n_minus999"]) or "none"), ""]

    # 2. cross-tabs
    L += ["## 2. Domain × variant cross-tabs", "",
          "### 2a. Goswami 9-domain mapping (original 250)", "", xtab(rows, "domain_goswami"), "",
          "### 2b. Official-source domain (derived from UCR_AnomalyDataSets.pptx; original 250)", "",
          xtab(rows, "domain_official"), "",
          "### 2c. TSB-AD UCR subset", "", "NA — A4 not run.", ""]

    # 3. non-medical subset
    nonmed = [r for r in rows if r["is_medical"] is False]
    nonmed_nd = [r for r in nonmed if not r["is_distorted"]]
    nonmed_plain = [r for r in nonmed if r["variant"] == "plain"]
    def bold_if_small(n):
        return f"**{n} (< 30)**" if n < 30 else str(n)
    L += ["## 3. Non-medical subset (is_medical = False ⇔ domain_goswami ∉ "
          f"{load_yaml('domains')['medical_domains']})", "",
          md_table(["subset", "n"], [
              ["non-medical, all variants", bold_if_small(len(nonmed))],
              ["non-medical, DISTORTED excluded", bold_if_small(len(nonmed_nd))],
              ["non-medical, plain only (DISTORTED and NOISE excluded)", bold_if_small(len(nonmed_plain))],
          ]), "",
          md_table(["domain_goswami", "n", "is_medical"],
                   [[d, n, m] for (d, m), n in sorted(Counter((r["domain_goswami"], r["is_medical"]) for r in rows).items())]),
          ""]

    # 4. anomaly length distribution by domain
    by_dom = defaultdict(list)
    for r in rows:
        by_dom[r["domain_goswami"]].append(r["anom_len"])
    L += ["## 4. Anomaly length (inclusive count end − begin + 1) by domain_goswami", "",
          md_table(["domain", "n", "min", "median", "max"],
                   [[d, len(v), min(v), int(np.median(v)), max(v)] for d, v in sorted(by_dom.items())]
                   + [["**all**", len(rows), min(r["anom_len"] for r in rows),
                       int(np.median([r["anom_len"] for r in rows])), max(r["anom_len"] for r in rows)]]),
          ""]

    # 5. GT disagreements
    diff_deck = [r for r in rows if "DECK_FIELDS_DIFFER" in r["_flags"]]
    L += ["## 5. Ground-truth disagreements", "",
          "TSB-AD Label column: NA — A4 not run.", "",
          f"Filename fields vs the fields printed in UCR_AnomalyDataSets.pptx for the same series "
          f"({len(diff_deck)} series). The archive filename is the ground truth (BRIEF D2); the deck was "
          f"written before the label fixes it itself describes on slide 116.", ""]
    L += ["Matching is by series name, so twins that share a name (e.g. the two `ECG4` files) are compared "
          "against every deck filename carrying that name; a row here means none of them equals the file.", ""]
    if diff_deck:
        body = []
        for r in diff_deck:
            prov = match_series(deck, r["name"])
            printed = "; ".join(f"{f['train_end']}_{f['begin']}_{f['end']}" for f in prov["deck_fields"])
            body.append([r["num"], r["name"], f"{r['train_end_raw']}_{r['begin_raw']}_{r['end_raw']}", printed, r["deck_slides"]])
        L += [md_table(["num", "name", "filename fields", "deck fields", "slides"], body), ""]
    notdeck = [r for r in rows if "NOT_IN_DECK" in r["_flags"]]
    L += [f"Series not described anywhere in the deck: {len(notdeck)}"
          + (" → " + ", ".join(r["filename"] for r in notdeck) if notdeck else ""), ""]

    # 6. index convention
    L += ["## 6. Index convention (BRIEF A2 / decision D3)", "",
          f"Active value in `configs/conventions.yaml`: **{conv.value if conv else 'null (undetermined)'}**"
          + (f" — {conv_err}" if conv_err else ""), "",
          "Documentary evidence (archive's own `UCR_AnomalyDataSets.pptx`, text in `docs/ucr_supplement_text.md`):", "",
          "- slide 4: `L = end – begin + 1` and the scoring rule "
          "`min(begin-L, begin-100) < P < max(end+L, end+100)` → `end` is inclusive",
          "- slide 5: \"From 1 to X is training data\", \"Files are '-ascii' format\" → 1-based (MATLAB) counting",
          "- slides 52, 53, 90–99: injection code `T(start_anomaly:end_anomaly) = …`, "
          "`randn(end_anomaly-start_anomaly+1, 1)` → the filename fields are MATLAB 1-based inclusive indices",
          "- slide 6: `a(200:280)` \"the anomaly starts at 200 … output 240\"", "",
          "Empirical evidence — series whose anomaly is 1–2 points long. For each series the mean local "
          "deviation of the points claimed by H1 (MATLAB 1-based inclusive → 0-based `[begin−1, end)`) is "
          "compared with the points claimed by H0 (0-based inclusive → `[begin, end+1)`). "
          "ratio ≥ 2 favours H1, ≤ 0.5 favours H0, otherwise not decisive (time-warp anomalies have no "
          "outlying point).", ""]
    if evidence:
        verdicts = Counter(e["verdict"] for e in evidence)
        L += [md_table(["verdict", "series"], [[k, v] for k, v in sorted(verdicts.items())]), "",
              md_table(["num", "file", "L", "H1 idx0", "H0 idx0", "dev H1", "dev H0", "ratio", "verdict"],
                       [[e["num"], e["filename"], e["anom_len_incl"], e["H1_idx0"], e["H0_idx0"],
                         f"{e['dev_H1']:.3g}", f"{e['dev_H0']:.3g}", f"{e['ratio_H1_over_H0']:.2f}",
                         e["verdict"]] for e in evidence]),
              "", f"Figures: `reports/figures/index_convention/` ({len(evidence)} series); "
              f"numbers: `reports/index_convention_evidence.csv`.", ""]
        # Twins = same recording + same fields under a different DISTORTED/NOISE
        # prefix (slide 100). A verdict that its twins do not share is a draw
        # of the added noise, not evidence about the convention.
        twin_key = {r["num"]: (r["name"].replace("DISTORTED", "").replace("NOISE", ""),
                               r["train_end_raw"], r["begin_raw"], r["end_raw"]) for r in rows}
        by_twin = defaultdict(list)
        for e in evidence:
            by_twin[twin_key[e["num"]]].append(e)
        h0 = [e for e in evidence if e["verdict"] == "H0 (0-based)"]
        if h0:
            L += ["Series favouring the 0-based reading, with the verdicts of their twins "
                  "(same recording and fields, other variant prefix):", ""]
            for e in h0:
                twins = [t for t in by_twin[twin_key[e["num"]]] if t["num"] != e["num"]]
                L.append(f"- {e['filename']} (ratio {e['ratio_H1_over_H0']:.2f}) — twins: "
                         + ("; ".join(f"{t['num']:03d} {t['verdict']} (ratio {t['ratio_H1_over_H0']:.2f})"
                                      for t in twins) or "none"))
            L.append("")
    L += ["**D3 status:** default set from the documentary evidence; researcher sign-off pending "
          "(BRIEF §8). Override `index_convention` in `configs/conventions.yaml` and re-run this script.", ""]

    # A5 provenance summary
    L += ["## A5. Domain mapping provenance", "",
          f"- `domain_goswami`: vendored from `{goswami_prov['repo']}` @ `{goswami_prov['commit']}`, "
          f"`{goswami_prov['file']}` lines {goswami_prov['lines']} (`{goswami_prov['symbol']}`, "
          f"{goswami_prov['n_entities']} entities). This is the authors' own mapping, not a reconstruction.",
          f"- `domain_official`: category read from the source statements in `UCR_AnomalyDataSets.pptx` "
          f"via the rules in `configs/domains.yaml` (`official_source_rules`). The verbatim statement is in "
          f"`source_official`; the category is our reading of it.",
          f"- agreement: official == Goswami for {sum(1 for r in rows if r['domain_agree'] is True)} series; "
          f"differs for {sum(1 for r in rows if r['domain_agree'] is False)}; official NA for "
          f"{sum(1 for r in rows if r['domain_official'] == NA)}. Details: `reports/domain_unmatched.csv`.", ""]
    if unmatched:
        L += [md_table(["num", "name", "official", "Goswami", "basis", "slides"],
                       [[r["num"], r["name"], r["domain_official"], r["domain_goswami"],
                         r["domain_official_basis"], r["deck_slides"]] for r in unmatched]), ""]

    # licence
    L += ["## Licence", "",
          "See `docs/DATA_LICENSE.md`. No licence text was found in any supplementary document "
          "(keyword pass over all slides and .m files; category `licence` in `docs/ucr_supplement_text.md`); "
          "the only usage statement is the citation request in `Start_Here_This_is_a_Read_me.pptx`.", ""]

    (reports_dir / "data_audit.md").write_text("\n".join(L))
    print(f"wrote {reports_dir / 'data_audit.md'}")

    print(f"\nflags: {dict(flag_counter)}")
    if sev_counter["STOP"]:
        print(f"STOP: {len(stop_rows)} series with STOP-severity findings (BRIEF §7-5)")
        return 2
    if sev_counter["WARN"]:
        print(f"WARN: {len(warn_rows)} series with WARN-severity findings")
        return 1
    print("audit OK: no STOP/WARN findings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
