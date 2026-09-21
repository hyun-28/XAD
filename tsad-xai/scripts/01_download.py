"""01_download.py -- BRIEF Task A, step A1.

Fetch the official UCR Time Series Anomaly Archive 2021 zip, extract it into a
separate read-only tree, hash every file, and compare it byte-for-byte against
the researcher's pre-existing local copy (both the local zip and the local
extracted tree).

Exit code != 0 (BRIEF §7 STOP 1) when:
  * the official archive does not contain exactly `expected_n_series` .txt
    series files, or
  * any series (.txt) file differs between local copy and official copy.
Differences confined to supplementary documents (pptx/pdf/m/mat) are reported
but do not stop the run (agreed 2026-09-21, DECISIONS.md D-A1-1).

Outputs
  data/raw/ucr/official/<zip>            downloaded zip (git-ignored)
  data/raw/ucr/official/extracted/       extraction, chmod a-w (git-ignored)
  data/raw/ucr/CHECKSUMS.sha256          zip + per-file SHA256 (committed)
  data/raw/ucr/audit_a1.json             machine-readable comparison, read by
                                         02_audit.py for the report header

Usage:
  python scripts/01_download.py [--force-download]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import REPO_ROOT, load_yaml, resolve  # noqa: E402
from src.runlog import env_header  # noqa: E402

# Files that are OS/editor litter, not archive content. They are listed in the
# JSON as `ignored` so nothing disappears silently.
LITTER_PREFIXES = ("~$",)
LITTER_NAMES = {".DS_Store", "Thumbs.db"}


def sha256_file(p: Path, bufsize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def is_litter(rel: Path) -> bool:
    return rel.name in LITTER_NAMES or rel.name.startswith(LITTER_PREFIXES)


def hash_tree(root: Path) -> tuple[dict[str, str], list[str]]:
    """{relative_posix_path: sha256} for every regular file under root."""
    hashes, ignored = {}, []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if is_litter(rel):
            ignored.append(rel.as_posix())
            continue
        hashes[rel.as_posix()] = sha256_file(p)
    return hashes, ignored


def find_anchor(root: Path, fulldata_subdir: Path) -> Path:
    """The unique directory D under root such that D/fulldata_subdir exists."""
    hits = [d for d in [root, *root.rglob("*")]
            if d.is_dir() and (d / fulldata_subdir).is_dir()]
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {fulldata_subdir} under {root}, found {hits}")
    return hits[0]


def inner_zip_vs_tree(inner_zip: Path, anchor: Path) -> dict:
    """Compare every member of inner_zip with the same path under anchor."""
    if not inner_zip.is_file():
        return {"present": False}
    same, diff = 0, []
    with zipfile.ZipFile(inner_zip) as z:
        for m in z.infolist():
            if m.is_dir():
                continue
            ext = anchor / m.filename
            if not ext.is_file():
                diff.append({"member": m.filename, "why": "missing in extracted tree"})
                continue
            if hashlib.sha256(z.read(m)).hexdigest() == sha256_file(ext):
                same += 1
            else:
                diff.append({"member": m.filename, "inner_bytes": m.file_size,
                             "extracted_bytes": ext.stat().st_size,
                             "inner_date": "%04d-%02d-%02d" % m.date_time[:3]})
    return {"present": True, "inner_zip": str(inner_zip), "n_same": same, "differences": diff}


def download(url: str, dest: Path) -> None:
    print(f"downloading {url}\n        -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as r, tmp.open("wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done/1e6:8.1f} / {total/1e6:.1f} MB", end="", flush=True)
        print()
        if total and done != total:
            raise RuntimeError(f"short download: {done} of {total} bytes")
    tmp.replace(dest)


def make_read_only(root: Path) -> None:
    """BRIEF §1.3: downloaded originals are read-only."""
    for p in root.rglob("*"):
        if p.is_file():
            p.chmod(p.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-download", action="store_true",
                    help="re-download even if the zip is already present")
    args = ap.parse_args()
    print(env_header())

    cfg = load_yaml("paths")["ucr"]
    url = cfg["official_url"]
    official_dir = resolve(cfg["official_dir"])
    zip_path = official_dir / Path(url).name
    extract_root = official_dir / "extracted"
    local_zip = resolve(cfg["local_zip"])
    local_extracted = resolve(cfg["local_extracted"])
    fulldata_subdir = Path(cfg["fulldata_subdir"])
    n_expected = int(cfg["expected_n_series"])

    for p in (local_zip, local_extracted):
        if not p.exists():
            raise FileNotFoundError(f"local copy not found (configs/paths.yaml): {p}")

    # ---- 1. official zip -------------------------------------------------
    if args.force_download or not zip_path.is_file():
        if zip_path.is_file():
            zip_path.chmod(zip_path.stat().st_mode | stat.S_IWUSR)
            zip_path.unlink()
        download(url, zip_path)
    else:
        print(f"official zip already present, not re-downloading: {zip_path}")
    official_zip_sha = sha256_file(zip_path)
    official_zip_size = zip_path.stat().st_size

    # ---- 2. extract (fresh) ----------------------------------------------
    if extract_root.exists():
        # previous extraction was made read-only; restore write bit to replace
        for p in extract_root.rglob("*"):
            p.chmod(p.stat().st_mode | stat.S_IWUSR)
        import shutil
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as z:
        bad = z.testzip()
        if bad is not None:
            raise RuntimeError(f"corrupt member in official zip: {bad}")
        z.extractall(extract_root)
    make_read_only(extract_root)
    print(f"extracted to {extract_root}")

    # ---- 2b. anchor both trees on the folder that CONTAINS the archive ------
    # MEASURED 2026-09-21 (unzip -l on the official zip): the official zip is
    # not the archive itself but a zip of the researcher's-style folder
    # 'AnomalyDatasets_2021/', which holds BOTH an inner
    # 'UCR_TimeSeriesAnomalyDatasets2021.zip' (92,026,813 B) AND its extracted
    # tree. The local copy is that same folder. So the comparable unit is
    # "the directory containing <fulldata_subdir>'s top folder", on both sides;
    # that also brings the inner zip vs local zip into the per-file table.
    official_anchor = find_anchor(extract_root, fulldata_subdir)
    local_anchor = local_extracted.parent
    if not (local_anchor / fulldata_subdir).is_dir():
        raise FileNotFoundError(f"local tree lacks {fulldata_subdir}: {local_anchor}")
    print(f"official anchor: {official_anchor}\nlocal anchor   : {local_anchor}")
    official_files, official_ignored = hash_tree(official_anchor)
    local_files, local_ignored = hash_tree(local_anchor)

    # ---- 2c. inner zip vs extracted tree (both inside the official zip) ----
    # The official zip ships the archive twice (inner zip + extracted tree).
    # Record whether the two agree so the report can say which copy is used.
    inner_zip_report = inner_zip_vs_tree(official_anchor / Path(url).name, official_anchor)

    # ---- 3. series count (STOP 1) ------------------------------------------
    fulldata_prefix = fulldata_subdir.as_posix() + "/"
    official_series = sorted(k for k in official_files
                             if k.startswith(fulldata_prefix) and k.endswith(".txt"))
    local_series = sorted(k for k in local_files
                          if k.startswith(fulldata_prefix) and k.endswith(".txt"))

    # ---- 4. compare ---------------------------------------------------------
    keys = sorted(set(official_files) | set(local_files))
    match, mismatch, local_only, official_only = [], [], [], []
    for k in keys:
        o, l = official_files.get(k), local_files.get(k)
        if o and l:
            (match if o == l else mismatch).append(k)
        elif o:
            official_only.append(k)
        else:
            local_only.append(k)

    def is_series(k: str) -> bool:
        return k.startswith(fulldata_prefix) and k.endswith(".txt")

    series_mismatch = [k for k in mismatch if is_series(k)]
    series_local_only = [k for k in local_only if is_series(k)]
    series_official_only = [k for k in official_only if is_series(k)]

    # ---- 5. CHECKSUMS.sha256 -------------------------------------------------
    checksums = resolve(cfg["checksums"])
    checksums.parent.mkdir(parents=True, exist_ok=True)
    with checksums.open("w") as f:
        f.write(f"# SHA256 of the official archive fetched {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"# source: {url}\n")
        f.write(f"# generated by scripts/01_download.py -- do not hand-edit\n")
        f.write(f"{official_zip_sha}  {zip_path.name}\n")
        for k in sorted(official_files):
            f.write(f"{official_files[k]}  {(official_anchor / k).relative_to(official_dir).as_posix()}\n")
    print(f"wrote {checksums} ({1 + len(official_files)} entries)")

    # ---- 6. machine-readable result ---------------------------------------
    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "env": env_header(),
        "official": {
            "url": url, "zip_path": str(zip_path), "zip_sha256": official_zip_sha,
            "zip_bytes": official_zip_size, "n_files": len(official_files),
            "n_series_txt": len(official_series), "ignored": official_ignored,
            "supplementary": sorted(k for k in official_files if not is_series(k)),
        },
        "local": {
            "zip_path": str(local_zip), "zip_sha256": sha256_file(local_zip),
            "zip_bytes": local_zip.stat().st_size,
            "zip_mtime": datetime.fromtimestamp(local_zip.stat().st_mtime, timezone.utc).isoformat(),
            "extracted_path": str(local_extracted), "n_files": len(local_files),
            "n_series_txt": len(local_series), "ignored": local_ignored,
        },
        "expected_n_series": n_expected,
        "anchors": {"official": str(official_anchor), "local": str(local_anchor)},
        "inner_zip_vs_extracted": inner_zip_report,
        "compare": {
            "match": match, "mismatch": mismatch,
            "local_only": local_only, "official_only": official_only,
            "series_mismatch": series_mismatch,
            "series_local_only": series_local_only,
            "series_official_only": series_official_only,
        },
    }
    stop_reasons = []
    if len(official_series) != n_expected:
        stop_reasons.append(f"official archive has {len(official_series)} series .txt files, expected {n_expected}")
    if series_mismatch or series_local_only or series_official_only:
        stop_reasons.append(
            f"series files differ: {len(series_mismatch)} mismatched, "
            f"{len(series_local_only)} local-only, {len(series_official_only)} official-only")
    result["stop"] = bool(stop_reasons)
    result["stop_reasons"] = stop_reasons
    result["supplementary_differences"] = bool(
        [k for k in mismatch + local_only + official_only if not is_series(k)])

    out = resolve(cfg["a1_result"])
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}")

    # ---- 7. summary -----------------------------------------------------------
    print("\n=== A1 summary ===")
    print(f"official zip : {official_zip_size:>12,} B  sha256={official_zip_sha}")
    print(f"local zip    : {result['local']['zip_bytes']:>12,} B  sha256={result['local']['zip_sha256']}")
    print(f"series .txt  : official={len(official_series)}  local={len(local_series)}  expected={n_expected}")
    print(f"all files    : match={len(match)}  mismatch={len(mismatch)}  "
          f"local_only={len(local_only)}  official_only={len(official_only)}")
    print(f"series only  : mismatch={len(series_mismatch)}  local_only={len(series_local_only)}  "
          f"official_only={len(series_official_only)}")
    if inner_zip_report.get("present"):
        print(f"inner zip vs extracted tree (both official): same={inner_zip_report['n_same']}  "
              f"different={len(inner_zip_report['differences'])}")
        for d in inner_zip_report["differences"]:
            print(f"  INNER-ZIP-DIFF {d}")
    for k in mismatch + local_only + official_only:
        tag = "MISMATCH" if k in mismatch else ("LOCAL-ONLY" if k in local_only else "OFFICIAL-ONLY")
        print(f"  {tag:13s} {k}")
    if stop_reasons:
        print("\nSTOP (BRIEF §7-1):")
        for r in stop_reasons:
            print(f"  - {r}")
        return 1
    print("\nA1 OK: series files identical; official copy is the reference from here on.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
