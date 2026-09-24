"""w2_sample.py -- W2 rev1 §5: draw the experiment sample from metadata only; write configs/w2_sample.yaml.

Rule (configs/w2.yaml `sample`, DECISIONS D-S-1):
  1. unit = content_group (89; D-F2-2). Group domain = domain_goswami of its members (each group
     has one domain; checked). Missing domain -> "other".
  2. one series per group: seeded choice among the group's members (sorted by num), for all 89.
  3. 60-group sample: proportional allocation over domains, at least 1 group per domain:
     floor(60 * share) raised to 1 where 0, then the remaining seats by largest fractional part among
     domains not raised by the minimum rule (ties -> larger domain, then name); then a seeded choice of
     groups within each domain. The 89-group sample is every group with the same series (nested).
  4. metadata per selected series: n, train_end, w, |GT|, detected (gate-1 P1, buffer = w),
     n_const_train (gate-1), scale(x) (D8, from the stored gate-1 score), admissible positions per
     length ({0.25, 0.5, 1.0} w and L = |GT|): number of admissible starts and the maximum number of
     pairwise-disjoint regions. Gate-1 detection failures are included (flag `detected`).

Uses only the manifest, gate-1 detection.csv and gate-1 stored unperturbed scores (for scale):
no perturbation is computed. The output is committed before any E2 result exists (rev1 §5).
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import w2  # noqa: E402
from src.config import REPO_ROOT, load_yaml  # noqa: E402
from src.runlog import env_header  # noqa: E402

OUT = REPO_ROOT / "configs" / "w2_sample.yaml"


def allocate(counts: dict[str, int], total: int) -> dict[str, int]:
    N = sum(counts.values())
    if total > N:
        raise ValueError(f"cannot draw {total} of {N} groups")
    quota = {d: total * c / N for d, c in counts.items()}
    seats = {d: int(np.floor(q)) for d, q in quota.items()}
    raised = {d for d, s in seats.items() if s == 0}
    for d in raised:
        seats[d] = 1
    left = total - sum(seats.values())
    if left < 0:
        raise ValueError("minimum-1 rule exceeds the total")
    order = sorted((d for d in counts if d not in raised and seats[d] < counts[d]),
                   key=lambda d: (-(quota[d] - np.floor(quota[d])), -counts[d], d))
    for d in order[:left]:
        seats[d] += 1
    if sum(seats.values()) != total or any(seats[d] > counts[d] for d in counts):
        raise RuntimeError(f"allocation failed: {seats}")
    return seats


def main() -> int:
    header = env_header()
    print(header)
    cfg = load_yaml("w2")["sample"]
    man, g1 = w2.manifest(), w2.gate1_primary()
    rng = np.random.default_rng(int(cfg["seed"]))
    groups = {}
    for cg, sub in man.groupby("content_group"):
        doms = set(sub.domain_goswami.fillna("other").astype(str))
        if len(doms) != 1:
            raise RuntimeError(f"{cg}: members span domains {doms}")
        groups[cg] = {"domain": doms.pop(), "members": sorted(int(v) for v in sub.num)}
    if len(groups) != 89:
        raise RuntimeError(f"{len(groups)} content groups, expected 89")
    # step 2: one series per group, all 89 groups (sorted group ids -> fixed rng consumption order)
    pick = {cg: int(rng.choice(groups[cg]["members"])) for cg in sorted(groups)}
    # step 3: 60 of 89, stratified by domain
    counts = {}
    for cg in sorted(groups):
        counts[groups[cg]["domain"]] = counts.get(groups[cg]["domain"], 0) + 1
    n_def = int(cfg["n_groups_default"])
    seats = allocate(counts, n_def)
    chosen60 = []
    for d in sorted(counts):
        pool = sorted(cg for cg in groups if groups[cg]["domain"] == d)
        chosen60 += sorted(str(v) for v in rng.choice(pool, size=seats[d], replace=False))
    chosen60 = sorted(chosen60)
    # step 4: metadata
    entries = []
    for cg in sorted(groups):
        num = pick[cg]
        lay = w2.layout(num, man, g1)
        sc = w2.scale_of(num, lay)
        L_gt = lay.stop0 - lay.start0
        pos = {}
        for f in cfg["lengths_w"]:
            L = w2.length_for(float(f), lay.w)
            pos[f"{f}w"] = {"L": L, "n_admissible": lay.n_admissible(L), "max_disjoint": lay.max_disjoint(L)}
        pos["GT"] = {"L": L_gt, "n_admissible": lay.n_admissible(L_gt), "max_disjoint": lay.max_disjoint(L_gt)}
        g = g1.loc[num]
        entries.append({
            "content_group": cg, "num": num, "name": str(man.loc[num, "name"]), "domain": groups[cg]["domain"],
            "group_size": len(groups[cg]["members"]), "in_sample_60": cg in chosen60,
            "n": lay.n, "train_end": lay.train_end, "gt": [lay.start0, lay.stop0], "gt_len": L_gt, "w": lay.w,
            "detected": bool(g.p1_bw), "n_const_train": int(g.n_const_train),
            "scale": round(sc["scale"], 12), "scale_ok": sc["scale_ok"], "scale_reason": sc["scale_reason"],
            "positions": pos,
        })
    doc = {
        "generated_by": "scripts/w2_sample.py", "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "env_header": header, "seed": int(cfg["seed"]), "rule": "DECISIONS D-S-1 (script docstring)",
        "domain_groups_89": dict(sorted(counts.items())), "domain_groups_60": dict(sorted(seats.items())),
        "sample_60": chosen60, "series": entries,
    }
    OUT.write_text("# GENERATED by scripts/w2_sample.py — do not edit (rev1 §5: no post-hoc changes).\n"
                   + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=120))
    bad60 = [e for e in entries if e["in_sample_60"] and not e["scale_ok"]]
    bad89 = [e for e in entries if not e["scale_ok"]]
    print(f"wrote {OUT}: 89 groups, 60-sample {len(chosen60)}; seats {seats}; "
          f"scale excluded: {len(bad60)}/60, {len(bad89)}/89")
    if len(bad60) > 0.10 * 60 or len(bad89) > 0.10 * 89:
        print("STOP (rev1 §11): scale(x) exclusions exceed 10% of the sample")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
