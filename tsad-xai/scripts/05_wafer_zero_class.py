"""05_wafer_zero_class.py -- BRIEF2 §D3 (the part due this week).

1. Reproduce the test accuracy of one trained Wafer model (default ResNet, the first
   seed) and compare it with `best_acc_test` in that model's `training_conf.json`.
   A mismatch beyond --tol is a STOP (BRIEF2 D3).
2. Run the upstream `identify_zero_class` procedure on all 25 Wafer models
   (5 architectures x 5 seeds) and write `reports/wafer_zero_class.csv`.

WHY THIS IS NOT `identify_zero_class.py`
    The upstream script imports `configs.dbconfig` and `pymongo` at module level
    (identify_zero_class.py:11-12) and pulls in captum through
    `interpretability_methods`. MongoDB is deliberately not installed (BRIEF2 D1-3).
    The procedure itself needs none of that, so it is re-expressed here from the
    upstream source, function by function, with the line numbers noted. The upstream
    data loader, network definitions and perturber are imported unchanged from the
    pinned clone -- nothing about the model or the data is reimplemented.

    Upstream procedure (identify_zero_class.py:31-163), reproduced exactly:
      * perturbation = SubSequencePerturber.OutOfDistHigh, window_size = 7  (:40-41)
      * at most 5 valid samples per class                                   (:43)
      * a sample is valid only if the model predicts the right class with
        probability >= 0.95                                                 (:109-118)
      * perturb a random window [start, start+7] (end+1 exclusive)          (:126-129)
      * count samples whose probability for the TRUE label rises to >= 0.99 (:140-141)
      * zero class = argmax of those counts; None if every count is 0       (:159-163)

Usage:
    python scripts/05_wafer_zero_class.py [--arch ResNet] [--tol 0.005] [--seed 0]
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import REPO_ROOT, load_yaml, resolve  # noqa: E402
from src.runlog import env_header  # noqa: E402

CLONE = REPO_ROOT / "third_party" / "simic_cmi"
DATASET = "Wafer"
ARCHES = ("Inception", "LSTM", "MLP", "ResNet", "ViT")
PERTURB_WINDOW = 7          # identify_zero_class.py:41
MAX_VALID_SAMPLES = 5       # :43
MIN_CONFIDENCE = 0.95       # :110
RISE_THRESHOLD = 0.99       # :140


@contextlib.contextmanager
def upstream_cwd():
    """The upstream loader uses the relative path 'data/UCR_datasets/Univariate'."""
    if not (CLONE / "utils" / "utils.py").is_file():
        raise FileNotFoundError(f"pinned clone missing: run scripts/00_fetch_simic.py ({CLONE})")
    old = os.getcwd()
    os.chdir(CLONE)
    sys.path.insert(0, str(CLONE))
    try:
        yield
    finally:
        sys.path.remove(str(CLONE))
        os.chdir(old)


def model_dirs() -> list[tuple[str, str, Path]]:
    out = []
    for arch in ARCHES:
        root = CLONE / "models" / DATASET / arch
        if not root.is_dir():
            raise FileNotFoundError(root)
        for d in sorted(root.iterdir(), key=lambda p: int(p.name.split("_")[1])):
            out.append((arch, d.name.split("_")[1], d))
    return out


def load_model(arch: str, model_dir: Path, input_length: int, n_channels: int, n_outputs: int):
    import torch
    from utils.networks import network_dict
    model = network_dict[arch](input_length, n_channels, n_outputs)
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
    model.eval()
    return model


def load_testset(conf: dict):
    """Exactly what identify_zero_class.py:43-59 does, including the label shift."""
    from utils.utils import load_univariate_UCR_dataset
    norm = conf.get("data_normalization_method")
    params = conf.get("normalization_parameters")
    test_x, test_y, _ = load_univariate_UCR_dataset(DATASET, is_testset=True,
                                                    normalization=norm, norm_params=params)
    test_y = np.asarray(test_y)
    uniq = np.unique(test_y)
    if uniq.min() > 0:                    # :56-58
        test_y = test_y - uniq.min()
    return test_x, test_y, np.unique(test_y)


def test_accuracy(model, test_x, test_y) -> tuple[float, int]:
    import torch
    with torch.no_grad():
        logits = model(torch.tensor(np.asarray(test_x), dtype=torch.float32))
        pred = torch.exp(logits).argmax(dim=1).numpy()
    return float((pred == test_y).mean()), int(len(test_y))


def zero_class(model, test_x, test_y, unique_labels, rng: random.Random) -> dict:
    """identify_zero_class.py:75-163, without the MongoDB/captum imports."""
    import torch
    from utils.subsequence_perturbation import SubSequencePerturber as ssp
    input_length = test_x.shape[-1]
    counts, examined = {}, {}
    for label in unique_labels:
        class_samples = test_x[test_y == label]
        counts[int(label)] = 0
        valid = 0
        for sample in class_samples:
            if valid >= MAX_VALID_SAMPLES:
                break
            target = np.array(sample, copy=True)
            with torch.no_grad():
                out = torch.exp(model(torch.tensor(target, dtype=torch.float32).unsqueeze(0))).squeeze()
            pred = int(out.argmax().item())
            if float(out[pred].item()) < MIN_CONFIDENCE or pred != int(label):
                continue                                   # :109-118
            valid += 1
            perturber = ssp.OutOfDistHigh(target[0])
            start = rng.randint(0, input_length - PERTURB_WINDOW)
            perturbed = perturber.perturb_subsequence(start, start + PERTURB_WINDOW + 1)  # :129
            with torch.no_grad():
                new_out = torch.exp(model(torch.tensor(
                    np.expand_dims(perturbed, axis=0), dtype=torch.float32).unsqueeze(0))).squeeze()
            if float(new_out[int(label)].item()) >= RISE_THRESHOLD:      # :140-141
                counts[int(label)] += 1
        examined[int(label)] = valid
    res = np.array([counts[k] for k in sorted(counts)])
    zc = None if (res == 0).all() else int(res.argmax())                  # :159-163
    return {"counts": counts, "examined": examined, "zero_class": zc, "res": res.tolist()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="ResNet", help="architecture for the accuracy reproduction")
    ap.add_argument("--tol", type=float, default=0.005, help="absolute accuracy tolerance (STOP above)")
    ap.add_argument("--seed", type=int, default=0, help="seed for the random perturbation window")
    args = ap.parse_args()
    print(env_header())

    reports = resolve(load_yaml("paths")["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)
    rows, stop_reasons = [], []

    with upstream_cwd():
        import torch
        torch.manual_seed(args.seed)
        cache: dict[str, tuple] = {}
        for arch, seed, mdir in model_dirs():
            conf = json.loads((mdir / "training_conf.json").read_text())
            key = f"{conf.get('data_normalization_method')}|{json.dumps(conf.get('normalization_parameters'), sort_keys=True)}"
            if key not in cache:
                cache[key] = load_testset(conf)
            test_x, test_y, uniq = cache[key]
            model = load_model(arch, mdir, test_x.shape[-1], test_x.shape[1], len(uniq))
            acc, n = test_accuracy(model, test_x, test_y)
            claimed = conf.get("best_acc_test")
            delta = acc - claimed if claimed is not None else float("nan")
            zc = zero_class(model, test_x, test_y, uniq, random.Random(args.seed))
            rows.append({
                "dataset": DATASET, "arch": arch, "seed": seed,
                "n_test": n, "acc_measured": round(acc, 6),
                "acc_in_training_conf": claimed, "acc_delta": round(delta, 6),
                "zero_class": "None" if zc["zero_class"] is None else zc["zero_class"],
                "counts_per_class": json.dumps(zc["counts"]),
                "valid_samples_per_class": json.dumps(zc["examined"]),
                "n_classes": len(uniq), "model_dir": str(mdir.relative_to(REPO_ROOT)),
            })
            flag = ""
            if claimed is not None and abs(delta) > args.tol:
                flag = "  ACCURACY MISMATCH"
                if arch == args.arch:
                    stop_reasons.append(f"{arch}/seed_{seed}: measured {acc:.6f} vs "
                                        f"training_conf {claimed:.6f} (|Δ| = {abs(delta):.6f} > {args.tol})")
            print(f"  {arch:<10} seed_{seed:<3} acc={acc:.6f} (conf {claimed:.6f}, Δ={delta:+.6f}) "
                  f"zero_class={rows[-1]['zero_class']}{flag}", flush=True)

    out = reports / "wafer_zero_class.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} models)")
    deltas = [abs(r["acc_delta"]) for r in rows if r["acc_in_training_conf"] is not None]
    print(f"accuracy reproduction: max |Δ| = {max(deltas):.6f} over {len(deltas)} models "
          f"(tolerance {args.tol})")
    if stop_reasons:
        print("\nSTOP (BRIEF2 D3): the reference model's accuracy did not reproduce")
        for r in stop_reasons:
            print(f"  - {r}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
