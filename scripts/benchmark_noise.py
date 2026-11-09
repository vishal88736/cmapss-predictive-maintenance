"""Measure a frozen pooled bundle under broadband sensor noise and dropouts.

Read-only benchmark: no fitting, no recalibration, no threshold or model
selection. The frozen bundle hash is recorded before evaluation and re-checked
afterwards, so a measurement can never silently become a repaired artifact.

Partitions:

* ``official`` -- the four C-MAPSS test files at their final endpoints. These
  benchmarks were inspected during earlier development, so the numbers are
  exploratory, not an untouched holdout.
* ``heldout`` -- the internal 10% engine test partition rebuilt from the frozen
  bundle's own split and verified against its saved partitions.

Noise levels are multiples of the fitting-only within-regime sensor scale. A
level is nested: the same per-engine innovations are scaled, so a sweep is
paired rather than resampled.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from grace_rul.artifacts import load_bundle
from grace_rul.calibration import interval_bounds
from grace_rul.config import SUBSETS
from grace_rul.data import WindowDataset, load_subset
from grace_rul.noise import (DEFAULT_DROPOUTS, DEFAULT_LEVELS, NOISE_PROTOCOL,
                             apply_measurement_noise, validate_noise)
from grace_rul.pooled import load_pooled_training, split_pooled_engines
from grace_rul.training import device_for, evaluate, metrics

DEFAULT_BUNDLE = ROOT / "artifacts/pooled_repair_20261101T230338Z_9ea6_dpk/final_model_verified/grace_rul_ALL.pt"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def official_frame(root):
    frames, truths = [], {}
    for index, subset in enumerate(SUBSETS, 5):
        _, test, truth = load_subset(root / "data", subset)
        test["source_unit"], test["source_subset"] = test.unit, subset
        test["unit"] = index * 1_000_000 + test.unit
        truths.update({index * 1_000_000 + int(unit): float(value) for unit, value in truth.items()})
        frames.append(test)
    return pd.concat(frames, ignore_index=True), truths


def heldout_frame(root, config, entry):
    frame, _ = load_pooled_training(root / "data", config)
    groups = split_pooled_engines(frame, config["split_seed"])
    if groups != entry["partitions"]:
        raise ValueError("Rebuilt partitions disagree with the frozen bundle.")
    return frame[frame.unit.isin(groups["test"])].copy(), None


def levels(value):
    values = [float(item) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Provide at least one level.")
    for level in values:
        validate_noise(level, 0.)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--partition", choices=["official", "heldout", "both"], default="both")
    parser.add_argument("--levels", type=levels, default=list(DEFAULT_LEVELS))
    parser.add_argument("--dropouts", type=levels, default=list(DEFAULT_DROPOUTS))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("batch_size must be positive.")
    for drop in args.dropouts:
        validate_noise(0., drop)
    root, bundle_path = Path(args.project_root).resolve(), Path(args.bundle).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(args.output_dir or root / "artifacts" / f"noise_benchmark_{stamp}").resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Noise benchmark output directory must be empty.")
    output.mkdir(parents=True, exist_ok=True)

    device = device_for({"device": args.device})
    frozen_hash = sha(bundle_path)
    bundle, loaded = load_bundle(bundle_path, device=str(device))
    item = loaded["POOLED"]
    entry, model, scaler = item["entry"], item["model"], item["scaler"]
    config = {**entry["training_config"], "device": str(device), "batch_size": args.batch_size}
    protocol = f"capped_{config['train_rul_cap']:g}"

    partitions = ["official", "heldout"] if args.partition == "both" else [args.partition]
    frames = {}
    if "official" in partitions:
        frames["official"] = official_frame(root)
    if "heldout" in partitions:
        frames["heldout"] = heldout_frame(root, config, entry)

    rows = []
    for partition in partitions:
        frame, truths = frames[partition]
        sources = frame[["unit", "source_subset"]].drop_duplicates()
        mode = "test" if truths is not None else "heldout"
        for level in args.levels:
            for drop in args.dropouts:
                noisy = apply_measurement_noise(frame, entry["scaler"], sigma_fraction=level,
                                                dropout_rate=drop, seed=args.seed)
                dataset = WindowDataset(noisy, scaler, config, mode=mode, family="clean",
                                        truths=truths, seed=args.seed)
                _, predictions = evaluate(model, dataset, config, fault_thresholds=entry["thresholds"])
                predictions = predictions.merge(sources, on="unit", how="left", validate="many_to_one")
                lower, upper = interval_bounds(predictions.prediction, entry["calibration"], protocol=protocol)
                predictions["covered"] = (predictions.target.to_numpy() >= lower) & (predictions.target.to_numpy() <= upper)
                groups = [("POOLED", predictions), *predictions.groupby("source_subset", sort=True)]
                for subset, group in groups:
                    rows.append({"partition": partition, "subset": subset, "protocol": protocol,
                                 "sigma_fraction": level, "dropout_rate": drop, "seed": args.seed,
                                 **metrics(group.prediction, group.target),
                                 "engine_balanced_RMSE": float(np.sqrt(((group.prediction - group.target) ** 2)
                                                                       .groupby(group.unit).mean().mean())),
                                 "interval_coverage": float(group.covered.mean()),
                                 "engines": int(group.unit.nunique()), "prefixes": int(len(group))})
                pooled = rows[-5] if len(groups) == 5 else rows[-1]
                print(f"{partition}: sigma={level:g} dropout={drop:g} -> pooled RMSE={pooled['RMSE']:.2f}", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(output / "noise_metrics.csv", index=False)
    if sha(bundle_path) != frozen_hash:
        raise RuntimeError("The frozen bundle changed during measurement.")
    manifest = {
        "status": "complete", "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": NOISE_PROTOCOL, "scope": "Read-only measurement; no fitting, recalibration or selection.",
        "bundle": str(bundle_path), "bundle_sha256": frozen_hash,
        "bundle_frozen_sha256": entry.get("source_bundle_sha256"),
        "partitions": partitions, "sigma_fraction_levels": args.levels,
        "dropout_rate_levels": args.dropouts, "noise_seed": args.seed,
        "noise_scale_source": "fitting-only within-regime sensor scale (sensor_fault_scale)",
        "settings_and_speeds_perturbed": False, "target_protocol": protocol,
        "device": str(device), "batch_size": args.batch_size,
        "official_benchmark_is_exploratory": True,
        "calibration_reused_not_refitted": True,
        "data_hashes": {name: sha(root / "data" / name) for name in
                        [f"{kind}_{subset}.txt" for subset in SUBSETS for kind in ("test", "RUL")]},
        "outputs": {"noise_metrics.csv": sha(output / "noise_metrics.csv")},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved noise benchmark: {output}", flush=True)


if __name__ == "__main__":
    main()
