"""Per-family recovery and detection benchmark of the frozen artifact.

Answers three questions the aggregate fault RMSE hides: how much of the damage
is detected, how accurately altered entries are reconstructed, and how far
healthy readings move. Operates on the manuscript's own evaluation seeds, so the
aggregate numbers reconcile with Table IX rather than resampling them.
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
from grace_rul.data import FAULTS, WindowDataset, load_subset
from grace_rul.training import device_for, evaluate, metrics

DEFAULT_BUNDLE = ROOT / "artifacts/pooled_repair_20261101T230338Z_9ea6_dpk/final_model_verified/grace_rul_ALL.pt"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--severity", type=float, default=.2)
    args = parser.parse_args(argv)
    root, bundle_path = ROOT, Path(args.bundle).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(args.output_dir or root / "artifacts" / f"recovery_benchmark_{stamp}").resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Recovery benchmark output directory must be empty.")
    output.mkdir(parents=True, exist_ok=True)

    frozen_hash = sha(bundle_path)
    bundle, loaded = load_bundle(bundle_path, device=str(device_for({"device": args.device})))
    item = loaded["POOLED"]
    entry, model, scaler = item["entry"], item["model"], item["scaler"]
    config = {**entry["training_config"], "device": args.device, "batch_size": args.batch_size}
    protocol = f"capped_{config['train_rul_cap']:g}"

    frames, truths = [], {}
    for index, subset in enumerate(SUBSETS, 5):
        _, test, truth = load_subset(root / "data", subset)
        test["source_unit"], test["source_subset"] = test.unit, subset
        test["unit"] = index * 1_000_000 + test.unit
        truths.update({index * 1_000_000 + int(unit): float(value) for unit, value in truth.items()})
        frames.append(test)
    frame = pd.concat(frames, ignore_index=True)

    rows = []
    for family in ["clean", *FAULTS, "mixed"]:
        for replicate in range(1 if family == "clean" else 3):
            dataset = WindowDataset(frame, scaler, config, mode="test", family=family, severity=args.severity,
                                    truths=truths, seed=config["validation_seed"] + 100000 + replicate)
            report, predictions = evaluate(model, dataset, config, fault_thresholds=entry["thresholds"])
            lower, upper = interval_bounds(predictions.prediction, entry["calibration"], protocol=protocol)
            rows.append({
                "family": family, "severity": 0. if family == "clean" else args.severity, "replicate": replicate,
                "detection_precision": report["fault_precision"], "detection_recall": report["fault_recall"],
                "detection_F1": report["fault_F1"], "false_repair_rate": report["fault_false_positive_rate"],
                "reconstruction_MAE": report["reconstruction_MAE"],
                "linear_reconstruction_RMSE": report["linear_reconstruction_RMSE"],
                "intact_displacement_RMSE": report["linear_intact_repair_RMSE"],
                **metrics(predictions.prediction, predictions.target),
                "interval_coverage": float(((predictions.target.to_numpy() >= lower) &
                                            (predictions.target.to_numpy() <= upper)).mean()),
            })
        print(f"{family} severity={args.severity:g} complete", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(output / "recovery_metrics.csv", index=False)
    if sha(bundle_path) != frozen_hash:
        raise RuntimeError("The frozen bundle changed during the benchmark.")
    summary = table.groupby("family", as_index=False).agg(
        detection_F1=("detection_F1", "mean"), detection_precision=("detection_precision", "mean"),
        detection_recall=("detection_recall", "mean"), false_repair_rate=("false_repair_rate", "mean"),
        reconstruction_MAE=("reconstruction_MAE", "mean"),
        intact_displacement_RMSE=("intact_displacement_RMSE", "mean"),
        interval_coverage=("interval_coverage", "mean"), replicates=("replicate", "nunique"))
    order = ["clean", *FAULTS, "mixed"]
    summary["order"] = summary.family.map({name: i for i, name in enumerate(order)})
    summary = summary.sort_values("order").drop(columns="order")
    summary.to_csv(output / "recovery_summary.csv", index=False)
    manifest = {
        "status": "complete", "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Post-freeze diagnostic on the exported artifact; no fitting, recalibration or selection.",
        "bundle": str(bundle_path), "bundle_sha256": frozen_hash, "device": args.device,
        "severity": args.severity, "target_protocol": protocol,
        "metric_space": "detection on altered-entry labels; reconstruction and displacement in original-linear units",
        "evaluation_seed_rule": "config['validation_seed'] + 100000 + replicate (manuscript scheme)",
        "official_benchmark_is_exploratory": True,
        "outputs": {name: sha(output / name) for name in ["recovery_metrics.csv", "recovery_summary.csv"]},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(summary.round(3).to_string(index=False), flush=True)
    print(f"Saved recovery benchmark: {output}", flush=True)


if __name__ == "__main__":
    main()
