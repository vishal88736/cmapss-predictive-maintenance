"""Post-freeze component ablation of the exported pooled artifact.

Four inference configurations of the *same* frozen weights are compared:

* ``full``          -- guarded recovery and the bounded corrector (deployed path)
* ``bypass``        -- recovery disabled, corrector kept
* ``no_correction`` -- recovery kept, corrector disabled
* ``plain``         -- neither, i.e. the branch-fusion predictor alone

This is a diagnostic, not a selection step: nothing is added, removed or
retuned from these numbers, and the deployed artifact is unchanged. Conditions
reuse the manuscript's own evaluation seeds and corruption replicates, so the
``full`` row reproduces the published fault values instead of resampling them.
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
VARIANTS = {
    "full": {},
    "bypass": {"use_repair": False},
    "no_correction": {"use_correction": False},
    "plain": {"use_repair": False, "use_correction": False},
}


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
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive.")
    root, bundle_path = ROOT, Path(args.bundle).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(args.output_dir or root / "artifacts" / f"component_ablation_{stamp}").resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Ablation output directory must be empty.")
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

    conditions = [("clean", 0., 0)]
    for family in [*FAULTS, "mixed"]:
        for replicate in range(3):
            conditions.append((family, args.severity, replicate))
    # Stuck severity sweep, full vs bypass only, to show where recovery pays.
    for severity in (.1, .3, .5):
        for replicate in range(3):
            conditions.append(("stuck", severity, replicate))

    rows = []
    for family, severity, replicate in conditions:
        dataset = WindowDataset(frame, scaler, config, mode="test", family=family, severity=severity,
                                truths=truths, seed=config["validation_seed"] + 100000 + replicate)
        for name, options in VARIANTS.items():
            if family == "stuck" and severity != args.severity and name in {"no_correction", "plain"}:
                continue
            _, predictions = evaluate(model, dataset, config, fault_thresholds=entry["thresholds"], **options)
            lower, upper = interval_bounds(predictions.prediction, entry["calibration"], protocol=protocol)
            width = upper - lower
            rows.append({"condition": "clean" if family == "clean" else f"{family}@{severity:g}",
                         "family": family, "severity": severity, "replicate": replicate, "variant": name,
                         **metrics(predictions.prediction, predictions.target),
                         "interval_coverage": float(((predictions.target.to_numpy() >= lower) &
                                                    (predictions.target.to_numpy() <= upper)).mean()),
                         "mean_interval_width": float(np.mean(width))})
        print(f"{family} severity={severity:g} replicate={replicate} complete", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(output / "ablation_metrics.csv", index=False)
    if sha(bundle_path) != frozen_hash:
        raise RuntimeError("The frozen bundle changed during ablation.")
    summary = table.groupby(["condition", "variant"], as_index=False).agg(
        RMSE=("RMSE", "mean"), MAE=("MAE", "mean"), interval_coverage=("interval_coverage", "mean"),
        mean_interval_width=("mean_interval_width", "mean"), replicates=("replicate", "nunique"))
    summary.to_csv(output / "ablation_summary.csv", index=False)
    context = "clean has no corruption; every other condition averages its replicates"
    manifest = {
        "status": "complete", "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Post-freeze diagnostic on the exported artifact; no fitting, recalibration or component selection.",
        "deployed_artifact_changed": False, "bundle": str(bundle_path), "bundle_sha256": frozen_hash,
        "variants": {name: options for name, options in VARIANTS.items()},
        "summary_rule": context, "severity_primary": args.severity,
        "evaluation_seed_rule": "config['validation_seed'] + 100000 + replicate (manuscript scheme)",
        "target_protocol": protocol, "device": args.device,
        "official_benchmark_is_exploratory": True,
        "outputs": {name: sha(output / name) for name in ["ablation_metrics.csv", "ablation_summary.csv"]},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    pivot = summary.pivot(index="condition", columns="variant", values="RMSE")
    print(pivot.round(2).to_string(), flush=True)
    print(f"Saved component ablation: {output}", flush=True)


if __name__ == "__main__":
    main()
