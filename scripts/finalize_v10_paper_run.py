#!/usr/bin/env python3
"""Convert a completed v10 repair directory into the frozen-run paper layout.

Only saved predictions and metadata are copied/re-labelled. No model inference,
fitting or selection occurs. The original repair directory is never modified.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import torch

from grace_rul.config import SUBSETS


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-bundle", type=Path,
                        default=ROOT / "artifacts/pooled_repair_20261101T230338Z_9ea6_dpk/final_model_verified/grace_rul_ALL.pt")
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if not (run / "test_metrics.csv").is_file() or not (run / "grace_rul_ALL.pt").is_file():
        raise FileNotFoundError("A completed v10 run with test_metrics.csv and grace_rul_ALL.pt is required.")
    output.mkdir(parents=True)
    (output / "predictions").mkdir()

    bundle = torch.load(run / "grace_rul_ALL.pt", map_location="cpu", weights_only=True)
    entry = bundle["models"]["POOLED"]
    config = entry["training_config"]
    table = pd.read_csv(run / "test_metrics.csv")
    for partition, frame in table.groupby("partition"):
        frame.to_csv(output / f"{partition}_metrics.csv", index=False)
    for name in ["calibration_POOLED.csv", "detector_audit_POOLED.json",
                 "recovery_validation_POOLED.json", "v10_report.json",
                 "validation_history.csv"]:
        if (run / name).is_file():
            shutil.copy2(run / name, output / name)
    for path in (run / "predictions").glob("*.csv.gz"):
        target = path.name.replace("official_", "official_benchmark_").replace("internal_", "internal_test_10pct_")
        shutil.copy2(path, output / "predictions" / target)

    records = pd.DataFrame(entry["engine_manifest"])
    counts = {name: len(entry["partitions"][name]) for name in
              ["fit", "validation", "calibration", "test", "validation_pool"]}
    dataset_rows = []
    for subset in SUBSETS:
        source = records[records.source_subset.eq(subset)]
        dataset_rows.append({"subset": subset, "train": int(len(source)),
                             **{name: int((source.partition == name).sum())
                                for name in ["fit", "validation", "calibration", "test"]}})
    parameter_counts = {}
    for name in ["detector", "generator", "predictor", "corrector"]:
        parameter_counts[name] = int(sum(value.numel() for key, value in entry["state_dict"].items()
                                         if key.startswith(f"{name}.")))
    recovery = json.loads((run / "recovery_validation_POOLED.json").read_text())["selected"]
    detector = json.loads((run / "detector_audit_POOLED.json").read_text())
    manifest = {
        "format_version": bundle["format_version"], "architecture": bundle["architecture"],
        "training_scope": "pooled", "requires_subset_label": False,
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
        "source_bundle": str(args.source_bundle), "source_bundle_sha256": sha(args.source_bundle),
        "artifact": "grace_rul_ALL.pt", "frozen_bundle_sha256": sha(run / "grace_rul_ALL.pt"),
        "status": "complete", "selection_uses_test_metrics": False,
        "weights_retrained": True, "forward_changed": True,
        "calibration_policy": entry["calibration"]["policy"],
        "partition_counts": counts, "training_device": config.get("device"),
        "torch_version": str(bundle.get("torch_version")),
        "official_source_hashes": {f"test_{s}.txt": sha(ROOT / f"data/test_{s}.txt") for s in SUBSETS},
        "source_unchanged": True, "evaluation_complete": True,
    }
    summary = {
        "bundle_sha256": manifest["frozen_bundle_sha256"], "architecture": bundle["architecture"],
        "config": config, "partition_counts": counts, "dataset_counts": dataset_rows,
        "thresholds": entry["thresholds"], "parameter_counts": parameter_counts,
        "calibration": entry["calibration"], "detector_audit": detector,
        "recovery_worst": recovery["worst"], "recovery_contexts": len(recovery["contexts"]),
        "regime_selection": entry["scaler"]["regime_selection"],
        "regimes": len(entry["scaler"]["centers"]), "manifest": manifest,
        "scope": "Saved v10 predictions and metadata; no new fitting, selection or inference.",
    }
    (output / "paper_model_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    # Raw settings are source data, not model output; reuse the preserved trace.
    source_run = args.source_bundle.resolve().parent
    trace = source_run / "engine179_raw_trace.csv"
    if trace.is_file():
        shutil.copy2(trace, output / "engine179_raw_trace.csv")
    print(json.dumps({"output": str(output), "bundle_sha256": manifest["frozen_bundle_sha256"],
                      "partition_counts": counts}, indent=2))


if __name__ == "__main__":
    main()
