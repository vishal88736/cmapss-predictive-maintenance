#!/usr/bin/env python3
"""Export tensor-free manuscript metadata from one completed, frozen run."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from grace_rul.artifacts import load_bundle
from grace_rul.config import SENSORS, SETTINGS
from grace_rul.pooled import load_pooled_training, sha, write_json


def export_paper_metadata(run, project_root=ROOT):
    run, root = Path(run).resolve(), Path(project_root).resolve()
    import json
    manifest = json.loads((run/"run_manifest.json").read_text())
    if manifest.get("status") != "complete" or not manifest.get("evaluation_complete"):
        raise ValueError("Paper metadata requires a completed frozen evaluation.")
    artifact = run/"grace_rul_ALL.pt"
    if sha(artifact) != manifest["frozen_bundle_sha256"]:
        raise ValueError("Bundle hash differs from the evaluated manifest.")
    bundle, loaded = load_bundle(artifact, device="cpu")
    model, scaler, entry = [loaded["POOLED"][key] for key in ["model", "scaler", "entry"]]
    frame, hashes = load_pooled_training(root/"data", bundle["config"])
    if hashes != entry["source_hashes"]:
        raise ValueError("Training data no longer match the frozen run.")
    groups = entry["partitions"]
    counts = []
    for source, rows in frame.groupby("source_subset", sort=True):
        units = set(rows.unit.unique())
        counts.append({"subset": str(source), "train": len(units),
                       **{key:len(units & set(groups[key])) for key in ["fit", "validation", "calibration", "test"]}})
    case = frame[frame.unit.eq(4000179)].copy()
    if len(case):
        _, regimes = scaler.transform(case)
        z = (case[SENSORS].to_numpy(float)-np.asarray(entry["scaler"]["sensor_mean"])[regimes])/np.asarray(entry["scaler"]["sensor_std"])[regimes]
        trace = case[["unit", "cycle", *SETTINGS, "sensor_2", "sensor_3", "sensor_4", "sensor_11"]].copy()
        trace["regime"] = regimes
        for index in [2, 3, 4, 11]:
            trace[f"z_sensor_{index}"] = z[:,index-1]
        trace.to_csv(run/"engine179_raw_trace.csv",index=False)
    data = {"bundle_sha256": sha(artifact), "architecture": bundle["architecture"],
            "config": bundle["config"], "partition_counts": {k:len(v) for k,v in groups.items()},
            "dataset_counts": counts, "thresholds": entry["thresholds"],
            "parameter_counts": {name:sum(p.numel() for p in getattr(model,name).parameters())
                                 for name in ["detector", "generator", "predictor", "corrector"]},
            "calibration": entry["calibration"], "detector_audit": entry["detector_audit"],
            "recovery_worst": entry["recovery_validation_audit"]["worst"],
            "recovery_contexts": entry["recovery_validation_audit"]["evaluated_contexts"],
            "regime_selection": entry["scaler"].get("regime_selection"),
            "regimes": len(entry["scaler"]["centers"]), "manifest": manifest,
            "scope": "Metadata and post-freeze descriptive raw-data tracing; no fitting, selection, or new model inference."}
    write_json(run/"paper_model_summary.json", data)
    return run/"paper_model_summary.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    print(export_paper_metadata(args.run))
