"""Audit structured corruption on real trajectories without model training/inference."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grace_rul.config import CORRUPTION_PROTOCOL, DEFAULT_CONFIG, SUBSETS
from grace_rul.data import FAULTS, OperatingRegimeScaler, WindowDataset, load_subset, source_hashes, split_engines


def audit(output, subsets, severities, replicates):
    if replicates < 1 or any(not np.isfinite(s) or not 0 < s <= .5 for s in severities):
        raise ValueError("Use positive replicates and severities in (0, .5].")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summaries, profiles, examples = [], {}, []
    for subset in subsets:
        train, test, truths = load_subset(ROOT / "data", subset)
        groups = split_engines(train, DEFAULT_CONFIG["split_seed"])
        scaler = OperatingRegimeScaler().fit(train[train.unit.isin(groups["fit"])], subset, DEFAULT_CONFIG["split_seed"])
        assert set(scaler.state["fit_units"]) == set(groups["fit"])
        assert not set(groups["fit"]) & set(groups["validation"] + groups["calibration"])
        profiles[subset] = {"scaler": scaler.state, "partitions": groups, "source_hashes": source_hashes(ROOT / "data", subset)}
        for family in [*FAULTS, "mixed"]:
            for severity in severities:
                active, entries, total_entries, endpoints = 0, 0, 0, 0
                domains, counts, onsets = Counter(), [], []
                for replicate in range(replicates):
                    dataset = WindowDataset(test, scaler, DEFAULT_CONFIG, mode="test", truths=truths,
                        family=family, severity=severity, seed=DEFAULT_CONFIG["validation_seed"] + 100000 + replicate)
                    for index in range(len(dataset)):
                        sample = dataset[index]
                        x, ref, valid, labels = [sample[name].numpy() for name in ["x", "reference", "valid", "altered"]]
                        np.testing.assert_array_equal(x[:, [0, 1, 2, 20, 21]], ref[:, [0, 1, 2, 20, 21]])
                        np.testing.assert_array_equal(labels[valid], x[valid, 3:] != ref[valid, 3:])
                        np.testing.assert_array_equal(x[valid, 3:][~labels[valid]], ref[valid, 3:][~labels[valid]])
                        assert float(sample["raw_target"]) == truths[sample["unit"]]
                        if family in ["noise", "bias", "drift", "stuck"]:
                            assert np.isfinite(x).all()
                        engine_index, _ = dataset.samples[index]
                        meta = dataset.fault_metadata[engine_index]
                        assert meta["scale_source"] == "fitting_engines"
                        active += int(labels[valid].any())
                        entries += int(labels[valid].sum())
                        total_entries += int(valid.sum() * 21)
                        endpoints += 1
                        domains[meta["domain"]] += 1
                        counts.append(len(meta["channels"]))
                        onsets.append(meta["onset_observed_offset"])
                        if index < 3 and replicate == 0:
                            examples.append({"subset": subset, "unit": sample["unit"], **meta})
                summaries.append({"subset": subset, "family": family, "severity": severity,
                    "replicates": replicates, "endpoints": endpoints, "active_endpoints": active,
                    "affected_endpoint_fraction": active / endpoints, "damaged_entry_fraction": entries / total_entries,
                    "median_selected_channels": float(np.median(counts)), "onset_min": min(onsets), "onset_max": max(onsets),
                    "domain_counts": dict(domains)})
            print(f"{subset}: {family} checked", flush=True)
    report = {"protocol": CORRUPTION_PROTOCOL,
        "scope": "Actual trajectories and fitting scalers; no model inference, training or accuracy claims",
        "profiles": profiles, "scenario_summary": summaries, "examples": examples}
    (output / "corruption_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    pd.DataFrame([{**row, "domain_counts": json.dumps(row["domain_counts"], sort_keys=True)} for row in summaries]).to_csv(
        output / "corruption_summary.csv", index=False)
    print(f"Saved corruption audit: {output}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "corruption_audit")
    parser.add_argument("--subsets", nargs="+", choices=SUBSETS, default=SUBSETS)
    parser.add_argument("--severities", nargs="+", type=float, default=[.2])
    parser.add_argument("--replicates", type=int, default=3)
    args = parser.parse_args()
    audit(args.output_dir, args.subsets, args.severities, args.replicates)
