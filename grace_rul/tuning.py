"""A bounded, validation-only search; official test files are never opened."""

import hashlib
import json
from pathlib import Path

import pandas as pd

from .data import load_training_subset, split_engines
from .training import fit_pipeline, validate_training_config


SEARCH_CANDIDATES = [
    ("default", {}),
    ("stride_5", {"window_stride": 5}),
    ("stronger_recovery", {"generator_reconstruction_loss_weight": .5, "trend_loss_weight": .2}),
    ("protect_intact", {"reconstruction_loss_weight": .5, "identity_loss_weight": 1.}),
    ("without_adversarial", {"adversarial_loss_weight": 0.}),
    ("more_clean", {"clean_fraction": .5, "clean_rul_loss_weight": .5}),
]


def comparison_candidates(config):
    """Equal trial counts with useful, method-specific validation candidates."""
    common = [("default", {}), ("stride_5", {"window_stride": 5}),
              ("more_clean", {"clean_fraction": .5, "clean_rul_loss_weight": .5}),
              ("lower_lr", {"learning_rate": .0005, "joint_learning_rate": .00005})]
    if config["train_repair"]:
        # A WGAN comparator must not silently inherit GRACE's zero-adversarial trial.
        return common + [("paper_context", {"sequence_length": 30}),
                         ("stronger_reconstruction", {"reconstruction_loss_weight": 2.})]
    if config["predictor_family"] == "cnn_gru":
        return common + [("local_20", {"local_length": 20}), ("local_50", {"local_length": 50})]
    return common + [("context_50", {"sequence_length": 50}), ("context_100", {"sequence_length": 100})]


def run_validation_search(project_root, config, output_dir, candidates=None):
    root, output = Path(project_root), Path(output_dir)
    candidates = SEARCH_CANDIDATES if candidates is None else candidates
    if not candidates or len({name for name, _ in candidates}) != len(candidates):
        raise ValueError("Search candidates need distinct names and cannot be empty.")
    allowed = {"window_stride", "sequence_length", "local_length", "learning_rate", "repair_learning_rate",
               "joint_learning_rate", "clean_fraction"} | {key for key in config if key.endswith("_loss_weight")}
    for name, overrides in candidates:
        if not name.replace("_", "").isalnum() or set(overrides) - allowed:
            raise ValueError("Search may change only training losses, rates, windows and clean fraction, with safe trial names.")
        validate_training_config({**config, **overrides})
    output.mkdir(parents=True, exist_ok=True)
    rows, selected, audit = [], {}, {}
    for subset in config["subsets"]:
        train = load_training_subset(root / "data", subset)
        groups = split_engines(train, config["split_seed"], config["maximum_engines"],
                               config["validation_fraction"], config["calibration_fraction_of_training"])
        fit = train[train.unit.isin(groups["fit"])]
        validation = train[train.unit.isin(groups["validation"])]
        best_score = float("inf")
        for name, overrides in candidates:
            trial = {**config, **overrides, "tune_before_training": False, "run_matched_benchmarks": False}
            model, _, history, score = fit_pipeline(fit, validation, subset, trial, config["seed"],
                checkpoint_path=output / "checkpoints" / f"{subset}_{name}.pt",
                progress_label=f"{subset} validation search {name}")
            rows.append({"subset": subset, "candidate": name, "validation_RMSE": score,
                         "repair_selected": bool(model.repair_enabled), "epochs": len(history),
                         "overrides": json.dumps(overrides, sort_keys=True),
                         "scope": "upstream validation only; correction fitted after selection"})
            if score < best_score:
                best_score, selected[subset] = score, dict(overrides)
            del model
        audit[subset] = {"partitions": groups, "training_sha256": hashlib.sha256(
            (root / "data" / f"train_{subset}.txt").read_bytes()).hexdigest()}
    table = pd.DataFrame(rows)
    table.to_csv(output / "validation_search.csv", index=False)
    (output / "selected_configs.json").write_text(json.dumps(selected, indent=2))
    (output / "search_manifest.json").write_text(json.dumps({"config": config, "audit": audit,
        "quick_run": config["quick_run"], "scope": "fitting/validation engines only; calibration and official test excluded"}, indent=2))
    return selected, table
