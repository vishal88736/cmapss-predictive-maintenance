"""Paired fitting/validation-only recovery-loss pilot; never opens official tests.

This is a short generator fine-tuning diagnostic with the existing main model's
fitting scope. Optional full-matrix mode checks all declared recovery conditions
and original clean RUL/NASA guards. It does not export a deployable model, refit
calibration, or replace a fresh main/OOF training run.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch

from grace_rul.artifacts import load_bundle
from grace_rul.config import DEFAULT_CONFIG
from grace_rul.data import FAULTS, WindowDataset, load_training_subset
from grace_rul.model import GraceRUL
from grace_rul.recovery_validation import DEFAULT_SEVERITIES, RecoveryValidationMatrix, audit_recovery_candidate
from grace_rul.training import checkpoint_guards, corrupt_validation_rmse, device_for, evaluate, loader, objectives, seed_all, validate_training_config


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_recovery_state(source, subset, entry, original, hashes):
    """Use a training checkpoint when present; frozen exports need none.

    Original clean references are read from the checkpoint or frozen upgrade
    report. Missing references never silently turn the clean guards into passes.
    """
    state, warm, history, references = deepcopy(original.state_dict()), None, [], None
    checkpoint_path = source / "checkpoints" / f"pipeline_{subset}_{entry['selected_seed']}.pt"
    reference_source = None
    if checkpoint_path.is_file():
        hashes[str(checkpoint_path)] = sha(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("identity") != entry["pipeline_identity"]:
            raise ValueError("Pilot checkpoint must match the selected source training identity.")
        warm, history = checkpoint.get("warm_recovery"), checkpoint.get("history", [])
        references = {"RMSE": checkpoint["clean_reference"],
                      "NASA_Score_Mean": checkpoint.get("clean_nasa_reference")}
        reference_source = str(checkpoint_path)
        if warm:
            state.update(warm)
    else:
        report_path = source / "safety_upgrade_report.json"
        if report_path.is_file():
            hashes[str(report_path)] = sha(report_path)
            report = json.loads(report_path.read_text())
            rows = [row for row in report.get("subsets", [])
                    if row.get("subset") == subset and row.get("seed") == entry["selected_seed"]]
            if report.get("status") != "accepted" or len(rows) != 1:
                raise ValueError("Frozen upgrade report does not match the selected source subset/seed.")
            references = rows[0].get("original_clean_references")
            reference_source = str(report_path)
    if references is not None:
        values = [references.get(key) for key in ["RMSE", "NASA_Score_Mean"]]
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in values):
            raise ValueError("Original clean RMSE/NASA references must be finite and nonnegative.")
        references = {key: float(references[key]) for key in ["RMSE", "NASA_Score_Mean"]}
    # Active repair is a diagnostic counterfactual if the saved model bypassed it.
    state["repair_enabled"] = torch.tensor(True)
    return state, bool(warm), history, references, reference_source


def check_balance(source_run, output_dir=None, *, subsets=("FD002", "FD004"), steps=40,
                  batch_size=16, device="cpu", full_matrix=False, retain_corrector=False, project_root=ROOT):
    root, source = Path(project_root).resolve(), Path(source_run).resolve()
    for value in [steps, batch_size]:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("Pilot steps/batch size must be positive integers.")
    if not isinstance(full_matrix, bool) or not isinstance(retain_corrector, bool):
        raise ValueError("full_matrix and retain_corrector must be booleans.")
    chosen_device = device_for({"device": device})
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (Path(output_dir) if output_dir else root / "artifacts" / f"recovery_balance_pilot_{stamp}").resolve()
    if output == source or source in output.parents:
        raise ValueError("Use a new pilot directory outside the preserved source run.")
    recipe = json.loads((root / "configs/v8_recovery_balance.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    access = {"active": True, "blocked_reads": [], "training_files": set()}

    def protect_tests(event, args):
        if not access["active"] or event != "open" or not args or not isinstance(args[0], (str, bytes)):
            return
        path = Path(args[0].decode() if isinstance(args[0], bytes) else args[0]).resolve()
        if not path.is_relative_to(root):
            return
        name = path.name
        if (path.parent == root / "data" and name.startswith(("test_", "RUL_"))) or (
                path.is_relative_to(root / "artifacts") and
                (name in {"test_metrics.csv", "robustness_summary.csv"} or
                 name.startswith(("predictions_", "calibration_")) or "predictions" in path.relative_to(root / "artifacts").parts)):
            access["blocked_reads"].append(str(path))
            raise PermissionError("This development pilot cannot read official test or calibration result files.")
        if path.parent == root / "data" and name.startswith("train_"):
            access["training_files"].add(str(path))

    sys.addaudithook(protect_tests)
    report = {"status": "running", "scope": __doc__, "source_run": str(source), "steps_per_variant": steps,
        "batch_size": batch_size, "device": str(chosen_device), "recipe": recipe,
        "full_recovery_matrix_requested": full_matrix,
        "retain_existing_corrector_for_RUL_validation": retain_corrector,
        "variants": ["legacy_objective", "imputation_only", "robust_imputation"],
        "official_tests_evaluated": False, "calibration_refitted": False, "deployable_model_exported": False,
        "original_files_sha256": {}, "subsets": {}}
    path = output / "pilot_report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    rows, training_rows, replay = [], [], []
    try:
        bundle_path = source / "grace_rul_ALL.pt"
        report["original_files_sha256"][str(bundle_path)] = sha(bundle_path)
        bundle, loaded = load_bundle(bundle_path, device="cpu")
        if not subsets or set(subsets) - set(loaded):
            raise ValueError("Pilot subsets must be present in the source bundle.")
        for subset in subsets:
            entry = loaded[subset]["entry"]
            original = loaded[subset]["model"]
            # Frozen upgrades can lack training-only options introduced later.
            # Fill them from declared defaults, retaining the saved inference law.
            config = {**DEFAULT_CONFIG, **original.config, **recipe, "device": str(chosen_device), "batch_size": batch_size,
                      "maximum_windows": 4096, "maximum_batches": None, "maximum_engines": None}
            if full_matrix:
                config["recovery_validation_severities"] = list(DEFAULT_SEVERITIES)
            validate_training_config(config)
            if not config["train_repair"]:
                raise ValueError("The recovery preflight requires a source model with a recovery module enabled in its config.")
            train_path = root / f"data/train_{subset}.txt"
            if sha(train_path) != entry["source_hashes"][train_path.name]:
                raise ValueError("Training source differs from the saved artifact.")
            frame = load_training_subset(root / "data", subset)
            partitions = entry["partitions"]
            fit_ids, val_ids, cal_ids = [set(partitions[k]) for k in ["fit", "validation", "calibration"]]
            if fit_ids & val_ids or fit_ids & cal_ids or val_ids & cal_ids:
                raise ValueError("Source training partitions overlap.")
            if fit_ids != set(loaded[subset]["scaler"].state["fit_units"]):
                raise ValueError("Pilot must retain the original fitting-only scaler scope.")
            fit = frame[frame.unit.isin(fit_ids)].copy()
            val = frame[frame.unit.isin(val_ids)].copy()
            state, warm, history, references, reference_source = source_recovery_state(
                source, subset, entry, original, report["original_files_sha256"])
            for row in history:
                if row["stage"] == "predictor":
                    replay.append({"subset": subset, "epoch": row["epoch"],
                        "validation_clean_RMSE": row["validation_clean_RMSE"],
                        "validation_macro_RMSE": row["validation_macro_RMSE"]})
            report["subsets"][subset] = {"fit_units": sorted(fit_ids), "validation_units": sorted(val_ids),
                "calibration_units_excluded": sorted(cal_ids), "warm_rejected_generator_used": bool(warm),
                "corrector_used": retain_corrector, "source_selected_repair": bool(original.repair_enabled),
                "source_quick_run": bool(original.config["quick_run"]),
                "training_policy": {k: config[k] for k in ["identity_tail_loss_weight", "identity_tail_fraction",
                    "recovery_sampling", "recovery_train_severities", "repair_learning_rate", "maximum_windows"]},
                "original_clean_references": references, "clean_reference_source": reference_source,
                "validation_checks": {}}
            scaler = loaded[subset]["scaler"]
            cases = [("clean", 0.), *((f, .2) for f in [*FAULTS, "mixed"]), ("intermittent", .5)]
            matrix = RecoveryValidationMatrix(val, scaler, config, seed=config["validation_seed"]) if full_matrix else None

            def validation(model, variant):
                audit, measured_cases = None, {}
                if matrix is not None:
                    print(f"{subset} {variant}: auditing all {len(matrix.datasets)} held-validation recovery conditions", flush=True)
                    audit = audit_recovery_candidate(model, matrix, config, use_repair=True, stop_on_failure=False)
                    (output / f"{subset}_{variant}_recovery_matrix.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
                    print(f"{subset} {variant}: matrix complete={audit['complete']}, accepted={audit['accepted']}, failures={len(audit['failing_contexts'])}", flush=True)
                for family, severity in cases:
                    dataset = (matrix.datasets[(family, severity)] if matrix is not None else
                        WindowDataset(val, scaler, config, mode="validation", seed=config["validation_seed"], family=family, severity=severity))
                    measured, predictions = evaluate(model, dataset, config,
                        fault_thresholds=entry["thresholds"], capture_recovery_tails=True, use_correction=retain_corrector)
                    measured_cases[(family, severity)] = measured
                    meta = {"subset": subset, "variant": variant, "family": family, "severity": severity}
                    rows.append({**meta, **measured})
                    predictions.assign(**meta).to_csv(output / f"{subset}_{variant}_{family}_s{severity:g}.csv", index=False)
                pd.DataFrame(rows).to_csv(output / "validation_metrics.csv", index=False)
                clean, mixed = measured_cases[("clean", 0.)], measured_cases[("mixed", .2)]
                guards = (checkpoint_guards(clean, mixed, references["RMSE"], config,
                    clean_nasa_reference=references["NASA_Score_Mean"]) if references is not None else None)
                fault_macro = sum(corrupt_validation_rmse(measured_cases[(f, .2)]) for f in [*FAULTS, "mixed"]) / 8
                current = model.state_dict()
                frozen_unchanged = all(torch.equal(value.detach().cpu(), state[name].detach().cpu())
                    for name, value in current.items() if not name.startswith("generator."))
                generator_changed = any(not torch.equal(value.detach().cpu(), state[name].detach().cpu())
                    for name, value in current.items() if name.startswith("generator."))
                checks = {"primary_guards": guards, "clean_RMSE": clean["RMSE"],
                    "RUL_validation_path": "retained_corrector" if retain_corrector else "upstream_without_corrector",
                    "clean_NASA_Score_Mean": clean["NASA_Score_Mean"], "active_fault_macro_RMSE": fault_macro,
                    "half_clean_half_active_macro_RMSE": .5 * (clean["RMSE"] + fault_macro),
                    "full_sensor_matrix_complete": audit["complete"] if audit is not None else False,
                    "full_sensor_matrix_accepted": audit["accepted"] if audit is not None else False,
                    "all_declared_validation_guards_passed": bool(audit is not None and audit["accepted"] and
                        guards is not None and all(guards.values())),
                    "frozen_modules_unchanged": frozen_unchanged, "generator_weights_changed": generator_changed}
                if not frozen_unchanged:
                    raise ValueError("Pilot altered frozen predictor/detector/correction weights or buffers.")
                report["subsets"][subset]["validation_checks"][variant] = checks
                path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

            model = GraceRUL(config).to(chosen_device)
            model.load_state_dict(state, strict=True)
            validation(model, "initial_active_repair")
            del model
            for variant in report["variants"]:
                seed_all(int(entry["selected_seed"]))
                trial = {**config, "identity_tail_penalty": "log_mse" if variant == "robust_imputation" else "mse",
                         "imputation_loss_weight": 0. if variant == "legacy_objective" else 1.}
                model = GraceRUL(trial).to(chosen_device)
                model.load_state_dict(state, strict=True)
                for name, parameter in model.named_parameters():
                    parameter.requires_grad_(name.startswith("generator."))
                model.generator.train(); model.detector.eval(); model.predictor.eval()
                for module in model.predictor.modules():
                    if isinstance(module, torch.nn.RNNBase):
                        module.train()
                optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                             lr=trial["repair_learning_rate"], weight_decay=.01)
                dataset = WindowDataset(fit, scaler, trial, seed=int(entry["selected_seed"]) + 2000)
                done, epoch = 0, 0
                while done < steps:
                    dataset.set_epoch(epoch, progress=1.)
                    for batch in loader(dataset, trial, shuffle=True, seed=int(entry["selected_seed"]) + epoch):
                        batch = {k: x.to(chosen_device) if isinstance(x, torch.Tensor) else x for k, x in batch.items()}
                        optimizer.zero_grad(set_to_none=True)
                        loss, result = objectives(model, batch, consistency=False)
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Nonfinite pilot loss; no candidate is accepted.")
                        loss.backward()
                        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5., error_if_nonfinite=True)
                        optimizer.step()
                        done += 1
                        training_rows.append({"subset": subset, "variant": variant, "step": done,
                            "loss": float(loss.detach()), "gradient_norm_before_clip": float(norm),
                            **{f"loss_{k}": float(x) for k, x in result["loss_terms"].items()}})
                        if done == 1 or done % 10 == 0:
                            print(f"{subset} {variant}: {done}/{steps} fitting steps", flush=True)
                        if done >= steps:
                            break
                    epoch += 1
                validation(model, variant)
                del model
            checks = report["subsets"][subset]["validation_checks"]
            initial, legacy, revised = [checks[key] for key in ["initial_active_repair", "legacy_objective", "robust_imputation"]]
            comparison = {"fault_macro_change_from_initial": revised["active_fault_macro_RMSE"] - initial["active_fault_macro_RMSE"],
                "fault_macro_change_from_paired_legacy": revised["active_fault_macro_RMSE"] - legacy["active_fault_macro_RMSE"],
                "all_declared_validation_guards_passed": revised["all_declared_validation_guards_passed"],
                "generator_updated": revised["generator_weights_changed"],
                "frozen_modules_unchanged": revised["frozen_modules_unchanged"]}
            supported = (full_matrix and comparison["all_declared_validation_guards_passed"] and
                         comparison["generator_updated"] and comparison["frozen_modules_unchanged"] and
                         comparison["fault_macro_change_from_initial"] <= 0 and
                         comparison["fault_macro_change_from_paired_legacy"] <= 0)
            comparison["recommendation"] = ("supports_a_full_training_trial_only" if supported else
                "investigate_failed_checks_before_full_training" if full_matrix else "short_pilot_only_full_matrix_not_requested")
            report["subsets"][subset]["paired_comparison"] = comparison
        for original_path, digest in report["original_files_sha256"].items():
            if sha(original_path) != digest:
                raise ValueError("A preserved source artifact changed during the pilot.")
        report.update(status="complete", original_artifacts_unchanged=True,
            training_files_read=sorted(access["training_files"]), prohibited_reads_attempted=access["blocked_reads"],
            completed_at_utc=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}",
                      prohibited_reads_attempted=access["blocked_reads"])
        raise
    finally:
        access["active"] = False
        pd.DataFrame(training_rows).to_csv(output / "fitting_loss_history.csv", index=False)
        pd.DataFrame(replay, columns=["subset", "epoch", "validation_clean_RMSE", "validation_macro_RMSE"]).to_csv(
            output / "clean_anchor_history_replay.csv", index=False)
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved fitting/validation development pilot: {output}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--subsets", nargs="+", default=["FD002", "FD004"])
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--full-matrix", action="store_true",
        help="Audit all 41 recovery contexts plus original clean RUL/NASA and primary reconstruction guards.")
    parser.add_argument("--retain-corrector", action="store_true",
        help="Evaluate RUL with the existing frozen corrector; never refits it. Sensor audits remain correction-free.")
    args = parser.parse_args()
    check_balance(args.source_run, args.output_dir, subsets=args.subsets, steps=args.steps,
                  batch_size=args.batch_size, device=args.device, full_matrix=args.full_matrix,
                  retain_corrector=args.retain_corrector)


if __name__ == "__main__":
    main()
