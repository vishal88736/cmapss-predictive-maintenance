"""Paired fitting/validation-only recovery-loss pilot; never opens official tests.

This is a short generator fine-tuning diagnostic with the existing main model's
fitting scope. It does not export a deployable model, refit calibration, certify
the full recovery matrix, or replace a fresh main/OOF training run.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch

from grace_rul.artifacts import load_bundle
from grace_rul.data import FAULTS, WindowDataset, load_training_subset
from grace_rul.model import GraceRUL
from grace_rul.training import device_for, evaluate, loader, objectives, seed_all


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_balance(source_run, output_dir=None, *, subsets=("FD002", "FD004"), steps=40,
                  batch_size=16, device="cpu", project_root=ROOT):
    root, source = Path(project_root).resolve(), Path(source_run).resolve()
    for value in [steps, batch_size]:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("Pilot steps/batch size must be positive integers.")
    chosen_device = device_for({"device": device})
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (Path(output_dir) if output_dir else root / "artifacts" / f"recovery_balance_pilot_{stamp}").resolve()
    if output == source or source in output.parents:
        raise ValueError("Use a new pilot directory outside the preserved source run.")
    output.mkdir(parents=True, exist_ok=False)
    recipe = json.loads((root / "configs/v8_recovery_balance.json").read_text())
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
                 name.startswith(("predictions_", "calibration_")))):
            access["blocked_reads"].append(str(path))
            raise PermissionError("This development pilot cannot read official test or calibration result files.")
        if path.parent == root / "data" and name.startswith("train_"):
            access["training_files"].add(str(path))

    sys.addaudithook(protect_tests)
    report = {"status": "running", "scope": __doc__, "source_run": str(source), "steps_per_variant": steps,
        "batch_size": batch_size, "device": str(chosen_device), "recipe": recipe,
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
            config = {**original.config, **recipe, "device": str(chosen_device), "batch_size": batch_size,
                      "maximum_windows": 4096, "maximum_batches": None, "maximum_engines": None}
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
            checkpoint_path = source / "checkpoints" / f"pipeline_{subset}_{entry['selected_seed']}.pt"
            report["original_files_sha256"][str(checkpoint_path)] = sha(checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            state = deepcopy(original.state_dict())
            warm = checkpoint.get("warm_recovery")
            if warm:
                state.update(warm)
            # This counterfactual active-repair state is diagnostic, not an accepted export.
            state["repair_enabled"] = torch.tensor(True)
            for row in checkpoint["history"]:
                if row["stage"] == "predictor":
                    replay.append({"subset": subset, "epoch": row["epoch"],
                        "validation_clean_RMSE": row["validation_clean_RMSE"],
                        "validation_macro_RMSE": row["validation_macro_RMSE"]})
            report["subsets"][subset] = {"fit_units": sorted(fit_ids), "validation_units": sorted(val_ids),
                "calibration_units_excluded": sorted(cal_ids), "warm_rejected_generator_used": bool(warm),
                "corrector_used": False, "source_selected_repair": bool(original.repair_enabled)}
            scaler = loaded[subset]["scaler"]
            cases = [("clean", 0.), *((f, .2) for f in [*FAULTS, "mixed"]), ("intermittent", .5)]

            def validation(model, variant):
                for family, severity in cases:
                    dataset = WindowDataset(val, scaler, config, mode="validation", seed=config["validation_seed"],
                                            family=family, severity=severity)
                    measured, predictions = evaluate(model, dataset, config,
                        fault_thresholds=entry["thresholds"], capture_recovery_tails=True, use_correction=False)
                    meta = {"subset": subset, "variant": variant, "family": family, "severity": severity}
                    rows.append({**meta, **measured})
                    predictions.assign(**meta).to_csv(output / f"{subset}_{variant}_{family}_s{severity:g}.csv", index=False)
                pd.DataFrame(rows).to_csv(output / "validation_metrics.csv", index=False)

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
        pd.DataFrame(replay).to_csv(output / "clean_anchor_history_replay.csv", index=False)
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
    args = parser.parse_args()
    check_balance(args.source_run, args.output_dir, subsets=args.subsets, steps=args.steps,
                  batch_size=args.batch_size, device=args.device)


if __name__ == "__main__":
    main()
