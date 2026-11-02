"""Preregistered development-only grouped CV, followed by a frozen final recipe.

Outer folds compare recipes; separate inner engines select epochs and gates.
Every upstream/OOF fit starts independently. Historical calibration and internal
test engines stay excluded from CV. Reused C-MAPSS remains exploratory evidence.
"""

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import torch

from .checkpoints import save_checkpoint
from .config import SUBSETS
from .data import FAULTS, WindowDataset
from .detection import audit_deployed_thresholds
from .pooled import (ROOT, load_pooled_training, pooled_config, run_pooled_experiment,
                     sha, split_pooled_engines, validate_pooled_config, write_json)
from .training import evaluate, fit_corrector, fit_pipeline


VERSION = 1
CELL_FILES = ["training_history.csv", "correction_history.csv", "inner_audits.json", "cv_model.pt", "engine_errors.csv"]
CANDIDATE_KEYS = {
    "prefix_consistency_loss_weight", "prefix_pair_max_lag", "physical_slew_loss_weight",
    "physical_rebound_loss_weight", "physical_slew_margin", "dropout", "learning_rate",
    "repair_learning_rate", "joint_learning_rate", "predictor_epochs", "generator_epochs",
    "joint_epochs", "correction_epochs", "patience", "generator_hidden", "generator_layers",
    "cnn_channels", "gru_hidden", "transformer_dim", "transformer_layers", "transformer_heads",
    "rul_loss_weight", "branch_loss_weight", "clean_rul_loss_weight", "bypass_rul_loss_weight",
    "reconstruction_loss_weight", "identity_loss_weight", "trend_loss_weight",
    "adversarial_loss_weight", "consistency_loss_weight", "finite_recovery_loss_weight",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def implementation_hashes():
    paths = [*sorted((ROOT / "grace_rul").glob("*.py")), ROOT / "scripts/run_grouped_study.py"]
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def validate_protocol(protocol, *, quick=False):
    expected = {"version", "outer_folds", "fold_seed", "inner_validation_fraction", "training_seeds",
                "final_seed", "reference", "clean_weight", "clean_rmse_tolerance", "clean_nasa_tolerance",
                "fault_families", "fault_severities", "corruption_replicates", "candidates"}
    if set(protocol) != expected or protocol["version"] != VERSION:
        raise ValueError("Unknown or missing grouped study protocol fields.")
    for key, minimum in [("outer_folds", 2 if quick else 3), ("corruption_replicates", 1)]:
        value = protocol[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"Invalid {key}.")
    seeds = protocol["training_seeds"]
    if (len(seeds) < (2 if quick else 3) or len(set(seeds)) != len(seeds) or
            any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in seeds) or
            protocol["final_seed"] not in seeds):
        raise ValueError("Use distinct predeclared training seeds and a predeclared final seed.")
    if not 0 < protocol["inner_validation_fraction"] < .5 or not 0 < protocol["clean_weight"] < 1:
        raise ValueError("Invalid inner split or clean/fault score weight.")
    for key in ["clean_rmse_tolerance", "clean_nasa_tolerance"]:
        if not np.isfinite(protocol[key]) or protocol[key] < 0:
            raise ValueError("Invalid cross-validation clean guard.")
    families, levels = protocol["fault_families"], protocol["fault_severities"]
    if (not families or len(set(families)) != len(families) or set(families) - {*FAULTS, "mixed"} or
            not levels or len(set(levels)) != len(levels) or
            any(not np.isfinite(v) or not 0 < v <= 1 for v in levels)):
        raise ValueError("Invalid assessment fault matrix.")
    candidates = protocol["candidates"]
    names = [c.get("name") for c in candidates]
    if len(names) < 2 or len(set(names)) != len(names) or protocol["reference"] not in names:
        raise ValueError("Predeclare a reference and at least one comparator.")
    for c in candidates:
        if (set(c) != {"name", "overrides"} or not isinstance(c["name"], str) or
                not re.fullmatch(r"[a-z][a-z0-9_]*", c["name"]) or
                not isinstance(c["overrides"], dict) or set(c["overrides"]) - CANDIDATE_KEYS):
            raise ValueError("Candidate changes must be named, global training/model settings.")


def validate_boundaries(frame, boundary):
    groups = boundary["partitions"]
    scopes = [set(groups[k]) for k in ["fit", "validation", "calibration", "test"]]
    if (any(not s or len(s) != len(groups[k]) for s,k in zip(scopes, ["fit", "validation", "calibration", "test"])) or
            any(scopes[i] & scopes[j] for i in range(4) for j in range(i)) or
            set.union(*scopes) != set(frame.unit.unique())):
        raise ValueError("Historical engine boundaries must be disjoint and cover the original training data.")
    if set(groups["validation_pool"]) != scopes[1] | scopes[2]:
        raise ValueError("Invalid historical development/calibration boundary.")
    return groups


def grouped_folds(frame, units, protocol):
    """Source-balanced outer and inner splits use engine identity, never targets."""
    metadata = frame[["unit", "source_subset"]].drop_duplicates()
    if metadata.unit.duplicated().any() or set(units) - set(metadata.unit):
        raise ValueError("Engine/source identity is invalid.")
    metadata = metadata[metadata.unit.isin(units)]
    if set(metadata.source_subset) != set(SUBSETS):
        raise ValueError("Every source must be represented in development.")
    k, seed = protocol["outer_folds"], protocol["fold_seed"]
    buckets = [[] for _ in range(k)]
    for source, name in enumerate(SUBSETS):
        ids = np.sort(metadata.loc[metadata.source_subset.eq(name), "unit"].to_numpy())
        if len(ids) < 2*k:
            raise ValueError("Too few engines per source for independent outer/inner folds.")
        shuffled = np.random.default_rng(np.random.SeedSequence([seed, source])).permutation(ids)
        for i, chunk in enumerate(np.array_split(shuffled, k)):
            buckets[i].extend(int(u) for u in chunk)
    result = []
    for fold, held in enumerate(buckets):
        fit, inner = [], []
        for source, name in enumerate(SUBSETS):
            remaining = np.sort(metadata.loc[metadata.source_subset.eq(name) & ~metadata.unit.isin(held), "unit"])
            shuffled = np.random.default_rng(np.random.SeedSequence([seed, source, fold, 911])).permutation(remaining)
            ninner = max(1, math.ceil(len(shuffled)*protocol["inner_validation_fraction"]))
            if ninner >= len(shuffled):
                raise ValueError("Inner selection leaves no fitting engines.")
            inner.extend(int(u) for u in shuffled[:ninner]); fit.extend(int(u) for u in shuffled[ninner:])
        result.append({"fold": fold+1, "fit": sorted(fit), "inner_validation": sorted(inner), "assessment": sorted(held)})
    return result


def assessment_cases(protocol):
    return [("clean", 0., 0)] + [(family, float(level), rep) for family in protocol["fault_families"]
        for level in protocol["fault_severities"] for rep in range(protocol["corruption_replicates"])]


@contextmanager
def selection_read_boundary(root, study, boundary):
    """Fail on accidental test, calibration-result or previous-model reads."""
    root, study, boundary = Path(root).resolve(), Path(study).resolve(), Path(boundary).resolve()
    active, log = [True], {"training_files": {}, "blocked": []}
    allowed = {f"train_{s}.txt" for s in SUBSETS}
    def audit(event, args):
        if not active[0] or event != "open" or not args or not isinstance(args[0], (str, bytes)):
            return
        mode = args[1]
        if isinstance(mode, str) and not ("r" in mode or "+" in mode):
            return
        path = Path(args[0].decode() if isinstance(args[0], bytes) else args[0]).resolve()
        denied = ((path.is_relative_to(root/"data") and path.name not in allowed) or
                  (path.is_relative_to(root/"artifacts") and not path.is_relative_to(study) and path != boundary) or
                  path.is_relative_to(root/"models"))
        if denied:
            log["blocked"].append(str(path))
            raise PermissionError("Development-only study blocked a test/previous-result/model read: " + str(path))
        if path.parent == root/"data":
            log["training_files"][path.name] = log["training_files"].get(path.name, 0)+1
    sys.addaudithook(audit)
    try:
        yield log
    finally:
        active[0] = False


def create_study(root, output, boundary_path, protocol, *, config=None, quick=False, device="auto"):
    root, output, boundary_path = Path(root).resolve(), Path(output).resolve(), Path(boundary_path).resolve()
    if output.exists():
        raise FileExistsError("Use a fresh study directory; plans are immutable.")
    protocol = deepcopy(protocol)
    if quick:
        protocol.update(outer_folds=2, training_seeds=protocol["training_seeds"][:2],
                        fault_families=["mixed", "stuck"], fault_severities=[.2], corruption_replicates=1)
    validate_protocol(protocol, quick=quick)
    recipe = pooled_config(quick=quick) if config is None else deepcopy(config)
    recipe.update(device=device, pooled_maximum_engines_per_source=None, maximum_engines=None,
                  seeds=protocol["training_seeds"], pooled_evaluate_official=False,
                  detector_validation_seed_policy="configured")
    if quick:
        recipe["maximum_batches"] = 1
    validate_pooled_config(recipe)
    if not quick and (recipe["quick_run"] or recipe["maximum_windows"] or recipe["maximum_batches"]):
        raise ValueError("Reduced-exposure execution checks cannot become scientific selection studies.")
    boundary = json.loads(boundary_path.read_text())
    frame, hashes = load_pooled_training(root/"data", recipe)
    if hashes != boundary["source_hashes"]:
        raise ValueError("Original training file hashes differ from the historical boundary plan.")
    groups = validate_boundaries(frame, boundary)
    if split_pooled_engines(frame, recipe["split_seed"]) != groups:
        raise ValueError("Base split settings would change historical calibration/test boundaries.")
    eligible = sorted(set(groups["fit"]) | set(groups["validation"]))
    units = eligible
    if quick:
        units = []
        for i, source in enumerate(SUBSETS):
            ids = np.sort(frame.loc[frame.unit.isin(eligible) & frame.source_subset.eq(source), "unit"].unique())
            units.extend(int(u) for u in np.random.default_rng(protocol["fold_seed"]+i).permutation(ids)[:12])
        units.sort()
    candidates = []
    for candidate in protocol["candidates"]:
        candidate_config = {**recipe, **candidate["overrides"]}
        validate_pooled_config(candidate_config)
        candidates.append({"name": candidate["name"], "config": candidate_config})
    plan = {"version": VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
            "quick_execution_check": bool(quick), "protocol": protocol, "candidates": candidates,
            "historical_boundary_path": str(boundary_path), "historical_boundary_sha256": sha(boundary_path),
            "source_hashes": hashes, "partitions": groups, "eligible_development_units": eligible,
            "study_units": units, "folds": grouped_folds(frame, units, protocol),
            "engine_sources": {str(int(u)): str(s) for u,s in frame.loc[frame.unit.isin(units), ["unit", "source_subset"]].drop_duplicates().itertuples(index=False, name=None)},
            "implementation_sha256": implementation_hashes(), "final_seed": protocol["final_seed"],
            "selection_rule": "Declared clean weight; equal sources, engines and fault contexts; mean over all seeds; per-source/per-seed clean guards against reference; declared candidate order breaks exact ties",
            "evidence_scope": "Exploratory development on reused C-MAPSS. Excluded old test/calibration engines do not restore an untouched benchmark. No engine-specific selection objective."}
    plan["identity_sha256"] = digest(plan)
    output.mkdir(parents=True)
    write_json(output/"study_plan.json", plan)
    write_json(output/"study_status.json", {"status": "planned", "identity_sha256": plan["identity_sha256"], "deployable": False})
    return plan


def read_study(root, output):
    output = Path(output).resolve()
    plan = json.loads((output/"study_plan.json").read_text())
    if digest({k:v for k,v in plan.items() if k != "identity_sha256"}) != plan["identity_sha256"]:
        raise ValueError("Study plan changed after preregistration.")
    if implementation_hashes() != plan["implementation_sha256"]:
        raise ValueError("Implementation changed; create a fresh preregistered study.")
    if sha(plan["historical_boundary_path"]) != plan["historical_boundary_sha256"]:
        raise ValueError("Historical boundary changed.")
    frame, hashes = load_pooled_training(Path(root)/"data", plan["candidates"][0]["config"])
    if hashes != plan["source_hashes"]:
        raise ValueError("Study training data changed.")
    validate_boundaries(frame, {"partitions": plan["partitions"]})
    if grouped_folds(frame, plan["study_units"], plan["protocol"]) != plan["folds"]:
        raise ValueError("Fold scopes disagree with the preregistered engines.")
    return plan, frame


def engine_errors(records, dataset):
    expected = {(int(dataset.engines[i]["unit"]), int(dataset.engines[i]["cycles"][end])) for i,end in dataset.samples}
    if records.duplicated(["unit", "cycle"]).any() or set(zip(records.unit, records.cycle)) != expected:
        raise ValueError("Assessment predictions omit or duplicate declared engine prefixes.")
    error = records.prediction.to_numpy()-records.target.to_numpy()
    if not np.isfinite(error).all():
        raise ValueError("Non-finite assessment errors.")
    values = records[["unit"]].assign(mse=error**2, mae=np.abs(error),
        nasa=np.expm1(np.where(error < 0, -error/13., error/10.)), prefixes=1)
    result = values.groupby("unit", sort=True).agg(mse=("mse", "mean"), mae=("mae", "mean"),
                nasa=("nasa", "mean"), prefixes=("prefixes", "sum")).reset_index()
    if not np.isfinite(result[["mse", "mae", "nasa"]]).all().all():
        raise ValueError("Non-finite engine-balanced assessment statistics.")
    result["source_subset"] = result.unit.map(dataset.source_by_unit)
    return result


def _cell_path(output, candidate, fold, seed):
    return Path(output)/"cells"/candidate/f"fold{fold}_seed{seed}"


def checked_cell(directory, plan, candidate, fold, seed):
    result = json.loads((directory/"result.json").read_text())
    if (result["study_identity_sha256"] != plan["identity_sha256"] or
            (result["candidate"], result["fold"], result["seed"]) != (candidate, fold, seed) or
            result["status"] != "complete"):
        raise ValueError("Cell result does not belong to this complete fold/seed/recipe.")
    scopes = next(item for item in plan["folds"] if item["fold"] == fold)
    if (result["fit_units"] != scopes["fit"] or result["inner_selection_units"] != scopes["inner_validation"] or
            result["assessment_units"] != scopes["assessment"] or set(result["output_sha256"]) != set(CELL_FILES)):
        raise ValueError("Cell evidence does not match the locked fold scopes.")
    for name, h in result["output_sha256"].items():
        if sha(directory/name) != h:
            raise ValueError("Completed fold evidence changed: " + name)
    return pd.read_csv(directory/"engine_errors.csv"), result


def run_cell(output, plan, frame, candidate, fold, seed):
    directory = _cell_path(output, candidate["name"], fold["fold"], seed)
    if (directory/"result.json").is_file():
        return checked_cell(directory, plan, candidate["name"], fold["fold"], seed)
    directory.mkdir(parents=True, exist_ok=True)
    config = {**candidate["config"], "seeds": [seed]}
    fit, inner, held = [frame[frame.unit.isin(fold[key])].copy() for key in ["fit", "inner_validation", "assessment"]]
    model, scaler, history, _ = fit_pipeline(fit, inner, "POOLED", config, seed,
        checkpoint_path=directory/"checkpoints/pipeline.pt", progress_label=f"CV {candidate['name']} fold{fold['fold']} seed{seed}")
    if set(scaler.state["fit_units"]) != set(fold["fit"]):
        raise ValueError("Fold preprocessing fitted outside its fitting engines.")
    model.scaler = scaler
    correction, oof, _ = fit_corrector(model, fit, inner, "POOLED", config, seed,
                                      checkpoint_dir=directory/"checkpoints")
    for scope in oof or []:
        a,b,c = [set(scope[k]) for k in ["fit_units", "inner_validation_units", "held_out_units"]]
        if a & b or a & c or b & c or a | b | c != set(fold["fit"]):
            raise ValueError("Corrector OOF scope crossed an outer/inner boundary.")
    detector = audit_deployed_thresholds(model, inner, scaler, config, config["validation_seed"], previous_audit=model.detector_audit)
    if not detector["accepted"]:
        raise ValueError("Inner-validation deployed detector guard failed.")
    history.to_csv(directory/"training_history.csv", index=False)
    correction.to_csv(directory/"correction_history.csv", index=False)
    write_json(directory/"inner_audits.json", {"detector": detector, "recovery": model.recovery_validation_audit, "oof": oof})
    # This checkpoint is for audit/resumption only. No calibration or deployable
    # bundle is produced, and outer predictions never select this cell's weights.
    save_checkpoint(directory/"cv_model.pt", {"deployable": False, "study_identity_sha256": plan["identity_sha256"],
        "config": config, "state_dict": model.state_dict(), "scaler": scaler.state, "scopes": fold})
    model.eval()
    rows = []
    assessment_config = {**config, "maximum_batches": None, "maximum_windows": None}
    for family, severity, replicate in assessment_cases(plan["protocol"]):
        ds = WindowDataset(held, scaler, assessment_config, mode="heldout", family=family, severity=severity,
                           seed=config["validation_seed"]+100000+replicate)
        _, predictions = evaluate(model, ds, assessment_config)
        rows.append(engine_errors(predictions, ds).assign(candidate=candidate["name"], fold=fold["fold"], seed=seed,
                    family=family, severity=severity, replicate=replicate))
    values = pd.concat(rows, ignore_index=True)
    values.to_csv(directory/"engine_errors.csv", index=False)
    result = {"status": "complete", "deployable": False, "study_identity_sha256": plan["identity_sha256"],
              "candidate": candidate["name"], "fold": fold["fold"], "seed": seed,
              "fit_units": fold["fit"], "inner_selection_units": fold["inner_validation"], "assessment_units": fold["assessment"],
              "repair_enabled": bool(model.repair_enabled), "training_device": str(next(model.parameters()).device),
              "output_sha256": {name:sha(directory/name) for name in CELL_FILES}}
    write_json(directory/"result.json", result)
    return values, result


def summarize(plan, records):
    """Equal engines within sources, equal fault contexts, and all fixed seeds."""
    records = pd.DataFrame(records)
    keys = ["candidate", "seed", "unit", "family", "severity", "replicate"]
    if records.duplicated(keys).any():
        raise ValueError("Repeated outer assessment engine/context.")
    if (set(records.candidate) != {c["name"] for c in plan["candidates"]} or
            set(records.seed) != set(plan["protocol"]["training_seeds"]) or
            set(zip(records.family, records.severity, records.replicate)) != set(assessment_cases(plan["protocol"]))):
        raise ValueError("Assessment contains undeclared recipes, seeds or contexts.")
    outer = {u:fold["fold"] for fold in plan["folds"] for u in fold["assessment"]}
    sources = {int(u):s for u,s in plan["engine_sources"].items()}
    if (not records.fold.eq(records.unit.map(outer)).all() or
            not records.source_subset.eq(records.unit.map(sources)).all() or
            not np.isfinite(records[["mse", "mae", "nasa"]]).all().all() or
            (records[["mse", "mae", "nasa"]] < 0).any().any()):
        raise ValueError("Assessment statistics or engine scopes are invalid.")
    expected_units = set(plan["study_units"])
    for candidate in plan["candidates"]:
        for seed in plan["protocol"]["training_seeds"]:
            for family, severity, replicate in assessment_cases(plan["protocol"]):
                selected = records[(records.candidate == candidate["name"]) & records.seed.eq(seed) &
                    records.family.eq(family) & records.severity.eq(severity) & records.replicate.eq(replicate)]
                if set(selected.unit) != expected_units:
                    raise ValueError("Incomplete assessment matrix; no partial-fold/seed selection is allowed.")
    by_context = records.groupby(["candidate", "seed", "source_subset", "family", "severity", "replicate"], as_index=False).agg(
        mse=("mse", "mean"), MAE=("mae", "mean"), NASA_Mean=("nasa", "mean"), engines=("unit", "nunique"))
    by_context["RMSE"] = np.sqrt(by_context.mse)
    seeds = []
    weight = plan["protocol"]["clean_weight"]
    for (name, seed), rows in by_context.groupby(["candidate", "seed"], sort=False):
        clean = float(rows.loc[rows.family.eq("clean"), "RMSE"].mean())
        fault = float(rows.loc[~rows.family.eq("clean"), "RMSE"].mean())
        seeds.append({"candidate": name, "seed": int(seed), "clean_macro_RMSE": clean,
                      "fault_macro_RMSE": fault, "score": weight*clean+(1-weight)*fault})
    reference = by_context[by_context.candidate.eq(plan["protocol"]["reference"]) & by_context.family.eq("clean")]
    candidates = []
    for candidate in plan["candidates"]:
        name = candidate["name"]
        clean = by_context[by_context.candidate.eq(name) & by_context.family.eq("clean")]
        matched = clean.merge(reference, on=["seed", "source_subset"], suffixes=("", "_reference"), validate="one_to_one")
        if len(matched) != len(plan["protocol"]["training_seeds"])*len(SUBSETS):
            raise ValueError("Missing source/seed clean guards.")
        guards = {
            f"{int(r.seed)}/{r.source_subset}/{metric}": bool(getattr(r, metric) <= getattr(r, metric+"_reference")*(1+tolerance))
            for r in matched.itertuples() for metric,tolerance in [
                ("RMSE", plan["protocol"]["clean_rmse_tolerance"]), ("NASA_Mean", plan["protocol"]["clean_nasa_tolerance"])]}
        scores = [r["score"] for r in seeds if r["candidate"] == name]
        candidates.append({"candidate": name, "eligible": all(guards.values()), "guards": guards,
                           "mean_score": float(np.mean(scores)), "between_seed_std": float(np.std(scores, ddof=1))})
    eligible = [c for c in candidates if c["eligible"]]
    if not eligible:
        raise ValueError("No candidate meets the declared population clean guards.")
    selected = min(eligible, key=lambda r:r["mean_score"])["candidate"]
    return {"selected_candidate": selected, "candidates": candidates, "seed_scores": seeds,
            "scope": "Recipe-selection CV on reused development engines; not independent final-test evidence. Seed SD is descriptive, not a fleet confidence interval."}, by_context


def run_study(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    with (output/"study.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("This study already has an active writer.") from exc
        plan, _ = read_study(root, output)
        write_json(output/"study_status.json", {"status": "running", "identity_sha256": plan["identity_sha256"], "deployable": False})
        access = None
        try:
            with selection_read_boundary(root, output, plan["historical_boundary_path"]) as access:
                plan, frame = read_study(root, output)
                all_records = []
                for candidate in plan["candidates"]:
                    for fold in plan["folds"]:
                        for seed in plan["protocol"]["training_seeds"]:
                            print(f"Grouped study: {candidate['name']} fold {fold['fold']} seed {seed}", flush=True)
                            values, _ = run_cell(output, plan, frame, candidate, fold, seed)
                            all_records.append(values)
                summary, metrics = summarize(plan, pd.concat(all_records, ignore_index=True))
        except BaseException:
            write_json(output/"study_status.json", {"status": "incomplete", "identity_sha256": plan["identity_sha256"], "deployable": False})
            raise
        finally:
            if access is not None:
                write_json(output/"file_access_audit.json", access)
        if access["blocked"]:
            raise ValueError("Selection attempted a prohibited read; study cannot be frozen.")
        metrics.to_csv(output/"context_metrics.csv", index=False)
        write_json(output/"cv_summary.json", summary)
        write_json(output/"study_status.json", {"status": "complete", "deployable": False,
            "quick_execution_check": plan["quick_execution_check"], "identity_sha256": plan["identity_sha256"]})
        return summary


def freeze_recipe(root, output):
    output = Path(output).resolve()
    plan, _ = read_study(root, output)
    if plan["quick_execution_check"]:
        raise ValueError("An execution check cannot freeze a scientific recipe or authorize final evaluation.")
    status = json.loads((output/"study_status.json").read_text())
    access = json.loads((output/"file_access_audit.json").read_text())
    if (status.get("status") != "complete" or status.get("identity_sha256") != plan["identity_sha256"] or
            access.get("blocked") or set(access.get("training_files", {})) != set(plan["source_hashes"])):
        raise ValueError("A complete development-only read audit is required before recipe freeze.")
    records, result_hashes = [], {}
    for candidate in plan["candidates"]:
        for fold in plan["folds"]:
            for seed in plan["protocol"]["training_seeds"]:
                directory = _cell_path(output, candidate["name"], fold["fold"], seed)
                values, _ = checked_cell(directory, plan, candidate["name"], fold["fold"], seed)
                records.append(values); result_hashes[str(directory.relative_to(output)/"result.json")] = sha(directory/"result.json")
    summary, _ = summarize(plan, pd.concat(records, ignore_index=True))
    selected = next(c for c in plan["candidates"] if c["name"] == summary["selected_candidate"])
    config = {**selected["config"], "seeds": [plan["final_seed"]], "pooled_evaluate_official": True}
    frozen = {"version": VERSION, "study_identity_sha256": plan["identity_sha256"],
              "selected_candidate": selected["name"], "config": config, "final_seed": plan["final_seed"],
              "summary": summary, "cell_result_sha256": result_hashes,
              "source_hashes": plan["source_hashes"], "partitions": plan["partitions"],
              "selection_uses_test_metrics": False, "independent_confirmatory_evidence": False}
    frozen["identity_sha256"] = digest(frozen)
    path = output/"frozen_recipe.json"
    if path.exists() and json.loads(path.read_text()) != frozen:
        raise ValueError("Frozen recipe/evidence changed; no reselection in this study.")
    if not path.exists():
        write_json(path, frozen)
    return frozen


def final_experiment(root, output, final_output):
    """Refit one predeclared seed from scratch, then use the existing strict export."""
    output, final_output = Path(output).resolve(), Path(final_output).resolve()
    saved = json.loads((output/"frozen_recipe.json").read_text())
    if freeze_recipe(root, output) != saved:
        raise ValueError("Frozen study changed.")
    if (final_output/"grace_rul_ALL.pt").exists():
        raise FileExistsError("Completed or frozen final models cannot be overwritten.")
    reference = {"study": str(output), "frozen_recipe_sha256": sha(output/"frozen_recipe.json"),
                 "study_identity_sha256": saved["study_identity_sha256"]}
    pointer = final_output/"frozen_study_reference.json"
    if final_output.exists():
        if not pointer.is_file() or json.loads(pointer.read_text()) != reference:
            raise FileExistsError("Final output is not this interrupted frozen study.")
    else:
        # prepare_pooled_run owns initial creation; a matching reference is added
        # after its plan is written, before any fit can begin.
        from .pooled import prepare_pooled_run
        _, groups, _ = prepare_pooled_run(root, saved["config"], final_output)
        if groups != saved["partitions"]:
            raise ValueError("Final refit would change the locked historical partitions.")
        write_json(pointer, reference)
    return run_pooled_experiment(root, saved["config"], final_output)
