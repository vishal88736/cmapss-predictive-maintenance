#!/usr/bin/env python3
"""Plan/run grouped development CV; freeze its recipe before final refitting."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grace_rul.grouped_study import create_study, final_experiment, freeze_recipe, run_study
from grace_rul.pooled import pooled_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["plan", "run", "freeze", "final"])
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--boundary-plan", type=Path, help="Existing pooled_plan.json; protects all old calibration/test IDs.")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--config", type=Path, help="Base training changes fixed before CV starts.")
    parser.add_argument("--quick", action="store_true", help="Reduced execution check; cannot freeze/evaluate a final model.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--final-output", type=Path)
    args = parser.parse_args()
    if args.stage != "plan" and (args.quick or args.config or args.protocol or args.boundary_plan or args.device != "auto"):
        parser.error("Recipe/device/scope choices are allowed only when creating a fresh plan.")
    if args.stage == "plan":
        if args.boundary_plan is None:
            parser.error("plan requires --boundary-plan from the preserved run.")
        config = pooled_config(quick=args.quick)
        if args.config:
            overrides = json.loads(args.config.read_text())
            if not isinstance(overrides, dict) or set(overrides)-set(config):
                parser.error("Base configuration uses unknown keys.")
            config.update(overrides)
        plan = create_study(ROOT, args.study_dir, args.boundary_plan,
            json.loads((args.protocol or ROOT/"configs/grouped_study.json").read_text()), config=config, quick=args.quick, device=args.device)
        cells = len(plan["candidates"])*len(plan["folds"])*len(plan["protocol"]["training_seeds"])
        print(json.dumps({"study": str(args.study_dir.resolve()), "development_engines": len(plan["study_units"]),
            "outer_fold_seed_recipe_cells": cells, "upstream_fits_including_corrector_OOF": cells*(1+config["oof_folds"]),
            "quick_execution_check": plan["quick_execution_check"], "identity_sha256": plan["identity_sha256"]}, indent=2))
    elif args.stage == "run":
        print(json.dumps(run_study(ROOT, args.study_dir), indent=2))
    elif args.stage == "freeze":
        frozen = freeze_recipe(ROOT, args.study_dir)
        print(json.dumps({"selected_candidate": frozen["selected_candidate"], "final_seed": frozen["final_seed"],
                          "identity_sha256": frozen["identity_sha256"]}, indent=2))
    else:
        if args.final_output is None:
            parser.error("final requires a fresh --final-output directory.")
        artifact, _ = final_experiment(ROOT, args.study_dir, args.final_output)
        print("Frozen final artifact:", artifact)


if __name__ == "__main__":
    main()
