"""Train one pooled GRACE-RUL model with an engine-disjoint 80/10/10 split."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grace_rul.pooled import pooled_config, prepare_pooled_run, run_pooled_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cuda")
    parser.add_argument("--quick", action="store_true", help="Small execution check across all four sources; not accuracy evidence.")
    parser.add_argument("--plan-only", action="store_true", help="Save actual engine partitions without fitting or reading official tests.")
    parser.add_argument("--config", type=Path, help="Known-key overrides, with the pooled split/routing contract preserved.")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--skip-official", action="store_true", help="Evaluate only the internal 10% engine holdout.")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = pooled_config(quick=args.quick)
    if args.config:
        overrides = json.loads(args.config.read_text())
        if not isinstance(overrides, dict) or set(overrides) - set(config):
            raise ValueError("Pooled configuration must be an object using known keys.")
        config.update(overrides)
    config["device"] = args.device
    if args.seeds:
        config["seeds"] = args.seeds
    if args.skip_official:
        config["pooled_evaluate_official"] = False
    output = args.output_dir
    if args.plan_only:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output = output or ROOT / "artifacts" / f"pooled_plan_{stamp}"
        _, _, plan = prepare_pooled_run(ROOT, config, output)
        print(json.dumps({"output_dir": str(output), "total_engines": plan["total_engines"], "counts": plan["counts"],
            "scope": "Split plan only; no model trained or official test data read."}, indent=2))
        return
    path, tables = run_pooled_experiment(ROOT, config, output)
    print("Artifact:", path)
    print(tables["test_metrics"].to_string(index=False))


if __name__ == "__main__":
    main()
