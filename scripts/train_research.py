"""Run validation tuning, final GRACE-RUL training, and matched comparisons."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grace_rul.artifacts import run_experiment
from grace_rul.benchmark import VARIANTS
from grace_rul.config import DEFAULT_CONFIG, SUBSETS, quick_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--quick", action="store_true", help="Small execution check, excluded from benchmark claims.")
    parser.add_argument("--subsets", nargs="+", choices=SUBSETS)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--output-dir", type=Path, help="Reuse an existing directory only with unchanged configuration.")
    parser.add_argument("--config", type=Path, help="JSON overrides for the default configuration.")
    parser.add_argument("--skip-tuning", action="store_true")
    parser.add_argument("--skip-benchmarks", action="store_true")
    parser.add_argument("--benchmark-variants", nargs="+", choices=sorted(VARIANTS))
    args = parser.parse_args()
    config = dict(DEFAULT_CONFIG)
    if args.config:
        overrides = json.loads(args.config.read_text())
        if not isinstance(overrides, dict) or set(overrides) - set(DEFAULT_CONFIG):
            raise ValueError("Configuration must be a JSON object using known configuration keys.")
        config.update(overrides)
    if args.quick:
        config = quick_config(config)
        # Exercise the whole research workflow at the small budget.
        config["tune_before_training"] = config["run_matched_benchmarks"] = True
    if args.device:
        config["device"] = args.device
    if args.subsets:
        config["subsets"] = args.subsets
    if args.seeds:
        config["seeds"] = args.seeds
    if args.benchmark_variants:
        config["benchmark_variants"] = args.benchmark_variants
    if args.skip_tuning:
        config["tune_before_training"] = False
    if args.skip_benchmarks:
        config["run_matched_benchmarks"] = False
    path, tables = run_experiment(ROOT, config, args.output_dir)
    print(f"Artifact: {path}")
    if "superiority_checks" in tables:
        print(tables["superiority_checks"].to_string(index=False))


if __name__ == "__main__":
    main()
