"""Generate global SHAP/LIME evidence from frozen, selected GRACE-RUL weights."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grace_rul.config import SUBSETS
from grace_rul.explain import run_explanation_study
from grace_rul.training import device_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--subsets", nargs="+", choices=[*SUBSETS, "POOLED"])
    parser.add_argument("--examples", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--family", choices=["clean", "mixed", "bias", "noise"], default="clean")
    args = parser.parse_args()
    summary = run_explanation_study(ROOT, args.bundle, args.output_dir,
        subsets=args.subsets, examples=args.examples, repeats=args.repeats, samples=args.samples,
        family=args.family, device=str(device_for({"device": args.device})))
    print(summary.groupby("subset", sort=False).head(5).to_string(index=False))


if __name__ == "__main__":
    main()
