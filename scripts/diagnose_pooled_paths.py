"""Compare pooled repair/corrector paths on saved model-selection engines only."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grace_rul.pooled_diagnostics import run_diagnostic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True, help="Completed pooled run containing the bundle and epoch checkpoint.")
    parser.add_argument("--output-dir", type=Path, help="Fresh directory outside the source run.")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--pilot-steps", type=int, default=0,
        help="Optional new fixed-step predictor-only fitting pilot; never reconstructs discarded joint weights.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--replicates", type=int, default=3, help="Seeded corruptions of the same engines, not training seeds.")
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = args.output_dir or ROOT / "artifacts" / f"pooled_path_diagnostic_{stamp}"
    path, table = run_diagnostic(ROOT, args.source_run, output, device=args.device,
        pilot_steps=args.pilot_steps, batch_size=args.batch_size, replicates=args.replicates,
        bootstrap_draws=args.bootstrap_draws, seed=args.seed, plot=not args.no_plot)
    print("Diagnostic saved:", path)
    print(table[table.subset.eq("POOLED")].groupby(["path", "family"]).RMSE.mean().to_string())


if __name__ == "__main__":
    main()
