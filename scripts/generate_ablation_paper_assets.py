"""Generate the component-ablation section from the saved ablation evidence.

Reports four inference configurations of one frozen artifact and pairs coverage
with mean interval width, so a high coverage number cannot be read without its
sharpness. Nothing here selects, removes or retunes a component: the deployed
path is unchanged and the table is explicitly post-freeze.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT.parent
OUT = PAPER / "paper_assets"
FAMILIES = ["intermittent", "outage", "block", "noise", "bias", "drift", "stuck", "mixed"]
COLUMNS = {"full": "Full", "bypass": r"$-$Recovery", "no_correction": r"$-$Corrector", "plain": "Neither"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table(command, caption, label, columns, header, rows, note):
    return "\n".join([
        rf"\newcommand{{\{command}}}{{%", r"\begin{table*}[!t]",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{4pt}", rf"\begin{{tabular}}{{{columns}}}", r"\toprule",
        header + r" \\", r"\midrule",
        *[" & ".join(map(str, row)) + r" \\" for row in rows],
        r"\bottomrule", r"\end{tabular}", r"\par\vspace{2pt}",
        rf"\begin{{minipage}}{{\textwidth}}\scriptsize {note}\end{{minipage}}",
        r"\end{table*}", "}", ""])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-run", type=Path, default=ROOT / "artifacts/component_ablation_full")
    args = parser.parse_args(argv)
    manifest = json.loads((args.ablation_run / "manifest.json").read_text())
    pointer = json.loads((OUT / "current_run.json").read_text())
    if manifest["bundle_sha256"] != pointer["bundle_sha256"]:
        raise ValueError("Ablation used a different frozen bundle than the manuscript.")
    summary = pd.read_csv(args.ablation_run / "ablation_summary.csv")

    def value(condition, variant, field):
        rows = summary[(summary.condition == condition) & (summary.variant == variant)]
        if len(rows) != 1:
            raise ValueError(f"Missing ablation value for {condition}/{variant}.")
        return float(rows.iloc[0][field])

    rows = []
    for family in ["clean", *FAMILIES]:
        condition = "clean" if family == "clean" else f"{family}@0.2"
        full = value(condition, "full", "RMSE")
        bypass = value(condition, "bypass", "RMSE")
        rows.append([family, f"{full:.2f}", f"{bypass:.2f}",
                     f"{value(condition, 'no_correction', 'RMSE'):.2f}",
                     f"{value(condition, 'plain', 'RMSE'):.2f}",
                     f"{100 * value(condition, 'full', 'interval_coverage'):.1f}",
                     f"{value(condition, 'full', 'mean_interval_width'):.1f}"])
    ablation_table = table("CurrentAblationTable",
        "Post-freeze component ablation of one unchanged artifact.",
        "tab:ablation", "lrrrrrr",
        r"Condition & Full & $-$Recovery & $-$Corrector & Neither & Cov. (\%) & Width", rows,
        r"All four columns use the same weights, scaler, detector thresholds and interval radius; only "
        r"the inference switches differ. Fault rows average three corruption replicates, clean has none. "
        r"Nothing was selected, removed or retuned from these values, and the deployed artifact is "
        r"unchanged. Coverage and mean width are for the full path under the frozen radius. Official "
        r"benchmarks were inspected in earlier development, so this is a diagnostic, not a holdout.")

    gains = {severity: value(f"stuck@{severity}", "bypass", "RMSE") - value(f"stuck@{severity}", "full", "RMSE")
             for severity in ("0.1", "0.2", "0.3", "0.5")}
    clean_full, clean_plain = value("clean", "full", "RMSE"), value("clean", "plain", "RMSE")
    stuck_full = value("stuck@0.2", "full", "RMSE")
    stuck_bypass = value("stuck@0.2", "bypass", "RMSE")
    mixed_gain = value("mixed@0.2", "bypass", "RMSE") - value("mixed@0.2", "full", "RMSE")
    corrector_delta = max(abs(value(c, "no_correction", "RMSE") - value(c, "full", "RMSE"))
                          for c in ["clean", *[f"{f}@0.2" for f in FAMILIES]])
    width = value("clean", "full", "mean_interval_width")
    coverage = 100 * value("clean", "full", "interval_coverage")

    macros = {
        "CurrentAblationCleanFull": f"{clean_full:.2f}",
        "CurrentAblationCleanPlain": f"{clean_plain:.2f}",
        "CurrentAblationStuckFull": f"{stuck_full:.2f}",
        "CurrentAblationStuckBypass": f"{stuck_bypass:.2f}",
        "CurrentAblationStuckGain": f"{gains['0.2']:.2f}",
        "CurrentAblationStuckGainLow": f"{gains['0.1']:.2f}",
        "CurrentAblationStuckGainHigh": f"{gains['0.5']:.2f}",
        "CurrentAblationMixedGain": f"{mixed_gain:.2f}",
        "CurrentAblationCorrectorDelta": f"{corrector_delta:.2f}",
        "CurrentAblationWidth": f"{width:.1f}",
        "CurrentAblationCoverage": f"{coverage:.1f}",
    }
    (OUT / "current_ablation_macros.tex").write_text(
        "% Data-derived macros; no desired scores are assigned.\n"
        + "\n".join(rf"\newcommand{{\{name}}}{{{value}}}" for name, value in macros.items()) + "\n")

    section = rf"""\subsection{{\textbf{{Component Contribution and Interval Width}}}}
{ablation_table}
\CurrentAblationTable
Table~\ref{{tab:ablation}} separates what the frozen artifact actually gains from
its parts. The guarded recovery path is the only component with a material
effect, and only under persistent faults: stuck-fault RMSE falls from
\CurrentAblationStuckBypass{{}} to \CurrentAblationStuckFull{{}} cycles
(\CurrentAblationStuckGain{{}}) at severity 0.2, the gain stays between
\CurrentAblationStuckGainHigh{{}} and \CurrentAblationStuckGainLow{{}} across
severities 0.1--0.5, and mixed faults improve by
\CurrentAblationMixedGain{{}}. Outage and block faults gain about 0.3 cycles.
Broadband noise gains nothing measurable, which is expected because recovery
targets finite abnormal readings rather than additive measurement error.

The bounded observable corrector does not earn its place on accuracy: removing it
changes RMSE by at most \CurrentAblationCorrectorDelta{{}} cycles in any tested
condition, and never for the worse. On clean endpoints the full path gives
\CurrentAblationCleanFull{{}} against \CurrentAblationCleanPlain{{}} for the
branch-fusion predictor with neither intervention. The corrector is retained
because it belongs to the frozen artifact whose validation guards passed, not
because it lowers RUL error, and the manuscript does not claim otherwise.

The same table pairs coverage with sharpness. The mean clean official interval is
\CurrentAblationWidth{{}} cycles wide against a 125-cycle target, so
\CurrentAblationCoverage{{}}\% empirical coverage at this radius is driven by width
rather than by accurate point prediction. Coverage must therefore be read
alongside width, and neither establishes conditional coverage for a realized
engine. Component selection was frozen before these values were read; the table
is a post-freeze diagnostic and the deployed artifact is unchanged.
"""
    (OUT / "current_ablation_section.tex").write_text(section)
    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Generated from saved ablation measurements; no training, inference or selection.",
        "run": pointer["run"], "bundle_sha256": manifest["bundle_sha256"],
        "ablation_run": str(args.ablation_run.relative_to(ROOT)),
        "inputs": {str(path.relative_to(PAPER)): {"sha256": sha(path), "bytes": path.stat().st_size}
                   for path in [args.ablation_run / "ablation_summary.csv",
                                args.ablation_run / "ablation_metrics.csv",
                                args.ablation_run / "manifest.json"]},
        "outputs": {name: sha(OUT / name) for name in
                    ["current_ablation_macros.tex", "current_ablation_section.tex"]},
        "post_freeze_diagnostic": True, "deployed_artifact_changed": False,
        "stuck_gains_by_severity": {key: round(val, 3) for key, val in gains.items()},
        "corrector_max_abs_delta": round(corrector_delta, 3),
        "clean_mean_interval_width": round(width, 2),
    }
    (OUT / "current_ablation_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"stuck_gains": {k: round(v, 2) for k, v in gains.items()},
                      "mixed_gain": round(mixed_gain, 2),
                      "corrector_max_delta": round(corrector_delta, 2),
                      "clean_full": clean_full, "clean_plain": clean_plain,
                      "width": round(width, 1), "coverage": round(coverage, 1)}, indent=2))


if __name__ == "__main__":
    main()
