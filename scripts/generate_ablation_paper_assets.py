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
COLUMNS = {"full": "Full", "bypass": r"$-$Recovery", "plain": "Neither"}


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
    args.ablation_run = args.ablation_run.resolve()
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
                     f"{value(condition, 'plain', 'RMSE'):.2f}",
                     f"{100 * value(condition, 'full', 'interval_coverage'):.1f}",
                     f"{value(condition, 'full', 'mean_interval_width'):.1f}"])
    ablation_table = table("CurrentAblationTable",
        "Post-freeze component ablation of one unchanged artifact.",
        "tab:ablation", "lrrrrr",
        r"Condition & Full & $-$Recovery & Neither & Cov. (\%) & Width", rows,
        r"All columns use the same weights, scaler, detector thresholds and interval radius; only "
        r"the inference switches differ. Fault rows average three corruption replicates, clean has none. "
        r"Nothing was selected, removed or retuned from these values, and the deployed artifact is "
        r"unchanged. The corrector is disabled in format 10, so a separate $-$Corrector column would be "
        r"identical to Full. Coverage and mean width are for the full path under the frozen radius. "
        r"Official benchmarks were inspected in earlier development, so this is a diagnostic, not a holdout.")

    gains = {severity: value(f"stuck@{severity}", "bypass", "RMSE") - value(f"stuck@{severity}", "full", "RMSE")
             for severity in ("0.1", "0.2", "0.3", "0.5")}
    clean_full, clean_plain = value("clean", "full", "RMSE"), value("clean", "plain", "RMSE")
    stuck_full = value("stuck@0.2", "full", "RMSE")
    stuck_bypass = value("stuck@0.2", "bypass", "RMSE")
    mixed_gain = value("mixed@0.2", "bypass", "RMSE") - value("mixed@0.2", "full", "RMSE")
    outage_gain = value("outage@0.2", "bypass", "RMSE") - value("outage@0.2", "full", "RMSE")
    block_gain = value("block@0.2", "bypass", "RMSE") - value("block@0.2", "full", "RMSE")
    noise_gain = value("noise@0.2", "bypass", "RMSE") - value("noise@0.2", "full", "RMSE")
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
        "CurrentAblationOutageGain": f"{outage_gain:.2f}",
        "CurrentAblationBlockGain": f"{block_gain:.2f}",
        "CurrentAblationNoiseGain": f"{noise_gain:.2f}",
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
\CurrentAblationMixedGain{{}}. Outage and block faults gain
\CurrentAblationOutageGain{{}} and \CurrentAblationBlockGain{{}} cycles,
respectively. Broadband noise gains only \CurrentAblationNoiseGain{{}} cycles,
consistent with recovery targeting persistent abnormal readings rather than
additive measurement error. On clean endpoints the full path gives
\CurrentAblationCleanFull{{}} against \CurrentAblationCleanPlain{{}} for the
branch-fusion predictor with neither intervention, so the recovery path is not
free when no fault is present.

The bounded observable corrector is disabled in format 10: its exported bound is
zero, so removing it changes RMSE by \CurrentAblationCorrectorDelta{{}} cycles by
construction. The source artifact's earlier null ablation, where removal changed
RMSE by at most 0.02 cycles and never for the worse, motivated this choice. No
corrector adjustment is present in the reported RUL, and the manuscript makes no
claim that the corrector improves accuracy.

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
    manuscript_assets = ROOT / "manuscript" / "paper_assets"
    if manuscript_assets.is_dir() and manuscript_assets.resolve() != OUT.resolve():
        import shutil
        for name in list(provenance["outputs"].keys()) + ["current_ablation_provenance.json"]:
            src = OUT / name
            if src.is_file():
                shutil.copy2(src, manuscript_assets / name)
    print(json.dumps({"stuck_gains": {k: round(v, 2) for k, v in gains.items()},
                      "mixed_gain": round(mixed_gain, 2),
                      "corrector_max_delta": round(corrector_delta, 2),
                      "clean_full": clean_full, "clean_plain": clean_plain,
                      "width": round(width, 1), "coverage": round(coverage, 1)}, indent=2))


if __name__ == "__main__":
    main()
