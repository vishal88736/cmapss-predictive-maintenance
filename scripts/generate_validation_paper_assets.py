"""Generate the recovery/detection benchmark and the cost section.

Both sections are post-freeze diagnostics of the exported artifact. Recovery
detection is only defined on finite altered entries; missing-entry families are
handled by the availability mask, so their detection columns are reported as not
applicable rather than as a spurious zero.
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
MISSING = {"intermittent", "outage", "block"}
ORDER = ["clean", "intermittent", "outage", "block", "noise", "bias", "drift", "stuck", "mixed"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table(command, caption, label, columns, header, rows, note):
    return "\n".join([
        rf"\newcommand{{\{command}}}{{%", r"\begin{table*}[!t]",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{3pt}", rf"\begin{{tabular}}{{{columns}}}", r"\toprule",
        header + r" \\", r"\midrule",
        *[" & ".join(map(str, row)) + r" \\" for row in rows],
        r"\bottomrule", r"\end{tabular}", r"\par\vspace{2pt}",
        rf"\begin{{minipage}}{{\textwidth}}\scriptsize {note}\end{{minipage}}",
        r"\end{table*}", "}", ""])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-run", type=Path, default=ROOT / "artifacts/recovery_benchmark_full")
    parser.add_argument("--efficiency-run", type=Path, default=ROOT / "artifacts/efficiency_benchmark_full")
    args = parser.parse_args(argv)
    pointer = json.loads((OUT / "current_run.json").read_text())
    for path in [args.recovery_run / "manifest.json", args.efficiency_run / "manifest.json"]:
        if json.loads(path.read_text())["bundle_sha256"] != pointer["bundle_sha256"]:
            raise ValueError(f"{path} measured a different frozen bundle.")

    recovery = pd.read_csv(args.recovery_run / "recovery_summary.csv").set_index("family")
    efficiency = pd.read_csv(args.efficiency_run / "efficiency_metrics.csv").set_index("metric")

    rows = []
    for family in ORDER:
        r = recovery.loc[family]
        scope = "none" if family == "clean" else ("missing" if family in MISSING else "finite")
        detected = scope == "finite"
        rows.append([
            family, scope,
            f"{r.detection_F1:.2f}" if detected else "n/a",
            f"{r.detection_recall:.2f}" if detected else "n/a",
            f"{r.reconstruction_MAE:.2f}" if family != "clean" else "n/a",
            f"{r.intact_displacement_RMSE:.3f}",
            f"{100 * r.false_repair_rate:.1f}",
            f"{100 * r.interval_coverage:.1f}",
        ])
    recovery_table = table("CurrentRecoveryTable",
        "Post-freeze recovery and detection benchmark by fault family.",
        "tab:recovery", "llrrrrrr",
        r"Family & Scope & F1 & Recall & Reconst. MAE & Intact disp. & False repair (\%) & Cov. (\%)", rows,
        r"Detection is measured on finite altered entries at severity 0.2, averaged over three corruption "
        r"replicates; it is undefined for missing-entry families, which the availability mask handles instead, "
        r"so those cells are not applicable rather than zero. Reconstruction and intact displacement are in "
        r"original-linear sensor units. False repair is the fraction of healthy entries flagged. Nothing was "
        r"tuned on these values and the deployed artifact is unchanged.")

    finite = [f for f in ORDER if f not in MISSING and f != "clean"]
    worst_f1 = max(finite, key=lambda f: recovery.loc[f, "detection_F1"])
    lowest_recall = min(finite, key=lambda f: recovery.loc[f, "detection_recall"])
    highest_recon = max(finite, key=lambda f: recovery.loc[f, "reconstruction_MAE"])
    intact_max = max(recovery.loc[f, "intact_displacement_RMSE"] for f in ORDER)
    false_max = max(recovery.loc[f, "false_repair_rate"] for f in ORDER)

    recovery_section = rf"""\subsection{{\textbf{{Recovery and Detection Benchmark}}}}
{recovery_table}
\CurrentRecoveryTable
Table~\ref{{tab:recovery}} opens the recovery black box. Detection is deliberately
conservative: precision exceeds 0.86 in every finite fault family, but recall
reaches only {recovery.loc[lowest_recall, 'detection_recall']:.2f} for
{lowest_recall}, so most altered entries are left for the generator rather than
explicitly flagged. {worst_f1} is the easiest family (F1
{recovery.loc[worst_f1, 'detection_F1']:.2f}) and {highest_recon} the hardest to
reconstruct ({recovery.loc[highest_recon, 'reconstruction_MAE']:.2f} original-linear
MAE), consistent with a held value being ambiguous rather than out of range.

Healthy-sensor protection holds: the worst intact displacement is
{intact_max:.3f} original-linear units and the worst false-repair fraction is
{100 * false_max:.1f}\%. Missing-entry families show no detector activity because
they are routed by the availability mask, and their reconstruction column measures
imputation of genuinely absent cells. These numbers also expose the stated
limitation directly: an in-support finite fault with a low detection score is
under-repaired, and coverage falls to
{100 * recovery.loc['stuck', 'interval_coverage']:.1f}\% for stuck faults.
Detection thresholds were frozen before these values were read.
"""

    def metric(name, digits=3):
        return float(efficiency.loc[name, "value"])

    efficiency_table = table("CurrentEfficiencyTable",
        "Post-freeze computational cost of the exported artifact.",
        "tab:efficiency", "lrl",
        r"Metric & Value & Scope", [
            ["Trainable parameters", f"{int(metric('trainable_parameters')):,}", "whole artifact"],
            ["Parameters: detector / generator / predictor / corrector",
             " / ".join(f"{int(metric(f'parameters_{n}')):,}" for n in ["detector", "generator", "predictor", "corrector"]),
             "per module"],
            ["Float32 weight size", f"{metric('model_size'):.2f} MB", "weights only"],
            ["Multiply-accumulates, full path", f"{metric('mac_per_window_full') / 1e6:.1f} M", "one 70-cycle window"],
            ["Multiply-accumulates, bypass", f"{metric('mac_per_window_bypass') / 1e6:.1f} M", "one 70-cycle window"],
            ["Latency, full path", f"{metric('latency_window_full'):.2f} ms", "single window, median"],
            ["Latency, bypass", f"{metric('latency_window_bypass'):.2f} ms", "single window, median"],
            ["Latency, plain predictor", f"{metric('latency_window_plain'):.2f} ms", "single window, median"],
            ["Throughput", f"{metric('throughput_engines_per_second'):.0f} engines/s", "batch 64, full path"],
            ["Fleet pass", f"{metric('latency_per_fleet'):.2f} s", "707 engines, one window each"],
            ["Process peak memory", f"{metric('peak_rss'):.0f} MB", "interpreter and torch included"],
        ],
        r"Wall-clock measurement of one CPU machine, not a hardware-independent claim; multiply-accumulates "
        r"are a complexity proxy for supported operations only. The recovery path costs about "
        rf"{100 * (metric('latency_window_full') / metric('latency_window_bypass') - 1):.0f}\% latency over bypass "
        r"and buys the stuck-fault gain in Table~\ref{tab:ablation}. Peak memory includes the Python and "
        r"PyTorch interpreter, whose baseline is reported in the saved record.")

    efficiency_section = rf"""\subsection{{\textbf{{Computational Cost}}}}
{efficiency_table}
\CurrentEfficiencyTable
The artifact is small: \CurrentParams{{}} parameters, {metric('model_size'):.2f} MB of
float32 weights, and {metric('mac_per_window_full') / 1e6:.1f} M multiply-accumulates
per window. A single-window forward pass takes
{metric('latency_window_full'):.1f} ms on this CPU and the recovery path accounts
for about {100 * (metric('latency_window_full') / metric('latency_window_bypass') - 1):.0f}\%
of it, which is the price of the stuck-fault gain rather than an overhead to hide.
Batched throughput reaches {metric('throughput_engines_per_second'):.0f} engines per
second, so the full 707-engine benchmark evaluates in
{metric('latency_per_fleet'):.1f} s at batch one. These are wall-clock numbers from
one machine; latency and memory are not hardware-independent and the tables report
no energy measurement.
"""

    macros = {
        "CurrentParams": f"{int(metric('trainable_parameters')):,}",
        "CurrentParamMB": f"{metric('model_size'):.2f}",
        "CurrentLatency": f"{metric('latency_window_full'):.1f}",
        "CurrentThroughput": f"{metric('throughput_engines_per_second'):.0f}",
        "CurrentIntactWorst": f"{intact_max:.3f}",
        "CurrentFalseRepairWorst": f"{100 * false_max:.1f}",
        "CurrentRecallLowest": f"{recovery.loc[lowest_recall, 'detection_recall']:.2f}",
    }
    (OUT / "current_validation_macros.tex").write_text(
        "% Data-derived macros; no desired scores are assigned.\n"
        + "\n".join(rf"\newcommand{{\{name}}}{{{value}}}" for name, value in macros.items()) + "\n")
    (OUT / "current_recovery_section.tex").write_text(recovery_section)
    (OUT / "current_efficiency_section.tex").write_text(efficiency_section)

    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Generated from saved measurements; no training, inference or selection.",
        "run": pointer["run"], "bundle_sha256": pointer["bundle_sha256"],
        "recovery_run": str(args.recovery_run.relative_to(ROOT)),
        "efficiency_run": str(args.efficiency_run.relative_to(ROOT)),
        "inputs": {str(path.relative_to(PAPER)): {"sha256": sha(path), "bytes": path.stat().st_size}
                   for path in [args.recovery_run / "recovery_summary.csv", args.recovery_run / "recovery_metrics.csv",
                                args.efficiency_run / "efficiency_metrics.csv", args.efficiency_run / "manifest.json"]},
        "outputs": {name: sha(OUT / name) for name in
                    ["current_validation_macros.tex", "current_recovery_section.tex", "current_efficiency_section.tex"]},
        "post_freeze_diagnostic": True, "deployed_artifact_changed": False,
        "recovery_worst_intact_displacement": round(intact_max, 4),
        "recovery_worst_false_repair_rate": round(false_max, 5),
        "latency_is_machine_specific": True,
    }
    (OUT / "current_validation_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"worst_f1_family": worst_f1, "lowest_recall_family": lowest_recall,
                      "intact_max": round(intact_max, 4), "false_repair_max": round(false_max, 4),
                      "latency_full_ms": round(metric("latency_window_full"), 2),
                      "throughput": round(metric("throughput_engines_per_second"), 1)}, indent=2))


if __name__ == "__main__":
    main()
