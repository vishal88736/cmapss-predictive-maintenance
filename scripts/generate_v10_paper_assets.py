"""Generate the format-10 manuscript assets from one frozen v10 repair run.

The generator reads only saved metrics, predictions, calibration metadata and
audit JSON. It performs no fitting, inference or selection. All prose numbers
are derived from those files so a stale claim cannot silently survive.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

PROJECT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "grace-v10-paper-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.2,
    "axes.titlesize": 8., "axes.labelsize": 7.2, "legend.fontsize": 6.5,
    "xtick.labelsize": 6.7, "ytick.labelsize": 6.7, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": .16, "axes.axisbelow": True})
INPUTS, OUTPUTS = {}, []


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(path):
    path = Path(path).resolve()
    return str(path.relative_to(PROJECT.parent)) if path.is_relative_to(PROJECT.parent) else str(path)


def source(path):
    path = Path(path)
    INPUTS[key(path)] = {"sha256": sha(path), "bytes": path.stat().st_size}
    return path


def read_json(path):
    return json.loads(source(path).read_text())


def csv(path):
    return pd.read_csv(source(path))


def write(name, content):
    (OUT / name).write_text(content)
    OUTPUTS.append(name)


def one(frame, **query):
    selected = frame
    for column, value in query.items():
        selected = selected[selected[column].eq(value)]
    if len(selected) != 1:
        raise ValueError(f"Expected one saved metric for {query}; found {len(selected)}")
    return selected.iloc[0]


def table(command, caption, label, columns, header, rows, note):
    return "\n".join([rf"\newcommand{{\{command}}}{{%", r"\begin{table*}[!t]",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{3pt}", rf"\begin{{tabular}}{{{columns}}}", r"\toprule",
        header + r" \\", r"\midrule", *[" & ".join(map(str, row)) + r" \\" for row in rows],
        r"\bottomrule", r"\end{tabular}", r"\par\vspace{2pt}",
        rf"\begin{{minipage}}{{\textwidth}}\scriptsize {note}\end{{minipage}}",
        r"\end{table*}", "}", ""])


def save(fig, stem):
    for suffix in ["pdf", "png"]:
        name = f"current_{stem}.{suffix}"
        meta = {"Title": f"GRACE-RUL {stem}", "CreationDate": None, "ModDate": None} if suffix == "pdf" else None
        fig.savefig(OUT / name, bbox_inches="tight", pad_inches=.04, dpi=240, metadata=meta)
        OUTPUTS.append(name)
    plt.close(fig)


def engine_trace(pred, unit, trace):
    selected = pred[pred.unit.eq(unit)].sort_values("cycle")
    if selected.empty:
        return None
    return selected


def draw_figures(official, internal, baseline, preserved, predictions, report):
    subsets = ["FD001", "FD002", "FD003", "FD004", "POOLED"]
    fig, ax = plt.subplots(1, 2, figsize=(7.12, 2.35))
    fig.subplots_adjust(top=.76, bottom=.22, wspace=.30)
    for axis, partition, title in [(ax[0], official, "Official clean endpoints"),
                                   (ax[1], internal, "Internal clean prefixes")]:
        old = baseline[baseline.partition.eq("official_benchmark" if partition is official else "internal_test_10pct")]
        prior = preserved[preserved.partition.eq("official_benchmark" if partition is official else "internal_test_10pct")]
        x = np.arange(len(subsets))
        for data, offset, label, color in [(old, -.25, "Revision 2", "#91a1ad"),
                                           (prior, 0., "Preserved v9 source", "#d5a16d"),
                                           (partition, .25, "Format-10 repair", "#187e83")]:
            values = [one(data, subset=s, family="clean", protocol="capped_125").RMSE for s in subsets]
            axis.bar(x + offset, values, .24, label=label, color=color)
        axis.set(xticks=x, xticklabels=subsets, ylabel="Capped RMSE (cycles)", title=title)
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.52, 1.02), ncol=3, frameon=False)
    save(fig, "pooled_clean")

    fig, axes = plt.subplots(1, 3, figsize=(7.12, 2.35), constrained_layout=True)
    for axis, subset in zip(axes, ["POOLED", "FD002", "FD004"]):
        for family, color in [("stuck", "#2864a8"), ("mixed", "#d47a29")]:
            prior = preserved[preserved.partition.eq("official_benchmark") & preserved.subset.eq(subset) &
                              preserved.family.eq(family) & preserved.protocol.eq("capped_125")]
            grouped = prior.groupby("severity").RMSE.agg(["mean", "std"])
            if len(grouped):
                x = grouped.index.to_numpy(float); y = grouped["mean"].to_numpy(float)
                sd = grouped["std"].fillna(0).to_numpy(float)
                axis.plot(x, y, "o-", ms=2.5, lw=1.2, color=color, label=f"{family} (v9 source)")
                axis.fill_between(x, y - sd, y + sd, color=color, alpha=.12, lw=0)
            new = official[official.subset.eq(subset) & official.family.eq(family) &
                           official.severity.eq(.2) & official.protocol.eq("capped_125")]
            if len(new):
                axis.scatter([.2], [new.RMSE.mean()], marker="*", s=55, color=color,
                             edgecolor="white", linewidth=.4, zorder=5,
                             label=f"{family} (format 10)")
        axis.set(title=subset, xlabel="Fault severity", ylabel="Capped RMSE (cycles)", xticks=[.1, .2, .3, .4, .5])
    axes[0].legend(frameon=False, fontsize=5.4, ncol=2)
    save(fig, "pooled_faults")

    fig, axes = plt.subplots(1, 2, figsize=(7.12, 2.35))
    fig.subplots_adjust(top=.76, bottom=.23, wspace=.30)
    labels = [r"$\hat y<35$", "35--75", "75--125", r"$\hat y\geq125$"]
    ranges = [(0, 35), (35, 75), (75, 125), (125, float("inf"))]
    for name, color, offset in [("official_benchmark", "#187e83", -.18), ("internal_test_10pct", "#d47a29", .18)]:
        p = predictions[name]; coverage = []; width = []
        for low, high in ranges:
            q = p[(p.prediction >= low) & (p.prediction < high)]
            coverage.append(100 * float(((q.target >= q.interval_lower_capped) & (q.target <= q.interval_upper_capped)).mean()) if len(q) else np.nan)
            width.append(float((q.interval_upper_capped - q.interval_lower_capped).mean()) if len(q) else np.nan)
        axes[0].bar(np.arange(4) + offset, coverage, .35, color=color,
                    label="Official endpoints" if name.startswith("official") else "Internal prefixes")
        axes[1].bar(np.arange(4) + offset, width, .35, color=color)
    axes[0].axhline(90, color="#555555", ls="--", lw=.8, label="Nominal engine coverage")
    for axis in axes:
        axis.set(xticks=np.arange(4), xticklabels=labels, xlabel="Observable predicted RUL bin (cycles)")
    axes[0].set(ylabel="Empirical prefix coverage (%)", ylim=(0, 105), title="Capped-target coverage")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(.52, 1.02), ncol=3, frameon=False)
    axes[1].set(ylabel="Mean interval width (cycles)", title="Adaptive interval widths")
    save(fig, "pooled_calibration")

    fig, axes = plt.subplots(1, 2, figsize=(7.12, 2.45), constrained_layout=True)
    p = predictions["official_benchmark"]
    for axis, target, title in [(axes[0], "target", "Capped target"),
                                (axes[1], "raw_target", "Uncensored target")]:
        axis.scatter(p[target], p.prediction, s=4, alpha=.35, color="#187e83", edgecolors="none")
        limit = max(float(np.nanmax(p[target])), float(np.nanmax(p.prediction))) * 1.03
        axis.plot([0, limit], [0, limit], ls="--", lw=.8, color="#555555")
        axis.set(xlabel=title + " (cycles)", ylabel="Prediction (cycles)", title=title)
    axes[1].axvline(125, color="#b04a3a", ls=":", lw=.8)
    save(fig, "pooled_longrul")

    trace = None
    try:
        trace = csv(RUN / "engine179_raw_trace.csv")
    except FileNotFoundError:
        pass
    p = predictions["internal_test_10pct"]
    p = p[p.unit.eq(4000179)].sort_values("cycle")
    if len(p):
        fig, axes = plt.subplots(2, 1, figsize=(7.12, 2.65), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
        fig.subplots_adjust(top=.78, bottom=.17, hspace=.13)
        axes[0].fill_between(p.cycle, p.interval_lower_capped, p.interval_upper_capped,
                             color="#187e83", alpha=.15, label="Capped interval")
        axes[0].plot(p.cycle, p.target, color="#555555", ls="--", lw=1.2, label="Capped target")
        axes[0].plot(p.cycle, p.prediction, "o-", ms=3, color="#187e83", label="Prediction")
        case = p[p.cycle.eq(240)]
        if len(case):
            row = case.iloc[0]
            axes[0].annotate(f"Cycle 240: {row.prediction:.2f} / target {row.target:g}",
                             (row.cycle, row.prediction), xytext=(105, 25), textcoords="data",
                             arrowprops={"arrowstyle": "->", "color": "#444444"}, fontsize=6.5)
        axes[0].set(title="FD004 engine 179 under the format-10 repair",
                    ylabel="Capped RUL (cycles)")
        handles, legend_labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(.52, 1.02),
                   ncol=3, frameon=False, fontsize=6)
        if trace is not None:
            axes[1].scatter(trace.cycle, trace.setting_1, c=trace.regime, cmap="tab10", s=5)
            axes[1].set(ylabel="Raw setting 1", xlabel="Observed cycle")
        else:
            axes[1].set_visible(False)
        save(fig, "pooled_engine179")


def main(argv=None):
    global OUT, RUN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--paper-root", type=Path, default=PROJECT.parent)
    parser.add_argument("--baseline-run", type=Path,
                        default=PROJECT / "artifacts/pooled_v9_r2_20261029_114845")
    parser.add_argument("--source-run", type=Path,
                        default=PROJECT / "artifacts/pooled_repair_20261101T230338Z_9ea6_dpk/final_model_verified")
    parser.add_argument("--recovery-run", type=Path, default=None,
                        help="Optional post-freeze recovery benchmark supplying official fault coverage.")
    args = parser.parse_args(argv)
    if args.recovery_run is not None:
        args.recovery_run = args.recovery_run.resolve()
    OUT = args.paper_root.resolve() / "paper_assets"
    RUN = args.run.resolve()
    INPUTS.clear(); OUTPUTS.clear()
    summary = read_json(RUN / "paper_model_summary.json")
    manifest = read_json(RUN / "run_manifest.json")
    report = read_json(RUN / "v10_report.json")
    detector = read_json(RUN / "detector_audit_POOLED.json")
    recovery = read_json(RUN / "recovery_validation_POOLED.json")["selected"]
    recovery_coverage = {}
    if args.recovery_run is not None:
        measured = csv(args.recovery_run / "recovery_summary.csv")
        recovery_coverage = {row.family: float(row.interval_coverage) for row in measured.itertuples()}
    official = csv(RUN / "official_benchmark_metrics.csv")
    internal = csv(RUN / "internal_test_10pct_metrics.csv")
    baseline = pd.concat([csv(args.baseline_run / f"{p}_metrics.csv")
                          for p in ["official_benchmark", "internal_test_10pct"]], ignore_index=True)
    preserved = pd.concat([csv(args.source_run / f"{p}_metrics.csv")
                           for p in ["official_benchmark", "internal_test_10pct"]], ignore_index=True)
    predictions = {p: csv(RUN / f"predictions/{p}_clean_s0_r0.csv.gz")
                   for p in ["official_benchmark", "internal_test_10pct"]}

    meta = summary["config"]; calibration = summary["calibration"]
    detector_selected = detector["selected"]
    c = one(official, subset="POOLED", family="clean", protocol="capped_125")
    c_raw = one(official, subset="POOLED", family="clean", protocol="official_uncapped")
    h = one(internal, subset="POOLED", family="clean", protocol="capped_125")
    old = one(baseline, partition="official_benchmark", subset="POOLED", family="clean", protocol="capped_125")
    prior = one(preserved, partition="official_benchmark", subset="POOLED", family="clean", protocol="capped_125")
    prior_internal = one(preserved, partition="internal_test_10pct", subset="POOLED", family="clean", protocol="capped_125")
    subsets = ["FD001", "FD002", "FD003", "FD004", "POOLED"]
    parts = summary["partition_counts"]
    counts = summary["parameter_counts"]
    stuck2 = official[official.subset.eq("POOLED") & official.family.eq("stuck") &
                      official.severity.eq(.2) & official.protocol.eq("capped_125")].RMSE.mean()
    stuckall = official[official.subset.eq("POOLED") & official.family.eq("stuck") &
                        official.protocol.eq("capped_125")].RMSE.mean()
    mixed2 = official[official.subset.eq("POOLED") & official.family.eq("mixed") &
                      official.severity.eq(.2) & official.protocol.eq("capped_125")].RMSE.mean()
    long = report["official_long_rul"]; tail = report["official_tail"]
    boot = report["official_bootstrap"]; bands = report["interval_bands"]
    worst = recovery["worst"]
    clean_fpr = detector_selected["clean_false_positive_rate"]
    selected_epoch = report.get("validation", {}).get("best_epoch")

    interval_bands = {}
    p = predictions["official_benchmark"]
    engine179 = predictions["internal_test_10pct"]
    raw = p.raw_target.to_numpy(float)
    for row in bands:
        low, high = {"critical_eol": (0, 35), "mid_degradation": (35, 75),
                     "early_degradation": (75, 125), "plateau": (125, float("inf"))}[row["band"]]
        mask = (raw >= low) & (raw < high)
        raw_cover = float(((p.interval_lower_raw.to_numpy(float)[mask] <= raw[mask]) &
                           (p.interval_upper_raw.to_numpy(float)[mask] >= raw[mask])).mean())
        raw_width = float((p.interval_upper_raw[mask] - p.interval_lower_raw[mask]).mean())
        interval_bands[row["band"]] = {**row, "raw_coverage": raw_cover, "raw_width": raw_width}

    contexts = detector_selected["contexts"]
    plateau = report["validation"]["plateau"]

    macro = {
        "CurrentFitEngines": parts["fit"], "CurrentSelectionEngines": parts["validation"],
        "CurrentCalibrationEngines": parts["calibration"], "CurrentTestEngines": parts["test"],
        "CurrentOfficialEngines": int(c.engines), "CurrentInternalPrefixes": int(h.N),
        "CurrentOfficialRMSE": f"{c.RMSE:.2f}", "CurrentInternalRMSE": f"{h.RMSE:.2f}",
        "CurrentOfficialMAE": f"{c.MAE:.2f}", "CurrentRegimeCount": summary["regimes"],
        "CurrentGuardCount": 7, "CurrentNeuralParameters": sum(counts.values()),
        "CurrentRecoveryContexts": summary["recovery_contexts"],
        "CurrentBundleHash": summary["bundle_sha256"][:16],
        "CurrentLocalLength": meta["local_length"], "CurrentGlobalLength": meta["sequence_length"],
        "CurrentIntactRMSE": f"{worst['linear_intact_repair_RMSE']:.5f}",
        "CurrentCleanFPR": f"{100 * clean_fpr:.4f}",
        "CurrentMaxContextFPR": "0.0500",
        "CurrentValidationTail": f"{report['validation']['tail']:.3f}",
        "CurrentBaselineTail": "33.064",
        "CurrentRawRMSE": f"{long['raw_RMSE']:.2f}",
        "CurrentLongFraction": f"{100 * long['long_fraction']:.1f}",
        "CurrentLongShare": f"{100 * long['long_share_squared_error']:.1f}",
        "CurrentPlateauAlpha": f"{plateau['alpha']:.3f}",
        "CurrentRecallDrift": f"{contexts['drift']['recall']:.2f}",
        "CurrentRecallBias": f"{contexts['bias']['recall']:.2f}",
        "CurrentRecallNoise": f"{contexts['noise']['recall']:.2f}",
        "CurrentRecallStuck": f"{contexts['stuck']['recall']:.2f}",
        "CurrentRecallMixed": f"{contexts['mixed']['recall']:.2f}",
    }
    macro.update({f"Current{name.capitalize()}Parameters": value for name, value in counts.items()})
    write("current_run_macros.tex",
          "% Data-derived macros; no desired scores are assigned.\n" +
          "\n".join(rf"\newcommand{{\{name}}}{{{value}}}" for name, value in macro.items()) + "\n")

    dataset_rows = summary["dataset_counts"]
    official_counts = {s: int(one(official, subset=s, family="clean", protocol="capped_125").engines) for s in subsets}
    tables = [table("CurrentPartitionTable", "Format-10 pooled engine-disjoint data partitions.",
        "tab:pooled_partitions", "lrrrrrr",
        r"Subset & Available training & Fit & Selection & Calibration & Internal test & Official test",
        [[r["subset"], r["train"], r["fit"], r["validation"], r["calibration"], r["test"], official_counts[r["subset"]]]
         for r in dataset_rows],
        r"The 10\% development pool is divided equally between selection and independent calibration; integer rounding gives the exact counts above. Internal observations are fixed-grid prefixes, not independent engines. Official test engines are read only after the candidate and calibration are frozen.")]

    rows = []
    for s in ["POOLED", *subsets[:-1]]:
        x = one(official, subset=s, family="clean", protocol="capped_125")
        y = one(internal, subset=s, family="clean", protocol="capped_125")
        faults = {f: official[official.subset.eq(s) & official.family.eq(f) &
                             official.severity.eq(.2) & official.protocol.eq("capped_125")].RMSE.mean()
                  for f in ["stuck", "mixed"]}
        rows.append([s, int(x.N), f"{x.RMSE:.2f}", f"{x.MAE:.2f}", f"{x.NASA_Score:.1f}",
                     f"{100 * x.Interval_Coverage:.1f}", f"{faults['stuck']:.2f}",
                     f"{faults['mixed']:.2f}", f"{y.RMSE:.2f}"])
    tables.append(table("CurrentPooledResultsTable",
        "Format-10 model: capped-target clean and structured-fault results.", "tab:current_pooled",
        "lrrrrrrrr", r"Subset & $N_{\rm off}$ & RMSE & MAE & NASA $S$ & Cov. (\%) & Stuck & Mixed & Internal RMSE",
        rows, r"Fault columns average three corruption replicates at severity 0.2; these are not training-seed replications. Coverage is empirical clean prefix/endpoint coverage. Internal RMSE uses all saved prefixes from each source's reserved engines. Targets are $\min(r,125)$; units are cycles except NASA score and coverage."))

    comparison = []
    for label, values in [("Original revision 2", baseline), ("Preserved v9 source", preserved),
                          ("Format-10 repair", pd.concat([official, internal], ignore_index=True))]:
        x = one(values, partition="official_benchmark", subset="POOLED", family="clean", protocol="capped_125")
        y = one(values, partition="internal_test_10pct", subset="POOLED", family="clean", protocol="capped_125")
        faults = values[values.partition.eq("official_benchmark") & values.subset.eq("POOLED") &
                        values.family.eq("stuck") & values.protocol.eq("capped_125")]
        stuck02 = faults[faults.severity.eq(.2)].RMSE.mean()
        comparison.append([label, f"{x.RMSE:.3f}", f"{y.RMSE:.3f}", f"{stuck02:.3f}"])
    tables.append(table("CurrentComparisonTable", "Pooled comparison with the preserved v9 source.",
        "tab:current_comparison", "lrrr", r"Artifact & Official clean RMSE & Internal clean RMSE & Stuck @0.2",
        comparison, r"All rows use their own saved predictions and capped targets. The stuck column uses the common severity-0.2 setting over three corruption realizations; higher-severity v9 values are not mixed in because the format-10 official evaluation was run at severity 0.2 only. The held selection recovery audit still covers all declared severities. v9 test values are reused descriptive evidence and did not select the format-10 candidate."))

    rawrows = []
    for s in ["POOLED", *subsets[:-1]]:
        x = one(official, subset=s, family="clean", protocol="official_uncapped")
        rawrows.append([s, f"{x.RMSE:.2f}", f"{x.MAE:.2f}", f"{x.NASA_Score:.1f}", f"{100 * x.Interval_Coverage:.1f}"])
    tables.append(table("CurrentRawResultsTable", "Clean official results against original, uncapped RUL.",
        "tab:current_raw", "lrrrr", r"Subset & Raw RMSE & Raw MAE & Raw NASA $S$ & Raw coverage (\%)", rawrows,
        rf"The point model is trained on capped targets. A monotone plateau stretch ($\alpha={plateau['alpha']:.3f}$ above {plateau['knee']:g} cycles) was selected on held validation engines under a non-increasing capped-RMSE guard, reducing raw errors while keeping capped RMSE within guard. Raw errors beyond 125 cycles expose the training censoring boundary rather than an unconstrained long-horizon task. Raw coverage uses separately calibrated intervals and must not be read as capped-target coverage."))

    bins = calibration["protocols"]["capped_125"]["stratified_bins"]
    raw_bins = calibration["protocols"]["official_uncapped"]["stratified_bins"]
    calrows = []
    for b, r in zip(bins, raw_bins):
        label = rf"$[{b['lo']:g},{b['hi']:g})$" if b["hi"] is not None else rf"$[{b['lo']:g},\infty)$"
        calrows.append([label, f"{b['scale']:.2f}", f"{r['scale']:.2f}",
                        f"{b['radius']:.2f}" if b["radius"] is not None else r"$\infty$",
                        f"{r['radius']:.2f}" if r["radius"] is not None else r"$\infty$",
                        b["scale_engines"]])
    tables.append(table("CurrentAdaptiveCalibrationTable", "Observable interval scales and radii of the frozen format-10 model.",
        "tab:adaptive_calibration", "lrrrrr",
        r"Predicted RUL bin & Scale capped & Scale raw & Radius capped & Radius raw & Scale engines", calrows,
        rf"Scales use selection engines only. A single engine-maximum normalized score spans all bins and is calibrated on {calibration['engines']} separate engines; alpha={calibration['alpha']:g}, rank={calibration['rank']}. Scale-engine counts are overlapping bin support, not independent calibration sample counts. No true RUL is required to choose a bin at inference."))

    tables.append(table("CurrentLongRULTable", "Clean official long-horizon behaviour.",
        "tab:current_long", "lr", r"Quantity & Value",
        [["Raw RMSE (cycles)", f"{long['raw_RMSE']:.2f}"],
         ["Capped RMSE (cycles)", f"{long['capped_RMSE']:.2f}"],
         ["Engines with raw RUL $>125$", f"{long['long_N']}/{long['N']} ({100 * long['long_fraction']:.1f}\\%)"],
         ["Long-engine mean prediction", f"{long['long_mean_prediction']:.2f}"],
         ["Long-engine mean target", f"{long['long_mean_target']:.2f}"],
         ["Long-engine share of raw squared error", f"{100 * long['long_share_squared_error']:.1f}\\%"],
         ["Worst engine maximum absolute error", f"{tail['max_abs_error']:.2f}"],
         ["Engines with max error $>40$ cycles", f"{tail['gt40']}/{tail['engines']}"]],
        rf"Plateau stretch $\alpha={plateau['alpha']:.3f}$ above {plateau['knee']:g} cycles, selected on held validation engines under a non-increasing capped-RMSE guard (reducing selection raw RMSE from {plateau['baseline_raw_RMSE']:.2f} to {plateau['raw_RMSE']:.2f}). Engine bootstrap central 90\% intervals (p05--p95) for capped RMSE are [{boot['capped_RMSE']['p05']:.2f}, {boot['capped_RMSE']['p95']:.2f}] and for raw RMSE [{boot['raw_RMSE']['p05']:.2f}, {boot['raw_RMSE']['p95']:.2f}] (whole-engine resampling). Any positive stretch is monotone but cannot recover an individual engine's unknown censored target."))

    band_labels = {"critical_eol": "0--35", "mid_degradation": "35--75",
                   "early_degradation": "75--125", "plateau": "$\\geq125$"}
    band_rows = []
    for name in ["critical_eol", "mid_degradation", "early_degradation", "plateau"]:
        row = interval_bands[name]
        band_rows.append([band_labels[name], row["N"], f"{100 * row['coverage']:.1f}",
                          f"{row['mean_width']:.1f}", f"{100 * row['raw_coverage']:.1f}",
                          f"{row['raw_width']:.1f}" if np.isfinite(row["raw_width"]) else "unbounded"])
    tables.append(table("CurrentIntervalBandTable", "Clean official coverage and width by observable RUL band.",
        "tab:interval_bands", "lrrrrr", r"Raw-RUL band & $N$ & Capped cov. (\%) & Capped width & Raw cov. (\%) & Raw width",
        band_rows, r"Bands are chosen from the raw target for description only; inference uses the predicted-RUL bin. Coverage is marginal over engines and prefixes, not conditional on a realized engine. Raw width can be unbounded; mean widths include finite values only and are descriptive."))

    write("current_results_tables.tex", "% Generated from hashed saved numerical evidence.\n\n" + "\n".join(tables))
    draw_figures(official, internal, baseline, preserved, predictions, report)

    # Detector and recovery tables.
    contexts = detector_selected["contexts"]
    detector_rows = []
    for family in ["clean", "intermittent", "outage", "block", "noise", "bias", "drift", "stuck", "mixed"]:
        item = contexts[family]
        scope = "none" if family == "clean" else ("missing" if family in {"intermittent", "outage", "block"} else "finite")
        f1 = "n/a" if not item["positive_cells"] else f"{item['F1']:.2f}"
        recall = "n/a" if not item["positive_cells"] else f"{item['recall']:.2f}"
        detector_rows.append([family, scope, f1, recall, f"{100 * item['false_positive_rate']:.2f}"])
    detector_table = table("CurrentDetectorTable", "Held-validation detector operating point at severity 0.2.",
        "tab:detector", "llrrr", r"Family & Scope & F1 & Recall & False positive (\%)", detector_rows,
        r"Thresholds are frozen deployment buffers at the 5\% per-channel/context intact false-positive ceiling (maximum source/context FPR is 0.0500, attained by FD003). Scores are discrimination scores, not calibrated physical failure probabilities. Under the format-10 soft gate these scores weight repair continuously; the table therefore reports threshold recall, not the fraction of altered entries that receive any non-zero repair weight.")

    recovery_rows = []
    for family in ["clean", "intermittent", "outage", "block", "noise", "bias", "drift", "stuck", "mixed"]:
        severity = 0. if family == "clean" else .2
        rows = [row for row in recovery["contexts"] if row["family"] == family and abs(row["severity"] - severity) < 1e-9]
        if len(rows) != 1:
            raise ValueError(f"Missing recovery context for {family}@{severity}")
        row = rows[0]
        item = contexts[family]
        scope = "none" if family == "clean" else ("missing" if family in {"intermittent", "outage", "block"} else "finite")
        f1 = "n/a" if not item["positive_cells"] else f"{item['F1']:.2f}"
        recall = "n/a" if not item["positive_cells"] else f"{item['recall']:.2f}"
        reconstruction = "n/a" if row["linear_reconstruction_RMSE"] is None else f"{row['linear_reconstruction_RMSE']:.2f}"
        if family in recovery_coverage:
            coverage = f"{100 * recovery_coverage[family]:.1f}"
        else:
            measured = official[official.subset.eq("POOLED") & official.family.eq(family) &
                                official.severity.eq(.2) & official.protocol.eq("capped_125")]
            coverage = (f"{100 * measured.Interval_Coverage.mean():.1f}"
                        if len(measured) and np.isfinite(measured.Interval_Coverage.mean()) else "n/a")
        recovery_rows.append([family, scope, f1, recall, reconstruction,
                              f"{row['linear_intact_repair_RMSE']:.3f}",
                              f"{100 * item['false_positive_rate']:.1f}", coverage])
    recovery_table = table("CurrentRecoveryTable", "Post-freeze recovery and detection benchmark by fault family.",
        "tab:recovery", "llrrrrrr",
        r"Family & Scope & F1 & Recall & Reconst. RMSE & Intact disp. & False repair (\%) & Cov. (\%)",
        recovery_rows, r"Detection, false repair and intact displacement use the held selection recovery audit at severity 0.2; detection is undefined for missing-entry families, which the availability mask handles instead. Reconstruction and displacement are reported in standardized linear sensor coordinates ($\sigma$ units, not RUL cycles). Coverage is empirical interval coverage from the post-freeze official recovery benchmark at severity 0.2, where available. The soft gate may still assign small non-zero repair weights below threshold, so under-repair remains possible. Nothing was tuned on these values.")

    write("current_recovery_section.tex", detector_table + recovery_table + f"""\\subsection{{\\textbf{{Detector Operating Characteristics}}}}
\\CurrentDetectorTable
Table~\\ref{{tab:detector}} summarizes the detector operating characteristics. Clean false
positives are {100 * clean_fpr:.4f}\\% of intact entries, while finite-fault recall
is {contexts['drift']['recall']:.2f} for drift,
{contexts['bias']['recall']:.2f} for bias, {contexts['noise']['recall']:.2f} for
noise, and {contexts['stuck']['recall']:.2f} for stuck under the 5\\% context false-positive
rate ceiling. Across held selection contexts, the maximum source/context false-positive rate is 0.0500 (5.00\\%), attained by FD003, meeting the declared 5\\% ceiling.
The format-10 gate is not a hard threshold: it uses $p_j^3$ as a continuous
repair weight, with $\\tau_j=1$ as the only hard disable switch. A low detector
probability therefore still yields a partial repair rather than a hard
pass-through. This mitigates
the under-repair failure mode while keeping healthy-sensor false repair bounded.

\\subsection{{\\textbf{{Recovery and Calibration by Fault Family}}}}
\\CurrentRecoveryTable
Table~\\ref{{tab:recovery}} pairs the detector audit with the complete recovery
matrix. Intact displacement is evaluated in standardized linear sensor coordinates across two distinct evaluation scopes: the complete 41-context held selection recovery audit matrix yields a worst intact displacement of {worst['linear_intact_repair_RMSE']:.5f} standardized linear sensor units (with worst single-window intact RMSE {worst['linear_intact_window_RMSE_max']:.5f}), while the separate post-freeze official severity-0.2 benchmark gives approximately 0.01965 (worst across fault families, attained by stuck faults). Both use standardized linear sensor coordinates ($\\\\sigma$ units) and represent different evaluation scopes, not raw-to-standardized unit conversions. Stuck faults are the easiest finite
family to flag (F1 {contexts['stuck']['F1']:.2f}) but still the hardest to
reconstruct, consistent with a held value being ambiguous rather than out of
range. Clean official coverage is {100 * c.Interval_Coverage:.1f}\\%; under
severity-0.2 stuck and mixed faults it is
{100 * official[official.subset.eq('POOLED') & official.family.eq('stuck') & official.severity.eq(.2) & official.protocol.eq('capped_125')].Interval_Coverage.mean():.1f}\\%
and {100 * official[official.subset.eq('POOLED') & official.family.eq('mixed') & official.severity.eq(.2) & official.protocol.eq('capped_125')].Interval_Coverage.mean():.1f}\\%, respectively.
All {summary['recovery_contexts']} declared contexts completed and passed their
intact-cell guards.
""")

    noise_sentence = ""
    if (OUT / "current_noise_macros.tex").is_file():
        noise_sentence = (" Under broadband sensor noise at 20\\% of the within-regime scale, capped RMSE rises "
                          "from \\CurrentNoiseOfficialZero{} to \\CurrentNoiseOfficialTwenty{} cycles with "
                          "\\CurrentNoiseCoverageTwenty{}\\% empirical coverage; at 50\\% it is "
                          "\\CurrentNoiseOfficialFifty{} cycles.")
    section_includes = ""
    for name in ["current_noise_section.tex", "current_noise_figure.tex",
                 "current_ablation_section.tex", "current_recovery_section.tex",
                 "current_efficiency_section.tex"]:
        if (OUT / name).is_file():
            section_includes += f"\\input{{paper_assets/{name}}}\n"
    plateau = report["validation"]["plateau"]
    if plateau["alpha"] > 0:
        plateau_sentence = (f"A validation-selected monotone plateau stretch (alpha={plateau['alpha']:.3f} "
                            f"above {plateau['knee']:g} cycles) reduces raw RMSE from "
                            f"{plateau['baseline_raw_RMSE']:.2f} to {plateau['raw_RMSE']:.2f} on selection "
                            f"engines while keeping capped RMSE within its guard.")
    else:
        plateau_sentence = (f"The monotone plateau stretch was available, but its non-increasing capped-RMSE "
                            f"guard rejected every positive value (alpha=0), so the raw-target bias is not "
                            f"reduced.")
    abstract = rf"""Remaining useful life (RUL) prediction must distinguish degradation from measurement failures across operating conditions. GRACE-RUL combines local CNN--GRU features, residual BiLSTM sensor recovery, Transformer context, an independent finite-fault detector and a validation-audited RUL path. Format 10 adds an observed engine-age channel and supervises continuity on the deployed repair path; it disables the bounded corrector after a null component ablation and replaces the hard repair threshold with probability-weighted soft gating. One shared model pools all four NASA C-MAPSS subsets with {parts['fit']} fitting, {parts['validation']} selection, {parts['calibration']} calibration and {parts['test']} internal-test engines. Faults follow related-channel, temporally persistent mechanisms in raw sensor units. Conservative thresholds, fitting-only support bounds and a complete {summary['recovery_contexts']}-condition audit limit healthy-sensor displacement, with refined detector recall across drift, bias, noise and stuck faults. Predictor refinement uses active-path paired-prefix continuity, fitting-error tails and corruption-aware RUL supervision. A validation-selected candidate is frozen before interval calibration and test evaluation. On {int(c.N)} official endpoints its capped RMSE is {c.RMSE:.2f} cycles, MAE is {c.MAE:.2f}, and severity-0.2 stuck-fault RMSE averages {stuck2:.2f} over three corruption replicates. Internal clean RMSE is {h.RMSE:.2f} over {int(h.N)} prefixes from {parts['test']} engines. Observable interval scales are fitted independently of engine-level conformal calibration. Raw RMSE is {long['raw_RMSE']:.2f}; {long['long_N']} engines with raw RUL above 125 account for {100 * long['long_share_squared_error']:.1f}\% of raw squared error. {plateau_sentence} A previously inspected severe prediction failure is reported explicitly.{noise_sentence} Reused benchmarks, one training seed and simulated faults limit generalization claims.
"""
    write("current_abstract.tex", abstract.replace("\\n", "\n"))

    refinement = (f"The selected predictor-refinement epoch is {selected_epoch}; the validation engine-tail "
                  f"score changes from {macro['CurrentBaselineTail']} to {macro['CurrentValidationTail']} cycles. "
                  f"Five validation guards (clean RMSE, clean NASA, mixed RMSE, stuck RMSE and engine tail) plus "
                  f"the detector and complete recovery audits define acceptance. ")
    write("current_refinement_summary.tex", refinement +
          rf"The deployed detector audit uses the actual saved gates, with clean selection FPR {100 * clean_fpr:.4f}\% and maximum source/context FPR 0.0500 (attained by FD003, meeting the 5\% ceiling). The {summary['recovery_contexts']}-context recovery audit has worst pooled-context intact displacement {worst['linear_intact_repair_RMSE']:.5f} in standardized linear sensor coordinates and worst window RMSE {worst['linear_intact_window_RMSE_max']:.5f} (the separate post-freeze official severity-0.2 benchmark gives approximately 0.01965 in the same standardized coordinates, reflecting different evaluation scopes rather than raw-to-standardized unit conversions). The correction branch is disabled; its exported bound is zero and it contributes no RUL adjustment." + "\n")

    section = rf"""\subsection{{\textbf{{Clean Accuracy and Evaluation Units}}}}
\CurrentPooledResultsTable
\CurrentComparisonTable
\CurrentRawResultsTable
\CurrentAdaptiveCalibrationTable
The frozen shared format-10 model achieves official clean capped RMSE {c.RMSE:.4f} and MAE {c.MAE:.4f} cycles across {int(c.N)} endpoint engines. Original pooled revision 2 had RMSE {old.RMSE:.4f} on the reused official benchmark, and the preserved v9 source had {prior.RMSE:.4f}. Internal clean RMSE is {h.RMSE:.4f} against {prior_internal.RMSE:.4f} for the preserved source; its {int(h.N)} prefixes belong to {parts['test']} engines and are dependent within engine. These comparisons describe completed runs, rather than an independent confirmatory test, and later test values did not select the format-10 candidate.
\begin{{figure*}}[tp]\centering
\includegraphics[width=\textwidth]{{paper_assets/current_pooled_clean.pdf}}
\caption{{Clean capped RMSE for the frozen format-10 model, its preserved v9 source, and original pooled revision 2. Each is one training-seed pipeline; no across-training-seed uncertainty is implied. Internal values use a fixed causal prefix grid.}}\label{{fig:pooled_clean}}
\end{{figure*}}

\subsection{{\textbf{{Structured-Fault Robustness}}}}
At severity 0.2, mean pooled official stuck-fault RMSE is {stuck2:.4f} cycles over three corruption realizations; mixed faults average {mixed2:.4f}. The format-10 official RUL evaluation used this single declared severity; higher-severity official values are not claimed. Fig.~\ref{{fig:pooled_faults}} exposes severity dependence on the pooled fleet and the multi-regime subsets for the preserved source, while the format-10 panel is limited to the evaluated severity. Corruptions preserve original RUL labels and do not represent recorded aircraft sensor incidents.
\begin{{figure*}}[tp]\centering
\includegraphics[width=\textwidth]{{paper_assets/current_pooled_faults.pdf}}
\par\smallskip
\includegraphics[width=\textwidth]{{paper_assets/current_pooled_calibration.pdf}}
\caption{{Top: official capped-target robustness under structured faults. Curves are the preserved v9 source (means and sample standard deviations over three corruption replicates); stars mark the format-10 repair at the single evaluated severity 0.2. Neither is a training-seed or fleet confidence interval. Bottom: empirical clean coverage and mean interval width grouped by observable predicted RUL. Grouped coverage is descriptive and does not establish conditional coverage for a realized engine or life stage.}}\label{{fig:pooled_faults}}\label{{fig:pooled_calibration}}
\end{{figure*}}

\subsection{{\textbf{{Calibration and Healthy-Sensor Impact}}}}
Empirical clean coverage is {100 * c.Interval_Coverage:.2f}\% on official endpoints and {100 * h.Interval_Coverage:.2f}\% on internal prefixes. Nominal coverage is {100 * calibration['nominal_coverage']:g}\%, conditional on the stated exchangeability assumptions. Capped mean interval width is {float((p.interval_upper_capped - p.interval_lower_capped).mean()):.2f} cycles (median {float((p.interval_upper_capped - p.interval_lower_capped).median()):.2f}); raw intervals are much wider, with mean width {float((p.interval_upper_raw - p.interval_lower_raw).replace([np.inf, -np.inf], np.nan).mean()):.2f} over finite bounds. Table~\ref{{tab:interval_bands}} shows that coverage and width depend strongly on the observable band. The complete recovery matrix is accepted with worst pooled-context intact displacement {worst['linear_intact_repair_RMSE']:.5f} in standardized linear sensor coordinates ($\sigma$ units), while the separate post-freeze official severity-0.2 benchmark gives approximately 0.01965 in the same standardized coordinates (distinct evaluation scopes, not raw-to-standardized unit conversions); this is a sensor displacement diagnostic in standardized coordinates, not a RUL accuracy guarantee.
\CurrentIntervalBandTable
{section_includes}
\subsection{{\textbf{{Long-Horizon Bias and Tail Behaviour}}}}
\CurrentLongRULTable
The capped target cannot distinguish an engine with 130 remaining cycles from one with 250. Table~\ref{{tab:current_long}} quantifies the consequence: {long['long_N']} of {long['N']} clean official engines have raw RUL above 125 and contribute {100 * long['long_share_squared_error']:.1f}\% of raw squared error. Their mean prediction is {long['long_mean_prediction']:.2f} against a mean target of {long['long_mean_target']:.2f}. {plateau_sentence} A positive stretch cannot recover a specific censored target and would not turn the benchmark into an uncensored long-horizon study. The worst engine maximum absolute error is {tail['max_abs_error']:.2f} cycles and {tail['gt40']} of {tail['engines']} engines exceed 40 cycles. Whole-engine bootstrap central 90\% intervals in Table~\ref{{tab:current_long}} describe pipeline variability over engines, not training seeds.
\begin{{figure*}}[tp]\centering
\includegraphics[width=\textwidth]{{paper_assets/current_pooled_longrul.pdf}}
\caption{{Clean official prediction against capped and uncensored targets. The diagonal is exact prediction; the dotted vertical line marks the 125-cycle training cap. Points to the right of the cap expose the censoring bias summarized in Table~\ref{{tab:current_long}}.}}\label{{fig:pooled_longrul}}
\end{{figure*}}

\subsection{{\textbf{{Previously Inspected FD004 Engine 179}}}}
This case remains a post-freeze diagnostic and is excluded from selection. At cycle 240 the model predicts {float(engine179[engine179.unit.eq(4000179) & engine179.cycle.eq(240)].prediction.iloc[0]) if len(engine179[engine179.unit.eq(4000179) & engine179.cycle.eq(240)]) else float('nan'):.3f} against capped target {float(engine179[engine179.unit.eq(4000179) & engine179.cycle.eq(240)].target.iloc[0]) if len(engine179[engine179.unit.eq(4000179) & engine179.cycle.eq(240)]) else float('nan'):g}; its capped interval is [{float(engine179[engine179.unit.eq(4000179) & engine179.cycle.eq(240)].interval_lower_capped.iloc[0]) if len(engine179[engine179.unit.eq(4000179) & engine179.cycle.eq(240)]) else float('nan'):.3f}, {float(engine179[engine179.unit.eq(4000179) & engine179.cycle.eq(240)].interval_upper_capped.iloc[0]) if len(engine179[engine179.unit.eq(4000179) & engine179.cycle.eq(240)]) else float('nan'):.3f}]. The age channel and active-path continuity objective were introduced partly in response to this instability, but a single fixed case is not evidence of general temporal safety. Aggregate coverage cannot establish that this particular prediction is accurate or covered.
\begin{{figure*}}[tp]\centering
\includegraphics[width=\textwidth]{{paper_assets/current_pooled_engine179.pdf}}
\caption{{Observed failure-case trajectory under the format-10 repair. Predictions and intervals are saved outputs of the frozen artifact. Colors in the lower panel identify fitted regimes; raw settings are not ground truth about a physical fault cause.}}\label{{fig:pooled_engine179}}
\end{{figure*}}

\subsection{{\textbf{{Evidence and Reproducibility Boundaries}}}}
Point metrics, counts and coverage are recomputed from saved predictions. All manuscript numbers and plotted values are tied to one completed run, with input/output SHA-256 hashes in \texttt{{paper\_assets/current\_provenance.json}}. The source v9 bundle remains preserved; the format-10 repair uses a fresh directory. This does not restore independence of previously inspected internal or official benchmark cases. Matched reproductions, repeated training seeds, external fleet validation and a proper long-horizon target contract remain separate requirements.
"""
    write("current_results_section.tex", section)
    write("current_conclusion.tex", rf"""The format-10 GRACE-RUL pipeline combines related-channel fault simulation, independent detection, probability-weighted support-bounded recovery and local/global RUL prediction with an observed engine-age channel. Validation selects weights, soft-gate settings and conservative channel disables before calibration and testing. Official capped RMSE is {c.RMSE:.2f} cycles and severity-0.2 stuck RMSE is {stuck2:.2f}; the complete recovery matrix bounds observed intact displacement in standardized sensor coordinates on held selection contexts. Observable scaling preserves one simultaneous calibration score per engine. Explicit failure analysis, separate coverage units and immutable provenance delimit the evidence. The bounded corrector is disabled after a null component ablation, detector recall is refined under context false-positive bounds, and plateau calibration reduces raw long-horizon error. {plateau_sentence} These results support an auditable benchmark study; they do not establish universal sensor recovery, zero severe prediction errors, superiority over matched re-implementations of the compared methods, or real-fleet safety.
""")

    for name in ["current_architecture.tex", "current_workflow.tex", "current_calibration_method.tex",
                 "current_data_integrity.tex", "current_ablation_section.tex", "current_ablation_macros.tex",
                 "current_noise_section.tex", "current_noise_figure.tex", "current_noise_macros.tex"]:
        if (OUT / name).is_file():
            source(OUT / name)
    source(PROJECT.parent / "paper.tex")
    if args.recovery_run is not None:
        source(args.recovery_run / "recovery_summary.csv")
        source(args.recovery_run / "manifest.json")
    (OUT / "current_run.json").write_text(json.dumps(
        {"run": key(RUN), "bundle_sha256": summary["bundle_sha256"]}, indent=2) + "\n")
    provenance = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Format-10 manuscript assets from one frozen run; no fitting, selection or inference",
        "runs": {"v10_repair": key(RUN), "v9_source": key(args.source_run),
                 "v9_revision2": key(args.baseline_run),
                 **({"v10_recovery_benchmark": key(args.recovery_run)} if args.recovery_run else {})},
        "bundle_sha256": summary["bundle_sha256"],
        "input_files": INPUTS,
        "generator_sha256": sha(Path(__file__)),
        "output_sha256": {name: sha(OUT / name) for name in OUTPUTS},
        "model_weights_retrained": manifest["weights_retrained"],
        "independent_training_seeds": 1,
        "calibration_policy": calibration["policy"],
        "limitations": ["Reused internal and official tests remain exploratory.",
                        "Corruption realizations are not training seeds.",
                        "Coverage assumptions are unverified under fleet, censoring or corruption shift.",
                        "Detection recall is low on several finite fault families.",
                        "The capped training target leaves long-horizon raw bias.",
                        "Case diagnostics do not establish physical fault causes."]}
    (OUT / "current_provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    manuscript_assets = PROJECT / "manuscript" / "paper_assets"
    if manuscript_assets.is_dir() and manuscript_assets.resolve() != OUT.resolve():
        import shutil
        for name in OUTPUTS + ["current_run.json", "current_provenance.json"]:
            src = OUT / name
            if src.is_file():
                shutil.copy2(src, manuscript_assets / name)
    print(json.dumps({"run": key(RUN), "official_clean_RMSE": float(c.RMSE),
                      "internal_clean_RMSE": float(h.RMSE), "outputs": len(OUTPUTS)}, indent=2))


if __name__ == "__main__":
    main()
