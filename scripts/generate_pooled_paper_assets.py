"""Generate the current pooled paper from one completed frozen numerical run.

No fitting or inference occurs here. Every reported metric, bin radius, count
and displayed failure-case prediction is read from hashed CSV/JSON evidence.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
OUT = ROOT / "paper_assets"
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir())/"grace-pooled-paper-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":7.2,
    "axes.titlesize":8., "axes.labelsize":7.2, "legend.fontsize":6.5,
    "xtick.labelsize":6.7, "ytick.labelsize":6.7, "pdf.fonttype":42,
    "axes.spines.top":False, "axes.spines.right":False, "axes.grid":True,
    "grid.alpha":.16, "axes.axisbelow":True})
INPUTS, OUTPUTS = {}, []


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(path):
    p = Path(path).resolve()
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def source(path):
    path = Path(path)
    INPUTS[key(path)] = {"sha256":sha(path), "bytes":path.stat().st_size}
    return path


def read_json(path):
    return json.loads(source(path).read_text())


def csv(path):
    return pd.read_csv(source(path))


def write(name, content):
    (OUT/name).write_text(content)
    OUTPUTS.append(name)


def one(frame, **query):
    selected = frame
    for column, value in query.items():
        selected = selected[selected[column].eq(value)]
    if len(selected) != 1:
        raise ValueError(f"Expected one saved metric for {query}; found {len(selected)}")
    return selected.iloc[0]


def table(command, caption, label, columns, header, rows, note):
    return "\n".join([rf"\newcommand{{\{command}}}{{%",r"\begin{table*}[!t]",
        rf"\caption{{{caption}}}",rf"\label{{{label}}}",r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",rf"\begin{{tabular}}{{{columns}}}",r"\toprule",
        header+r" \\",r"\midrule",*[" & ".join(map(str,row))+r" \\" for row in rows],
        r"\bottomrule",r"\end{tabular}",r"\par\vspace{2pt}",
        rf"\begin{{minipage}}{{\textwidth}}\scriptsize {note}\end{{minipage}}",
        r"\end{table*}","}",""])


def save(fig, stem):
    for suffix in ["pdf","png"]:
        name = f"current_{stem}.{suffix}"
        meta = {"Title":f"GRACE-RUL {stem}","CreationDate":None,"ModDate":None} if suffix=="pdf" else None
        fig.savefig(OUT/name,bbox_inches="tight",pad_inches=.04,dpi=240,metadata=meta)
        OUTPUTS.append(name)
    plt.close(fig)


def draw_figures(official, internal, baseline, preserved, run, predictions):
    subsets = ["FD001","FD002","FD003","FD004","POOLED"]
    fig, ax = plt.subplots(1,2,figsize=(7.12,2.35))
    fig.subplots_adjust(top=.76,bottom=.22,wspace=.30)
    for axis, partition, title in [(ax[0],official,"Official clean endpoints"),(ax[1],internal,"Internal clean prefixes")]:
        old = baseline[baseline.partition.eq("official_benchmark" if partition is official else "internal_test_10pct")]
        prior = preserved[preserved.partition.eq("official_benchmark" if partition is official else "internal_test_10pct")]
        x=np.arange(len(subsets))
        for data,offset,label,color in [(old,-.25,"Original revision 2","#91a1ad"),(prior,0.,"Preserved fine-tuned source","#d5a16d"),(partition,.25,"Current frozen model","#187e83")]:
            values=[one(data,subset=s,family="clean",protocol="capped_125").RMSE for s in subsets]
            axis.bar(x+offset,values,.24,label=label,color=color)
        axis.set(xticks=x,xticklabels=subsets,ylabel="Capped RMSE (cycles)",title=title)
    handles,labels=ax[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.52,1.02),ncol=3,frameon=False)
    save(fig,"pooled_clean")

    fig,axes=plt.subplots(1,3,figsize=(7.12,2.35),constrained_layout=True)
    for axis,subset in zip(axes,["POOLED","FD002","FD004"]):
        for family,color in [("stuck","#2864a8"),("mixed","#d47a29"),("noise","#187e83")]:
            values=official[official.subset.eq(subset)&official.family.eq(family)&official.protocol.eq("capped_125")]
            grouped=values.groupby("severity").RMSE.agg(["mean","std"])
            if len(grouped):
                x=grouped.index.to_numpy(float);y=grouped["mean"].to_numpy(float);sd=grouped["std"].fillna(0).to_numpy(float)
                axis.plot(x,y,"o-",ms=2.5,lw=1.2,color=color,label=family)
                axis.fill_between(x,y-sd,y+sd,color=color,alpha=.12,lw=0)
        axis.set(title=subset,xlabel="Fault severity",ylabel="Capped RMSE (cycles)",xticks=[.1,.2,.3,.4,.5])
    axes[0].legend(frameon=False)
    save(fig,"pooled_faults")

    fig,axes=plt.subplots(1,2,figsize=(7.12,2.35))
    fig.subplots_adjust(top=.76,bottom=.23,wspace=.30)
    labels=[r"$\hat y<35$","35--75","75--125",r"$\hat y\geq125$"]
    ranges=[(0,35),(35,75),(75,125),(125,float("inf"))]
    for name,color,offset in [("official_benchmark","#187e83",-.18),("internal_test_10pct","#d47a29",.18)]:
        p=predictions[name];coverage=[];width=[]
        for low,high in ranges:
            q=p[(p.prediction>=low)&(p.prediction<high)]
            coverage.append(100*float(((q.target>=q.interval_lower_capped)&(q.target<=q.interval_upper_capped)).mean()) if len(q) else np.nan)
            width.append(float((q.interval_upper_capped-q.interval_lower_capped).mean()) if len(q) else np.nan)
        axes[0].bar(np.arange(4)+offset,coverage,.35,color=color,label="Official endpoints" if name.startswith("official") else "Internal prefixes")
        axes[1].bar(np.arange(4)+offset,width,.35,color=color)
    axes[0].axhline(90,color="#555555",ls="--",lw=.8,label="Nominal engine coverage")
    for axis in axes:axis.set(xticks=np.arange(4),xticklabels=labels,xlabel="Observable predicted RUL bin (cycles)")
    axes[0].set(ylabel="Empirical prefix coverage (%)",ylim=(0,105),title="Capped-target coverage")
    handles,legend_labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc="upper center",bbox_to_anchor=(.52,1.02),ncol=3,frameon=False)
    axes[1].set(ylabel="Mean interval width (cycles)",title="Adaptive interval widths")
    save(fig,"pooled_calibration")

    p=predictions["internal_test_10pct"];p=p[p.unit.eq(4000179)].sort_values("cycle")
    trace=csv(run/"engine179_raw_trace.csv")
    fig,axes=plt.subplots(2,1,figsize=(7.12,2.65),sharex=True,gridspec_kw={"height_ratios":[2,1]})
    fig.subplots_adjust(top=.78,bottom=.17,hspace=.13)
    axes[0].fill_between(p.cycle,p.interval_lower_capped,p.interval_upper_capped,color="#187e83",alpha=.15,label="Current capped interval")
    axes[0].plot(p.cycle,p.target,color="#555555",ls="--",lw=1.2,label="Capped target")
    axes[0].plot(p.cycle,p.prediction,"o-",ms=3,color="#187e83",label="Current prediction")
    case=p[p.cycle.eq(240)]
    if len(case):
        row=case.iloc[0];axes[0].annotate(f"Cycle 240: {row.prediction:.2f} / target {row.target:g}",
            (row.cycle,row.prediction),xytext=(105,25),textcoords="data",arrowprops={"arrowstyle":"->","color":"#444444"},fontsize=6.5)
    axes[0].set(title="Previously inspected FD004 engine 179: post-freeze diagnostic",ylabel="Capped RUL (cycles)")
    handles,legend_labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc="upper center",bbox_to_anchor=(.52,1.02),ncol=3,frameon=False,fontsize=6)
    axes[1].scatter(trace.cycle,trace.setting_1,c=trace.regime,cmap="tab10",s=5)
    axes[1].set(ylabel="Raw setting 1",xlabel="Observed cycle")
    save(fig,"pooled_engine179")


def main(argv=None):
    global ROOT, OUT
    INPUTS.clear(); OUTPUTS.clear()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path)
    parser.add_argument("--paper-root",type=Path,default=ROOT)
    parser.add_argument("--baseline-run",type=Path,default=PROJECT/"artifacts/pooled_v9_r2_20261029_114845")
    args=parser.parse_args(argv)
    ROOT=args.paper_root.resolve(); OUT=ROOT/"paper_assets"
    pointer=OUT/"current_run.json"
    run=args.run or (ROOT/json.loads(pointer.read_text())["run"] if pointer.exists() else None)
    if run is None:raise ValueError("Specify --run for a completed repaired model before generating the paper.")
    run=run.resolve();summary=read_json(run/"paper_model_summary.json");manifest=read_json(run/"run_manifest.json")
    if manifest.get("status")!="complete" or summary["bundle_sha256"]!=manifest["frozen_bundle_sha256"]:
        raise ValueError("Paper source must be a completed, hash-consistent frozen run.")
    meta=summary["config"];calibration=summary["calibration"]
    if calibration["policy"]!="observable_scaled_engine_max":
        raise ValueError("This manuscript method requires the repaired simultaneous observable-scale contract.")
    official=csv(run/"official_benchmark_metrics.csv");internal=csv(run/"internal_test_10pct_metrics.csv")
    baseline=pd.concat([csv(args.baseline_run/f"{p}_metrics.csv") for p in ["official_benchmark","internal_test_10pct"]],ignore_index=True)
    preserved_run=Path(manifest["source_bundle"]).parent
    preserved=pd.concat([csv(preserved_run/f"{p}_metrics.csv") for p in ["official_benchmark","internal_test_10pct"]],ignore_index=True)
    predictions={p:csv(run/f"predictions/{p}_clean_s0_r0.csv.gz") for p in ["official_benchmark","internal_test_10pct"]}
    audit=read_json(run/"active_path_validation.json")
    c=one(official,subset="POOLED",family="clean",protocol="capped_125")
    h=one(internal,subset="POOLED",family="clean",protocol="capped_125")
    old=one(baseline,partition="official_benchmark",subset="POOLED",family="clean",protocol="capped_125")
    prior=one(preserved,partition="official_benchmark",subset="POOLED",family="clean",protocol="capped_125")
    subsets=["FD001","FD002","FD003","FD004","POOLED"]
    parts=summary["partition_counts"];detector=summary["detector_audit"];worst=summary["recovery_worst"]
    counts=summary["parameter_counts"]
    macro={"CurrentFitEngines":parts["fit"],"CurrentSelectionEngines":parts["validation"],
           "CurrentCalibrationEngines":parts["calibration"],"CurrentTestEngines":parts["test"],
           "CurrentOfficialEngines":int(c.engines),"CurrentInternalPrefixes":int(h.N),
           "CurrentOfficialRMSE":f"{c.RMSE:.2f}","CurrentInternalRMSE":f"{h.RMSE:.2f}",
           "CurrentOfficialMAE":f"{c.MAE:.2f}","CurrentRegimeCount":summary["regimes"],
           "CurrentGuardCount":len(audit["guards"]),"CurrentNeuralParameters":sum(counts.values()),
           "CurrentRecoveryContexts":summary["recovery_contexts"],"CurrentBundleHash":summary["bundle_sha256"][:16],
           "CurrentLocalLength":meta["local_length"],"CurrentGlobalLength":meta["sequence_length"],
           "CurrentIntactRMSE":f"{worst['linear_intact_repair_RMSE']:.5f}",
           "CurrentCleanFPR":f"{100*detector['selected']['clean_false_positive_rate']:.4f}",
           "CurrentValidationTail":f"{audit['selected_engine_tail']:.3f}",
           "CurrentBaselineTail":f"{audit['baseline_engine_tail']:.3f}"}
    macro.update({f"Current{name.capitalize()}Parameters":value for name,value in counts.items()})
    write("current_run_macros.tex","% Data-derived macros; no desired scores are assigned.\n"+"\n".join(rf"\newcommand{{\{name}}}{{{value}}}" for name,value in macro.items())+"\n")
    official_counts={s:int(one(official,subset=s,family="clean",protocol="capped_125").engines) for s in subsets}
    tables=[table("CurrentPartitionTable","Current pooled engine-disjoint data partitions.","tab:pooled_partitions","lrrrrrr",
        r"Subset & Available training & Fit & Selection & Calibration & Internal test & Official test",
        [[r["subset"],r["train"],r["fit"],r["validation"],r["calibration"],r["test"],official_counts[r["subset"]]] for r in summary["dataset_counts"]],
        r"The 10\% development pool is divided equally between selection and independent calibration; integer rounding gives the exact counts above. Internal observations are fixed-grid prefixes, not independent engines.")]
    rows=[]
    for s in ["POOLED",*subsets[:-1]]:
        x=one(official,subset=s,family="clean",protocol="capped_125");y=one(internal,subset=s,family="clean",protocol="capped_125")
        faults={f:official[official.subset.eq(s)&official.family.eq(f)&official.severity.eq(.2)&official.protocol.eq("capped_125")].RMSE.mean() for f in ["stuck","mixed"]}
        rows.append([s,int(x.N),f"{x.RMSE:.2f}",f"{x.MAE:.2f}",f"{x.NASA_Score:.1f}",f"{100*x.Interval_Coverage:.1f}",f"{faults['stuck']:.2f}",f"{faults['mixed']:.2f}",f"{y.RMSE:.2f}"])
    tables.append(table("CurrentPooledResultsTable","Current format-9 model: capped-target clean and structured-fault results.","tab:current_pooled","lrrrrrrrr",
        r"Subset & $N_{\rm off}$ & RMSE & MAE & NASA $S$ & Cov. (\%) & Stuck & Mixed & Internal RMSE",rows,
        r"Fault columns average three corruption replicates at severity 0.2; these are not training-seed replications. Coverage is empirical clean prefix/endpoint coverage. Internal RMSE uses all saved prefixes from each source's reserved engines. Targets are $\min(r,125)$; units are cycles except NASA score and coverage."))
    comparison=[]
    for label,values in [("Original revision 2",baseline),("Preserved fine-tuned source",preserved),("Current frozen repair",pd.concat([official,internal],ignore_index=True))]:
        x=one(values,partition="official_benchmark",subset="POOLED",family="clean",protocol="capped_125")
        y=one(values,partition="internal_test_10pct",subset="POOLED",family="clean",protocol="capped_125")
        faults=values[values.partition.eq("official_benchmark")&values.subset.eq("POOLED")&values.family.eq("stuck")&values.protocol.eq("capped_125")]
        comparison.append([label,f"{x.RMSE:.3f}",f"{x.MAE:.3f}",f"{y.RMSE:.3f}",f"{faults[faults.severity.eq(.2)].RMSE.mean():.3f}",f"{faults.RMSE.mean():.3f}"])
    tables.append(table("CurrentComparisonTable","Pooled comparison including the immediate preserved source.","tab:current_comparison","lrrrrr",
        r"Artifact & Official clean RMSE & MAE & Internal clean RMSE & Stuck @0.2 & Stuck all levels",comparison,
        r"All rows use their own saved predictions and capped targets. Fault values are arithmetic means across declared corruption realizations; the last column also averages severities. Reused test results are descriptive and do not select the current candidate."))
    rawrows=[]
    for s in ["POOLED",*subsets[:-1]]:
        x=one(official,subset=s,family="clean",protocol="official_uncapped")
        rawrows.append([s,f"{x.RMSE:.2f}",f"{x.MAE:.2f}",f"{x.NASA_Score:.1f}",f"{100*x.Interval_Coverage:.1f}"])
    tables.append(table("CurrentRawResultsTable","Clean official results against original, uncapped RUL.","tab:current_raw","lrrrr",
        r"Subset & Raw RMSE & Raw MAE & Raw NASA $S$ & Raw coverage (\%)",rawrows,
        r"The point model was trained on capped targets. Raw errors and separately calibrated raw intervals expose long-horizon limitations; capped coverage must not be presented as raw-target coverage."))
    bins=calibration["protocols"]["capped_125"]["stratified_bins"];raw_bins=calibration["protocols"]["official_uncapped"]["stratified_bins"]
    calrows=[]
    for b,r in zip(bins,raw_bins):
        label=rf"$[{b['lo']:g},{b['hi']:g})$" if b["hi"] is not None else rf"$[{b['lo']:g},\infty)$"
        calrows.append([label,f"{b['scale']:.2f}",f"{r['scale']:.2f}",f"{b['radius']:.2f}" if b["radius"] is not None else r"$\infty$",f"{r['radius']:.2f}" if r["radius"] is not None else r"$\infty$",b["scale_engines"]])
    tables.append(table("CurrentAdaptiveCalibrationTable","Observable interval scales and radii of the frozen model.","tab:adaptive_calibration","lrrrrr",
        r"Predicted RUL bin & Scale capped & Scale raw & Radius capped & Radius raw & Scale engines",calrows,
        rf"Scales use selection engines only. A single engine-maximum normalized score across all bins is calibrated on {calibration['engines']} separate engines; alpha={calibration['alpha']:g}, rank={calibration['rank']}. Scale-engine counts are overlapping bin support, not independent calibration sample counts. No true RUL is required to choose a bin at inference."))
    write("current_results_tables.tex","% Generated from hashed saved numerical evidence.\n\n"+"\n".join(tables))
    draw_figures(official,internal,baseline,preserved,run,predictions)
    stuck=official[official.subset.eq("POOLED")&official.family.eq("stuck")&official.protocol.eq("capped_125")]
    stuck2=stuck[stuck.severity.eq(.2)].RMSE.mean();stuckall=stuck.RMSE.mean()
    p=predictions["internal_test_10pct"];covered=(p.target>=p.interval_lower_capped)&(p.target<=p.interval_upper_capped)
    engine_coverage=float(covered.groupby(p.unit).all().mean())
    case=p[p.unit.eq(4000179)&p.cycle.eq(240)].iloc[0];rebound=p[p.unit.eq(4000179)&p.cycle.eq(270)].iloc[0]
    trace=csv(run/"engine179_raw_trace.csv");s240=trace[trace.cycle.eq(240)].iloc[0];s245=trace[trace.cycle.eq(245)].iloc[0]
    settings=lambda r:", ".join(f"{r[f'setting_{i}']:g}" for i in range(1,4))
    abstract=rf"""Remaining useful life (RUL) prediction must distinguish degradation from measurement failures across operating conditions. GRACE-RUL integrates local CNN--GRU features, residual BiLSTM sensor recovery, Transformer context, independent finite-fault detection and a bounded supervised corrector. One shared model pools all four NASA C-MAPSS subsets with {parts['fit']} fitting, {parts['validation']} selection, {parts['calibration']} calibration and {parts['test']} internal-test engines. Faults follow related-channel, temporally persistent mechanisms in raw sensor units. Conservative validated thresholds, fitting-only support bounds and a complete {summary['recovery_contexts']}-condition audit limit healthy-sensor displacement. Predictor refinement uses deterministic paired-prefix continuity, fitting-error tail supervision and corruption-aware RUL supervision. A validation-selected candidate is frozen before interval calibration and test evaluation. On {int(c.N)} official endpoints its capped RMSE is {c.RMSE:.2f} cycles, MAE is {c.MAE:.2f}, and severity-0.2 stuck-fault RMSE averages {stuck2:.2f} over three corruption replicates. Internal clean RMSE is {h.RMSE:.2f} over {int(h.N)} prefixes from {parts['test']} engines. Observable interval scales are fitted independently of engine-level conformal calibration. A previously inspected severe prediction failure is reported explicitly. Reused benchmarks, one training seed and simulated faults limit generalization claims.\n"""
    write("current_abstract.tex",abstract.replace("\\n","\n"))
    selection=manifest.get("selection") or {};selected_epoch=selection.get("selected_epoch")
    refinement=rf"The selected predictor-refinement epoch is {selected_epoch}; the validation engine-tail score changes from {audit['baseline_engine_tail']:.3f} to {audit['selected_engine_tail']:.3f} cycles. Every one of the {len(audit['guards'])} final active-path guards passes. " if selected_epoch is not None else "No new predictor weights are claimed in this post-training audit refresh. "
    write("current_refinement_summary.tex",refinement+rf"The deployed detector audit uses the actual saved thresholds, with clean selection FPR {100*detector['selected']['clean_false_positive_rate']:.4f}\%. Its earlier training operating point is retained as historical metadata. The {summary['recovery_contexts']}-context recovery audit has worst pooled-context intact original-linear RMSE {worst['linear_intact_repair_RMSE']:.5f} and worst window RMSE {worst['linear_intact_window_RMSE_max']:.5f}.\n".replace("\\n","\n"))
    if manifest.get("weights_retrained"):
        rate=lambda value: f"{value:.6f}".rstrip("0").rstrip(".")
        recipe=(rf"The reported refinement uses AdamW with learning rate {rate(meta['finetune_learning_rate'])}, "
            rf"weight decay {rate(meta['finetune_weight_decay'])}, batch size {meta['batch_size']}, "
            rf"and a cosine schedule to {rate(meta['finetune_min_learning_rate'])}. "
            rf"The budget is {meta['finetune_epochs']} epochs and {meta['finetune_maximum_windows']} fitting draws per epoch. "
            rf"Clean, auxiliary-branch, prefix, tail and active corrupted-RUL weights are "
            rf"{meta['rul_loss_weight']:g}, {meta['branch_loss_weight']:g}, {meta['prefix_consistency_loss_weight']:g}, "
            rf"{meta['finetune_tail_loss_weight']:g} and {meta['finetune_corrupted_rul_loss_weight']:g}; "
            rf"the tail fraction is {meta['finetune_tail_fraction']:g}. Slew and rebound weights inside the prefix term are "
            rf"{meta['physical_slew_loss_weight']:g} and {meta['physical_rebound_loss_weight']:g}.\n").replace("\\n","\n")
        with (OUT/"current_refinement_summary.tex").open("a") as stream:stream.write(recipe)
    section=rf"""\subsection{{\textbf{{Clean Accuracy and Evaluation Units}}}}
\CurrentPooledResultsTable
\CurrentComparisonTable
\CurrentRawResultsTable
\CurrentAdaptiveCalibrationTable
The frozen shared model achieves official clean capped RMSE {c.RMSE:.4f} and MAE {c.MAE:.4f} cycles across {int(c.N)} endpoint engines. Original pooled revision 2 had RMSE {old.RMSE:.4f} on the reused official benchmark. Internal clean RMSE is {h.RMSE:.4f}; its {int(h.N)} prefixes belong to {parts['test']} engines and are dependent within engine. These comparisons describe completed runs, rather than an independent confirmatory test.
The immediate preserved fine-tuned source had official RMSE {prior.RMSE:.4f}; Table~\ref{{tab:current_comparison}} reports that comparison as well, including any regressions. Validation guards and the held selection-tail objective fixed the current model before these test values were read.
\begin{{figure*}}[tp]\centering
\includegraphics[width=\textwidth]{{paper_assets/current_pooled_clean.pdf}}
\caption{{Clean capped RMSE for the frozen current model, its immediate preserved source, and original pooled revision 2. Each is one training-seed pipeline; no across-training-seed uncertainty is implied. Internal values use a fixed causal prefix grid.}}\label{{fig:pooled_clean}}
\end{{figure*}}
\subsection{{\textbf{{Structured-Fault Robustness}}}}
At severity 0.2, mean pooled official stuck-fault RMSE is {stuck2:.4f} cycles over three corruption realizations. Averaging all declared stuck severities and realizations instead gives {stuckall:.4f}; these summaries must not be interchanged. Fig.~\ref{{fig:pooled_faults}} exposes severity dependence on the pooled fleet and the multi-regime subsets. Corruptions preserve original RUL labels and do not represent recorded aircraft sensor incidents.
\begin{{figure*}}[tp]\centering
\includegraphics[width=\textwidth]{{paper_assets/current_pooled_faults.pdf}}
\par\smallskip
\includegraphics[width=\textwidth]{{paper_assets/current_pooled_calibration.pdf}}
\caption{{Top: official capped-target robustness under structured faults; curves are means and bands are sample standard deviations over three corruption replicates of one frozen model, not training seeds or fleet confidence intervals. Bottom: empirical clean coverage and mean interval width grouped by observable predicted RUL. Grouped coverage is descriptive and does not establish conditional coverage for a realized engine or life stage. The simultaneous calibration score spans all bins for each engine.}}\label{{fig:pooled_faults}}\label{{fig:pooled_calibration}}
\end{{figure*}}
\subsection{{\textbf{{Calibration and Healthy-Sensor Impact}}}}
Empirical clean coverage is {100*c.Interval_Coverage:.2f}\% on official endpoints and {100*h.Interval_Coverage:.2f}\% on internal prefixes. Internal whole-engine coverage over the evaluated grid is {100*engine_coverage:.2f}\%, a different unit from prefix coverage and from every-cycle trajectory coverage. Nominal coverage is {100*calibration['nominal_coverage']:g}\%, conditional on the stated exchangeability assumptions. The complete recovery matrix is accepted with worst pooled-context intact original-linear RMSE {worst['linear_intact_repair_RMSE']:.5f}; this is a sensor displacement diagnostic, not a RUL accuracy guarantee.
\subsection{{\textbf{{Previously Inspected FD004 Engine 179}}}}
This case is a post-freeze diagnostic, excluded from the current selection objective. At cycle 240 the model predicts {case.prediction:.3f} against capped target {case.target:g} (raw RUL {case.raw_target:g}); its capped interval is [{case.interval_lower_capped:.3f}, {case.interval_upper_capped:.3f}]. At cycle 270 it predicts {rebound.prediction:.3f}. Aggregate coverage cannot establish that this particular prediction is accurate or covered. Raw settings at cycles 240 and 245 are [{settings(s240)}] and [{settings(s245)}], respectively, and correspond to different fitted regimes. Consequently the earlier identical-setting physical-excursion explanation is unsupported. The local branch uses {meta['local_length']} cycles and the Transformer uses {meta['sequence_length']}; a local-window-only causal explanation is also incomplete.
\begin{{figure*}}[tp]\centering
\includegraphics[width=\textwidth]{{paper_assets/current_pooled_engine179.pdf}}
\caption{{Observed failure-case trajectory and operating-setting changes. Predictions and intervals are saved outputs of the frozen current artifact. Colors in the lower panel identify fitted regimes; raw settings are not ground truth about a physical fault cause.}}\label{{fig:pooled_engine179}}
\end{{figure*}}
\subsection{{\textbf{{Evidence and Reproducibility Boundaries}}}}
Point metrics, counts and coverage are recomputed from saved predictions. All manuscript numbers and plotted values are tied to a single completed run, with input/output SHA-256 hashes in \texttt{{paper\_assets/current\_provenance.json}}. Asset generation performs no model fitting or inference. The source model and earlier audit directories remain preserved; new artifacts use fresh directories. This does not restore independence of previously inspected internal or official benchmark cases. Matched reproductions of the three supplied papers, repeated training seeds and external fleet validation remain separate requirements.
"""
    write("current_results_section.tex",section)
    write("current_conclusion.tex",rf"The current shared GRACE-RUL pipeline combines related-channel fault simulation, independent detection, bounded recurrent recovery and local/global RUL prediction. Validation selects weights and conservative thresholds before calibration and testing. Official capped RMSE is {c.RMSE:.2f} cycles and severity-0.2 stuck RMSE is {stuck2:.2f}; the full recovery matrix bounds observed intact displacement on held selection contexts. Observable scaling preserves one simultaneous calibration score per engine. Explicit failure analysis, separate coverage units and immutable provenance delimit the evidence. These results support an auditable benchmark study; they do not establish universal sensor recovery, zero severe prediction errors, superiority over matched paper implementations, or real-fleet safety.\n".replace("\\n","\n"))
    for name in ["current_architecture.tex","current_workflow.tex","current_calibration_method.tex","current_data_integrity.tex"]:
        source(OUT/name)
    source(ROOT/"paper.tex")
    pointer.write_text(json.dumps({"run":key(run),"bundle_sha256":summary["bundle_sha256"]},indent=2)+"\n")
    provenance={"created_at_utc":datetime.now(timezone.utc).isoformat(),"scope":"Current pooled model; data-derived text/tables/figures; no model execution",
        "runs":{"v9_pooled":key(run)},"baseline_run":key(args.baseline_run),"bundle_sha256":summary["bundle_sha256"],
        "input_files":INPUTS,"generator_sha256":sha(OUT/"generate_current_assets.py"),
        "generator_sources":{key(path):sha(path) for path in [OUT/"generate_current_assets.py",OUT/"generate_pooled_assets.py",Path(__file__)]},
        "output_sha256":{name:sha(OUT/name) for name in OUTPUTS},"model_weights_retrained":manifest["weights_retrained"],
        "independent_training_seeds":1,"calibration_policy":calibration["policy"],
        "limitations":["Reused internal and official tests remain exploratory.","Corruption realizations are not training seeds.",
                        "Coverage assumptions are unverified under fleet/censoring/corruption shift.","Case diagnostics do not establish physical fault causes."]}
    (OUT/"current_provenance.json").write_text(json.dumps(provenance,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"run":key(run),"inputs":len(INPUTS),"outputs":len(OUTPUTS),"official_clean_RMSE":float(c.RMSE)},indent=2))


if __name__=="__main__":
    main()
