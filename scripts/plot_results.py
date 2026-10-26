"""Render scientific figures from a saved GRACE-RUL result snapshot; no training."""

import argparse
import hashlib
from html import escape
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "grace-results-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE, GREEN, RED, PURPLE = "#2563eb", "#d97706", "#0f766e", "#dc2626", "#7c3aed"
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
STAGES = ["predictor", "generator", "joint"]
STAGE_COLORS = ["#e2e8f0", "#ccfbf1", "#ede9fe"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.results_dir.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    inputs = {}

    def read(name):
        path = source / name
        inputs[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return pd.read_csv(path) if path.suffix == ".csv" else json.loads(path.read_text())

    run_meta = read("run_manifest.json")
    SCOPE = f"{source.name} | {run_meta['corruption_protocol']}"
    summary = read("test_seed_summary.csv")
    test = read("test_metrics.csv")
    seeds = read("seed_summary.csv")
    history = read("training_history.csv")
    selected = read("review_20261026/selected_model_verification.csv").set_index("subset").loc[SUBSETS]
    faults = read("review_20261026/frozen_model_fault_check.csv")
    detector = read("review_20261026/detector_validation_review.json")
    diagnostics = read("review_20261026/corruption_diagnostics.json")
    architecture = read("architecture_evidence.json")
    for entry in architecture["models"]:
        assert entry["selected_seed"] == int(selected.loc[entry["subset"], "selected_seed"])
    for subset in SUBSETS:
        for protocol in ["capped_125", "official_uncapped"]:
            original = test[(test.subset == subset) & (test.protocol == protocol)]
            aggregate = summary[(summary.subset == subset) & (summary.protocol == protocol)].iloc[0]
            assert len(original) == 5 and original.seed.nunique() == 5
            np.testing.assert_allclose([original.RMSE.mean(), original.RMSE.std(ddof=1)],
                                       [aggregate.RMSE_mean, aggregate.RMSE_std], atol=1e-10)
    assert len(faults) == 32 and set(faults.severity) == {.2} and set(faults.replicate) == {0}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
                         "axes.titleweight": "bold", "axes.labelcolor": "#334155", "text.color": "#0f172a",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "savefig.facecolor": "white"})
    figures = []

    def save(fig, stem, title, scope, caption):
        for ext in ["png", "svg"]:
            fig.savefig(output / f"{stem}.{ext}", dpi=190, bbox_inches="tight", pad_inches=.18)
        plt.close(fig)
        figures.append({"file": stem, "title": title, "scope": scope, "caption": caption})

    def label_bars(ax, bars, fmt=".2f", unit=""):
        ax.bar_label(bars, labels=[f"{b.get_height():{fmt}}{unit}" for b in bars], padding=4, fontsize=9)

    def finish_axis(ax):
        ax.grid(axis="y", alpha=.18, zorder=0)
        ax.set_axisbelow(True)

    def header(fig, title, subtitle, footer, *, bottom=.16, top=.80):
        fig.suptitle(title, fontsize=17, fontweight="bold", y=.99)
        fig.text(.5, .927, subtitle, ha="center", fontsize=10, color="#475569")
        fig.text(.015, .018, footer, fontsize=9, color="#475569", va="bottom")
        fig.subplots_adjust(top=top, bottom=bottom, left=.08, right=.98)

    x = np.arange(4)
    capped = summary[summary.protocol == "capped_125"].set_index("subset").loc[SUBSETS]
    raw = summary[summary.protocol == "official_uncapped"].set_index("subset").loc[SUBSETS]
    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    a = ax.bar(x-.19, capped.RMSE_mean, .36, yerr=capped.RMSE_std, capsize=4, color=BLUE, label="Capped target: min(RUL, 125)", zorder=3)
    b = ax.bar(x+.19, raw.RMSE_mean, .36, yerr=raw.RMSE_std, capsize=4, color=ORANGE, label="Original official RUL", zorder=3)
    label_bars(ax, a); label_bars(ax, b)
    ax.set(xticks=x, xticklabels=SUBSETS, ylim=(0, 31), ylabel="Test RMSE (cycles; lower is better)")
    ax.legend(frameon=False, loc="upper left", ncol=2)
    finish_axis(ax)
    header(fig, "Clean official-endpoint RUL error", "Mean ± sample standard deviation over five training seeds per subset",
           f"{SCOPE}\nCapped and uncapped targets are distinct evaluation protocols. Error bars are SD, not confidence intervals.")
    save(fig, "01_test_rmse", "Clean test RMSE", "Original run; five seeds; final official endpoints", "Capped and uncapped RMSE are reported separately. Bars are means; error bars are sample SD across five seeds.")

    predictions = {}
    for subset in SUBSETS:
        seed = int(selected.loc[subset, "selected_seed"])
        predictions[subset] = read(f"predictions_{subset}_{seed}.csv")
    limit = np.ceil(max(max(d.raw_target.max(), d.prediction.max()) for d in predictions.values()) / 25) * 25
    fig, axes = plt.subplots(2, 2, figsize=(10.7, 9.0), sharex=True, sharey=True)
    for subset, ax in zip(SUBSETS, axes.flat):
        d = predictions[subset]
        long = d.raw_target > 125
        ax.scatter(d.raw_target[~long], d.prediction[~long], s=23, color=BLUE, alpha=.65, label="True RUL ≤ 125", linewidths=0)
        ax.scatter(d.raw_target[long], d.prediction[long], s=32, color=ORANGE, alpha=.85, label="True RUL > 125", edgecolor="white", linewidths=.4)
        ax.plot([0, limit], [0, limit], "--", color="#64748b", lw=1.2, label="Perfect prediction")
        ax.axvline(125, color=ORANGE, ls=":", alpha=.6)
        rmse = np.sqrt(np.mean((d.prediction-d.raw_target)**2))
        ax.set(title=f"{subset} · validation-selected seed {int(selected.loc[subset, 'selected_seed'])}",
               xlim=(0, limit), ylim=(0, limit), xlabel="Official true RUL (cycles)", ylabel="Predicted RUL (cycles)")
        ax.text(.04, .94, f"Uncapped RMSE: {rmse:.2f}\nN = {len(d)} engines", transform=ax.transAxes, va="top", fontsize=9)
        ax.grid(alpha=.15); ax.set_aspect("equal", adjustable="box")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .075), ncol=3, frameon=False)
    header(fig, "Predicted versus actual remaining useful life", "Saved original predictions for the validation-selected model in each subset",
           "One point per official test engine. Selected seeds are not chosen by test RMSE.\nOrange points expose the long-RUL limitation of capped-target training.", bottom=.20, top=.84)
    fig.subplots_adjust(hspace=.34, wspace=.28)
    save(fig, "02_predicted_vs_actual", "Predicted versus actual RUL", "Original predictions; one validation-selected seed per subset", "The diagonal denotes exact prediction. Orange observations have official true RUL above the training cap of 125 cycles.")

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.4))
    for subset, ax in zip(SUBSETS, axes.flat):
        h = history[(history.subset == subset) & (history.seed == int(selected.loc[subset, "selected_seed"]))]
        offset = 0
        for stage, color in zip(STAGES, STAGE_COLORS):
            part = h[h.stage == stage].sort_values("epoch")
            xx = np.arange(1, len(part)+1) + offset
            ax.axvspan(offset+.5, offset+len(part)+.5, color=color, alpha=.75, zorder=0)
            for field, name, linecolor, linestyle in [("validation_clean_RMSE", "Clean validation", BLUE, "-"),
                 ("validation_corrupt_RMSE", "Active mixed faults", ORANGE, "-"),
                 ("validation_macro_RMSE", "Selection score", PURPLE, "--")]:
                ax.plot(xx, part[field], color=linecolor, ls=linestyle, lw=1.5)
            ax.text(offset+len(part)/2+.5, .97, stage, ha="center", va="top", fontsize=9,
                    transform=ax.get_xaxis_transform(), bbox={"facecolor":"white", "alpha":.65, "edgecolor":"none", "pad":1})
            if offset:
                ax.axvline(offset+.5, color="#94a3b8", lw=.7)
            offset += len(part)
        ax.set(title=f"{subset} · seed {int(selected.loc[subset, 'selected_seed'])}", xlabel="Completed upstream training epochs", ylabel="Validation RMSE (cycles)")
        ax.margins(y=.18); ax.grid(axis="y", alpha=.17)
    handles = [plt.Line2D([0], [0], color=color, lw=1.5, ls=style, label=name) for name,color,style in
               [("Clean validation",BLUE,"-"),("Active mixed faults",ORANGE,"-"),("Selection score",PURPLE,"--")]]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5,.075), ncol=3, frameon=False)
    header(fig, "Validation curves across the three upstream phases", "Observed epoch histories; the final residual corrector is excluded",
           "Curves show attempted epochs, including rejected updates; the deployed model restores the selected checkpoint.\nPhase lengths differ due to early stopping. Loss/objective magnitudes are not plotted as RUL RMSE.", bottom=.20, top=.83)
    fig.subplots_adjust(hspace=.50, wspace=.27)
    save(fig, "03_validation_curves", "Upstream validation curves", "Selected seed per subset; all attempted upstream epochs", "Shading identifies predictor, generator and joint phases. Rejected updates remain visible, so the last curve point need not describe the deployed checkpoint.")

    kept = seeds.groupby("subset").repair_selected.sum().reindex(SUBSETS).astype(int)
    attempted = history[history.stage != "predictor"]
    best_rec = attempted.groupby("subset").validation_reconstruction_RMSE.min().reindex(SUBSETS)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.2))
    bars = axes[0].bar(x, kept, color=[GREEN if v else RED for v in kept], width=.6)
    axes[0].bar_label(bars, labels=[f"{v}/5" for v in kept], padding=5)
    axes[0].set(xticks=x, xticklabels=SUBSETS, ylim=(0,5.8), yticks=range(6), ylabel="Seeds retaining learned repair", title="Repair retained in the saved models")
    bars = axes[1].bar(x, best_rec, color=[GREEN if v<=1.5 else RED for v in best_rec], width=.6, zorder=3)
    label_bars(axes[1], bars)
    axes[1].set(xticks=x, xticklabels=SUBSETS, yscale="log", ylim=(.2,1500), ylabel="Normalized sensor RMSE · logarithmic scale", title="Best attempted reconstruction error")
    axes[1].axhline(1.5,color="#0f172a",ls="--",lw=1.2,label="Acceptance limit = 1.5")
    axes[1].legend(frameon=False,fontsize=9,loc="upper left")
    for ax in axes: finish_axis(ax)
    header(fig, "Recovery acceptance reveals the main integration failure", "Five-seed retention and the minimum reconstruction RMSE over all attempted recovery/joint epochs",
           "Passing reconstruction alone is insufficient: clean and intact-reading constraints also apply.\nFD002/FD004 exceed the reconstruction limit in every attempted recovery/joint epoch.")
    fig.subplots_adjust(wspace=.35)
    save(fig, "04_recovery_acceptance", "Recovery acceptance", "All five seeds; observed upstream histories", "The reconstruction panel uses a logarithmic axis. It shows the best attempted value, not a claim that this checkpoint passed the other selection guards.")

    families = ["noise", "bias", "drift", "stuck", "intermittent", "outage", "block", "mixed"]
    all_fault = faults.pivot(index="subset", columns="family", values="RMSE").loc[SUBSETS, families]
    active_fault = faults.pivot(index="subset", columns="family", values="active_fault_RMSE").loc[SUBSETS, families]
    clean = selected.capped_RMSE.to_numpy()
    all_matrix = np.column_stack([clean, all_fault.to_numpy()])
    assert np.isfinite(all_matrix).all() and np.isfinite(active_fault.to_numpy()).all()
    vmax = float(np.ceil(max(all_matrix.max(), active_fault.to_numpy().max()) / 5) * 5)
    fig, axes = plt.subplots(2, 1, figsize=(12.8, 7.6))
    for ax, values, cols, title in [(axes[0], all_matrix, ["clean",*families], "All official endpoints, including unaffected histories"),
                                   (axes[1], active_fault.to_numpy(), families, "Only endpoints with actual corruption in the observed window")]:
        im = ax.imshow(values, cmap="YlOrRd", vmin=0, vmax=vmax, aspect="auto")
        ax.set(xticks=np.arange(len(cols)), xticklabels=cols, yticks=np.arange(4), yticklabels=[f"{s} · seed {int(selected.loc[s,'selected_seed'])}" for s in SUBSETS], title=title)
        ax.tick_params(axis="both", length=0)
        for i in range(4):
            for j in range(len(cols)):
                ax.text(j,i,f"{values[i,j]:.2f}",ha="center",va="center",color="white" if values[i,j]>.6*vmax else "#0f172a",fontsize=10)
        for spine in ax.spines.values(): spine.set_visible(False)
    fig.subplots_adjust(top=.82,bottom=.18,left=.15,right=.88,hspace=.60)
    fig.suptitle("Frozen-model sensor-fault check",fontsize=17,fontweight="bold",y=.99)
    fig.text(.5,.93,"Capped RMSE (cycles) · one validation-selected model per subset · severity .2 · replicate 0",ha="center",fontsize=10,color="#475569")
    cax=fig.add_axes([.91,.23,.018,.53]);fig.colorbar(im,cax=cax,label="Capped RUL RMSE (cycles)")
    fig.text(.015,.018,"Additional CPU inference from the recorded review; no retraining or tuning. This is not the full multi-seed robustness study.\nSeverity is a mechanism control, not a percentage of corrupted readings. Active-fault endpoint counts are preserved in the source CSV.",fontsize=9,color="#475569")
    save(fig, "05_fault_robustness", "Frozen-model fault robustness", "One selected model; severity .2; one replicate; additional CPU inference", "Both panels share the same color scale. All-endpoint and active-fault-only RMSE are distinct. No multi-seed uncertainty or published-paper comparison is claimed.")

    coverage = test.groupby(["subset","protocol"]).Interval_Coverage.agg(["mean","std"])
    fig, ax = plt.subplots(figsize=(10.8,5.4))
    a = coverage.xs("capped_125",level="protocol").loc[SUBSETS]
    b = coverage.xs("official_uncapped",level="protocol").loc[SUBSETS]
    p=ax.bar(x-.19,100*a["mean"],.36,yerr=100*a["std"],capsize=4,color=BLUE,label="Capped targets",zorder=3)
    q=ax.bar(x+.19,100*b["mean"],.36,yerr=100*b["std"],capsize=4,color=ORANGE,label="Official uncapped targets",zorder=3)
    label_bars(ax,p,fmt=".1f",unit="%");label_bars(ax,q,fmt=".1f",unit="%")
    ax.axhline(90,color=RED,ls="--",lw=1.4,label="Nominal target: 90%")
    ax.set(xticks=x,xticklabels=SUBSETS,ylim=(0,110),ylabel="Official endpoint coverage (%)")
    ax.legend(frameon=False,loc="upper left",ncol=3,fontsize=9)
    finish_axis(ax)
    header(fig,"Empirical prediction-interval coverage","Mean ± sample standard deviation over five training seeds",
           "Intervals use independent calibration residual quantiles; they are not conformal coverage guarantees.\nError bars are SD across seeds, not confidence intervals. Clean endpoint coverage does not establish coverage under faults.")
    save(fig,"06_interval_coverage","Prediction-interval coverage","Original run; five seeds; clean official endpoints","The red line marks nominal 90% coverage. Report capped and uncapped coverage separately; neither is a guaranteed operational probability.")

    d={row["subset"]:row for row in detector}
    recall=np.array([d[s]["saved_threshold_validation_metrics"]["recall"] for s in SUBSETS])
    f1=np.array([d[s]["saved_threshold_validation_metrics"]["F1"] for s in SUBSETS])
    fig,axes=plt.subplots(1,2,figsize=(11.0,5.2))
    a=axes[0].bar(x,recall*100,color=RED,width=.6)
    label_bars(axes[0],a,fmt=".2f",unit="%")
    axes[0].set(xticks=x,xticklabels=SUBSETS,ylim=(0,100),title="Recall at saved thresholds",ylabel="Finite-fault recall (%)")
    b=axes[1].bar(x,f1,color=RED,width=.6)
    label_bars(axes[1],b,fmt=".4f")
    axes[1].set(xticks=x,xticklabels=SUBSETS,ylim=(0,1),title="F1 at saved thresholds",ylabel="Finite-fault F1 (0–1)")
    for ax in axes: finish_axis(ax)
    header(fig,"Finite-fault detection remains weak","Validation-only diagnostic · selected models · original saved thresholds",
           "Inputs combine clean, bias, drift and noise validation scenarios. Zero recall/F1 reflects actual positive labels with no detections.\nFD002/FD004 detector heads remain at initialization; missingness detection is a separate availability signal.")
    save(fig,"07_detector_validation","Finite-fault validation detection","Validation diagnostic; selected models; saved thresholds unchanged","Recall and F1 have different units and are shown on separate full-range axes. No SHAP/LIME, physical fault probability or test-generalization claim is implied.")

    gain=seeds.assign(gain_pct=100*(seeds.validation_no_correction_RMSE-seeds.validation_selection_RMSE)/seeds.validation_no_correction_RMSE)
    fig,ax=plt.subplots(figsize=(10.4,5.2))
    means=gain.groupby("subset").gain_pct.mean().reindex(SUBSETS)
    bars=ax.bar(x,means,color=PURPLE,width=.60,alpha=.75,zorder=2)
    for i,s in enumerate(SUBSETS):
        values=gain[gain.subset==s].sort_values("seed").gain_pct.to_numpy()
        ax.scatter(np.linspace(i-.15,i+.15,len(values)),values,s=32,color="#0f172a",edgecolor="white",linewidth=.6,zorder=3)
        ax.text(i,float(values.max())+.10,f"mean {means.loc[s]:.3f}%",ha="center",fontsize=9)
    ax.axhline(0,color="#64748b",lw=1)
    ax.set(xticks=x,xticklabels=SUBSETS,ylabel="Validation selection-RMSE reduction (%)",ylim=(-.1,max(3.,float(gain.gain_pct.max())*1.2)))
    finish_axis(ax)
    header(fig,"Contribution of the bounded residual corrector","Bars: mean across five seeds · points: individual seeds",
           "Reduction is relative to the same upstream model before correction, on clean/active-mixed validation RMSE.\nThis is not a retrained comparator, an official-test gain, or improvement over published papers.")
    save(fig,"08_correction_gain","Correction validation gain","Original seed summary; five validation comparisons","Positive values indicate lower validation selection RMSE after correction; exact zero denotes no selected improvement. Points preserve seed variability.")

    diag = {row["subset"]: row for row in diagnostics}
    fig,axes = plt.subplots(1,2,figsize=(12.0,5.6))
    categories=[("NRf",RED),("Nf",ORANGE),("phi",PURPLE),("P30",BLUE),("Other", "#94a3b8")]
    left=np.zeros(2)
    for name,color in categories:
        values=[]
        for subset in ["FD002","FD004"]:
            contributions={entry["name"]:entry["SSE_fraction"] for entry in diag[subset]["channels"]}
            value=(1-sum(contributions.get(n,0) for n,_ in categories[:-1])) if name=="Other" else contributions.get(name,0)
            values.append(100*value)
        bars=axes[0].barh(np.arange(2),values,left=left,height=.52,color=color,label=name)
        for bar,value in zip(bars,values):
            if value>7:
                axes[0].text(bar.get_x()+bar.get_width()/2,bar.get_y()+bar.get_height()/2,f"{value:.1f}%",ha="center",va="center",color="white",fontsize=10)
        left+=np.array(values)
    np.testing.assert_allclose(left,[100.,100.],atol=1e-8)
    axes[0].set(yticks=[0,1],yticklabels=["FD002","FD004"],xlim=(0,100),xlabel="Share of unrepaired altered-input squared error (%)",title="Speed sensors dominate normalized error")
    axes[0].invert_yaxis();axes[0].legend(frameon=False,ncol=3,fontsize=9,loc="lower center",bbox_to_anchor=(.5,-.30))
    worst=[diag[s]["worst_finite_case"]["normalized_error"] for s in ["FD002","FD004"]]
    bars=axes[1].bar([0,1],worst,color=[RED,ORANGE],width=.55)
    label_bars(axes[1],bars)
    axes[1].set(xticks=[0,1],xticklabels=["FD002 · NRf","FD004 · Nf"],ylim=(0,6500),ylabel="Absolute normalized reading error",title="Largest finite single-entry errors")
    finish_axis(axes[1])
    header(fig,"Why recovery struggles across operating regimes","Fixed mixed validation scenario · preserved corruption diagnostics · selected fitting-only scalers",
           "Left: sensor shares of altered INPUT error before recovery, not final model reconstruction error.\nRight: largest finite reading errors, not RUL RMSE. A physically stuck speed becomes extreme after regime normalization.",bottom=.25,top=.79)
    fig.subplots_adjust(wspace=.40)
    save(fig,"09_sensor_error_contributions","Normalization-error diagnosis","Fixed mixed validation scenario; unrepaired input diagnostics","Left shows per-sensor squared-error contributions; right shows worst individual finite reading errors. These are sensor-space diagnostics, not RUL errors or a new test benchmark.")

    fig,axes=plt.subplots(2,2,figsize=(12.0,8.8))
    ax=axes[0,0]
    a=ax.bar(x-.18,capped.RMSE_mean,.34,color=BLUE,label="Capped",zorder=3)
    b=ax.bar(x+.18,raw.RMSE_mean,.34,color=ORANGE,label="Uncapped",zorder=3)
    label_bars(ax,a,fmt=".1f");label_bars(ax,b,fmt=".1f")
    ax.set(title="A · Clean RMSE: five-seed mean",xticks=x,xticklabels=SUBSETS,ylim=(0,32),ylabel="Cycles");ax.legend(frameon=False,fontsize=8)
    ax=axes[0,1];stuck=all_fault["stuck"].to_numpy()
    bars=ax.bar(x,stuck,color=ORANGE,width=.6);label_bars(ax,bars)
    ax.set(title="B · Stuck fault: one frozen selected model",xticks=x,xticklabels=SUBSETS,ylim=(0,60),ylabel="Capped RMSE (cycles)")
    ax=axes[1,0];bars=ax.bar(x,kept,color=[GREEN if v else RED for v in kept],width=.6)
    ax.bar_label(bars,labels=[f"{v}/5" for v in kept],padding=4)
    ax.set(title="C · Seeds retaining learned recovery",xticks=x,xticklabels=SUBSETS,ylim=(0,5.8),yticks=range(6),ylabel="Model count")
    ax=axes[1,1];bars=ax.bar(x,100*capped_coverage(coverage,SUBSETS),color=BLUE,width=.6);label_bars(ax,bars,fmt=".1f",unit="%")
    ax.axhline(90,color=RED,ls="--",lw=1.2)
    ax.set(title="D · Capped interval coverage: five-seed mean",xticks=x,xticklabels=SUBSETS,ylim=(0,110),ylabel="Coverage (%)")
    for ax in axes.flat: finish_axis(ax)
    header(fig,"GRACE-RUL · trained-results overview","Five-seed primary results and a separate frozen-model fault check",
           "A/C/D: original full run, five seeds. B: additional CPU check, one validation-selected model, severity .2, replicate 0.\nNo matched paper-comparison or SHAP/LIME results were generated. Detailed figures retain protocol and uncertainty labels.",bottom=.14,top=.82)
    fig.subplots_adjust(hspace=.48,wspace=.30)
    save(fig,"00_results_overview","Trained-results overview","Mixed scopes, separately identified in each panel and footer","Overview panels preserve the difference between five-seed primary evidence and the additional single-replicate frozen-model fault check.")

    gallery=["# GRACE-RUL graphs and diagrams", "", f"Saved run: `{SCOPE}`. All plots derive from preserved numerical evidence; no model training was performed.", "",
             "PNG files are for viewing/slides; SVG files preserve vector quality. The plot source is `scripts/plot_results.py`.", "",
             "The bundle contains one validation-selected model per subset. Five-seed aggregate plots and selected-model diagnostics are different scopes.", ""]
    for figure in [figures[-1],*figures[:-1]]:
        stem=figure['file'];gallery.extend([f"## {figure['title']}","",f"Scope: {figure['scope']}","",figure['caption'],"",f"![{figure['title']}]({stem}.png)","",f"[Vector SVG]({stem}.svg) · [PNG]({stem}.png)",""])
    gallery.extend(["## Architecture and workflow diagrams","",
        "[Complete architecture](../../diagrams/GRACE_RUL_current_architecture.svg) · [Training workflow](../../diagrams/GRACE_RUL_training_workflow.svg) · [Three-paper integration](../../diagrams/GRACE_RUL_paper_integration.svg)","",
        "[Architecture/workflow explanation](../../ARCHITECTURE_AND_WORKFLOW.md) · [Saved results](../../RESULTS_2026-10-26.md)","",
        "## Reproduce the result plots","", "Run from the project root using the existing environment:","","```bash",
        "minor/bin/python scripts/plot_results.py \\", f"  --results-dir docs/results/{source.name} \\", f"  --output-dir docs/figures/{output.name}","```",""])
    (output/"GALLERY.md").write_text("\n".join(gallery))
    write_html_gallery(output, [figures[-1],*figures[:-1]], SCOPE)
    manifest={"run":SCOPE,"matplotlib_version":matplotlib.__version__,"plot_count":len(figures),"figures":figures,"input_sha256":inputs,
              "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope":"Saved numerical results only; no retraining, parameter changes, baseline fits or SHAP/LIME generation",
              "rendered_files":[{"file":p.name,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(output.iterdir()) if p.suffix in {".svg",".png"}]}
    (output/"plot_manifest.json").write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n")
    print(f"Saved {len(figures)} plots in PNG/SVG, gallery and provenance manifest to {output}")


def capped_coverage(coverage, subsets):
    return coverage.xs("capped_125",level="protocol").loc[subsets,"mean"].to_numpy()


def write_html_gallery(output, figures, scope):
    """Create a local, dependency-free document for browsing the exported images."""
    cards=[]
    for index,figure in enumerate(figures):
        stem=figure['file']
        cards.append(f'''<section id="figure-{index}"><h2>{escape(figure['title'])}</h2>
<p class="scope">{escape(figure['scope'])}</p><p>{escape(figure['caption'])}</p>
<p><a href="{stem}.svg">Open vector SVG</a> · <a href="{stem}.png">Open PNG</a></p>
<a href="{stem}.svg"><img src="{stem}.png" alt="{escape(figure['title'])}" loading="lazy"></a></section>''')
    for title,stem,caption in [
        ('Complete architecture','GRACE_RUL_current_architecture','Detailed module dimensions, recovery fallback, training-only critic and actual exported paths.'),
        ('Training workflow','GRACE_RUL_training_workflow','Fitting, validation, OOF correction, calibration, test, export and optional studies.'),
        ('Three-paper integration','GRACE_RUL_paper_integration','Paper-inspired modules and their implemented adaptations. This is not proof of superiority.')]:
        cards.append(f'''<section><h2>{escape(title)}</h2><p>{escape(caption)}</p>
<p><a href="../../diagrams/{stem}.svg">Open vector SVG</a> · <a href="../../diagrams/{stem}.png">Open PNG</a></p>
<a href="../../diagrams/{stem}.svg"><img src="../../diagrams/{stem}.png" alt="{escape(title)}" loading="lazy"></a></section>''')
    content='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GRACE-RUL results, graphs and diagrams</title><style>
*{box-sizing:border-box}body{margin:0;background:#f1f5f9;color:#0f172a;font:16px/1.6 system-ui,sans-serif}
main{max-width:1240px;margin:0 auto;padding:32px 20px}header{margin-bottom:32px}h1{line-height:1.2;font-size:clamp(28px,4vw,42px)}
h2{line-height:1.3;margin-top:0}section{background:white;border:1px solid #cbd5e1;border-radius:12px;padding:24px;margin:28px 0}
img{display:block;width:100%;height:auto;margin-top:20px}a{color:#1d4ed8;text-underline-offset:3px}.scope{color:#475569;font-size:14px}
code{overflow-wrap:anywhere}nav{display:flex;gap:16px;flex-wrap:wrap}footer{margin:32px 0;color:#475569}
</style><main><header><h1>GRACE-RUL: result graphs and diagrams</h1>'''
    content+=f'<p><code>{escape(scope)}</code></p>'
    content+='''<p>Ten graphs drawn from the preserved numerical results, plus three architecture/workflow diagrams.
Five-seed aggregates and single-model fault checks retain separate scope labels. Error bars show sample SD, not confidence intervals.
The figures preserve recovery and detector failures; matched paper comparisons and SHAP/LIME were not generated.</p>
<nav><a href="GALLERY.md">Markdown gallery</a><a href="../../ARCHITECTURE_AND_WORKFLOW.md">Complete explanation</a>
<a href="../../RESULTS_2026-10-26.md">Saved results</a><a href="plot_manifest.json">Data provenance</a></nav></header>'''
    content+=''.join(cards)+'<footer>All model weights and original result payloads remain unchanged. Click any image for its vector version.</footer></main></html>'
    (output/'GALLERY.html').write_text(content)


if __name__ == "__main__":
    main()
