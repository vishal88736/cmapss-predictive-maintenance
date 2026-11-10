"""Recreate manuscript figures/tables from the preserved run without training."""
from pathlib import Path
import hashlib
import json
import os
import runpy
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "cmapss-predictive-maintenance"
SOURCE = PROJECT / "docs/results/full_20261025T211530540186Z"
OUT = ROOT / "paper_assets"
OUT.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "grace-paper-matplotlib"))
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
from matplotlib.figure import Figure
import matplotlib.text
import pandas as pd
import numpy as np

save_original = Figure.savefig
exports = []


def paper_save(self, filename, *args, **kwargs):
    filename = Path(filename)
    if filename.suffix == ".png":
        width, height = self.get_size_inches()
        self.set_size_inches(7.12, height * 7.12 / width)
        if filename.stem in {'01_test_rmse', '06_interval_coverage', '08_correction_gain'}:
            self.set_size_inches(3.45, 2.8)
            self.axes[0].set_position([.17, .16, .80, .66])
        elif filename.stem == '07_detector_validation':
            self.set_size_inches(3.45, 4.0)
            self.axes[0].set_position([.17, .59, .80, .31])
            self.axes[1].set_position([.17, .12, .80, .31])
        # The IEEE captions carry scope/title information. Avoid duplicated
        # poster headers/footnotes inside the journal-sized vector assets.
        for text in self.texts:
            text.set_visible(False)
        for text in self.findobj(matplotlib.text.Text):
            if text.get_visible():
                text.set_fontsize(7.4)
        for axis in self.axes:
            axis.title.set_fontsize(8.2)
        if filename.stem == '01_test_rmse':
            self.axes[0].set_ylabel('Test RMSE (cycles)')
            self.axes[0].legend(frameon=False, fontsize=6.7, loc='lower left',
                                bbox_to_anchor=(0, 1.02))
            for label in self.axes[0].texts:
                label.set_fontsize(6.3)
        elif filename.stem == '04_recovery_acceptance':
            self.axes[1].legend(frameon=False, fontsize=7.4,
                                loc='upper right', bbox_to_anchor=(1, 1.24))
        elif filename.stem == '06_interval_coverage':
            self.axes[0].legend(frameon=False, fontsize=6.7,
                                loc='lower left', bbox_to_anchor=(0, 1.10))
            for label, bar in zip(self.axes[0].texts, self.axes[0].patches):
                label.set_text(label.get_text().rstrip('%'))
                label.xy = (bar.get_x() + bar.get_width()/2, bar.get_height() - 8)
                label.set_position((0, 0))
                label.set_verticalalignment('center')
                label.set_color('white')
                label.set_fontsize(6.3)
        elif filename.stem == '08_correction_gain':
            self.axes[0].set_ylabel('Validation RMSE reduction (%)')
            for label in self.axes[0].texts:
                label.set_fontsize(6.6)
        elif filename.stem == '09_sensor_error_contributions':
            self.axes[0].legend(frameon=False, fontsize=7.4, ncol=3,
                                loc='lower center', bbox_to_anchor=(.5, -.60))
        target = OUT / (filename.stem + ".pdf")
        save_original(self, target, format="pdf", bbox_inches="tight", pad_inches=.04,
                      metadata={"Title": filename.stem, "Subject": "Saved GRACE-RUL numerical evidence; no new training"})
        exports.append(target.name)
    return save_original(self, filename, *args, **kwargs)


Figure.savefig = paper_save
sys.argv = ["plot_results.py", "--results-dir", str(SOURCE), "--output-dir", "/tmp/grace-paper-build/plot_work"]
try:
    runpy.run_path(str(PROJECT / "scripts/plot_results.py"), run_name="__main__")
finally:
    Figure.savefig = save_original

summary = pd.read_csv(SOURCE / "test_seed_summary.csv")
test = pd.read_csv(SOURCE / "test_metrics.csv")
seeds = pd.read_csv(SOURCE / "seed_summary.csv")
hist = pd.read_csv(SOURCE / "training_history.csv")
selected = json.loads((SOURCE / "architecture_evidence.json").read_text())["models"]
faults = pd.read_csv(SOURCE / "review_20261026/frozen_model_fault_check.csv")
detector = {x["subset"]: x for x in json.loads((SOURCE / "review_20261026/detector_validation_review.json").read_text())}
subsets = ["FD001", "FD002", "FD003", "FD004"]


def table(command, caption, label, columns, header, rows, wide=False, note=""):
    env = "table*" if wide else "table"
    lines = [rf"\newcommand{{\{command}}}{{%", rf"\begin{{{env}}}[!t]", rf"\caption{{{caption}}}",
             rf"\label{{{label}}}", r"\centering\footnotesize", r"\setlength{\tabcolsep}{3.5pt}",
             rf"\begin{{tabular}}{{{columns}}}", r"\toprule", header + " " + chr(92)*2, r"\midrule"]
    lines += [" & ".join(map(str,row)) + " " + chr(92)*2 for row in rows]
    lines += [r"\bottomrule",r"\end{tabular}"]
    if note:
        lines += [r"\par\vspace{2pt}",rf"\begin{{minipage}}{{\{'textwidth' if wide else 'columnwidth'}}}\scriptsize {note}\end{{minipage}}"]
    lines += [rf"\end{{{env}}}","}",""]
    return "\n".join(lines)


tables = ["% Auto-generated from the preserved results. Regenerate with generate_assets.py."]
rows=[]
for subset in subsets:
    c=summary[(summary.subset==subset)&(summary.protocol=="capped_125")].iloc[0]
    u=summary[(summary.subset==subset)&(summary.protocol=="official_uncapped")].iloc[0]
    n=int(test[(test.subset==subset)&(test.protocol=="capped_125")].N.iloc[0])
    rows.append([subset,n,rf"${c.RMSE_mean:.2f}\pm{c.RMSE_std:.2f}$",f"{c.MAE_mean:.2f}",f"{c.NASA_Score_mean:.2f}",
                 rf"${u.RMSE_mean:.2f}\pm{u.RMSE_std:.2f}$",f"{u.MAE_mean:.2f}",f"{u.NASA_Score_mean:.2f}"])
tables.append(table("PrimaryResultsTable","Clean official-endpoint results across five training seeds. RMSE and MAE are in cycles.","tab:clean_results","lrrrrrrr",
    r"Subset & $N$ & Capped RMSE & MAE & NASA $S$ & Uncapped RMSE & MAE & NASA $S$",rows,True,
    r"RMSE is mean $\pm$ sample standard deviation over seeds. Other entries are seed means. Capped targets are $\min(y,125)$; uncapped targets are original official RUL. NASA $S$ is the total over each test fleet, averaged over seeds; it is not normalized by $N$.") )
rows=[]
for entry in selected:
    s=seeds[seeds.subset==entry["subset"]]
    rows.append([entry["subset"],entry["selected_seed"],"On" if entry["repair_enabled"] else "Off",
                 r"$\Delta=0$" if not entry["correction_enabled_by_weights"] else "Learned",f"{entry['correction_bound']:g}",f"{int(s.repair_selected.sum())}/5"])
tables.append(table("ExportedModelsTable","Validation-selected exported paths and all-seed recovery retention.","tab:exported_paths","lrrrrr",
    r"Subset & Seed & Repair & Correction & $b$ & Retained",rows,False,
    r"$b$ is the correction bound in cycles, not a guaranteed nonzero correction. Retained counts seeds that keep learned recovery."))
rows=[]
for entry in selected:
    s=entry["subset"];rows.append([s,entry["train_engines"],entry["test_engines"],entry["split_counts"]["fit"],entry["split_counts"]["calibration"],entry["split_counts"]["validation"],1 if s in ["FD001","FD003"] else 6])
tables.append(table("DataPartitionTable","Engine counts and partitions in the local C-MAPSS snapshot.","tab:cmapss_subsets","lrrrrrr",
    r"Subset & Train & Test & Fit & Cal. & Val. & Reg.",rows,False,
    r"All models use 21 sensor columns, three settings, a 70-cycle input window and a 30-cycle local branch. FD004 counts describe the supplied files rather than an assumed canonical ordering."))
families=["noise","bias","drift","stuck","intermittent","outage","block","mixed"]
rows=[];active=[]
for s in subsets:
    seed=next(x["selected_seed"] for x in selected if x["subset"]==s)
    d=faults[faults.subset==s].set_index("family")
    pred=pd.read_csv(SOURCE/f"review_20261026/selected_model_verification.csv").set_index("subset").loc[s]
    rows.append([s,seed,f"{pred.capped_RMSE:.2f}",*[f"{d.loc[f,'RMSE']:.2f}" for f in families]])
    active.append([s,*[rf"${d.loc[f,'active_fault_RMSE']:.2f}\;({int(d.loc[f,'active_fault_N'])})$" for f in families]])
tables.append(table("FaultResultsTable","Additional frozen-selected-model check: capped RMSE, severity $v=0.2$, one fixed replicate.","tab:fault_results","lrrrrrrrrrr",
    r"Subset & Seed & Clean & Noise & Bias & Drift & Stuck & Intermit. & Outage & Block & Mixed",rows,True,
    r"These are separate CPU review evaluations, not five-seed means or a full severity sweep. All official endpoints are included, including histories with no active corruption in their last window. Severity is a mechanism control, not a percentage of bad entries."))
tables.append(table("ActiveFaultTable","Active-fault-only capped RMSE with affected endpoint count in parentheses.","tab:active_faults","lrrrrrrrr",
    r"Subset & Noise & Bias & Drift & Stuck & Intermit. & Outage & Block & Mixed",active,True,
    r"Same frozen models, severity and replicate as Table~\ref{tab:fault_results}. An endpoint is active only when its observed window contains at least one changed sensor reading. Counts differ by fault and subset."))
rows=[]
for entry in selected:
    s=entry["subset"];d=test[test.subset==s];m=detector[s]["saved_threshold_validation_metrics"]
    g=seeds[seeds.subset==s]
    gain=(100*(g.validation_no_correction_RMSE-g.validation_selection_RMSE)/g.validation_no_correction_RMSE).mean()
    rows.append([s,entry["split_counts"]["calibration"],f"{100*d[d.protocol=='capped_125'].Interval_Coverage.mean():.1f}",
        f"{100*d[d.protocol=='official_uncapped'].Interval_Coverage.mean():.1f}",f"{100*m['recall']:.2f}",f"{m['F1']:.4f}",f"{gain:.3f}"])
tables.append(table("CalibrationDetectionTable","Calibration, finite-fault detection and correction diagnostics with distinct scopes.","tab:calibration_detection","lrrrrrr",
    r"Subset & Cal. engines & Capped cov. (\%) & Uncapped cov. (\%) & Recall (\%) & F1 & Gain (\%)",rows,True,
    r"Coverage: five-seed mean on clean official endpoints, nominal target 90\%. Recall/F1: selected-model validation diagnostic under saved thresholds, combining clean/bias/drift/noise cases. Gain: five-seed mean reduction of clean/active-mixed validation RMSE from correction. These columns are not one common test experiment."))
rows=[]
for s in subsets:
    a=hist[(hist.subset==s)&(hist.stage!='predictor')]
    rows.append([s,len(a),int(a.checkpoint_accepted.sum()),f"{a.validation_reconstruction_RMSE.min():.3f}"])
tables.append(table("RepairRejectionTable","Recovery/joint checkpoint outcomes across the five primary seeds.","tab:recovery_rejection","lrrr",
    r"Subset & Tried & Admissible & Min. recon. RMSE",rows,False,
    r"Admissibility includes clean, intact-entry and reconstruction constraints. A candidate must also improve the selection score to replace the saved checkpoint. Reconstruction limit: 1.5."))
(OUT/"results_tables.tex").write_text("\n".join(tables))
source_hashes_verified = 0
for line in (SOURCE/"SHA256SUMS").read_text().splitlines():
    expected, name = line.split('  ', 1)
    assert hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() == expected, name
    source_hashes_verified += 1
provenance={"source_snapshot":str(SOURCE.relative_to(ROOT)),"run":"full_20261025T211530540186Z",
    "figures":exports,"tables":"results_tables.tex","scope":"Reformatted existing plots and exact numeric tables only; no retraining or new experiments",
    "asset_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob('*.pdf'))},
    "source_manifest_sha256":hashlib.sha256((SOURCE/"snapshot_manifest.json").read_bytes()).hexdigest(),
    "tables_sha256":hashlib.sha256((OUT/"results_tables.tex").read_bytes()).hexdigest(),
    "generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "source_result_hashes_verified":source_hashes_verified}
(OUT/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
print("Paper vector figures:",len(exports),"; data-driven LaTeX tables: 7")
