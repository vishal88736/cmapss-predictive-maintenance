"""Generate the noise-robustness section and the cross-study comparison.

Two evidence classes are kept strictly separate:

* Saved measurements of the frozen artifact: per-subset clean metrics and the
  completed broadband-noise benchmark. These are read from hashed CSVs.
* Literature values transcribed from the three supplied papers. They are
  labelled ``reported`` and are never recomputed, averaged with ours, or used
  to select anything.

Only third-party baselines are ranked. The proposed row of TRANS-D3 is retained
for completeness but is not treated as a like-for-like target, because its
FD001/FD003 values are far outside every third-party result in the same table.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT.parent
OUT = PAPER / "paper_assets"
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]

# RMSE at the 125-cycle target as printed in TRANS-D3, Table 7. ``None`` means
# the model was not reported for that subset. These are transcriptions; the
# generator asserts the resulting positions so a typo cannot silently move a
# claim.
THIRD_PARTY = {
    "SVR": {"FD001": 25.12, "FD002": 27.94, "FD003": 26.65, "FD004": 28.13},
    "MLP": {"FD001": 29.15, "FD002": 33.65, "FD003": 31.52, "FD004": 33.13},
    "CNN": {"FD001": 21.51, "FD002": 26.54, "FD003": 21.36, "FD004": 26.77},
    "LSTM": {"FD001": 19.40, "FD002": 20.81, "FD003": 17.79, "FD004": 27.17},
    "CNN+LSTM": {"FD001": 16.12, "FD002": 20.45, "FD003": 17.12, "FD004": 23.25},
    "BiLSTM": {"FD001": 13.65, "FD002": 23.18, "FD003": 13.74, "FD004": 24.86},
    "DCNN": {"FD001": 12.61, "FD002": 22.36, "FD003": 12.64, "FD004": 23.31},
    "GCT": {"FD001": 11.27, "FD002": 22.81, "FD003": 11.42, "FD004": 24.86},
    "BiLSTM-Att": {"FD001": 13.78, "FD002": 15.94, "FD003": 14.36, "FD004": 16.96},
    "ELSTMNN": {"FD001": 18.22, "FD002": None, "FD003": 23.21, "FD004": None},
    "DATCN": {"FD001": 11.78, "FD002": 16.95, "FD003": 11.56, "FD004": 18.23},
    "AGCNN": {"FD001": 12.42, "FD002": 19.43, "FD003": 13.39, "FD004": 21.50},
    "DAST": {"FD001": 11.43, "FD002": 15.25, "FD003": 11.32, "FD004": 18.23},
    "DLformer": {"FD001": None, "FD002": 15.93, "FD003": None, "FD004": 15.86},
    "1D-CNN-LSTM": {"FD001": 16.10, "FD002": None, "FD003": None, "FD004": None},
    "CNN-LSTM-SAM": {"FD001": 12.60, "FD002": 18.90, "FD003": 12.50, "FD004": 20.50},
    "BiLSTM-DAE-Transformer": {"FD001": 10.98, "FD002": 16.12, "FD003": 11.14, "FD004": 18.15},
    "STAR": {"FD001": 10.61, "FD002": 13.47, "FD003": 10.71, "FD004": 15.87},
}
BEST_THIRD_PARTY = {"FD001": "STAR", "FD002": "STAR", "FD003": "STAR", "FD004": "DLformer"}
PAPER_A = {"FD001": 13.1, "FD002": 16.8, "FD003": 14.5, "FD004": 18.9}
PAPER_A_NOTE = "Hybrid CNN-GRU, mean of five repeats (MSSP 2026)"
TRANSD3 = {"FD001": 1.66, "FD002": 13.23, "FD003": 6.61, "FD004": 12.26}
PAPER_B_POOLED = 12.15
# Independently recomputed positions; a transcription error trips this. Format 10
# moved FD001 and FD003 above the best third-party entry, so every position is 1.
EXPECTED_RANK = {"FD001": 1, "FD002": 1, "FD003": 1, "FD004": 1}


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
    parser.add_argument("--noise-run", type=Path,
                        default=ROOT / "artifacts/noise_benchmark_full")
    args = parser.parse_args(argv)
    args.noise_run = args.noise_run.resolve()

    pointer = json.loads((OUT / "current_run.json").read_text())
    run = (PAPER / pointer["run"]).resolve()
    manifest = json.loads((run / "run_manifest.json").read_text())
    if manifest["frozen_bundle_sha256"] != pointer["bundle_sha256"]:
        raise ValueError("Paper pointer and run manifest disagree on the frozen bundle.")
    noise_manifest = json.loads((args.noise_run / "manifest.json").read_text())
    if noise_manifest["bundle_sha256"] != manifest["frozen_bundle_sha256"]:
        raise ValueError("Noise benchmark used a different frozen bundle.")

    official = pd.read_csv(run / "official_benchmark_metrics.csv")
    internal = pd.read_csv(run / "internal_test_10pct_metrics.csv")
    noise = pd.read_csv(args.noise_run / "noise_metrics.csv")

    def clean(frame, subset, partition=None):
        query = frame[(frame.subset == subset) & (frame.family == "clean") & (frame.protocol == "capped_125")]
        if partition is not None:
            query = query[query.partition == partition]
        if len(query) != 1:
            raise ValueError(f"Expected one clean metric for {subset}/{partition}.")
        return query.iloc[0]

    ours = {s: float(clean(official, s).RMSE) for s in SUBSETS}
    ours_mae = {s: float(clean(official, s).MAE) for s in SUBSETS}
    internal_rmse = {s: float(clean(internal, s).RMSE) for s in SUBSETS}

    # Rank against third-party baselines only, never against a proposed row.
    best, rank = {}, {}
    for subset in SUBSETS:
        values = {name: table[subset] for name, table in THIRD_PARTY.items() if table[subset] is not None}
        best[subset] = min(values.values())
        rank[subset] = 1 + sum(value < ours[subset] for value in values.values())
    if rank != EXPECTED_RANK:
        raise ValueError(f"Third-party positions changed: {rank} != {EXPECTED_RANK}")

    # ---- Noise measurements -------------------------------------------------
    def noise_row(partition, sigma, dropout):
        query = noise[(noise.partition == partition) & (noise.subset == "POOLED")
                      & (noise.sigma_fraction == sigma) & (noise.dropout_rate == dropout)]
        if len(query) != 1:
            raise ValueError(f"Expected one noise row for {partition}/{sigma}/{dropout}.")
        return query.iloc[0]

    levels = [0., .05, .1, .2, .3, .5]
    rows = []
    for level in levels:
        off = noise_row("official", level, 0.)
        held = noise_row("heldout", level, 0.)
        label = "none" if level == 0 else f"{100 * level:.0f}\\%"
        rows.append([label, f"{off.RMSE:.2f}", f"{off.MAE:.2f}", f"{100 * off.interval_coverage:.1f}",
                     f"{held.RMSE:.2f}", f"{100 * held.interval_coverage:.1f}"])
    for dropout in (.02, .05):
        off = noise_row("official", 0., dropout)
        held = noise_row("heldout", 0., dropout)
        rows.append([f"{100 * dropout:.0f}\\% cells", f"{off.RMSE:.2f}", f"{off.MAE:.2f}",
                     f"{100 * off.interval_coverage:.1f}", f"{held.RMSE:.2f}",
                     f"{100 * held.interval_coverage:.1f}"])
    noise_table = table("CurrentNoiseTable",
        "Frozen-artifact accuracy under broadband sensor noise and channel dropout.",
        "tab:noise", "lrrrrr",
        r"Noise & Official RMSE & Official MAE & Cov. (\%) & Internal RMSE & Cov. (\%)", rows,
        r"Rows 1--6 add per-channel Gaussian noise; the level is a multiple of the fitting-only "
        r"within-regime sensor scale. Rows 7--8 instead drop whole cells at zero added noise. "
        r"Settings and speed commands are unperturbed. Noise is applied after model freeze; no "
        r"weight, threshold, corrector or interval radius was fitted on these levels. Coverage is "
        r"empirical clean endpoint/prefix coverage under the frozen radius, not a new guarantee.")

    # ---- Literature comparison ---------------------------------------------
    rows = [["GRACE-RUL (one shared model, no subset label)",
             *[rf"\textbf{{{ours[s]:.2f}}}" for s in SUBSETS]],
            ["Hybrid CNN--GRU (reported, five repeats)", *[f"{PAPER_A[s]:.1f}" for s in SUBSETS]],
            ["TRANS-D3 (reported)", *[f"{TRANSD3[s]:.2f}" for s in SUBSETS]],
            ["Best third-party baseline (reported)",
             *[f"{best[s]:.2f} ({BEST_THIRD_PARTY[s]})" for s in SUBSETS]],
            ["Reported value ahead of GRACE-RUL",
             *[("TRANS-D3 " + f"{TRANSD3[s]:.2f}") if TRANSD3[s] < ours[s] else "none" for s in SUBSETS]],
            ["Position among third-party baselines (proposed row excluded)",
             *[f"{rank[s]}" for s in SUBSETS]]]
    literature_table = table("CurrentLiteratureTable",
        "Reported C-MAPSS results at the 125-cycle target against the supplied studies.",
        "tab:literature", "lrrrr",
        r"Model & FD001 & FD002 & FD003 & FD004", rows,
        r"Reported columns are transcribed from the cited papers and are not recomputed here. "
        r"Rank counts only third-party baselines, so a proposed row cannot define its own benchmark. "
        r"The penultimate row names every reported value still ahead of this work, including the "
        r"TRANS-D3 proposed row where it applies; TRANS-D3 remains ahead on FD001 and FD003, so the "
        r"table does not establish first place overall. "
        r"This work is one shared model without a subset label, whereas the compared entries are "
        r"subset-specific. Official C-MAPSS tests were inspected in earlier development, so all "
        r"values are exploratory rather than a pristine holdout.")

    macros = {
        "CurrentNoiseOfficialZero": f"{noise_row('official', 0., 0.).RMSE:.2f}",
        "CurrentNoiseOfficialTwenty": f"{noise_row('official', .2, 0.).RMSE:.2f}",
        "CurrentNoiseOfficialFifty": f"{noise_row('official', .5, 0.).RMSE:.2f}",
        "CurrentNoiseHeldoutZero": f"{noise_row('heldout', 0., 0.).RMSE:.2f}",
        "CurrentNoiseHeldoutFifty": f"{noise_row('heldout', .5, 0.).RMSE:.2f}",
        "CurrentNoiseCoverageTwenty": f"{100 * noise_row('official', .2, 0.).interval_coverage:.1f}",
        "CurrentNoiseCoverageFifty": f"{100 * noise_row('official', .5, 0.).interval_coverage:.1f}",
        "CurrentNoiseDropoutTwo": f"{noise_row('official', 0., .02).RMSE:.2f}",
        "CurrentNoiseDropoutFive": f"{noise_row('official', 0., .05).RMSE:.2f}",
        "CurrentPaperAOursFdA": f"{ours['FD001']:.2f}",
        "CurrentPaperAFdA": f"{PAPER_A['FD001']:.1f}",
        "CurrentPaperBOfficial": f"{PAPER_B_POOLED:.2f}",
        "CurrentBestFdA": f"{best['FD001']:.2f}",
        "CurrentBestFdB": f"{best['FD003']:.2f}",
        "CurrentRankFdA": str(rank["FD001"]),
        "CurrentRankFdB": str(rank["FD003"]),
    }
    (OUT / "current_noise_macros.tex").write_text(
        "% Data-derived macros; no desired scores are assigned.\n"
        + "\n".join(rf"\newcommand{{\{name}}}{{{value}}}" for name, value in macros.items()) + "\n")

    rise = 100 * (float(noise_row('official', .2, 0.).RMSE) / float(noise_row('official', 0., 0.).RMSE) - 1)
    gains = {subset: PAPER_A[subset] - ours[subset] for subset in SUBSETS}
    min_gain, max_gain = min(gains.values()), max(gains.values())
    ahead = [subset for subset in SUBSETS if TRANSD3[subset] < ours[subset]]
    if all(rank[subset] == 1 for subset in SUBSETS):
        rank_sentence = (r"Among the third-party baselines transcribed in Table~\ref{tab:literature}, "
                         r"this work has the lowest RMSE on every subset. Its FD002 and FD004 values "
                         r"are also first among every reported row. TRANS-D3's proposed FD001 and "
                         r"FD003 values remain lower, so no first place overall is claimed.")
    else:
        rank_sentence = (r"Against the third-party baselines transcribed in Table~\ref{tab:literature}, "
                         rf"this work ranks {rank['FD001']} on FD001, {rank['FD002']} on FD002, "
                         rf"{rank['FD003']} on FD003 and {rank['FD004']} on FD004. "
                         + (r"TRANS-D3's proposed values remain ahead on "
                            + ", ".join(ahead) + r", so no first place overall is claimed." if ahead else
                            r"No transcribed proposed value is ahead of this work."))
    section = rf"""\subsection{{\textbf{{Broadband Sensor Noise and Channel Dropout}}}}
{noise_table}
\CurrentNoiseTable
The structured protocol models failures confined to related channels. Real
acquisition also carries broadband noise on every channel at once and occasional
missing cells. We therefore add an independent measurement layer that perturbs
raw sensor readings before the frozen fitting-only scaler: per-channel Gaussian
noise scaled to the within-regime sensor scale, plus independent cell dropouts.
Settings and speed commands stay untouched, and the layer is applied only after
freeze, so no weight, detector threshold, corrector or interval radius was fitted
on it. Levels are nested: one per-engine draw is scaled, so a sweep is paired
rather than resampled.

Table~\ref{{tab:noise}} shows graceful degradation. At 20\% of the within-regime
scale, pooled official RMSE moves from \CurrentNoiseOfficialZero{{}} to
\CurrentNoiseOfficialTwenty{{}} cycles, an increase of {rise:.1f}\%, and empirical coverage
remains \CurrentNoiseCoverageTwenty{{}}\%. At 50\% it is
\CurrentNoiseOfficialFifty{{}} cycles with \CurrentNoiseCoverageFifty{{}}\% coverage.
The internal partition follows the same pattern, from \CurrentNoiseHeldoutZero{{}}
to \CurrentNoiseHeldoutFifty{{}}. Modest dropout is not harmful: dropping 2\% of
cells leaves pooled official RMSE at \CurrentNoiseDropoutTwo{{}}, and 5\% gives
\CurrentNoiseDropoutFive{{}}. These are sensitivity measurements of one frozen
artifact, not a claim that any level matches a measured sensor specification.

\subsection{{\textbf{{Comparison with the Supplied Studies}}}}
{literature_table}
\CurrentLiteratureTable
Table~\ref{{tab:literature}} transcribes the reported values of the supplied
studies. {rank_sentence} The hybrid CNN--GRU study reports
13.1, 16.8, 14.5 and 18.9, so this model improves every subset by {min_gain:.2f} to {max_gain:.2f}
cycles at a comparable 125-cycle target.

Two qualifications matter more than the ordering. First, the compared entries are
subset-specific models, whereas this work uses one shared model without a subset
label, so the comparison is not budget-matched in the harder direction. Second,
the TRANS-D3 FD001 and FD003 values of 1.66 and 6.61 lie far below every
third-party entry in the same table, whose best values are \CurrentBestFdA{{}}
and \CurrentBestFdB{{}}; no capped C-MAPSS protocol in this study reaches them, so they are
retained for completeness rather than treated as a reproducible target. The
partial-failure study reports a single pooled fault-conditioned RMSE of
\CurrentPaperBOfficial{{}} with unspecified channel coverage and severity, which is
not decomposable by subset and is therefore not entered as a per-subset cell.
"""
    (OUT / "current_noise_section.tex").write_text(section)

    # ---- Figure -------------------------------------------------------------
    figure, axis = plt.subplots(figsize=(7.2, 2.7))
    for partition, style, label in [("official", "-o", "Official endpoints"),
                                    ("heldout", "-s", "Internal prefixes")]:
        values = [float(noise_row(partition, level, 0.).RMSE) for level in levels]
        axis.plot([100 * level for level in levels], values, style, label=label, linewidth=1.4, markersize=3.5)
    axis.set_xlabel("Sensor noise (percent of within-regime scale)")
    axis.set_ylabel("Capped RMSE (cycles)")
    axis.grid(alpha=.3, linewidth=.5)
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(OUT / f"current_noise_curves.{suffix}", bbox_inches="tight", pad_inches=.04, dpi=240,
                       metadata={"Title": "GRACE-RUL noise sensitivity", "CreationDate": None, "ModDate": None}
                       if suffix == "pdf" else None)
    plt.close(figure)
    figure_block = (r"\begin{figure*}[tp]\centering"
                    "\n" r"\includegraphics[width=0.86\textwidth]{paper_assets/current_noise_curves.pdf}"
                    "\n" r"\caption{Capped RMSE of the frozen artifact against broadband sensor noise. "
                    r"Every point is one evaluation of the same weights, scaler and interval "
                    r"radius, with the bounded corrector disabled; levels are paired draws, not "
                    r"independent realizations or training seeds.}"
                    "\n" r"\label{fig:noise_curves}\end{figure*}" "\n")
    (OUT / "current_noise_figure.tex").write_text(figure_block)

    provenance = {
        "created_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "scope": "Generated from saved measurements only; no training, inference or recalibration.",
        "run": str(run.relative_to(PAPER)), "bundle_sha256": manifest["frozen_bundle_sha256"],
        "noise_run": str(args.noise_run.relative_to(ROOT)),
        "noise_protocol": noise_manifest["protocol"],
        "inputs": {str(path.relative_to(PAPER)): {"sha256": sha(path), "bytes": path.stat().st_size} for path in
                   [run / "official_benchmark_metrics.csv", run / "internal_test_10pct_metrics.csv",
                    args.noise_run / "noise_metrics.csv", args.noise_run / "manifest.json"]},
        "outputs": {name: sha(OUT / name) for name in
                    ["current_noise_macros.tex", "current_noise_section.tex", "current_noise_curves.pdf",
                     "current_noise_curves.png", "current_noise_figure.tex"]},
        "literature_values_are_transcribed_not_recomputed": True,
        "literature_sources": {
            "hybrid_cnn_gru": "1787574110586-178711702.pdf, Table 4",
            "partial_sensor_failure": "1787574110595-209939890.pdf, Table 4",
            "trans_d3": "1787574864583-423798354.pdf, Table 7",
        },
        "ranking_scope": "Third-party baselines in the TRANS-D3 table only; proposed rows excluded.",
        "positions": rank,
    }
    (OUT / "current_noise_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    manuscript_assets = ROOT / "manuscript" / "paper_assets"
    if manuscript_assets.is_dir() and manuscript_assets.resolve() != OUT.resolve():
        import shutil
        for name in list(provenance["outputs"].keys()) + ["current_noise_provenance.json"]:
            src = OUT / name
            if src.is_file():
                shutil.copy2(src, manuscript_assets / name)

    print(json.dumps({"ours": {s: round(v, 2) for s, v in ours.items()}, "rank": rank,
                      "best_third_party": {s: round(best[s], 2) for s in SUBSETS},
                      "noise_official": {f"{100 * l:.0f}%": round(float(noise_row('official', l, 0.).RMSE), 2) for l in levels}},
                     indent=2))


if __name__ == "__main__":
    main()
