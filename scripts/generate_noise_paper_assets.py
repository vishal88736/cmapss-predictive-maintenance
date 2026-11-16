"""Generate the broadband-noise section from saved measurements.

Only saved measurements of the frozen artifact are used: the completed
broadband-noise benchmark and the run manifest that binds it to the exported
bundle. No training, inference or recalibration is performed.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT.parent
OUT = PAPER / "paper_assets"


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

    noise = pd.read_csv(args.noise_run / "noise_metrics.csv")

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
    }
    (OUT / "current_noise_macros.tex").write_text(
        "% Data-derived macros; no desired scores are assigned.\n"
        + "\n".join(rf"\newcommand{{\{name}}}{{{value}}}" for name, value in macros.items()) + "\n")

    rise = 100 * (float(noise_row('official', .2, 0.).RMSE) / float(noise_row('official', 0., 0.).RMSE) - 1)
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
"""
    (OUT / "current_noise_section.tex").write_text(section)

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
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Generated from saved noise measurements only; no training, inference or recalibration.",
        "run": str(run.relative_to(PAPER)), "bundle_sha256": manifest["frozen_bundle_sha256"],
        "noise_run": str(args.noise_run.relative_to(ROOT)),
        "noise_protocol": noise_manifest["protocol"],
        "inputs": {str(path.relative_to(PAPER)): {"sha256": sha(path), "bytes": path.stat().st_size}
                   for path in [args.noise_run / "noise_metrics.csv", args.noise_run / "manifest.json"]},
        "outputs": {name: sha(OUT / name) for name in
                    ["current_noise_macros.tex", "current_noise_section.tex", "current_noise_curves.pdf",
                     "current_noise_curves.png", "current_noise_figure.tex"]},
        "noise_levels_are_paired_draws": True,
    }
    (OUT / "current_noise_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    manuscript_assets = ROOT / "manuscript" / "paper_assets"
    if manuscript_assets.is_dir() and manuscript_assets.resolve() != OUT.resolve():
        import shutil
        for name in list(provenance["outputs"].keys()) + ["current_noise_provenance.json"]:
            src = OUT / name
            if src.is_file():
                shutil.copy2(src, manuscript_assets / name)

    print(json.dumps({"noise_official": {f"{100 * l:.0f}%": round(float(noise_row('official', l, 0.).RMSE), 2)
                                        for l in levels}}, indent=2))


if __name__ == "__main__":
    main()
