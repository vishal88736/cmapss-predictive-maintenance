# Improved research training

The format-5 pipeline turns the three papers' ideas into a trainable recovery and
prediction system. Changes improve training signals and model selection; their
accuracy benefit remains an experimental question.

| Paper idea | Implemented improvement |
| --- | --- |
| A: CNN–GRU, dense windows, grouped validation | Stride-1 full training, engine-balanced sampling, fixed partitions across seeds, three validation prefixes per engine |
| B: reconstruction plus RUL feedback | Residual BiLSTM recovery initialized to identity; direct generator reconstruction and affected-pair temporal-trend losses; class-balanced finite-fault detection |
| B: staged WGAN-GP | Configurable losses, validation-driven learning-rate reduction, and checkpoint limits on clean RUL, intact-entry distortion, and altered-entry reconstruction |
| C: bounded prediction correction | Engine-disjoint OOF features, six correction prefixes per engine, validation-selected 5/10/15/30-cycle bounds, and zero-correction fallback |
| A/B/C: empirical comparison | Separately trained architecture adaptations and GRACE ablations on identical data, target conventions, engines, and corruption draws |

These extensions are project design choices, rather than findings established by
the papers. The residual generator returns observed zero-filled sensors plus a
learned adjustment. A zero-initialized decoder preserves finite readings initially;
missing readings start at the fitting-regime mean. Trend supervision compares
consecutive real timesteps touching a changed entry and excludes padding.

Default loss weights are RUL 1, repaired reconstruction 1, direct generator
reconstruction 0.25, identity 0.5, detection 0.2, trend 0.1, adversarial 0.01,
consistency 0.1, and branch supervision 0.1. Detector positive weights use the
square root of negative/positive counts, bounded between 1 and 5 per channel.
Recovery is accepted only if clean validation RMSE grows at most 5%, intact-entry
normalized RMSE is at most 0.15, altered-entry RMSE is at most 1.5, and the macro
RUL score improves. These initial limits are configurable research choices.
If no repair candidate passes, the pretrained predictor is retained and the saved
`repair_enabled` buffer is false. This fallback also lacks trained finite-fault
detection; the website distinguishes predictor input from recovered readings.
Correction separately protects clean RMSE and
can retain zero correction. The website reports the selected repair status and bounds.

Validation-only search tries defaults, stride 5, stronger recovery/trend, greater
identity protection, and zero adversarial loss. It opens only training files,
fits on fitting engines, and selects using validation engines; calibration and
official test labels are excluded. The search selects upstream settings before
OOF correction and uses one common training seed. Model seeds are selected after
correction. The fixed split seed is 42; fixed validation corruption seed is 31415.
Every seed and comparator shares the selected per-subset settings. This modest
search does not establish an optimal configuration.

`ReduceLROnPlateau` lowers the learning rate by 0.5 with patience 5,
down to 1e-6. Epoch checkpoints include scheduler state and resume without repeating
updates. Engine-balanced training uses a seeded weighted sampler with reciprocal
window counts; each engine has equal total sampling weight. See the official
[scheduler documentation](https://docs.pytorch.org/docs/2.14/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html)
and [sampling documentation](https://docs.pytorch.org/docs/2.14/data.html).

## Run the workflow

```bash
bash scripts/setup_env.sh --cuda
.venv/bin/python scripts/check_cuda.py --training-check
.venv/bin/python scripts/train_research.py --device cuda --output-dir artifacts/research_v5
```

This defaults to FD001–FD004, five seeds, validation search, and three grouped OOF folds.
There are up to 20 search fits and 80 primary upstream/fold fits. Separate comparison
training and extended robustness/ablation reports are disabled by default. The model,
training losses, epoch budgets, validation, calibration, and seed selection are unchanged.
Enabling the six comparison variants adds 360 upstream/fold fits of differing sizes
and training stages, so fit counts alone do not determine wall-clock savings.
Early stopping can reduce epochs. A functioning CUDA PyTorch environment is needed
for a practical full study. CPU execution requires explicit `--device cpu`.

For the complete comparison and reporting workflow:

```bash
.venv/bin/python scripts/train_research.py --device cuda --benchmarks --extended-evaluation --output-dir artifacts/research_with_reports
```

The notebook exposes the same switches through `CONFIG['run_matched_benchmarks']`
and `CONFIG['run_extended_evaluation']`. Its optional SHAP/LIME example uses
`RUN_EXPLANATIONS = True`; on-demand website explanations remain available.

```bash
.venv/bin/python scripts/train_research.py --quick --device cpu --output-dir /tmp/grace-research-check
```

The quick CLI deliberately checks tuning and comparisons on tiny dimensions/data.
Quick notebook mode checks the main pipeline and skips those additional experiments.
Neither quick run qualifies as research performance evidence.

`--config settings.json` overrides known configuration keys; `--subsets`, `--seeds`,
and `--benchmark-variants` choose a reduced study. Specifying comparison variants
enables benchmarks unless `--skip-benchmarks` is also given. `--skip-tuning` and
`--skip-benchmarks` skip those steps explicitly. Rerunning with the same output
directory resumes matching upstream checkpoints. Report switches do not invalidate
completed epochs; changed training data/settings require a
fresh directory. OOF collection, correction search, and evaluation rerun.

## Evaluate the evidence

The three default paper-inspired comparators are local CNN–GRU, BiLSTM recovery
plus CNN–LSTM, and Transformer plus supervised residual correction. They share
preprocessing and training settings and adapt the architectures to this project's
protocol. The Transformer comparator does **not** reproduce original TD3.
The repair comparator shares the estimated detector and residual decoder rather
than reproducing Paper B's precise recovery implementation. Original reported
paper scores cannot be compared directly with these experiments. Independent
baseline tuning under an equal budget remains necessary for a strong superiority claim.

Trained GRACE ablations remove adversarial loss, temporal-trend loss, and repair.
Optional variants include pure Transformer and separately trained local/global
predictors. Existing `ablation_metrics.csv` contains inference interventions and
retains that label; these are different experiments.

Core outputs include training/correction histories, clean test metrics, calibration,
seed selection, upstream cross-validation, and the model bundle. Optional outputs include:

- `validation_search.csv` and `tuning/search_manifest.json`: trial scores, selected
  settings, partitions, and training-file hashes.
- `test_seed_summary.csv`: clean test mean/std across training seeds, separate
  capped-125 and official-uncapped target conventions.
- `robustness_summary.csv` and `ablation_metrics.csv`: generated only with extended evaluation.
- `benchmark_metrics.csv`, `benchmark_summary.csv`, and compressed predictions (with benchmarks enabled):
  every comparator, seed, condition, and corruption replicate.
- `benchmark_comparison.csv`: paired RMSE differences and 95% bootstrap intervals.
- `superiority_checks.csv`: corrupted macro improvement, uncertainty, clean-data
  preservation, protocol completeness, and the predeclared acceptance check.
- `benchmark_training_audit.csv`: actual trained epochs, selected repair and bounds.
- `seed_models/`: every primary seed model; the website bundle contains the
  validation-selected seed per subset.

Bootstrap resamples whole engines and training seeds using the same indices across
variants and conditions. Each engine's corruption replicates stay together. The
corrupted macro weights subsets, fault families, and severities equally. A full
acceptance check requires all four subsets, at least five distinct seeds, no quick
data/batch limits, all specified conditions, at least 10% macro improvement, a
positive lower improvement interval, and clean RMSE within 5% per subset.
Individual regressions remain visible. Acceptance is against harmonized adaptations;
`published_paper_superiority_established` remains false in the manifest.

No full format-5 study is saved in the project's model slot. Verification results
and the limited FD001 validation pilot are recorded in [VALIDATION.md](VALIDATION.md).
