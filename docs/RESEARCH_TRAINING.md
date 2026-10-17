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
consistency 0.1, and branch supervision 0.1.
Additional clean-reference RUL supervision has weight 0.25 during joint training;
the reference pass disables dropout and contributes gradients only to the predictor.
The corruption curriculum reaches full difficulty within ten epochs, before the
default early-stopping patience expires. Detector positive weights use the
square root of negative/positive counts, bounded between 1 and 5 per channel.
Recovery is accepted only if clean validation RMSE grows at most 1%, intact-entry
normalized RMSE is at most 0.15, altered-entry RMSE is at most 1.5, and the macro
RUL score improves. These initial limits are configurable research choices.
If no repair candidate passes, the pretrained predictor is retained and the saved
`repair_enabled` buffer is false. This fallback also lacks trained finite-fault
detection; the website distinguishes predictor input from recovered readings.
Correction uses the same original clean-predictor reference, preventing consecutive
stages from each spending another copy of the tolerance, and
can retain zero correction. The website reports the selected repair status and bounds.

Validation-only search tries defaults, stride 5, stronger recovery/trend, greater
identity protection, zero adversarial loss, and a larger clean fraction/anchor. It opens only training files,
fits on fitting engines, and selects using validation engines; calibration and
official test labels are excluded. The search selects upstream settings before
OOF correction and uses one common training seed. Model seeds are selected after
correction. The fixed split seed is 42; fixed validation corruption seed is 31415.
Primary seeds share selected per-subset settings. Paper adaptations instead receive
their own six-trial searches on the same engine partitions. Their candidates tune
learning rates, context, stride, clean fraction, or reconstruction as appropriate;
the WGAN comparator keeps a positive adversarial weight. Controlled GRACE ablations
retain the selected primary settings. This modest
search does not establish an optimal configuration.

Sensor corruption uses `structured-sensor-faults-v4`. Related measurement domains
replace arbitrary channel sets; speed commands and settings are preserved. Added
noise is temporally correlated with common/individual innovations. Bursty missingness
shares a contact-loss state within a domain. Observed regime transitions increase
noise amplitude and burst-entry risk. Bias/drift are additive physical-unit offsets;
stuck sensors hold a physical reading across changes in regime. Mixed faults share
channels/onset and combine contact-noise/dropout, calibration-bias/drift, or acquisition
freeze/block mechanisms. These groupings and parameters are simulation assumptions.
The fitting-only scaler saves `sensor_fault_scale`: median within-regime p10–p90
range divided by 2.563, floored by observed resolution or an explicit constant-channel
fallback. This is not a measured sensor error specification. Full equations and
limitations are in [STRUCTURED_SENSOR_FAULTS.md](STRUCTURED_SENSOR_FAULTS.md).

Onsets are sampled from absolute observed offsets 0 through `fault_start_horizon`
(100), independently of failure time or total trajectory length. Drift grows over
`drift_ramp_cycles` (100), then stays at its sampled additive offset. Missing blocks
recur after gaps between half of `block_gap_cycles` and that maximum (50 cycles).
An onset beyond observed history has no effect. Independent mechanism streams
preserve prefixes in mixed scenarios; overlapping windows share cached faults.
`fault_noise_autocorrelation` (.9), `fault_transition_gain` (1), and
`intermittent_mean_burst` (4 cycles) are configurable and included in training identity.

The timing controls can be changed in notebook configuration or a CLI JSON config.
This is a revised training-data protocol: pre-correction checkpoints cannot resume
under it, even when the tensor architecture is still format 5. Use a fresh run
directory. New epoch identities, bundles and run manifests record the protocol;
existing format-5 weights can still be read for inference under their original
training semantics. Accuracy under the corrected protocol requires a new run.

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
There are up to 24 search fits and 80 primary upstream/fold fits. Separate comparison
training and extended robustness/ablation reports are disabled by default. The model,
epoch budgets, grouped correction and calibration remain intact. Comparisons add
independent baseline searches and controlled ablations of differing sizes/stages.
The no-correction ablation reuses unchanged upstream weights; identical selected
ablations can reuse their exact weights. Fit counts alone do not determine wall-clock savings.
Early stopping can reduce epochs. A functioning CUDA PyTorch environment is needed
for a practical full study. CPU execution requires explicit `--device cpu`.

For the complete comparison and reporting workflow:

```bash
.venv/bin/python scripts/train_research.py --device cuda --paper-study --output-dir artifacts/research_faults_v3
```

The notebook exposes `RUN_PAPER_STUDY = True` and the individual report switches.
`RUN_EXPLANATIONS = True` now calculates local explanations and global sensor importance
over distinct held-out engines, repeated with different perturbation seeds. Fitting
backgrounds are stratified across engines, regimes and life stages; provenance is in
`background_audit.csv`. Full defaults use 48 backgrounds, 12 explained engines per
subset and three explanation seeds. SHAP sparsity is not forced to ten features;
SHAP additivity, LIME local error/fidelity and rank agreement are recorded separately.
On-demand website explanations remain available. Existing trained weights can be
explained separately without retraining:

```bash
.venv/bin/python scripts/explain_research.py artifacts/<run>/grace_rul_ALL.pt --device cuda
```

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
positive lower improvement interval, and clean RMSE within the configured tolerance
(default 1%) per subset. Paper-comparator acceptance also requires independent tuning.
Individual regressions remain visible. Acceptance is against harmonized adaptations;
`published_paper_superiority_established` remains false in the manifest.
`benchmark_active_comparison.csv` adds paired confidence intervals on faults present
in both compared models' observed windows. The bootstrap retains whole engines and
their corruption replicates. Acceptance also requires at least 10% active-fault macro
improvement with a positive lower interval and complete active-scenario coverage.

No full format-5 study is saved in the project's model slot. Verification results
and the limited FD001 validation pilot are recorded in [VALIDATION.md](VALIDATION.md).
The v3 repair checks and current neural-validation limitations are recorded in
[the October 15 repair record](REPAIR_VALIDATION_2026-10-15.md). Fault reports now retain
capped and official uncapped targets separately, plus active-fault and unaffected
endpoint metrics. Selection averages clean and active-mixed-fault RMSE; validation
faults are fixed independently of the training clean fraction. The three-fold tables
remain upstream CV, not nested CV of the final corrected model.
