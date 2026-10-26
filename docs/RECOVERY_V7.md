# GRACE-RUL format 7: repair guide and retraining protocol

Current healthy-sensor protections and the frozen-weight upgrade are documented in
[RECOVERY_V8.md](RECOVERY_V8.md). This guide preserves the format-7 implementation.

Format 7 repairs confirmed training and reporting inconsistencies found in the
FD002/FD004 format-6 review. It does not establish improved final test accuracy,
successful sensor recovery, or real-fleet coverage. A fresh full CUDA run remains
necessary to measure the revised pipeline.

The artifact architecture identifier is
`validated-detector-linear-identity-gated-cnn-gru-transformer-residual-v7`.
The physical fault simulator remains `structured-sensor-faults-v4`.

## Confirmed problems and source changes

| Format-6 finding | Format-7 change | Practical limit |
| --- | --- | --- |
| Intact-sensor identity loss used compressed asinh values, while checkpoint acceptance measured displacement in original linear regime z-score units. Small compressed errors could hide large original-scale changes. | `intact_identity_loss` in `grace_rul/training.py` evaluates intact displacement after the inverse transform, normalized by the intact-error budget. The stable float64 sinh-difference calculation masks faulty cells before inversion. | The intact RMSE guard stays at 0.15 linear z-score units; it is not relaxed to force repair selection. |
| The exported clean-predictor fallback could discard a learned detector. Joint RUL/reconstruction gradients could also alter the detector to reduce repair exposure. | The detector is fitted separately, selected using validation finite-fault labels, and frozen during generator/joint fitting. Gate scores are detached from recovery/RUL gradients. The learned detector remains available when repair is bypassed. | Updated weights establish an available detector, not accurate or physically calibrated fault probabilities. |
| Threshold validation did not protect intact channels in every structured-fault context. | Per-sensor thresholds maximize validation context F1 subject to empirical false-positive rate no greater than 5% in each channel/context. Validation covers all nine contexts: clean, intermittent, outage, block, noise, bias, drift, stuck, and mixed. | The bound describes the selected validation cells and simulator settings. It does not guarantee a fleet or official-test false-positive rate. |
| A continuous finite-sensor gate could modify apparently healthy measurements. | A finite channel receives repair weight only when its detached score reaches its validated threshold. Threshold 1 explicitly disables that channel, including a saturated score of 1. Missing readings remain eligible for reconstruction when the repair path is enabled. | False-positive repairs remain possible; checkpoint intact-error and clean-prediction guards still apply. |
| RMSE-only acceptance could tolerate deterioration in the asymmetric NASA score. | Later pipeline stages and residual correction share a guard on clean validation `NASA_Score_Mean`: at most 1% above the original selected clean-predictor reference. The clean RMSE guard also stays at 1%. | The score used for selection is still clean/corrupted validation RMSE. Validation guards cannot guarantee matching official-test behavior. |
| One randomly selected calibration prefix per engine and interpolated signed percentiles provided a weak endpoint contract; capped offsets were also reused for raw targets. | Separate capped/raw split-conformal contracts use one maximum absolute error over all observed prefixes per held calibration engine. The order statistic uses independent engine scores without percentile interpolation. | This conservative calibration can produce wide intervals. Exchangeability and scope assumptions remain explicit. |
| Capped predictions were displayed as unrestricted remaining life and physical failure-cycle forecasts. Cycle gaps were accepted although model timesteps represent consecutive cycles. | Website reports expose the cap, target convention, separate capped/raw metrics, and a horizon-limited cycle proxy. Uploads require consecutive cycles. | A raw-target interval does not turn a capped-trained point predictor into a validated long-horizon predictor. |

Repair and bypass candidates are compared on validation data. The history now
retains repair-specific clean RMSE, NASA score, original-scale intact error, and
individual guard decisions even when the bypass candidate wins. An admissible
predictor fallback remains valid behavior; format 7 does not force the generator
into deployment.

## Architecture and training order

```text
Raw operating settings + 21 sensors, with missingness preserved
  -> fitting-only operating-regime normalization -> asinh sensor encoding
  -> observed history window: 70 positions, with padding mask
  -> separately trained, frozen CNN-GRU finite-fault detector
  -> validated per-channel finite gate + directly observed missingness mask
  -> residual BiLSTM sensor generator -> selective reconstruction
  -> local CNN-GRU (30 positions) + global Transformer (70 positions)
  -> quality-conditioned fusion -> nonnegative capped-target RUL
  -> bounded supervised residual corrector trained from grouped OOF predictions
  -> capped/raw interval contracts + optional genuine Kernel SHAP/LIME
```

The clean predictor is fitted first. Separate detector fitting then uses only
fitting engines for optimization and validation engines for threshold/checkpoint
selection. Its parameters and thresholds are frozen for generator and joint
training. Generator and joint stages can retain the clean or bypassed predictor
when the reconstruction and clean-prediction guards reject recovery.

The residual corrector uses three grouped upstream folds with fold-specific
fitting-only preprocessing. Its validation selection shares the original clean
RMSE/NASA references rather than receiving a new tolerance after each stage.
Independent calibration engines enter neither optimization nor model selection.
Setting normalization statistics, regime centers, fault scales, and explanation
backgrounds are fitted or selected from fitting data only.

Corruptions continue to model related sensor/process faults: temporal outages,
stuck measurements, persistent bias, drift, and correlated disturbances, including
operating transitions and sensor relationships. Stochastic event sampling remains
part of this simulator; it is not evidence that every real sensor fault follows
the simulated distribution. See [STRUCTURED_SENSOR_FAULTS.md](STRUCTURED_SENSOR_FAULTS.md).

## Interval interpretation

For each reserved calibration engine, the default `engine_prefix_max` policy
evaluates every observed prefix, including short padded histories. For each target
convention, it stores the engine's largest absolute RUL error. With `n` independent
engine scores and `alpha = 0.1`, the selected rank is
`ceil((n + 1) * (1 - alpha))`.

When this rank exceeds `n`, the interval is `[0, infinity)`. The bundle stores a
null radius/offsets with `unbounded: true`; the website emits `interval_upper: null`,
`interval_width: null`, and `interval_unbounded: true`, and displays an infinity
symbol. If any engine interval is unbounded, the report's mean interval width is
also null and its unbounded count remains explicit. Small execution-check runs
must not substitute a finite maximum score to hide insufficient calibration data.

This limitation also affects full default FD001/FD003 runs: each subset has 100
training engines, the outer split leaves an 80-engine training pool, and the 10%
calibration reservation contains only 8 engines. At nominal 90% coverage, rank
`ceil(9 * 0.9) = 9` exceeds 8, so the honest default interval is unbounded even
after full training. A future protocol seeking a finite 90% radius must reserve
at least 9 independent calibration engines in advance, with the resulting smaller
fitting partition declared explicitly. Do not increase the reservation after
examining coverage results. FD002/FD004 have 21/20 calibration engines under the
defaults used by the command below, so this particular sample-size obstruction
does not apply to those subsets.

Under a fixed model/preprocessing and exchangeable complete engine trajectories,
the maximum-prefix score supports simultaneous clean-prefix coverage. Dataset
membership alone does not establish that assumption. Fleet shift, censoring
shift, corruption, and out-of-distribution inputs require separate validation.
The displayed nominal level is distinct from measured coverage.

The primary protocol is `capped_125` under the default training cap; the second
protocol is `official_uncapped`. They use separate residual scores. Legacy
format-5/6 bundles retain their original capped interval behavior. Their raw-target
coverage is unavailable rather than computed from incompatible capped offsets.

A training-only calibration audit applied this method to **frozen format-6
seed-42 models**, without modifying their checkpoints or evaluating official test
data:

| Subset | Calibration engines | Old capped signed offsets | New capped radius | New raw radius |
| --- | ---: | --- | ---: | ---: |
| FD002 | 21 | [-3.7499, +9.9240] | 51.3708 | 147.7940 |
| FD004 | 20 | [-27.7422, +9.7585] | 53.8929 | 178.2754 |

The old offsets were asymmetric empirical quantiles, not symmetric conformal
radii. The wider bounds reflect the stronger simultaneous-prefix objective and
the capped point predictor's raw-target errors. These values are calibration
measurements on old models, not format-7 accuracy or verified official-test/fleet
coverage. Local evidence is in
`artifacts/repair_v7_audit_20261026/calibration/summary.json`.

## Available format-7 execution evidence

The saved CPU pilot at
`artifacts/repair_v7_audit_20261026/pilot_final/report.json` used the full model widths
with 25 fitting and 8 validation engines per subset. Predictor, generator, and
joint stages had three epochs each; the separate detector had up to three epochs.
The pilot limited optimization to 256 windows, at most eight batches per epoch,
and stride 5. It did not evaluate official test data or establish a complete
OOF-corrected benchmark.

| Subset | Selected detector validation macro F1 | Maximum channel/context validation FPR | Selected detector epoch | Repair exported by pilot |
| --- | ---: | ---: | ---: | --- |
| FD002 | 0.10649 | 0.04814 | 3 | Bypassed |
| FD004 | 0.07558 | 0.04948 | 2 | Bypassed |

Macro F1 averages contexts containing finite faults; the maximum FPR is taken
over channel/context pairs with intact finite cells. These modest F1 values show
that a learned detector and the conservative selection path execute; they do not
establish useful field detection or improved RUL. Both pilot models retained
repair bypass. A fresh full run must determine whether recovery qualifies.

Focused regression checks cover scale-aligned identity gradients, detached and
disabled gates, per-context threshold guards, original-reference NASA acceptance,
engine-level conformal ranking, target-specific coverage, checkpoint boundaries,
and website JSON/input contracts. JSX parsing/transpilation was checked with an
already-installed local esbuild. This checkout has no installed Vite dependencies,
so that check is not a production build or visual-browser validation.

## Fresh CUDA run in the `minor` environment

From the repository root, activate the existing environment and verify CUDA
before starting a long run:

```bash
cd "/home/vishal/D DRIVE/minor/cmapss-predictive-maintenance"
source minor/bin/activate
python scripts/check_cuda.py --training-check
```

After that check passes, run the primary FD002/FD004 experiment in a new directory:

```bash
set -o pipefail
run_dir="artifacts/recovery_v7_$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "$run_dir"
python -u scripts/train_research.py \
  --device cuda \
  --subsets FD002 FD004 \
  --seeds 42 \
  --skip-tuning \
  --skip-benchmarks \
  --extended-evaluation \
  --output-dir "$run_dir" \
  2>&1 | tee "$run_dir/training.log"
```

The command uses full default widths and budgets, disables the separate
hyperparameter search/comparison fits, and saves clean, robustness, and inference
ablation reports. It is a single-seed primary run, not the full multi-seed paper
study. An explicit CUDA request fails if CUDA is unavailable rather than silently
switching the long run to CPU. `pipefail` preserves a failing training exit status
when output is piped through `tee`.

Use the same recorded directory and unchanged training configuration to resume
an interrupted format-7 run. Do not reuse a format-6 checkpoint directory.
Inference can still load supported format-5/6 bundles; format-7 training has a new
checkpoint identity and learned threshold buffer.

`--paper-study` is optional. It trains additional architectural adaptations and
comparators; it is not required to train the primary model and does not improve
already-trained primary weights by itself. Strong superiority claims require
matched comparisons, all required subsets/seeds, and the declared checks.

## What to review after training

Check `training_history.csv` for original-reference RMSE/NASA guards, repair/bypass
selection, and intact/reconstruction errors. Inspect `detector_history_<subset>_42.csv`
and `detector_audit_<subset>_42.json` for selected F1, channel/context FPR, and disabled
thresholds. Review `correction_history.csv`, `seed_summary.csv`, calibration
contracts/prefix records, and both target conventions in `test_metrics.csv` and
`robustness_summary.csv`. Separate all-endpoint from active-fault results and
nominal interval levels from observed coverage.

FD002/FD004 official test results have already informed successive redesigns.
Subsequent results on those reused test sets are exploratory, even though source
selection uses validation engines. Freeze the next protocol before running it and
report this test-set reuse; source changes cannot restore an unseen holdout.

Historical bundles, reports, and the paper remain historical records and are not
overwritten with pilot or hypothetical future results. Publish source changes
only; generated runs, model weights, datasets, and audit artifacts remain local.
The repairs address the identified failure mechanisms, with remaining performance
and calibration limitations measured and reported rather than assumed resolved.
