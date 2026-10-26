# Format 6: cross-regime sensor recovery repair

This preserves the format-6 implementation rationale. Current training uses
[format 7](RECOVERY_V7.md). Completed format-5/6 runs remain frozen local evidence;
the earlier prospective statements below describe the format-6 development stage.

## What changed

The fitting-only regime scaler still computes the same setting clusters, sensor
means, standard deviations, resolution floors, and raw-unit fault scales. Neural
sensor features now use `u = asinh(z)` by default, where `z` is the original regime
z-score. This transform is monotone and invertible, approximately linear near zero,
and logarithmic for large magnitudes. It retains the sign and size information of
a stuck reading without passing thousands of normalized units into recurrent and
attention layers. `sensor_transform_scale` defaults to one; `linear` is available
for a controlled ablation.

Fault simulation decodes the representation, applies the existing physical-unit
mechanisms, and re-encodes changed entries only. Unchanged entries retain their
exact bits. Settings, commanded speeds, masks, shared channels/onset, persistence,
and prefix causality are preserved. This changes the model representation rather
than reducing physical fault severity. The physical protocol remains
`structured-sensor-faults-v4`.

Changed-entry generator/repaired losses and affected-pair trend loss now use
Smooth L1, with beta one, to bound outlier gradients. Intact-entry identity still
uses MSE. Finite-fault BCE averages positive and negative classes within each
available sensor channel, then averages channels. Missing readings and padding
do not train this finite-fault classifier. Legacy weighted BCE and MSE remain
available for controlled ablations. No altered-entry label enters inference.

The input dimensions and neural modules remain unchanged: detector, residual
BiLSTM, local CNN-GRU, global Transformer, quality gate, and supervised OOF bounded
corrector. No TD3 or LQR is introduced. Training targets still cap at 125 cycles;
this repair does not solve the separate long-RUL target-horizon limitation.

## Checkpoint selection and fallback

Joint training includes corrupted-input RUL supervision with recovery bypassed
(weight 0.5, including auxiliary branch supervision). Both repaired and bypassed
paths are evaluated on the same clean and active-mixed validation endpoints.
Either can improve the retained pipeline, but the original pretrained clean RMSE
tolerance of 1% remains mandatory. Repair checkpoints must additionally satisfy:

- Altered-entry RMSE at most 1.5 in the declared model representation.
- Intact-entry RMSE at most 0.15 **before asinh compression**.
- Repaired/unrepaired altered-entry RMSE ratio at most one, also before compression.

These checks prevent compressed error values alone from making a damaging repair
look acceptable. `linear_*` metrics use the original regime z-score units; raw
sensor units differ by channel, so they are not pooled into one raw-unit RMSE.
`validation_path`, separate repair/bypass candidate scores, admissibility flags,
and uncompressed error metrics are recorded in training history.

After generator training, an accepted recovery checkpoint starts joint training.
If none qualified, the last finite detector/generator weights warm-start joint
optimization instead. The safe
deployable checkpoint is preserved separately. Rejection never automatically
exports the training-only warm start. Epoch checkpoints preserve this warm start,
optimizers, schedulers, RNG states, and the deployable best state for exact resume.

The website reverses the saved transform before showing raw sensor readings.
Updated finite-fault detector scores can be available even when recovery is
bypassed; the original constant initialization is marked unavailable. This is
availability of an estimate, not evidence of high detection accuracy.

## Validate before a full run

Use the existing `minor` environment from the repository directory:

```bash
source minor/bin/activate
python -m unittest discover -s tests -v
python scripts/check_recovery_fix.py
python scripts/check_recovery_fix.py --pilot --device cuda
```

The audit reads **training files only**, fits on fitting engines, and checks
physical equivalence, changed-entry/missingness semantics, and prefix invariance
on validation engines. The optional pilot trains the full tensor dimensions for
three epochs per stage, using 25 fitting and eight validation engines. It exercises
WGAN-GP and path selection but omits OOF correction, test evaluation, and paper
comparisons. These are limited execution/learning checks, not benchmark results.
Use `--device cpu` intentionally when CUDA is unavailable. Outputs and pilot
checkpoints stay under ignored `artifacts/` and are not published with source.

Then run a fresh, full-dimension one-seed study on the two affected subsets:

```bash
python scripts/check_cuda.py --training-check
python scripts/train_research.py --device cuda --subsets FD002 FD004 --seeds 42 --skip-tuning --skip-benchmarks --output-dir artifacts/recovery_v6_fd002_fd004_seed42
```

This command still runs complete phases, grouped OOF correction, calibration, and
official endpoint evaluation. It does not reuse the failed format-5 checkpoints.
Inspect retained paths, finite-fault validation metrics, uncompressed recovery
error, clean/active-mixed validation scores, and capped/uncapped test metrics before
investing in the five-seed study. Never tune new settings on inspected test results.

To run the normal four-subset, five-seed workflow with validation search:

```bash
python scripts/train_research.py --device cuda --output-dir artifacts/research_v6
```

## Compatibility and evidence

Format-5 bundles remain loadable with their original linear scalers and unchanged
neural definitions. Automatic website discovery accepts supported full format-5/6
bundles, excludes quick checks, and reports the loaded format. Formats below five
remain incompatible. All format-5 training checkpoints require fresh format-6
training; inference compatibility is not permission to resume a changed experiment.

Numerical input compression is not a demonstrated RUL improvement. Model-space
reconstruction RMSE across representations is not directly comparable: report the
representation, original-scale errors, fault exposure, and RUL outcomes. Existing
format-5 figures and manuscript results should not be relabeled as format-6 results.
