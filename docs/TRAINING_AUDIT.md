# Training audit — 11 October 2026

This records the initial format-4 audit and cleanup. The subsequent authorized
improvements advance the pipeline to **format 5**; see
[RESEARCH_TRAINING.md](RESEARCH_TRAINING.md) and [VALIDATION.md](VALIDATION.md).

The three supplied PDFs were checked against data preparation, the model,
staged optimization, correction folds, checkpoint selection, and notebook output.
The integration is implemented, but superiority over the papers is unproven.

## How the papers are integrated

| Supplied paper | Implemented contribution | Adaptation or limitation |
| --- | --- | --- |
| Özüpak: hybrid CNN–GRU | 30-cycle CNN–GRU branch, grouped validation, repeated seeds, SHAP/LIME | Added quality gating, explicit detection, and recovery; initial audit used stride 5, subsequently changed to full-mode stride 1 with validation search. |
| Tang: partial sensor failure | Paired trajectory corruption, BiLSTM recovery, conditional recurrent WGAN-GP critic, staged and task-aware optimization | A learned detector supplies repair weights at deployment. The clean trajectories are simulation references, not perfectly noiseless measurements. |
| Paredes et al.: Trans-D3 | Transformer branch and bounded, penalized additive correction | The implementation uses supervised residual correction, not TD3. Sections 3.1.2 and 3.3 describe true RUL/test values in correction, while the state equation describes prediction and sample index. Deployment must use observable inputs. |

See [the architecture analysis](FINAL_ARCHITECTURE.md) for PDF filenames, reported
results, and method details. The [archived Trans-D3 article](https://pmc.ncbi.nlm.nih.gov/articles/PMC13210881/)
also confirms the correction-state inconsistency. Tang's
[September correction](https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0357561&type=printable)
was retrieved during the 14 October follow-up and verified to correct the funding
statement only. See [the current integration audit](INTEGRATION_AUDIT_2026-10-14.md)
for the renewed paper/code review and the open fault-simulation protocol finding.

## Verified problems and changes

1. **Unaligned correction inputs.** Independent fold models supplied hidden
   embeddings whose coordinates need not agree. Removed those embeddings from
   the corrector. Its 67 inputs now contain sensor quality summaries and three
   scaled RUL predictions with consistent meanings across folds.
2. **Correction and seed selection.** Corrector checkpoints used mixed validation
   alone, and final seeds were selected before correction. Both now use the
   final pipeline's average clean/mixed validation RMSE. Correction fitting
   includes clean and mixed examples. An epoch-zero, zero-correction candidate
   prevents accepting correction when every candidate worsens this objective.
3. **Joint-stage regression.** Joint fine-tuning could discard a better repair
   checkpoint. That checkpoint is now an eligible joint-stage candidate.
4. **Dropout during repair supervision.** A frozen predictor still used dropout,
   and consistency reference targets were stochastic. Predictor dropout is
   disabled during repair training and all reference target passes. Its GRU
   stays in training mode for cuDNN gradients into the generator; that GRU has
   no dropout. The reference pass restores every module's previous mode.
5. **Contradictory drift labels.** Multiplicative drift marked unchanged zero
   readings and its unchanged first ramp point as faulty. Labels now identify
   measurements actually changed, including missingness.
6. **Extreme normalization of quantized channels.** With the previous scaler,
   official FD002 sensor_10 reached approximately 54 standard deviations.
   Sensor scales now have fitting-only measurement-increment floors, recorded
   in the scaler. For the same seed-42 partition, sensor_10's maximum absolute
   standardized test value is approximately 1.00, and the maximum across all
   sensors is 6.56. Constant channels use a scale of 1.
7. **Expensive recurrent gradient penalty.** The recorded interruption happened
   inside `penalty.backward()`, which computes second derivatives through the
   critic. Gradient penalty now uses a uniformly sampled sub-batch of up to 16;
   the Wasserstein term still uses the entire batch. This is a stochastic
   estimate of the same regularizer, not a removal of adversarial training.
8. **Incomplete robustness coverage.** Full runs now also evaluate explicit
   two-family mixed faults at each configured severity and replicate. These
   combinations are not held out from training.

Training rejects nonpositive epoch/batch budgets before creating a run and stops
on nonfinite losses or gradients. Architecture/bundle format is now **4**;
old checkpoints and bundles require fresh training. Subsequent improvements use format 5.

## What the previous run established

The notebook ended with `KeyboardInterrupt` during FD001 seed 2026, correction
fold 1, generator epoch 28. Its checkpoint retained 27 completed generator epochs.
This was a manual interruption, not evidence of a programming exception.

Only seeds 42 and 123 had saved official FD001 predictions:

| Seed | Capped-at-125 RMSE | Official uncapped RMSE |
| --- | ---: | ---: |
| 42 | 13.183 | 14.437 |
| 123 | 12.983 | 14.049 |

Those are historical partial results from the old implementation. They are not
results of the corrected pipeline, a completed four-subset study, or a matched
comparison against the three papers. No final bundle or run manifest existed.

## Cleanup

Removed the interrupted `artifacts/full_20261008T175019310667Z/` run: ten old
checkpoints and four partial prediction/calibration CSVs, 61,645,351 bytes
(58.8 MiB). Removed its plot cache and regenerated the notebook with no execution
output, including removal of its interrupt traceback and embedded plots.

Removed `data/x.txt`: its 218 values exactly matched the first 218 of the 259
values in `RUL_FD002.txt`. It was an unused truncated duplicate, not another
training dataset. Preserved all 12 official train/test/RUL files and all papers.
Generated `artifacts/` files are now ignored by Git. Audit verification runs use
`/tmp` so they do not populate the project's model slot with test artifacts.

## What is needed to establish a better model

Run fresh full training across FD001–FD004 and five seeds. Retrain the three
paper approaches and module ablations with the same engine partitions,
normalization, target convention, windowing, and corruption cases. Report mean
and standard deviation, RMSE/NASA score, recovery error, clean-sensor distortion,
and detector precision/recall at the selected thresholds. Existing inference
interventions cannot replace retrained baselines.

Validation-only tuning of stride 1 versus 5, repair/adversarial loss weights, and
correction bounds is now implemented in format 5. Detector usefulness on finite
noise still needs full evaluation. A larger
network or more paper modules does not establish better generalization. Keep
official test labels out of every optimizer and selection decision. The fixes
improve the training procedure; their effect on full benchmark accuracy must
still be measured.

Validation results are recorded in [VALIDATION.md](VALIDATION.md). The initial
audit used CPU PyTorch in a workspace that hid GPU devices. Subsequent authorized
host checks confirmed the RTX 4050, installed CUDA PyTorch, and verified GPU training.
