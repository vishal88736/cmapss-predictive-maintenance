# Validation-only repair and correction diagnostic

Run from the repository root using the existing `minor` environment:

```bash
source minor/bin/activate
python scripts/diagnose_pooled_paths.py \
  --source-run artifacts/pooled_v9_r2_20261029_114845 \
  --device cpu
```

The command creates a new timestamped directory under `artifacts/`. It evaluates
the four repair on/off × exported corrector on/off paths on exactly the saved
model-selection engines, with clean, stuck and mixed faults at severity 0.2.
Three seeded corruption replicates reuse the same engines; they are not neural
training seeds. Faults use the existing correlated, raw-unit protocol. All paths
share the exact prefix grid, fault exposure, scaler, detector thresholds and
fitting-support bounds. No production guard is relaxed and no deployable bundle
is written. CPU and saved CUDA arithmetic can differ slightly.

## Which saved weights are available

The completed revision-2 epoch checkpoint restores its accepted fallback before
saving. The rejected joint predictor was overwritten. Its separately retained
warm generator is available, but cannot represent the missing joint predictor.
The default four paths therefore use the **exported predictor**, with the warm
generator enabled only for the repair counterfactuals. The exported bypass with
correction is the reference path. `state_inventory.json` records the actual
checkpoint selection, decoder norms and limitations.

For a new, short corruption-trained predictor diagnostic, add an explicit step
budget. For example:

```bash
python scripts/diagnose_pooled_paths.py \
  --source-run artifacts/pooled_v9_r2_20261029_114845 \
  --pilot-steps 40 --batch-size 16 --device cpu
```

The pilot adapts only predictor parameters using the original fitting engines.
It freezes the warm detector/generator, support and corrector. It retains the
existing repair and bypass RUL objectives, clean anchors, branch and paired-prefix
supervision, without fitting the critic. It evaluates the final fixed-step state;
there is no validation-based pilot checkpoint selection. This is a new predictor
adaptation, not a reproduction of the discarded historical joint model or a full
joint-training experiment. The four pilot paths are evaluated in addition to the
saved paths. The diagnostic checkpoint is stored only in the new output folder.

## Interpretation and leakage boundaries

- Full original training files are read to verify hashes and reconstruct engine
  namespaces. Only fitting rows enter the pilot and only model-selection rows
  enter evaluation. Calibration/internal-test engines and official test files are
  excluded from fitting/evaluation. Preprocessing is never refitted.
- Repair toggles compare the same predictor with the same correction setting.
  Corrector toggles compare the same predictor/repair path. Comparing the pilot
  bypass without correction against the saved bypass without correction isolates
  the new predictor adaptation, without a correction confound.
- The existing corrector was fitted from OOF features for the exported upstream
  path. Applying it after changing repair/predictor is a **transfer intervention**,
  not a separately trained matched corrector ablation. No OOF fitting is done here.
- Clean-only identity probes zero the generator's final residual layer while
  retaining fault/quality routing. They help separate predictor sensitivity to
  quality features from displacement of healthy sensor values.
- Coverage columns apply the already exported capped-target radius descriptively.
  The original contract does not transfer to changed paths or corrupted inputs.
  The diagnostic never uses selection residuals to fit a new radius, never selects
  bins using true RUL, and never claims tighter calibrated intervals.
- Paired bootstrap intervals resample complete selection engines, retaining all
  their prefixes and corruption replicates. They estimate engine-balanced RMSE
  changes for this reused selection set, not independent test confirmation,
  multiple-training-seed uncertainty or multiplicity-adjusted significance.
- Clean guard columns use the original clean RMSE/NASA references and budgets.
  This limited study does not replace the full sensor validation matrix or
  authorize a model for export. No holdout is reused to tune a guard budget.

## Saved evidence

`manifest.json` records scopes, input hashes and source immutability.
`state_inventory.json` describes the available tensors. `metrics.csv` reports
pooled and FD-wise point errors, original-linear healthy-sensor displacement,
clean guard checks and descriptive coverage. `predictions.csv.gz` and
`engine_errors.csv` retain the underlying observations and prefix error summaries.
`paired_contrasts.csv` reports paired engine-bootstrap differences.
`validation_paths.png` plots measured errors. A requested pilot additionally
produces `pilot_history.csv` and a non-deployable `predictor_pilot.pt`.
