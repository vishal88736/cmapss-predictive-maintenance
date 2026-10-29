# Run the remaining GRACE-RUL studies

The current implementation direction is the [shared format-9 pipeline](POOLED_V9.md):
combine all four training sources, use one model without FD routing, and split
whole engines 80/10/10. Use that document and `GRACE_RUL_pooled.ipynb` for the next
training run. The stages below retain the earlier format-8 experiment history.

Run these stages in order. Each stage has a distinct purpose. Evaluation and
explanations do not retrain weights. Fresh format-8 training is needed to measure
the revised recovery/selection recipe. A passing guard does not force better RUL accuracy.
The frozen evaluation and first window-guard training run have already been
completed locally. The frozen model and its evaluation remain available; the first
training run's results/review are archived under `artifacts/retired_evidence/` after
its weaker weights and checkpoints were removed. Continue with the revised step 3;
repeating evaluation does not repair the model. See
[the FD002/FD004 investigation](RECOVERY_BALANCE.md).

Before the expensive step 3, use the
[short FD002 preflight](RECOVERY_BALANCE.md#short-development-pilot-and-its-limits).
It audits all 41 sensor conditions after 40 generator fitting steps per objective,
without training OOF models, recalibrating or reading official tests. The completed
preflight passed the guards but did not improve RUL over the retained starting
model, so investigate that development-validation regression before launching the
full study. A guard pass alone is not a reason to claim improved accuracy.

## 1. Activate the existing minor environment

```bash
cd "/home/vishal/D DRIVE/minor/cmapss-predictive-maintenance"
source minor/bin/activate
set -o pipefail
```

No installation is required for the existing environment. Check GPU visibility
in the same terminal before CUDA evaluation or training:

```bash
python scripts/check_cuda.py
```

The current assistant session cannot see a CUDA device, despite a CUDA PyTorch
build being installed. A terminal with GPU access can give a different result.
If that check fails, inspect `nvidia-smi` in that terminal. CPU evaluation remains
available with `--device cpu`. Full neural training on CPU can take substantial
time; the CUDA command stops when a GPU is unavailable.

## 2. Evaluate the existing frozen format-8 model

This preserves the source bundle, scaler, support, detector thresholds,
corrector, and calibration. It performs no search, training, or recalibration.
The evaluation plan is saved before official data is read. Source-file hashes
must agree with the saved artifact; prediction metrics are calculated anew.

```bash
v8_eval_dir="artifacts/v8_frozen_eval_$(date -u +%Y%m%d_%H%M%S)"
python -u scripts/evaluate_research.py \
  artifacts/recovery_v8_safety_20261027/grace_rul_ALL.pt \
  --device auto \
  --output-dir "$v8_eval_dir" \
  2>&1 | tee "${v8_eval_dir}.log"
```

`auto` uses visible CUDA or falls back to CPU and records the actual device. Use
`--device cuda` to require GPU execution, or `--device cpu` explicitly. For a
short clean-endpoint evaluation, add `--clean-only`; that omits the fault sweep
and is not the complete robustness study.

Do not create the evaluation directory yourself. The evaluator requires a new
directory outside the source run and creates it. The adjacent `.log` file is
separate. If interrupted, completed cases remain inspectable, but the manifest
does not say `complete`; rerun in a new evaluation directory.

Default full scope for this two-subset bundle:

- One clean case and 120 fault cases per subset: eight families, five declared
  severities, and three corruption replicates. Each uses every official final
  observed endpoint; this is not an all-prefix test study.
- `test_metrics.csv`: capped/raw clean RMSE, MAE, NASA score and interval coverage.
- `robustness_summary.csv`: 480 capped/raw fault rows for FD002/FD004, with
  active-fault/unaffected endpoint metrics and measured exposure fractions.
- `recovery_diagnostics.csv`: original-linear aggregate, channel and window-tail
  healthy displacement diagnostics. Optional tail collection does not change
  legacy training/evaluation defaults.
- `predictions/*.csv.gz`: every case's per-engine predictions, actual targets,
  intervals and healthy-window displacement. These support recomputation.
- `evaluation_plan.json` and `run_manifest.json`: source/data hashes, selected
  seeds, fixed cases, device, completion and unchanged weights/buffers checks.

The bundle contains one validation-selected seed per subset. Repeating corruption
does not create independent training seeds. Existing benchmark tests were
inspected during development; this evaluation is exploratory. Keep model and
threshold selection on fitting/validation engines. Do not use a better-looking
test row to choose settings, seeds, guards or explanations.

## 3. Train the new objective on FD002/FD004 first

The first `v8_window_guard.json` run retained FD002 bypass after missing-input
reconstruction rejection and regressed on FD004 clean test RMSE. Its result records
remain under `artifacts/retired_evidence/v8_training_20261028_183317/`; its weights
and checkpoints were retired at the user's request. The next full-dimension
experiment uses
`configs/v8_recovery_balance.json` at one training seed on the two problem subsets.
It adds separate physical-scale imputation supervision, stable log-tail training,
balanced fitting fault contexts, a clean pretraining anchor and equal-family
validation selection. The short development pilot supports FD002 imputation, but
does not establish a full FD004 RUL improvement.

The recipe retains **validation** maximum channel and maximum window healthy-RMSE
limits of 0.15, in original-linear regime z-score units, along with the existing
aggregate, reconstruction and clean RUL/NASA guards. It can reject recovery and
retain a bypass. These are declared policies, not desired accuracies, and they
provide no universal unseen-window guarantee. Observed-inlier protection remains
conditional on fitting support. Selection now evaluates nine fixed validation
contexts per call instead of two, so validation takes longer.

```bash
v8_train_dir="artifacts/v8_recovery_balance_$(date -u +%Y%m%d_%H%M%S)"
python -u scripts/train_research.py \
  --device cuda --subsets FD002 FD004 --seeds 42 \
  --config configs/v8_recovery_balance.json \
  --skip-tuning --skip-benchmarks --extended-evaluation \
  --output-dir "$v8_train_dir" \
  2>&1 | tee "${v8_train_dir}.log"
```

The command includes full staged training and three engine-disjoint OOF folds.
It evaluates capped/raw clean endpoints and the complete structured-fault sweep.
It deliberately skips the additional paper-comparator fits at this stage. Use
the saved validation records to inspect retained repair/bypass paths, clean
RMSE/NASA protection, matrix completeness, channel/window tails, and original
corrector references. Report any RUL regression alongside gains.

If interrupted, rerun the exact command with the **same actual output directory
and training settings** to resume completed upstream epochs. Do not substitute a
fresh timestamp when trying to resume. Correction/evaluation may rerun. Changing
guards, model dimensions, subsets or seed lists requires a fresh training
directory; never resume format-7 checkpoints as format 8.

The supplied recipe is an explicit development experiment. Omitting `--config`
retains the earlier training policy, without these changes. Changing the recipe
requires a fresh directory. Do not select between policies using test errors.

## 4. Run model explanations after a bundle exists

Use the fresh run's bundle, or substitute the frozen-upgrade bundle to explain
that exact artifact. These calculations use saved fitting backgrounds and do
not train the predictor.

```bash
python -u scripts/explain_research.py \
  "$v8_train_dir/grace_rul_ALL.pt" \
  --device cuda --subsets FD002 FD004 \
  --examples 12 --repeats 3 --samples 256 --family clean \
  --output-dir "$v8_train_dir/explanations_clean" \
  2>&1 | tee "${v8_train_dir}_explanations_clean.log"

python -u scripts/explain_research.py \
  "$v8_train_dir/grace_rul_ALL.pt" \
  --device cuda --subsets FD002 FD004 \
  --examples 12 --repeats 3 --samples 256 --family mixed \
  --output-dir "$v8_train_dir/explanations_mixed" \
  2>&1 | tee "${v8_train_dir}_explanations_mixed.log"
```

Inspect the generated JSON/CSV for SHAP/LIME importance, repeated perturbation
stability and LIME held-out fidelity. These repeats vary explanation randomness;
they are not independently trained neural seeds. Fix the engine sampling policy
before inspecting outcomes. Attribution explains model behavior; it does not
establish a physical sensor diagnosis or improved RUL accuracy. Larger explanation
studies cost more computation; a smaller declared sample is an exploratory check.

## 5. Expand to all subsets, multiple training seeds and matched comparisons

Freeze the training policy using development/validation evidence. A new run over
all four subsets and five training seeds provides the broader replication:

```bash
v8_study_dir="artifacts/v8_all_subsets_$(date -u +%Y%m%d_%H%M%S)"
python -u scripts/train_research.py \
  --device cuda --subsets FD001 FD002 FD003 FD004 \
  --seeds 42 123 2026 314 777 \
  --config configs/v8_recovery_balance.json \
  --skip-tuning --skip-benchmarks --extended-evaluation \
  --output-dir "$v8_study_dir" \
  2>&1 | tee "${v8_study_dir}.log"
```

This is expensive: each subset/seed includes temporary upstream OOF fits. The
earlier two-subset, one-seed directory has a different training identity and
cannot simply be resumed with this wider configuration.

Add the matched comparators and controlled ablations later, using the **same**
all-subset directory and unchanged training settings. Reporting/benchmark switches
can change without invalidating completed primary upstream checkpoints:

```bash
python -u scripts/train_research.py \
  --device cuda --subsets FD001 FD002 FD003 FD004 \
  --seeds 42 123 2026 314 777 \
  --config configs/v8_recovery_balance.json \
  --skip-tuning --paper-study \
  --output-dir "$v8_study_dir" \
  2>&1 | tee "${v8_study_dir}_comparisons.log"
```

This still reruns correction/calibration/evaluation and trains comparator models;
it is not a cheap plotting switch. Paper adaptations receive independent
validation searches, and controlled ablations inherit primary settings. They are
harmonized adaptations, including supervised correction instead of TD3, not exact
reproductions of every paper. Examine the generated benchmark, paired-comparison,
active-fault and acceptance tables whether they show gains or failures. A
configured improvement target must never replace measured metrics.

All-subset explanation studies can subsequently use `scripts/explain_research.py`
with the all-subset bundle and `--subsets FD001 FD002 FD003 FD004`. Default full
FD001/FD003 calibration has eight engines; honest nominal-90% engine-max-prefix
intervals are unbounded. Preserve that result instead of inventing finite bounds.

## 6. Independent confirmation and paper refresh

A fresh, genuinely uninspected holdout or external dataset is still required for
an independent confirmation. The existing inspected C-MAPSS tests cannot become
untouched again by changing a seed or randomly resplitting them. No additional
independent dataset is configured here. That stage requires a declared dataset,
compatible measured-input/target adapter, predeclared endpoint/fault protocol,
fitting-only preprocessing, separate calibration, and a model fixed before its
test labels are inspected. No command here fabricates this missing evidence.

After the runs, retain the actual run paths, prediction files, manifests and
validation reports. Verify completed manifests and unchanged source hashes before
updating the manuscript. The manuscript generator currently points to its named
historical/safety snapshots; rerunning it alone will not import a new training
directory. Update those explicit evidence bindings and provenance when reviewing
the new results. Never copy historical test numbers onto the new artifact.

Generated data, weights and logs stay under ignored `artifacts/`. Commit/push only
source, configuration, tests and documentation.
