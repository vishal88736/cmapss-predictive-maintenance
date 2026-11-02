# Pooled audit, calibration and predictor repair

Completed source runs are preserved. Selection uses fitting/held-selection
engines; previously inspected internal and official benchmarks remain
exploratory. A validated change may still have worse test metrics.

## Implemented changes

- Deployment detector audits use the actual gate buffers and bind thresholds,
  detector state, held engine IDs, seed and the configured FPR budget. Earlier
  detector metrics remain historical. Source/channel/context rates are reported.
- Threshold tightening changes only violating channels, using the smallest
  representable feasible cutoff while preserving conservative floors. The new
  gate must still pass clean/source and corrupted-RUL guards.
- Predictor refinement uses deterministic paired-prefix passes, clean RUL,
  branch supervision, fitting-error tails and the active repaired corrupted-RUL
  path. Detector, generator and corrector weights remain frozen.
- Candidate selection protects original clean RMSE/NASA references, per-source
  mixed/stuck accuracy and an engine-balanced maximum-prefix error tail. No
  engine-specific output floor or monotonic inference rule is used.
- `observable_scaled_engine_max` fits positive observable-bin scales on
  selection engines, then calibrates **one maximum normalized residual per
  independent engine across all bins** on separate calibration engines.
  Unsupported finite-sample ranks stay unbounded. Old artifacts retain their
  own contracts. New records bind model/configuration, scale scope and radii.
- Every completed refinement epoch saves optimizer, scheduler, global RNG,
  sampled-window generator and best-selection state. Completed runs cannot be
  overwritten; interrupted matching runs can resume.

## Validation-only run

From the repository root, using the existing `minor` environment:

```bash
minor/bin/python scripts/finetune_predictor_continuity.py \
  --source-bundle artifacts/pooled_v9_r2_finetuned_final/grace_rul_ALL.pt \
  --output-dir artifacts/predictor_repair_validation \
  --epochs 16 --lr 0.00002 --corrupted-weight 2 \
  --tail-weight 1 --tail-fraction 0.1 \
  --maximum-training-windows 8192 --validation-only --device auto
```

Use a fresh directory. The window budget limits draws per epoch; all fitting
windows remain eligible under engine-balanced sampling. Omit it for full
fitting-window exposure. The budget is not a promise of better predictions.

The output `validation_candidate.pt` explicitly stores `deployable=False`.
`validation_history.csv` and `validation_selection.json` record acceptance. If
no epoch qualifies, inspect `validation_rejected.json`. No calibration or holdout
evaluation runs in validation-only mode.

For an interrupted run, repeat the same command with `--resume` and the exact
output directory, recipe, source and implementation. Its checkpoint is
`checkpoints/predictor_refinement.pt`. Changed recipes require fresh runs.

## Freeze and evaluate

```bash
minor/bin/python scripts/refresh_pooled_bundle.py \
  --source-run artifacts/pooled_v9_r2_finetuned_final \
  --candidate artifacts/predictor_repair_validation/validation_candidate.pt \
  --output-dir artifacts/predictor_repair_frozen --device auto
```

Source hash and partitions must match. The script repeats active-path, detector
and complete recovery validation before calibration. Strict reload precedes
official test reads. No test result selects weights or gates. A verified
identical forward path can reuse point predictions with targets rechecked and
all intervals/metrics recomputed; changed gates or weights require inference.

## Generate the paper

```bash
minor/bin/python scripts/export_paper_metadata.py \
  --run artifacts/predictor_repair_frozen
minor/bin/python scripts/generate_pooled_paper_assets.py \
  --run artifacts/predictor_repair_frozen --paper-root ..
```

One completed run supplies tensor-free numerical evidence. The generator
derives prose, partitions, capped/raw tables, original/immediate-source
comparisons, fault/calibration plots and the engine-179 diagnostic. It records
input/output hashes and `paper_assets/current_run.json`; it does no fitting or
inference. Architecture/workflow diagrams remain native editable LaTeX.

Report regressions and unresolved cases. A plausible explanation or high fleet
coverage does not repair an individual prediction. Do not tune against the
previously inspected engine, use true RUL to select inference bins, substitute
desired scores, or treat corruption realizations as independent training seeds.
Keep datasets, checkpoints and generated evidence out of source-only Git pushes.
