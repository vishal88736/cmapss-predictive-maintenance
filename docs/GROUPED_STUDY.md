# Population-level recipe selection

Use this workflow for the next model experiment. The earlier predictor-refinement
scripts remain historical tools; they are not the recommended recipe-selection
path. No engine-specific prediction repair or case-specific acceptance condition
is used here.

## Fixed boundaries and training

The original `pooled_plan.json` supplies immutable historical boundaries. With
the retained full split, development contains **603 engines** (567 fitting + 36
selection); all 35 calibration and 71 internal-test engines stay excluded from
the cross-validation study. Official test/RUL files and previous model/result
artifacts are blocked during study execution. The four training files contain
all original training trajectories, so separation is additionally enforced on
the DataFrames passed to each fit.

The default plan compares two global recipes: the shared reference without
paired-prefix regularization, and the same model with paired-prefix
regularization. These are declared hypotheses, not established improvements.
Three source-stratified outer engine folds and three fixed training seeds give
18 recipe/fold/seed cells. Within each outer training group, separate inner
engines select epochs, detector gates, repair paths and correction bounds.

Every cell fits its scaler, regime discovery, fault scales, support envelope,
detector, generator, predictor and OOF corrector independently. Temporary OOF
models also exclude the outer assessment engines. All compared recipes share
the same engine folds, assessment prefixes and corruption realizations. Operating
settings and measurements enter inference; source/engine labels are bookkeeping.
New detector threshold selection uses the configured inner-validation corruption
seed, matching its final audit across training seeds. Detector fitting retains
its own training randomness. Older artifacts keep their original seed contract.

## Selection rule

For each source/context, average squared prefix errors within each engine, then
average equally across engines and take the square root. Average equally across
sources and declared fault contexts. The default score is half clean macro RMSE
and half fault macro RMSE, averaged over **all** declared training seeds.

Every source and training seed must also satisfy the declared clean RMSE/NASA
budgets relative to the reference. Failed guards disqualify the comparator. No
fold, seed or fault context can be omitted. Exact score ties use the declared
candidate order. Seed standard deviations are descriptive; folds share training
data, and neither fold nor corruption variation is an independent-fleet confidence
interval. The selected CV score is model-selection evidence, not an unbiased
final performance estimate.

The plan locks configuration, candidates, seeds, folds, assessment matrix, data
hashes and implementation before fitting. Changes require a fresh study. These
are local integrity checks, not independent public preregistration or proof
against deliberate evidence rewriting.

## Commands

Run from the repository root in the existing `minor` environment. Use a new
directory name for each declared study:

```bash
minor/bin/python scripts/run_grouped_study.py plan \
  --boundary-plan artifacts/pooled_v9_r2_20261029_114845/pooled_plan.json \
  --study-dir artifacts/grouped_study_population --device auto

minor/bin/python -u scripts/run_grouped_study.py run \
  --study-dir artifacts/grouped_study_population

minor/bin/python scripts/run_grouped_study.py freeze \
  --study-dir artifacts/grouped_study_population

minor/bin/python -u scripts/run_grouped_study.py final \
  --study-dir artifacts/grouped_study_population \
  --final-output artifacts/grouped_study_population_final
```

The default full study trains 18 main pipelines and 54 temporary OOF pipelines.
The final refit trains one further main pipeline and its three OOF models. It
can take substantial time on CPU. A plan is cheap and does no fitting. Device
and any base training changes must be declared at `plan`; `run` cannot change
the recipe. `configs/grouped_study.json` defines the fixed scientific matrix.

If interrupted, repeat `run` with the same study directory and unchanged code.
Completed cells are hash-checked and reused; incomplete cells use the existing
epoch checkpoints. One file lock prevents concurrent writers. Preserve the
study directory until it finishes. A completed/frozen final model cannot be
overwritten. Do not retry with modified parameters in that directory.

For a small CPU execution check, create a separate plan using `--quick --device
cpu` and then run it. It retains two seeds, two folds, both recipes and all
training components, using 12 development engines per source and a small
training budget. Such a run **cannot** freeze a research recipe or invoke final
evaluation. Its errors establish execution behavior only.

## Freeze, calibration and evaluation

`freeze` verifies every completed cell and derives `frozen_recipe.json` from the
saved per-engine statistics. The final seed is fixed in the original plan; it is
not the best-performing CV seed. Neither CV weights nor earlier bundles warm-start
the final fit.

`final` refits the selected recipe from scratch on the original fitting/selection
groups. Independent calibration follows model selection, then strict serialized
loading, then evaluation. Observable interval scales use selection engines;
conformal scores use the separate calibration engines. The existing pooled
exporter freezes the bundle before reading official test/RUL files. Final results
cannot trigger another candidate selection or overwrite the study's frozen recipe.

Outputs include cell checkpoints/scopes, inner detector/recovery/OOF audits,
per-engine errors, source/context metrics, all-seed comparisons, read-access
audit and the frozen recipe. Source publication excludes these artifacts.

## Evidence limits

Previously inspected engines and official benchmarks remain **reused exploratory
data**. New splits, new checkpoints and stricter boundaries do not erase earlier
development exposure. Independent confirmatory performance requires genuinely
unseen data or a separately controlled evaluation. Keep previously inspected
individual failures as descriptive diagnostics, outside recipe selection. No
paper result or deployment claim should be replaced by execution-check metrics.
