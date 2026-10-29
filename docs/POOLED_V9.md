# Shared GRACE-RUL pipeline: FD001–FD004, 80/10/10

Format 9 trains one detector, generator, CNN–GRU/Transformer predictor and residual
corrector across all four C-MAPSS training sources. One shared scaler assigns
operating regimes from measured settings. Inference accepts compatible sensor
histories without an FD label. Optional source labels group engine histories and
reports; they never select model weights or normalization statistics.

The neural dimensions remain 70 global cycles, 30 local cycles, 21 sensors and
three operating settings. Recovery uses the existing separate detector, validated
finite-repair thresholds, fitting-support bounds and full recovery matrix.
The generator has structured fault supervision; healthy examples remain present.
The new loss recipe is experimental and does not guarantee better RUL accuracy.

## Training recipe revision 2

The first full pooled run selected repair bypass. Its audited failures include
mild-noise reconstruction regressions and a large FD004 prefix underprediction.
Revision 2 changes fitting objectives and generator initialization; the neural
architecture, 24 measured features and inference equations remain compatible.

- **Finite-fault regret:** for each altered finite window/channel, compare the
  proposed repair MSE against the unchanged-input MSE in the original, uncompressed
  normalized sensor space. For ratio `r`, train with `log(1+r) + max(log(r), 0)`.
  Only a machine-precision floor handles a zero baseline. This gives mild faults
  meaningful weight alongside large faults and directly penalizes harmful moves.
  The loss uses the gated proposal before projection so a saturated bound cannot
  hide its gradient. Actual deployed repairs still face the original full matrix.
- **Paired clean prefixes:** each fitting window receives an earlier prefix of
  the same engine, with an observed lag from 1 to 30 rows. Supervise the earlier
  prediction and the change between the two predictions using their actual capped
  labels. A healthy plateau and a declining late-life target are both supported.
  Dropout is disabled for these paired passes, while predictor gradients remain
  active. Pairs never supply future measurements, source IDs or RUL to inference.
  Calibration does not construct training pairs.
- **Sensor-based generator warm start:** generator-stage early stopping and the
  initialization carried into joint fitting use the best validation sensor score,
  rather than the last rejected generator epoch. The score combines finite-fault
  regret, missing imputation and healthy displacement over mixed faults and the
  lowest declared training noise severity. Pooled selection minimizes the worst
  source's average of these contexts. Warm state and score survive interruption.
  This does not authorize exporting a failed sensor-recovery candidate.

All three additions use fitting labels or held-validation measurements. The 80/10/10
engine identities, calibration exclusion, correlated corruption and repair/RUL/NASA
guard budgets remain the same. Paired predictor passes add training compute; they
do not add inference parameters or a prediction-smoothing rule. Reports include
paired-prefix loss, warm sensor score/selection and validation RUL-step errors.
The next full run is needed to measure whether these changes improve recovery.

## Data split and leakage boundaries

Only `train_FD001.txt` through `train_FD004.txt` enter the new split. Original
official test/RUL files remain a separate benchmark and never enter fitting.
Source/unit pairs receive unique numeric engine IDs before combining data. IDs
and source names are bookkeeping, not neural inputs. Complete engines are
stratified by source using a fixed seed; windows are created after partitioning.

The actual full-data plan contains 709 engines:

| Role | Engines | Purpose |
| --- | ---: | --- |
| Training: approximately 80% | 567 | Fit preprocessing, detector, generator and RUL predictor; OOF correction training |
| Validation pool: approximately 10% | 71 | Contains the two disjoint groups below |
| Validation: model selection | 36 | Checkpoints, detector thresholds, repair/bypass decisions, correction bound and seed selection |
| Validation: interval calibration | 35 | Calibrate the already selected frozen model; no weight/seed selection |
| Internal test: approximately 10% | 71 | Final clean/fault evaluation after selection and calibration |

The 36 and 35 engines are subdivisions of the 71-engine validation pool, not extra
data. Integer-engine rounding is recorded per source. All 567 training engines
remain in the fitting pool. Calibration does not reuse model-selection engines.
Each temporary OOF model has its own fitting/inner-validation split and never
fits or selects on the engines whose correction features it produces.

The shared preprocessor fits setting statistics on training engines only. It
selects 1–6 groups using Gaussian-mixture BIC on a deterministic fitting-settings
sample, then fits KMeans centers on the fitting settings. Sensor normalization,
resolution floors, raw-unit fault scales and recovery support use the same
fitting scope. Each OOF model refits these components within its own fitting scope.
The [scikit-learn GaussianMixture documentation](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)
describes the BIC API. The selected groups are statistical operating contexts,
not inferred FD/failure-mode labels.

## Training, validation, calibration and testing

1. Save `pooled_plan.json` and `engine_partitions.csv` before model fitting. The
   plan binds data hashes, configuration, engine namespaces and all partitions.
2. Train one shared clean predictor with paired-prefix supervision, then independently fit the finite-fault
   detector, train recovery with the conditional WGAN-GP critic, and jointly
   optimize recovery/prediction. Corruption uses temporally correlated, related
   sensor-domain mechanisms with fitting-only physical amplitude scales.
3. Select clean pretraining by clean validation RMSE. Later stages use the
   configured clean/eight-family validation objective and original clean
   RMSE/NASA guards, both pooled and separately for every source. Every improving
   recovery candidate must pass every declared sensor condition: 41 pooled cases
   plus the same 41 cases within each source, including all five severities. This
   prevents a good pooled average from hiding an individual source's failure. Source
   identity is validation bookkeeping; all these checks use the same model. Rejected
   recovery can leave a validated bypass; the saved decision remains explicit.
4. Fit the bounded residual corrector from engine-disjoint OOF predictions,
   selecting bounds on the selection engines and retaining zero correction.
5. Choose the seed using validation only. Afterwards, calibrate separate capped
   and raw-target engine-maximum-prefix conformal intervals on the 35 independent
   calibration engines. Small execution checks can honestly produce unbounded
   intervals; nominal coverage requires the documented exchangeability conditions.
6. Serialize and strictly reload the one-model bundle before evaluating the
   internal test engines. If enabled, read original official test/RUL files only
   after this freeze. Evaluation never updates weights, preprocessing or thresholds.

Internal testing uses a fixed inspection schedule at observed cycles 30, 60, 90,
and so on. Existing prefixes do not move when future observations are appended.
Ground-truth RUL is calculated offline for scoring; it is not a model feature.
This avoids evaluating only complete-trajectory final rows, where every RUL would
be zero. The schedule also applies to model-selection prefixes. Official benchmark
evaluation retains its original final observed endpoints.

Reports contain pooled and per-source RMSE/MAE/NASA values, capped/raw target
conventions, active-fault exposure, healthy displacement and per-case predictions.
Both prefix-weighted RMSE and equal-engine RMSE are recorded. FD names are used
only to break down measurements; all rows use identical model/scaler parameters.
The simulator does not provide measured field-fault validation. Reusing these
previously studied simulated datasets remains exploratory, including a new split.

## Commands in the existing minor environment

```bash
cd "/home/vishal/D DRIVE/minor/cmapss-predictive-maintenance"
source minor/bin/activate
set -o pipefail
```

Inspect the full-data split without fitting or opening official tests:

```bash
python scripts/train_pooled.py --plan-only \
  --output-dir "artifacts/pooled_plan_$(date -u +%Y%m%d_%H%M%S)"
```

Run a small CPU execution check across all four sources:

```bash
python -u scripts/train_pooled.py --quick --device cpu --skip-official \
  --output-dir "artifacts/pooled_quick_$(date -u +%Y%m%d_%H%M%S)"
```

Quick mode caps the number of engines, optimizer batches, epochs and model widths.
Its metrics are execution evidence only. It still exercises namespaces, all source
files, OOF exclusion, calibration, strict loading and the internal test workflow.
Quick bundles are excluded from automatic website discovery.

Start the complete full-dimension run in a terminal with a visible CUDA device:

```bash
python scripts/check_cuda.py
pooled_run_dir="artifacts/pooled_v9_r2_$(date -u +%Y%m%d_%H%M%S)"
python -u scripts/train_pooled.py --device cuda \
  --output-dir "$pooled_run_dir" \
  2>&1 | tee "${pooled_run_dir}.log"
```

The full command includes the internal holdout fault sweep and the separate
original official benchmark. It trains one seed (42), one primary pipeline and
three temporary OOF pipelines. To add training seeds, specify `--seeds 42 123 2026`.
To omit only the extra official benchmark, add `--skip-official`. That reporting
switch does not change the training identity. Changing data, seeds, dimensions or
loss settings requires a fresh directory. Resume an interrupted run by using its
actual printed directory and unchanged training settings.
Revision 2 is loaded automatically from `configs/pooled_v9.json`. Do not resume
the completed revision-1 training directory with this changed recipe. Its saved
format-9 model remains loadable and its measurements remain historical evidence.

The notebook equivalent is `GRACE_RUL_pooled.ipynb`. Its default is full training
on CUDA. `QUICK_RUN = True` and `DEVICE = 'cpu'` explicitly select an
execution check. SHAP/LIME studies remain optional and use fitting-only backgrounds.

## Outputs and deployment

- `grace_rul_ALL.pt`: one format-9 model under the key `POOLED`, with shared
  preprocessing, independent calibration, detector/recovery audits and provenance.
- `pooled_plan.json`, `engine_partitions.csv`: exact source/engine identities and
  optimizer/selection/calibration/test exclusions.
- `checkpoints/`: resumable primary and OOF epochs; no held-test fitting.
- `training_history.csv`, `correction_history.csv`, `seed_summary.csv`,
  `cross_validation_metrics.csv`: actual learning/selection records. Generator
  history distinguishes `warm_sensor_score` / `warm_recovery_selected` from
  deployment checkpoint acceptance. RUL-step columns measure residual changes
  between observed validation prefixes; they do not modify model predictions.
- `detector_history_POOLED_<seed>.csv`, `detector_audit_POOLED_<seed>.json`,
  `recovery_validation_POOLED_<seed>.json`: measured detector fitting and selected/
  rejected sensor-recovery audits, including source-level checks when reached.
- `test_metrics.csv`, `robustness_summary.csv`, `recovery_diagnostics.csv`: explicit
  `internal_test_10pct` versus `official_benchmark` scopes, with per-source reports.
- `predictions/*.csv.gz`: every evaluated case, targets and source metadata for
  numerical recomputation. `run_manifest.json` becomes `complete` only after
  evaluation finishes and the frozen bundle hash is rechecked.

The website supports both the retained format-8 model and new format-9 bundles.
A completed full pooled bundle can be discovered automatically. When connected,
the FD selector disappears and uploads need only engine ID, cycle, the three
settings and 21 named sensors, with blank sensor cells for missing readings.
Optional source labels distinguish repeated engine IDs but do not affect routing.
Unknown fleets still need compatible sensor meanings/units and separate evaluation.

The existing verified FD002/FD004 bundle remains preserved. No pooled accuracy
claim replaces its measured results until a full pooled run is completed and audited.
Generated weights, logs, predictions and figures stay under ignored `artifacts/`.
