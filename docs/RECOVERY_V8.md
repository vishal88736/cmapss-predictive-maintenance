# Format 8: fitting-support recovery safety

Format 8 addresses a measured failure in format 7: a detector can assign a high
fault score to an intact channel when another channel is damaged. Its generator
can then propose a large change. A small average healthy error can conceal a
large error in one sensor or window, and an unbounded change in asinh coordinates
can become a much larger change after inverse transformation.

This is a recovery safeguard and validation change. It does not establish a new
paper benchmark, improved official test accuracy, or safety on a real fleet.

## What the validation diagnosis found

The frozen format-7 seed-42 models were examined on their reserved **training
validation engines**, using clean inputs and every structured fault/mixed family
at severities 0.1–0.5. No official test inputs or labels entered this diagnosis.
The local evidence is under
`artifacts/repair_v8_audit_20261027/validation_diagnosis/`.

- FD002: under a pressure-domain stuck fault in sensors 5, 6, and 11, the intact
  P30 sensor 7 moved from 138.29 to approximately 105.35 in its source unit. The
  original-linear standardized displacement was −74.50. Its observed value was
  within the fitting range [136.80, 140.35].
- FD004: under a fan-speed stuck fault in sensors 8 and 13, intact coolant-flow
  sensor 21 moved from 8.6328 to approximately 11.1126, a +32.09 standardized
  displacement. Its fitting range was [8.3202, 8.8879].
- Merely clipping proposals to the fitting range is insufficient. A separate
  FD004 intact core-speed example moved by approximately −73.51 source units
  while both its original and proposed values remained within the fitting range.

Paired clean/fault evaluations used the same validation engine, observed window,
and unchanged target sensor. Changing other channels changed detector scores and
generator proposals substantially. These observations motivate an explicit bound
on movement of an observed inlier, rather than relying entirely on detector scores.
The sigmoid outputs remain discrimination scores, not calibrated probabilities
of a physical sensor failure.

## Fitting-only support and finite repair bounds

`fit_recovery_envelope` computes a minimum and maximum for each sensor in each
operating regime using exactly the scaler's fitting engines. It records fitting
IDs, regime centers, representation, and bounds. Validation and calibration
engines do not fit these statistics. Source observations define an empirical
support, not a manufacturer's sensor tolerance or a physical failure boundary.

The neural generator and validated detector gate first propose a repair. The
safety layer bounds the resulting finite update in the **original linear regime
z-score representation**, before asinh compression. Let `x` be the observed
linear value, `[L,U]` the fitting support, and `b` the existing
`max_intact_repair_rmse` budget, normally 0.15:

| Observed reading | Allowed finite repair interval |
| --- | --- |
| Within `[L,U]` | `[x-b, x+b]` |
| Above `U` | `[L-b, x+b]` |
| Below `L` | `[x-b, U+b]` |

Thus an observed inlier has at most 0.15 original-linear movement per finite
entry under the default budget. An outlier may move back through the fitting
support in one step; it is not limited to a tiny step toward a very distant
valid range. Movement farther outward remains budgeted. The bound is transformed
back to encoded coordinates and rounded inward to representable values.

This is not a raw-unit limit of 0.15. A raw displacement also depends on that
sensor's regime standard deviation. Pooled raw error across different sensors
mixes different units; per-sensor physical errors remain necessary diagnostics.

Missing observations retain the generator's imputation path. Padding stays
excluded. An unmodified future reading can itself lie outside the historical
fitting support, so the inlier bound is a conditional safeguard, not a universal
healthy-sensor or fleet guarantee.

## Training and complete validation

Fresh format-8 training adds a tail identity penalty on the proposed repair
**before projection**. It penalizes the worst 10% of window/sensor healthy-error
groups instead of allowing many unchanged entries to hide a few large errors.
Applying that loss before projection gives the generator a learning signal when
a proposal saturates at a bound. The projection itself uses ordinary clipping;
it does not use a straight-through gradient estimator.

Full validation evaluates **41 distinct contexts**: clean once, plus seven fault
families and mixed faults at five severities. Each context must pass the existing
healthy original-linear error and recovery-versus-input requirements, and the
audit must finish every declared window. A partial shortlist cannot certify a
candidate. Channel/window tail errors are always reported; optional additional
tail limits are explicit configuration choices.

The original clean RMSE and NASA-score guards remain separate requirements. They
are checked on the final pipeline with its selected corrector. Recovery-matrix
audits disable correction to measure sensor repair itself. The primary mixed
validation reconstruction guard also remains in force. Quick execution runs use
their declared smaller matrix and must not be described as a completed full study.

## Fast upgrade of existing format-7 weights

A full retraining run is optional if the frozen-weight upgrade passes its guards.
The upgrader loads a format-7 bundle on CPU, fits the support from its original
fitting engines, and adds only the four support buffers. Every existing neural
tensor, detector threshold, selected correction bound, and corrector tensor must
remain unchanged. It performs no optimizer step and no hyperparameter search.

It then evaluates the complete 41-context training-validation matrix and the
clean/mixed validation guards. Original clean references come from the source
pipeline checkpoint, or the selected predictor row in its training history.
If any guard fails, the command writes a rejection report and exports no model;
it does not silently disable recovery or replace the selected corrector.

After acceptance, the exact frozen model with the new recovery rule is
recalibrated on the original independent calibration engines. Its capped and raw
intervals are separate contracts. Old interval offsets cannot be reused after
changing the forward rule. The output includes a new format-8 bundle, seed entries,
calibration CSVs, validation report, fitting provenance, and explicit
`weight_training_config` / `run_type="post_training_safety_upgrade"` metadata.

The original format-7 run remains unchanged. Its test/CV metrics are not copied
into the new run as results for the modified model. Post-training upgrades do not
claim that their frozen weights received the new format-8 tail-loss training.

Use the existing `minor` environment from the repository root:

```bash
cd "/home/vishal/D DRIVE/minor/cmapss-predictive-maintenance"
minor/bin/python scripts/upgrade_recovery_safety.py \
  --source-model artifacts/recovery_v7_20261026_191710/grace_rul_ALL.pt \
  --output-dir "artifacts/safety_upgrade_v8_$(date +%Y%m%d_%H%M%S)"
```

The directory must be new and outside the source run. No packages or weights are
downloaded by this command. If the `minor` environment is already activated,
`python` can replace `minor/bin/python`.

## Optional fresh CUDA training

Fresh training also trains with the new tail objective. Run it only in a new
directory; format-7 checkpoints describe a different recovery rule.

```bash
cd "/home/vishal/D DRIVE/minor/cmapss-predictive-maintenance"
minor/bin/python scripts/check_cuda.py --training-check
minor/bin/python scripts/train_research.py --device cuda --subsets FD002 FD004 \
  --skip-tuning --skip-benchmarks \
  --output-dir "artifacts/recovery_v8_$(date +%Y%m%d_%H%M%S)"
```

Paper comparator training and explanations remain separate experiments. Changing
them does not improve the primary frozen weights. The default target remains
`min(raw_RUL, 125)`; the safety change does not repair long-horizon point accuracy
by changing the target definition.

## Evidence status

The frozen-weight CPU migration completed successfully on 27 October 2026:
`artifacts/recovery_v8_safety_20261027/grace_rul_ALL.pt`. Both subsets passed all
41 declared recovery contexts, their original clean RMSE/NASA guards, and the
primary mixed-context reconstruction guards. The complete serialized artifact
also passed a strict loader round trip. These are measured training-validation
results, not official test results:

| Validation measurement | FD002 | FD004 |
| --- | ---: | ---: |
| Clean capped-target RMSE, retained corrector | 11.622525 | 10.820696 |
| Clean NASA score mean | 2.308244 | 1.944239 |
| Worst context healthy original-linear RMSE | 0.023773 | 0.021229 |
| Worst individual window healthy original-linear RMSE | 0.061684 | 0.174113 |
| Primary mixed healthy original-linear RMSE | 0.013140 | 0.012868 |
| Recalibrated capped-target radius | 49.902321 | 50.267166 |
| Recalibrated raw-target radius | 147.934731 | 178.490784 |

The FD004 worst window can exceed 0.15 because the pointwise bound is conditional
on an observed value being inside the fitting support. Historical support does
not contain every unmodified future observation. Individual-window limits remain
an optional, separately declared guard; they were not asserted by this run.

No weights were retrained and every original state tensor remained unchanged.
Calibration used the original independent calibration engines. The source bundle
was preserved. The migration's file-access audit observed only the two training
trajectory files and no attempted access to official test inputs/labels or old
test/robustness/prediction tables. The report is
`artifacts/recovery_v8_safety_20261027/safety_upgrade_report.json`.

No official test metrics were evaluated for this new artifact. Later test metrics
would remain exploratory because benchmark subsets were inspected during earlier
development; the safety upgrade cannot undo that previous reuse. Test values must
not tune the support, safety budget, detector thresholds, or checkpoint selection.

Only source code, tests, and documentation belong in the GitHub changes. Generated
diagnostics, model bundles, datasets, checkpoints, and run outputs remain local
artifacts.
