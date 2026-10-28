# FD002/FD004 recovery and selection repair

The October 28 fresh format-8 run did not establish better RUL performance.
FD002 retained a bypass because missing-input reconstruction failed validation;
FD004 passed the recovery matrix but its clean test RMSE regressed. The local
review is now archived at
`artifacts/retired_evidence/v8_training_20261028_183317/review_20261029/RESULTS_REVIEW.md`.
The user requested removal of the weaker run's weights/checkpoints. Its historical
measurements and reviews remain unchanged in that archive. The retained model is
`artifacts/recovery_v8_safety_20261027/grace_rul_ALL.pt`, with its measured frozen
evaluation in `artifacts/v8_eval_20261028_180741/`. This revision requires a new run.
It does not change the inference architecture, format-8 tensor schema, physical
fault simulator, or measured historical results.

## What changed and why

| Evidence from fitting/validation records | Revised training policy |
| --- | --- |
| FD002's two otherwise-promising joint candidates failed intermittent missing-input reconstruction, with original-linear ratios 1.354 and 1.022. The healthy-window guard was not the cause. | Add a distinct original-linear missing-input objective, rather than pooling all faults only in compressed reconstruction loss. |
| Mean healthy-proposal tail losses reached 745.6/215.5 for FD002/FD004. Large proposals can dominate gradient clipping. | Use the same upper-tail window/channel ranking with a stable `log(1 + normalized MSE)` penalty. Ordinary RMSE guards remain unchanged. |
| Fault families were independently drawn per trajectory and could have unequal epoch exposure. | Stratify fitting engines across seven single mechanisms plus mixed, at severities 0.1/0.2/0.3, retaining the configured clean share. Rotate assignments every epoch. |
| Clean-only pretraining selected a weaker clean checkpoint because it scored better on mixed faults. | Select pretraining by clean validation RMSE, including its scheduler and early stopping. Carry the selected checkpoint's fault-macro score into later stages for consistent comparisons. |
| Later checkpoint selection used one mixed context even though evaluation reports all mechanisms. | Score recovery, joint repair/bypass, correction, tuning and seeds using half clean RMSE plus half the equal-family mean active-fault RMSE at severity 0.2. |

The configuration is explicit: `configs/v8_recovery_balance.json`. Neutral library
defaults preserve the earlier policy; the notebook enables this file through
`USE_RECOVERY_BALANCE_RECIPE = True`. CLI runs must supply `--config`. Existing
bundles retain their saved inference behavior. Do not resume an old training run
under the new configuration.

For each fitting-batch window/channel with genuinely missing real entries, the
new imputation term is

```text
log(1 + repaired_missing_MSE / (mean_fill_missing_MSE + budget^2))
```

Both MSEs use original-linear, regime-standardized sensor units. Mean fill is zero
in that coordinate system. The existing budget is 0.15. Groups are averaged
equally; clean entries and left padding do not enter this term. Encoded
reconstruction, temporal trend and RUL objectives still train the generator.
This is a smooth training surrogate; it does not certify the global reconstruction
ratio. Validation continues to measure that ratio directly.

The tail term applies `log(1 + group_MSE / budget^2)` to the largest configured
fraction of intact proposal groups (10% by default). The transform is monotone,
so their ordering is unchanged. Log-space sinh-difference calculations avoid
overflow without clipping the loss or removing extreme-proposal gradients.
The original mean intact loss remains separate. Reported errors use ordinary
sensor/RUL units.

Balanced exposure is an engine/epoch plan, not an exact batch quota. Fault onset,
duration and affected channels still come from the structured raw-unit simulator.
Engine count, epoch and local seeds determine assignment; future observations,
failure cycles and RUL do not. Overlapping windows preserve one corruption
realization. Training window counts are unchanged. Small OOF fits may not cover
every family/severity in one epoch; assignment rotation provides coverage over
epochs. The ten-epoch curriculum remains active.

## Guards and data boundaries

- Fitting engines alone train weights, preprocessing, support and loss scaling.
  Each OOF model retains its own disjoint inner validation engines. The corrector
  learns from OOF fitting-engine predictions and observable features.
- Fixed held training-validation engines select checkpoints, thresholds, correction
  bounds and seeds. Active-fault RMSE falls back to all-endpoint RMSE only when
  that context has no active endpoints. Every context remains logged separately.
- Calibration engines remain separate and are used only after weights are fixed.
  Official test values never enter fitting or selection. Previously inspected
  benchmark tests remain exploratory evidence.
- Mixed encoded reconstruction RMSE must still be at most 1.5; original-linear
  reconstruction ratio must still be at most 1.0. Clean RMSE and mean NASA score
  can increase by at most 1% relative to the original selected clean predictor.
- Every improving repair candidate must pass the complete declared recovery
  matrix (41 conditions in full mode). Aggregate, maximum-channel and
  maximum-window intact displacement limits remain 0.15. A failed repair can
  still leave a validated bypass. No repair is forced into a deployable bundle.

Full-family selection evaluates nine validation contexts instead of two. It adds
validation compute, without adding optimizer windows or new neural modules.
History files record the selection policy, every family score, admissibility
(`checkpoint_accepted`) and actual best-checkpoint replacement (`checkpoint_selected`).
Resumption stores the separate clean pretraining selection score.

## Short development pilot and its limits

`scripts/check_recovery_balance.py` compares three generator objectives on the
same fitting batches and starting weights: legacy losses, added imputation only,
and log-tail plus imputation. It uses existing fitting scopes and held validation
prefixes, freezes the detector/predictor, disables correction and exports no
model. FD002's rejected warm generator is activated only for this diagnostic.
It does not replace fresh main/OOF training or certify the full recovery matrix.
An access guard rejects official test/RUL data and known test/calibration reports.

The 40-step CPU pilot is saved locally at
`artifacts/recovery_balance_validation_20261029/`. Its report verifies unchanged
source artifacts, excluded calibration engines and no prohibited read attempts.
All ten predeclared validation cases and all variants remain saved.

For FD002, the new objective reduced the worst missing-family reconstruction
ratio from 1.088 before fine-tuning to 0.966, versus 0.984 for the paired legacy
objective. Its eight-family active-fault RMSE was 13.751, versus 13.902 for legacy
fine-tuning. FD004 active-fault RMSE was 14.340, versus 14.333 for legacy and
14.236 before fine-tuning: this pilot does **not** show a general FD004 improvement.
Individual faults can still worsen. These are development-validation measurements
from short generator fine-tuning, not new official test results, full-run acceptance
or evidence of superiority to the papers.

The earlier pilot's source weights have since been retired at the user's request;
its saved evidence remains available. A preflight can instead start from the
retained frozen bundle, which needs no training checkpoint. `--full-matrix` checks
all 41 sensor contexts without an early exit and retains the original clean
RMSE/NASA references from its saved upgrade report. `--retain-corrector` evaluates
RUL with the existing fixed corrector; it never refits that module. Sensor audits
always exclude correction. Only the generator receives fitting updates.

This is a short check before committing to full retraining:

```bash
source minor/bin/activate
python -u scripts/check_recovery_balance.py \
  --source-run artifacts/recovery_v8_safety_20261027 \
  --subsets FD002 --steps 40 --device cpu \
  --full-matrix --retain-corrector \
  --output-dir "artifacts/fd002_preflight_$(date -u +%Y%m%d_%H%M%S)"
```

The preflight fills missing training-only options from declared defaults, since a
frozen upgrade retains the older weight-training configuration. The report logs
the effective tail weight/fraction, fitting scope, original clean references,
all per-case measurements, frozen-module preservation and complete matrix audits.
Absent original clean references prevent a clean-guard pass. It compares the new
objective both to the original retained state and to a paired old-objective
fine-tune using the same fitting batches. It saves no deployable model.

The completed FD002 preflight is
`artifacts/fd002_preflight_complete_recipe_20261029/`. Each variant used 40 fitting
steps at batch size 16, full model dimensions, and all 52 held validation engines
with three fixed observed prefixes. It did not open official test data or use the
21 calibration engines. The revised candidate passed all 41 sensor conditions
and the original clean RMSE/NASA guards. Its eight-family active-fault RMSE was
12.965, versus 13.083 for the paired old objective and 12.781 before fine-tuning.
Thus it improved over the short old-objective update but did not beat the retained
starting model. This is not a demonstrated accuracy gain. Keep the retained bundle
and investigate that validation regression before the expensive training study.
A fine-tune of an existing working model cannot prove that fresh main/OOF training
will select recovery or achieve better accuracy.

## Next full run

Use the existing `minor` environment and a fresh directory. The GPU check must
pass in the same terminal. This command runs full dimensions on FD002/FD004 at
seed 42, including independent OOF training, correction, calibration and extended
evaluation. It skips tuning/comparator fits to keep this next trial focused.

```bash
cd "/home/vishal/D DRIVE/minor/cmapss-predictive-maintenance"
source minor/bin/activate
python scripts/check_cuda.py
set -o pipefail
v8_train_dir="artifacts/v8_recovery_balance_$(date -u +%Y%m%d_%H%M%S)"
python -u scripts/train_research.py \
  --device cuda --subsets FD002 FD004 --seeds 42 \
  --config configs/v8_recovery_balance.json \
  --skip-tuning --skip-benchmarks --extended-evaluation \
  --output-dir "$v8_train_dir" \
  2>&1 | tee "${v8_train_dir}.log"
```

After interruption, use the printed directory literally with unchanged settings.
Review all validation guards and the saved repair/bypass and correction decisions
before expanding to multiple seeds. Full accuracy improvement remains unmeasured
until retraining; a passing implementation test does not establish it. Preserve
failures alongside gains. The manuscript remains bound to its existing measured
runs until a new complete run is audited.

The [remaining studies runbook](RUN_REMAINING_STUDIES.md) covers explanations,
multiple seeds, matched comparisons and later paper updates. Generated results,
weights and logs remain under ignored `artifacts/`; only source, tests,
configuration, notebook and documentation are committed.
