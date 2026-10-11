# Implementation validation

Validated locally on 11 October 2026 after the training audit and improvements, using Python 3.12,
the project's `.venv`, and the original C-MAPSS files under `data/`. Initial checks
used PyTorch 2.14.1+cpu; the environment now has **2.14.1+cu130** and actual CUDA
training has passed. Current artifacts use format 5.

## Completed checks

- All ten notebook code cells executed successfully in 18.48 seconds in quick
  CPU mode across FD001–FD004, including
  staged training with actual WGAN-GP updates, three correction folds, empirical
  calibration, corruption evaluations, export/reload, plots, Kernel SHAP, and LIME.
- Thirty-seven regression checks passed in 18.36 seconds with
  `python -m unittest discover -s tests -v`.
  These cover all four dataset counts and normalization regimes, engine separation,
  finite faults versus missingness, persistent corruption across overlapping windows,
  exclusion of future input, padding behavior, complete-outage recovery, second
  derivatives for gradient penalty, and ordered empirical intervals.
  Added checks cover fitting-only sensor resolution floors, accurate drift labels,
  reproducible mixed faults, deterministic reference targets and restored module
  modes, correction invariance to hidden-coordinate changes, zero-correction and
  repair-stage checkpoint fallbacks, final-pipeline selection scores, and the
  gradient-penalty sub-batch. Checkpoint interruption/resume remains bit-exact on
  CPU, including completed-pipeline reuse and rejection of changed data/config.
  New checks cover residual identity initialization, affected-pair trend gradients,
  balanced engine sampling, stable multi-prefix validation, strict checkpoint limits,
  repair bypass, correction-bound selection, scheduler reduction across interruption,
  train-only tuning, comparator gradients, paired bootstrap scope, and trained
  comparisons sharing identical evaluation datasets, engines, and targets.
- API checks exercise multipart uploads, pending-model responses, explicit quick
  selection, real predictions, recovered readings, dropout spread, explanation
  additivity, and rejection of legacy model artifacts.
- The complete quick CLI workflow finished: five validation trials, primary staged
  training and OOF correction, all six separately trained comparators/ablations,
  shared corruptions, bootstrap intervals, CSVs, and gzip predictions. All quick
  acceptance checks correctly report incomplete protocol and no established claim.
- `pip check`, `git diff --check`, and the frontend production build passed.
  The website displays format 5, actual selected bounds/repair status, and connected
  trained-comparison tables. API checks include the repair-bypass explanation behavior.
  A project-local `grace-rul` Jupyter kernel is available.
- Default model discovery reports `awaiting_model`, format 5; quick artifacts
  remain excluded. No trained bundle remains in the project's model slot.
- SHA-256 checks confirmed that all 12 official train/test/RUL files were
  unchanged. The regenerated notebook has zero outputs and execution counts.

The interrupted October 8 run, its plot cache, and an unused truncated duplicate
label file were removed during cleanup; see [TRAINING_AUDIT.md](TRAINING_AUDIT.md).
Future notebook validation
creates fresh records under `artifacts/validation/` through
`scripts/validate_notebook.py`.

The checked-in notebook now defaults to `QUICK_RUN = False` and
`CONFIG['device'] = 'cuda'`. The validation runner changes these settings only
in its execution namespace to perform short CPU checks; it can use `--device cuda`
on an accessible GPU. Updated all-subset validation used a temporary output
directory, `/tmp/grace-research-notebook-v5-final`, and no saved run results remain
under the project's `artifacts/`.
The complete quick CLI run is under `/tmp/grace-research-workflow-v5-final`.
Test output is `/tmp/grace-final-tests.log`; the production build log is
`/tmp/grace-research-frontend-build.log`.

The split counts for seed 42 are:

| Subset | Training pool | Optimizer fitting | Calibration | Validation |
| --- | ---: | ---: | ---: | ---: |
| FD001 | 80 | 72 | 8 | 20 |
| FD002 | 208 | 187 | 21 | 52 |
| FD003 | 80 | 72 | 8 | 20 |
| FD004 | 199 | 179 | 20 | 50 |

Training pool = fitting + calibration. Whole-engine rounding applies.

## Scope of validation

Notebook cells were executed sequentially in one Python namespace without
launching a Jupyter server.
The installed Jupyter kernel points at the project's `.venv`. API routing and
multipart handling were exercised through HTTPX's in-process ASGI transport, with
thread dispatch made synchronous only in the test harness. Production endpoints
retain FastAPI's normal worker dispatch. A live server and browser interaction
were not verified in this restricted workspace.

The restricted workspace hides NVIDIA device files. An authorized host check
confirmed the RTX 4050 Laptop GPU and working NVIDIA driver 615.71.09. The earlier
GPU-unavailable report applied to the restricted session, rather than the host.
The CPU-only wheel was replaced with the matching official PyTorch 2.14.1+cu130
build. GPU commands must run in the normal terminal or an authorized execution
with device access. The training configuration continues to request CUDA explicitly
and never silently falls back to CPU.

## CUDA verification

- Real CUDA tensor forward/backward passed on the RTX 4050 with CUDA runtime 13.0.
- `scripts/check_cuda.py --training-check` passed predictor, generator, and joint
  stages using full default model dimensions, batch size 128, and WGAN-GP sub-batch
  16. One batch per stage was used; official test files were excluded. Peak allocated
  PyTorch tensor memory was 0.234 GiB, excluding driver/context and reserved memory.
- All ten notebook cells passed in **16.72 seconds** with quick CUDA training
  across FD001–FD004, including OOF correction, threshold selection, calibration,
  export/reload, and explanations. The model reload/explanation cell intentionally
  uses CPU, as supported by the website artifact contract.
- All **37 regression checks** passed again in **30.53 seconds** after replacing
  PyTorch, and `pip check`, shell syntax, and `git diff --check` passed.
- Setup now defaults to CUDA and verifies real GPU execution. `--cpu` is an
  explicit alternative. The training driver prints its device, GPU name, and
  PyTorch build before training.

GPU stage output is `/tmp/grace-cuda-training-check.log`; notebook records are
under `/tmp/grace-cuda-notebook-check`; post-install regression output is
`/tmp/grace-cuda-regression-tests.log`. These checks establish GPU execution,
not full five-seed benchmark accuracy. No full training job was started.

Small quick runs establish executable integration and produce explicitly labeled
validation artifacts. The separately trained comparison workflow is implemented
and executed at a quick budget. Full five-seed training, original-paper baseline
reproductions, full retrained ablations, real sensor-fault accuracy, and superiority
over the papers remain research experiments. Full-size model dimensions are
implemented but have not been benchmarked. Empirical interval coverage under
corruption or new operating conditions remains an evaluation result, not a guarantee.

## FD001 validation-only pilot

A separate limited pilot used all 72 fitting engines and the same 20 validation
engines, leaving eight calibration engines untouched. It opened only
`train_FD001.txt`, and used seed 42, fixed validation seed 31415, three prefixes per
validation engine, stride 5, batch size 128, six predictor epochs, three repair
epochs, and two joint epochs. It used reduced dimensions: generator/GRU hidden 16,
CNN channels 8, Transformer dimension 32, and gradient-penalty sub-batch 2.
There were no engine/window/batch limits. It did not fit final correction or
evaluate official test engines.

| Trial | Upstream clean/corrupted validation macro RMSE |
| --- | ---: |
| Default improved losses | 15.131099 |
| Greater direct reconstruction and trend weights | 15.131114 |
| Zero adversarial loss | 15.118617 |

The search selected zero adversarial loss. The improvement is approximately 0.08%
in this short single-seed pilot; stronger recovery weights did not help. These
results validate making adversarial/trend weights selectable rather than assuming
more loss terms improve accuracy. They do not establish a statistically reliable
gain, compare the old pipeline, or demonstrate superiority over any paper.
Pilot configuration, checkpoints, partitions, training hash, and scores are under
`/tmp/grace-improvement-pilot`; its log is `/tmp/grace-training-pilot.log`.
