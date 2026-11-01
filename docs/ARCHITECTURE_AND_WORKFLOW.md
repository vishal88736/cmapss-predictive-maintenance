# GRACE-RUL: format-5 as-trained architecture and workflow

Frozen run description, 26 October 2026. Current format-6 source changes are in
[RECOVERY_V6.md](RECOVERY_V6.md); the results below remain format-5 evidence.

Source-verified baseline description, 26 October 2026. Architecture format 5; corruption
protocol `structured-sensor-faults-v4`. This document describes the current code and
the trained run `full_20261025T211530540186Z`, including its observed failures.

Read [the saved results](RESULTS_2026-10-26.md) alongside this guide. The
[results snapshot](results/full_20261025T211530540186Z/README.md) contains the actual
tables, predictions, diagnostics, selected configurations and integrity manifest.
The [visual gallery](figures/full_20261025T211530540186Z/GALLERY.md) plots the saved
results, training histories, recovery failures and calibration evidence.

## 1. What the system does

GRACE-RUL estimates the number of operational cycles remaining before an engine
fails. It receives the engine's observed sensor history and operating settings.
During training, it creates additional sensor-error scenarios, learns to detect
finite errors and recover readings, and learns to estimate RUL from those readings.

The intended computation is:

**Observed history → normalization → availability and fault detection → selective
BiLSTM recovery → local CNN–GRU and global Transformer → quality-based fusion → base
RUL → bounded supervised correction → final RUL.**

Empirical intervals are added using calibration residuals. SHAP/LIME can subsequently
explain a frozen prediction. They are optional analysis steps, not prediction layers
and not part of the gradient-based training loss in this implementation.

The architecture contains the recovery path, but each saved model can bypass it
when validation rejects recovery. FD002/FD004 currently do so. This distinction is
necessary when describing what the trained system actually accomplishes.

![Architecture, including optional recovery and training-only critic](diagrams/GRACE_RUL_current_architecture.svg)

## 2. How the three papers contribute

| Source | Retained idea | Actual implementation and boundary |
| --- | --- | --- |
| Paper A: CNN–GRU, `1787574110586-178711702.pdf` | Convolutional local features, recurrent degradation representation, interpretation | Last-30-cycle CNN–GRU branch; post-training SHAP/LIME infrastructure. The paper's generalized REE objectives are not reproduced literally. |
| Paper B: partial sensor failure, `1787574110595-209939890.pdf` | Paired corruption/recovery, BiLSTM generator, WGAN-GP and staged training | Residual BiLSTM recovery, CNN–LSTM critic, reconstruction and RUL losses. A learned detector supplies deployment-available finite-fault weights. |
| Paper C: Trans-D3, `1787574864583-423798354.pdf` | Transformer context and bounded prediction correction | A 70-cycle Transformer branch plus an engine-disjoint supervised residual corrector. It is not TD3 or an actor–critic policy. |

The reliability gate, availability handling, detector and combined protocol are
integration choices. These are adaptations of the papers, not exact reproductions.
See [the paper/design analysis](FINAL_ARCHITECTURE.md) for the earlier paper review.
Only matched experiments can establish improvement over comparators.

## 3. Data and prediction targets

Every trajectory row has 26 columns: engine ID, cycle, three settings and 21 sensor
columns. IDs/cycles organize the data; the neural input has **24 measured features**.
The final two speed-demand columns among the 21 sensor columns are preserved by the
sensor-fault simulator because they describe commands.

These are the actual local file counts, not assumed counts from a paper/readme:

| Subset | Train engines | Test engines | Train rows | Test rows | Fitting / calibration / validation engines |
| --- | ---: | ---: | ---: | ---: | --- |
| FD001 | 100 | 100 | 20,631 | 13,096 | 72 / 8 / 20 |
| FD002 | 260 | 259 | 53,759 | 33,991 | 187 / 21 / 52 |
| FD003 | 100 | 100 | 24,720 | 16,596 | 72 / 8 / 20 |
| FD004 | 249 | 248 | 61,249 | 41,214 | 179 / 20 / 50 |

For a training engine, `raw_RUL(t) = final_training_cycle - t`. For a test history,
the official endpoint RUL plus the distance from that endpoint supplies labels for
earlier observations. These labels are targets/evaluation information, never inputs.

Training uses `y(t) = min(125, raw_RUL(t))`; prediction losses divide RUL by 125.
Evaluation separately reports capped and original official targets. A model trained
on capped targets is learning a limited horizon. Its output is nonnegative but is
not hard-clamped to 125, so a post-processing clamp change cannot remove the
limited-horizon target assumption.

### 3.1 Engine-level partitions

Use split seed 42. First reserve 20% of training engines for validation; reserve
10% of the remaining 80% for calibration. Roughly 72% of engines fit neural weights,
8% calibrate intervals, and 20% select settings/checkpoints. Rounding occurs at the
engine level. All observations/windows from one engine stay together.

The same partitions are used across initialization seeds and tuning candidates.
Official test engines do not fit normalizers, neural weights, thresholds or selected
seeds. Test metrics are inspection evidence, not a further tuning objective.

### 3.2 Normalization and operating regimes

`OperatingRegimeScaler` fits only the fitting engines:

1. Standardize the three operating settings using fitting means/standard deviations.
2. Use one regime for FD001/FD003, or fit six setting-based K-means centers for
   FD002/FD004. Assign each row to its closest fitting center.
3. Standardize each sensor using its fitting mean and standard deviation for that
   regime. A fitting measurement-increment floor protects nearly constant channels.
4. Separately save raw-unit fault scales from within-regime fitting variation and
   resolution/fallback floors. These scale added faults, not the model's targets.

The saved scaler is reused for validation, calibration, test and live inference.
The tiny within-regime scale of some speed sensors is a current weakness when a
physically stuck sensor crosses regimes: a plausible raw error can become thousands
of normalized units. See section 11 for observed evidence.

### 3.3 Window construction and masks

For an endpoint at cycle `t`, use at most the last 70 observations through `t`.
Short histories are left-padded with zeros and a separate boolean validity mask.
Windows never cross an engine boundary. Models do not receive observations after
the prediction endpoint.

| Object | Shape for batch size B | Meaning |
| --- | --- | --- |
| Input history X | B × 70 × 24 | Settings plus normalized observed sensors; missing sensors can be NaN |
| Validity V | B × 70 | Real observation versus left padding |
| Reference Y | B × 70 × 21 | Original sensor readings for training supervision; already naturally noisy |
| Availability M | B × 70 × 21 | Whether a real sensor reading is finite |
| Altered labels A | B × 70 × 21 | Which readings synthetic corruption actually changed; training/audit only |
| RUL y | B | Capped training target |

`M` identifies missingness. It does not identify a finite bias or drift. That is
why the detector is separate. A numerical zero is a valid standardized measurement;
zero filling alone is not an availability signal.

Training endpoints use stride 1 by default; validation tuning can choose stride 5.
Validation samples three reproducible prefixes per engine, correction six, and
calibration one. Official test metrics use one final observed endpoint per engine.
Engine-balanced sampling gives each fitting engine equal total sampling weight.

## 4. Structured corruption: what is added and why

Original C-MAPSS histories already contain simulated noise, manufacturing variation
and degradation. The original histories are reference data, not noise-free truth.
Added errors model sensor/acquisition failures while preserving original engine
degradation and RUL labels. Simulating additional component damage would require a
consistent engine model and new failure-time labels.

Faults select one related measurement domain: temperature, pressure, fan speed,
core speed, coolant flow, or an individual derived quantity. The domains are
measurement-based hypotheses, not verified C-MAPSS electrical wiring. Settings and
speed commands stay intact.

| Family | Mechanism |
| --- | --- |
| Intermittent | Shared burst dropout on selected channels; minimum two observed cycles, configured mean four |
| Outage | Channels become unavailable persistently after onset |
| Block | Missing blocks recur; default maximum intact gap 50 cycles |
| Noise | Added AR(1) error with rho .9; 80% common-source and 20% individual innovation variance |
| Bias | Persistent additive raw-unit calibration offset |
| Drift | Additive raw-unit offset ramps over 100 cycles then holds its bound |
| Stuck | A physical reading is held at onset even when the operating regime changes |
| Mixed | Connector: noise/dropout; calibration: bias/drift; acquisition: stuck/block |

Mixed mechanisms share a domain, channels and onset, with independent random
streams. Observed regime transitions increase noise amplitude and burst-entry risk
through a causal stress proxy. A real error process can be stochastic; the important
structure here is its source, related channels, persistence and operating context.

Onset is an absolute observed offset between 0 and 100 by default, independent of
eventual failure time. Fault streams preserve an observed prefix if unseen future
rows are appended with the same profile/seed. Cached trajectories give overlapping
windows identical fault histories. Actual changed-entry labels include finite
errors as well as missingness.

Training samples severity .1/.2/.3 under a ten-epoch curriculum. Expected clean
trajectory-realization probability is .25 by default, .50 for selected FD001. It is
not an exact per-batch quota. A severity value is a mechanism control, not a global
percentage of bad cells. Stuck faults can have a large physical amplitude across
regimes even at a small severity. Full equations/assumptions are in
[STRUCTURED_SENSOR_FAULTS.md](STRUCTURED_SENSOR_FAULTS.md).

## 5. Neural architecture, module by module

Dimensions below describe all four exported models. Their training hyperparameters
can differ, but these tensor dimensions are identical. Recurrent processing ignores
padding using packed sequences and then restores the original window positions.

### 5.1 Fault detector: available does not necessarily mean correct

Build a 45-feature vector per cycle: 21 zero-filled observed sensors + 21 availability
indicators + 3 settings. Parallel temporal convolutions have kernels 3 and 5, each
mapping 45 to 32 channels with ReLU. Concatenation produces 64 channels. A one-layer
GRU with hidden size 64 produces a representation at each observed time. A linear
64 → 21 head gives sensor logits; sigmoid gives `p[t,j]`.

The head starts with zero weights and bias -4, approximately p=.018. Its loss is
weighted binary cross entropy on finite, valid entries. Per-sensor positive weight
is `sqrt((negative_count+1)/(positive_count+1))`, clipped between 1 and 5. Missing
entries already have observable availability, so this classifier targets finite
measurement errors. Scores are learned outputs, not automatically calibrated
probabilities of a physical fault.

### 5.2 Residual BiLSTM generator

Input per cycle: observed sensors (21), availability (21), detector scores (21),
settings (3): **66 features**. A two-layer bidirectional LSTM has hidden size 64 in
each direction, producing 128 features per cycle. The decoder is 128 → 64 → 21 with
ReLU between linear layers. It predicts a residual sensor correction:

```text
generated = observed_zero_filled + decoder(BiLSTM(inputs))
```

The decoder's final layer starts at zero. Initially it preserves finite observed
readings and gives mean imputation (zero in standardized space) for missing ones.
The residual has no hard amplitude bound. Bidirectionality uses the already
observed 70-cycle window, not observations after the endpoint.

### 5.3 Selective recovery and the fallback

For real, non-padding entries:

```text
w = (1 - M) + M * p
R = (1 - w) * observed_zero_filled + w * generated
```

A missing sensor has M=0, so it uses the generated estimate. A finite sensor uses
a blend determined by its estimated fault score. Small p preserves most of the
observed value. Both reconstructed and original values matter later through the
quality summary.

If the saved `repair_enabled` buffer is false, R equals zero-filled observations
and recovery weights passed to the predictor are zero. The main configuration
still instantiates/computes its detector, but that does not establish a trained
detector. In FD002/FD004 the saved head is still at its initial constant score.

### 5.4 Local CNN–GRU branch

Each cycle gets R (21), settings (3), M (21) and w (21): **66 features**. The local
branch takes the last 30 observations. Parallel kernel-3/kernel-5 convolutions
map 66 → 32 each, concatenate to 64, and feed a GRU with hidden size 64. Its last
valid output is projected 64 → 64 to obtain local feature h_local.

This branch extracts short-range patterns and their recent temporal evolution.
With recovery enabled, those 30 repaired readings can indirectly depend on the
full observed recovery window. It is not a strictly isolated 30-cycle model.

### 5.5 Global Transformer branch

The same 66 input features over the 70-cycle window are projected to 64 dimensions
and receive sinusoidal positions. Two Transformer encoder layers each use four
attention heads, feed-forward dimension 128, GELU and dropout .2. The last valid
token is global feature h_global (64 dimensions).

Attention masks padding. It can attend throughout the observed window; there is
no future after the prediction endpoint in that window. This is endpoint inference,
not a claim of strictly causal attention at every intermediate token.

### 5.6 Quality vector and learned fusion

The 64-dimensional quality vector q contains:

- 21 mean availabilities over real observations;
- 21 mean recovery weights;
- 21 mean absolute differences between repaired and observed values;
- one observed-history fraction, real observations / 70.

Concatenate h_local (64), h_global (64) and q (64). A 192 → 32 → 2 MLP with ReLU
and final softmax gives nonnegative local/global weights summing to one:

```text
[g_local, g_global] = softmax(gate([h_local, h_global, q]))
h_fused = g_local * h_local + g_global * h_global
```

The weights are learned model behavior, not proof of sensor reliability. Optional
ablation modes force a single branch or equal .5/.5 fusion.

### 5.7 Base and auxiliary RUL heads

The main head takes h_fused (64) + q (64). Layers are 128 → 64, ReLU, dropout .2,
and 64 → 1. `softplus(output) * 125` gives nonnegative base RUL. Auxiliary local
and global 64 → 1 heads have the same softplus/scale transformation and support
branch supervision.

### 5.8 Bounded supervised residual corrector

Its 67 observable features are q (64), base/125, local prediction/125 and global
prediction/125. A 67 → 32 → 1 MLP with ReLU gives a scalar transformed by tanh:

```text
delta = selected_bound * tanh(corrector(features))
final_RUL = max(0, base_RUL + delta)
```

Validation searches correction bounds 5, 10, 15 and 30 cycles. The output layer
starts at zero, and zero correction remains an explicit fallback. It is fitted
on out-of-fold features from engines the temporary upstream model did not train
or select on. Arbitrary hidden embeddings are excluded because independently
trained fold models need not share their coordinate basis.

The corrector never receives true current RUL or a test label as an input. It is
supervised correction rather than TD3. Bound limits delta, not the maximum base RUL.

### 5.9 Training-only WGAN-GP critic

Input: reference or repaired sensors (21) plus settings (3), or 24 features.
Temporal convolutions 24 → 32 → 64 use kernel 3 and LeakyReLU .2. An LSTM with
hidden size 64 and a linear 64 → 1 head produce a conditional Wasserstein score.

The critic loss is mean(fake score) - mean(real score) plus gradient penalty with
coefficient 10. The generator adversarial term is -mean(fake score). The penalty
uses an interpolated sub-batch of at most 16 for memory cost, while the Wasserstein
term uses the full batch. Recurrent gradient-penalty computation supports second
derivatives through an unfused backend. Critic optimization uses Adam with betas
.5/.999 and learning rate .0001; critic updates default to one per training batch.
The first five generator/joint epochs omit the adversarial term.

The critic is not loaded as a deployed prediction component. Its training scores
are not sensor-fault probabilities.

### 5.10 Measured parameter counts

| Component | Parameters per subset |
| --- | ---: |
| Detector | 37,909 |
| BiLSTM residual generator | 176,533 |
| CNN–GRU + Transformer + quality gate + RUL heads | 132,005 |
| Residual corrector | 2,209 |
| **Instantiated deployed model total** | **348,656** |
| Separate training-only critic | 41,889 |

The bundle contains four independently selected models, one per subset. Instantiated
parameter count does not mean every module is active or successfully learned.

## 6. Training objectives

Except during clean predictor pretraining, the total objective sums the following
terms. Sensor terms exclude padding. RUL terms use targets/predictions divided by
125. `A` is actual changed-entry supervision, never a deployment feature.

| Term | Default weight | Purpose |
| --- | ---: | --- |
| Base RUL MSE | 1.0 | Predict capped RUL from the processed input |
| Repaired altered-entry MSE | 1.0 | Recover damaged readings after selective blending |
| Direct generated altered-entry MSE | .25 | Train the generator even when initial detector weights are small |
| Intact-entry identity MSE | .5 | Limit changes to readings left intact by the simulator |
| Finite-fault weighted BCE | .2 | Learn which available readings were changed |
| First-difference trend MSE | .1 | Preserve temporal changes on valid adjacent pairs touched by corruption |
| Generator adversarial loss | .01 | Encourage repaired sequences that resemble references conditional on settings |
| Mild-fault consistency MSE | .1 | Match a detached clean-reference prediction for mildly affected histories |
| Auxiliary branch RUL MSE | .1 | Give local and global branches direct target supervision |
| Clean-reference RUL MSE | .25 | Anchor the trainable predictor to the reference input |

Consistency applies only when at least 90% of readings are available and average
finite perturbation is at most .05 normalized units. The reference pass disables
dropout; this is the current model evaluated on reference inputs, not an EMA teacher.
The corrector has a separate normalized RUL MSE plus `.02 * mean((delta/125)^2)`.
Exact selected loss overrides are in section 8.

## 7. Complete training workflow

![Training, validation, OOF correction, selection and export](diagrams/GRACE_RUL_training_workflow.svg)

### Step 1 — Validate configuration and source data

Check source format, engine/cycle uniqueness, disjoint partitions and numerical
configuration. Explicit CUDA requests fail if the current process cannot use CUDA.
Read actual file counts and preserve source hashes. Create a fresh timestamped run
directory for a new experiment.

### Step 2 — Validation-only hyperparameter search

Each subset gets six bounded trials: default; stride 5; stronger direct/trend
recovery; stronger intact protection; zero adversarial loss; more clean examples
and a stronger clean anchor. All trials share fitting/validation engines and seed
42. They open only training data, not official test inputs/labels. The lowest
accepted validation clean/active-mixed objective selects per-subset overrides.
This modest search is not an exhaustive optimization or an original-paper reproduction.

### Step 3 — Train the upstream model for each initialization seed

Use seeds 42, 123, 2026, 314 and 777. Each subset/seed gets these phases:

| Phase | Trainable modules | Inputs and losses | Maximum epochs / initial LR |
| --- | --- | --- | --- |
| Clean predictor | Predictor, including branches/gate/heads | Reference input; main and branch RUL loss; recovery/correction bypassed | 100 / .001 |
| Recovery | Detector + generator; separate critic when warmup ends | Corrupted/reference pairs; recovery, detection, RUL and other applicable losses; predictor frozen | 50 / .0001 |
| Joint | Upstream detector, generator and predictor; corrector excluded | Corrupted inputs with combined objectives and clean anchor | 20 / .0001 |

Neural optimizers are AdamW with weight decay .01, batch size 128 and gradient
norm clipping at 5. The sampler balances engines. ReduceLROnPlateau halves learning
rate after five stagnant validation epochs, down to .000001. Early-stopping patience
is 15. Schedules are upper limits; training restores selected checkpoints.

### Step 4 — Validate and guard each checkpoint

Clean and fixed mixed-fault validation use the same held-out endpoints across
seeds. The selection score is:

```text
0.5 * (clean validation RMSE + active-mixed-fault validation RMSE)
```

If a scenario exposes no active faults, the implementation falls back to its all-
endpoint RMSE. A repair/joint checkpoint must additionally satisfy:

- clean RMSE no more than 1% above the selected pretrained clean predictor;
- intact-entry normalized repair RMSE at most .15;
- altered-entry reconstruction RMSE at most 1.5.

Among admissible improvements, retain the best score. If repair never improves
within those limits, retain the predictor fallback with recovery disabled. Joint
training can retain the selected recovery checkpoint if its updates are rejected.
In the current multi-regime run, rejection discarded all joint detector/predictor
updates too; a dedicated corrupted-input fallback needs further work.

Atomic epoch checkpoints save weights, critic, optimizers, schedulers, RNG states,
history and selection state. Resuming a matching run restores completed epochs.
Architecture/protocol/configuration/data identity must match. Do not overwrite the
saved baseline when changing the representation or training protocol.

### Step 5 — Produce engine-disjoint out-of-fold correction features

Split only fitting engines into three folds. For each held fold, divide the other
engines into inner fitting and inner validation groups. Fit a temporary upstream
model and its own scaler without the held fold. Generate clean/mixed held-engine
predictions and the 67 observable correction features from that model.

Collect those features over the three held folds. No engine supplies correction
features from an upstream model trained or checkpoint-selected on that engine.
Saved cross-validation metrics describe these upstream models before final
correction; they are not nested CV of the complete corrected system.

### Step 6 — Fit and select the bounded corrector

Freeze the final upstream model. Fit the correction MLP on OOF features/labels for
each candidate bound, at most 50 epochs with learning rate .001 and patience 15.
Select by clean/active-mixed validation RMSE and the original clean-predictor
tolerance. The zero-correction baseline remains available if trained correction
does not improve. Clean tolerance is not compounded at successive stages.

### Step 7 — Calibrate finite-fault thresholds and intervals

For each sensor, select a detector threshold using validation clean/bias/drift/noise
inputs under a per-channel false-positive constraint of at most 5%. Current source
searches only 60 thresholds from .05 to .99; this is an observed calibration weakness,
especially when learned score boundaries lie outside/between candidates.

On separate calibration engines, evaluate one reference endpoint per engine. Residual
quantiles at alpha/2 and 1-alpha/2, with alpha=.1, become lower/upper interval offsets.
Apply offsets to final predictions with nonnegative bounds. This is empirical
calibration, not a conformal or distribution-shift coverage guarantee.

### Step 8 — Evaluate official endpoints and select the exported seed

Each trained seed produces one clean final-window prediction per official test
engine. Report RMSE, MAE, NASA score, R² and interval coverage for capped and uncapped
targets separately. Aggregate mean and sample standard deviation across five seeds.

All 20 primary seed models are saved. The exported seed for each subset is the one
with lowest post-correction validation score. It is not chosen by best test score.

### Step 9 — Export a complete inference bundle

`grace_rul_ALL.pt` contains four selected model states, their selected configurations,
scalers, threshold vectors, repair/correction metadata, empirical intervals, engine
partitions/OOF audit, source hashes and 48 fitting reference backgrounds per model.
Background selection spreads regimes, life stages and distinct fitting engines.
The original critic is not a deployed model component.

The loader validates format/version, feature order, thresholds, correction metadata
and finite weights. Run manifests identify the architecture, protocol, subsets and
whether the artifact is a quick execution check or a full experiment.

### Step 10 — Optional comparisons, robustness and explanations

`RUN_PAPER_STUDY=False` / omitted `--paper-study` skips separate comparator training
and extended reports. It does not reduce the primary architecture or remove its
core staged training, tuning, OOF correction or calibration. Core defaults schedule
104 upstream fits: 24 search trials plus 20 main and 60 OOF models.

Enabled comparisons independently tune paper-inspired CNN–GRU, repair CNN–LSTM and
Transformer/correction adaptations, plus controlled GRACE ablations. Full robustness
uses each fault family plus mixed scenarios, five severities and three replicates.
Reports distinguish all endpoints, active faults and unaffected cases. Paired
bootstrap comparisons preserve engine clusters and corruption replicates. Original
published scores are not automatically comparable to this harmonized protocol.

These optional studies and SHAP/LIME were off in the saved run. The added result
review contains only a clearly scoped frozen-selected-model fault check, not a full
multi-seed robustness/comparator study.

## 8. What was actually selected and exported

Validation chose these overrides to the default training configuration:

| Subset | Selected search changes |
| --- | --- |
| FD001 | Clean realization probability .50; clean-reference RUL loss weight .50 |
| FD002 | Training-window stride 5 |
| FD003 | Repaired reconstruction weight .50; intact identity weight 1.0 |
| FD004 | Training-window stride 5 |

These are the actual exported computation paths, not a test-selected best case:

| Subset | Exported seed | Learned repair retained | Corrector output retained | Selected correction bound |
| --- | ---: | --- | --- | ---: |
| FD001 | 2026 | Yes | Zero correction fallback | 15 |
| FD002 | 42 | No | Nonzero learned correction | 30 |
| FD003 | 42 | Yes | Nonzero learned correction | 30 |
| FD004 | 42 | No | Nonzero learned correction | 5 |

Across all five seeds, repair was retained in 2/5 FD001, 0/5 FD002, 3/5 FD003 and
0/5 FD004 models. The 15-cycle bound in FD001 does not imply correction is active:
its selected corrector's final weights/bias are zero, so delta is exactly zero.

## 9. Deployment and explanation workflow

### 9.1 Prediction

1. Load a full bundle, automatically discover the latest full run, or select one
   explicitly with `GRACE_MODEL_PATH`. The notebook and API use the same model classes.
2. Parse the uploaded measurements; validate schema, subset, settings, engine/cycle
   order and duplicate rows. Keep missing sensor values as missingness information.
3. Group by subset and engine. Apply that subset's saved scaler without refitting.
4. Build its last 70-cycle window and masks. The website also samples earlier
   endpoints for a trajectory display; the latest endpoint is the current estimate.
5. Run detector, selected recovery/fallback, dual branches, fusion, base head and
   selected correction. No synthetic corruption is injected into a user upload.
6. Attach saved empirical intervals. Convert repaired sensor values back to raw
   units using the corresponding observed regimes.
7. Return final/base RUL, correction, local/global gates, availability, sensor scores,
   suspected spans, intervals and optional reference-label errors. Labels are only
   used to calculate evaluation errors when present.

Optional MC dropout runs the selected model repeatedly to show prediction variability
(default 20 passes). This is distinct from calibrated residual intervals and does
not turn them into guarantees.

### 9.2 SHAP and LIME

Use a frozen model, one real input window and saved fitting backgrounds. Divide
each sensor's observed history into three coherent temporal blocks: up to 63
coalition features. Select up to three background references by observed settings,
preferring complete horizons; shorter references are resampled monotonically.

A coalition keeps an observed sensor block or replaces its finite values with
background values. Operating settings, actual missingness and padding stay fixed.
Each coalition reruns the complete selected pipeline. Actual Kernel SHAP calculates
contributions, and actual LIME fits a local weighted Ridge surrogate. Aggregate
temporal contributions to sensor-level values, retaining separate SHAP/LIME fields.

Explain final RUL, and optionally a sensor's estimated finite-fault score if learned
recovery/detection is available. Report SHAP local accuracy, LIME holdout fidelity,
local errors and rank agreement. The global study defaults to 12 distinct held-out
engines, three perturbation seeds and 256 samples per case; it records equal-engine
mean absolute importance and stability. These settings are optional compute costs.

An attribution describes model dependence under this background/perturbation scheme.
It does not establish the physical cause of a failure. Current FD002/FD004 detector
scores are constant, so their learned finite-fault explanation is unavailable. No
actual SHAP/LIME outputs were generated in the preserved training run.

### 9.3 API interfaces

| Endpoint | Role |
| --- | --- |
| `GET /api/status`, `GET /api/model` | Artifact status, selected paths and available result tables |
| `POST /api/preview` | Parse/inspect an upload |
| `POST /api/predict` | Actual trained-model inference and optional MC dropout |
| `POST /api/explain` | Calculated SHAP/LIME for a selected engine/window |
| `GET /api/sample`, `GET /health` | Example input and service status |

This documents the implemented service contract. The result review verified the
bundle/CPU model directly; it did not establish a new live browser or API validation.

## 10. Main source files and saved outputs

| File | Responsibility |
| --- | --- |
| `grace_rul/config.py` | Dimensions, targets, protocol and default configuration |
| `grace_rul/data.py` | Parsing, engine split, fitting scaler, windows, masks and backgrounds |
| `grace_rul/corruption.py` | Structured, prefix-consistent sensor-error simulation |
| `grace_rul/model.py` | Detector, generator, critic, predictors, gate and corrector |
| `grace_rul/training.py` | Objectives, staged fitting, guarded selection, OOF correction, thresholding and calibration |
| `grace_rul/tuning.py` | Training/validation-only search |
| `grace_rul/checkpoints.py` | Atomic checkpoints, RNG state and compatibility identity |
| `grace_rul/artifacts.py` | Main experiment driver, result tables, bundle export/load |
| `grace_rul/benchmark.py` | Optional independent comparisons and paired analysis |
| `grace_rul/explain.py` | Actual local/global SHAP/LIME calculations |
| `scripts/train_research.py`, notebook | User-facing training entrypoints |
| `website/backend/service.py` | Bundle discovery and model-backed prediction/explanation |

`test_seed_summary.csv` summarizes seeds; `test_metrics.csv` retains each protocol/
seed; `predictions_*.csv` retain actual endpoint predictions; `seed_summary.csv`
records selections; histories record acceptance/rejection. Calibration, OOF audit,
background audit and tuning files make these decisions reviewable. The saved
snapshot manifest hashes all copied results and references the original trained
bundle plus 20 seed-model binaries. Model binaries remain in their original run.

## 11. What this trained run establishes—and where it fails

The clean five-seed capped RMSE values are 13.45, 11.92, 12.69 and 13.71 cycles for
FD001–FD004. Original uncapped RMSE values are 14.46, 24.89, 14.07 and 25.18.
See [the results report](RESULTS_2026-10-26.md) for full tables and interpretation.

The main integration failure is multi-regime recovery. Every attempted FD002/FD004
repair/joint checkpoint exceeded the reconstruction guard. Stuck speed readings
can produce approximately 5090/1977 normalized error units in diagnostic cases.
The stored models consequently bypass recovery and retain constant detector scores
near .018. A single-replicate frozen check gave stuck-fault capped RMSE 43.10 on
FD002 and 48.54 on FD004. Detection is weak on the other subsets too.

Long-life engines above the 125-cycle training horizon dominate much of FD002/FD004
uncapped error. Nominal 90% intervals cover only about 77–79% of clean capped test
targets on FD001/FD002/FD004. No matched comparison establishes superiority over
the three papers. The architecture should be described as implemented and trained,
with these limitations, rather than as a proven improved three-paper model.

Prioritize scale-aware recovery and retained detector/corrupted-input training,
then validate threshold calibration and intervals. Further tuning must use fitting/
validation engines rather than selecting changes by the inspected official test
outcomes. Retain this run as baseline evidence when implementing those changes.

## 12. Commands using the existing `minor` environment

From the project root, the existing local environment is `minor/bin/python`.

```bash
# Read and inspect the preserved run; these commands do not retrain it.
minor/bin/python -c "from grace_rul.artifacts import load_bundle; b,m=load_bundle('artifacts/full_20261025T211530540186Z/grace_rul_ALL.pt', device='cpu'); print({s: bool(v['model'].repair_enabled) for s,v in m.items()})"

# Runtime/neural checks before a future experiment.
minor/bin/python scripts/check_cuda.py --training-check
minor/bin/python -m unittest discover -s tests

# A small execution run; its outputs are not accuracy evidence.
minor/bin/python scripts/train_research.py --quick --device cuda --skip-tuning --skip-benchmarks

# Full primary training after the outstanding design issues are addressed.
# Omitting --output-dir creates a fresh timestamped directory.
minor/bin/python scripts/train_research.py --device cuda
```

Optional post-training explanations use `scripts/explain_research.py`; optional
comparisons use `--paper-study`. Neither is needed to load the existing trained
bundle. Changing the training representation/protocol requires a fresh run, not
resuming or overwriting this preserved baseline. Optional report switches can reuse
compatible upstream checkpoints, but correction/calibration/evaluation may rerun.

Notebook configuration for primary training remains `QUICK_RUN=False`,
`RUN_PAPER_STUDY=False`, `RUN_EXPLANATIONS=False`. Choosing a local Python environment
does not make a GPU visible; verify CUDA in the terminal used for training.
