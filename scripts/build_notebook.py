"""Rebuild the checked-in notebook from readable, deterministic source cells."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(source):
    add_cell("markdown", source)


def code(source, tags=None):
    compile(source.strip(), "notebook cell", "exec")
    add_cell("code", source, tags)


def add_cell(kind, source, tags=None):
    source = source.strip()
    cell = {"cell_type": kind, "id": hashlib.sha256(f"{len(cells)}:{source}".encode()).hexdigest()[:12],
            "metadata": {"tags": tags or []} if kind == "code" else {},
            "source": source.splitlines(keepends=True)}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    cells.append(cell)


md("""
# GRACE-RUL: corruption, recovery, prediction, and sensor fault explanations

This notebook implements the current architecture in `docs/RECOVERY_V8.md`;
`docs/FINAL_ARCHITECTURE.md` preserves the original paper integration:
**fault detector → selective BiLSTM recovery with WGAN-GP → gated CNN–GRU / Transformer → bounded residual correction → SHAP + LIME**.

It uses the `grace_rul/` package so the website loads exactly the same model definitions.
Keep this notebook, that package, and `data/` together. Install `requirements.txt` or run
`bash scripts/setup_env.sh`, then choose the **Python (GRACE-RUL)** kernel.

The default `QUICK_RUN = False` tunes on validation data and trains all four subsets
with five seeds using CUDA. Separate comparison models and extended evaluation reports
are disabled by default to reduce runtime while preserving the main training procedure.
PyTorch must detect an accessible GPU; a CUDA request never silently falls back to CPU.
Set it to `True` only for a small FD001 execution check. Quick artifacts are clearly
labeled and are not benchmark evidence. A full run has a substantial compute cost.
Neither this notebook nor its artifact claims superiority before matched experiments.

Format 8 adds fitting-only clean support for directed finite-repair bounds,
upper-tail healthy-proposal supervision, and mandatory recovery validation across
all declared fault families/severities. Large outlier recovery remains possible.
A frozen format-7 safety upgrade is available in `scripts/upgrade_recovery_safety.py`;
it recalibrates the exact changed model and exports only if held-validation guards pass.
New full training requires a fresh directory. Format-5/6/7 artifacts preserve
their historical inference behavior; improved accuracy must be measured.
""")
code("""
from pathlib import Path
import os, sys, json, platform

start = Path(os.environ.get('GRACE_PROJECT_ROOT', Path.cwd())).resolve()
PROJECT_ROOT = next((p for p in [start, *start.parents] if (p / 'grace_rul').is_dir() and (p / 'data').is_dir()), None)
if PROJECT_ROOT is None:
    raise FileNotFoundError('Open the notebook alongside grace_rul/ and data/, or set GRACE_PROJECT_ROOT.')
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(PROJECT_ROOT / 'artifacts' / '.matplotlib'))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, SVG
from grace_rul.config import DEFAULT_CONFIG, quick_config, SENSORS, SENSOR_NAMES
from grace_rul.model import GraceRUL
from grace_rul.data import load_subset, split_engines, OperatingRegimeScaler, WindowDataset
from grace_rul.training import seed_all, device_for
from grace_rul.artifacts import run_experiment, load_bundle
from grace_rul.explain import explain_window, run_explanation_study

print('Project:', PROJECT_ROOT)
print('Python:', platform.python_version(), '| Torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
""")
md("""
## Configuration and execution budget

The full configuration uses a 70-cycle global window, 30-cycle local branch, 21 sensors,
three settings, a two-layer BiLSTM generator, and two Transformer layers with four heads.
The residual generator starts by preserving observed values. Direct reconstruction,
temporal-trend supervision, and intact-entry identity losses train recovery. Full mode
uses stride-1 windows and balances sampling by engine. Learning-rate reduction and
clean/intact/reconstruction checkpoint limits are configurable. Joint training includes
clean-reference RUL supervision and permits at most 1% clean RMSE and mean NASA-score
regression against the original clean predictor by default. Detector fitting is separate;
its weights stay frozen during recovery/joint training. Finite repairs require per-sensor
thresholds selected under a 5% empirical intact-cell false-positive limit in every
validation fault context. Recovery/RUL gradients cannot change detector gates.
It also supervises prediction with recovery bypassed. Joint validation can retain
that path independently when recovery fails. Recovery must not increase original
linear sensor error; intact-reading protection is checked before asinh compression.
The curriculum reaches full fault severity within ten epochs, before default patience expires.
Corruption labels supervise detection/recovery only; they are never inference features.
The critic is training-only. Training targets are capped at 125; evaluation reports capped
and original official targets separately. Fitting-only per-regime support bounds
limit finite changes inside the clean fitting range to the intact-error budget;
outliers can return throughout the fitting range. The recovery-balance recipe adds
a separate original-linear missing-input loss relative to the mean-fill baseline.
It stratifies fitting engines over all eight fault contexts and three severities,
retaining clean examples and consistent overlapping windows. A top-10%
window/channel healthy-proposal term retains gradients even when the deployed
repair is clamped. Its log(1 + normalized MSE) form limits extreme-loss domination;
acceptance still measures ordinary RMSE. Every improving repair candidate must pass
all declared fault/severity conditions on held training-validation engines.
Intact identity loss uses original linear
sensor error normalized by its protection budget, even with asinh model inputs.

Partition and validation/evaluation corruption seeds are fixed across training seeds. Validation samples
three prefixes per engine; correction uses six. Interval calibration evaluates every
clean prefix of each reserved engine, uses one maximum absolute error per engine,
and takes the finite-sample conformal order statistic. Capped and raw targets have
separate contracts. Small calibration samples may produce honest unbounded intervals.
Coverage requires exchangeable whole trajectories and is not guaranteed under
fleet/censoring shift, corrupted data, or OOD conditions. Full runs train separate models for each subset. Seeds are selected using validation
performance, never test scores. Training explicitly requests `CONFIG['device'] = 'cuda'`.
Protocol `structured-sensor-faults-v4` selects a related measurement domain, protects
operating settings and demanded speeds, and uses raw-unit amplitude scales estimated
only on fitting engines. Fault starts use absolute observed offsets up to
`CONFIG['fault_start_horizon']`; additive calibration drift ramps over
`CONFIG['drift_ramp_cycles']`. Defaults are 100 cycles for both controls.
Noise has temporal memory and a shared component within the selected domain;
observed regime changes raise its amplitude and the burst-dropout entry rate.
Stuck sensors hold physical readings across regime changes. Mixed scenarios combine
related mechanisms on the same channels and onset. Each mechanism has an independent
random stream. Temporary blocks recur with gaps up to `block_gap_cycles` (50).
These are structured sensor-error simulations; measured field-fault parameters are
unavailable. Reports separate active faults from unaffected windows.
The recovery-balance recipe selects clean pretraining by clean validation RMSE.
Later stages, correction and seeds use half clean RMSE plus half the mean active-fault
RMSE over all eight contexts at severity 0.2. The unchanged mixed reconstruction
guards and full recovery matrix remain separate acceptance requirements. Changing
the training clean fraction does not change fixed validation faults. These are
training changes awaiting full retraining evidence, not established accuracy gains.
Use a fresh directory for this recipe; do not resume an earlier recipe's checkpoints.
Use `nvidia-smi` and the GPU check in README.md before starting. An intentional CPU
execution check requires changing the device to `'cpu'`.
""")
code("""
QUICK_RUN = False
RUN_PAPER_STUDY = False
RUN_EXPLANATIONS = False
USE_RECOVERY_BALANCE_RECIPE = True
CONFIG = dict(DEFAULT_CONFIG)
CONFIG['device'] = 'cuda'
if USE_RECOVERY_BALANCE_RECIPE:
    CONFIG.update(json.loads((PROJECT_ROOT / 'configs' / 'v8_recovery_balance.json').read_text()))
if QUICK_RUN:
    CONFIG = quick_config(CONFIG)
if RUN_PAPER_STUDY:
    CONFIG['run_matched_benchmarks'] = True
    CONFIG['run_extended_evaluation'] = True

seed_all(CONFIG['seed'], CONFIG['cpu_threads'])
TRAINING_DEVICE = device_for(CONFIG)
print('Training device:', TRAINING_DEVICE)
if TRAINING_DEVICE.type == 'cuda':
    print('GPU:', torch.cuda.get_device_name(TRAINING_DEVICE))
display(pd.DataFrame({'setting': list(CONFIG), 'value': [str(v) for v in CONFIG.values()]}))
display(Markdown('**Execution check only: small data and reduced dimensions.**' if QUICK_RUN else '**Full training: four subsets, five seeds, grouped out-of-fold correction.**'))
""", ["configuration"])
md("## Architecture")
code("""
display(SVG(filename=str(PROJECT_ROOT / 'docs' / 'GRACE_RUL_architecture.svg')))
model = GraceRUL(CONFIG)
display(pd.DataFrame([{'module': name, 'parameters': sum(p.numel() for p in module.parameters())}
                      for name, module in [('detector', model.detector), ('generator', model.generator),
                                           ('predictor', model.predictor), ('corrector', model.corrector)]]))
print(model)
del model
""")
md("""
## Dataset integrity and engine partitions

Read actual file counts instead of trusting conflicting FD004 tables. Split complete
engines into an 80% training pool and 20% validation split. Reserve 10% of the
training pool for independent interval calibration: approximately 72% optimizer
fitting, 8% calibration, and 20% validation overall, with integer-engine rounding.
Regime centers and sensor statistics come only from fitting engines. Windows never
cross an engine boundary. FD002/FD004 fit six operating-regime clusters using the
three settings and normalize sensors separately within each regime. FD001/FD003 use
one regime. Left padding is explicitly masked in all model stages.
""")
code("""
inventory = []
for subset in CONFIG['subsets']:
    train, test, truths = load_subset(PROJECT_ROOT / 'data', subset)
    partition = split_engines(train, CONFIG['split_seed'], CONFIG['maximum_engines'],
                              CONFIG['validation_fraction'], CONFIG['calibration_fraction_of_training'])
    inventory.append({'subset': subset, 'train_engines': train.unit.nunique(), 'test_engines': test.unit.nunique(),
                      'train_rows': len(train), 'test_rows': len(test),
                      'training_pool_engines': len(partition['fit']) + len(partition['calibration']),
                      'operating_conditions': 6 if subset in ['FD002', 'FD004'] else 1,
                      **{name + '_engines': len(units) for name, units in partition.items()}})
    assert len(truths) == test.unit.nunique()
display(pd.DataFrame(inventory))
""")
md("""
## Visualize introduced corruption before training

Persistent faults are applied to reference trajectories before making windows, so
overlapping windows share the same fault realization. Reference C-MAPSS measurements
already contain simulation noise and engine degradation. Added sensor errors include
bursty contact loss, channel outage, recurring telemetry blocks, correlated measurement
noise, calibration bias/drift, and stuck readings. Channel groups are documented
measurement-domain hypotheses, not verified C-MAPSS wiring. Severity controls a
mechanism's magnitude/duration and domain coverage, not a global percentage of bad
entries. The audit below shows actual changes. Plots use physical sensor units.
Unmodified examples remain in training. The damage mask is a target, not a model input.
""")
code("""
subset = CONFIG['subsets'][0]
train, test, truths = load_subset(PROJECT_ROOT / 'data', subset)
partition = split_engines(train, CONFIG['split_seed'], CONFIG['maximum_engines'],
                          CONFIG['validation_fraction'], CONFIG['calibration_fraction_of_training'])
fit = train[train.unit.isin(partition['fit'])]
scaler = OperatingRegimeScaler().fit(fit, subset, CONFIG['split_seed'],
    sensor_transform=CONFIG['sensor_transform'], sensor_transform_scale=CONFIG['sensor_transform_scale'])
# Start the illustrations immediately; training retains its configured onset range.
DEMO_CONFIG = {**CONFIG, 'fault_start_horizon': 0}
fig, axes = plt.subplots(4, 2, figsize=(14, 12))
fault_audit = []
for family, ax in zip(['intermittent', 'outage', 'block', 'noise', 'bias', 'drift', 'stuck', 'mixed'], axes.flat):
    demo_data = WindowDataset(fit, scaler, DEMO_CONFIG, mode='correction', family=family, seed=42)
    demo = demo_data[0]
    engine_index, end = demo_data.samples[0]
    metadata = demo_data.fault_metadata[engine_index]
    affected = np.flatnonzero(demo['altered'].any(0).numpy())
    j = int(affected[0]) if len(affected) else SENSORS.index(metadata['channels'][0])
    valid = demo['valid'].numpy()
    regimes = demo_data.engines[engine_index]['regimes'][max(0, end-CONFIG['sequence_length']+1):end+1]
    for key, label, style in [('reference', 'Reference', '-'), ('x', 'Sensor fault', '--')]:
        physical = scaler.inverse_sensors(demo[key].numpy()[valid, 3:], regimes)
        ax.plot(physical[:, j], style, label=label)
    missing = ~np.isfinite(demo['x'].numpy()[valid, 3+j])
    if missing.any():
        ax.scatter(np.flatnonzero(missing), scaler.inverse_sensors(demo['reference'].numpy()[valid, 3:], regimes)[missing, j],
                   marker='x', color='red', s=15, label='Unavailable')
    ax.set_title(f'{family}: {metadata["domain"]}, {SENSOR_NAMES[j]}')
    ax.set_xlabel('Observed window position'); ax.set_ylabel('Physical sensor reading'); ax.legend(fontsize=8)
    fault_audit.append({'family': family, 'scenario': metadata['scenario'], 'domain': metadata['domain'],
                       'channels': ', '.join(metadata['channels']), 'onset': metadata['onset_observed_offset'],
                       'window_changed_fraction': float(demo['altered'].numpy()[valid].mean())})
plt.tight_layout(); plt.show()
display(pd.DataFrame(fault_audit))
""")
md("""
## Train and export the complete pipeline

`run_experiment` first searches loss weights and stride on fitting/validation engines
without opening official test files. It then performs predictor pretraining, independent finite-fault detector fitting, generator/WGAN-GP training,
joint recovery/RUL optimization, three-fold engine-disjoint correction fitting, validation
fault-threshold selection, and engine-disjoint, target-specific conformal calibration. Temporary OOF
models also use independent inner validation engines. The exported audit records these
partitions. No training or calibration stage fits on the official test engines.

Progress includes batch counts, validation, and each temporary OOF model's epochs.
WGAN-GP uses up to 16 randomly sampled examples for gradient penalty and the full
batch for Wasserstein loss. Repair must improve the clean/corrupted validation
objective within clean-RMSE, intact-repair, and reconstruction limits; otherwise
an independently validated predictor is retained with repair bypassed. Recovery
training weights warm-start joint optimization even when not accepted for export;
the deployable fallback is preserved separately. Joint training can retain
the selected repair checkpoint. Correction uses observable summaries and predictions
across folds, searches bounds of 5/10/15/30 cycles, and retains a zero-correction fallback.
Final seeds are selected after
correction using the configured held-validation objective. The enabled recovery-balance
recipe uses the equal-family fault macro; legacy defaults use clean/mixed RMSE. Completed upstream
epochs (including OOF models) are saved in the printed run directory's `checkpoints/`.
To resume after interrupting, restart the kernel and rerun setup/configuration with
the same settings, then pass that run directory as `output_dir` to `run_experiment`.
Completed upstream models load from checkpoints; correction and evaluation rerun.
Format 8 adds fitting-support bounds, upper-tail healthy-proposal training,
and recovery validation over every declared fault family/severity while preserving physical `structured-sensor-faults-v4`
mechanisms. All format-5/6/7 training checkpoints require a fresh directory.
Format-5/6/7 bundles remain inference-compatible;
older format-3/4 bundles are incompatible.

Three-fold metrics evaluate the detector/recovery/predictor on held-out fitting engines
before the final corrector learns from their out-of-fold targets. Fold-level records
and mean/std summaries are saved. They are explicitly upstream CV; the complete
corrected pipeline is evaluated separately on official test files.

It evaluates clean official test endpoints and records upstream cross-validation metrics.
Set `RUN_PAPER_STUDY = True` before training for both comparison and robustness evidence.
Set `CONFIG['run_extended_evaluation'] = True` for extra corrupted endpoint predictions,
unseen severities, and inference-intervention ablations. Set
`CONFIG['run_matched_benchmarks'] = True` to separately train
CNN–GRU, repair/CNN–LSTM, and Transformer/correction adaptations plus retrained
GRACE ablations. These optional experiments do not update the primary model.
Paper adaptations get their own equal-count validation search. Controlled GRACE
ablations retain the selected settings and remaining branch/reliability interfaces.
No-repair models still receive corrupted-input predictor training. Separate clean-only
references are available explicitly. Retrained local/global/equal-fusion/no-correction
variants are included in the full comparison list. Every fault report includes its
actual affected-endpoint/entry rates and active-fault RMSE, for capped and raw targets.
Identical test engines and corruptions support paired seed/engine
bootstrap comparisons. The Transformer comparator uses supervised residual correction,
not the original TD3. These are harmonized adaptations, not reproductions of published
paper scores. Quick notebook mode skips tuning/comparators; `scripts/train_research.py
--quick --device cpu` checks that additional workflow. See `docs/RESEARCH_TRAINING.md`.
""")
code("""
BUNDLE_PATH, TABLES = run_experiment(PROJECT_ROOT, CONFIG)
RUN_DIR = BUNDLE_PATH.parent
print('Saved:', BUNDLE_PATH)
display(TABLES['seed_summary'])
""", ["training"])
md("""
## Training curves and recorded results

The following figures and tables use this run's CSV records. Capped and uncapped
test targets remain separate. Quick results are execution checks. The calibration
contracts distinguish capped and raw targets. Simultaneous clean-prefix coverage
requires exchangeable engine trajectories; fault/shift coverage must not be assumed.
Official test subsets were inspected in earlier development runs, so subsequent
benchmark results are exploratory rather than a fresh untouched holdout.
""")
code("""
history = TABLES['training_history']
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for axis, stage in zip(axes, ['predictor', 'generator', 'joint']):
    for (subset, seed), group in history[history.stage == stage].groupby(['subset', 'seed']):
        axis.plot(group.epoch, group.validation_macro_RMSE, marker='o', label=f'{subset}/{seed}')
    axis.set_title(stage.capitalize()); axis.set_xlabel('Epoch'); axis.set_ylabel('Validation RUL RMSE (cycles)')
axes[-1].legend(fontsize=7); fig.tight_layout()
fig.savefig(RUN_DIR / 'training_curves.png', dpi=150); plt.show()
display(TABLES['test_metrics'])
display(TABLES['cross_validation_metrics'])
display(TABLES['cross_validation_summary'])
for name in ['robustness_summary', 'ablation_metrics', 'validation_search', 'test_seed_summary', 'benchmark_summary', 'benchmark_active_comparison', 'benchmark_training_audit', 'superiority_checks', 'background_audit']:
    if name in TABLES:
        display(Markdown(f'**{name}**'))
        display(TABLES[name])
""")
md("""
## Reload the exact website artifact and inspect recovery

The format-8 bundle contains weights, selected per-subset settings, repair/bound buffers, correction, sensor thresholds,
validated repair gates, detector fitting audits, separate capped/raw interval contracts,
fitting-only explanation backgrounds, source hashes, and
engine split audits. It can be loaded with `weights_only=True`. Format-5/6/7 bundles
retain their original preprocessing and inference behavior; new training requires a
fresh run directory. Incompatible older artifacts are rejected.
""")
code("""
bundle, loaded = load_bundle(BUNDLE_PATH, device='cpu')
subset = CONFIG['subsets'][0]
entry = loaded[subset]
train, test, truths = load_subset(PROJECT_ROOT / 'data', subset)
sample_ds = WindowDataset(test[test.unit == test.unit.min()], entry['scaler'], entry['model'].config,
                          mode='test', family='bias', severity=.2, seed=42, truths=truths)
sample = sample_ds[0]
with torch.no_grad():
    out = entry['model'](sample['x'][None], sample['valid'][None])
valid = sample['valid'].numpy()
fault_scores = out['fault_scores'][0, valid].numpy()
available = np.isfinite(sample['x'][valid, 3:].numpy())
ranked = np.where(available, fault_scores, 0).sum(0) / np.maximum(1, available.sum(0))
suspect = int(ranked.argmax())
sensor_table = pd.DataFrame({'sensor': SENSORS, 'name': SENSOR_NAMES, 'mean_fault_score': ranked,
                             'validation_threshold': entry['entry']['thresholds']})
display(sensor_table.sort_values('mean_fault_score', ascending=False))
plt.figure(figsize=(11, 3))
plt.plot(sample['reference'][valid, 3+suspect], label='Reference')
plt.plot(sample['x'][valid, 3+suspect], label='Observed', linestyle='--')
plt.plot(out['repaired'][0, valid, suspect], label='Recovered')
plt.title(f'Estimated sensor recovery: {SENSORS[suspect]} ({SENSOR_NAMES[suspect]})')
plt.legend(); plt.tight_layout(); plt.savefig(RUN_DIR / 'sensor_recovery.png', dpi=150); plt.show()
print('Final RUL:', float(out['prediction'][0]), 'cycles; correction:', float(out['delta'][0]))
print('Learned repair selected:', bool(entry['model'].repair_enabled), '| Correction bound:', float(entry['model'].correction_bound))
""")
md("""
## Final SHAP and LIME sensor-fault explanations

Use actual Kernel SHAP and LIME over coherent sensor/time blocks. Explain the selected
sensor's detector score and final RUL separately. Perturbations preserve missingness,
settings, and padding, and replace sensor blocks with unmodified fitting references.
Fitting backgrounds are stratified across engines, regimes, and life stages, then
matched by setting history. Short references are never cyclically repeated. LIME reports weighted
held-out perturbation fidelity. SHAP contributions and LIME coefficients have different
scales; agreement is not forced. High RUL importance alone does not establish a fault.

Set `RUN_EXPLANATIONS = True` in the configuration cell to generate these optional plots.
They are skipped by default and do not affect saved model weights.
This also calculates global sensor importance over distinct held-out engines, with
repeated explanation seeds and separate within-engine and between-engine variability.
Full mode defaults to 12 engines and three explanation seeds per subset; quick mode
uses two engines and one seed, so stability is unavailable. No forced top-ten SHAP
sparsity is applied. These calculations use the selected trained model, without fitting it.
The full run uses three temporal blocks per sensor; quick mode uses one to reduce cost.
Detailed website explanations are computed on demand after an engine is selected.
""")
code("""
if RUN_EXPLANATIONS:
    explanation = explain_window(entry['model'], sample['x'], sample['valid'],
                             entry['entry']['background'], entry['entry']['background_valid'],
                             target_sensor=suspect if bool(entry['model'].repair_enabled) else None, samples=CONFIG['explanation_samples'],
                             seed=CONFIG['seed'], time_blocks=1 if QUICK_RUN else 3)
    (RUN_DIR / f'explanation_{subset}.json').write_text(json.dumps(explanation, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, target in zip(axes, explanation['targets']):
        values = pd.DataFrame(target['sensors']).head(10).iloc[::-1]
        axis.barh(values.sensor, values.shap)
        axis.set_title(f\"SHAP: {target['target']}\")
        axis.set_xlabel(target['units'])
        print(target['target'], 'LIME held-out fidelity:', target['lime_holdout_fidelity'])
        display(pd.DataFrame(target['sensors']).head(10))
    fig.tight_layout(); fig.savefig(RUN_DIR / 'shap_sensor_evidence.png', dpi=150); plt.show()
    global_importance = run_explanation_study(PROJECT_ROOT, BUNDLE_PATH,
                                             device=str(TRAINING_DEVICE))
    display(global_importance.groupby('subset', sort=False).head(10))
else:
    print('Optional SHAP/LIME plots skipped. Set RUN_EXPLANATIONS = True to generate them.')
""")
md("""
## Connect the website

Full bundles in `artifacts/<run>/grace_rul_ALL.pt` are discovered automatically. The
website keeps a model placeholder until one exists. It never invents model metrics.
Quick bundles are excluded from automatic discovery; an explicit selection shows a
quick-validation label.

From the project root, build the frontend with `cd website && npm run build`, then run
`.venv/bin/python -m uvicorn website.backend.app:app --host 127.0.0.1 --port 8000`.
Open `http://127.0.0.1:8000`. Use `GRACE_MODEL_PATH` to select a particular artifact.
""")
code("""
print('Model mode:', 'QUICK validation only' if QUICK_RUN else 'FULL training')
print('Artifact:', BUNDLE_PATH)
print('Explicit model selection:')
print(f'GRACE_MODEL_PATH=\"{BUNDLE_PATH}\" .venv/bin/python -m uvicorn website.backend.app:app --port 8000')
print('Automatic loading remains a placeholder until a full run exists.' if QUICK_RUN else 'Refresh the website model status to load this full run.')
""")

notebook = {"nbformat": 4, "nbformat_minor": 5, "cells": cells, "metadata": {
    "kernelspec": {"display_name": "Python (GRACE-RUL)", "language": "python", "name": "grace-rul"},
    "language_info": {"name": "python", "version": "3.12"}}}
try:
    import nbformat
except ImportError:
    pass  # The builder only needs the standard library; runtime validation uses nbformat.
else:
    nbformat.validate(notebook)
(ROOT / "GRACE_RUL_all_subsets.ipynb").write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(f"Created {len(cells)} cells.")

# Shared-model workflow. Keep the earlier per-FD notebook available for old runs.
cells = []
md("""
# GRACE-RUL: one shared model across FD001–FD004

This is the format-9 workflow in `docs/POOLED_V9.md`. It combines the four original
training files and trains one detector, selective BiLSTM recovery, gated CNN–GRU /
Transformer and bounded residual corrector. Operating regimes come from fitting
settings, and predictions require no FD label.

Split **whole engines** into **80% training / 10% validation pool / 10% internal
test**. The validation pool has independent selection and interval-calibration
halves. Original official test files remain an additional frozen benchmark.
The current retained format-8 model is preserved; this notebook does not overwrite
it. Full pooled accuracy must be measured after training. Quick metrics establish
execution only. No packages are installed by this notebook.
""")
code("""
from pathlib import Path
from datetime import datetime, timezone
import os, sys, json

start = Path(os.environ.get('GRACE_PROJECT_ROOT', Path.cwd())).resolve()
PROJECT_ROOT = next((p for p in [start, *start.parents] if (p / 'grace_rul').is_dir() and (p / 'data').is_dir()), None)
if PROJECT_ROOT is None:
    raise FileNotFoundError('Open alongside grace_rul/ and data/, or set GRACE_PROJECT_ROOT.')
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(PROJECT_ROOT / 'artifacts' / '.matplotlib'))
import pandas as pd
import torch
import matplotlib.pyplot as plt
from IPython.display import display
from grace_rul.pooled import pooled_config, prepare_pooled_run, run_pooled_experiment
from grace_rul.training import device_for
from grace_rul.artifacts import load_bundle

print('Project:', PROJECT_ROOT)
print('PyTorch:', torch.__version__, '| CUDA visible:', torch.cuda.is_available())
""")
md("""
## Configuration

Full mode uses the original dimensions and one training seed (42), with three
complete temporary OOF pipelines. Extra seeds require additional fitting.
Set `QUICK_RUN = True` for a reduced execution check across all four sources;
select `DEVICE = 'cpu'` explicitly for CPU. Use a new directory after changing
training settings; `GRACE_RUN_DIR` can name an unchanged run to resume.
""")
code("""
QUICK_RUN = False
DEVICE = 'cuda'
RUN_EXPLANATIONS = False
CONFIG = pooled_config(quick=QUICK_RUN)
CONFIG['device'] = DEVICE
if QUICK_RUN:
    CONFIG['pooled_evaluate_official'] = False
TRAINING_DEVICE = device_for(CONFIG)
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
RUN_DIR = Path(os.environ.get('GRACE_RUN_DIR', str(PROJECT_ROOT / 'artifacts' / f"pooled_{'quick_' if QUICK_RUN else ''}{stamp}")))
print('Mode:', 'QUICK EXECUTION CHECK' if QUICK_RUN else 'FULL POOLED TRAINING')
print('Device:', TRAINING_DEVICE, '| Run directory:', RUN_DIR)
display(pd.DataFrame({'setting': CONFIG.keys(), 'value': [str(x) for x in CONFIG.values()]}))
""", ["configuration"])
md("""
## Engine partitions before fitting

Engine IDs are namespaced to avoid collisions across source files. FD/source tags
are used for split balance and reporting only; the model tensor contains 24 measured
features. Fitting statistics, support and corruption scales exclude validation,
calibration and test engines. For the full 709-engine inventory the counts are
567 training, 36 selection, 35 calibration and 71 internal-test engines. The middle
two groups together form the requested 10% validation pool.
""")
code("""
COMBINED, PARTITIONS, PLAN = prepare_pooled_run(PROJECT_ROOT, CONFIG, RUN_DIR)
display(pd.DataFrame([{'role': k, 'engines': len(v)} for k, v in PARTITIONS.items()]))
inventory = pd.DataFrame(PLAN['engine_manifest'])
display(pd.crosstab(inventory.source_subset, inventory.partition))
groups = [set(PARTITIONS[k]) for k in ['fit', 'validation', 'calibration', 'test']]
assert sum(map(len, groups)) == len(set.union(*groups)) == inventory.unit.nunique()
print('Official test files have not been opened by this planning stage.')
""")
md("""
## Shared architecture and training

Observed settings → shared fitting-only operating normalization → finite-fault
detector → residual BiLSTM recovery with fitting-support bounds → local CNN–GRU
and global Transformer → quality gate → bounded OOF residual correction → RUL.

The WGAN-GP critic is used during training only. Missingness is directly observed;
the simulator's altered-entry labels supervise fitting and never enter inference.
Noise has temporal memory and a common component within related sensor groups.
Recovery candidates must pass the declared matrix and clean RMSE/NASA guards,
both pooled and within every source. These source tags are validation bookkeeping;
inference always uses the same shared model without an FD label.
Rejected recovery can retain a validated bypass. This is an experimental pooled
recipe; neither a configured guard nor passing tests proves improved accuracy.

After seed selection, separate calibration engines produce capped/raw conformal
contracts. Then the bundle is serialized and frozen before either test evaluation.
Internal testing uses fixed observed-prefix cycles 30, 60, 90, and so on, avoiding
the misleading all-zero-RUL final rows of complete trajectories. Official tests
use their original final observed endpoints. Reused C-MAPSS results remain exploratory.
""")
code("""
BUNDLE_PATH, TABLES = run_pooled_experiment(PROJECT_ROOT, CONFIG, RUN_DIR)
print('Saved:', BUNDLE_PATH)
""", ["training"])
md("""
## Measured results and learning curves

Pooled/per-source rows all use the same model. Keep `internal_test_10pct` and
`official_benchmark` scopes separate. Report every degradation or failed guard;
test values never choose weights or thresholds. Quick results are execution checks.
""")
code("""
results = pd.read_csv(RUN_DIR / 'test_metrics.csv')
display(results[['partition', 'subset', 'protocol', 'engines', 'N', 'RMSE', 'engine_balanced_RMSE', 'MAE', 'NASA_Score', 'Interval_Coverage']])
history = pd.read_csv(RUN_DIR / 'training_history.csv')
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for stage, group in history.groupby('stage', sort=False):
    axes[0].plot(group.epoch, group.validation_macro_RMSE, marker='.', label=stage)
axes[0].set(xlabel='Epoch within stage', ylabel='Validation RUL RMSE (cycles)')
axes[0].legend()
primary = results[(results.partition == 'internal_test_10pct') & (results.protocol == f"capped_{CONFIG['train_rul_cap']:g}")]
axes[1].bar(primary.subset, primary.RMSE)
axes[1].set(ylabel='Internal holdout RMSE (cycles)', title='Shared weights across every source')
plt.tight_layout(); plt.show()
""")
md("""
## Reload and optional explanations

The inference artifact has one `POOLED` entry. Uploads need engine IDs, consecutive
cycles, known operating settings and the named sensors; an FD label is optional.
SHAP/LIME use saved fitting-only backgrounds and explain model behavior, not a
proven physical diagnosis. Interval coverage under a new fleet or corruption is
not guaranteed by clean calibration. The full run must be audited before replacing
historical manuscript measurements.
""")
code("""
bundle, loaded = load_bundle(BUNDLE_PATH, device='cpu')
assert bundle['format_version'] == 9 and list(loaded) == ['POOLED']
assert bundle['requires_subset_label'] is False
print('One shared model; FD labels are not required at inference.')
print('Parameters:', sum(p.numel() for p in loaded['POOLED']['model'].parameters()))
if RUN_EXPLANATIONS:
    from grace_rul.explain import run_explanation_study
    display(run_explanation_study(PROJECT_ROOT, BUNDLE_PATH, device=str(TRAINING_DEVICE)))
else:
    print('Optional explanations skipped; set RUN_EXPLANATIONS = True to compute them.')
print(f'GRACE_MODEL_PATH="{BUNDLE_PATH}" minor/bin/python -m uvicorn website.backend.app:app --port 8000')
""")
notebook = {"nbformat": 4, "nbformat_minor": 5, "cells": cells, "metadata": {
    "kernelspec": {"display_name": "Python (GRACE-RUL)", "language": "python", "name": "grace-rul"},
    "language_info": {"name": "python", "version": "3.12"}}}
try:
    import nbformat
except ImportError:
    pass
else:
    nbformat.validate(notebook)
(ROOT / "GRACE_RUL_pooled.ipynb").write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(f"Created pooled notebook with {len(cells)} cells.")
