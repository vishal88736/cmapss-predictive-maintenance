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

This notebook implements the finalized architecture in `docs/FINAL_ARCHITECTURE.md`:
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
clean-reference RUL supervision and permits at most 1% clean RMSE regression by default.
The curriculum reaches full fault severity within ten epochs, before default patience expires.
Corruption labels supervise detection/recovery only; they are never inference features.
The critic is training-only. Training targets are capped at 125; evaluation reports capped
and original official targets separately. Empirical intervals are not conformal guarantees.

Partition and validation/evaluation corruption seeds are fixed across training seeds. Validation samples
three prefixes per engine; correction uses six; calibration retains one independent
endpoint per engine. Full runs train separate models for each subset. Seeds are selected using validation
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
Selection averages clean and active-mixed-fault RMSE; changing the training clean
fraction does not change the fixed validation faults. Checkpoints from earlier
simulators require a fresh run directory.
Use `nvidia-smi` and the GPU check in README.md before starting. An intentional CPU
execution check requires changing the device to `'cpu'`.
""")
code("""
QUICK_RUN = False
RUN_PAPER_STUDY = False
RUN_EXPLANATIONS = False
CONFIG = dict(DEFAULT_CONFIG)
CONFIG['device'] = 'cuda'
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
scaler = OperatingRegimeScaler().fit(fit, subset, CONFIG['split_seed'])
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
    mu = np.asarray(scaler.state['sensor_mean'])[regimes, j]
    sd = np.asarray(scaler.state['sensor_std'])[regimes, j]
    for key, label, style in [('reference', 'Reference', '-'), ('x', 'Sensor fault', '--')]:
        ax.plot(demo[key].numpy()[valid, 3+j] * sd + mu, style, label=label)
    missing = ~np.isfinite(demo['x'].numpy()[valid, 3+j])
    if missing.any():
        ax.scatter(np.flatnonzero(missing), (demo['reference'].numpy()[valid, 3+j] * sd + mu)[missing],
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
without opening official test files. It then performs predictor pretraining, detector/generator and WGAN-GP training,
joint recovery/RUL optimization, three-fold engine-disjoint correction fitting, validation
fault-threshold selection, and separate empirical interval calibration. Temporary OOF
models also use independent inner validation engines. The exported audit records these
partitions. No training or calibration stage fits on the official test engines.

Progress includes batch counts, validation, and each temporary OOF model's epochs.
WGAN-GP uses up to 16 randomly sampled examples for gradient penalty and the full
batch for Wasserstein loss. Repair must improve the clean/corrupted validation
objective within clean-RMSE, intact-repair, and reconstruction limits; otherwise
the pretrained predictor is retained with repair bypassed. Joint training can retain
the selected repair checkpoint. Correction uses observable summaries and predictions
across folds, searches bounds of 5/10/15/30 cycles, and retains a zero-correction fallback.
Final seeds are selected after
correction using clean/mixed validation RMSE. Completed upstream
epochs (including OOF models) are saved in the printed run directory's `checkpoints/`.
To resume after interrupting, restart the kernel and rerun setup/configuration with
the same settings, then pass that run directory as `output_dir` to `run_experiment`.
Completed upstream models load from checkpoints; correction and evaluation rerun.
The corrected `structured-sensor-faults-v4` simulator has a new training identity;
pre-correction checkpoints require a fresh directory even if their bundle format is 5.
Older format-3/4 runs are incompatible with the residual-recovery format-5 pipeline and
require fresh training. The interrupted October 8 artifacts have been removed.

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
interval scope is clean, capped-target endpoints; fault/shift coverage is measured
and must not be assumed.
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

The format-5 bundle contains weights, selected per-subset settings, repair/bound buffers, correction, sensor thresholds,
empirical interval offsets, fitting-only explanation backgrounds, source hashes, and
engine split audits. It can be loaded with `weights_only=True`. Legacy artifacts are
rejected because their architecture differs.
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
