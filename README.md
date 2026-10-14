# GRACE-RUL

Corrupt sensor data during training, detect damaged readings, recover degradation
signals, and predict remaining useful life. The notebook and website share one
PyTorch implementation in `grace_rul/`.

- [Final architecture and paper analysis](docs/FINAL_ARCHITECTURE.md)
- [Training audit, fixes, and cleanup](docs/TRAINING_AUDIT.md)
- [Current paper integration audit](docs/INTEGRATION_AUDIT_2026-10-14.md)
- [Improved training and comparison workflow](docs/RESEARCH_TRAINING.md)
- [Training notebook](GRACE_RUL_all_subsets.ipynb)
- [Python requirements](requirements.txt)
- [Website instructions](website/README.md)

## Environment and notebook

```bash
bash scripts/setup_env.sh --cuda
source .venv/bin/activate
python -m jupyterlab GRACE_RUL_all_subsets.ipynb
```

Choose **Python (GRACE-RUL)**. Setup defaults to CUDA, replaces a CPU-only PyTorch
installation with the tested CUDA 13.0 build, and verifies actual GPU tensor
execution. It preserves an existing CUDA build. Use `--cpu` only for an intentional
CPU environment. The implementation uses PyTorch, pandas, scikit-learn, SHAP,
LIME, and FastAPI. Installation staging stays under `.venv/` to avoid filling a
memory-backed `/tmp`. Restart the notebook kernel after changing PyTorch.
To choose a different official wheel, set `GRACE_CUDA_WHEEL`
and `GRACE_TORCH_VERSION` to a supported combination.

The default `QUICK_RUN = False` trains the finalized dimensions on FD001–FD004
with five seeds and three engine-disjoint correction folds, preceded by validation-only
tuning. Separate comparison models, extended robustness/ablation reports, and notebook
SHAP/LIME plots are skipped by default. It explicitly requests
CUDA and stops if no CUDA GPU is accessible. Full training has a substantial compute
cost. Set `QUICK_RUN = True` only for a small FD001 execution check; select
`CONFIG['device'] = 'cpu'` explicitly if that check should use CPU.
These defaults retain the full training procedure and reduce the scheduled upstream
fits from 460 to 100: 20 tuning fits and 80 primary/fold fits. Actual runtime savings
depend on the model and early stopping. Full training uses stride-1 windows with balanced engine sampling; validation search
also tries stride 5. Official test predictions use the last observed endpoint per engine.

The same workflow is available without a notebook:

```bash
.venv/bin/python scripts/train_research.py --device cuda --output-dir artifacts/research_v5
```

For a small CPU check including tuning and trained comparisons:

```bash
.venv/bin/python scripts/train_research.py --quick --device cpu --output-dir /tmp/grace-research-check
```

The default command still includes many temporary fold models. To also run the
separate comparisons and extended evaluation reports, use:

```bash
.venv/bin/python scripts/train_research.py --device cuda --benchmarks --extended-evaluation --output-dir artifacts/research_with_reports
```

In the notebook, set `CONFIG['run_matched_benchmarks'] = True` and
`CONFIG['run_extended_evaluation'] = True` before training to enable these reports.
Set `RUN_EXPLANATIONS = True` in its configuration cell for optional SHAP/LIME plots.
Website explanations remain available on demand using the saved fitting backgrounds.
See [research training](docs/RESEARCH_TRAINING.md) for outputs, scope, and CLI options.

Keep the notebook, `grace_rul/`, `data/`, and `docs/` together. All required C-MAPSS
files are already under `data/`. Outputs are written to timestamped `artifacts/`
directories; source data is not modified.

Training prints the active stage, epoch, batch progress, validation, and each
out-of-fold (OOF) correction fold. The OOF step trains three additional complete
upstream models per seed, so a pause after an epoch line can be a fold transition.
WGAN-GP starts after the five-epoch warmup and takes longer than predictor training.
Early stopping restores the best admissible validation weights. Learning-rate
reduction adapts to validation plateaus. Repair checkpoints must protect clean RUL
accuracy and intact readings while meeting reconstruction limits. The clean predictor
is retained with repair bypassed if repair cannot improve within those limits.
Joint fine-tuning can retain the selected repair-stage checkpoint. Recurrent WGAN-GP
uses a random sub-batch of up to 16 for gradient penalty while its Wasserstein
term uses the full batch; configure `gradient_penalty_batch_size` to change this.

Each completed upstream epoch is saved atomically under the printed run directory's
`checkpoints/`, including temporary OOF models, optimizers, schedulers, and random states.
After an interruption, restart the notebook kernel, rerun setup/configuration,
and run the training cell with the same configuration and run directory:

```python
BUNDLE_PATH, TABLES = run_experiment(
    PROJECT_ROOT, CONFIG,
    output_dir=PROJECT_ROOT / "artifacts" / "full_<your_run_timestamp>",
)
```

Training resumes from the last completed upstream epoch. Completed pipelines are
loaded; correction features, the small residual corrector, and evaluations are
recomputed. A changed training configuration, architecture, or engine data requires
a new run directory. Report switches can change without invalidating completed
upstream epochs; rerun with the same output directory to reuse them.
Format-3/4 checkpoints and bundles cannot be reused with the
improved format-5 pipeline; retrain in a fresh directory. The interrupted October
8 run and its notebook outputs were removed during the October 11 audit.

The corrected simulator uses protocol `persistent-causal-faults-v2`. Fault onset is
sampled at an absolute observed offset from 0 through `fault_start_horizon` (100
by default). Drift ramps over `drift_ramp_cycles` (100 by default), then holds its
sampled gain. Fault streams are independent, including mixed faults, so extending
an engine's unseen future cannot change its already corrupted observations.
The observation rows in C-MAPSS represent consecutive cycles.

This changes training data. Start a fresh run directory for checkpoints created
before this correction; the new training identity rejects the earlier protocol.
Bundle format 5 and model tensor shapes are unchanged, and earlier format-5 bundles
remain loadable for inference. Their results describe the earlier simulator.
New bundles and run manifests record the corruption protocol explicitly.

## What the notebook implements

1. Split complete engines into an 80% training pool and 20% validation. Reserve
   10% of the training pool for calibration, giving approximately 72% optimizer
   fitting / 8% calibration / 20% validation. Integer-engine rounding applies.
   Fit operating-regime normalization and sensor resolution floors only on fitting engines. FD002/FD004 use
   six setting-based regimes with separate sensor statistics; FD001/FD003 use one.
2. Introduce intermittent loss, outage, blocks, noise, bias, drift, and mixed faults
   on trajectories before building windows, retaining unmodified examples.
3. Train the detector, selective residual BiLSTM recovery, conditional WGAN-GP critic,
   and quality-gated CNN–GRU / Transformer predictor in stages, then jointly. Direct
   reconstruction and temporal-trend losses preserve degradation information;
   class-balanced detector supervision and identity loss protect intact readings.
4. Fit a bounded residual corrector using observable quality summaries and RUL
   predictions from engine-disjoint folds, retaining a zero-correction candidate;
   select correction bounds and sensor thresholds using validation data and calibrate empirical intervals
   using separate engines.
5. Record held-out upstream cross-validation metrics and their mean/std summaries,
   official clean/corrupted test results, capped and original target conventions,
   all seeds, and explicitly labeled inference-intervention ablations. Upstream CV
   covers detector/recovery/prediction before the final corrector is fitted; it
   does not claim nested cross-validation of the completed corrected pipeline.
6. Train matched architecture comparators and module ablations in full mode; record
   paired seed/engine bootstrap intervals, per-condition metrics, and acceptance checks.
   These are adaptations to a common protocol; Paper C uses supervised correction.
7. Reload the exported bundle and calculate actual Kernel SHAP and LIME for final
   RUL and a selected sensor's finite-fault score, with LIME held-out fidelity.

SHAP and LIME explain model behavior. The detector and directly observed missingness
provide fault evidence. Importance alone does not establish a sensor fault.
Empirical intervals have no distribution-free coverage guarantee, particularly
under new corruption or operating conditions.

## Website and model placeholder

```bash
cd website
npm install
npm run build
cd ..
.venv/bin/python -m uvicorn website.backend.app:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. A visible placeholder accepts upload previews while
training is pending. Full format-5 bundles are discovered automatically after
training. The model page displays only results recorded by the connected run.
Sensor reports show affected cycles, scores and thresholds, recovered readings,
and on-demand SHAP/LIME explanations. Quick artifacts are excluded from automatic
discovery. [Explicit selection](website/README.md#model-bundle) labels them as quick
validation models.

## GPU check

Run these from your normal terminal before starting full training:

```bash
nvidia-smi
.venv/bin/python scripts/check_cuda.py
.venv/bin/python scripts/check_cuda.py --training-check
```

The NVIDIA RTX 4050 Laptop GPU is accessible from the normal host.
The restricted workspace hides GPU devices, which caused the earlier unavailable-GPU
report. Run GPU commands in your normal terminal or an authorized unrestricted
execution. The setup installs CUDA-enabled PyTorch; `check_cuda.py` tests real
tensor forward/backward. Its optional training check runs one batch in each stage
using full model dimensions and WGAN-GP; this checks execution, not accuracy.
For a particular accessible GPU, use `CONFIG['device'] = 'cuda:0'`.
Actual full-dimension staged GPU training and all-subset quick notebook checks
have passed; see [validation records](docs/VALIDATION.md#cuda-verification).
Setup follows the [official PyTorch CUDA installation workflow](https://pytorch.org/get-started/locally/).

## Validation

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/validate_notebook.py
.venv/bin/python scripts/validate_notebook.py --explanations
```

The notebook validator overrides full-training settings only in memory to use a
quick CPU check. It executes every code cell directly, writes execution records
under `artifacts/validation/`, and includes a real small training run. This works
when a workspace prohibits the local sockets that Jupyter normally needs. The
checked-in notebook remains free of execution output. Frontend validation is
`npm --prefix website run build`.

Use `scripts/validate_notebook.py --all-subsets` to validate all four subsets with
small dimensions, or `--device cuda` to validate on an accessible GPU. These
checks do not modify the checked-in notebook's full-mode CUDA defaults.

The quick run validates execution, not research performance. The optional comparison
workflow trains matched comparators, and those experiments must complete before any superiority
claim. Published-paper superiority additionally requires faithful original-protocol
reproductions and disclosed comparable tuning budgets.
