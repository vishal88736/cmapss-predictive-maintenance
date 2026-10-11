# GRACE-RUL

Corrupt sensor data during training, detect damaged readings, recover degradation
signals, and predict remaining useful life. The notebook and website share one
PyTorch implementation in `grace_rul/`.

- [Final architecture and paper analysis](docs/FINAL_ARCHITECTURE.md)
- [Training notebook](GRACE_RUL_all_subsets.ipynb)
- [Python requirements](requirements.txt)
- [Website instructions](website/README.md)

## Environment and notebook

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
python -m jupyterlab GRACE_RUL_all_subsets.ipynb
```

Choose **Python (GRACE-RUL)**. The existing local `.venv` has been configured and
its dependency consistency checked. The implementation uses PyTorch, pandas,
scikit-learn, SHAP, LIME, and FastAPI. Install a suitable PyTorch GPU build before
the remaining requirements when using a different machine.

The default `QUICK_RUN = False` trains the finalized dimensions on FD001–FD004
with five seeds and three engine-disjoint correction folds. It explicitly requests
CUDA and stops if no CUDA GPU is accessible. Full training has a substantial compute
cost. Set `QUICK_RUN = True` only for a small FD001 execution check; select
`CONFIG['device'] = 'cpu'` explicitly if that check should use CPU.
Training windows use a five-cycle stride to control
cost, while official test predictions use the last observed endpoint per engine.

Keep the notebook, `grace_rul/`, `data/`, and `docs/` together. All required C-MAPSS
files are already under `data/`. Outputs are written to timestamped `artifacts/`
directories; source data is not modified.

Training prints the active stage, epoch, batch progress, validation, and each
out-of-fold (OOF) correction fold. The OOF step trains three additional complete
upstream models per seed, so a pause after an epoch line can be a fold transition.
WGAN-GP starts after the five-epoch warmup and takes longer than predictor training.
Early stopping restores the best validation weights when a stage finishes before
its configured epoch limit.

Each completed upstream epoch is saved atomically under the printed run directory's
`checkpoints/`, including temporary OOF models, optimizers, and random states.
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
recomputed. A changed training configuration or engine data requires a new run
directory. Runs created before checkpoint support contain only partial CSV results
and cannot restore their previous model weights.

## What the notebook implements

1. Split complete engines into an 80% training pool and 20% validation. Reserve
   10% of the training pool for calibration, giving approximately 72% optimizer
   fitting / 8% calibration / 20% validation. Integer-engine rounding applies.
   Fit operating-regime normalization only on fitting engines. FD002/FD004 use
   six setting-based regimes with separate sensor statistics; FD001/FD003 use one.
2. Introduce intermittent loss, outage, blocks, noise, bias, drift, and mixed faults
   on trajectories before building windows, retaining unmodified examples.
3. Train the detector, selective BiLSTM recovery, conditional WGAN-GP critic, and
   quality-gated CNN–GRU / Transformer predictor in stages, then jointly.
4. Fit a bounded residual corrector using features from engine-disjoint folds;
   select sensor thresholds using validation data and calibrate empirical intervals
   using separate engines.
5. Record held-out upstream cross-validation metrics and their mean/std summaries,
   official clean/corrupted test results, capped and original target conventions,
   all seeds, and explicitly labeled inference-intervention ablations. Upstream CV
   covers detector/recovery/prediction before the final corrector is fitted; it
   does not claim nested cross-validation of the completed corrected pipeline.
6. Reload the exported bundle and calculate actual Kernel SHAP and LIME for final
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
training is pending. Full format-3 bundles are discovered automatically after
training. The model page displays only results recorded by the connected run.
Sensor reports show affected cycles, scores and thresholds, recovered readings,
and on-demand SHAP/LIME explanations. Quick artifacts are excluded from automatic
discovery. [Explicit selection](website/README.md#model-bundle) labels them as quick
validation models.

## GPU check

Run these from your normal terminal before starting full training:

```bash
nvidia-smi
.venv/bin/python -c 'import torch; print("CUDA:", torch.cuda.is_available()); print("GPUs:", torch.cuda.device_count())'
```

The machine identifies an NVIDIA RTX 4050, but the restricted workspace currently
reports zero accessible CUDA GPUs and `nvidia-smi` cannot communicate with the driver.
The installed PyTorch build includes CUDA 12.8. If these checks also fail in your
normal terminal, GPU access or the driver must be resolved before CUDA training.
For a particular accessible GPU, use `CONFIG['device'] = 'cuda:0'`.

## Validation

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/validate_notebook.py
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

The quick run validates execution, not research performance. Establishing better
performance than the three papers still requires complete training and matched,
retrained baselines; the provided intervention ablations are not substitutes for
those experiments.
