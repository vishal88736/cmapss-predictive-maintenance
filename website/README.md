# GRACE-RUL website

A local sensor-recovery and remaining-life workspace using the exact model
definitions trained by `GRACE_RUL_all_subsets.ipynb`.

## Run

From the project root:

```bash
bash scripts/setup_env.sh
npm --prefix website install
npm --prefix website run build
.venv/bin/python -m uvicorn website.backend.app:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. For frontend development, run
`npm --prefix website run dev` in a second terminal and open
<http://127.0.0.1:5173>; Vite proxies `/api` to port 8000.

## Model bundle

The visible model placeholder remains until full training finishes. Upload previews
and sample downloads work without weights; inference and explanations return a
clear model-unavailable response. The website never trains inside a web request.

Run the notebook with `QUICK_RUN = False`. Full format-5 bundles under
`artifacts/<run>/grace_rul_ALL.pt` are discovered automatically, with their
`run_manifest.json`. Refresh the model status to load the newest run.

To select a specific bundle, including a quick execution check:

```bash
GRACE_MODEL_PATH="/absolute/path/to/grace_rul_ALL.pt" \
  .venv/bin/python -m uvicorn website.backend.app:app --host 127.0.0.1 --port 8000
```

An explicitly selected quick artifact is labeled **Quick validation model**;
automatic discovery excludes it. Legacy artifacts have incompatible architecture
and are rejected. `models/model_manifest.json` documents the pending model slot
without pretending to contain trained weights.

Optional environment settings: `GRACE_ARTIFACTS_DIR` selects another artifact root,
`GRACE_DEVICE` selects `cpu` or `cuda`, and `GRACE_CPU_THREADS` defaults to `2`.

## Data and reports

Upload headered CSV or original whitespace-delimited C-MAPSS text. Required columns:
`unit`, `cycle`, `setting_1`–`setting_3`, and `sensor_1`–`sensor_21`. Blank sensor
cells indicate missing readings. Choose the subset, or include a `subset` column
for mixed-subset CSVs. Optional `true_rul` values at engine endpoints enable error
metrics against those exact uploaded targets. Settings must be complete.

The console shows final RUL, bounded correction, local/global gates, empirical
residual intervals, and optional MC-dropout spread. Sensor reports use the final
70 observed positions and show missing/suspected cycles, validation thresholds,
repair magnitude, and observed/recovered readings in original sensor units.

Select an engine and sensor, then **Run SHAP + LIME**. The service separately explains
final RUL and that sensor's estimated finite-fault score with coherent temporal
blocks and fitting-data reference backgrounds. LIME held-out fidelity is reported.
Entirely missing sensors have directly observable outages; select an available
channel for finite-fault explanations.

The model page uses actual connected-run CSVs. Reported test metrics keep the
125-cycle capped target convention together; original-target results remain in the
run directory. Empirical interval calibration uses separate unmodified capped-target
endpoints, so corrupted-data coverage requires measurement.

The updated training protocol uses an 80% training pool and 20% validation split
by complete engine, with 10% of that pool reserved for calibration. FD002/FD004
use six operating-regime normalizers fitted on fitting engines only. The notebook
defaults to full CUDA training. Per-fold upstream CV results and mean/std summaries
are available through `/api/model` and the run CSVs; official test tables evaluate
the complete corrected model.

Uploads are processed in memory. **Export JSON** includes predictions, sensor
evidence, and requested explanations. No uploaded file is persisted by the API.

## API

| Route | Purpose |
| --- | --- |
| `GET /health` | Service health and model state |
| `GET /api/status` | Placeholder, quick, or trained state |
| `GET /api/model` | Configuration and recorded experiment tables |
| `GET /api/sample?subset=FD001` | Five-engine sample CSV |
| `POST /api/preview` | Validate multipart `file`, `subset` |
| `POST /api/predict` | Predict; optional `mc_dropout` |
| `POST /api/explain` | Explain multipart `file`, `subset`, `unit`, optional `sensor` (1–21) |
