# NASA C-MAPSS RUL Prediction

Focused test-file upload UI and an adapter around the existing modular model in `../devesh/training/src`. The active interface has Prediction, Model, Metrics, and About pages, with Light, Dark, and System themes.

## Start the application

Use the Python environment already containing the project's PyTorch, NumPy, Pandas, and Scikit-learn dependencies. From `website/`:

```bash
python -m pip install -r backend/requirements.txt
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
npm install
npm run dev
```

The existing project environment can also be invoked explicitly:

```bash
../my_gnome_project/bin/python -m uvicorn backend.app:app --port 8000
```

Vite runs on `http://localhost:5183` and proxies `/api` to the local Python API. `CMAPSS_API_PROXY` overrides the development proxy target. For production, serve the frontend with a reverse proxy for `/api`, or configure `VITE_API_BASE_URL` for your deployment.

## Model-artifact placeholder

**No trained weights or fitted preprocessing artifacts are present in this checkout.** The upload, validation, test telemetry, model documentation, and recorded metrics work now. Inference explicitly reports unavailable until your real trained bundle is supplied.

Default directory:

```text
backend/artifacts/
  FD001/
    manifest.json
    predictor.pt
    td3_actor.pt
    td3_critic.pt
    preprocessor.joblib
    safety_gate.json
```

Other supported subsets use `FD002/`, `FD003/`, or `FD004/`. Set `CMAPSS_MODEL_DIR` to override this root. Only server-owned local artifacts are loaded; model files are not uploaded through the test-data interface.

Export objects from a **completed, actually trained** notebook session using `backend/export_artifacts.py`. Ensure the repository's `website` directory is on the notebook's Python path:

```python
from backend.export_artifacts import export_bundle

export_bundle(
    "/path/to/website/backend/artifacts",
    dataset_name,
    predictor, td3_agent, lqr_filter, safety_gate, dm,
    window_selection="last_window",
)
```

Use `last_window` for the current `cmapss_ag.ipynb` evaluation policy; use `all_windows` for the current `.py` training script's full-window inference policy. The adapter preserves this recorded difference. Export each subset's objects before the training loop replaces them with the next subset. No fitting, training, or recalibration is performed by the export helper or upload API.

## Input and inference contract

- Test data: the existing raw loader's **headerless whitespace-separated `.txt`**, with the columns declared in `src/preprocessing.py`: unit ID, cycle, operational settings, and sensors.
- Units must be grouped in ascending ID order, with consecutive cycles starting at one. Headers, CSV, missing measurements, duplicate cycles, and incompatible columns produce explicit validation errors; the API does not silently fix the data.
- Optional RUL file: one non-negative numeric value per engine, using the existing C-MAPSS convention of row one → engine one. It is used for evaluation only, not as a prediction input.
- The existing fitted `RegimeAwarePreprocessor.transform`, `_build_windows`, `SupervisedPredictor`, TD3 agent, LQR filter, calibrated gate, and `run_full_inference` are reused.
- One final model prediction per uploaded engine is returned. Short histories use the existing window builder's first-reading padding policy.
- When ground truth is absent, the label-shaped argument required by the existing inference function contains NaNs. It does not affect inference and is never displayed as actual RUL.
- Metric calculation for an upload reuses `src/metrics.calculate_metrics` and lives on the separate Metrics page.
- Uploaded files are held in server memory for a temporary session, with a one-hour expiry. Inference uses a temporary directory that is removed after execution. Files are not permanently saved.

### API routes

`GET /api/status`, `GET /api/model`, `GET /api/metrics`

`POST /api/uploads` — multipart `test_file`, `subset`, optional `rul_file`

`GET /api/uploads/{uploadId}/engines/{engineId}` — raw uploaded telemetry

`DELETE /api/uploads/{uploadId}` — discard uploaded data

`POST /api/predictions` — JSON `{ "uploadId": "..." }`

`GET /api/predictions/{jobId}` — actual stage states and final outputs

## Metric provenance

The checked-in notebook contains recorded final test RMSE and NASA scores. `/api/metrics` reads those numeric values directly from the notebook's saved output, attaching its cell ID, execution count, and SHA-256 fingerprint. It does **not** substitute them for a new run or imply that the corresponding checkpoint is deployed. MAE is not recorded there and is marked unavailable. No paired evaluation vectors are saved with those notebook records, so no actual-vs-predicted chart is fabricated.

## Checks

Install the optional test client dependency before running the backend tests:

```bash
python -m pip install -r backend/requirements-dev.txt
```

```bash
npm run build
npm run typecheck
../my_gnome_project/bin/python -m unittest backend.tests -v
```

The backend tests use actual local C-MAPSS source files and check validation, optional labels, lossless telemetry, missing-model behavior, upload discard, and notebook metric provenance. They do not manufacture trained checkpoints.
