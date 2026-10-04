# AeroLife / RUL

Independent C-MAPSS turbofan research visualization. Not an official NASA product.

## Run

```bash
npm install
npm run dev
```

The checked-in `public/data/FD00*.json` files are lossless numeric exports of the **training** text files in `../data`. All four C-MAPSS subsets can be explored without a Python server. To regenerate and verify against the local source files:

```bash
python3 scripts/export_cmapss.py
python3 scripts/verify_export.py
npm run build
```

If the source files are not in `../data`, the committed JSON exports still work in the browser. The export script requires those local originals when regenerating.

## Data integrity

- The explorer plots raw operational settings and sensor measurements. It does **not** interpolate, simulate, smooth, or predict readings.
- The uncapped run-to-failure reference is `last observed training cycle − selected cycle`. Training trajectories in C-MAPSS run to failure. The separate label visualization applies the training code's cap of 125 cycles. The raw reference is **not** a model output.
- The training code under `../devesh/training` implements regime-aware normalization, selected sensors, 40-cycle windows, a denoising convolutional autoencoder, CNN–Transformer predictor, and TD3/LQR correction stages. The website does not execute that Python code. No trained checkpoint or inference service is bundled with this app.
- Evaluation cards intentionally say "Not evaluated" until verified experiment output is connected. Saved notebook stdout is not imported as live results.

## Optional inference service contract

Set `VITE_API_BASE_URL` to the service base URL (default `/api`). The app checks `GET /api/status` and enables the prediction button **only** when its JSON response is `{ "modelConnected": true }`.

It then sends `POST /api/predict` with JSON:

```json
{ "subset": "FD001", "split": "train", "engineId": 1, "cycle": 1 }
```

The service must return a real inference output with the same `engineId` and `cycle`, and a finite numeric `predictedRUL`. A production service must use its actual model output, implement the same preprocessing and window construction used at training time, and expose actual model metadata/experiment records before those views are populated. No fake fallback prediction is generated on failure.

## Frontend

React, TypeScript, Tailwind CSS, local shadcn/ui-style primitives, Recharts, and Vite. `src/services/api.ts` isolates dataset loading and optional model inference from UI components.
