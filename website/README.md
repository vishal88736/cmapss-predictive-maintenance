# AeroGuard · GRACE-RUL Predictive Maintenance Platform

A fault-tolerant turbofan prognostics, sensor recovery, and assurance diagnostics platform built on NASA C-MAPSS data and the frozen Format-10 GRACE-RUL model (`pooled_v10_paper_20261114_repaired/grace_rul_ALL.pt`).

Inspired by the AeroGuard predictive maintenance architecture, this platform features a pure Python/FastAPI backend and a native vanilla ES modules frontend (zero Node.js or npm build step required to run).

---

## 🚀 Quick Start

Run the server from the repository root:

```bash
./minor/bin/python -m uvicorn website.backend.app:app --host 127.0.0.1 --port 8000
```

Then open **<http://127.0.0.1:8000>** in your browser.

---

## 🎯 Key Features & Views

### 1. Fleet Dashboard (`/dashboard`)
- **Prognostic KPIs**: Total evaluated engines, health status breakdown (Healthy, Watch, Attention, Critical), fleet mean RUL, and flagged sensor faults.
- **Certified Performance Banner**: Discloses verified format-10 bundle metrics:
  - Official Capped Clean RMSE: **11.08 cycles**
  - Stuck-fault RMSE (@0.20): **12.11 cycles**
  - Mixed-fault RMSE (@0.20): **11.48 cycles**
  - Healthy Sensor Displacement Ceiling: **≤ 0.02364 σ**
- **Interactive Visualizations**:
  - Remaining Useful Life Distribution (Histogram)
  - Fleet Health Breakdown (Donut Chart)
  - Engine Watchlist Table with RUL meters and 90% conformal intervals.

### 2. Engine Explorer (`/engines` and `/engines/:asset`)
- Search by engine ID/asset, filter by subset (`FD001`–`FD004`) and health status.
- **Deep Engine Detail View**:
  - Predicted RUL with central 90% split conformal prediction interval `[p05, p95]`.
  - RUL Degradation Trajectory line chart with conformal confidence envelope.
  - **21-Sensor Robust Health Matrix**:
    - Raw observed vs soft-recovered sensor telemetry.
    - Fault detector score $p_j \in [0, 1]$.
    - Soft cubic gate weight $w_j = p_j^3$.
    - Sensor repair displacement $\Delta x_j = \hat{x}_j - x_j$.
    - Sensor status classification (`Clean`, `Suspect`, `Fault Flagged`).
    - Interactive 40-cycle historical telemetry sparklines.

### 3. Assurance & Diagnostic Cases (`/assurance`)
- **Case 1: FD004 Engine 179 Failure Diagnostic**:
  - Full 420-cycle lifetime inspector.
  - Deep dive into the famous cycle 240 underprediction (predicted 7.7 vs target 125 cycles).
  - Operating regime transitions across flight envelopes.
  - Honest safety disclosure: conformal bands expand to reflect epistemic uncertainty under censored outliers.
- **Case 2: Live Fault Injection & Sensor Recovery Assurance Bench**:
  - Live interactive fault simulation: select target engine, fault family (`stuck`, `bias`, `drift`, `noise`, `outage`), sensor channel (1–21), and severity $\alpha \in [0.05, 0.50]$.
  - Verifies healthy-sensor displacement $\delta_{\text{healthy}} = \frac{1}{|S_{\text{intact}}|}\sum_{k \in S_{\text{intact}}} |x_k^{\text{repaired}} - x_k^{\text{clean}}|$.
  - Displays formal **ASSURANCE PASS** badge when $\delta_{\text{healthy}} \le 0.02364 \sigma$.
  - Real-time telemetry chart: Clean vs Injected Fault vs Soft-Repaired sensor signal.
- **Case 3: Conformal Prediction & Plateau Assurance**:
  - Observable 4-bin predicted-RUL adaptive calibration.
  - Monotone plateau stretch ($\alpha = 0.150$, knee 105) bounding raw long-horizon bias.

### 4. Test Data Upload & Batch Evaluation (`/upload`)
- **File Upload**: Drag-and-drop or browse NASA C-MAPSS whitespace-delimited text files (e.g. `test_FD001.txt`) or headered CSV files up to 25 MB.
- **1-Click Preloaded Benchmark Fleets**:
  - `Sample Fleet (5 Engines)`
  - `Official FD001 (100 Engines)`
  - `Official FD002 (259 Engines)`
  - `Official FD003 (100 Engines)`
  - `Official FD004 (248 Engines)`
- **Batch Evaluation Summary**: Engines evaluated, inference latency, fleet mean RUL, capped RMSE (if ground truth available), and empirical coverage.
- **Export**: One-click **Download Results CSV** button (`/api/download-results`).

### 5. Model Registry & Provenance (`/models`)
- Frozen format-10 bundle details (`pooled_v10_paper_20261114_repaired/grace_rul_ALL.pt`).
- Cryptographic SHA-256 hashes for model weights (`9b2b86ef0236...`), preprocessors, and manuscript assets.
- Official benchmark comparison table across release iterations.

---

## 📡 REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health status and format version |
| `GET` | `/api/status` | Model bundle readiness and hardware device |
| `GET` | `/api/dashboard` | Fleet summary KPIs, health breakdown, and benchmark values |
| `GET` | `/api/engines` | Filterable, paginated fleet engine list |
| `GET` | `/api/engines/{asset}` | Detailed engine diagnostics and 21-sensor matrix |
| `POST` | `/api/upload` | Multipart file upload for batch prognostics |
| `POST` | `/api/load-sample` | 1-Click quick load of official NASA test sets |
| `GET` | `/api/assurance/engine179` | Full telemetry and diagnostic data for Engine 179 |
| `POST` | `/api/assurance/simulate-fault` | Live sensor fault injection and displacement verification |
| `GET` | `/api/models` | Model registry, parameters, benchmarks, and provenance |
| `GET` | `/api/download-results` | Export current batch prognostic results as CSV |
| `GET` | `/api/sample` | Download standard sample test CSV |
