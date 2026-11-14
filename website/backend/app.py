"""FastAPI application for AeroGuard · GRACE-RUL Predictive Maintenance Platform.

Run from the project root:
    uvicorn website.backend.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
import pandas as pd
from pydantic import BaseModel

from grace_rul.config import COLUMNS, SENSORS, SUBSETS
from .inference import MAX_FILE_BYTES, PROJECT_ROOT, parse_csv
from .service import InferenceService, InputError, ModelUnavailable, clean_json

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"

app = FastAPI(
    title="AeroGuard · GRACE-RUL Predictive Maintenance Platform",
    description="Fault-tolerant turbofan prognostics, sensor recovery, and assurance diagnostics.",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = InferenceService()


# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------

@app.get("/api/status")
def get_status():
    return clean_json(service.status())


@app.get("/api/dashboard")
def get_dashboard(subset: str = Query("ALL")):
    return clean_json(service.get_dashboard(subset=subset))


@app.get("/api/engines")
def get_engines(
    subset: str = Query("ALL"),
    health: str = Query("ALL"),
    q: str = Query(""),
    page: int = Query(0, ge=0),
    size: int = Query(20, ge=1, le=100)
):
    return clean_json(service.get_engines(subset=subset, health=health, q=q, page=page, size=size))


@app.get("/api/engines/{asset:path}")
def get_engine_detail(asset: str):
    detail = service.get_engine_detail(asset)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Engine {asset} not found in evaluated fleet.")
    return clean_json(detail)


@app.post("/api/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    subset: str = Form("POOLED"),
    mc_dropout: bool = Form(False)
):
    """Upload test dataset (C-MAPSS text format or headered CSV) and execute batch inference."""
    content = await file.read(MAX_FILE_BYTES + 1)
    filename = file.filename or "uploaded_data.csv"
    await file.close()
    
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds the 25 MB upload limit.")
        
    try:
        frame = parse_csv(content, subset)
        results = service.evaluate_frame(frame, filename)
        return clean_json(results)
    except InputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc


class SampleRequest(BaseModel):
    subset: str = "FD001"


@app.post("/api/load-sample")
def load_sample_dataset(payload: SampleRequest):
    """1-Click quick load of official C-MAPSS test fleet."""
    subset = payload.subset.upper()
    if subset not in [*SUBSETS, "SAMPLE"]:
        raise HTTPException(status_code=422, detail=f"Unknown subset: {subset}")

    if subset == "SAMPLE":
        source = PROJECT_ROOT / "data" / "test_FD001.txt"
        limit = 5
        label = "Sample Quick Test (5 Engines)"
    else:
        source = PROJECT_ROOT / "data" / f"test_{subset}.txt"
        limit = None
        label = f"Official {subset} Test Fleet"

    if not source.is_file():
        raise HTTPException(status_code=404, detail=f"Dataset {source.name} not found.")

    try:
        content = source.read_bytes()
        frame = parse_csv(content, "POOLED")
        if limit:
            frame = frame[frame["unit"] <= limit].copy()
        results = service.evaluate_frame(frame, label)
        return clean_json(results)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load sample: {exc}") from exc


@app.get("/api/assurance/cases")
def get_assurance_cases():
    """Catalog of assurance cases available on the platform."""
    return {
        "cases": [
            {
                "id": "engine179",
                "title": "Failure Mode Diagnostic: FD004 Engine 179",
                "category": "Inspected Outlier Analysis",
                "description": "Deep inspection of the famous inspected underprediction case at cycle 240, demonstrating regime shift effects, age-continuity bounds, and honest safety boundary disclosure."
            },
            {
                "id": "fault_injection",
                "title": "Live Fault Recovery & Intact Protection Assurance",
                "category": "Fault-Tolerance Verification",
                "description": "Simulate structured sensor failures (stuck, bias, drift, noise, outage) on any engine and verify whether intact sensors remain protected within the certified 0.02364 sigma bound."
            },
            {
                "id": "conformal_calibration",
                "title": "Conformal Coverage & Long-Horizon Plateau Assurance",
                "category": "Uncertainty Quantification",
                "description": "Observable 4-bin predicted-RUL adaptive calibration and monotone plateau stretch (alpha=0.150 above knee 105) bounding raw long-horizon bias."
            }
        ]
    }


@app.get("/api/assurance/engine179")
def get_engine179_diagnostic():
    return clean_json(service.get_engine179_case())


class FaultSimulationRequest(BaseModel):
    unit: int = 1
    family: str = "stuck"
    severity: float = 0.2
    sensor: int = 7


@app.post("/api/assurance/simulate-fault")
def simulate_fault_assurance(payload: FaultSimulationRequest):
    """Run live fault simulation and verify healthy-sensor displacement assurance bound."""
    if payload.family not in ["stuck", "bias", "drift", "noise", "outage", "mixed"]:
        raise HTTPException(status_code=422, detail=f"Unsupported fault family: {payload.family}")
    try:
        res = service.simulate_fault_assurance(
            unit=payload.unit,
            family=payload.family,
            severity=payload.severity,
            sensor_idx=payload.sensor
        )
        return clean_json(res)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {exc}") from exc


@app.get("/api/models")
def get_models_registry():
    return clean_json(service.get_models_registry())


@app.get("/api/download-results")
def download_results():
    """Export evaluated fleet predictions as CSV."""
    fleet = service.cached_fleet
    if not fleet:
        raise HTTPException(status_code=404, detail="No evaluated fleet available for export.")
    
    rows = []
    for e in fleet:
        rows.append({
            "asset": e["asset"],
            "unit": e["unit"],
            "subset": e["subset"],
            "cycle": e["cycle"],
            "predicted_rul": e["rul"],
            "interval_lower": e["interval_lower"],
            "interval_upper": e["interval_upper"],
            "interval_width": e["interval_width"],
            "health_status": e["health"],
            "health_score_pct": e["health_score"],
            "anomaly_percentile": e["anomaly_percentile"],
            "flagged_faults": ";".join(e["detected_faults"]),
            "true_rul": e["true_rul"],
            "error": e["error"]
        })
    
    df = pd.DataFrame(rows)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return Response(
        buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="grace_rul_evaluated_results.csv"'}
    )


@app.get("/api/sample")
def download_sample_data(subset: str = "FD001"):
    if subset not in [*SUBSETS, "POOLED"]:
        raise HTTPException(status_code=422, detail="Unknown C-MAPSS subset.")
    source = PROJECT_ROOT / "data" / f"test_{'FD001' if subset == 'POOLED' else subset}.txt"
    if not source.exists():
        raise HTTPException(status_code=404, detail="C-MAPSS sample data is not available.")
    data = pd.read_csv(source, sep=r"\s+", header=None).dropna(axis=1, how="all")
    data.columns = COLUMNS
    data = data[data["unit"] <= 5]
    buf = io.StringIO()
    data.to_csv(buf, index=False)
    return Response(
        buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="sample_{subset}.csv"'}
    )


@app.get("/health")
def health_check():
    state = service.status()
    return {"status": "ok", "model_ready": state["ready"], "format_version": state["format_version"]}


# -----------------------------------------------------------------------------
# Frontend Static Files & SPA Fallback
# -----------------------------------------------------------------------------

if FRONTEND_DIR.is_dir():
    # Mount subdirectories if they exist
    if (FRONTEND_DIR / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")
    if (FRONTEND_DIR / "vendor").is_dir():
        app.mount("/vendor", StaticFiles(directory=FRONTEND_DIR / "vendor"), name="vendor")

    @app.get("/styles.css")
    def get_styles():
        return FileResponse(FRONTEND_DIR / "styles.css", media_type="text/css")

    @app.get("/charts.js")
    def get_charts():
        return FileResponse(FRONTEND_DIR / "charts.js", media_type="application/javascript")

    @app.get("/app.js")
    def get_app_js():
        return FileResponse(FRONTEND_DIR / "app.js", media_type="application/javascript")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa_fallback(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API route not found.")
        file_path = (FRONTEND_DIR / full_path).resolve()
        if file_path.is_file() and file_path.is_relative_to(FRONTEND_DIR.resolve()):
            return FileResponse(file_path)
        index_file = FRONTEND_DIR / "index.html"
        if index_file.is_file():
            return FileResponse(index_file)
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
