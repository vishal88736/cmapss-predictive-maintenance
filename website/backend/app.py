"""Run from the project root: uvicorn website.backend.app:app --port 8000."""

import io
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
import pandas as pd

from .inference import COLUMNS, MAX_FILE_BYTES, PROJECT_ROOT, SUBSETS, clean_json, parse_csv, preview
from .service import InputError, InferenceService, ModelUnavailable

app = FastAPI(title="GRACE-RUL · Sensor recovery workspace", version="3.0.0")
service = InferenceService()


@app.get("/api/status")
def status():
    return clean_json(service.status())


@app.get("/api/model")
def model():
    return clean_json(service.model_report())


async def read_upload(file, subset):
    content = await file.read(MAX_FILE_BYTES + 1)
    try:
        return parse_csv(content, subset)
    except InputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        await file.close()


@app.post("/api/preview")
async def preview_upload(file: UploadFile = File(...), subset: str = Form("FD001")):
    frame = await read_upload(file, subset)
    return clean_json(preview(frame, file.filename or "upload.csv"))


@app.post("/api/predict")
def predict_upload(file: UploadFile = File(...), subset: str = Form("FD001"), mc_dropout: bool = Form(False)):
    # A sync endpoint runs inference in FastAPI's thread pool rather than blocking its event loop.
    content = file.file.read(MAX_FILE_BYTES + 1)
    file.file.close()
    try:
        frame = parse_csv(content, subset)
        return service.predict(frame, file.filename or "upload.csv", mc_dropout)
    except InputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/explain")
def explain_upload(file: UploadFile = File(...), subset: str = Form("FD001"), unit: int = Form(...), sensor: int | None = Form(None)):
    content = file.file.read(MAX_FILE_BYTES + 1)
    file.file.close()
    try:
        frame = parse_csv(content, subset)
        return service.explain(frame, subset, unit, sensor)
    except InputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (ModelUnavailable, ImportError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/health")
def health():
    state = service.status()
    return {"status": "ok", "model_ready": state["ready"], "model_state": state["state"]}


@app.get("/api/sample")
def sample(subset: str = "FD001"):
    if subset not in SUBSETS:
        raise HTTPException(status_code=422, detail="Unknown C-MAPSS subset.")
    source = PROJECT_ROOT / "data" / f"test_{subset}.txt"
    if not source.exists():
        raise HTTPException(status_code=404, detail="The C-MAPSS sample data is not available.")
    data = pd.read_csv(source, sep=r"\s+", header=None).dropna(axis=1, how="all")
    data.columns = COLUMNS
    data = data[data["unit"] <= 5]
    buffer = io.StringIO()
    data.to_csv(buffer, index=False)
    return Response(buffer.getvalue(), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="sample_{subset}.csv"'})


DIST = Path(__file__).resolve().parents[1] / "dist"
if DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/")
    def homepage():
        return FileResponse(DIST / "index.html")
