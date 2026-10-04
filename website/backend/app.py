"""Local upload/inference API. Run: python -m uvicorn backend.app:app --port 8000."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import asyncio
from datetime import datetime, timezone
import logging
import threading
import time
from uuid import uuid4
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .inference import ARTIFACT_ROOT, ModelUnavailable, artifact_status, predict_uploaded
from .metrics import recorded_evaluations
from .project import SUBSETS, model_metadata
from .validation import MAX_UPLOAD_BYTES, InputError, validate_test, validate_truth

@asynccontextmanager
async def lifespan(application):
    async def expire_sessions():
        while True:
            await asyncio.sleep(60)
            prune()
    expiry = asyncio.create_task(expire_sessions())
    try:
        yield
    finally:
        expiry.cancel()
        try:
            await expiry
        except asyncio.CancelledError:
            pass


app = FastAPI(title="C-MAPSS existing-model adapter", lifespan=lifespan)
uploads = {}
jobs = {}
storage_lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=1)
SESSION_TTL = 3600
STAGES = ("uploaded", "validation", "preprocessing", "features", "inference", "prediction")


def problem(status, message, code):
    return HTTPException(status_code=status, detail={"message": message, "code": code})


def prune():
    cutoff = time.time() - SESSION_TTL
    with storage_lock:
        for key in list(uploads):
            if uploads[key]["created"] < cutoff:
                del uploads[key]
        for key in list(jobs):
            if jobs[key]["created"] < cutoff and jobs[key]["status"] != "running":
                del jobs[key]


@app.get("/api/status")
def status():
    return {"backendConnected": True, "models": [artifact_status(subset) for subset in SUBSETS], "artifactDirectory": str(ARTIFACT_ROOT), "artifactConfigured": any(artifact_status(subset)["ready"] for subset in SUBSETS), "acceptedFormat": ".txt", "maxUploadBytes": MAX_UPLOAD_BYTES}


@app.get("/api/model")
def model():
    return model_metadata()


@app.get("/api/metrics")
def metrics():
    return {"records": recorded_evaluations(), "implementedMetrics": ["MAE", "RMSE", "CMAPSS_Score"], "source": "devesh/training/src/metrics.py"}


async def read_upload(file):
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    await file.close()
    if len(content) > MAX_UPLOAD_BYTES:
        raise problem(413, "Input file exceeds the upload limit of 20 MiB.", "file_too_large")
    return content


@app.post("/api/uploads")
async def upload(test_file: UploadFile = File(...), subset: str = Form(...), rul_file: UploadFile | None = File(None)):
    if subset not in SUBSETS:
        raise problem(422, "Select a supported C-MAPSS subset.", "invalid_subset")
    prune()
    try:
        data = validate_test(test_file.filename or "", await read_upload(test_file))
        truth_text, truth = None, None
        truth_filename = None
        if rul_file is not None:
            truth_filename = rul_file.filename
            truth_text, truth = validate_truth(truth_filename or "", await read_upload(rul_file), data.engine_ids)
    except InputError as error:
        raise problem(422, str(error), "invalid_input") from error
    token = str(uuid4())
    with storage_lock:
        uploads[token] = {"data": data, "truthText": truth_text, "truth": truth, "subset": subset, "created": time.time()}
    return {"uploadId": token, "subset": subset, **data.metadata(), "groundTruthFile": truth_filename, "groundTruthAvailable": truth is not None, "stages": {name: "completed" if name in ("uploaded", "validation") else "pending" for name in STAGES}}


@app.delete("/api/uploads/{token}")
def discard(token: str):
    with storage_lock:
        uploads.pop(token, None)
    return {"discarded": True}


def get_upload(token):
    prune()
    with storage_lock:
        item = uploads.get(token)
    if item is None:
        raise problem(410, "Upload is no longer available. Upload the test file again.", "upload_expired")
    return item


@app.get("/api/uploads/{token}/engines/{unit}")
def telemetry(token: str, unit: int):
    item = get_upload(token)
    data = item["data"]
    if unit not in data.engine_ids:
        raise problem(404, "Engine ID is not present in the uploaded file.", "engine_not_found")
    rows = data.rows[data.rows[:, 0] == unit]
    return {"engineId": unit, "readings": rows[:, 1:].tolist(), "sensorNames": data.metadata()["sensorColumns"], "conditionNames": ["setting_1", "setting_2", "setting_3"]}


def job_snapshot(job):
    return {key: dict(job[key]) if key == "stages" else job[key] for key in ("jobId", "status", "stages", "fileName", "engineCount", "timestamp", "error", "result")}


def run_job(job_id, item):
    active_stage = "inference"

    def progress(stage, state):
        nonlocal active_stage
        active_stage = stage
        with storage_lock:
            jobs[job_id]["stages"][stage] = state

    try:
        result = predict_uploaded(item["data"], item["subset"], item["truthText"], item["truth"], progress)
        with storage_lock:
            jobs[job_id]["result"] = result
            jobs[job_id]["status"] = "completed"
    except Exception as error:
        logging.exception("C-MAPSS inference failed")
        message = str(error) if isinstance(error, ModelUnavailable) else f"Prediction could not be completed. {error}"
        with storage_lock:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = message
            jobs[job_id]["stages"][active_stage] = "failed"


@app.post("/api/predictions")
def start_prediction(payload: dict):
    token = payload.get("uploadId")
    if not isinstance(token, str):
        raise problem(422, "A validated test upload is required.", "missing_upload")
    item = get_upload(token)
    if not artifact_status(item["subset"])["ready"]:
        raise problem(503, artifact_status(item["subset"])["reason"], "model_unavailable")
    job_id = str(uuid4())
    job = {"jobId": job_id, "status": "running", "created": time.time(), "stages": {name: "completed" if name in ("uploaded", "validation") else "pending" for name in STAGES}, "fileName": item["data"].filename, "engineCount": len(item["data"].engine_ids), "timestamp": datetime.now(timezone.utc).isoformat(), "error": None, "result": None}
    with storage_lock:
        jobs[job_id] = job
        snapshot = job_snapshot(job)
    executor.submit(run_job, job_id, item)
    return snapshot


@app.get("/api/predictions/{job_id}")
def prediction_job(job_id: str):
    prune()
    with storage_lock:
        job = jobs.get(job_id)
        if job is None:
            raise problem(410, "Prediction session is no longer available.", "job_expired")
        return job_snapshot(job)
