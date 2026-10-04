"""Read recorded evaluation outputs with explicit provenance; never fill missing scores."""
import hashlib
import json
import re
from .project import TRAINING_ROOT


def recorded_evaluations():
    path = TRAINING_ROOT / "cmapss_ag.ipynb"
    if not path.is_file():
        return []
    content = path.read_bytes()
    notebook = json.loads(content)
    records = []
    for cell in notebook.get("cells", []):
        if cell.get("execution_count") is None:
            continue
        text = "".join("".join(output.get("text", [])) for output in cell.get("outputs", []) if output.get("name") == "stdout")
        if "Dataset" not in text or "Full System RMSE" not in text or "NASA Score" not in text:
            continue
        for match in re.finditer(r"^(FD00[1-4])\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*$", text, re.MULTILINE):
            subset, baseline, rmse, score = match.groups()
            records.append({
                "id": f"notebook-{subset}", "subset": subset,
                "modelName": "Notebook full system: DM-TCAE + CNN–Transformer + TD3 / LQR / Safety Gate",
                "metrics": {"RMSE": float(rmse), "CMAPSS_Score": float(score)},
                "baselineRMSE": float(baseline),
                "source": "Recorded notebook test evaluation",
                "sourceFile": "devesh/training/cmapss_ag.ipynb",
                "cellId": cell.get("id"), "executionCount": cell["execution_count"],
                "sourceSHA256": hashlib.sha256(content).hexdigest(),
                "scope": "Last window per test engine, as implemented in the notebook",
                "predictions": [],
                "note": "Read directly from saved notebook output. The trained checkpoint and per-engine evaluation arrays are not included with this record; this API has not re-evaluated that run.",
            })
    return records
