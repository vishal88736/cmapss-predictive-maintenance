"""CSV validation, artifact discovery, and batched GRACE-RUL reports."""

import io
import math
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd

from grace_rul.config import COLUMNS, SETTINGS, SENSORS, SUBSETS
from .service import InputError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_ROWS = 200_000
MAX_ENGINES = 1_000


def parse_csv(content, subset=None):
    subset = (subset or "POOLED").strip().upper()
    if subset not in [*SUBSETS, "POOLED"]:
        raise InputError("Choose the shared model or a supported legacy FD subset.")
    if not content or len(content) > MAX_FILE_BYTES:
        raise InputError("Upload a non-empty CSV smaller than 25 MB.")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise InputError("Save the CSV using UTF-8 encoding.") from exc
    aliases = {"engine_id": "unit", "engine": "unit", "cycles": "cycle", "time": "cycle", "rul": "true_rul"}
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    comma_separated = "," in first
    tokens = next(csv.reader([first])) if comma_separated else re.split(r"\s+", first)
    headerless = len(tokens) >= 2 and all(_is_number(v) for v in tokens[:2])
    if not headerless:
        headers = [aliases.get(c.strip().lower(), c.strip().lower()) for c in tokens]
        if len(headers) != len(set(headers)):
            raise InputError("The CSV contains duplicate column names.")
    try:
        frame = pd.read_csv(io.StringIO(text), sep="," if comma_separated else r"\s+",
                            header=None if headerless else "infer", nrows=MAX_ROWS + 1)
        if headerless:
            if frame.shape[1] != len(COLUMNS):
                raise InputError("A headerless file must contain exactly 26 C-MAPSS columns.")
            if len(frame) and np.array_equal(frame.iloc[0].to_numpy(dtype=float), np.arange(len(COLUMNS), dtype=float)):
                frame = frame.iloc[1:].reset_index(drop=True)
            frame.columns = COLUMNS
    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as exc:
        raise InputError(f"Could not read this CSV: {exc}") from exc
    names = [aliases.get(str(c).strip().lower(), str(c).strip().lower()) for c in frame.columns]
    if len(names) != len(set(names)):
        raise InputError("The CSV contains duplicate column names.")
    frame.columns = names
    missing = [name for name in COLUMNS if name not in frame]
    if missing:
        raise InputError("Missing required columns: " + ", ".join(missing) + ". Download the sample CSV for the expected format.")
    if not 1 <= len(frame) <= MAX_ROWS:
        raise InputError("The CSV must contain between 1 and 200,000 observations.")
    frame["subset"] = subset if "subset" not in frame else frame["subset"].astype(str).str.strip().str.upper()
    if subset != "POOLED" and not frame["subset"].isin(SUBSETS).all():
        raise InputError("Every subset value must be FD001, FD002, FD003, or FD004.")
    if subset == "POOLED" and not frame["subset"].str.fullmatch(r"[A-Z0-9_.:-]{1,64}").all():
        raise InputError("Optional source labels must contain 1–64 letters, numbers, dots, dashes, colons or underscores.")
    for column in COLUMNS + (["true_rul"] if "true_rul" in frame else []):
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        except (ValueError, TypeError) as exc:
            raise InputError(f"Column {column} contains a non-numeric value. Use blank sensor cells for missing readings.") from exc
        values = frame[column].to_numpy(dtype=float)
        if np.isinf(values).any():
            raise InputError(f"Column {column} contains an infinite value.")
        if column in ["unit", "cycle"] + SETTINGS and not np.isfinite(values).all():
            raise InputError(f"Column {column} must be complete and finite.")
    for column in ["unit", "cycle"]:
        if (frame[column] < 1).any() or (frame[column] % 1 != 0).any():
            raise InputError(f"Column {column} must contain positive whole numbers.")
        frame[column] = frame[column].astype(int)
    if "true_rul" in frame and (frame["true_rul"].dropna() < 0).any():
        raise InputError("Ground-truth RUL must be non-negative.")
    if frame.duplicated(["subset", "unit", "cycle"]).any():
        raise InputError("Duplicate subset / unit / cycle rows found. Each engine observation must be unique.")
    if frame.groupby(["subset", "unit"]).ngroups > MAX_ENGINES:
        raise InputError("A report can contain up to 1,000 engines.")
    frame = frame.sort_values(["subset", "unit", "cycle"]).reset_index(drop=True)
    gaps = frame.groupby(["subset", "unit"])["cycle"].diff().dropna()
    if (gaps != 1).any():
        raise InputError("Each engine history must contain consecutive cycles. Include a row for every missing cycle, with blank sensor readings and complete operating settings.")
    return frame


def _is_number(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def preview(frame, filename):
    missing = int(frame[SENSORS].isna().sum().sum())
    groups = frame.groupby(["subset", "unit"], sort=True)
    return {
        "filename": filename, "rows": len(frame), "engines": groups.ngroups,
        "subsets": sorted(frame["subset"].unique().tolist()),
        "missing_readings": missing,
        "availability": 100 * (1 - missing / (len(frame) * len(SENSORS))),
        "has_targets": "true_rul" in frame and bool(groups.tail(1)["true_rul"].notna().any()),
        "columns": list(frame.columns),
        "sample": frame[COLUMNS + ["subset"]].head(4).to_dict("records"),
    }


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def regression_metrics(engines, protocol="official_uncapped"):
    """Keep uploaded raw truth separate from the model's capped training target."""
    if protocol not in {"capped", "official_uncapped"}:
        raise ValueError("Unknown uploaded-target metric protocol.")
    labeled = [engine for engine in engines if engine["true_rul"] is not None]
    if not labeled:
        return None
    truth = np.asarray([engine["true_rul"] for engine in labeled], dtype=np.float64)
    caps = np.asarray([engine.get("train_rul_cap", 125.) for engine in labeled], dtype=np.float64)
    if protocol == "capped":
        truth = np.minimum(truth, caps)
    predictions = np.asarray([engine["predicted_rul"] for engine in labeled], dtype=np.float64)
    error = predictions - truth
    variance = ((truth - truth.mean()) ** 2).sum()
    with np.errstate(over="ignore"):
        penalties = np.expm1(np.where(error >= 0, error / 10.0, -error / 13.0))
    intervals = [engine.get("intervals", {}).get(protocol) for engine in labeled]
    # Old callers supplied only one interval without an explicit target contract.
    # New reports never apply a capped-only legacy calibration to raw truth.
    intervals = [interval if "intervals" in engine else {
        "available": True, "interval_lower": engine["interval_lower"],
        "interval_upper": engine["interval_upper"], "interval_unbounded": engine.get("interval_unbounded", False)
    } for engine, interval in zip(labeled, intervals)]
    covered = []
    for target, interval in zip(truth, intervals):
        if not interval or not interval.get("available", False):
            continue
        upper = math.inf if interval.get("interval_unbounded", False) else interval["interval_upper"]
        covered.append(interval["interval_lower"] <= target <= upper)
    target_convention = (f"capped_{caps[0]:g}" if np.all(caps == caps[0]) else "capped_per_model") if protocol == "capped" else protocol
    return clean_json({
        "protocol": target_convention, "target_convention": target_convention,
        "training_horizons": sorted(set(caps.tolist())),
        "N": len(labeled), "RMSE": np.sqrt(np.mean(error ** 2)), "MAE": np.abs(error).mean(),
        "R2": 1 - (error ** 2).sum() / variance if variance > 0 else None,
        "NASA_Score": penalties.sum(), "NASA_Score_Mean": penalties.mean(),
        "Interval_Coverage": np.mean(covered) if covered else None, "Interval_N": len(covered),
        "Interval_Coverage_Scope": "matching target calibration only; nominal coverage is not guaranteed for uploaded data",
        "Mean_Error": error.mean(),
        "NASA_Score_Overflow": not bool(np.isfinite(penalties.sum())),
    })
