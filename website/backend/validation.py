"""Validate exactly the raw text layout expected by the project's preprocessor."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .project import RAW_COLUMNS, CANONICAL_14_SENSORS

MAX_UPLOAD_BYTES = 20 * 1024 * 1024


class InputError(ValueError):
    pass


@dataclass
class ValidatedData:
    filename: str
    text: str
    rows: np.ndarray
    engine_ids: list[int]

    def metadata(self):
        return {
            "fileName": self.filename,
            "format": "Whitespace-separated .txt",
            "rowCount": len(self.rows),
            "columnCount": self.rows.shape[1],
            "columns": list(RAW_COLUMNS),
            "engineIds": self.engine_ids,
            "sensorColumns": [name for name in RAW_COLUMNS if name.startswith("sensor_")],
            "modelFeatures": list(CANONICAL_14_SENSORS),
        }


def decode_text(filename: str, content: bytes) -> str:
    if Path(filename).suffix.lower() != ".txt":
        raise InputError("Unsupported input format. The existing pipeline expects a headerless, whitespace-separated .txt file.")
    if not content or not content.strip():
        raise InputError("Input file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise InputError("Input file exceeds the upload limit of 20 MiB.")
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise InputError("Unsupported input format. Upload a UTF-8 text file.") from error


def validate_test(filename: str, content: bytes) -> ValidatedData:
    text = decode_text(filename, content)
    rows = []
    for line_number, line in enumerate(text.splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != len(RAW_COLUMNS):
            missing = ", ".join(RAW_COLUMNS[len(fields):]) if len(fields) < len(RAW_COLUMNS) else "extra fields"
            raise InputError(f"Input file does not match the expected C-MAPSS format. Line {line_number} has {len(fields)} fields; expected {len(RAW_COLUMNS)}. Missing or unexpected: {missing}.")
        try:
            values = [float(field) for field in fields]
        except ValueError as error:
            raise InputError(f"Line {line_number} contains non-numeric values. Headers and comma-separated CSV files are not supported by the existing raw loader.") from error
        if not np.isfinite(values).all():
            raise InputError(f"Line {line_number} contains a missing or non-finite measurement.")
        if any(values[i] < 1 or values[i] != int(values[i]) for i in (0, 1)):
            raise InputError(f"Line {line_number}: engine IDs and cycles must be positive integers.")
        if any(values[i] > np.iinfo(np.int32).max for i in (0, 1)):
            raise InputError(f"Line {line_number}: engine ID or cycle exceeds the integer range supported by the project pipeline.")
        rows.append(values)
    if not rows:
        raise InputError("Input file contains no engine observations.")
    array = np.asarray(rows, dtype=np.float64)
    last_unit, last_cycle = None, None
    for unit, cycle in array[:, :2].astype(np.int64):
        if last_unit is not None and unit < last_unit:
            raise InputError("Engine units must be grouped in ascending ID order. The adapter does not silently reorder your input.")
        if unit != last_unit:
            if cycle != 1:
                raise InputError(f"Engine {unit} must start at cycle 1, as in the supported C-MAPSS test files.")
        elif cycle != last_cycle + 1:
            raise InputError(f"Engine {unit} has duplicate, missing, or out-of-order cycles near cycle {cycle}. Consecutive cycles are required.")
        last_unit, last_cycle = unit, cycle
    return ValidatedData(Path(filename).name, text, array, np.unique(array[:, 0]).astype(int).tolist())


def validate_truth(filename: str, content: bytes, engine_ids: list[int]) -> tuple[str, dict[int, float]]:
    text = decode_text(filename, content)
    values = []
    for line_number, line in enumerate(text.splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 1:
            raise InputError(f"Ground-truth RUL line {line_number} must contain one numeric RUL value.")
        try:
            value = float(fields[0])
        except ValueError as error:
            raise InputError(f"Ground-truth RUL line {line_number} is not numeric.") from error
        if not np.isfinite(value) or value < 0:
            raise InputError(f"Ground-truth RUL line {line_number} must be finite and non-negative.")
        values.append(value)
    if engine_ids != list(range(1, len(engine_ids) + 1)):
        raise InputError("A C-MAPSS RUL file maps row 1 to engine 1, row 2 to engine 2, and so on. This test upload must contain those consecutive engine IDs to attach ground truth.")
    if len(values) != len(engine_ids):
        raise InputError(f"Ground-truth RUL has {len(values)} entries, but the test file contains {len(engine_ids)} engines.")
    return text, dict(zip(engine_ids, values))
