"""Import the modular project, never the top-level training/bootstrap script."""
import ast
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = Path(os.environ.get("CMAPSS_TRAINING_DIR", PROJECT_ROOT / "devesh" / "training")).resolve()
sys.path.insert(0, str(TRAINING_ROOT))

from src.preprocessing import RAW_COLUMNS, CANONICAL_14_SENSORS, CANONICAL_14_DESCRIPTIONS  # noqa: E402

SUBSETS = ("FD001", "FD002", "FD003", "FD004")


def source_configuration():
    """Read literal script settings without running training or its downloader."""
    tree = ast.parse((TRAINING_ROOT / "cmapss_dm_tcae_td3_lqr.py").read_text())
    config = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("WINDOW_LENGTH", "MAX_RUL", "SEED"):
                    config[target.id] = ast.literal_eval(node.value)
    return config


def model_metadata():
    config = source_configuration()
    return {
        "name": "DM-TCAE + CNN–Transformer + TD3 / LQR / Safety Gate",
        "dataset": "NASA C-MAPSS",
        "inputFormat": "Headerless, whitespace-separated .txt",
        "columns": RAW_COLUMNS,
        "features": [{"name": name, "description": description} for name, description in zip(CANONICAL_14_SENSORS, CANONICAL_14_DESCRIPTIONS)],
        "output": "Remaining Useful Life in operational cycles",
        "configuredWindowLength": config["WINDOW_LENGTH"],
        "configuredRULCap": config["MAX_RUL"],
        "preprocessing": ["Select the project's canonical sensor features", "Assign operating regimes using the fitted training K-Means", "Apply training-fitted, regime-specific Z-score normalization", "Construct engine-specific sliding windows; repeat the first reading for short histories"],
        "architecture": ["DM-TCAE reconstruction + latent representation", "Feature fusion and 1D convolution", "Transformer encoding and last-token pooling", "Regression head", "Trained TD3 residual correction", "LQR filter and validation-calibrated safety gate"],
        "sources": ["devesh/training/src/preprocessing.py", "devesh/training/src/dataset.py", "devesh/training/src/models/predictor.py", "devesh/training/src/evaluation.py"],
    }
