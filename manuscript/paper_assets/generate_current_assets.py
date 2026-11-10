"""Compatibility entrypoint for the versioned pooled paper generator."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cmapss-predictive-maintenance"))
from scripts.generate_pooled_paper_assets import main

if __name__ == "__main__":
    main()
