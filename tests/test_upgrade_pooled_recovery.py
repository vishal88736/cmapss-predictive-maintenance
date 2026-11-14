"""Unit tests for the pooled recovery upgrade workflow."""

from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from grace_rul.artifacts import load_bundle
from grace_rul.config import FEATURE_COLUMNS
from grace_rul.model import GraceRUL
from grace_rul.pooled import MODEL_KEY
from scripts.upgrade_pooled_recovery import PooledUpgradeRejected, upgrade_pooled_run


class TestUpgradePooledRecovery(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source_dir = self.root / "source_run"
        self.source_dir.mkdir(parents=True)
        (self.source_dir / "checkpoints").mkdir(parents=True)

        # Clone metadata from the real saved run to ensure realistic test bundle
        real_bundle_path = Path("artifacts/pooled_v9_r2_20261029_114845/grace_rul_ALL.pt")
        real_ckpt_path = Path("artifacts/pooled_v9_r2_20261029_114845/checkpoints/pipeline_POOLED_42.pt")
        if not real_bundle_path.is_file() or not real_ckpt_path.is_file():
            self.skipTest("Real pooled artifact not available.")

        bundle = torch.load(real_bundle_path, map_location="cpu", weights_only=True)
        ckpt = torch.load(real_ckpt_path, map_location="cpu")

        # Save to test directory
        torch.save(bundle, self.source_dir / "grace_rul_ALL.pt")
        torch.save(ckpt, self.source_dir / "checkpoints" / f"pipeline_{MODEL_KEY}_42.pt")

    def tearDown(self):
        self.temp.cleanup()

    def test_missing_warm_recovery_raises(self):
        bad_ckpt = self.source_dir / "checkpoints" / f"pipeline_{MODEL_KEY}_42.pt"
        torch.save({"identity": {}}, bad_ckpt)
        with self.assertRaises(ValueError):
            upgrade_pooled_run(Path("."), self.source_dir, device="cpu", skip_official=True)

    def test_missing_bundle_raises(self):
        nonexistent = self.root / "empty_dir"
        nonexistent.mkdir()
        with self.assertRaises(FileNotFoundError):
            upgrade_pooled_run(Path("."), nonexistent, device="cpu", skip_official=True)


if __name__ == "__main__":
    unittest.main()
