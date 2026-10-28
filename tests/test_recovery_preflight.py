"""Preflight scope, frozen-export support and complete recovery checks."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import torch

from grace_rul.config import COLUMNS, DEFAULT_CONFIG, quick_config
from grace_rul.data import OperatingRegimeScaler
from grace_rul.model import GraceRUL
from grace_rul.training import seed_all
from scripts.check_recovery_balance import check_balance, source_recovery_state


ROOT = Path(__file__).resolve().parents[1]


class RecoveryPreflight(unittest.TestCase):
    def setUp(self):
        seed_all(42)
        self.config = {**quick_config(DEFAULT_CONFIG), "device": "cpu", "fault_start_horizon": 0,
                       "recovery_safety": "none"}
        # Emulate a frozen upgrade whose saved config predates tail supervision.
        self.config.pop("identity_tail_loss_weight")
        self.config.pop("identity_tail_fraction")

    def fixture(self, root):
        (root / "data").mkdir()
        (root / "configs").mkdir()
        (root / "configs/v8_recovery_balance.json").write_bytes((ROOT / "configs/v8_recovery_balance.json").read_bytes())
        data = pd.DataFrame([[u, t, *[float(j + t / 10 + u / 20) for j in range(24)]]
            for u in range(1, 5) for t in range(1, 11)], columns=COLUMNS)
        train = root / "data/train_FD001.txt"
        data.to_csv(train, sep=" ", header=False, index=False)
        scaler = OperatingRegimeScaler().fit(data[data.unit <= 2], "FD001", 42,
            sensor_transform=self.config["sensor_transform"], sensor_transform_scale=self.config["sensor_transform_scale"])
        source = root / "artifacts/source"
        source.mkdir(parents=True)
        (source / "grace_rul_ALL.pt").write_bytes(b"Synthetic fixture; load_bundle is mocked, not trained-model evidence.")
        (source / "safety_upgrade_report.json").write_text(json.dumps({"status": "accepted", "subsets": [
            {"subset": "FD001", "seed": 42, "original_clean_references": {"RMSE": 0., "NASA_Score_Mean": 0.}}]}))
        entry = {"selected_seed": 42, "partitions": {"fit": [1, 2], "validation": [3], "calibration": [4]},
                 "source_hashes": {train.name: hashlib.sha256(train.read_bytes()).hexdigest()},
                 "thresholds": [.5] * 21, "pipeline_identity": {"seed": 42}}
        model = GraceRUL(self.config)
        return source, entry, model, scaler

    def test_frozen_export_without_checkpoints_preserves_original_clean_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, entry, model, _ = self.fixture(Path(tmp))
            model.repair_enabled.fill_(False)
            hashes = {}
            state, warm, history, reference, origin = source_recovery_state(source, "FD001", entry, model, hashes)
        self.assertFalse(bool(model.repair_enabled))
        self.assertTrue(bool(state["repair_enabled"]))
        self.assertFalse(warm)
        self.assertEqual(history, [])
        self.assertEqual(reference, {"RMSE": 0., "NASA_Score_Mean": 0.})
        self.assertTrue(origin.endswith("safety_upgrade_report.json"))
        self.assertIn(origin, hashes)

    def test_missing_clean_reference_is_not_replaced_with_a_new_tolerance(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, entry, model, _ = self.fixture(Path(tmp))
            (source / "safety_upgrade_report.json").unlink()
            _, _, _, reference, origin = source_recovery_state(source, "FD001", entry, model, {})
        self.assertIsNone(reference)
        self.assertIsNone(origin)

    def test_different_training_checkpoint_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, entry, model, _ = self.fixture(Path(tmp))
            (source / "checkpoints").mkdir()
            torch.save({"identity": {"seed": 999}}, source / "checkpoints/pipeline_FD001_42.pt")
            with self.assertRaisesRegex(ValueError, "training identity"):
                source_recovery_state(source, "FD001", entry, model, {})

    def test_real_fixture_audits_all_41_cases_without_test_files_or_export(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            source, entry, model, scaler = self.fixture(root)
            loaded = {"FD001": {"entry": entry, "model": model, "scaler": scaler}}
            with patch("scripts.check_recovery_balance.load_bundle", return_value=({}, loaded)):
                output = check_balance(source, root / "pilot", subsets=["FD001"], steps=1,
                    batch_size=2, device="cpu", full_matrix=True, retain_corrector=True, project_root=root)
            report = json.loads((output / "pilot_report.json").read_text())
            for variant in ["initial_active_repair", *report["variants"]]:
                audit = json.loads((output / f"FD001_{variant}_recovery_matrix.json").read_text())
                self.assertTrue(audit["complete"])
                self.assertEqual(audit["evaluated_contexts"], 41)
                self.assertFalse(audit["use_correction"])
                checked = report["subsets"]["FD001"]["validation_checks"][variant]
                self.assertTrue(checked["frozen_modules_unchanged"])
                self.assertFalse(checked["all_declared_validation_guards_passed"])
            self.assertEqual(report["subsets"]["FD001"]["training_policy"]["identity_tail_loss_weight"], .5)
            self.assertEqual(report["prohibited_reads_attempted"], [])
            self.assertFalse(report["official_tests_evaluated"])
            self.assertFalse(report["calibration_refitted"])
            self.assertFalse(report["deployable_model_exported"])
            self.assertEqual(len(list(output.rglob("*.pt"))), 0)

    def test_official_test_read_attempt_is_blocked(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            source, _, _, _ = self.fixture(root)
            def forbidden(*args, **kwargs):
                return (root / "data/test_FD001.txt").read_text()
            with patch("scripts.check_recovery_balance.load_bundle", side_effect=forbidden):
                with self.assertRaisesRegex(PermissionError, "cannot read official"):
                    check_balance(source, root / "pilot", subsets=["FD001"], steps=1, device="cpu", project_root=root)
            report = json.loads((root / "pilot/pilot_report.json").read_text())
            self.assertEqual(report["status"], "error")
            self.assertTrue(report["prohibited_reads_attempted"][0].endswith("data/test_FD001.txt"))


if __name__ == "__main__":
    unittest.main()
