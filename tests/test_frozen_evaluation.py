"""Frozen evaluation uses real fixture inference and never fits on test data."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from grace_rul.calibration import calibrate_endpoint_intervals
from grace_rul.config import COLUMNS, DEFAULT_CONFIG, FEATURE_COLUMNS, SUPPORTED_ARCHITECTURES, quick_config
from grace_rul.data import OperatingRegimeScaler, source_hashes
from grace_rul.model import GraceRUL
from grace_rul.training import evaluate, seed_all
from scripts import evaluate_research as frozen


class FrozenEvaluation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "data").mkdir()
        (self.root / "source").mkdir()
        seed_all(42)
        config = {**quick_config(DEFAULT_CONFIG), "device": "cpu", "subsets": ["FD001"],
            "sequence_length": 8, "local_length": 4, "recovery_safety": "none",
            "recovery_validation": False, "maximum_windows": 1, "maximum_engines": 2}
        train = pd.DataFrame([[u, t, *[float(t + j + u / 10) for j in range(24)]]
            for u in range(1, 7) for t in range(1, 6)], columns=COLUMNS)
        test = pd.DataFrame([[u, t, *[float(t + j + u / 10) for j in range(24)]]
            for u in [10, 20, 30] for t in range(1, 5)], columns=COLUMNS)
        for frame in [train, test]:
            frame[["unit", "cycle"]] = frame[["unit", "cycle"]].astype(int)
        for name, frame in [("train", train), ("test", test)]:
            frame.to_csv(self.root / f"data/{name}_FD001.txt", sep=" ", header=False, index=False)
        np.savetxt(self.root / "data/RUL_FD001.txt", [6., 12., 18.])
        scaler = OperatingRegimeScaler().fit(train[train.unit.isin([1, 2])], "FD001", 42,
            sensor_transform=config["sensor_transform"], sensor_transform_scale=config["sensor_transform_scale"])
        model = GraceRUL(config).eval()
        calibration, _ = calibrate_endpoint_intervals(model, train[train.unit.isin([5, 6])], scaler, config, 31417)
        entry = {"state_dict": deepcopy(model.state_dict()), "training_config": config, "scaler": scaler.state,
            "thresholds": [1.] * 21, "calibration": calibration, "selected_seed": 42,
            "correction_bound": float(model.correction_bound), "repair_selected": bool(model.repair_enabled),
            "source_hashes": source_hashes(self.root / "data", "FD001")}
        self.bundle = self.root / "source/grace_rul_ALL.pt"
        torch.save({"format_version": 7, "architecture": SUPPORTED_ARCHITECTURES[7],
            "corruption_protocol": "structured-sensor-faults-v4", "feature_columns": FEATURE_COLUMNS,
            "config": config, "models": {"FD001": entry}}, self.bundle)

    def run_evaluation(self, clean_only=True):
        return frozen.evaluate_bundle(self.bundle, self.root / "evaluation", project_root=self.root,
                                      device="cpu", clean_only=clean_only)

    def test_clean_metrics_are_recomputable_and_all_endpoints_survive_source_caps(self):
        original = self.bundle.read_bytes()
        output = self.root / "evaluation"
        real_hashes = frozen.source_hashes
        def hashes_after_plan(*args):
            self.assertTrue((output / "evaluation_plan.json").exists())
            return real_hashes(*args)
        with patch.object(frozen, "source_hashes", side_effect=hashes_after_plan), \
                patch("grace_rul.training.fit_pipeline", side_effect=AssertionError("No training")), \
                patch("grace_rul.calibration.calibrate_endpoint_intervals", side_effect=AssertionError("No recalibration")), \
                patch.object(OperatingRegimeScaler, "fit", side_effect=AssertionError("No preprocessing fit")):
            self.run_evaluation()
        self.assertEqual(self.bundle.read_bytes(), original)
        table = pd.read_csv(output / "test_metrics.csv")
        records = pd.read_csv(output / "predictions/FD001_clean_s0_r0.csv.gz")
        self.assertEqual(len(records), 3)
        self.assertEqual(set(records.unit), {10, 20, 30})
        self.assertEqual(len(table), 2)
        for _, row in table.iterrows():
            column = "raw_target" if row.protocol == "official_uncapped" else "target"
            expected = np.sqrt(np.mean((records.prediction - records[column]) ** 2))
            self.assertAlmostEqual(row.RMSE, expected)
            self.assertEqual(row.N, 3)
            self.assertTrue(row.Interval_Unbounded)
        self.assertTrue(np.isinf(records.raw_interval_upper).all())
        self.assertTrue((records.raw_interval_lower == 0).all())
        manifest = json.loads((output / "run_manifest.json").read_text())
        self.assertEqual(manifest["status"], "complete")
        self.assertTrue(manifest["source_quick_run"])
        self.assertTrue(manifest["weights_and_buffers_unchanged"])
        self.assertFalse(manifest["selection_performed"])
        self.assertFalse((output / "robustness_summary.csv").exists())

    def test_complete_declared_sweep_has_unique_cases_and_per_case_predictions(self):
        output = self.run_evaluation(clean_only=False)
        table = pd.read_csv(output / "robustness_summary.csv")
        self.assertEqual(len(table), 8 * 5 * 3 * 2)
        self.assertEqual(set(table.replicate), {0, 1, 2})
        self.assertFalse(table.duplicated(["family", "severity", "replicate", "protocol"]).any())
        self.assertEqual(len(list((output / "predictions").glob("*.csv.gz"))), 121)
        self.assertTrue((table.N == 3).all())

    def test_existing_or_source_output_is_rejected_before_data_access(self):
        output = self.root / "evaluation"
        output.mkdir()
        with patch.object(frozen, "source_hashes", side_effect=AssertionError("No data access")):
            with self.assertRaises(FileExistsError):
                self.run_evaluation()
            with self.assertRaisesRegex(ValueError, "outside"):
                frozen.evaluate_bundle(self.bundle, self.bundle.parent / "eval", project_root=self.root, device="cpu")

    def test_source_data_mismatch_is_rejected_without_inference(self):
        with (self.root / "data/RUL_FD001.txt").open("a") as stream:
            stream.write("24\n")
        with patch.object(frozen, "evaluate", side_effect=AssertionError("No inference")):
            with self.assertRaisesRegex(ValueError, "provenance"):
                self.run_evaluation()
        manifest = json.loads((self.root / "evaluation/run_manifest.json").read_text())
        self.assertEqual(manifest["status"], "error")

    def test_missing_label_provenance_is_rejected_without_inference(self):
        saved = torch.load(self.bundle, map_location="cpu", weights_only=True)
        del saved["models"]["FD001"]["source_hashes"]["RUL_FD001.txt"]
        torch.save(saved, self.bundle)
        with patch.object(frozen, "evaluate", side_effect=AssertionError("No inference")):
            with self.assertRaisesRegex(ValueError, "provenance"):
                self.run_evaluation()

    def test_changed_corrector_state_is_detected_as_incomplete(self):
        real_evaluate = frozen.evaluate
        def mutate(model, *args, **kwargs):
            result = real_evaluate(model, *args, **kwargs)
            model.corrector[0].bias.add_(1.)
            return result
        with patch.object(frozen, "evaluate", side_effect=mutate):
            with self.assertRaisesRegex(ValueError, "parameters/buffers changed"):
                self.run_evaluation()
        self.assertEqual(json.loads((self.root / "evaluation/run_manifest.json").read_text())["status"], "error")


class TailMetrics(unittest.TestCase):
    def test_window_channel_and_aggregate_rmse_are_distinct(self):
        class Dataset:
            fault_profile = {"sensor_transform": "linear", "sensor_transform_scale": 1.}
            def __len__(self):
                return 2
            def __getitem__(self, index):
                x = torch.zeros(2, 24)
                x[:, 0] = 2. * index
                return {"x": x, "reference": x.clone(), "valid": torch.ones(2, dtype=torch.bool),
                    "altered": torch.zeros(2, 21, dtype=torch.bool), "target": torch.tensor(1.),
                    "raw_target": torch.tensor(1.), "unit": index + 1, "cycle": 2}
        class Model(torch.nn.Module):
            config = {}
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
            def forward(self, x, valid):
                observed = x[..., 3:]
                repaired = observed.clone()
                repaired[..., 0] += x[..., 0]
                return {"prediction": torch.ones(len(x)), "repaired": repaired,
                    "observed": observed, "mask": torch.ones_like(observed), "fault_scores": torch.zeros_like(observed)}
        config = {**quick_config(DEFAULT_CONFIG), "device": "cpu", "batch_size": 2}
        with torch.no_grad():
            report, records = evaluate(Model(), Dataset(), config, capture_recovery_tails=True)
        self.assertAlmostEqual(report["linear_intact_repair_RMSE"], 2 / np.sqrt(42))
        self.assertAlmostEqual(report["linear_intact_window_RMSE_max"], 2 / np.sqrt(21))
        self.assertAlmostEqual(report["linear_intact_channel_RMSE_max"], np.sqrt(2))
        self.assertAlmostEqual(records.linear_intact_repair_RMSE.max(), 2 / np.sqrt(21))
        class MissingHealthySupport(Dataset):
            def __getitem__(self, index):
                item = super().__getitem__(index)
                if index == 0:
                    item["altered"].fill_(True)
                return item
        with torch.no_grad():
            report, records = evaluate(Model(), MissingHealthySupport(), config, capture_recovery_tails=True)
        self.assertEqual(report["linear_intact_window_N"], 1)
        self.assertTrue(pd.isna(records.iloc[0].linear_intact_repair_RMSE))
        self.assertEqual(records.iloc[0].intact_sensor_entries, 0)
        self.assertAlmostEqual(report["linear_intact_window_RMSE_p95"], 2 / np.sqrt(21))


if __name__ == "__main__":
    unittest.main()
