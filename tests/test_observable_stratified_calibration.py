"""Unit tests for observable-stratified conformal calibration and physical slew regularization."""

import math
import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd
import torch

from grace_rul.calibration import (calibrate_endpoint_intervals, interval_bounds,
                                   DEFAULT_OBSERVABLE_BINS, POLICIES)
from grace_rul.training import prefix_consistency_loss, fit_corrector


class DummyScaler:
    def __init__(self, fit_units=(999,)):
        self.state = {"fit_units": list(fit_units), "setting_mean": [0., 0., 0.], "setting_std": [1., 1., 1.],
                      "centers": [[0., 0., 0.]], "sensor_mean": [[0.] * 21], "sensor_std": [[1.] * 21],
                      "sensor_transform": "asinh"}

    def transform(self, frame):
        return np.zeros((len(frame), 24), dtype=np.float32), np.zeros(len(frame), dtype=int)


def mock_records(dataset, pred_fn):
    rows = []
    for engine_index, end in dataset.samples:
        engine = dataset.engines[engine_index]
        raw = float(engine["targets"][end])
        rows.append({
            "unit": engine["unit"],
            "cycle": int(engine["cycles"][end]),
            "prediction": float(pred_fn(engine, end)),
            "raw_target": raw,
            "target": min(dataset.config["train_rul_cap"], raw),
            "altered_entries": 0,
        })
    return pd.DataFrame(rows)


class TestObservableStratifiedCalibration(unittest.TestCase):
    def test_policy_in_policies(self):
        self.assertIn("observable_stratified", POLICIES)

    def test_observable_stratified_calibration_and_bounds(self):
        # 10 engines of length 150 each
        frame = pd.DataFrame([
            {"unit": u, "cycle": c}
            for u in range(1, 11)
            for c in range(1, 151)
        ])
        config = {
            "interval_alpha": 0.1,
            "train_rul_cap": 125.0,
            "calibration_policy": "observable_stratified",
            "window_stride": 1,
            "maximum_windows": None,
            "maximum_batches": None,
        }

        # Predictor simulates accurate predictions near end of life (error ~ 5),
        # but larger error near plateau (error ~ 25)
        def mock_pred(engine, end):
            raw = float(engine["targets"][end])
            if raw < 35:
                return raw + 5.0  # error = 5
            elif raw < 75:
                return raw + 12.0  # error = 12
            else:
                return raw + 25.0  # error = 25

        def evaluator(model, dataset, cfg):
            return {}, mock_records(dataset, mock_pred)

        result, records = calibrate_endpoint_intervals(
            Mock(), frame, DummyScaler(), config, seed=42, evaluator=evaluator
        )

        self.assertEqual(result["policy"], "observable_stratified")
        primary = result["protocols"]["capped_125"]
        self.assertIn("stratified_bins", primary)
        bins = primary["stratified_bins"]
        self.assertEqual(len(bins), len(DEFAULT_OBSERVABLE_BINS))

        # Check that the critical_eol bin has smaller radius than plateau bin
        eol_bin = [b for b in bins if b["name"] == "critical_eol"][0]
        plateau_bin = [b for b in bins if b["name"] == "plateau"][0]
        self.assertLess(eol_bin["radius"], plateau_bin["radius"])

        # Test interval bounds application
        test_preds = np.array([10.0, 50.0, 100.0, 130.0])
        lo, hi = interval_bounds(test_preds, result, protocol="capped_125")
        widths = hi - lo

        # Near failure (pred=10), interval width must be significantly tighter than near plateau (pred=130)
        self.assertLess(widths[0], widths[3])
        self.assertTrue(np.all(lo >= 0.0))
        self.assertTrue(np.all(hi >= lo))


class TestPhysicalSlewRegularization(unittest.TestCase):
    def setUp(self):
        self.config = {
            "prefix_consistency_loss_weight": 1.0,
            "target_scale": 125.0,
            "physical_slew_loss_weight": 1.0,
            "physical_rebound_loss_weight": 1.0,
            "physical_slew_margin": 0.0,
            "predictor_family": "gated",
            "train_rul_cap": 125.0,
        }

    def test_physical_slew_penalizes_crash_and_rebound(self):
        class MockPredictor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, x, valid, use_repair=False, use_correction=False):
                pass

        class MockModel(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.predictor = MockPredictor()

        model = MockModel(self.config)

        # Case 1: Normal physical degradation
        # previous cycle = 100, current cycle = 120 (delta_t = 20)
        # previous pred = 80, current pred = 65 (drop = 15 <= 20)
        batch_normal = {
            "reference": torch.zeros((1, 70, 24)),
            "previous_reference": torch.zeros((1, 70, 24)),
            "valid": torch.ones((1, 70), dtype=torch.bool),
            "previous_valid": torch.ones((1, 70), dtype=torch.bool),
            "target": torch.tensor([65.0]),
            "previous_target": torch.tensor([80.0]),
            "pair_valid": torch.tensor([True]),
            "cycle": torch.tensor([120]),
            "previous_cycle": torch.tensor([100]),
        }

        # Mock model forward to return custom predictions
        with torch.no_grad():
            # If current_prediction is passed directly
            curr_pred = torch.tensor([65.0])
            # Mock forward for previous_reference
            model.forward = lambda x, v, **kw: {"base": torch.tensor([80.0])}
            loss_normal = prefix_consistency_loss(model, batch_normal, current_prediction=curr_pred)

        # Case 2: Non-physical crash
        # previous cycle = 100, current cycle = 120 (delta_t = 20)
        # previous pred = 110, current pred = 10 (drop = 100 >> 20)
        batch_crash = {
            "reference": torch.zeros((1, 70, 24)),
            "previous_reference": torch.zeros((1, 70, 24)),
            "valid": torch.ones((1, 70), dtype=torch.bool),
            "previous_valid": torch.ones((1, 70), dtype=torch.bool),
            "target": torch.tensor([90.0]),
            "previous_target": torch.tensor([110.0]),
            "pair_valid": torch.tensor([True]),
            "cycle": torch.tensor([120]),
            "previous_cycle": torch.tensor([100]),
        }
        with torch.no_grad():
            curr_pred_crash = torch.tensor([10.0])
            model.forward = lambda x, v, **kw: {"base": torch.tensor([110.0])}
            loss_crash = prefix_consistency_loss(model, batch_crash, current_prediction=curr_pred_crash)

        # Case 3: Non-physical rebound
        # previous cycle = 100, current cycle = 120 (delta_t = 20)
        # previous pred = 20, current pred = 120 (increase = 100)
        batch_rebound = {
            "reference": torch.zeros((1, 70, 24)),
            "previous_reference": torch.zeros((1, 70, 24)),
            "valid": torch.ones((1, 70), dtype=torch.bool),
            "previous_valid": torch.ones((1, 70), dtype=torch.bool),
            "target": torch.tensor([20.0]),
            "previous_target": torch.tensor([40.0]),
            "pair_valid": torch.tensor([True]),
            "cycle": torch.tensor([120]),
            "previous_cycle": torch.tensor([100]),
        }
        with torch.no_grad():
            curr_pred_rebound = torch.tensor([120.0])
            model.forward = lambda x, v, **kw: {"base": torch.tensor([20.0])}
            loss_rebound = prefix_consistency_loss(model, batch_rebound, current_prediction=curr_pred_rebound)

        # The loss under crash and rebound must be much greater than under normal degradation
        self.assertLess(float(loss_normal), float(loss_crash))
        self.assertLess(float(loss_normal), float(loss_rebound))


class TestCorrectorBypass(unittest.TestCase):
    def test_corrector_bypass_when_disabled(self):
        mock_model = Mock()
        mock_model.parameters.return_value = []
        cfg = {"train_corrector": False}
        corr, oof, cv = fit_corrector(mock_model, Mock(), Mock(), "POOLED", cfg, 42)
        self.assertEqual(corr, [])
        self.assertIsNone(oof)
        self.assertEqual(cv, [])


if __name__ == "__main__":
    unittest.main()
