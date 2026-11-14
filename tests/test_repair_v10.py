"""Format-10 repair invariants: age input, soft gate, corrector removal, calibration."""

from copy import deepcopy
import unittest

import numpy as np
import pandas as pd
import torch

from grace_rul.config import (COLUMNS, DEFAULT_CONFIG, POOLED_ARCHITECTURES, POOLED_FORMATS,
                              POOLED_FORMAT_VERSION, POOLED_REPAIR_FORMAT_VERSION,
                              SUPPORTED_ARCHITECTURES, quick_config)
from grace_rul.data import OperatingRegimeScaler, WindowDataset
from grace_rul.model import GraceRUL, expand_predictor_for_age
from grace_rul.repair import (apply_plateau_calibration, fit_plateau_calibration,
                              interval_band_report, long_rul_report, per_engine_tail)
from grace_rul.training import prefix_consistency_loss, validate_training_config


def model_config(**overrides):
    return {**quick_config({**DEFAULT_CONFIG, "device": "cpu"}),
            "recovery_safety": "none", "train_repair": True,
            "finite_repair_gate": "validated_threshold", "use_corrector": True,
            "sequence_length": 8, "local_length": 4, **overrides}


def sensor_batch(batch=3, length=8, seed=7):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, length, 24, generator=generator)
    x[..., :3] *= .1
    x[0, :2, 5] = float("nan")
    return x, torch.ones(batch, length, dtype=torch.bool)


class ConstantDetector(torch.nn.Module):
    def __init__(self, logit):
        super().__init__()
        self.logit = float(logit)

    def forward(self, observed, mask, settings, valid):
        return torch.full_like(observed, self.logit)


class RepairV10(unittest.TestCase):
    def test_format_ten_is_declared_and_format_nine_remains_loadable(self):
        self.assertEqual(POOLED_FORMATS, {9, 10})
        self.assertEqual((POOLED_FORMAT_VERSION, POOLED_REPAIR_FORMAT_VERSION), (9, 10))
        self.assertEqual(SUPPORTED_ARCHITECTURES[9], POOLED_ARCHITECTURES[9])
        self.assertEqual(SUPPORTED_ARCHITECTURES[10], POOLED_ARCHITECTURES[10])
        self.assertNotEqual(POOLED_ARCHITECTURES[9], POOLED_ARCHITECTURES[10])

    def test_age_expansion_is_exactly_prediction_preserving_before_training(self):
        torch.manual_seed(11)
        source_config = model_config(age_feature=False)
        source = GraceRUL(source_config).eval()
        expanded_config = {**source_config, "age_feature": True, "age_normalizer": 500.}
        expanded = GraceRUL(expanded_config).eval()
        expanded.load_state_dict(expand_predictor_for_age(source.state_dict(), expanded_config), strict=True)
        x, valid = sensor_batch()
        with torch.no_grad():
            reference = source(x, valid, use_repair=False, use_correction=False)
            for age in [torch.zeros(3), torch.full((3,), .8)]:
                candidate = expanded(x, valid, age=age, use_repair=False, use_correction=False)
                torch.testing.assert_close(candidate["base"], reference["base"], rtol=0, atol=1e-6)

    def test_age_channel_can_change_predictions_once_it_has_weights(self):
        torch.manual_seed(12)
        config = model_config(age_feature=True)
        model = GraceRUL(config).eval()
        with torch.no_grad():
            model.predictor.conv3.weight[:, -1].normal_(0., .5)
            model.predictor.conv5.weight[:, -1].normal_(0., .5)
        x, valid = sensor_batch()
        with torch.no_grad():
            young = model(x, valid, age=torch.zeros(3), use_repair=False, use_correction=False)["base"]
            old = model(x, valid, age=torch.ones(3), use_repair=False, use_correction=False)["base"]
        self.assertFalse(torch.allclose(young, old, rtol=0, atol=1e-6))

    def test_soft_gate_is_probability_weighted_and_respects_disabled_channels(self):
        config = model_config(finite_repair_gate="soft", soft_gate_power=2.,
                              soft_gate_temperature=1., use_corrector=False)
        model = GraceRUL(config).eval()
        model.detector = ConstantDetector(1.2)
        model.finite_repair_thresholds.fill_(.5)
        x, valid = sensor_batch()
        with torch.no_grad():
            result = model(x, valid)
        expected = torch.sigmoid(torch.tensor(1.2)).pow(2.).item()
        finite = torch.isfinite(x[..., 3:]) & valid.unsqueeze(-1)
        torch.testing.assert_close(result["weights"][finite],
                                   torch.full_like(result["weights"][finite], expected),
                                   rtol=1e-6, atol=1e-7)
        model.finite_repair_thresholds[0] = 1.
        with torch.no_grad():
            disabled = model(x, valid)
        self.assertTrue(torch.equal(disabled["weights"][..., 0][finite[:, :, 0]],
                                    torch.zeros_like(disabled["weights"][..., 0][finite[:, :, 0]])))
        self.assertGreater(float(disabled["weights"][..., 1][finite[:, :, 1]].mean()), 0.)

    def test_threshold_gate_remains_available_for_format_nine_state(self):
        config = model_config(finite_repair_gate="validated_threshold")
        model = GraceRUL(config).eval()
        model.detector = ConstantDetector(1.2)
        model.finite_repair_thresholds.fill_(.9)
        x, valid = sensor_batch()
        with torch.no_grad():
            result = model(x, valid)
        finite = torch.isfinite(x[..., 3:]) & valid.unsqueeze(-1)
        self.assertTrue(torch.equal(result["weights"][finite], torch.zeros_like(result["weights"][finite])))

    def test_disabling_the_corrector_zeroes_delta_even_with_nonzero_head(self):
        config = model_config(use_corrector=False)
        model = GraceRUL(config).eval()
        model.correction_bound.fill_(5.)
        with torch.no_grad():
            model.corrector[-1].weight.normal_(0., 1.)
        x, valid = sensor_batch()
        with torch.no_grad():
            disabled = model(x, valid, use_correction=True)
            enabled = model(x, valid)
        self.assertTrue(torch.equal(disabled["delta"], torch.zeros_like(disabled["delta"])))
        model.config["use_corrector"] = True
        with torch.no_grad():
            active = model(x, valid)
        self.assertGreater(float(active["delta"].abs().max()), 0.)
        self.assertFalse(torch.allclose(enabled["delta"], active["delta"]))

    def test_plateau_stretch_is_monotone_and_only_added_above_the_knee(self):
        config = model_config(plateau_calibration={"alpha": .5, "knee": 100.})
        model = GraceRUL(config).eval()
        model.correction_bound.fill_(0.)
        x, valid = sensor_batch()
        with torch.no_grad():
            result = model(x, valid)
        expected = result["base"] + .5 * torch.relu(result["base"] - 100.)
        torch.testing.assert_close(result["prediction"], expected)

    def test_plateau_fit_never_trades_capped_error_for_raw_error(self):
        good = pd.DataFrame({"prediction": [110., 110.], "raw_target": [130., 130.]})
        fitted = fit_plateau_calibration(good, knee=100., max_capped_increase=0.)
        self.assertGreater(fitted["alpha"], 0.)
        self.assertLess(fitted["raw_RMSE"], fitted["baseline_raw_RMSE"])
        self.assertLessEqual(fitted["capped_RMSE"], fitted["baseline_capped_RMSE"])

        bad = pd.DataFrame({"prediction": [125., 125.], "raw_target": [200., 50.]})
        refused = fit_plateau_calibration(bad, knee=100., max_capped_increase=0.)
        self.assertEqual(refused["alpha"], 0.)
        np.testing.assert_allclose(apply_plateau_calibration(bad.prediction, refused), bad.prediction)

    def test_active_prefix_loss_reads_the_deployed_input_path(self):
        config = model_config(prefix_consistency_loss_weight=1., prefix_consistency_path="clean",
                              finite_repair_gate="soft", soft_gate_power=3., age_feature=True)
        model = GraceRUL(config).eval()
        x, valid = sensor_batch()
        reference = x.clone()
        reference[..., 3:] = torch.nan_to_num(reference[..., 3:], nan=1.)
        previous_x = x.clone()
        previous_x[..., 3:] = torch.nan_to_num(previous_x[..., 3:], nan=1.) + 4.
        previous_reference = reference.clone()
        batch = {"x": x, "reference": reference, "valid": valid,
                 "target": torch.tensor([40., 80., 120.]), "cycle": torch.tensor([100, 200, 300]),
                 "previous_x": previous_x, "previous_reference": previous_reference,
                 "previous_valid": valid, "previous_target": torch.tensor([45., 90., 125.]),
                 "previous_cycle": torch.tensor([90, 180, 280]),
                 "pair_valid": torch.tensor([True, True, True])}
        torch.manual_seed(3)
        clean = prefix_consistency_loss(model, batch)
        model.config["prefix_consistency_path"] = "active"
        torch.manual_seed(3)
        active = prefix_consistency_loss(model, batch)
        self.assertTrue(torch.isfinite(clean) and torch.isfinite(active))
        self.assertGreater(abs(float(clean.detach()) - float(active.detach())), 1e-6)

    def test_training_dataset_emits_the_earlier_damaged_prefix(self):
        frame = pd.DataFrame([[u, t, 0., 0., 100., *[10. + t + .01 * u + j for j in range(21)]]
                              for u in range(1, 5) for t in range(1, 40)], columns=COLUMNS)
        scaler = OperatingRegimeScaler().fit(frame, "FD001", 42, regime_selection="per_subset")
        config = model_config(prefix_consistency_loss_weight=1., sequence_length=8,
                              fault_start_horizon=0, clean_fraction=0.,
                              evaluation_prefix_policy="seeded")
        dataset = WindowDataset(frame, scaler, config, mode="train", family="stuck", severity=.5, seed=5)
        sample = dataset[len(dataset) - 1]
        self.assertIn("previous_x", sample)
        self.assertIn("previous_cycle", sample)
        self.assertFalse(torch.equal(sample["previous_x"], sample["previous_reference"]))

    def test_long_rul_and_interval_reports_use_raw_and_capped_targets(self):
        records = pd.DataFrame({
            "unit": [1, 1, 2, 2], "raw_target": [200., 150., 60., 20.],
            "prediction": [110., 105., 65., 30.],
            "interval_lower_capped": [80., 80., 40., 10.],
            "interval_upper_capped": [125., 125., 90., 50.]})
        long = long_rul_report(records)
        self.assertEqual(long["long_N"], 2)
        self.assertGreater(long["long_share_squared_error"], .5)
        bands = {row["band"]: row for row in interval_band_report(records)}
        self.assertEqual(bands["plateau"]["N"], 2)
        self.assertEqual(bands["critical_eol"]["N"], 1)
        tail = per_engine_tail(records)
        self.assertEqual(tail["engines"], 2)
        self.assertAlmostEqual(tail["max_abs_error"], 90.)

    def test_invalid_soft_gate_settings_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Soft gate"):
            validate_training_config(model_config(finite_repair_gate="soft", soft_gate_power=0.))


if __name__ == "__main__":
    unittest.main()
