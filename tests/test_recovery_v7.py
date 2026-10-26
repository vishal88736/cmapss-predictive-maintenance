"""Regression checks for the failures seen in the completed recovery run."""

from copy import deepcopy
import unittest

import numpy as np
import torch

from grace_rul.artifacts import validate_calibration_contract
from grace_rul.config import DEFAULT_CONFIG, quick_config
from grace_rul.detection import _select_thresholds, _summarize
from grace_rul.model import GraceRUL
from grace_rul.training import acceptable_checkpoint, intact_identity_loss, seed_all


class RecoveryV7(unittest.TestCase):
    def setUp(self):
        seed_all(42)
        self.config = quick_config({**DEFAULT_CONFIG, "device": "cpu"})

    def test_identity_penalty_matches_inverse_space_and_guard_budget(self):
        reference = torch.tensor([0., 10.], dtype=torch.float64)
        repaired = (reference + .01).requires_grad_()
        mask = torch.ones(2, dtype=torch.bool)
        loss = intact_identity_loss(repaired, reference, mask, self.config)
        expected = ((torch.sinh(repaired) - torch.sinh(reference)) / .15).square().mean()
        torch.testing.assert_close(loss, expected, rtol=1e-10, atol=1e-12)
        loss.backward()
        self.assertTrue(torch.isfinite(repaired.grad).all())
        self.assertGreater(float(repaired.grad[1]), 1e6 * float(repaired.grad[0]))

    def test_faulty_values_are_masked_before_inverse_transform(self):
        repaired = torch.tensor([.1, 1e5], requires_grad=True)
        reference = torch.zeros_like(repaired)
        loss = intact_identity_loss(repaired, reference, torch.tensor([True, False]), self.config)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(repaired.grad[1]), 0.)
        empty = intact_identity_loss(repaired, reference, torch.tensor([False, False]), self.config)
        self.assertEqual(float(empty.detach()), 0.)

    def test_legacy_encoded_identity_is_unchanged(self):
        config = {**self.config, "identity_loss_space": "encoded"}
        a, b, mask = torch.tensor([2., 7.]), torch.tensor([1., 3.]), torch.tensor([True, False])
        self.assertEqual(float(intact_identity_loss(a, b, mask, config)), 1.)

    def test_finite_repairs_wait_for_validation_but_missing_values_are_repaired(self):
        model = GraceRUL(self.config).eval()
        with torch.no_grad():
            model.generator.decoder[-1].bias.fill_(.2)
        x = torch.randn(2, 7, 24)
        x[0, -1, 3] = float("nan")
        out = model(x)
        finite = torch.isfinite(x[..., 3:])
        torch.testing.assert_close(out["repaired"][finite], x[..., 3:][finite], rtol=0, atol=0)
        self.assertAlmostEqual(float(out["repaired"][0, -1, 0].detach()), .2, places=6)
        with torch.no_grad():
            model.finite_repair_thresholds.fill_(.01)
        self.assertGreater(float((model(x)["repaired"][finite] - x[..., 3:][finite]).abs().sum().detach()), 0.)

    def test_recovery_gradients_cannot_retrain_detector_through_gate_or_generator(self):
        model = GraceRUL(self.config).eval()
        with torch.no_grad():
            model.finite_repair_thresholds.fill_(.01)
            model.generator.decoder[-1].weight.fill_(.01)
            model.generator.decoder[-1].bias.fill_(.1)
        out = model(torch.randn(2, 7, 24))
        (out["prediction"].mean() + out["repaired"].square().mean()).backward()
        self.assertTrue(all(p.grad is None or not bool(p.grad.count_nonzero()) for p in model.detector.parameters()))
        self.assertTrue(any(p.grad is not None and bool(p.grad.count_nonzero()) for p in model.generator.parameters()))
        model.zero_grad(set_to_none=True)
        model(torch.randn(2, 7, 24))["logits"].mean().backward()
        self.assertGreater(float(model.detector.head.weight.grad.abs().sum()), 0.)

    def test_disabled_threshold_remains_disabled_when_sigmoid_rounds_to_one(self):
        model = GraceRUL(self.config).eval()
        with torch.no_grad():
            model.detector.head.bias.fill_(100.)
            model.generator.decoder[-1].bias.fill_(1.)
        x = torch.randn(1, 7, 24)
        torch.testing.assert_close(model(x)["repaired"], x[..., 3:], rtol=0, atol=0)

    def test_saturated_positive_scores_select_an_active_representable_threshold(self):
        pair = (np.array([.1, .1, 1., 1.], np.float32), np.array([False, False, True, True]))
        contexts = {"stuck": [pair] * 21}
        thresholds = _select_thresholds(contexts, .05)
        self.assertTrue(all(0.1 < threshold < 1 for threshold in thresholds))
        self.assertEqual(_summarize(contexts, thresholds)["score"], 1.)
        disabled = _summarize(contexts, [1.] * 21)
        self.assertEqual(disabled["score"], 0.)
        self.assertEqual(disabled["max_context_false_positive_rate"], 0.)

    def test_nasa_guard_rejects_rmse_improvement_with_asymmetric_score_regression(self):
        damaged = {"reconstruction_RMSE": .1, "linear_intact_repair_RMSE": .01, "linear_reconstruction_ratio": .5}
        self.assertFalse(acceptable_checkpoint({"RMSE": 9., "NASA_Score_Mean": 2.1}, damaged, 10.,
            self.config, clean_nasa_reference=2.))
        self.assertTrue(acceptable_checkpoint({"RMSE": 9., "NASA_Score_Mean": 2.01}, damaged, 10.,
            self.config, clean_nasa_reference=2.))

    def test_legacy_model_state_keys_keep_original_contract(self):
        config = {**self.config}
        config.pop("finite_repair_gate")
        self.assertNotIn("finite_repair_thresholds", GraceRUL(config).state_dict())
        self.assertIn("finite_repair_thresholds", GraceRUL(self.config).state_dict())

    def test_artifact_contract_rejects_fabricated_finite_small_sample_radius(self):
        contract = {"n": 2, "engines": 2, "alpha": .1, "rank": 3,
                    "unbounded": True, "radius": None, "lower_offset": None, "upper_offset": None}
        calibration = {"method": "split_conformal_absolute", "primary_protocol": "capped_125",
                       **{key: contract[key] for key in ["n", "rank", "unbounded", "radius", "lower_offset", "upper_offset"]},
                       "protocols": {"capped_125": deepcopy(contract), "official_uncapped": deepcopy(contract)}}
        validate_calibration_contract(calibration)
        bad = deepcopy(calibration)
        bad["protocols"]["official_uncapped"].update(unbounded=False, radius=10., lower_offset=-10., upper_offset=10.)
        with self.assertRaisesRegex(ValueError, "finite conformal"):
            validate_calibration_contract(bad)


if __name__ == "__main__":
    unittest.main()
