"""Directed finite repair preserves large outlier recovery and budgets inliers."""

from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import torch

from grace_rul.normalization import decode_sensors, encode_sensors
from grace_rul.recovery_safety import bounded_finite_repair, fit_recovery_envelope


class DirectedRecoverySafety(unittest.TestCase):
    def setUp(self):
        self.config = {"sensor_transform": "asinh", "sensor_transform_scale": 1.,
                       "max_intact_repair_rmse": .15}
        self.valid = torch.ones((1, 2), dtype=torch.bool)
        self.finite = torch.ones((1, 2, 21), dtype=torch.bool)
        self.lower = torch.full((21,), float(np.arcsinh(-3.)))
        self.upper = torch.full((21,), float(np.arcsinh(3.)))

    def encoded(self, value, *, dtype=torch.float32):
        return torch.tensor(encode_sensors(np.full((1, 2, 21), value), self.config), dtype=dtype)

    def run_bound(self, observed, proposal, **kwargs):
        return bounded_finite_repair(observed, proposal, self.finite, self.valid,
                                     self.lower, self.upper, self.config, **kwargs)

    def test_inliers_obey_original_linear_budget_even_for_huge_proposals(self):
        for observed_z in [-2., 0., 2.]:
            for target in [-1e4, 1e4]:
                observed = self.encoded(observed_z)
                bounded, audit = self.run_bound(observed, self.encoded(target), return_diagnostics=True)
                delta = decode_sensors(bounded.numpy(), self.config) - decode_sensors(observed.numpy(), self.config)
                self.assertLessEqual(float(np.abs(delta).max()), .15 + 1e-12)
                self.assertTrue(bool(audit["inlier_mask"].all()))
                self.assertTrue(bool(audit["clipped_mask"].all()))

    def test_huge_outliers_can_be_repaired_to_zero_without_global_step_cap(self):
        for observed_z in [-1e5, 1e5]:
            observed, target = self.encoded(observed_z), self.encoded(0.)
            bounded, audit = self.run_bound(observed, target, return_diagnostics=True)
            torch.testing.assert_close(bounded, target, rtol=0, atol=0)
            self.assertTrue(bool(audit["outlier_mask"].all()))
            self.assertFalse(bool(audit["clipped_mask"].any()))

    def test_outliers_may_use_full_support_but_not_arbitrarily_overshoot(self):
        for observed_z, target, expected in [(100., -100., -3.15), (-100., 100., 3.15)]:
            bounded = self.run_bound(self.encoded(observed_z), self.encoded(target))
            z = decode_sensors(bounded.numpy(), self.config)
            np.testing.assert_allclose(z, expected, atol=1e-6)
        for observed_z, target in [(100., 1000.), (-100., -1000.)]:
            observed = self.encoded(observed_z)
            bounded = self.run_bound(observed, self.encoded(target))
            delta = decode_sensors(bounded.numpy(), self.config) - decode_sensors(observed.numpy(), self.config)
            self.assertLessEqual(float(np.abs(delta).max()), .15 + 1e-12)

    def test_missing_imputation_and_padding_are_untouched(self):
        observed, proposal = self.encoded(0.), self.encoded(100.)
        self.finite[0, 0, 0] = False
        observed[0, 0, 0] = float("nan")
        self.valid[0, 1] = False
        proposal[0, 1, 2] = float("nan")
        bounded, audit = self.run_bound(observed, proposal, return_diagnostics=True)
        self.assertEqual(float(bounded[0, 0, 0]), float(proposal[0, 0, 0]))
        torch.testing.assert_close(bounded[0, 1], proposal[0, 1], rtol=0, atol=0, equal_nan=True)
        self.assertFalse(bool(audit["finite_valid_mask"][0, 1].any()))
        self.assertFalse(bool(audit["finite_valid_mask"][0, 0, 0]))

    def test_missing_repairs_keep_a_finite_unit_gradient(self):
        observed, proposal = self.encoded(0.), self.encoded(100.)
        self.finite[0, 0, 0] = False
        observed[0, 0, 0] = float("nan")
        proposal.requires_grad_(True)
        bounded = self.run_bound(observed, proposal)
        bounded.sum().backward()
        self.assertTrue(bool(torch.isfinite(proposal.grad).all()))
        self.assertEqual(float(proposal.grad[0, 0, 0]), 1.)

    def test_identity_preserves_exact_bits_including_negative_zero(self):
        observed = self.encoded(0.)
        observed[0, 0, 0] = -0.
        self.config["max_intact_repair_rmse"] = 0.
        bounded = self.run_bound(observed, observed.clone())
        self.assertTrue(torch.equal(observed.view(torch.int32), bounded.view(torch.int32)))

    def test_actual_clamp_gradient_without_straight_through_estimator(self):
        observed = self.encoded(0.)
        proposal = self.encoded(.05)
        proposal[0, 0, 0] = float(np.arcsinh(10.))
        proposal.requires_grad_(True)
        bounded = self.run_bound(observed, proposal)
        bounded.sum().backward()
        self.assertTrue(bool(torch.isfinite(proposal.grad).all()))
        self.assertEqual(float(proposal.grad[0, 0, 0]), 0.)
        self.assertEqual(float(proposal.grad[0, 0, 1]), 1.)

    def test_identity_initialized_generator_keeps_its_learning_gradient(self):
        observed = self.encoded(1.)
        proposal = observed.clone().requires_grad_(True)
        self.run_bound(observed, proposal).sum().backward()
        torch.testing.assert_close(proposal.grad, torch.ones_like(proposal), rtol=0, atol=0)

    def test_per_timestep_profiles_select_their_own_inlier_support(self):
        observed = self.encoded(5.)
        lower = torch.stack([torch.full((21,), float(np.arcsinh(-3.))),
                             torch.full((21,), float(np.arcsinh(4.)))])[None]
        upper = torch.stack([torch.full((21,), float(np.arcsinh(3.))),
                             torch.full((21,), float(np.arcsinh(6.)))])[None]
        bounded = bounded_finite_repair(observed, self.encoded(0.), self.finite, self.valid,
                                        lower, upper, self.config)
        z = decode_sensors(bounded.numpy(), self.config)
        np.testing.assert_allclose(z[0, 0], 0., atol=0.)
        self.assertGreaterEqual(float(z[0, 1].min()), 4.84999)

    def test_legacy_linear_and_nondefault_asinh_scale(self):
        for method, scale in [("linear", 1.), ("asinh", .25), ("asinh", 3.)]:
            self.config.update(sensor_transform=method, sensor_transform_scale=scale)
            self.lower = torch.tensor(encode_sensors(np.full(21, -3.), self.config), dtype=torch.float32)
            self.upper = torch.tensor(encode_sensors(np.full(21, 3.), self.config), dtype=torch.float32)
            observed = self.encoded(1.)
            bounded = self.run_bound(observed, self.encoded(100.))
            delta = decode_sensors(bounded.numpy(), self.config) - decode_sensors(observed.numpy(), self.config)
            self.assertLessEqual(float(np.abs(delta).max()), .15 + 1e-12)

    def test_float32_rounding_near_large_supported_values_stays_within_budget(self):
        observed = self.encoded(5000.)
        self.lower = self.encoded(4000.)[0, 0]
        self.upper = self.encoded(6000.)[0, 0]
        for target in [0., 1e5]:
            bounded = self.run_bound(observed, self.encoded(target))
            delta = decode_sensors(bounded.numpy(), self.config) - decode_sensors(observed.numpy(), self.config)
            self.assertLessEqual(float(np.abs(delta).max()), .15 + 1e-9)

    def test_invalid_bounds_masks_and_nonfinite_valid_proposals_are_rejected(self):
        x = self.encoded(0.)
        for lower, upper in [(self.upper, self.lower), (torch.full((21,), float("nan")), self.upper),
                             (torch.zeros(20), self.upper)]:
            with self.assertRaises(ValueError):
                bounded_finite_repair(x, x, self.finite, self.valid, lower, upper, self.config)
        bad = x.clone(); bad[0, 0, 0] = float("inf")
        with self.assertRaisesRegex(ValueError, "proposals"):
            self.run_bound(x, bad)
        self.finite = self.finite.float(); self.finite[0, 0, 0] = .5
        with self.assertRaisesRegex(ValueError, "zero/one"):
            self.run_bound(x, x)


class FittingEnvelopeScope(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({"unit": [1, 1, 2, 2], "value": [0., 2., -1., 3.],
                                   "regime": [0, 1, 0, 1]})
        def transform(frame):
            values = np.zeros((len(frame), 24), np.float32)
            values[:, 3:] = frame.value.to_numpy()[:, None]
            return values, frame.regime.to_numpy()
        self.scaler = SimpleNamespace(state={"fit_units": [1, 2], "centers": [[0., 0., 0.], [1., 0., 0.]],
                                             "sensor_transform": "asinh", "sensor_transform_scale": 1.},
                                      transform=transform)

    def test_envelope_uses_only_provided_fitting_observations_and_regimes(self):
        profile = fit_recovery_envelope(self.frame, self.scaler)
        np.testing.assert_array_equal(profile["lower"], [[-1.] * 21, [2.] * 21])
        np.testing.assert_array_equal(profile["upper"], [[0.] * 21, [3.] * 21])
        self.assertEqual(profile["regime_counts"], [2, 2])
        self.assertEqual(profile["fit_units"], [1, 2])

    def test_extra_held_engine_missing_fit_engine_and_empty_regime_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly"):
            fit_recovery_envelope(self.frame[self.frame.unit == 1], self.scaler)
        extra = pd.concat([self.frame, pd.DataFrame({"unit": [3], "value": [1000.], "regime": [0]})])
        with self.assertRaisesRegex(ValueError, "exactly"):
            fit_recovery_envelope(extra, self.scaler)
        no_second_regime = self.frame.copy(); no_second_regime.regime = 0
        with self.assertRaisesRegex(ValueError, "no fitting"):
            fit_recovery_envelope(no_second_regime, self.scaler)

    def test_fractional_engine_ids_cannot_disguise_the_fitting_scope(self):
        fractional = self.frame.copy(); fractional.unit = fractional.unit.astype(float)
        fractional.loc[fractional.unit == 1, "unit"] = 1.5
        with self.assertRaisesRegex(ValueError, "integer engine"):
            fit_recovery_envelope(fractional, self.scaler)


if __name__ == "__main__":
    unittest.main()
