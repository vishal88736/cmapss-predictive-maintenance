"""Regression cases for cross-regime faults and rejected-recovery fallbacks."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from grace_rul.artifacts import load_bundle
from grace_rul.config import DEFAULT_CONFIG, FEATURE_COLUMNS, SUPPORTED_ARCHITECTURES, quick_config
from grace_rul.corruption import FAULTS, corrupt_trajectory
from grace_rul.data import OperatingRegimeScaler, load_training_subset
from grace_rul.model import GraceRUL, has_finite_fault_detector
from grace_rul.normalization import decode_sensors, encode_sensors
from grace_rul.training import acceptable_checkpoint, finite_fault_loss, fit_pipeline, masked_repair_loss, objectives, seed_all

ROOT = Path(__file__).resolve().parents[1]


class CrossRegimeRecovery(unittest.TestCase):
    @staticmethod
    def profile():
        return {"centers": [[0., 0., 0.], [2., 0., 0.]],
                "sensor_mean": [np.zeros(21).tolist(), (np.ones(21)*100).tolist()],
                "sensor_std": [(np.ones(21)*.01).tolist(), (np.ones(21)*.02).tolist()],
                "sensor_fault_scale": (np.ones(21)*.05).tolist()}

    def test_compression_is_invertible_monotonic_and_preserves_small_signals(self):
        profile = {"sensor_transform": "asinh", "sensor_transform_scale": 1.}
        z = np.array([-1e5, -5090., -1., -.01, 0., .01, 1., 5090., 1e5])
        encoded = encode_sensors(z, profile)
        self.assertTrue((np.diff(encoded) > 0).all())
        self.assertLess(abs(encoded[-2]), 10.)
        np.testing.assert_allclose(decode_sensors(encoded, profile), z, rtol=1e-12)
        np.testing.assert_allclose(encoded[[3,5]], z[[3,5]], atol=2e-7)

    def test_compression_preserves_physical_faults_and_intact_bits(self):
        raw_reference = np.zeros((180, 24), np.float32)
        regimes = np.arange(180) % 2
        legacy = self.profile()
        compressed = {**legacy, "sensor_transform": "asinh"}
        scaler_old, scaler_new = OperatingRegimeScaler(legacy), OperatingRegimeScaler(compressed)
        for family in [*FAULTS, "mixed", "clean"]:
            for seed in range(8):
                with self.subTest(family=family, seed=seed):
                    options = dict(family=family, severity=.2, start_horizon=0, regimes=regimes)
                    linear, labels_old = corrupt_trajectory(raw_reference, np.random.default_rng(seed), profile=legacy, **options)
                    encoded, labels_new = corrupt_trajectory(raw_reference, np.random.default_rng(seed), profile=compressed, **options)
                    np.testing.assert_array_equal(labels_old, labels_new)
                    np.testing.assert_array_equal(encoded[:, :3], raw_reference[:, :3])
                    np.testing.assert_array_equal(encoded[:, 3:][~labels_new], raw_reference[:, 3:][~labels_new])
                    np.testing.assert_allclose(scaler_old.inverse_sensors(linear[:, 3:], regimes),
                                               scaler_new.inverse_sensors(encoded[:, 3:], regimes), atol=5e-5, rtol=1e-6)

    def test_compressed_faults_are_prefix_invariant_and_missingness_stays_missing(self):
        profile = {**self.profile(), "sensor_transform": "asinh"}
        reference = np.random.default_rng(20).normal(size=(160,24)).astype(np.float32)
        reference[:, 3:] = np.arcsinh(reference[:, 3:])
        regimes = np.arange(160) % 2
        for family in [*FAULTS, "mixed", None, "clean"]:
            for seed in range(4):
                full, labels = corrupt_trajectory(reference, np.random.default_rng(seed), family, profile=profile, regimes=regimes)
                for length in [1, 30, 80, 120]:
                    short, short_labels = corrupt_trajectory(reference[:length], np.random.default_rng(seed), family,
                                                             profile=profile, regimes=regimes[:length])
                    np.testing.assert_array_equal(short, full[:length])
                    np.testing.assert_array_equal(short_labels, labels[:length])
        damaged, _ = corrupt_trajectory(reference, np.random.default_rng(42), 'outage', profile=profile,
                                         regimes=regimes, start_horizon=0)
        self.assertTrue(np.isnan(damaged[:, 3:]).any())

    def test_clean_comparison_can_use_zero_severity(self):
        reference = np.ones((10,24), np.float32)
        out, altered = corrupt_trajectory(reference, np.random.default_rng(42), 'clean', severity=0)
        np.testing.assert_array_equal(out, reference)
        self.assertFalse(altered.any())

    def test_rare_positive_gradient_is_not_diluted_by_more_intact_timesteps(self):
        gradients = []
        for n in [100, 10000]:
            logits = torch.full((1,n,1), -4., requires_grad=True)
            labels = torch.zeros_like(logits, dtype=torch.bool); labels[0,0,0] = True
            finite_fault_loss(logits, labels, torch.ones_like(labels), DEFAULT_CONFIG).backward()
            gradients.append(float(logits.grad[0,0,0]))
            self.assertGreater(abs(gradients[-1]), .4)
        self.assertAlmostEqual(*gradients, places=7)

    def test_missing_entries_do_not_train_the_finite_fault_classifier(self):
        logits = torch.zeros((1,3,2), requires_grad=True)
        available = torch.zeros_like(logits, dtype=torch.bool)
        loss = finite_fault_loss(logits, torch.ones_like(available), available, DEFAULT_CONFIG)
        loss.backward()
        self.assertEqual(float(loss.detach()), 0.)
        self.assertEqual(float(logits.grad.abs().sum()), 0.)

    def test_extreme_repair_error_has_finite_bounded_gradient(self):
        error = torch.tensor([1e4, 0., 100.], requires_grad=True)
        loss = masked_repair_loss(error, torch.tensor([True,True,False]), DEFAULT_CONFIG)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertLessEqual(float(error.grad.abs().max()), .5)
        self.assertEqual(float(error.grad[-1]), 0.)

    def test_recovery_guard_also_checks_uncompressed_sensor_error(self):
        clean = {"RMSE":10.}
        damaged = {"reconstruction_RMSE":.1, "intact_repair_RMSE":.01,
                   "linear_intact_repair_RMSE":.2, "linear_reconstruction_ratio":.5}
        self.assertFalse(acceptable_checkpoint(clean, damaged, 10., DEFAULT_CONFIG))
        damaged.update(linear_intact_repair_RMSE=.01, linear_reconstruction_ratio=1.2)
        self.assertFalse(acceptable_checkpoint(clean, damaged, 10., DEFAULT_CONFIG))
        self.assertTrue(acceptable_checkpoint(clean, damaged, 10., DEFAULT_CONFIG, use_repair=False))
        self.assertFalse(acceptable_checkpoint({"RMSE":10.2}, damaged, 10., DEFAULT_CONFIG, use_repair=False))

    def test_joint_bypass_can_retain_predictor_and_detector_when_recovery_fails(self):
        config = {**quick_config({**DEFAULT_CONFIG, "device":"cpu"}), "maximum_batches":1,
                  "adversarial_loss_weight":0., "validate_bypass":True}
        frame = load_training_subset(ROOT/'data', 'FD001')
        before, calls = {}, []
        def validate(model, *args, **options):
            calls.append(options['use_repair'])
            if len(calls)==1:
                before.update(deepcopy(model.state_dict()))
                return 10., {"RMSE":10.}, {"RMSE":10., "reconstruction_RMSE":1.}
            if len(calls)<4:
                return 9., {"RMSE":10.}, {"RMSE":8., "reconstruction_RMSE":500.}
            return 8.5, {"RMSE":10.}, {"RMSE":7., "reconstruction_RMSE":500.}
        with patch('grace_rul.training.validation_score', side_effect=validate):
            model, _, history, score = fit_pipeline(frame[frame.unit<=6], frame[frame.unit.isin([7,8])],
                                                   'FD001',config,42,verbose=False)
        self.assertEqual(calls,[False,True,True,False])
        self.assertEqual(score,8.5)
        self.assertFalse(bool(model.repair_enabled))
        self.assertEqual(history.validation_path.tolist(),['bypass','repair','bypass'])
        self.assertFalse(history.iloc[-1].repair_checkpoint_accepted)
        self.assertEqual(history.iloc[-1].repair_validation_clean_RMSE, 10.)
        self.assertFalse(history.iloc[-1].repair_guard_reconstruction)
        self.assertTrue(history.iloc[-1].bypass_guard_clean_rmse)
        for component in ['predictor','detector']:
            self.assertTrue(any(not torch.equal(before[k],v) for k,v in model.state_dict().items() if k.startswith(component)))

    def test_legacy_format5_weights_load_with_original_linear_preprocessing(self):
        seed_all(42)
        config = quick_config({**DEFAULT_CONFIG, "device":"cpu"})
        for key in ['sensor_transform','sensor_transform_scale','repair_loss','repair_huber_beta','detector_loss',
                    'validate_bypass','bypass_rul_loss_weight','max_linear_reconstruction_ratio',
                    'finite_repair_gate','detach_repair_gate','detector_training','identity_loss_space',
                    'recovery_safety']:
            config.pop(key)
        model = GraceRUL(config).eval()
        profile = self.profile()
        entry = {'state_dict':model.state_dict(),'scaler':profile,'training_config':config,
                 'thresholds':[.5]*21,'calibration':{'lower_offset':-2.,'upper_offset':2.},
                 'correction_bound':15.,'repair_selected':True}
        bundle = {'format_version':5,'architecture':SUPPORTED_ARCHITECTURES[5],
                  'feature_columns':FEATURE_COLUMNS,'config':config,'models':{'FD001':entry}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'legacy.pt';torch.save(bundle,path)
            _, loaded = load_bundle(path)
            x = torch.randn(2,12,24)
            with torch.no_grad():
                torch.testing.assert_close(model(x)['prediction'],loaded['FD001']['model'](x)['prediction'],rtol=0,atol=0)
            self.assertNotIn('sensor_transform',loaded['FD001']['scaler'].state)

    def test_rejected_recovery_warm_starts_joint_but_is_never_exported_unguarded(self):
        config = {**quick_config({**DEFAULT_CONFIG,'device':'cpu'}),'maximum_batches':1,'adversarial_loss_weight':0.}
        frame = load_training_subset(ROOT/'data','FD001')
        before, recovery, inspected = {}, {}, []
        def validate(model,*args,**options):
            if not before:
                before.update(deepcopy(model.state_dict()))
                return 10., {'RMSE':10.}, {'RMSE':10.,'reconstruction_RMSE':1.}
            if not recovery:
                recovery.update({k:v.clone() for k,v in model.state_dict().items() if k.startswith(('detector.','generator.'))})
            return 9., {'RMSE':11.}, {'RMSE':7.,'reconstruction_RMSE':500.}
        def inspect(model,*args,**options):
            if any(p.requires_grad for p in model.predictor.parameters()) and not inspected:
                for key,value in recovery.items():
                    torch.testing.assert_close(model.state_dict()[key],value,rtol=0,atol=0)
                inspected.append(True)
            return objectives(model,*args,**options)
        with patch('grace_rul.training.validation_score',side_effect=validate), \
             patch('grace_rul.training.objectives',side_effect=inspect):
            model,_,_,score = fit_pipeline(frame[frame.unit<=6],frame[frame.unit.isin([7,8])],
                                           'FD001',config,42,verbose=False)
        self.assertEqual(inspected,[True])
        self.assertEqual(score,10.)
        for key,value in before.items():
            if key.startswith(('predictor.', 'generator.', 'corrector.')):
                torch.testing.assert_close(model.state_dict()[key],value,rtol=0,atol=0)
        for key,value in recovery.items():
            if key.startswith('detector.'):
                torch.testing.assert_close(model.state_dict()[key],value,rtol=0,atol=0)
        np.testing.assert_array_equal(model.finite_repair_thresholds.cpu(),
            np.asarray(model.detector_audit['thresholds'], np.float32))

    def test_detector_availability_is_independent_of_recovery_bypass(self):
        model = GraceRUL(quick_config(DEFAULT_CONFIG))
        model.repair_enabled.fill_(False)
        self.assertFalse(has_finite_fault_detector(model))
        with torch.no_grad():model.detector.head.weight[0,0] = .01
        self.assertTrue(has_finite_fault_detector(model))


if __name__ == '__main__':
    unittest.main()
