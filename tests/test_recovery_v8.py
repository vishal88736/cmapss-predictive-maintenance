"""Training and model integration for healthy-reading displacement protection."""
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from grace_rul.config import DEFAULT_CONFIG, quick_config
from grace_rul.data import load_training_subset
from grace_rul.model import GraceRUL
from grace_rul.training import (fit_pipeline, intact_identity_loss,
    intact_tail_identity_loss, objectives, seed_all)

ROOT = Path(__file__).resolve().parents[1]


class RecoveryV8(unittest.TestCase):
    def setUp(self):
        seed_all(42)
        self.config = quick_config({**DEFAULT_CONFIG, "device": "cpu"})

    def test_tail_loss_highlights_a_single_bad_window_channel(self):
        ref = torch.zeros(4, 8, 21)
        repaired = ref.clone(); repaired[0, :, 0] = 1.
        repaired.requires_grad_()
        intact = torch.ones_like(ref, dtype=torch.bool)
        mean = intact_identity_loss(repaired, ref, intact, self.config)
        tail = intact_tail_identity_loss(repaired, ref, intact, self.config)
        self.assertGreater(float(tail.detach()), 8 * float(mean.detach()))
        tail.backward()
        self.assertTrue(torch.isfinite(repaired.grad).all())
        self.assertGreater(float(repaired.grad[0, :, 0].abs().sum()), 0.)
        self.assertEqual(float(repaired.grad[1:].abs().sum()), 0.)

    def test_tail_excludes_faults_and_padding_before_inverse(self):
        repaired = torch.full((2, 8, 21), 1e5, requires_grad=True)
        ref = torch.zeros_like(repaired)
        intact = torch.zeros_like(repaired, dtype=torch.bool)
        loss = intact_tail_identity_loss(repaired, ref, intact, self.config)
        loss.backward()
        self.assertEqual(float(loss.detach()), 0.)
        self.assertEqual(float(repaired.grad.abs().sum()), 0.)

    def test_unsafe_proposal_keeps_training_gradient_after_projection(self):
        config = {**self.config, **{key: 0. for key in self.config if key.endswith('_loss_weight')},
                  'identity_tail_loss_weight': 1.}
        model = GraceRUL(config).eval()
        with torch.no_grad():
            model.finite_repair_thresholds.fill_(.01)
            model.detector.head.bias.fill_(20.)
            model.generator.decoder[-1].bias.fill_(2.)
        x = torch.zeros(2, 8, 24)
        batch = {'x': x, 'reference': x.clone(), 'valid': torch.ones(2, 8, dtype=torch.bool),
                 'altered': torch.zeros(2, 8, 21, dtype=torch.bool), 'target': torch.ones(2) * 70}
        loss, result = objectives(model, batch, consistency=False)
        self.assertTrue((torch.sinh(result['repaired'].double()).abs() <= .15).all())
        self.assertTrue((result['repair_proposal'] > 1.).all())
        loss.backward()
        self.assertGreater(float(model.generator.decoder[-1].bias.grad.abs().sum()), 0.)

    def test_support_setter_rejects_wrong_transform_and_bad_shapes(self):
        model = GraceRUL(self.config)
        profile = {'centers': [[0., 0., 0.]], 'lower': [[-1.] * 21], 'upper': [[1.] * 21],
                   'sensor_transform': 'asinh', 'sensor_transform_scale': 1., 'fit_units': [1]}
        model.set_recovery_envelope(profile)
        self.assertEqual(int(model.recovery_regime_count), 1)
        self.assertEqual(model.recovery_support['fit_units'], [1])
        with self.assertRaisesRegex(ValueError, 'transforms disagree'):
            model.set_recovery_envelope({**profile, 'sensor_transform': 'linear'})
        with self.assertRaisesRegex(ValueError, 'Invalid fitting-support'):
            model.set_recovery_envelope({**profile, 'lower': [[2.] * 21]})

    def test_full_matrix_rejection_prevents_a_better_primary_candidate_export(self):
        frame = load_training_subset(ROOT/'data', 'FD001')
        config = {**self.config, 'maximum_batches': 1, 'adversarial_loss_weight': 0.,
                  'train_corrupted_predictor': False}
        calls = []
        def primary(model, *args, **kwargs):
            score = 10. if not calls else 9.
            calls.append(score)
            return score, {'RMSE': 10.}, {'RMSE': score, 'reconstruction_RMSE': .1,
                'linear_intact_repair_RMSE': .01, 'linear_reconstruction_ratio': .5}
        rejection = {'complete': True, 'accepted': False,
                     'worst': {'linear_intact_repair_RMSE': .3}}
        with patch('grace_rul.training.validation_score', side_effect=primary), \
             patch('grace_rul.recovery_validation.audit_recovery_candidate', return_value=rejection) as audit:
            model, _, history, score = fit_pipeline(frame[frame.unit <= 6], frame[frame.unit.isin([7, 8])],
                'FD001', config, 42, verbose=False)
        self.assertEqual(audit.call_count, 1)
        self.assertEqual(score, 10.)
        self.assertFalse(bool(model.repair_enabled))
        self.assertEqual(history.iloc[-1].repair_matrix_status, 'rejected')
        self.assertIsNone(model.recovery_validation_audit)


if __name__ == '__main__':
    unittest.main()
