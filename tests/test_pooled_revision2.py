"""Physical repair regret, causal prefix pairs and safe generator warm starts."""

import contextlib
from copy import deepcopy
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from grace_rul.config import COLUMNS, DEFAULT_CONFIG, quick_config
from grace_rul.checkpoints import save_checkpoint
from grace_rul.data import OperatingRegimeScaler, WindowDataset
from grace_rul.model import GraceRUL
from grace_rul.pooled import pooled_config
from grace_rul.training import (finite_recovery_loss, fit_pipeline, objectives,
    prefix_consistency_loss, recovery_warm_score, seed_all, trajectory_metrics, validate_training_config)


def source(units=4, cycles=60):
    return pd.DataFrame([[u, t, 0., 0., 100., *[j + .01*t + .03*u for j in range(21)]]
        for u in range(1, units+1) for t in range(1, cycles+1)], columns=COLUMNS)


class PhysicalFiniteLoss(unittest.TestCase):
    def test_same_relative_fault_skill_has_same_weight_at_different_amplitudes(self):
        cfg = {**DEFAULT_CONFIG, 'sensor_transform': 'linear'}
        ref = torch.zeros(1, 5, 2, dtype=torch.float64)
        observed = torch.tensor([.001, 1000.], dtype=torch.float64).expand_as(ref)
        repaired = (.5*observed).clone().requires_grad_()
        loss = finite_recovery_loss(repaired, observed, ref, torch.ones_like(ref, dtype=torch.bool), cfg)
        self.assertAlmostEqual(float(loss.detach()), np.log1p(.25), places=12)
        loss.backward()
        self.assertTrue(torch.isfinite(repaired.grad).all())
        self.assertTrue((repaired.grad > 0).all())
        # Relative weighting prevents the large-amplitude channel dominating.
        self.assertTrue(torch.allclose(repaired.grad[..., 0]*.001, repaired.grad[..., 1]*1000.))

    def test_regret_penalizes_worsening_in_original_linear_space(self):
        cfg = {**DEFAULT_CONFIG, 'sensor_transform': 'asinh'}
        ref = torch.tensor([[[2., -3.], [2.1, -3.1]]], dtype=torch.float64)
        observed_linear = torch.sinh(ref) + .1
        observed = torch.asinh(observed_linear)
        proposal = torch.asinh(torch.sinh(ref) + .2).requires_grad_()
        mask = torch.ones_like(ref, dtype=torch.bool)
        baseline = finite_recovery_loss(observed, observed, ref, mask, cfg)
        worse = finite_recovery_loss(proposal, observed, ref, mask, cfg)
        self.assertAlmostEqual(float(baseline), np.log(2), places=10)
        self.assertAlmostEqual(float(worse.detach()), np.log1p(4)+np.log(4), places=10)
        self.assertEqual(float(finite_recovery_loss(ref, observed, ref, mask, cfg)), 0.)
        worse.backward()
        self.assertTrue((proposal.grad > 0).all())

    def test_empty_masks_and_extreme_encoded_errors_have_finite_gradients(self):
        for active in [False, True]:
            proposal = torch.full((2, 4, 3), 1000., requires_grad=True)
            ref, observed = torch.zeros_like(proposal), torch.ones_like(proposal)
            loss = finite_recovery_loss(proposal, observed, ref,
                torch.full_like(ref, active, dtype=torch.bool), DEFAULT_CONFIG)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(proposal.grad).all())
            self.assertEqual(bool(proposal.grad.abs().sum()>0), active)

    def test_healthy_and_padding_cells_do_not_train_finite_repairs(self):
        p = torch.full((1, 3, 2), .2, requires_grad=True)
        mask = torch.zeros_like(p, dtype=torch.bool); mask[0, 1, 0] = True
        finite_recovery_loss(p, torch.ones_like(p), torch.zeros_like(p), mask,
            {**DEFAULT_CONFIG, 'sensor_transform': 'linear'}).backward()
        self.assertEqual(int(torch.count_nonzero(p.grad)), 1)


class PrefixPairs(unittest.TestCase):
    def setUp(self):
        seed_all(42)
        self.cfg = {**quick_config(DEFAULT_CONFIG), 'maximum_windows': None,
            'window_stride': 1, 'prefix_consistency_loss_weight': 1., 'prefix_pair_max_lag': 30}
        self.scaler = OperatingRegimeScaler().fit(source(4, 90), 'FD001', 42)

    def test_pairs_use_same_engine_past_and_never_move_when_future_is_appended(self):
        short = WindowDataset(source(4, 60), self.scaler, self.cfg, seed=13)
        long = WindowDataset(source(4, 90), self.scaler, self.cfg, seed=13)
        short.set_epoch(2); long.set_epoch(2)
        for i, (engine, end) in enumerate(short.samples):
            j = long.samples.index((engine, end))
            a,b = short[i], long[j]
            torch.testing.assert_close(a['previous_reference'], b['previous_reference'], rtol=0, atol=0)
            self.assertEqual(a['previous_cycle'], b['previous_cycle'])
            self.assertLessEqual(a['previous_cycle'], a['cycle'])
            self.assertLessEqual(a['cycle']-a['previous_cycle'], 30)
            self.assertEqual(bool(a['pair_valid']), end > 0)
            if end:
                previous_end = a['previous_cycle']-1
                expected = short.engines[engine]['values'][max(0,previous_end-self.cfg['sequence_length']+1):previous_end+1]
                np.testing.assert_array_equal(a['previous_reference'][a['previous_valid']], expected)
        for mode in ['validation','heldout','test','correction']:
            self.assertNotIn('previous_target', WindowDataset(source(), self.scaler, self.cfg, mode=mode)[0])

    def test_supervised_plateau_and_life_decline_both_have_zero_consistency_error(self):
        class ScalarModel(torch.nn.Module):
            def __init__(self, config):
                super().__init__(); self.config = config
                self.predictor = torch.nn.Sequential(torch.nn.Linear(1,1,bias=False),torch.nn.Dropout(.8))
                self.predictor[0].weight.data.fill_(1.)
            def forward(self, x, valid, **kwargs):
                return {'base': self.predictor(x[:, -1, :1]).flatten()}

        model = ScalarModel(self.cfg).train()
        cur = torch.zeros(2,3,24); prev = cur.clone()
        cur[:, -1, 0] = torch.tensor([125.,30.]); prev[:, -1, 0] = torch.tensor([125.,60.])
        batch = {'reference':cur,'previous_reference':prev,'valid':torch.ones(2,3,dtype=torch.bool),
            'previous_valid':torch.ones(2,3,dtype=torch.bool),'target':torch.tensor([125.,30.]),
            'previous_target':torch.tensor([125.,60.]),'pair_valid':torch.ones(2,dtype=torch.bool)}
        loss = prefix_consistency_loss(model,batch)
        self.assertEqual(float(loss.detach()),0.)
        self.assertTrue(model.training and model.predictor[-1].training)
        batch['reference'] = cur.clone();batch['reference'][0,-1,0] = 3.
        bad = prefix_consistency_loss(model,batch)
        self.assertGreater(float(bad.detach()),.1)
        bad.backward()
        self.assertTrue(torch.isfinite(model.predictor[0].weight.grad).all())
        with self.assertRaisesRegex(ValueError,'paired fitting'):
            prefix_consistency_loss(model,{})

    def test_trajectory_metrics_respect_engine_boundaries_and_true_target_change(self):
        records = pd.DataFrame({'unit':[1,2,1,2], 'cycle':[30,30,60,60],
            'prediction':[125.,60.,125.,30.], 'target':[125.,60.,125.,30.]})
        measured = trajectory_metrics(records)
        self.assertEqual(measured['RUL_step_pairs'],2)
        self.assertEqual(measured['RUL_step_RMSE'],0.)

    def test_optional_pair_lag_uses_the_validated_default(self):
        cfg = {k:v for k,v in self.cfg.items() if k!='prefix_pair_max_lag'}
        validate_training_config(cfg)
        dataset = WindowDataset(source(),self.scaler,cfg)
        sample = dataset[30]
        self.assertTrue(bool(sample['pair_valid']))
        self.assertLessEqual(sample['cycle']-sample['previous_cycle'],30)


class WarmSelection(unittest.TestCase):
    def test_worst_source_sensor_score_cannot_be_hidden_by_another_source(self):
        dataset = SimpleNamespace(source_by_unit={1:'A',2:'B'}, samples=[(0,0),(1,0)],
            engines=[{'unit':1},{'unit':2}])
        reports = [({'finite_recovery_relative_loss':v,'missing_imputation_relative_loss':0.,
                    'linear_intact_repair_RMSE':0.},None) for v in [.1,.3,1.,2.]]
        with patch('grace_rul.training.evaluate',side_effect=reports) as evaluated:
            score = recovery_warm_score(None,dataset,dataset,DEFAULT_CONFIG)
        self.assertAlmostEqual(score,1.5)
        self.assertEqual(evaluated.call_count,4)
        self.assertEqual([len(call.args[1].samples) for call in evaluated.call_args_list],[1,1,1,1])

    def test_joint_warm_start_uses_best_sensor_epoch_instead_of_last_rejected_epoch(self):
        seed_all(42)
        cfg = {**quick_config(DEFAULT_CONFIG), 'device':'cpu','maximum_batches':1,
            'generator_selection':'sensor_skill','finite_recovery_loss_weight':1.,
            'generator_epochs':3,'patience':4,'fault_start_horizon':0}
        snapshots, joint_starts = [], []
        sensor_scores = iter([1.,.5,.2,.7])

        def sensor_score(model,*args,**kwargs):
            snapshots.append(deepcopy(model.generator.state_dict()))
            return next(sensor_scores)

        def objective(model,*args,**kwargs):
            if next(model.predictor.parameters()).requires_grad:
                joint_starts.append(deepcopy(model.generator.state_dict()))
            return objectives(model,*args,**kwargs)

        def guards(*args,**kwargs):
            return {'clean':not kwargs.get('use_repair',True)}

        f = source(8,30)
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()), \
                patch('grace_rul.training.recovery_warm_score',side_effect=sensor_score), \
                patch('grace_rul.training.objectives',side_effect=objective), \
                patch('grace_rul.training.checkpoint_guards',side_effect=guards):
            path = Path(tmp)/'pipeline.pt'
            model,_,history,_ = fit_pipeline(f[f.unit<=6],f[f.unit>6],'FD001',cfg,42,checkpoint_path=path)
            checkpoint = torch.load(path,map_location='cpu',weights_only=True)
        self.assertEqual(checkpoint['warm_recovery_score'],.2)
        self.assertEqual(history[history.stage.eq('generator')].warm_recovery_selected.tolist(),[True,True,False])
        self.assertEqual(len(joint_starts),1)
        for name,value in snapshots[2].items():
            torch.testing.assert_close(joint_starts[0][name],value,rtol=0,atol=0)
        self.assertTrue(any(not torch.equal(snapshots[2][name],snapshots[3][name]) for name in snapshots[2]))
        self.assertFalse(bool(model.repair_enabled))

    def test_interrupted_generator_restores_sensor_best_without_restarting_selection(self):
        seed_all(42)
        cfg = {**quick_config(DEFAULT_CONFIG), 'device':'cpu','maximum_batches':1,
            'generator_selection':'sensor_skill','finite_recovery_loss_weight':1.,
            'generator_epochs':3,'patience':4,'fault_start_horizon':0}
        f = source(8,30)

        def interrupt(path,state):
            save_checkpoint(path,state)
            if state['stage']=='generator' and state['next_epoch']==1:
                raise InterruptedError('fixture interruption after generator checkpoint')

        def guards(*args,**kwargs):
            return {'clean':not kwargs.get('use_repair',True)}

        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()), \
                patch('grace_rul.training.checkpoint_guards',side_effect=guards):
            path = Path(tmp)/'pipeline.pt'
            with patch('grace_rul.training.recovery_warm_score',side_effect=[1.,.2]), \
                    patch('grace_rul.training.save_checkpoint',side_effect=interrupt):
                with self.assertRaises(InterruptedError):
                    fit_pipeline(f[f.unit<=6],f[f.unit>6],'FD001',cfg,42,checkpoint_path=path)
            first = torch.load(path,map_location='cpu',weights_only=True)
            self.assertEqual(first['warm_recovery_score'],.2)
            with patch('grace_rul.training.recovery_warm_score',side_effect=[.7,.1]) as score:
                _,_,history,_ = fit_pipeline(f[f.unit<=6],f[f.unit>6],'FD001',cfg,42,checkpoint_path=path)
            last = torch.load(path,map_location='cpu',weights_only=True)
        self.assertEqual(score.call_count,2)
        self.assertEqual(last['warm_recovery_score'],.1)
        self.assertEqual(history[history.stage.eq('generator')].warm_recovery_selected.tolist(),[True,False,True])

    def test_recipe_two_keeps_inference_state_and_old_recipe_defaults_compatible(self):
        cfg = {**pooled_config(quick=True),'device':'cpu'}
        self.assertEqual(cfg['pooled_protocol_revision'],2)
        validate_training_config(cfg)
        old = {k:v for k,v in cfg.items() if k not in ['finite_recovery_loss_weight',
            'prefix_consistency_loss_weight','prefix_pair_max_lag','generator_selection']}
        old['pooled_protocol_revision'] = 1
        a,b = GraceRUL(old).eval(),GraceRUL(cfg).eval()
        b.load_state_dict(a.state_dict(),strict=True)
        x = torch.randn(2,cfg['sequence_length'],24);valid = torch.ones(x.shape[:2],dtype=torch.bool)
        with torch.inference_mode():
            torch.testing.assert_close(a(x,valid)['prediction'],b(x,valid)['prediction'],rtol=0,atol=0)
        for value in [0,True,-1,1.5]:
            with self.assertRaisesRegex(ValueError,'prefix_pair_max_lag'):
                validate_training_config({**cfg,'prefix_pair_max_lag':value})


if __name__ == '__main__':
    unittest.main()
