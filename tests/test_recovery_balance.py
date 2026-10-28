"""Physical imputation skill, stable tail gradients and validation-only selection."""

from collections import Counter
import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from grace_rul.artifacts import load_bundle, run_experiment
from grace_rul.checkpoints import save_checkpoint
from grace_rul.config import COLUMNS, DEFAULT_CONFIG, quick_config
from grace_rul.data import FAULTS, OperatingRegimeScaler, WindowDataset
from grace_rul.training import (fit_pipeline, intact_tail_identity_loss, missing_imputation_loss,
    seed_all, validate_training_config, validation_datasets, validation_score)


ROOT = Path(__file__).resolve().parents[1]
RECIPE = json.loads((ROOT / "configs/v8_recovery_balance.json").read_text())


def frame(units=8, cycles=8):
    result = pd.DataFrame([[u, t, *[float(j + t / 10 + u / 20) for j in range(24)]]
        for u in range(1, units + 1) for t in range(1, cycles + 1)], columns=COLUMNS)
    result[["unit", "cycle"]] = result[["unit", "cycle"]].astype(int)
    return result


class RecoveryObjectives(unittest.TestCase):
    def setUp(self):
        seed_all(42)
        self.config = {**DEFAULT_CONFIG, **RECIPE}

    def test_log_tail_matches_original_physical_group_mse_ranking(self):
        a = torch.randn(4, 7, 3, dtype=torch.float64, requires_grad=True)
        b = torch.randn_like(a)
        mask = torch.rand_like(a) > .3
        error = torch.sinh(a) - torch.sinh(b)
        count = mask.sum(1)
        mse = ((error / .15).square().masked_fill(~mask, 0).sum(1) / count.clamp_min(1))[count > 0]
        k = max(1, math.ceil(mse.numel() * .1))
        expected = torch.log1p(mse).topk(k).values.mean()
        measured = intact_tail_identity_loss(a, b, mask, self.config)
        self.assertTrue(torch.allclose(measured, expected, rtol=1e-12, atol=1e-12))
        measured.backward()
        self.assertTrue(torch.isfinite(a.grad).all())
        self.assertTrue((a.grad[~mask] == 0).all())

    def test_extreme_proposal_and_exact_identity_have_finite_gradients(self):
        for value in [0., 1000., -1000.]:
            a = torch.full((2, 5, 3), value, requires_grad=True)
            ref = torch.zeros_like(a)
            loss = intact_tail_identity_loss(a, ref, torch.ones_like(a, dtype=torch.bool), self.config)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(a.grad).all())
            if value:
                self.assertGreater(float(a.grad.abs().sum()), 0.)
            else:
                self.assertEqual(float(loss.detach()), 0.)
                self.assertEqual(float(a.grad.abs().sum()), 0.)

    def test_missing_skill_beats_mean_fill_only_when_prediction_is_better(self):
        config = {**self.config, "sensor_transform": "linear"}
        ref = torch.ones(1, 3, 2)
        mask = torch.ones_like(ref, dtype=torch.bool)
        baseline = torch.zeros_like(ref, requires_grad=True)
        zero = missing_imputation_loss(baseline, ref, mask, config)
        expected = math.log1p(1 / (1 + .15 ** 2))
        self.assertAlmostEqual(float(zero.detach()), expected)
        zero.backward()
        self.assertTrue((baseline.grad < 0).all())
        perfect = missing_imputation_loss(ref, ref, mask, config)
        wrong = missing_imputation_loss(-ref, ref, mask, config)
        self.assertEqual(float(perfect), 0.)
        self.assertGreater(float(wrong), float(zero.detach()))

    def test_empty_or_fault_masks_never_transform_invalid_references(self):
        a = torch.full((2, 5, 3), 1e5, requires_grad=True)
        ref = torch.full_like(a, -1e5)
        empty = torch.zeros_like(a, dtype=torch.bool)
        loss = intact_tail_identity_loss(a, ref, empty, self.config) + missing_imputation_loss(a, ref, empty, self.config)
        loss.backward()
        self.assertEqual(float(loss.detach()), 0.)
        self.assertEqual(float(a.grad.abs().sum()), 0.)


class BalancedRecoveryData(unittest.TestCase):
    def setUp(self):
        self.config = {**quick_config(DEFAULT_CONFIG), **RECIPE, "maximum_windows": None,
                       "window_stride": 1, "fault_start_horizon": 0}
        self.fitting = frame(8)
        self.scaler = OperatingRegimeScaler().fit(self.fitting, "FD001", 42)

    def test_all_families_severities_and_clean_allocation_without_extra_windows(self):
        f = frame(96)
        ds = WindowDataset(f, self.scaler, self.config, seed=42)
        ds.set_epoch(0)
        counts = Counter(ds.recovery_plans.values())
        self.assertEqual(counts[("clean", 0.)], 24)
        for family in [*FAULTS, "mixed"]:
            for severity in [.1, .2, .3]:
                self.assertEqual(counts[(family, severity)], 3)
        self.assertEqual(len(ds), len(f))
        before = dict(ds.recovery_plans)
        ds.set_epoch(1)
        self.assertNotEqual(before, ds.recovery_plans)
        self.assertEqual(Counter(ds.recovery_plans.values()), counts)

    def test_plan_and_corruption_are_independent_of_future_trajectory_rows(self):
        short = WindowDataset(frame(8, 8), self.scaler, self.config, seed=70)
        long = WindowDataset(frame(8, 12), self.scaler, self.config, seed=70)
        short.set_epoch(2, progress=.5)
        long.set_epoch(2, progress=.5)
        self.assertEqual(short.recovery_plans, long.recovery_plans)
        for i in range(8):
            short[next(j for j, (engine, _) in enumerate(short.samples) if engine == i)]
            long[next(j for j, (engine, _) in enumerate(long.samples) if engine == i)]
            np.testing.assert_allclose(short.cache[i][0], long.cache[i][0][:8], equal_nan=True)
            np.testing.assert_array_equal(short.cache[i][1], long.cache[i][1][:8])

    def test_explicit_fault_validation_and_clean_extremes_keep_their_contract(self):
        validation = WindowDataset(self.fitting, self.scaler, self.config, mode="validation", family="mixed")
        self.assertIsNone(validation.recovery_plans)
        explicit = WindowDataset(self.fitting, self.scaler, self.config, family="stuck")
        self.assertIsNone(explicit.recovery_plans)
        for fraction in [0., 1.]:
            ds = WindowDataset(self.fitting, self.scaler, {**self.config, "clean_fraction": fraction})
            clean = sum(f == "clean" for f, _ in ds.recovery_plans.values())
            self.assertEqual(clean, len(ds.engines) if fraction else 0)


class SelectionPolicy(unittest.TestCase):
    def setUp(self):
        seed_all(42)

    def test_all_faults_enter_score_while_mixed_reconstruction_guard_stays_separate(self):
        config = {**DEFAULT_CONFIG, "rul_selection": "fault_macro"}
        contexts = {family: object() for family in [*FAULTS, "mixed"]}
        reports = [({"RMSE": 10.}, None), ({"RMSE": 20., "active_fault_RMSE": 30., "reconstruction_RMSE": 1.2}, None)]
        reports += [({"RMSE": float(i), "active_fault_RMSE": float(i)}, None) for i in range(2, 9)]
        with patch("grace_rul.training.evaluate", side_effect=reports) as evaluated:
            score, clean, mixed = validation_score(None, None, None, config, selection_datasets=contexts)
        self.assertEqual(evaluated.call_count, 9)
        self.assertAlmostEqual(score, .5 * (10 + (30 + sum(range(2, 9))) / 8))
        self.assertEqual(mixed["reconstruction_RMSE"], 1.2)
        self.assertEqual(set(mixed["selection_family_RMSE"]), {*FAULTS, "mixed"})
        with patch("grace_rul.training.evaluate", return_value=({"RMSE": 1.}, None)):
            with self.assertRaisesRegex(ValueError, "every declared"):
                validation_score(None, None, None, config, selection_datasets={"mixed": None})

    def test_selection_contexts_share_held_engines_and_observed_endpoints(self):
        config = {**quick_config(DEFAULT_CONFIG), **RECIPE}
        f = frame()
        fit, val = f[f.unit <= 6], f[f.unit > 6]
        scaler = OperatingRegimeScaler().fit(fit, "FD001", 42)
        clean, mixed, contexts = validation_datasets(val, scaler, config)
        self.assertIs(mixed, contexts["mixed"])
        self.assertEqual(set(contexts), {*FAULTS, "mixed"})
        for ds in contexts.values():
            self.assertEqual(ds.samples, clean.samples)
            self.assertEqual({e["unit"] for e in ds.engines}, {7, 8})
            self.assertEqual(ds.severity, .2)
            self.assertIsNone(ds.recovery_plans)

    def test_clean_anchor_selection_keeps_macro_baseline_for_next_stage(self):
        f = frame()
        config = {**quick_config(DEFAULT_CONFIG), "device": "cpu", "predictor_epochs": 2,
            "train_repair": False, "train_corrupted_predictor": False,
            "maximum_batches": 1, "clean_pretraining_selection": "clean_rmse"}
        reports = [(20., {"RMSE": 8., "NASA_Score_Mean": 1.}, {"RMSE": 32., "reconstruction_RMSE": None}),
                   (10., {"RMSE": 9., "NASA_Score_Mean": 1.1}, {"RMSE": 11., "reconstruction_RMSE": None})]
        with tempfile.TemporaryDirectory() as tmp, patch("grace_rul.training.validation_score", side_effect=reports):
            path = Path(tmp) / "pipeline.pt"
            model, _, history, macro = fit_pipeline(f[f.unit <= 6], f[f.unit > 6], "FD001", config, 42,
                                                     verbose=False, checkpoint_path=path)
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        self.assertEqual(model.pretrained_clean_rmse, 8.)
        self.assertEqual(macro, 20.)
        self.assertEqual(checkpoint["best_score"], 20.)
        self.assertEqual(checkpoint["best_selection_score"], 8.)
        self.assertEqual(history.checkpoint_selected.tolist(), [True, False])

    def test_resumed_pretraining_preserves_clean_selection_scale(self):
        f = frame()
        config = {**quick_config(DEFAULT_CONFIG), "device": "cpu", "predictor_epochs": 2,
            "train_repair": False, "train_corrupted_predictor": False,
            "maximum_batches": 1, "clean_pretraining_selection": "clean_rmse"}

        def interrupt_after_save(path, state):
            save_checkpoint(path, state)
            raise InterruptedError("fixture interruption after the first persisted epoch")

        first = (20., {"RMSE": 8., "NASA_Score_Mean": 1.}, {"RMSE": 32., "reconstruction_RMSE": None})
        second = (10., {"RMSE": 9., "NASA_Score_Mean": 1.1}, {"RMSE": 11., "reconstruction_RMSE": None})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pipeline.pt"
            with patch("grace_rul.training.validation_score", return_value=first), \
                    patch("grace_rul.training.save_checkpoint", side_effect=interrupt_after_save):
                with self.assertRaises(InterruptedError):
                    fit_pipeline(f[f.unit <= 6], f[f.unit > 6], "FD001", config, 42, verbose=False, checkpoint_path=path)
            with patch("grace_rul.training.validation_score", return_value=second) as measured:
                model, _, history, macro = fit_pipeline(f[f.unit <= 6], f[f.unit > 6], "FD001", config, 42,
                    verbose=False, checkpoint_path=path)
        self.assertEqual(measured.call_count, 1)
        self.assertEqual(model.pretrained_clean_rmse, 8.)
        self.assertEqual(macro, 20.)
        self.assertEqual(history.checkpoint_selected.tolist(), [True, False])

    def test_recovery_compares_macro_with_macro_after_clean_anchor(self):
        f = frame()
        config = {**quick_config(DEFAULT_CONFIG), "device": "cpu", "predictor_epochs": 1,
            "generator_epochs": 1, "train_corrupted_predictor": False, "maximum_batches": 1,
            "detector_training": "joint", "finite_repair_gate": "continuous", "recovery_validation": False,
            "clean_pretraining_selection": "clean_rmse"}
        reports = [(20., {"RMSE": 8., "NASA_Score_Mean": 1.}, {"RMSE": 32., "reconstruction_RMSE": None}),
                   (15., {"RMSE": 8.01, "NASA_Score_Mean": 1.}, {"RMSE": 22., "reconstruction_RMSE": .1,
                    "linear_intact_repair_RMSE": .01, "linear_reconstruction_ratio": .5})]
        with patch("grace_rul.training.validation_score", side_effect=reports):
            model, _, history, macro = fit_pipeline(f[f.unit <= 6], f[f.unit > 6], "FD001", config, 42, verbose=False)
        self.assertEqual(macro, 15.)
        self.assertTrue(bool(model.repair_enabled))
        self.assertTrue(history.iloc[-1].checkpoint_selected)

    def test_repaired_recipe_validates_without_relaxing_guards(self):
        config = {**DEFAULT_CONFIG, **RECIPE}
        validate_training_config(config)
        self.assertEqual(config["max_linear_reconstruction_ratio"], 1.)
        self.assertEqual(config["max_reconstruction_rmse"], 1.5)
        self.assertEqual(config["max_clean_rmse_increase"], .01)
        self.assertEqual(config["max_clean_nasa_increase"], .01)
        for override in [{"recovery_train_severities": []}, {"recovery_train_severities": [.6]},
                         {"recovery_train_severities": [[.2]]}, {"rul_selection": []},
                         {"identity_tail_penalty": "clip"}, {"imputation_loss_weight": -1.}]:
            with self.assertRaises(ValueError):
                validate_training_config({**config, **override})


class RecipeIntegration(unittest.TestCase):
    def test_synthetic_workflow_exports_and_reuses_benchmark_with_family_macro(self):
        # Synthetic files exercise training, OOF, correction, export and benchmarks.
        # Their metrics are fixtures, not evidence of learned C-MAPSS accuracy.
        config = {**quick_config(DEFAULT_CONFIG), **RECIPE, "device": "cpu", "maximum_batches": 1,
            "run_matched_benchmarks": True, "run_extended_evaluation": False, "benchmark_tune": False,
            "benchmark_variants": ["grace_without_correction", "paper_A_cnn_gru"]}
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            (root / "data").mkdir()
            frame(18, 12).to_csv(root / "data/train_FD001.txt", sep=" ", header=False, index=False)
            frame(3, 8).to_csv(root / "data/test_FD001.txt", sep=" ", header=False, index=False)
            (root / "data/RUL_FD001.txt").write_text("30\n40\n50\n")
            path, tables = run_experiment(root, config, root / "run")
            _, loaded = load_bundle(path)
        self.assertTrue(loaded["FD001"]["entry"]["training_config"]["quick_run"])
        self.assertEqual(loaded["FD001"]["entry"]["training_config"]["rul_selection"], "fault_macro")
        history, correction = tables["training_history"], tables["correction_history"]
        self.assertEqual(history.iloc[0].selection_policy, "clean_rmse")
        self.assertTrue(history.iloc[1:].selection_policy.eq("fault_macro").all())
        self.assertTrue(correction.selection_policy.eq("fault_macro").all())
        for family in [*FAULTS, "mixed"]:
            self.assertTrue(history[f"validation_{family}_selection_RMSE"].notna().all())
            self.assertTrue(correction[f"validation_{family}_selection_RMSE"].notna().all())
        audit = tables["benchmark_training_audit"].set_index("variant")
        self.assertTrue(bool(audit.loc["grace_without_correction"].reused_identical_weights))
        self.assertFalse(bool(audit.loc["paper_A_cnn_gru"].reused_identical_weights))
        self.assertFalse(tables["superiority_checks"].acceptance_target_met.any())


if __name__ == "__main__":
    unittest.main()
