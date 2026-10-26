"""Checks for paper-informed training and matched comparison safeguards."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from grace_rul.artifacts import load_bundle, run_experiment
from grace_rul.benchmark import _active_comparison_tables, _comparison_tables, paired_bootstrap
from grace_rul.checkpoints import load_checkpoint, pipeline_identity, save_checkpoint
from grace_rul.config import DEFAULT_CONFIG, quick_config
from grace_rul.data import OperatingRegimeScaler, WindowDataset, load_subset, load_training_subset
from grace_rul.model import GraceRUL
from grace_rul.training import (acceptable_checkpoint, evaluate, fit_corrector, fit_pipeline,
                                loader, objectives, repair_trend_loss, seed_all, validate_training_config, validation_score)
from grace_rul.tuning import run_validation_search
from website.backend.service import InferenceService

ROOT = Path(__file__).resolve().parents[1]


class ResearchTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = quick_config({**DEFAULT_CONFIG, "device": "cpu"})
        cls.frame = load_training_subset(ROOT / "data", "FD001")
        cls.fit = cls.frame[cls.frame.unit <= 9]
        cls.val = cls.frame[cls.frame.unit.isin([10, 11])]
        cls.scaler = OperatingRegimeScaler().fit(cls.fit, "FD001", 42)
        seed_all(42)

    def test_residual_recovery_starts_with_identity_on_observed_readings(self):
        model = GraceRUL(self.config).eval()
        values = torch.randn(2, 70, 24)
        values[0, -5:, 4] = float("nan")
        result = model(values)
        finite = torch.isfinite(values[..., 3:])
        torch.testing.assert_close(result["generated"][finite], values[..., 3:][finite], rtol=0, atol=0)
        torch.testing.assert_close(result["repaired"][finite], values[..., 3:][finite], rtol=0, atol=1e-6)
        self.assertTrue(torch.isfinite(result["repaired"]).all())
        self.assertLess(float(result["fault_scores"][finite].max().detach()), .02)

    def test_no_repair_predictor_still_trains_on_corruptions(self):
        config = {**self.config, "train_repair": False, "clean_fraction": 0.,
                  "fault_start_horizon": 0, "maximum_batches": 1}
        batches = []

        def inspect(model, batch, *args, **kwargs):
            batches.append(bool(batch["altered"].any()))
            self.assertFalse(bool(model.repair_enabled))
            return objectives(model, batch, *args, **kwargs)

        def fixed_training_fault(*args, **kwargs):
            if kwargs.get("mode", "train") == "train":
                kwargs["family"] = "bias"
            return WindowDataset(*args, **kwargs)

        with patch("grace_rul.training.objectives", side_effect=inspect), \
                patch("grace_rul.training.WindowDataset", side_effect=fixed_training_fault), \
                patch("grace_rul.model.SensorGenerator.forward", side_effect=AssertionError("Disabled recovery executed")):
            model, _, history, _ = fit_pipeline(self.fit, self.val, "FD001", config, 42, verbose=False)
        self.assertEqual(history.stage.tolist(), ["predictor", "joint"])
        self.assertEqual(batches, [True])
        self.assertFalse(bool(model.repair_enabled))

    def test_trained_single_branch_cannot_use_the_disabled_branch(self):
        x = torch.randn(2, 70, 24)
        valid = torch.ones(2, 70, dtype=torch.bool)
        for branch, inactive in [("local", "global"), ("global", "local")]:
            with self.subTest(branch=branch):
                model = GraceRUL({**self.config, "predictor_branch": branch}).train()
                blocked = model.predictor.transformer if branch == "local" else model.predictor.gru
                with patch.object(blocked, "forward", side_effect=AssertionError("Inactive branch executed")):
                    out = model(x, valid)
                    out["prediction"].sum().backward()
                self.assertTrue(torch.isfinite(out["prediction"]).all())
                self.assertTrue((out[f"{inactive}_prediction"] == 0).all())
                self.assertTrue((out["correction_features"][:, -1 if inactive == "global" else -2] == 0).all())
                inactive_head = model.predictor.global_head if inactive == "global" else model.predictor.local_head
                self.assertIsNone(inactive_head.weight.grad)

    def test_clean_anchor_updates_predictor_without_reconstruction(self):
        config = {**self.config, **{k: 0. for k in self.config if k.endswith("_loss_weight")},
                  "clean_rul_loss_weight": 1., "train_repair": False}
        model = GraceRUL(config).train()
        sample = WindowDataset(self.fit, self.scaler, config, family="bias")[0]
        batch = {k: v[None] for k, v in sample.items() if isinstance(v, torch.Tensor)}
        loss, out = objectives(model, batch, consistency=False)
        loss.backward()
        self.assertGreater(float(out["loss_terms"]["clean_rul"]), 0.)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(model.predictor.head[-1].weight.grad.abs().sum()), 0.)

    def test_fault_metrics_separate_unaffected_windows(self):
        config = {**self.config, "fault_start_horizon": 0}
        clean = WindowDataset(self.val, self.scaler, config, mode="test", family="clean")[0]
        faulty = WindowDataset(self.val, self.scaler, config, mode="test", family="block")[1]
        report, records = evaluate(GraceRUL(config).eval(), [clean, faulty], config)
        self.assertEqual(records.active_fault.tolist(), [False, True])
        self.assertEqual((report["active_fault_N"], report["unaffected_N"]), (1, 1))
        self.assertEqual(report["affected_endpoint_fraction"], .5)
        self.assertAlmostEqual(report["active_fault_RMSE"], abs(records.iloc[1].prediction - records.iloc[1].target))
        with patch("grace_rul.training.evaluate", side_effect=[({"RMSE": 10.}, None),
                ({"RMSE": 5., "active_fault_RMSE": 14.}, None)]):
            score, _, _ = validation_score(None, None, None, config)
        self.assertEqual(score, 12.)

    def test_trend_loss_uses_affected_real_pairs_and_has_gradients(self):
        repaired = torch.tensor([[[99.], [0.], [2.], [5.]]], requires_grad=True)
        reference = torch.tensor([[[0.], [0.], [1.], [3.]]])
        valid = torch.tensor([[False, True, True, True]])
        altered = torch.tensor([[[True], [False], [True], [False]]])
        loss = repair_trend_loss(repaired, reference, altered, valid)
        self.assertEqual(float(loss.detach()), 1.)
        loss.backward()
        self.assertEqual(float(repaired.grad[0, 0]), 0.)
        self.assertGreater(float(repaired.grad.abs().sum()), 0.)
        self.assertEqual(float(repair_trend_loss(repaired, reference, torch.zeros_like(altered), valid).detach()), 0.)

    def test_balanced_sampling_gives_equal_total_weight_to_each_engine(self):
        short = self.fit[self.fit.unit == 1].head(40)
        long = self.fit[self.fit.unit == 2].head(120)
        config = {**self.config, "maximum_windows": None}
        dataset = WindowDataset(pd.concat([short, long]), self.scaler, config)
        sampling = loader(dataset, config, shuffle=True, seed=81).sampler
        units = np.array([engine for engine, _ in dataset.samples])
        self.assertNotEqual(int((units == 0).sum()), int((units == 1).sum()))
        self.assertAlmostEqual(float(sampling.weights[units == 0].sum()), 1.)
        self.assertAlmostEqual(float(sampling.weights[units == 1].sum()), 1.)
        self.assertEqual(list(sampling), list(loader(dataset, config, shuffle=True, seed=81).sampler))

    def test_multiple_validation_prefixes_are_stable_per_engine(self):
        config = {**self.config, "validation_prefixes": 4, "correction_prefixes": 6}
        full = WindowDataset(self.fit, self.scaler, config, mode="validation", seed=91)
        partial = WindowDataset(self.fit[self.fit.unit.isin([2, 6])], self.scaler, config, mode="validation", seed=91)
        endpoints = lambda ds: {(ds.engines[i]["unit"], end) for i, end in ds.samples}
        self.assertEqual(endpoints(partial), {pair for pair in endpoints(full) if pair[0] in {2, 6}})
        self.assertEqual(len(full), 4 * self.fit.unit.nunique())
        calibration = WindowDataset(self.fit, self.scaler, config, mode="calibration", seed=91)
        self.assertEqual(len(calibration), self.fit.unit.nunique())

    def test_checkpoint_constraints_reject_clean_and_repair_regressions(self):
        damaged = {"intact_repair_RMSE": .1, "reconstruction_RMSE": 1.}
        self.assertTrue(acceptable_checkpoint({"RMSE": 10.05}, damaged, 10., self.config))
        self.assertFalse(acceptable_checkpoint({"RMSE": 10.2}, damaged, 10., self.config))
        self.assertFalse(acceptable_checkpoint({"RMSE": 5.}, {**damaged, "intact_repair_RMSE": .2}, 10., self.config))
        self.assertFalse(acceptable_checkpoint({"RMSE": 5.}, {**damaged, "reconstruction_RMSE": 2.}, 10., self.config))

    def test_rejected_repair_preserves_pretrained_predictor_and_disables_repair(self):
        config = {**self.config, "maximum_batches": 1, "adversarial_loss_weight": 0.}
        snapshot = {}

        def validate(model, *args, **options):
            if not snapshot:
                snapshot.update({name: tensor.clone() for name, tensor in model.state_dict().items()})
                return 10., {"RMSE": 10.}, {"RMSE": 10., "reconstruction_RMSE": 1.}
            # Better macro RMSE alone is insufficient if clean accuracy regresses.
            return 8., {"RMSE": 11.}, {"RMSE": 5., "reconstruction_RMSE": 1.}

        with patch("grace_rul.training.validation_score", side_effect=validate):
            model, _, history, score = fit_pipeline(self.fit, self.val, "FD001", config, 42, verbose=False)
        self.assertEqual(score, 10.)
        self.assertFalse(bool(model.repair_enabled))
        self.assertEqual(history.checkpoint_accepted.tolist(), [True, False, False])
        for name, weights in snapshot.items():
            if name.startswith(("predictor.", "generator.", "corrector.")):
                torch.testing.assert_close(model.state_dict()[name], weights, rtol=0, atol=0)
        self.assertIsNotNone(model.detector_audit)

    def test_scheduler_reduction_survives_interruption(self):
        config = {**self.config, "train_repair": False, "train_corrupted_predictor": False, "predictor_epochs": 4,
                  "maximum_batches": 1, "lr_patience": 1}
        flat_score = lambda *args, **kwargs: (10., {"RMSE": 10.}, {"RMSE": 10., "reconstruction_RMSE": 1.})
        with patch("grace_rul.training.validation_score", side_effect=flat_score):
            expected, _, full_history, _ = fit_pipeline(self.fit, self.val, "FD001", config, 42, verbose=False)
        self.assertEqual(full_history.learning_rate.tolist(), [.001, .001, .001, .0005])
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "pipeline.pt"
            calls = 0

            def interrupt(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 4:
                    raise KeyboardInterrupt
                return flat_score()

            with patch("grace_rul.training.validation_score", side_effect=interrupt), self.assertRaises(KeyboardInterrupt):
                fit_pipeline(self.fit, self.val, "FD001", config, 42, verbose=False, checkpoint_path=checkpoint)
            saved = torch.load(checkpoint, weights_only=True)
            self.assertEqual(saved["optimizer"]["param_groups"][0]["lr"], .0005)
            with patch("grace_rul.training.validation_score", side_effect=flat_score):
                resumed, _, history, _ = fit_pipeline(self.fit, self.val, "FD001", config, 42, verbose=False, checkpoint_path=checkpoint)
            pd.testing.assert_frame_equal(history.drop(columns="duration_seconds"), full_history.drop(columns="duration_seconds"))
            for name, weights in expected.state_dict().items():
                torch.testing.assert_close(resumed.state_dict()[name], weights, rtol=0, atol=0)

    def test_corrector_search_selects_admissible_bound(self):
        config = {**self.config, "correction_bounds": [5., 30.], "correction_epochs": 1}
        model = GraceRUL(config).eval()
        model.scaler = self.scaler

        def temporary(fit, val, subset, config, seed, **kwargs):
            return GraceRUL(config).eval(), OperatingRegimeScaler().fit(fit, subset, 42), pd.DataFrame(), 0.

        scores = [(10., {"RMSE": 10.}, {"RMSE": 10.}),
                  (8., {"RMSE": 10.05}, {"RMSE": 5.95}),
                  (7., {"RMSE": 10.2}, {"RMSE": 3.8})]
        with patch("grace_rul.training.fit_pipeline", side_effect=temporary), \
                patch("grace_rul.training.validation_score", side_effect=scores), contextlib.redirect_stdout(io.StringIO()):
            history, _, _ = fit_corrector(model, self.fit, self.val, "FD001", config, 42)
        self.assertEqual(float(model.correction_bound), 5.)
        self.assertEqual(history.checkpoint_accepted.tolist(), [True, True, False])
        loaded = GraceRUL(config)
        loaded.load_state_dict(model.state_dict())
        self.assertEqual(float(loaded.correction_bound), 5.)

    def test_corrector_cannot_compound_the_clean_regression_tolerance(self):
        config = {**self.config, "correction_bounds": [5.], "correction_epochs": 1}
        model = GraceRUL(config).eval()
        model.scaler, model.pretrained_clean_rmse = self.scaler, 10.

        def temporary(fit, val, subset, cfg, seed, **kwargs):
            return GraceRUL(cfg).eval(), OperatingRegimeScaler().fit(fit, subset, 42), pd.DataFrame(), 0.

        scores = [(10., {"RMSE": 10.05}, {"RMSE": 9.95}),
                  (8., {"RMSE": 10.12}, {"RMSE": 5.88})]
        with patch("grace_rul.training.fit_pipeline", side_effect=temporary), \
                patch("grace_rul.training.validation_score", side_effect=scores), contextlib.redirect_stdout(io.StringIO()):
            history, _, _ = fit_corrector(model, self.fit, self.val, "FD001", config, 42)
        self.assertEqual(history.checkpoint_accepted.tolist(), [True, False])
        self.assertTrue((model.corrector[-1].weight == 0).all())

    def test_no_correction_ablation_reuses_exact_frozen_upstream_weights(self):
        config = {**self.config, "maximum_batches": 1, "run_matched_benchmarks": True,
                  "benchmark_variants": ["grace_without_correction"]}
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            with patch("grace_rul.benchmark.fit_pipeline", side_effect=AssertionError("Unchanged upstream model retrained")), \
                    patch("grace_rul.benchmark.fit_corrector", side_effect=AssertionError("Correction ablation fitted a corrector")):
                path, tables = run_experiment(ROOT, config, Path(directory))
            _, loaded = load_bundle(path)
            item = loaded["FD001"]
            _, test, truths = load_subset(ROOT / "data", "FD001")
            dataset = WindowDataset(test[test.unit <= 3], item["scaler"], config, mode="test", family="clean", truths=truths)
            _, expected = evaluate(item["model"], dataset, config, use_correction=False)
            records = pd.read_csv(path.parent / "benchmark_predictions.csv.gz")
            measured = records[(records.variant == "grace_without_correction") & (records.family == "clean")].sort_values("unit")
            np.testing.assert_allclose(measured.prediction, expected.sort_values("unit").prediction)
            self.assertTrue(bool(tables["benchmark_training_audit"].iloc[0].reused_identical_weights))
            self.assertEqual(int(tables["benchmark_training_audit"].iloc[0].upstream_epochs), 0)

    def test_tuning_uses_only_training_files_and_common_partitions(self):
        calls = []

        def trial(fit, val, subset, config, seed, **kwargs):
            calls.append((set(fit.unit), set(val.unit), seed))
            return GraceRUL(config), None, pd.DataFrame([{}]), 4. if config["trend_loss_weight"] == .2 else 5.

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            # No official test/RUL files exist in this deliberately restricted project.
            (root / "data" / "train_FD001.txt").symlink_to(ROOT / "data" / "train_FD001.txt")
            with patch("grace_rul.tuning.fit_pipeline", side_effect=trial):
                selected, table = run_validation_search(root, self.config, root / "tuning",
                    candidates=[("default", {}), ("trend", {"trend_loss_weight": .2})])
            self.assertEqual(selected, {"FD001": {"trend_loss_weight": .2}})
            self.assertEqual(calls[0], calls[1])
            audit = json.loads((root / "tuning" / "search_manifest.json").read_text())["audit"]["FD001"]
            self.assertFalse(set(audit["partitions"]["calibration"]) & (calls[0][0] | calls[0][1]))
            self.assertEqual(table.validation_RMSE.min(), 4.)
            with self.assertRaises(ValueError):
                run_validation_search(root, self.config, root / "invalid", [("unsafe", {"train_rul_cap": 50.})])

    def test_comparator_backpropagation_is_finite_on_short_histories(self):
        valid = torch.zeros(2, 70, dtype=torch.bool)
        valid[0, -2:], valid[1, -8:] = True, True
        values = torch.randn(2, 70, 24)
        for family in ["cnn_gru", "cnn_lstm", "transformer"]:
            with self.subTest(family=family):
                model = GraceRUL({**self.config, "predictor_family": family}).train()
                out = model(values, valid)
                out["prediction"].sum().backward()
                self.assertTrue(torch.isfinite(out["prediction"]).all())
                gradients = [p.grad for p in model.predictor.parameters() if p.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(g).all() for g in gradients))

    def test_paired_bootstrap_retains_corruption_replicates_and_quick_scope(self):
        grace = np.array([[1., 4.], [4., 9.]])
        seeds, engines = np.array([[0, 1], [1, 1]]), np.array([[0, 1], [1, 1]])
        g, b = paired_bootstrap(grace, 4 * grace, seeds, engines)
        np.testing.assert_allclose(b, 2 * g)
        self.assertEqual(g[1], 3.)
        with self.assertRaises(ValueError):
            paired_bootstrap(grace, grace[:, :1], seeds, engines)
        errors = {}
        for family in ["clean", "bias"]:
            for variant, error in [("GRACE_RUL", 1.), ("baseline", 4.)]:
                errors[variant, "FD001", family, .0 if family == "clean" else .2, "capped_125"] = {
                    42: {0: np.array([error, error]), 1: np.array([error, error])}}
        comparison, checks = _comparison_tables(errors, self.config)
        self.assertTrue((comparison.delta_RMSE == -1.).all())
        self.assertAlmostEqual(float(checks.iloc[0].relative_improvement), .5)
        self.assertFalse(bool(checks.iloc[0].acceptance_target_met))
        self.assertFalse(bool(checks.iloc[0].full_protocol_complete))

    def test_active_comparisons_pair_only_commonly_damaged_windows(self):
        errors, masks = {}, {}
        for variant, squared, mask in [("GRACE_RUL", [1., 999.], [True, True]),
                                       ("baseline", [4., 0.], [True, False])]:
            key = (variant, "FD001", "block", .2, "capped_125")
            errors[key] = {42: {0: np.array(squared)}}
            masks[key] = {42: {0: np.array(mask)}}
        comparison, checks = _active_comparison_tables(errors, masks, self.config)
        self.assertEqual(float(comparison.iloc[0].GRACE_RMSE), 1.)
        self.assertEqual(float(comparison.iloc[0].comparator_RMSE), 2.)
        self.assertEqual(float(comparison.iloc[0].common_affected_endpoint_fraction), .5)
        self.assertAlmostEqual(float(checks.iloc[0].active_relative_improvement), .5)
        self.assertFalse(bool(checks.iloc[0].active_fault_acceptance))
        masks[("baseline", "FD001", "block", .2, "capped_125")][42][0][:] = False
        comparison, checks = _active_comparison_tables(errors, masks, self.config)
        self.assertIsNone(comparison.iloc[0].GRACE_RMSE)
        self.assertFalse(bool(checks.iloc[0].active_fault_acceptance))

    def test_invalid_seed_and_numeric_settings_are_rejected(self):
        for overrides in [{"seeds": [42, 42]}, {"bootstrap_samples": 0},
                          {"minimum_learning_rate": float("nan")}, {"detector_positive_weight_cap": .5},
                          {"fault_start_horizon": -1}, {"fault_start_horizon": True},
                          {"drift_ramp_cycles": 0}, {"fault_noise_autocorrelation": 1.},
                          {"fault_transition_gain": -1.}, {"intermittent_mean_burst": 1}]:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                validate_training_config({**self.config, **overrides})

    def test_reporting_switches_reuse_epochs_but_training_changes_do_not(self):
        original = pipeline_identity(self.fit, self.val, "FD001", self.config, 42)
        # Report keys in a checkpoint may differ from the current report choices.
        legacy = {**original, "config": {**original["config"],
                  "run_matched_benchmarks": True, "run_extended_evaluation": True,
                  "benchmark_variants": DEFAULT_CONFIG["benchmark_variants"]}}
        reduced = {**self.config, "run_extended_evaluation": False,
                   "run_matched_benchmarks": False, "benchmark_variants": []}
        identity = pipeline_identity(self.fit, self.val, "FD001", reduced, 42)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pipeline.pt"
            save_checkpoint(path, {"identity": legacy, "next_epoch": 3})
            self.assertEqual(load_checkpoint(path, identity)["next_epoch"], 3)
            changed = pipeline_identity(self.fit, self.val, "FD001",
                                        {**reduced, "window_stride": 1}, 42)
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_checkpoint(path, changed)

    def test_old_corruption_protocol_checkpoints_require_fresh_training(self):
        identity = pipeline_identity(self.fit, self.val, "FD001", self.config, 42)
        legacy = {**identity, "version": 2}
        legacy.pop("corruption_protocol")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pipeline.pt"
            save_checkpoint(path, {"identity": legacy, "next_epoch": 3})
            with self.assertRaisesRegex(ValueError, "corruption protocol"):
                load_checkpoint(path, identity)

    def test_skipping_extended_reports_preserves_exported_weights(self):
        config = {**self.config, "maximum_batches": 1}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "quick"
            with patch("grace_rul.benchmark.run_matched_benchmarks", side_effect=AssertionError("Unexpected benchmark training")):
                path, full_tables = run_experiment(ROOT, config, output)
                before, _ = load_bundle(path)
                with patch("grace_rul.training.critic_update", side_effect=AssertionError("Cached model retrained")):
                    path, reduced_tables = run_experiment(ROOT, {**config, "run_extended_evaluation": False}, output)
                after, _ = load_bundle(path)
            for report in ["robustness_summary", "ablation_metrics"]:
                self.assertFalse(full_tables[report].empty)
                self.assertNotIn(report, reduced_tables)
            for name, weights in before["models"]["FD001"]["state_dict"].items():
                torch.testing.assert_close(after["models"]["FD001"]["state_dict"][name], weights, rtol=0, atol=0)
            for name, frame in reduced_tables.items():
                pd.testing.assert_frame_equal(full_tables[name], frame)

    def test_trained_comparisons_share_engines_targets_and_corruption_draws(self):
        config = {**self.config, "maximum_batches": 1, "tune_before_training": True,
                  "run_matched_benchmarks": True,
                  "benchmark_variants": ["paper_A_cnn_gru", "paper_C_transformer_corrected"]}
        evaluated = []

        def inspect_dataset(model, dataset, config, **kwargs):
            evaluated.append(dataset)
            return evaluate(model, dataset, config, **kwargs)

        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            with patch("grace_rul.tuning.SEARCH_CANDIDATES", [("default", {}), ("without_adversarial", {"adversarial_loss_weight": 0.})]), \
                    patch("grace_rul.tuning.comparison_candidates", return_value=[("default", {}), ("more_clean", {"clean_fraction": .5})]), \
                    patch("grace_rul.benchmark.evaluate", side_effect=inspect_dataset):
                path, tables = run_experiment(ROOT, config, Path(directory) / "quick")
            bundle, loaded = load_bundle(path)
            self.assertEqual(set(tables["benchmark_training_audit"].variant), set(config["benchmark_variants"]))
            self.assertTrue((tables["benchmark_training_audit"].upstream_epochs == 2).all())
            self.assertTrue(tables["benchmark_training_audit"].independent_tuning.all())
            self.assertTrue(tables["benchmark_training_audit"].corruption_trained_predictor.all())
            self.assertFalse(tables["superiority_checks"].acceptance_target_met.any())
            predictions = pd.read_csv(path.parent / "benchmark_predictions.csv.gz")
            keys = ["unit", "cycle", "family", "severity", "replicate"]
            for variant in config["benchmark_variants"]:
                primary = predictions[predictions.variant == "GRACE_RUL"].sort_values(keys)
                comparator = predictions[predictions.variant == variant].sort_values(keys)
                pd.testing.assert_frame_equal(primary[keys + ["target", "raw_target"]].reset_index(drop=True),
                                              comparator[keys + ["target", "raw_target"]].reset_index(drop=True))
            self.assertEqual(len(evaluated), 12)
            for i in range(0, len(evaluated), 3):
                self.assertIs(evaluated[i], evaluated[i + 1])
                self.assertIs(evaluated[i], evaluated[i + 2])
            self.assertEqual(set(predictions.variant), {"GRACE_RUL", *config["benchmark_variants"]})
            with patch.dict("os.environ", {"GRACE_MODEL_PATH": str(path)}):
                service = InferenceService()
                status = service.model_report()
                detector = service.loaded["FD001"]["model"].detector
                service.loaded["FD001"]["model"].repair_enabled.fill_(False)
                sample = self.val[self.val.unit == 10].copy()
                sample["subset"] = "FD001"
                # Updated detector scores remain available on a bypassed path.
                with torch.no_grad():
                    detector.head.weight.fill_(.01)
                with patch("website.backend.service.explain_window", return_value={"targets": []}) as explain:
                    service.explain(sample, "FD001", 10, sensor=3)
                self.assertEqual(explain.call_args.kwargs["target_sensor"], 2)
                # A constant initialization is not finite-fault evidence.
                with torch.no_grad():
                    detector.head.weight.zero_()
                with patch("website.backend.service.explain_window", return_value={"targets": []}) as explain:
                    report = service.explain(sample, "FD001", 10, sensor=3)
                self.assertIsNone(explain.call_args.kwargs["target_sensor"])
                self.assertIn("final RUL only", report["note"])
            self.assertEqual(status["selected_correction_bounds"]["FD001"], float(loaded["FD001"]["model"].correction_bound))
            self.assertEqual(status["selected_repair"]["FD001"], bool(loaded["FD001"]["model"].repair_enabled))
            self.assertTrue(status["tables"]["benchmark_summary"])
            bundle["models"]["FD001"]["state_dict"]["correction_bound"].fill_(float("nan"))
            invalid = Path(directory) / "invalid.pt"
            torch.save(bundle, invalid)
            with self.assertRaisesRegex(ValueError, "correction bound"):
                load_bundle(invalid)


if __name__ == "__main__":
    unittest.main()
