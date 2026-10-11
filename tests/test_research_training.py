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
from grace_rul.benchmark import _comparison_tables, paired_bootstrap
from grace_rul.config import DEFAULT_CONFIG, quick_config
from grace_rul.data import OperatingRegimeScaler, WindowDataset, load_training_subset
from grace_rul.model import GraceRUL
from grace_rul.training import (acceptable_checkpoint, evaluate, fit_corrector, fit_pipeline,
                                loader, repair_trend_loss, seed_all, validate_training_config)
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
        self.assertTrue(acceptable_checkpoint({"RMSE": 10.4}, damaged, 10., self.config))
        self.assertFalse(acceptable_checkpoint({"RMSE": 10.6}, damaged, 10., self.config))
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
            if name != "repair_enabled":
                torch.testing.assert_close(model.state_dict()[name], weights, rtol=0, atol=0)

    def test_scheduler_reduction_survives_interruption(self):
        config = {**self.config, "train_repair": False, "predictor_epochs": 4,
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
                  (8., {"RMSE": 10.4}, {"RMSE": 5.6}),
                  (7., {"RMSE": 10.6}, {"RMSE": 3.4})]
        with patch("grace_rul.training.fit_pipeline", side_effect=temporary), \
                patch("grace_rul.training.validation_score", side_effect=scores), contextlib.redirect_stdout(io.StringIO()):
            history, _, _ = fit_corrector(model, self.fit, self.val, "FD001", config, 42)
        self.assertEqual(float(model.correction_bound), 5.)
        self.assertEqual(history.checkpoint_accepted.tolist(), [True, True, False])
        loaded = GraceRUL(config)
        loaded.load_state_dict(model.state_dict())
        self.assertEqual(float(loaded.correction_bound), 5.)

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

    def test_invalid_seed_and_numeric_settings_are_rejected(self):
        for overrides in [{"seeds": [42, 42]}, {"bootstrap_samples": 0},
                          {"minimum_learning_rate": float("nan")}, {"detector_positive_weight_cap": .5}]:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                validate_training_config({**self.config, **overrides})

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
                    patch("grace_rul.benchmark.evaluate", side_effect=inspect_dataset):
                path, tables = run_experiment(ROOT, config, Path(directory) / "quick")
            bundle, loaded = load_bundle(path)
            self.assertEqual(set(tables["benchmark_training_audit"].variant), set(config["benchmark_variants"]))
            self.assertTrue((tables["benchmark_training_audit"].upstream_epochs == 1).all())
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
                service.loaded["FD001"]["model"].repair_enabled.fill_(False)
                sample = self.val[self.val.unit == 10].copy()
                sample["subset"] = "FD001"
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
