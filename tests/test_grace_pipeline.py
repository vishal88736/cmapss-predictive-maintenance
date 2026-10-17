"""Regression checks for leakage, corruption, recovery, and the website contract."""

import asyncio
import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import httpx

from grace_rul.artifacts import load_bundle, run_experiment
from grace_rul.config import DEFAULT_CONFIG, quick_config
from grace_rul.data import OperatingRegimeScaler, WindowDataset, corrupt_trajectory, load_subset, make_window, split_engines
from grace_rul.model import GraceRUL, TemporalCritic, masked_rnn
from grace_rul.training import critic_update, device_for, evaluate, fit_corrector, fit_pipeline, interval_bounds, objectives, seed_all
from website.backend import app as web
from website.backend.inference import InputError, parse_csv
from website.backend.service import InferenceService

ROOT = Path(__file__).resolve().parents[1]


class InProcessClient:
    """Exercise ASGI routing and multipart handling without thread wakeup sockets.

    Only test scheduling changes: production endpoints retain their worker pool.
    """
    def __init__(self, app):
        self.app = app

    def request(self, method, path, **kwargs):
        async def inline(function, *args, **options):
            return function(*args)

        async def send():
            with patch("anyio.to_thread.run_sync", inline):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://test") as client:
                    return await client.request(method, path, **kwargs)
        return asyncio.run(send())

    def get(self, path, **kwargs):
        return self.request("GET", path, **kwargs)

    def post(self, path, **kwargs):
        return self.request("POST", path, **kwargs)


class PipelineInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = quick_config({**DEFAULT_CONFIG, "device": "cpu"})
        seed_all(42)
        cls.train, cls.test, cls.truth = load_subset(ROOT / "data", "FD001")

    def test_engine_partitions_and_scaler_have_no_held_out_engines(self):
        groups = split_engines(self.train, 42)
        sets = [set(v) for v in groups.values()]
        self.assertEqual(len(set.union(*sets)), self.train.unit.nunique())
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        scaler = OperatingRegimeScaler().fit(self.train[self.train.unit.isin(groups["fit"])], "FD001", 42)
        self.assertEqual(set(scaler.state["fit_units"]), sets[0])

    def test_outer_80_20_and_calibration_reservation_for_all_subsets(self):
        for subset in DEFAULT_CONFIG["subsets"]:
            train, _, _ = load_subset(ROOT / "data", subset)
            with self.subTest(subset=subset):
                groups = split_engines(train, 42)
                n = train.unit.nunique()
                self.assertEqual(len(groups["validation"]), int(np.ceil(.2 * n)))
                pool = set(groups["fit"]) | set(groups["calibration"])
                self.assertEqual(len(pool), n - len(groups["validation"]))
                self.assertEqual(len(groups["calibration"]), max(2, int(np.ceil(.1 * len(pool)))))
                self.assertFalse(pool & set(groups["validation"]))
                self.assertFalse(set(groups["fit"]) & set(groups["calibration"]))
                self.assertEqual(groups, split_engines(train, 42))
        for validation, calibration in [(0., .1), (.2, 0.), (1., .1), (.2, 1.)]:
            with self.assertRaises(ValueError):
                split_engines(self.train, 42, validation_fraction=validation, calibration_fraction_of_training=calibration)

    def test_explicit_cuda_never_silently_falls_back(self):
        self.assertEqual(DEFAULT_CONFIG["device"], "cuda")
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "cannot access a CUDA GPU"):
                device_for({"device": "cuda"})
            self.assertEqual(device_for({"device": "cpu"}).type, "cpu")
            self.assertEqual(device_for({"device": "auto"}).type, "cpu")
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "failed_run"
                with self.assertRaises(RuntimeError):
                    run_experiment(ROOT, DEFAULT_CONFIG, output)
                self.assertFalse(output.exists())
        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.device_count", return_value=1):
            self.assertEqual(device_for({"device": "cuda"}).type, "cuda")
            with self.assertRaisesRegex(ValueError, "index"):
                device_for({"device": "cuda:1"})

    def test_all_subsets_and_multi_regime_normalization(self):
        counts = [(100, 100), (260, 259), (100, 100), (249, 248)]
        for subset, expected in zip(DEFAULT_CONFIG["subsets"], counts):
            with self.subTest(subset=subset):
                train, test, truth = load_subset(ROOT / "data", subset)
                self.assertEqual((train.unit.nunique(), test.unit.nunique()), expected)
                fit = train[train.unit <= 12]
                scaler = OperatingRegimeScaler().fit(fit, subset, 42)
                normalized, regimes = scaler.transform(test.head(40))
                self.assertTrue(np.isfinite(normalized).all())
                self.assertEqual(len(scaler.state["centers"]), 6 if subset in {"FD002", "FD004"} else 1)
                self.assertTrue((regimes >= 0).all())

    def test_near_constant_sensor_scaling_uses_fitting_resolution(self):
        fit = self.train[self.train.unit == 1].head(100).copy()
        fit["sensor_10"] = 1.30
        fit.loc[fit.index[-1], "sensor_10"] = 1.31
        scaler = OperatingRegimeScaler().fit(fit, "FD001", 42)
        held = fit.iloc[:1].copy()
        held["sensor_10"] = 1.31
        normalized, _ = scaler.transform(held)
        self.assertAlmostEqual(scaler.state["sensor_scale_floor"][9], .01)
        self.assertLess(abs(float(normalized[0, 12])), 1.)
        self.assertEqual(scaler.state["fit_units"], [1])

    def test_finite_bias_and_drift_do_not_become_missingness(self):
        reference = np.random.default_rng(8).normal(size=(120, 24)).astype(np.float32)
        for family in ["bias", "drift", "noise"]:
            x, altered = corrupt_trajectory(reference, np.random.default_rng(42), family, .3)
            self.assertTrue(np.isfinite(x).all())
            self.assertTrue(altered.any())
            self.assertTrue(np.array_equal(x[:, :3], reference[:, :3]))
            self.assertTrue(np.array_equal(x[:, 3:][~altered], reference[:, 3:][~altered]))
        x, altered = corrupt_trajectory(reference, np.random.default_rng(42), "outage", .3)
        self.assertTrue(np.array_equal(np.isnan(x[:, 3:]), altered))

    def test_drift_labels_only_measurements_that_actually_change(self):
        reference = np.zeros((100, 24), dtype=np.float32)
        reference[:, 3::2] = 1.
        damaged, altered = corrupt_trajectory(reference, np.random.default_rng(42), "drift", .5)
        self.assertTrue(altered.any())
        np.testing.assert_array_equal(altered, damaged[:, 3:] != reference[:, 3:])
        # Calibration drift is additive: zero standardized readings can drift.
        self.assertTrue(altered[:, 1::2].any())

    def test_explicit_mixed_fault_evaluation_is_reproducible(self):
        reference = np.random.default_rng(8).normal(size=(120, 24)).astype(np.float32)
        first, labels = corrupt_trajectory(reference, np.random.default_rng(42), "mixed", .4)
        repeated, repeated_labels = corrupt_trajectory(reference, np.random.default_rng(42), "mixed", .4)
        np.testing.assert_array_equal(first, repeated)
        np.testing.assert_array_equal(labels, repeated_labels)
        np.testing.assert_array_equal(labels, first[:, 3:] != reference[:, 3:])
        self.assertTrue(labels.any())
        np.testing.assert_array_equal(first[:, :3], reference[:, :3])

    def test_consistency_teacher_disables_dropout_and_restores_modes(self):
        model = GraceRUL(self.config).train()
        valid = torch.ones(2, 70, dtype=torch.bool)
        batch = {"x": torch.randn(2, 70, 24), "valid": valid,
                 "altered": torch.zeros(2, 70, 21, dtype=torch.bool), "target": torch.ones(2) * 70}
        batch["reference"] = batch["x"].clone()
        calls = []

        def record_mode(module, inputs, output):
            calls.append((module.training, module.head[2].training, output["base"].detach().clone()))

        hook = model.predictor.register_forward_hook(record_mode)
        try:
            loss, _ = objectives(model, batch)
            loss.backward()
            objectives(model, batch)
        finally:
            hook.remove()
        self.assertTrue(calls[0][0] and calls[0][1])
        self.assertFalse(calls[1][0] or calls[1][1])
        torch.testing.assert_close(calls[1][2], calls[3][2], rtol=0, atol=0)
        self.assertTrue(all(module.training for module in model.modules()))

    def test_corrector_ignores_fold_specific_hidden_coordinates(self):
        model = GraceRUL(self.config).eval()
        torch.nn.init.normal_(model.corrector[-1].weight)
        x = torch.randn(2, 70, 24)
        valid = torch.ones(2, 70, dtype=torch.bool)
        with torch.no_grad():
            before = model(x, valid)

        def permute_hidden_coordinates(module, inputs, output):
            return {**output, "features": output["features"].flip(-1)}

        hook = model.predictor.register_forward_hook(permute_hidden_coordinates)
        try:
            with torch.no_grad():
                after = model(x, valid)
        finally:
            hook.remove()
        self.assertFalse(torch.equal(before["features"], after["features"]))
        torch.testing.assert_close(before["prediction"], after["prediction"], rtol=0, atol=0)

    def test_corrector_retains_zero_candidate_when_validation_worsens(self):
        fit, val = self.train[self.train.unit <= 9], self.train[self.train.unit.isin([10, 11])]
        model = GraceRUL(self.config).eval()
        model.scaler = OperatingRegimeScaler().fit(fit, "FD001", 42)

        def untrained_fold(train, validation, subset, config, seed, **options):
            return GraceRUL(config).eval(), OperatingRegimeScaler().fit(train, subset, seed), pd.DataFrame(), 0.

        scores = [(5., {"RMSE": 4.}, {"RMSE": 6.}), (10., {"RMSE": 9.}, {"RMSE": 11.}),
                  (9., {"RMSE": 8.}, {"RMSE": 10.})]
        with patch("grace_rul.training.fit_pipeline", side_effect=untrained_fold), \
                patch("grace_rul.training.validation_score", side_effect=scores), contextlib.redirect_stdout(io.StringIO()):
            history, _, _ = fit_corrector(model, fit, val, "FD001", self.config, 42)
        self.assertEqual(history.validation_macro_RMSE.min(), 5.)
        self.assertTrue(bool((model.corrector[-1].weight == 0).all()))
        self.assertTrue(bool((model.corrector[-1].bias == 0).all()))

    def test_invalid_epoch_budget_fails_before_creating_run(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "invalid"
            with self.assertRaisesRegex(ValueError, "joint_epochs"):
                run_experiment(ROOT, {**self.config, "joint_epochs": 0}, output)
            self.assertFalse(output.exists())

    def test_joint_stage_keeps_repair_checkpoint_if_all_updates_worsen(self):
        config = {**self.config, "maximum_batches": 1, "adversarial_warmup": 10}
        fit, val = self.train[self.train.unit <= 6], self.train[self.train.unit.isin([7, 8])]
        snapshots = []

        def validation(model, *args, **options):
            snapshots.append({name: weights.clone() for name, weights in model.state_dict().items()})
            score = [5., 4., 6.][len(snapshots) - 1]
            return score, {"RMSE": score}, {"RMSE": score, "reconstruction_RMSE": 1.}

        with patch("grace_rul.training.validation_score", side_effect=validation):
            model, _, _, score = fit_pipeline(fit, val, "FD001", config, 42, verbose=False)
        self.assertEqual(score, 4.)
        for name, weights in model.state_dict().items():
            torch.testing.assert_close(weights, snapshots[1][name], rtol=0, atol=0)

    def test_windows_exclude_future_and_padding_is_ignored(self):
        values = np.random.default_rng(7).normal(size=(100, 24)).astype(np.float32)
        x, valid = make_window(values, 8, 70)
        future_changed = values.copy(); future_changed[9:] += 10000
        self.assertTrue(np.array_equal(x, make_window(future_changed, 8, 70)[0]))
        dirty_padding = x.copy(); dirty_padding[~valid] = 100000
        model = GraceRUL(self.config).eval()
        with torch.no_grad():
            a = model(torch.tensor(x[None]), torch.tensor(valid[None]))
            b = model(torch.tensor(dirty_padding[None]), torch.tensor(valid[None]))
        torch.testing.assert_close(a["prediction"], b["prediction"])
        torch.testing.assert_close(a["repaired"][:, valid], b["repaired"][:, valid])

    def test_persistent_corruption_and_finite_recovery_under_complete_outage(self):
        fit = self.train[self.train.unit == 1]
        scaler = OperatingRegimeScaler().fit(fit, "FD001", 42)
        dataset = WindowDataset(fit, scaler, self.config, family="bias")
        first, second = dataset[3], dataset[4]
        overlap = min(int(first["valid"].sum()), int(second["valid"].sum()) - 5)
        torch.testing.assert_close(first["x"][-overlap:], second["x"][-overlap - 5:-5])
        x = first["x"].clone(); x[first["valid"], 3:] = float("nan")
        model = GraceRUL(self.config).eval()
        with torch.no_grad():
            out = model(x[None], first["valid"][None])
        self.assertTrue(torch.isfinite(out["prediction"]).all())
        self.assertTrue(torch.isfinite(out["repaired"]).all())
        self.assertTrue(bool((out["weights"][0, first["valid"]] == 1).all()))
        self.assertLessEqual(float(out["delta"].abs().max()), self.config["correction_bound"])

    def test_gradient_penalty_supports_second_derivatives(self):
        critic = TemporalCritic(self.config)
        optimizer = torch.optim.Adam(critic.parameters(), lr=.001)
        valid = torch.ones(2, 12, dtype=torch.bool); valid[0, :4] = False
        batch = {"reference": torch.randn(2, 12, 24), "valid": valid}
        result = {"repaired": torch.randn(2, 12, 21)}
        backend_flags = (torch.backends.cudnn.enabled, torch.backends.mkldnn.enabled)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loss = critic_update(critic, optimizer, result, batch, self.config)
        self.assertTrue(np.isfinite(loss))
        self.assertFalse(any("TF32 acceleration on top of oneDNN" in str(w.message) for w in caught))
        self.assertEqual((torch.backends.cudnn.enabled, torch.backends.mkldnn.enabled), backend_flags)
        with patch.object(critic, "forward", side_effect=RuntimeError("critic failed")):
            with self.assertRaisesRegex(RuntimeError, "critic failed"):
                critic_update(critic, optimizer, result, batch, self.config)
        self.assertEqual((torch.backends.cudnn.enabled, torch.backends.mkldnn.enabled), backend_flags)

    def test_gradient_penalty_sub_batch_preserves_full_wasserstein_batch(self):
        critic = TemporalCritic(self.config)
        batch = {"reference": torch.randn(4, 12, 24), "valid": torch.ones(4, 12, dtype=torch.bool)}
        result = {"repaired": torch.randn(4, 12, 21)}
        sizes = []
        hook = critic.register_forward_pre_hook(lambda module, inputs: sizes.append(len(inputs[0])))
        try:
            loss = critic_update(critic, torch.optim.Adam(critic.parameters()), result, batch,
                                 {**self.config, "gradient_penalty_batch_size": 2})
        finally:
            hook.remove()
        self.assertTrue(np.isfinite(loss))
        self.assertEqual(sizes, [2, 4, 4])

    def test_vectorized_rnn_preserves_ragged_outputs_and_gradients(self):
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

        valid = torch.tensor([[False, False, True, True, True, True],
                              [True, False, True, False, True, True],
                              [False, False, False, False, False, True]])
        for kind in [torch.nn.GRU, torch.nn.LSTM]:
            with self.subTest(rnn=kind.__name__):
                rnn = kind(4, 5, batch_first=True, bidirectional=True)
                x = torch.randn(3, 6, 4).masked_fill(~valid.unsqueeze(-1), float("nan")).requires_grad_()
                actual = masked_rnn(rnn, x, valid)
                compact = pad_sequence([row[mask] for row, mask in zip(x, valid)], batch_first=True)
                packed = pack_padded_sequence(compact, valid.sum(1), batch_first=True, enforce_sorted=False)
                packed_output, _ = rnn(packed)
                unpacked, _ = pad_packed_sequence(packed_output, batch_first=True)
                expected = x.new_zeros(3, 6, 10)
                for i, mask in enumerate(valid):
                    expected[i, mask] = unpacked[i, :int(mask.sum())]
                torch.testing.assert_close(actual, expected)
                variables = (x, *rnn.parameters())
                actual_grad = torch.autograd.grad(actual.square().sum(), variables)
                expected_grad = torch.autograd.grad(expected.square().sum(), variables)
                for a, b in zip(actual_grad, expected_grad):
                    torch.testing.assert_close(a, b)

    def test_epoch_checkpoint_resumes_without_repeating_updates(self):
        config = {**self.config, "predictor_epochs": 3, "generator_epochs": 3, "joint_epochs": 3,
                  "maximum_batches": 1, "progress_interval_seconds": 0}
        fit = self.train[self.train.unit <= 6]
        val = self.train[self.train.unit.isin([7, 8])]
        expected, _, expected_history, expected_score = fit_pipeline(fit, val, "FD001", config, 42, verbose=False)
        for stage_index, stage in enumerate(["predictor", "generator", "joint"]):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                checkpoint = Path(directory) / "pipeline.pt"
                calls = 0

                def interrupt_validation(*args, **kwargs):
                    nonlocal calls
                    calls += 1
                    if calls == 5 + 6 * stage_index:
                        raise KeyboardInterrupt("Interrupted in the third epoch")
                    return evaluate(*args, **kwargs)

                with patch("grace_rul.training.evaluate", side_effect=interrupt_validation):
                    with self.assertRaises(KeyboardInterrupt):
                        fit_pipeline(fit, val, "FD001", config, 42, verbose=False, checkpoint_path=checkpoint)
                saved = torch.load(checkpoint, weights_only=True)
                self.assertEqual(saved["stage"], stage)
                self.assertEqual(saved["next_epoch"], 2)
                self.assertFalse(saved["stage_complete"])
                log = io.StringIO()
                with contextlib.redirect_stdout(log):
                    resumed, _, history, score = fit_pipeline(fit, val, "FD001", config, 42, checkpoint_path=checkpoint)
                self.assertIn(f"resuming checkpoint: {stage} epoch 2 completed", log.getvalue())
                self.assertIn("batch 1/1 complete", log.getvalue())
                self.assertNotIn("predictor 1/3: training", log.getvalue())
                self.assertEqual(score, expected_score)
                pd.testing.assert_frame_equal(history.drop(columns="duration_seconds"), expected_history.drop(columns="duration_seconds"))
                for name, weights in expected.state_dict().items():
                    torch.testing.assert_close(resumed.state_dict()[name], weights, rtol=0, atol=0)
                with patch("grace_rul.training.evaluate", side_effect=AssertionError("Completed pipeline trained again")):
                    cached, _, cached_history, cached_score = fit_pipeline(fit, val, "FD001", config, 42, verbose=False, checkpoint_path=checkpoint)
                self.assertEqual(cached_score, score)
                pd.testing.assert_frame_equal(cached_history, history)
                for name, weights in resumed.state_dict().items():
                    torch.testing.assert_close(cached.state_dict()[name], weights, rtol=0, atol=0)
                for changed_fit, changed_config in [(fit, {**config, "batch_size": 4}), (fit.iloc[1:], config)]:
                    with self.assertRaisesRegex(ValueError, "does not match"):
                        fit_pipeline(changed_fit, val, "FD001", changed_config, 42, verbose=False, checkpoint_path=checkpoint)

    def test_empirical_intervals_are_ordered_and_nonnegative(self):
        lo, hi = interval_bounds(np.array([0., 20., 100.]), {"lower_offset": -30., "upper_offset": -10.})
        self.assertTrue((lo >= 0).all()); self.assertTrue((hi >= lo).all())


class UploadAndAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, test, truths = load_subset(ROOT / "data", "FD001")
        cls.frame = test[test.unit <= 2].copy()
        cls.frame["true_rul"] = cls.frame.unit.map(truths)
        cls.frame.loc[(cls.frame.unit == 1) & (cls.frame.cycle >= 20), "sensor_3"] = np.nan
        cls.csv = cls.frame.to_csv(index=False).encode()

    def test_original_whitespace_and_csv_preserve_first_observation(self):
        original = (ROOT / "data" / "test_FD001.txt").read_bytes()
        parsed = parse_csv(original, "FD001")
        self.assertEqual(len(parsed), len(load_subset(ROOT / "data", "FD001")[1]))
        self.assertEqual(int(parsed.iloc[0].cycle), 1)
        csv = self.frame.drop(columns="true_rul").to_csv(index=False, header=False).encode()
        parsed = parse_csv(csv, "FD001")
        self.assertEqual(len(parsed), len(self.frame))
        self.assertTrue(parsed.sensor_3.isna().any())

    def test_duplicate_headers_and_rows_and_invalid_settings_are_rejected(self):
        header = self.csv.decode().splitlines()[0]
        duplicate = self.csv.decode().replace(header, header.replace("sensor_2", "sensor_1"), 1)
        with self.assertRaisesRegex(InputError, "duplicate column"):
            parse_csv(duplicate.encode(), "FD001")
        duplicate_rows = pd.concat([self.frame, self.frame.iloc[:1]]).to_csv(index=False).encode()
        with self.assertRaisesRegex(InputError, "Duplicate"):
            parse_csv(duplicate_rows, "FD001")
        incomplete = self.frame.copy(); incomplete.loc[0, "setting_1"] = np.nan
        with self.assertRaisesRegex(InputError, "complete and finite"):
            parse_csv(incomplete.to_csv(index=False).encode(), "FD001")

    def test_placeholder_api_previews_without_fabricating_predictions(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"GRACE_MODEL_PATH": "", "GRACE_ARTIFACTS_DIR": directory}), patch.object(web, "service", InferenceService()):
            client = InProcessClient(web.app)
            status = client.get("/api/status").json()
            self.assertFalse(status["ready"]); self.assertEqual(status["mode"], "pending")
            self.assertEqual(client.get("/api/model").json()["tables"]["test_metrics"], [])
            preview = client.post("/api/preview", files={"file": ("input.csv", self.csv)}, data={"subset": "FD001"})
            self.assertEqual(preview.status_code, 200)
            self.assertEqual(preview.json()["engines"], 2)
            prediction = client.post("/api/predict", files={"file": ("input.csv", self.csv)}, data={"subset": "FD001"})
            self.assertEqual(prediction.status_code, 503)
            explanation = client.post("/api/explain", files={"file": ("input.csv", self.csv)}, data={"subset": "FD001", "unit": 1})
            self.assertEqual(explanation.status_code, 503)
            self.assertEqual(client.get("/api/sample?subset=FD004").status_code, 200)

    def test_exported_model_recovery_explanations_and_discovery(self):
        # A real short staged run exercises the training-to-website artifact contract.
        with tempfile.TemporaryDirectory() as directory:
            config = quick_config({**DEFAULT_CONFIG, "device": "cpu"})
            path, tables = run_experiment(ROOT, config, Path(directory) / "quick")
            bundle, loaded = load_bundle(path)
            entry = bundle["models"]["FD001"]
            with patch("grace_rul.training.critic_update", side_effect=AssertionError("Cached model retrained")), contextlib.redirect_stdout(io.StringIO()) as resume_log:
                resumed_path, resumed_tables = run_experiment(ROOT, config, Path(directory) / "quick")
            self.assertEqual(resumed_path, path)
            self.assertEqual(resume_log.getvalue().count("resuming checkpoint"), 4)
            resumed_bundle, _ = load_bundle(resumed_path)
            for name, weights in entry["state_dict"].items():
                torch.testing.assert_close(resumed_bundle["models"]["FD001"]["state_dict"][name], weights, rtol=0, atol=0)
            for name, frame in tables.items():
                pd.testing.assert_frame_equal(resumed_tables[name], frame)
            held = set(entry["partitions"]["validation"] + entry["partitions"]["calibration"])
            self.assertFalse(held & set(entry["scaler"]["fit_units"]))
            pool = set(entry["split_protocol"]["training_pool_units"])
            self.assertEqual(pool, set(entry["partitions"]["fit"] + entry["partitions"]["calibration"]))
            self.assertEqual(len(entry["partitions"]["validation"]), 4)
            self.assertEqual(len(tables["cross_validation_metrics"]), 6)
            self.assertEqual(set(tables["cross_validation_metrics"].fold), {1, 2, 3})
            self.assertTrue((tables["cross_validation_summary"].folds == 3).all())
            self.assertTrue(np.isfinite(tables["cross_validation_summary"].RMSE_mean).all())
            correction_score = tables["correction_history"].validation_macro_RMSE.min()
            self.assertEqual(tables["seed_summary"].iloc[0].validation_selection_RMSE, correction_score)
            self.assertLessEqual(correction_score, tables["seed_summary"].iloc[0].validation_no_correction_RMSE)
            for fold in entry["oof_audit"]:
                self.assertFalse(set(fold["held_out_units"]) & set(fold["fit_units"] + fold["inner_validation_units"]))
            with patch.dict(os.environ, {"GRACE_MODEL_PATH": "", "GRACE_ARTIFACTS_DIR": directory}):
                self.assertFalse(InferenceService().status()["ready"])
            with patch.dict(os.environ, {"GRACE_MODEL_PATH": str(path)}), patch.object(web, "service", InferenceService()):
                client = InProcessClient(web.app)
                self.assertEqual(client.get("/api/status").json()["mode"], "quick")
                response = client.post("/api/predict", files={"file": ("input.csv", self.csv)}, data={"subset": "FD001", "mc_dropout": True})
                self.assertEqual(response.status_code, 200, response.text)
                report = response.json()
                self.assertEqual(len(report["engines"]), 2)
                first = report["engines"][0]
                self.assertIn("missing", first["sensors"][2]["status"])
                self.assertTrue(np.isfinite(first["sensor_history"][-1]["repaired"]).all())
                self.assertIsNone(first["sensor_history"][-1]["observed"][2])
                self.assertGreaterEqual(first["mc_std_cycles"], 0.)
                self.assertEqual(report["metrics"]["N"], 2)
                response = client.post("/api/explain", files={"file": ("input.csv", self.csv)}, data={"subset": "FD001", "unit": 2, "sensor": 3})
                self.assertEqual(response.status_code, 200, response.text)
                explanation = response.json()
                self.assertEqual([t["target"] for t in explanation["targets"]], ["final_rul", "finite_fault_score"])
                for target in explanation["targets"]:
                    self.assertEqual(len(target["sensors"]), 21)
                    self.assertTrue(np.isfinite(target["lime_holdout_fidelity"]))
                    # SHAP local additivity is in each target's original output units.
                    self.assertAlmostEqual(target["shap_baseline"] + sum(s["shap"] for s in target["sensors"]), target["prediction"], places=4)
            legacy = Path(directory) / "legacy.pt"
            torch.save({"format_version": 2}, legacy)
            with self.assertRaisesRegex(ValueError, "Legacy"):
                load_bundle(legacy)


if __name__ == "__main__":
    unittest.main()
