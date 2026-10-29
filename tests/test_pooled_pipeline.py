"""Shared-model engine boundaries, fixed prefixes, artifacts and label-free API."""

import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from starlette.datastructures import UploadFile

from grace_rul.artifacts import load_bundle
from grace_rul.config import COLUMNS, FEATURE_COLUMNS, SUBSETS
from grace_rul.data import OperatingRegimeScaler, WindowDataset
from grace_rul.pooled import (load_pooled_training, pooled_config, prepare_pooled_run,
                              run_pooled_experiment, split_pooled_engines)
from grace_rul.recovery_validation import RecoveryValidationMatrix, audit_recovery_candidate, validate_recovery_audit
from grace_rul.training import checkpoint_guards, fit_pipeline, seed_all
from website.backend.inference import parse_csv
from website.backend.service import InferenceService


def synthetic_source(units=20, cycles=60):
    return pd.DataFrame([[u, t, 0., 0., 100., *[j + .01 * t + .03 * u for j in range(21)]]
        for u in range(1, units + 1) for t in range(1, cycles + 1)], columns=COLUMNS)


class PooledProtocol(unittest.TestCase):
    def setUp(self):
        seed_all(42)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "data").mkdir()
        for subset in SUBSETS:
            synthetic_source().to_csv(self.root / f"data/train_{subset}.txt", sep=" ", header=False, index=False)
        self.config = {**pooled_config(quick=True), "device": "cpu", "maximum_batches": 1,
            "pooled_evaluate_official": False}

    def tearDown(self):
        self.tmp.cleanup()

    def test_namespaces_and_split_use_whole_engines_and_no_official_files(self):
        frame, hashes = load_pooled_training(self.root / "data", self.config)
        groups = split_pooled_engines(frame, 42)
        self.assertEqual(frame.unit.nunique(), 80)
        self.assertEqual(len(frame[frame.source_unit.eq(1)].unit.unique()), 4)
        self.assertEqual({k: len(v) for k, v in groups.items()},
            {"fit": 64, "validation": 4, "calibration": 4, "test": 8, "validation_pool": 8})
        self.assertEqual(set(hashes), {f"train_{s}.txt" for s in SUBSETS})
        all_ids = []
        for name in ["fit", "validation", "calibration", "test"]:
            all_ids.extend(groups[name])
        self.assertEqual(len(set(all_ids)), len(all_ids))
        np.testing.assert_array_equal(FEATURE_COLUMNS, [f"setting_{i}" for i in range(1, 4)] + [f"sensor_{i}" for i in range(1, 22)])

    def test_fixed_prefix_schedule_is_stable_when_future_rows_are_appended(self):
        config = {**self.config, "maximum_windows": None}
        frame, _ = load_pooled_training(self.root / "data", config)
        groups = split_pooled_engines(frame, 42)
        fit = frame[frame.unit.isin(groups["fit"])]
        scaler = OperatingRegimeScaler().fit(fit, "POOLED", 42, regime_selection="auto")
        unit = groups["test"][0]
        short_frame = frame[frame.unit.eq(unit) & frame.cycle.le(45)]
        long_frame = frame[frame.unit.eq(unit)]
        short = WindowDataset(short_frame, scaler, config, mode="heldout", family="clean")
        long = WindowDataset(long_frame, scaler, config, mode="heldout", family="clean")
        self.assertEqual([short.engines[i]["cycles"][e] for i, e in short.samples], [30])
        self.assertEqual([long.engines[i]["cycles"][e] for i, e in long.samples], [30, 60])
        torch.testing.assert_close(short[0]["x"], long[0]["x"], rtol=0, atol=0)
        self.assertEqual(short[0]["x"].shape[-1], 24)

    def test_plan_change_cannot_reuse_a_training_directory(self):
        prepare_pooled_run(self.root, self.config, self.root / "plan")
        with self.assertRaisesRegex(ValueError, "fresh directory"):
            prepare_pooled_run(self.root, {**self.config, "imputation_loss_weight": .5}, self.root / "plan")

    def test_full_grid_sensor_audit_binds_schedule_and_allows_more_than_three_prefixes(self):
        frame, _ = load_pooled_training(self.root / "data", self.config)
        groups = split_pooled_engines(frame, 42)
        fit = frame[frame.unit.isin(groups["fit"])]
        scaler = OperatingRegimeScaler().fit(fit, "POOLED", 42, regime_selection="auto")
        held = synthetic_source(1, 180)

        class MeanFillFixture(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
            def forward(self, x, valid, **kwargs):
                observed = torch.nan_to_num(x[..., 3:])
                return {"repaired": observed, "observed": observed}

        model = MeanFillFixture(self.config)
        matrix = RecoveryValidationMatrix(held, scaler, self.config, seed=self.config["validation_seed"])
        audit = audit_recovery_candidate(model, matrix, self.config)
        partitions = {"fit": groups["fit"], "validation": [1], "calibration": [2]}
        validate_recovery_audit(audit, self.config, partitions, model=model)
        self.assertEqual(audit["contexts"][0]["windows"], 6)
        self.assertEqual(audit["prefix_policy"]["stride"], 30)
        with self.assertRaisesRegex(ValueError, "prefixes|scenario"):
            validate_recovery_audit(audit, {**self.config, "evaluation_stride": 15}, partitions)

    def test_pooled_average_cannot_hide_a_source_clean_rul_regression(self):
        clean = {"RMSE": 9., "NASA_Score_Mean": 1., "source_metrics": {
            "FD001": {"RMSE": 8., "NASA_Score_Mean": .8}, "FD004": {"RMSE": 12., "NASA_Score_Mean": 1.}}}
        references = {s: {"RMSE": 10., "NASA_Score_Mean": 1.} for s in ["FD001", "FD004"]}
        guards = checkpoint_guards(clean, {}, 10., self.config, use_repair=False,
            clean_nasa_reference=1., clean_source_reference=references)
        self.assertTrue(guards["clean_rmse"])
        self.assertTrue(guards["clean_rmse_FD001"])
        self.assertFalse(guards["clean_rmse_FD004"])

    def test_pooled_sensor_ratio_cannot_hide_a_source_imputation_failure(self):
        config = {**self.config, "fault_start_horizon": 0}
        fit = synthetic_source(1, 60)
        fit["unit"] = 99
        fit.loc[:, FEATURE_COLUMNS[3:]] = 0.
        scaler = OperatingRegimeScaler().fit(fit, "POOLED", 42, regime_selection="auto")
        held = synthetic_source(2, 60)
        held["source_subset"] = np.where(held.unit.eq(1), "FD001", "FD002")
        held.loc[held.unit.eq(1), FEATURE_COLUMNS[3:]] = .1
        held.loc[held.unit.eq(2), FEATURE_COLUMNS[3:]] = 100.

        class FixedImputationFixture(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = config
            def forward(self, x, valid, **kwargs):
                return {"repaired": torch.nan_to_num(x[..., 3:], nan=1.),
                        "observed": torch.nan_to_num(x[..., 3:], nan=0.)}

        audit = audit_recovery_candidate(FixedImputationFixture(), RecoveryValidationMatrix(held, scaler, config), config)
        self.assertTrue(all(c["accepted"] for c in audit["contexts"]))
        self.assertFalse(audit["accepted"])
        self.assertFalse(audit["source_audits"]["FD001"]["accepted"])
        self.assertTrue(any(c.get("source_subset") == "FD001" for c in audit["failing_contexts"]))


class SharedWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        (cls.root / "data").mkdir()
        for subset in SUBSETS:
            synthetic_source().to_csv(cls.root / f"data/train_{subset}.txt", sep=" ", header=False, index=False)
            synthetic_source(2, 40).to_csv(cls.root / f"data/test_{subset}.txt", sep=" ", header=False, index=False)
            (cls.root / f"data/RUL_{subset}.txt").write_text("12\n24\n")
        cls.config = {**pooled_config(quick=True), "device": "cpu", "maximum_batches": 1}
        combined, _ = load_pooled_training(cls.root / "data", cls.config)
        cls.groups = split_pooled_engines(combined, 42)
        active = {"enabled": True, "reads": []}
        output = cls.root / "run"

        def audit(event, args):
            if not active["enabled"] or event != "open" or not args or not isinstance(args[0], str):
                return
            p = Path(args[0]).resolve()
            if p.parent == cls.root / "data" and p.name.startswith(("test_", "RUL_")):
                manifest = output / "run_manifest.json"
                if not manifest.is_file() or json.loads(manifest.read_text())["status"] not in {"evaluating", "complete"}:
                    raise AssertionError("Official test data was read before selection, calibration and serialization finished.")
                active["reads"].append(p.name)

        sys.addaudithook(audit)
        cls.fitting_calls = []

        def capture(fit, val, *args, **kwargs):
            cls.fitting_calls.append((set(fit.unit), set(val.unit)))
            return fit_pipeline(fit, val, *args, **kwargs)

        try:
            with contextlib.redirect_stdout(io.StringIO()), patch("grace_rul.pooled.fit_pipeline", side_effect=capture), \
                    patch("grace_rul.training.fit_pipeline", side_effect=capture):
                cls.path, cls.tables = run_pooled_experiment(cls.root, cls.config, output)
        finally:
            active["enabled"] = False
        cls.official_reads = active["reads"]
        cls.bundle, cls.loaded = load_bundle(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_complete_shared_artifact_and_fitting_boundaries(self):
        self.assertEqual(self.bundle["format_version"], 9)
        self.assertEqual(set(self.loaded), {"POOLED"})
        entry = self.loaded["POOLED"]["entry"]
        self.assertEqual(set(entry["scaler"]["fit_units"]), set(self.groups["fit"]))
        self.assertEqual(len(self.fitting_calls), 4)
        forbidden = set(self.groups["test"]) | set(self.groups["calibration"])
        for fit, val in self.fitting_calls:
            self.assertFalse((fit | val) & forbidden)
            self.assertFalse(fit & val)
        self.assertEqual(self.fitting_calls[0], (set(self.groups["fit"]), set(self.groups["validation"])))
        self.assertEqual(set(entry["calibration"]["units"]), set(self.groups["calibration"]))
        self.assertEqual(set(self.official_reads), {f"{prefix}_{s}.txt" for prefix in ["test", "RUL"] for s in SUBSETS})
        self.assertTrue(json.loads((self.path.parent / "run_manifest.json").read_text())["bundle_unchanged_during_evaluation"])
        manifest = json.loads((self.path.parent / "run_manifest.json").read_text())
        self.assertEqual(manifest["training_device"], "cpu")
        for name in ["detector_history_POOLED_42.csv", "detector_audit_POOLED_42.json", "recovery_validation_POOLED_42.json"]:
            self.assertTrue((self.path.parent / name).is_file())

    def test_saved_metrics_recompute_and_internal_test_is_not_only_failure_endpoints(self):
        predictions = pd.read_csv(self.path.parent / "predictions/internal_test_10pct_clean_s0_r0.csv.gz")
        self.assertEqual(set(predictions.unit), set(self.groups["test"]))
        self.assertTrue(predictions.raw_target.gt(0).any())
        self.assertTrue(predictions.cycle.mod(30).eq(0).all())
        row = self.tables["test_metrics"].query("partition == 'internal_test_10pct' and subset == 'POOLED' and protocol == 'capped_125'").iloc[0]
        self.assertAlmostEqual(row.RMSE, float(np.sqrt(np.mean((predictions.prediction - predictions.target) ** 2))))

    def test_unknown_source_and_missing_fd_label_use_identical_shared_weights(self):
        sample = synthetic_source(1, 40)
        with patch.dict("os.environ", {"GRACE_MODEL_PATH": str(self.path)}):
            service = InferenceService()
            a = service.predict(parse_csv(sample.to_csv(index=False).encode()), "unknown.csv")
            sample["subset"] = "FD004"
            b = service.predict(parse_csv(sample.to_csv(index=False).encode(), "POOLED"), "named.csv")
            self.assertEqual(a["engines"][0]["predicted_rul"], b["engines"][0]["predicted_rul"])
            self.assertEqual(a["engines"][0]["model_key"], "POOLED")
            self.assertFalse(service.status()["requires_subset_label"])

    def test_upload_handler_accepts_no_subset_label(self):
        from website.backend.app import predict_upload
        upload = UploadFile(io.BytesIO(synthetic_source(1, 40).to_csv(index=False).encode()), filename="telemetry.csv")
        with patch.dict("os.environ", {"GRACE_MODEL_PATH": str(self.path)}):
            report = predict_upload(upload, subset="", mc_dropout=False)
        self.assertEqual(report["engines"][0]["model_key"], "POOLED")

    def test_shared_loader_rejects_partition_overlap(self):
        damaged = deepcopy(self.bundle)
        entry = damaged["models"]["POOLED"]
        entry["partitions"]["test"].append(entry["partitions"]["fit"][0])
        path = self.path.parent / "invalid.pt"
        torch.save(damaged, path)
        with self.assertRaisesRegex(ValueError, "overlap"):
            load_bundle(path)

    def test_shared_loader_rejects_calibration_scope_and_routing_config_tampering(self):
        for change, message in [("calibration", "separate calibration"), ("routing", "configurations disagree")]:
            with self.subTest(change=change):
                damaged = deepcopy(self.bundle)
                if change == "calibration":
                    damaged["models"]["POOLED"]["calibration"]["units"] = self.groups["validation"]
                else:
                    damaged["config"]["training_scope"] = "per_subset"
                path = self.path.parent / f"invalid_{change}.pt"
                torch.save(damaged, path)
                with self.assertRaisesRegex(ValueError, message):
                    load_bundle(path)


if __name__ == "__main__":
    unittest.main()
