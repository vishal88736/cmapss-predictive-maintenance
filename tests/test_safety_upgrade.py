"""Frozen-weight migration, fail-closed export, and untouched-data safeguards."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from grace_rul.calibration import calibrate_endpoint_intervals as real_calibrate
from grace_rul.config import COLUMNS, DEFAULT_CONFIG, FEATURE_COLUMNS, SUPPORTED_ARCHITECTURES, quick_config
from grace_rul.data import OperatingRegimeScaler, WindowDataset
from grace_rul.model import GraceRUL
from scripts import upgrade_recovery_safety as upgrade


class SafetyUpgrade(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source_dir = self.root / "source_run"
        self.source_dir.mkdir()
        (self.root / "data").mkdir()
        self.config = {**quick_config(DEFAULT_CONFIG), "device": "cpu", "subsets": ["FD001"],
                       "recovery_safety": "legacy", "max_clean_rmse_increase": .01,
                       "max_clean_nasa_increase": .01}
        rows = []
        for unit in range(1, 7):
            for cycle in range(1, 5):
                rows.append([unit, cycle, *[float(cycle + j + unit / 10) for j in range(24)]])
        self.frame = pd.DataFrame(rows, columns=COLUMNS)
        self.frame[["unit", "cycle"]] = self.frame[["unit", "cycle"]].astype(int)
        train_path = self.root / "data/train_FD001.txt"
        self.frame.to_csv(train_path, sep=" ", header=False, index=False)
        fit = self.frame[self.frame.unit.isin([1, 2])]
        val = self.frame[self.frame.unit.isin([3, 4])]
        scaler = OperatingRegimeScaler().fit(fit, "FD001", 42)
        model = GraceRUL(self.config).eval()
        self.entry = {"state_dict": deepcopy(model.state_dict()), "scaler": scaler.state,
                      "training_config": self.config, "selected_seed": 42,
                      "thresholds": [1.] * 21, "calibration": {"lower_offset": -2., "upper_offset": 2.},
                      "partitions": {"fit": [1, 2], "validation": [3, 4], "calibration": [5, 6]},
                      "correction_bound": float(model.correction_bound), "repair_selected": bool(model.repair_enabled),
                      "source_hashes": {train_path.name: hashlib.sha256(train_path.read_bytes()).hexdigest()},
                      "pipeline_identity": {"subset": "FD001", "seed": 42,
                                            "config": deepcopy(self.config),
                                            "fit": upgrade._fingerprint(fit), "validation": upgrade._fingerprint(val)}}
        self.bundle = {"format_version": 7, "architecture": SUPPORTED_ARCHITECTURES[7],
                       "feature_columns": FEATURE_COLUMNS, "config": self.config, "models": {"FD001": self.entry}}
        self.source = self.source_dir / "grace_rul_ALL.pt"
        torch.save(self.bundle, self.source)
        checkpoint = self.source_dir / "checkpoints/pipeline_FD001_42.pt"
        checkpoint.parent.mkdir()
        torch.save({"identity": self.entry["pipeline_identity"], "model": deepcopy(self.entry["state_dict"]),
                    "clean_reference": 1., "clean_nasa_reference": 1.}, checkpoint)
        self.good_report = {"RMSE": .5, "NASA_Score_Mean": .5, "linear_intact_repair_RMSE": .01,
                            "linear_reconstruction_ratio": .5, "reconstruction_RMSE": .1}
        self.good_audit = {"complete": True, "accepted": True, "contexts": [], "worst": {},
                           "expected_contexts": 41, "evaluated_contexts": 41}

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def calibration(model, frame, scaler, config, seed):
        def evaluator(model, dataset, settings):
            rows = []
            for i, end in dataset.samples:
                engine = dataset.engines[i]
                target = float(engine["targets"][end])
                rows.append({"unit": engine["unit"], "cycle": int(engine["cycles"][end]),
                             "prediction": target, "raw_target": target,
                             "target": min(settings["train_rul_cap"], target)})
            return {}, pd.DataFrame(rows)
        return real_calibrate(model, frame, scaler, config, seed, evaluator=evaluator)

    def run_upgrade(self, audit=None, reports=None, output=None, matrix_object=None):
        output = output or self.root / "upgrade"
        with patch.object(upgrade, "RecoveryValidationMatrix", return_value=matrix_object) as matrix, \
                patch.object(upgrade, "audit_recovery_candidate", return_value=self.good_audit if audit is None else audit) as audited, \
                patch.object(upgrade, "evaluate", side_effect=[(row, None) for row in (reports or [self.good_report, self.good_report])]) as evaluator, \
                patch.object(upgrade, "calibrate_endpoint_intervals", side_effect=self.calibration) as calibrated:
            result = upgrade.upgrade_recovery_safety(self.source, output, project_root=self.root)
            return result, matrix, audited, evaluator, calibrated

    def test_success_preserves_every_source_tensor_and_retains_corrector(self):
        before = self.source.read_bytes()
        config = upgrade.upgrade_config(self.config, 2)
        scaler = OperatingRegimeScaler(self.entry["scaler"])
        fit = self.frame[self.frame.unit.isin([1, 2])]
        val = self.frame[self.frame.unit.isin([3, 4])]
        model = upgrade.migrate_frozen_model(self.entry, config)
        model.set_recovery_envelope(upgrade.fit_recovery_envelope(fit, scaler))
        real_matrix = upgrade.RecoveryValidationMatrix(val, scaler, config, seed=config["validation_seed"])
        real_audit = upgrade.audit_recovery_candidate(model, real_matrix, config)
        self.assertTrue(real_audit["accepted"])
        (path, report), matrix, audited, evaluator, calibrated = self.run_upgrade(audit=real_audit, matrix_object=real_matrix)
        saved = torch.load(path, map_location="cpu", weights_only=True)
        new = saved["models"]["FD001"]
        self.assertEqual(saved["format_version"], 8)
        self.assertEqual(report["status"], "accepted")
        self.assertFalse(saved["weights_retrained"])
        self.assertTrue(report["serialized_bundle_strictly_loaded"])
        self.assertEqual(self.source.read_bytes(), before)
        self.assertEqual(set(new["state_dict"]), set(self.entry["state_dict"]) | upgrade.SUPPORT_BUFFERS)
        for key, value in self.entry["state_dict"].items():
            self.assertTrue(torch.equal(new["state_dict"][key], value), key)
        self.assertEqual(new["weight_training_config"], self.config)
        self.assertEqual(new["training_config"]["recovery_safety"], "fitted_support")
        self.assertEqual(new["recovery_support"]["fit_units"], [1, 2])
        self.assertEqual(new["calibration"]["units"], [5, 6])
        self.assertTrue(new["calibration"]["unbounded"])
        self.assertEqual(set(calibrated.call_args.args[1].unit), {5, 6})
        self.assertEqual(set(matrix.call_args.args[0].unit), {3, 4})
        self.assertTrue(audited.call_args.kwargs["use_repair"])
        self.assertEqual(evaluator.call_count, 2)
        self.assertFalse((path.parent / "test_metrics.csv").exists())
        self.assertFalse((path.parent / "cross_validation_metrics.csv").exists())
        self.assertTrue((path.parent / "calibration_FD001_42.csv").exists())
        self.assertEqual(len(list((self.root / "data").iterdir())), 1)

    def test_real_upgrade_never_opens_official_test_or_copies_old_metric_tables(self):
        """Use real model inference, full matrix, calibration, and both loaders."""
        model = GraceRUL(self.config).eval()
        model.load_state_dict(self.entry["state_dict"], strict=True)
        scaler = OperatingRegimeScaler(self.entry["scaler"])
        val = self.frame[self.frame.unit.isin([3, 4])]
        dataset = WindowDataset(val, scaler, self.config, mode="validation",
                                seed=self.config["validation_seed"], family="clean")
        measured, _ = upgrade.evaluate(model, dataset, self.config)
        # References are measured from this fixture's original frozen model,
        # rather than chosen to force an accuracy result or mock acceptance.
        torch.save({"identity": self.entry["pipeline_identity"],
                    "model": deepcopy(self.entry["state_dict"]),
                    "clean_reference": measured["RMSE"],
                    "clean_nasa_reference": measured["NASA_Score_Mean"]},
                   self.source_dir / "checkpoints/pipeline_FD001_42.pt")
        for name in ["test_metrics.csv", "cross_validation_metrics.csv", "robustness_summary.csv", "predictions_FD001_42.csv"]:
            (self.source_dir / name).write_text("fixture old table; must not read or copy\n")
        state = {"active": True, "train_reads": set()}

        def block_held_test_files(event, args):
            if not state["active"] or event != "open" or not args or not isinstance(args[0], (str, bytes)):
                return
            name = Path(args[0].decode() if isinstance(args[0], bytes) else args[0]).name
            if (name.startswith(("test_FD", "RUL_FD", "predictions_FD")) or
                    name in {"test_metrics.csv", "robustness_summary.csv", "cross_validation_metrics.csv"}):
                raise AssertionError(f"Upgrade attempted forbidden held-test/old-metric file access: {name}")
            if name.startswith("train_FD"):
                state["train_reads"].add(name)

        sys.addaudithook(block_held_test_files)
        try:
            path, report = upgrade.upgrade_recovery_safety(self.source, self.root / "real_upgrade", project_root=self.root)
        finally:
            # Audit hooks cannot be removed; disable it before other tests run.
            state["active"] = False
        self.assertEqual(state["train_reads"], {"train_FD001.txt"})
        saved = torch.load(path, map_location="cpu", weights_only=True)
        entry = saved["models"]["FD001"]
        self.assertEqual(report["status"], "accepted")
        self.assertEqual(report["subsets"][0]["recovery_validation"]["evaluated_contexts"], 41)
        self.assertEqual(report["subsets"][0]["clean_validation_retained_corrector"]["RMSE"], measured["RMSE"])
        self.assertEqual(set(entry["calibration"]["units"]), {5, 6})
        self.assertFalse(set(entry["calibration"]["units"]) & {1, 2, 3, 4})
        for name in ["test_metrics.csv", "cross_validation_metrics.csv", "robustness_summary.csv", "predictions_FD001_42.csv"]:
            self.assertFalse((path.parent / name).exists())

    def test_rejected_matrix_exports_nothing_and_reports_failure(self):
        with self.assertRaises(upgrade.SafetyUpgradeRejected):
            self.run_upgrade(audit={**self.good_audit, "accepted": False})
        out = self.root / "upgrade"
        self.assertEqual(json.loads((out / "safety_upgrade_report.json").read_text())["status"], "rejected")
        self.assertFalse((out / "grace_rul_ALL.pt").exists())
        self.assertFalse((out / "seed_models").exists())
        self.assertFalse(list(out.glob("calibration*.csv")))

    def test_incomplete_matrix_cannot_certify_even_if_it_claims_acceptance(self):
        with self.assertRaises(upgrade.SafetyUpgradeRejected):
            self.run_upgrade(audit={**self.good_audit, "complete": False})

    def test_original_nasa_guard_cannot_be_replaced_by_source_final_metrics(self):
        weak = {**self.good_report, "NASA_Score_Mean": 1.02}
        with self.assertRaises(upgrade.SafetyUpgradeRejected):
            self.run_upgrade(reports=[weak, self.good_report])
        report = json.loads((self.root / "upgrade/safety_upgrade_report.json").read_text())
        self.assertFalse(report["subsets"][0]["guards"]["clean_nasa"])

    def test_missing_non_support_weight_is_rejected(self):
        entry = deepcopy(self.entry)
        del entry["state_dict"]["corrector.0.weight"]
        with self.assertRaisesRegex(ValueError, "four fitting-support buffers"):
            upgrade.migrate_frozen_model(entry, upgrade.upgrade_config(self.config, 2))

    def test_serialized_contract_failure_does_not_publish_artifact_or_manifest(self):
        actual_loader = upgrade.load_bundle

        def loader(path, device="cpu"):
            if Path(path).resolve() == self.source.resolve():
                return actual_loader(path, device=device)
            raise ValueError("Fixture serialized contract rejected")

        with patch.object(upgrade, "load_bundle", side_effect=loader), \
                self.assertRaisesRegex(ValueError, "serialized contract rejected"):
            self.run_upgrade()
        out = self.root / "upgrade"
        report = json.loads((out / "safety_upgrade_report.json").read_text())
        self.assertEqual(report["status"], "error")
        self.assertFalse(report["artifact_exported"])
        self.assertFalse((out / "grace_rul_ALL.pt").exists())
        self.assertFalse((out / "run_manifest.json").exists())
        self.assertFalse((out / ".grace_rul_ALL.candidate.pt").exists())

    def test_upgrade_preserves_existing_stricter_optional_tail_limits(self):
        config = upgrade.upgrade_config({**self.config, "recovery_validation_max_channel_intact_rmse": .02}, 2)
        self.assertEqual(config["recovery_validation_max_channel_intact_rmse"], .02)
        self.assertEqual(config["recovery_validation_severities"], [.1, .2, .3, .4, .5])
        self.assertEqual(config["max_intact_repair_rmse"], self.config["max_intact_repair_rmse"])

    def test_existing_output_and_source_subdirectory_are_never_written(self):
        target = self.root / "existing"
        target.mkdir()
        sentinel = target / "sentinel"
        sentinel.write_text("keep")
        with self.assertRaises(FileExistsError):
            upgrade.upgrade_recovery_safety(self.source, target, project_root=self.root)
        self.assertEqual(sentinel.read_text(), "keep")
        with self.assertRaisesRegex(ValueError, "outside the immutable"):
            upgrade.upgrade_recovery_safety(self.source, self.source_dir / "child", project_root=self.root)
        self.assertFalse((self.source_dir / "child").exists())

    def test_source_must_be_format7(self):
        self.bundle["format_version"] = 6
        self.bundle["architecture"] = SUPPORTED_ARCHITECTURES[6]
        torch.save(self.bundle, self.source)
        with self.assertRaisesRegex(ValueError, "only a format-7"):
            self.run_upgrade()
        self.assertFalse((self.root / "upgrade").exists())

    def test_partition_overlap_and_training_source_changes_fail_closed(self):
        bad = deepcopy(self.entry)
        bad["partitions"]["calibration"] = [2, 5]
        with self.assertRaisesRegex(ValueError, "overlap"):
            upgrade._training_partitions(self.root, "FD001", bad)
        (self.root / "data/train_FD001.txt").write_text("changed")
        with self.assertRaisesRegex(ValueError, "hash"):
            upgrade._training_partitions(self.root, "FD001", self.entry)

    def test_history_fallback_uses_selected_macro_predictor_not_minimum_clean(self):
        (self.source_dir / "checkpoints/pipeline_FD001_42.pt").unlink()
        pd.DataFrame([
            {"subset":"FD001","seed":42,"stage":"predictor","epoch":1,"checkpoint_accepted":True,"validation_macro_RMSE":3.,"validation_clean_RMSE":1.,"validation_clean_NASA_Score_Mean":2.},
            {"subset":"FD001","seed":42,"stage":"predictor","epoch":2,"checkpoint_accepted":True,"validation_macro_RMSE":2.,"validation_clean_RMSE":1.5,"validation_clean_NASA_Score_Mean":1.8},
            {"subset":"FD001","seed":42,"stage":"generator","epoch":3,"checkpoint_accepted":True,"validation_macro_RMSE":1.,"validation_clean_RMSE":.5,"validation_clean_NASA_Score_Mean":.5},
        ]).to_csv(self.source_dir / "training_history.csv", index=False)
        result = upgrade.original_clean_references(self.source_dir, "FD001", self.entry)
        self.assertEqual(result["epoch"], 2)
        self.assertEqual(result["RMSE"], 1.5)
        self.assertEqual(result["NASA_Score_Mean"], 1.8)

    def test_same_partitions_with_another_training_law_cannot_supply_references(self):
        path = self.source_dir / "checkpoints/pipeline_FD001_42.pt"
        state = torch.load(path, map_location="cpu", weights_only=True)
        state["identity"]["config"]["learning_rate"] *= 2
        torch.save(state, path)
        with self.assertRaisesRegex(ValueError, "complete source training identity"):
            upgrade.original_clean_references(self.source_dir, "FD001", self.entry)

    def test_same_training_identity_with_stale_upstream_weights_is_rejected(self):
        path = self.source_dir / "checkpoints/pipeline_FD001_42.pt"
        state = torch.load(path, map_location="cpu", weights_only=True)
        state["model"]["generator.decoder.2.weight"].add_(1.)
        torch.save(state, path)
        with self.assertRaisesRegex(ValueError, "upstream weights/state"):
            upgrade.original_clean_references(self.source_dir, "FD001", self.entry)

    def test_original_reference_binding_allows_post_training_corrector_selection(self):
        path = self.source_dir / "checkpoints/pipeline_FD001_42.pt"
        state = torch.load(path, map_location="cpu", weights_only=True)
        state["model"]["corrector.0.weight"].add_(1.)
        state["model"]["correction_bound"].add_(1.)
        torch.save(state, path)
        reference = upgrade.original_clean_references(self.source_dir, "FD001", self.entry)
        self.assertTrue(reference["upstream_state_verified"])


if __name__ == "__main__":
    unittest.main()
