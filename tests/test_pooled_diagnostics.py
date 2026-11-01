"""Leakage, intervention provenance, and paired-engine uncertainty safeguards."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from grace_rul.checkpoints import pipeline_identity
from grace_rul.config import COLUMNS, SUBSETS
from grace_rul.data import OperatingRegimeScaler
from grace_rul.model import GraceRUL
from grace_rul.pooled import load_pooled_training, pooled_config, split_pooled_engines
from grace_rul.pooled_diagnostics import (checked_frames, fit_predictor_pilot, identity_intervention,
                                         paired_engine_bootstrap, run_diagnostic, warm_intervention)
from grace_rul.training import seed_all


def small_config():
    return {**pooled_config(quick=True), "device": "cpu", "sequence_length": 16,
            "local_length": 8, "batch_size": 4, "maximum_windows": 16}


def telemetry(units=20, cycles=35):
    return pd.DataFrame([[u, t, 0., 0., 100., *[j + .01 * t + .03 * u for j in range(21)]]
        for u in range(1, units + 1) for t in range(1, cycles + 1)], columns=COLUMNS)


class Interventions(unittest.TestCase):
    def setUp(self):
        seed_all(42)
        self.config = small_config()
        self.model = GraceRUL(self.config).eval()
        self.identity = {"config": self.config, "seed": 42, "fit": "fixture-fit", "validation": "fixture-val"}
        self.entry = {"pipeline_identity": self.identity}
        warm = {k: v.clone() for k, v in self.model.state_dict().items() if k.startswith(("detector.", "generator."))}
        warm["generator.decoder.2.weight"].fill_(.2)
        self.checkpoint = {"identity": self.identity, "warm_recovery": warm}

    def test_warm_merge_preserves_predictor_corrector_thresholds_support_and_source(self):
        before = {k: v.clone() for k, v in self.model.state_dict().items()}
        result = warm_intervention(self.model, self.checkpoint, self.entry)
        self.assertGreater(float(result.generator.decoder[-1].weight.detach().norm()), 0.)
        for key, value in result.state_dict().items():
            if not key.startswith(("generator.", "detector.")) and key != "repair_enabled":
                self.assertTrue(torch.equal(value, before[key]), key)
        for key, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]), key)

    def test_warm_merge_rejects_extra_missing_nonfinite_and_unpaired_detector_tensors(self):
        for kind in ["extra", "missing", "nonfinite", "detector", "identity"]:
            bad = deepcopy(self.checkpoint)
            if kind == "extra":
                bad["warm_recovery"]["predictor.head.0.weight"] = self.model.predictor.head[0].weight.detach().clone()
            elif kind == "missing":
                del bad["warm_recovery"]["generator.decoder.2.weight"]
            elif kind == "nonfinite":
                bad["warm_recovery"]["generator.decoder.2.weight"][0, 0] = float("nan")
            elif kind == "detector":
                bad["warm_recovery"]["detector.head.bias"][0] += .1
            else:
                bad["identity"]["fit"] = "different engines"
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                warm_intervention(self.model, bad, self.entry)

    def test_identity_probe_keeps_clean_sensors_within_interpolation_roundoff(self):
        warm = warm_intervention(self.model, self.checkpoint, self.entry)
        warm.finite_repair_thresholds.zero_()
        result = identity_intervention(warm)
        x, valid = torch.randn(2, 16, 24), torch.ones(2, 16, dtype=torch.bool)
        with torch.no_grad():
            predicted = result(x, valid, use_correction=False)
        self.assertTrue(torch.equal(predicted["generated"], predicted["observed"]))
        torch.testing.assert_close(predicted["repaired"], predicted["observed"], rtol=0., atol=4 * torch.finfo(x.dtype).eps)
        self.assertGreater(float(predicted["weights"].sum()), 0.)
        self.assertGreater(float(warm.generator.decoder[-1].weight.detach().norm()), 0.)

    def test_short_pilot_only_updates_predictor_and_samples_fitting_engines(self):
        frame = telemetry(units=8, cycles=20)
        scaler = OperatingRegimeScaler().fit(frame, "POOLED", 42, regime_selection="auto")
        warm = warm_intervention(self.model, self.checkpoint, self.entry)
        before = {k: v.clone() for k, v in warm.state_dict().items()}
        fitted, history, sampled = fit_predictor_pilot(warm, frame, scaler, self.config, steps=2, seed=7)
        self.assertEqual(len(history), 2)
        self.assertTrue(set(sampled).issubset(set(frame.unit)))
        self.assertTrue(any(not torch.equal(v, before[k]) for k, v in fitted.state_dict().items() if k.startswith("predictor.")))
        for key, value in fitted.state_dict().items():
            if not key.startswith("predictor."):
                self.assertTrue(torch.equal(value, before[key]), key)
        for key, value in warm.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]), key)


class ScopeChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "data").mkdir()
        self.config = small_config()
        for source in SUBSETS:
            telemetry().to_csv(self.root / f"data/train_{source}.txt", header=False, index=False, sep=" ")
            # An attempted official-file read would fail parsing these sentinels.
            (self.root / f"data/test_{source}.txt").write_text("DO NOT READ THIS OFFICIAL TEST\n")
            (self.root / f"data/RUL_{source}.txt").write_text("DO NOT READ THIS OFFICIAL LABEL\n")
        frame, hashes = load_pooled_training(self.root / "data", self.config)
        self.groups = split_pooled_engines(frame, 42)
        fit, val = [frame[frame.unit.isin(self.groups[k])].copy() for k in ["fit", "validation"]]
        self.entry = {"source_hashes": hashes, "partitions": self.groups, "selected_seed": 42,
            "scaler": {"fit_units": self.groups["fit"]},
            "pipeline_identity": pipeline_identity(fit, val, "POOLED", self.config, 42)}

    def tearDown(self):
        self.tmp.cleanup()

    def test_only_saved_fitting_and_selection_engines_are_returned_without_official_reads(self):
        fit, selection, hashes = checked_frames(self.root, self.entry, self.config)
        self.assertEqual(set(fit.unit), set(self.groups["fit"]))
        self.assertEqual(set(selection.unit), set(self.groups["validation"]))
        forbidden = set(self.groups["calibration"]) | set(self.groups["test"])
        self.assertFalse((set(fit.unit) | set(selection.unit)) & forbidden)
        self.assertEqual(len(hashes), 4)

    def test_partition_tampering_is_rejected(self):
        entry = deepcopy(self.entry)
        entry["partitions"]["validation"][0] = entry["partitions"]["test"][0]
        with self.assertRaisesRegex(ValueError, "partitions"):
            checked_frames(self.root, entry, self.config)

    def test_changed_training_files_are_rejected(self):
        with (self.root / "data/train_FD001.txt").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "hashes"):
            checked_frames(self.root, self.entry, self.config)

    def test_invalid_runtime_options_fail_before_creating_outputs(self):
        for options in [{"pilot_steps": -1}, {"pilot_steps": True}, {"replicates": 0}, {"batch_size": 0}, {"bootstrap_draws": 0}]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                run_diagnostic(self.root, self.root / "missing", self.root / "output", **options)
        self.assertFalse((self.root / "output").exists())


class EngineUncertainty(unittest.TestCase):
    def records(self):
        return pd.DataFrame([{"unit": unit, "cycle": cycle, "family": "stuck", "replicate": 0,
            "prediction": 1. if unit == 1 else 3., "target": 0., "raw_target": 0.,
            "altered_entries": 1, "valid_sensor_entries": 21}
            for unit, cycles in [(1, [1]), (2, [1, 2, 3, 4, 5])] for cycle in cycles])

    def test_resampling_unit_is_engine_despite_unequal_prefix_counts(self):
        reference, changed = self.records(), self.records()
        changed["prediction"] = 0.
        report = paired_engine_bootstrap(reference, changed, draws=100)
        self.assertEqual(report["engines"], 2)
        self.assertEqual(report["observations"], 6)
        self.assertAlmostEqual(report["engine_balanced_RMSE_delta"], -np.sqrt(5.))
        self.assertLess(report["bootstrap_95_high"], 0.)

    def test_identical_predictions_give_zero_change_and_interval(self):
        report = paired_engine_bootstrap(self.records(), self.records().iloc[::-1], draws=100)
        for key in ["engine_balanced_RMSE_delta", "bootstrap_95_low", "bootstrap_95_high"]:
            self.assertEqual(report[key], 0.)

    def test_mismatched_prefixes_targets_or_fault_exposure_are_rejected(self):
        for kind in ["missing", "duplicate", "target", "exposure"]:
            changed = self.records()
            if kind == "missing":
                changed = changed.iloc[1:]
            elif kind == "duplicate":
                changed = pd.concat([changed, changed.iloc[:1]])
            elif kind == "target":
                changed.loc[0, "target"] = 10.
            else:
                changed.loc[0, "altered_entries"] = 2
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                paired_engine_bootstrap(self.records(), changed, draws=100)


if __name__ == "__main__":
    unittest.main()
