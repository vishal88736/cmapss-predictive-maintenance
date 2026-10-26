"""Declared validation matrices reject hidden stuck/mixed recovery damage."""

from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset

from grace_rul.config import COLUMNS, DEFAULT_CONFIG
from grace_rul.data import OperatingRegimeScaler
from grace_rul.recovery_validation import (RECOVERY_FAMILIES, RecoveryValidationMatrix,
                                            audit_recovery_candidate, recovery_state_sha256,
                                            validate_recovery_audit)


def fixture_frame():
    frame = pd.DataFrame(np.zeros((64, 26)), columns=COLUMNS)
    frame["unit"] = np.repeat([1, 2], 32)
    frame["cycle"] = np.tile(np.arange(1, 33), 2)
    return frame


def fixture_config(**overrides):
    return {**DEFAULT_CONFIG, "device": "cpu", "sensor_transform": "linear",
        "sequence_length": 4, "validation_prefixes": 2, "batch_size": 2,
        "maximum_windows": 1, "maximum_batches": 1, "fault_start_horizon": 0,
        **overrides}


def fixture_scaler(config):
    return OperatingRegimeScaler({"fit_units": [99], "setting_mean": [0.] * 3,
        "setting_std": [1.] * 3, "centers": [[0.] * 3], "sensor_mean": [[0.] * 21],
        "sensor_std": [[1.] * 21], "sensor_fault_scale": [.1] * 21,
        "sensor_transform": config["sensor_transform"], "sensor_transform_scale": 1.})


class ScenarioDataset(Dataset):
    """Transparent controlled sensor errors; settings encode fixture scenario IDs."""
    def __init__(self, frame, scaler, config, *, mode, seed, family, severity):
        self.config, self.family, self.severity = config, family, severity
        self.epoch, self.progress = 0, 1.

    def __len__(self):
        return 4

    def __getitem__(self, index):
        x = torch.zeros(4, 24)
        x[:, 0] = RECOVERY_FAMILIES.index(self.family)
        x[:, 1], x[:, 2] = self.severity, index
        if self.config.get("_fixture_reference_linear") is not None:
            value = torch.tensor(self.config["_fixture_reference_linear"])
            x[:, 4] = torch.asinh(value) if self.config["sensor_transform"] == "asinh" else value
        reference = x.clone()
        altered = torch.zeros(4, 21, dtype=torch.bool)
        if self.family != "clean":
            x[:, 3] = self.config.get("_fixture_input_damage", 1.)
            altered[:, 0] = True
        valid = torch.ones(4, dtype=torch.bool)
        if self.config.get("_fixture_padding"):
            valid[0] = False
            x[0, 3:], reference[0, 3:] = float("inf"), 0.
        return {"x": x, "reference": reference, "valid": valid, "altered": altered,
                "unit": index // 2 + 1, "cycle": 3 if index % 2 == 0 else 6}


class ScenarioRepair(nn.Module):
    def __init__(self, config, *, family=None, severity=.5, damage=0., channel_only=False,
                 window_only=False, changed_output=0., linear_shift=0., nonfinite=False):
        super().__init__()
        self.config = config
        self.anchor = nn.Parameter(torch.zeros(()))
        self.dropout = nn.Dropout(.5)
        self.family, self.severity, self.damage = family, severity, damage
        self.channel_only, self.window_only = channel_only, window_only
        self.changed_output, self.linear_shift, self.nonfinite = changed_output, linear_shift, nonfinite
        self.calls = []

    def forward(self, x, valid, *, use_repair, use_correction):
        self.calls.append((use_repair, use_correction, self.training))
        observed = torch.nan_to_num(x[..., 3:], nan=0., posinf=0., neginf=0.)
        repaired = observed.clone()
        if use_repair:
            faulty = x[:, 0, 0] != 0
            repaired[faulty, :, 0] = self.changed_output
            if self.family is not None:
                selected = (x[:, 0, 0] == RECOVERY_FAMILIES.index(self.family)) & torch.isclose(
                    x[:, 0, 1], x.new_tensor(self.severity))
                if self.window_only:
                    selected &= x[:, 0, 2] == 3
                if self.channel_only:
                    repaired[selected, :, 1] += self.damage
                else:
                    repaired[selected, :, 1:] += self.damage
            if self.linear_shift:
                repaired[:, :, 1] = torch.asinh(torch.sinh(observed[:, :, 1].double()) + self.linear_shift).float()
        if self.config.get("_fixture_padding"):
            repaired[:, 0] = float("inf")
        if self.nonfinite:
            repaired[:, -1, 1] = float("nan")
        return {"observed": observed, "repaired": repaired}


def scenario_matrix(config):
    with patch("grace_rul.recovery_validation.WindowDataset", ScenarioDataset):
        return RecoveryValidationMatrix(fixture_frame(), fixture_scaler(config), config)


class RecoveryMatrixGuards(unittest.TestCase):
    def test_default_matrix_has_41_cached_cases_and_ignores_global_shortcuts(self):
        config = fixture_config()
        matrix = RecoveryValidationMatrix(fixture_frame(), fixture_scaler(config), config)
        self.assertEqual(len(matrix.datasets), 41)
        self.assertEqual(list(matrix.datasets)[0], ("clean", 0.))
        self.assertEqual(set(family for family, _ in matrix.datasets), set(RECOVERY_FAMILIES))
        sample = matrix.datasets[("stuck", .2)][0]
        cache = matrix.datasets[("stuck", .2)].cache
        cached_trajectory = next(iter(cache.values()))
        model = ScenarioRepair(config)
        result = audit_recovery_candidate(model, matrix)
        self.assertTrue(result["accepted"])
        self.assertTrue(result["complete"])
        self.assertTrue(all(row["windows"] == 4 for row in result["contexts"]))
        self.assertIs(next(iter(cache.values())), cached_trajectory)
        torch.testing.assert_close(matrix.datasets[("stuck", .2)][0]["x"], sample["x"], equal_nan=True)

    def test_stuck_failure_cannot_hide_behind_passing_primary_mixed_context(self):
        config = fixture_config()
        matrix = scenario_matrix(config)
        model = ScenarioRepair(config, family="stuck", severity=.5, damage=.3)
        result = audit_recovery_candidate(model, matrix)
        primary = next(row for row in result["contexts"] if row["family"] == "mixed" and row["severity"] == .2)
        self.assertTrue(primary["accepted"])
        self.assertTrue(result["complete"])
        self.assertFalse(result["accepted"])
        self.assertEqual(result["failing_contexts"], [{"family": "stuck", "severity": .5, "failed_guards": ["intact"]}])
        self.assertEqual(result["worst"]["contexts"]["linear_intact_repair_RMSE"]["family"], "stuck")

    def test_high_severity_mixed_failure_is_not_certified_by_severity_point_two(self):
        config = fixture_config()
        result = audit_recovery_candidate(ScenarioRepair(config, family="mixed", severity=.5, damage=.3), scenario_matrix(config))
        primary = next(row for row in result["contexts"] if row["family"] == "mixed" and row["severity"] == .2)
        self.assertTrue(primary["accepted"])
        self.assertFalse(result["accepted"])
        self.assertEqual(result["failing_contexts"][0]["severity"], .5)

    def test_clean_healthy_damage_rejects_and_early_exit_cannot_certify(self):
        config = fixture_config()
        model = ScenarioRepair(config, family="clean", severity=0., damage=.3)
        result = audit_recovery_candidate(model, scenario_matrix(config), stop_on_failure=True)
        self.assertFalse(result["accepted"])
        self.assertFalse(result["complete"])
        self.assertEqual(result["evaluated_contexts"], 1)
        self.assertEqual(result["failing_contexts"][0]["family"], "clean")

    def test_channel_tails_are_reported_and_guarded_only_when_declared(self):
        config = fixture_config()
        matrix = scenario_matrix(config)
        model = ScenarioRepair(config, family="stuck", damage=.3, channel_only=True)
        default = audit_recovery_candidate(model, matrix)
        self.assertTrue(default["accepted"])
        self.assertLess(default["worst"]["linear_intact_repair_RMSE"], .15)
        self.assertAlmostEqual(default["worst"]["linear_intact_channel_RMSE_max"], .3, places=6)
        guarded = audit_recovery_candidate(model, matrix, {**config, "recovery_validation_max_channel_intact_rmse": .15})
        self.assertFalse(guarded["accepted"])
        self.assertIn("intact_channel", guarded["failing_contexts"][0]["failed_guards"])

    def test_window_tails_preserve_worst_engine_cycle_and_optional_budget(self):
        config = fixture_config()
        matrix = scenario_matrix(config)
        model = ScenarioRepair(config, family="mixed", damage=.2, window_only=True)
        default = audit_recovery_candidate(model, matrix)
        self.assertTrue(default["accepted"])
        self.assertAlmostEqual(default["worst"]["linear_intact_repair_RMSE"], .1, places=6)
        self.assertAlmostEqual(default["worst"]["linear_intact_window_RMSE_max"], .2, places=6)
        context = next(row for row in default["contexts"] if row["family"] == "mixed" and row["severity"] == .5)
        self.assertEqual((context["worst_windows"][0]["unit"], context["worst_windows"][0]["cycle"]), (2, 6))
        self.assertGreater(context["linear_intact_window_RMSE_p99"], context["linear_intact_window_RMSE_p95"])
        guarded = audit_recovery_candidate(model, matrix, {**config, "recovery_validation_max_window_intact_rmse": .15})
        self.assertFalse(guarded["accepted"])
        self.assertIn("intact_window", guarded["failing_contexts"][0]["failed_guards"])

    def test_linear_reconstruction_ratio_rejects_worsened_faulty_cells(self):
        config = fixture_config()
        result = audit_recovery_candidate(ScenarioRepair(config, changed_output=2.), scenario_matrix(config))
        self.assertFalse(result["accepted"])
        self.assertTrue(all("linear_reconstruction" in row["failed_guards"] for row in result["failing_contexts"]))
        self.assertAlmostEqual(result["worst"]["linear_reconstruction_ratio"], 2.)

    def test_encoded_reconstruction_cap_is_opt_in_for_the_full_matrix(self):
        config = fixture_config(_fixture_input_damage=3.)
        matrix = scenario_matrix(config)
        model = ScenarioRepair(config, changed_output=2.)
        result = audit_recovery_candidate(model, matrix)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["worst"]["reconstruction_RMSE"], 2.)
        guarded = audit_recovery_candidate(model, matrix, {**config, "recovery_validation_max_reconstruction_rmse": 1.5})
        self.assertFalse(guarded["accepted"])
        self.assertTrue(all("reconstruction" in row["failed_guards"] for row in guarded["failing_contexts"]))

    def test_shortlist_calls_never_certify_even_when_the_fixture_fits_one_batch(self):
        config = fixture_config(batch_size=16)
        result = audit_recovery_candidate(ScenarioRepair(config), scenario_matrix(config), max_batches=1)
        self.assertTrue(all(row["complete"] for row in result["contexts"]))
        self.assertFalse(result["complete"])
        self.assertFalse(result["accepted"])
        self.assertEqual(result["mode"], "shortlist")
        config = fixture_config(batch_size=1)
        truncated = audit_recovery_candidate(ScenarioRepair(config), scenario_matrix(config), max_batches=1)
        self.assertTrue(all(row["windows"] == 1 for row in truncated["contexts"]))
        self.assertFalse(truncated["accepted"])

    def test_original_linear_scale_exposes_small_encoded_large_healthy_errors(self):
        config = fixture_config(sensor_transform="asinh", _fixture_reference_linear=1000.)
        result = audit_recovery_candidate(ScenarioRepair(config, linear_shift=1.), scenario_matrix(config))
        self.assertFalse(result["accepted"])
        self.assertAlmostEqual(result["worst"]["linear_intact_channel_RMSE_max"], 1., delta=.001)
        self.assertGreater(result["worst"]["linear_intact_repair_RMSE"], .15)

    def test_padding_is_excluded_but_valid_nonfinite_output_rejects_strict_json_report(self):
        config = fixture_config(_fixture_padding=True)
        matrix = scenario_matrix(config)
        good = audit_recovery_candidate(ScenarioRepair(config), matrix)
        self.assertTrue(good["accepted"])
        self.assertEqual(good["worst"]["nonfinite_contexts"], 0)
        bad = audit_recovery_candidate(ScenarioRepair(config, nonfinite=True), matrix)
        self.assertFalse(bad["accepted"])
        self.assertEqual(bad["worst"]["nonfinite_contexts"], 41)
        self.assertTrue(all("finite" in row["failed_guards"] for row in bad["failing_contexts"]))
        json.dumps(bad, allow_nan=False)

    def test_model_modes_restore_and_corrector_is_always_disabled(self):
        config = fixture_config()
        model = ScenarioRepair(config)
        model.train()
        model.dropout.eval()
        audit_recovery_candidate(model, scenario_matrix(config))
        self.assertTrue(model.training)
        self.assertFalse(model.dropout.training)
        self.assertTrue(all(not use_correction and not training for _, use_correction, training in model.calls))
        self.assertIsNone(model.anchor.grad)

    def test_invalid_provenance_axes_and_mutated_cache_are_rejected(self):
        config = fixture_config()
        frame = fixture_frame()
        frame["unit"] = 99
        with self.assertRaisesRegex(ValueError, "overlap"):
            RecoveryValidationMatrix(frame.drop_duplicates(["unit", "cycle"]), fixture_scaler(config), config)
        with self.assertRaisesRegex(ValueError, "distinct"):
            RecoveryValidationMatrix(fixture_frame(), fixture_scaler(config), {**config, "recovery_validation_severities": [.2, .2]})
        matrix = scenario_matrix(config)
        with self.assertRaisesRegex(ValueError, "changed"):
            audit_recovery_candidate(ScenarioRepair(config), matrix, {**config, "recovery_validation_severities": [.2]})
        with self.assertRaisesRegex(ValueError, "scenario configuration"):
            audit_recovery_candidate(ScenarioRepair(config), matrix, {**config, "fault_start_horizon": 30})
        matrix.datasets.pop(("stuck", .5))
        with self.assertRaisesRegex(ValueError, "contexts"):
            audit_recovery_candidate(ScenarioRepair(config), matrix)
        matrix = scenario_matrix(config)
        matrix.datasets[("clean", 0.)].epoch = 1
        with self.assertRaisesRegex(ValueError, "mutated"):
            audit_recovery_candidate(ScenarioRepair(config), matrix)


class PersistedRecoveryProof(unittest.TestCase):
    def setUp(self):
        self.config = fixture_config()
        self.partitions = {"fit": [99], "calibration": [100], "validation": [1, 2]}
        self.model = ScenarioRepair(dict(self.config))
        # Named sensor and RUL modules exercise the production hash boundary.
        self.model.detector = nn.Linear(1, 1)
        self.model.generator = nn.Linear(1, 1)
        self.model.predictor = nn.Linear(1, 1)
        self.model.corrector = nn.Linear(1, 1)
        self.model.register_buffer("finite_repair_thresholds", torch.full((21,), .5))
        self.model.register_buffer("recovery_lower", torch.zeros(1, 21))
        self.model.register_buffer("recovery_upper", torch.ones(1, 21))
        self.model.register_buffer("recovery_centers", torch.zeros(1, 3))
        self.model.register_buffer("recovery_regime_count", torch.tensor(1))
        self.model.register_buffer("repair_enabled", torch.tensor(True))
        self.model.register_buffer("correction_bound", torch.tensor(15.))
        self.audit = audit_recovery_candidate(self.model, scenario_matrix(self.config))

    def test_complete_default_proof_survives_strict_json_roundtrip(self):
        audit = json.loads(json.dumps(self.audit, allow_nan=False))
        self.assertIs(validate_recovery_audit(audit, self.config, self.partitions, self.model), audit)
        self.assertEqual(audit["expected_contexts"], 41)
        self.assertEqual(audit["scenario_options"]["fault_start_horizon"], self.config["fault_start_horizon"])
        self.assertEqual(audit["sequence_length"], self.config["sequence_length"])

    def test_quick_nine_context_proof_cannot_certify_full_41_configuration(self):
        quick = {**self.config, "recovery_validation_severities": [.2]}
        audit = audit_recovery_candidate(self.model, scenario_matrix(quick))
        validate_recovery_audit(audit, quick, self.partitions, self.model)
        self.assertEqual(audit["expected_contexts"], 9)
        with self.assertRaisesRegex(ValueError, "contexts/severities"):
            validate_recovery_audit(audit, self.config, self.partitions, self.model)

    def test_wrong_scope_seed_prefix_sequence_and_scenario_options_reject(self):
        changes = [
            ("validation_units", [1, 3]), ("seed", self.config["validation_seed"] + 1),
            ("validation_prefixes", 1), ("sequence_length", self.config["sequence_length"] + 1),
            ("prefix_policy", {"mode": "seeded_validation_prefixes", "prefixes_per_engine": 1}),
            ("scenario_options", {**self.audit["scenario_options"], "fault_start_horizon": 999}),
            ("representation", ["asinh", 1.])]
        for key, value in changes:
            with self.subTest(key=key):
                audit = deepcopy(self.audit); audit[key] = value
                with self.assertRaises(ValueError):
                    validate_recovery_audit(audit, self.config, self.partitions, self.model)
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_recovery_audit(self.audit, self.config,
                {**self.partitions, "calibration": [1, 100]}, self.model)

    def test_incomplete_reordered_failed_and_miscounted_contexts_reject(self):
        mutations = [lambda a: a.update(mode="shortlist"), lambda a: a.update(complete=False),
            lambda a: a.update(accepted=False), lambda a: a.update(use_correction=True),
            lambda a: a.update(use_repair=False), lambda a: a["contexts"].pop(),
            lambda a: a["contexts"].reverse(), lambda a: a["contexts"][0].update(complete=False),
            lambda a: a["contexts"][0].update(windows=3),
            lambda a: a["contexts"][0]["guards"].update(intact=False),
            lambda a: a["contexts"][0].update(nonfinite_entries=1)]
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                audit = deepcopy(self.audit); mutate(audit)
                with self.assertRaises(ValueError):
                    validate_recovery_audit(audit, self.config, self.partitions, self.model)

    def test_changed_budget_and_metrics_cannot_keep_old_passed_flags(self):
        with self.assertRaisesRegex(ValueError, "budgets"):
            validate_recovery_audit(self.audit, {**self.config, "max_intact_repair_rmse": .3}, self.partitions)
        bad = deepcopy(self.audit); bad["contexts"][0]["linear_intact_repair_RMSE"] = .3
        with self.assertRaisesRegex(ValueError, "metric"):
            validate_recovery_audit(bad, self.config, self.partitions, self.model)
        bad = deepcopy(self.audit); bad["worst"]["linear_intact_repair_RMSE"] = .123
        with self.assertRaisesRegex(ValueError, "worst intact"):
            validate_recovery_audit(bad, self.config, self.partitions, self.model)

    def test_sensor_weights_gates_support_and_clipping_policy_are_bound(self):
        state = deepcopy(self.model.state_dict())
        mutations = [lambda m: m.detector.weight.add_(1.), lambda m: m.generator.bias.add_(1.),
            lambda m: m.finite_repair_thresholds.add_(.1), lambda m: m.recovery_lower.sub_(.1),
            lambda m: m.recovery_upper.add_(.1), lambda m: m.recovery_centers.add_(.1),
            lambda m: m.recovery_regime_count.fill_(0), lambda m: m.repair_enabled.fill_(False)]
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                self.model.load_state_dict(state)
                with torch.no_grad():
                    mutate(self.model)
                with self.assertRaisesRegex(ValueError, "fingerprint"):
                    validate_recovery_audit(self.audit, self.config, self.partitions, self.model)
        self.model.load_state_dict(state)
        self.model.config["max_intact_repair_rmse"] = .3
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            validate_recovery_audit(self.audit, self.config, self.partitions, self.model)

    def test_predictor_corrector_and_correction_bound_do_not_invalidate_sensor_proof(self):
        before = recovery_state_sha256(self.model)
        with torch.no_grad():
            self.model.predictor.weight.add_(1.)
            self.model.corrector.bias.add_(1.)
            self.model.correction_bound.fill_(30.)
        self.assertEqual(recovery_state_sha256(self.model), before)
        validate_recovery_audit(self.audit, self.config, self.partitions, self.model)

    def test_empty_state_models_have_stable_fingerprints(self):
        class Empty(nn.Module):
            pass
        model = Empty()
        self.assertEqual(recovery_state_sha256(model), recovery_state_sha256(model))
        self.assertEqual(len(recovery_state_sha256(model)), 64)

    def test_missing_or_invalid_fingerprint_rejects_even_without_a_model(self):
        for fingerprint in [None, "0" * 63, "z" * 64]:
            audit = deepcopy(self.audit); audit["recovery_state_sha256"] = fingerprint
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                validate_recovery_audit(audit, self.config, self.partitions)


if __name__ == "__main__":
    unittest.main()
