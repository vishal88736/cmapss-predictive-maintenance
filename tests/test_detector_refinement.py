"""Finite-label, grouped-scope and unchanged-RUL detector regression cases."""

from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from grace_rul.config import DEFAULT_CONFIG, quick_config
from grace_rul.detection import (DETECTOR_FAMILIES, _select_thresholds, _summarize,
                                 balanced_finite_fault_loss, refine_bypassed_detector,
                                 select_detector_thresholds)
from grace_rul.model import GraceRUL


class TinyFaultWindows(Dataset):
    """Known finite faults are visible in measured inputs, never oracle inputs."""
    scopes = []

    def __init__(self, frame, scaler, config, mode="train", seed=42, family=None):
        self.units = sorted(frame.unit.unique().tolist())
        self.config, self.family, self.epoch = config, family, 0
        self.engines = [{"unit": unit} for unit in self.units]
        self.samples = [(i, end) for i in range(len(self.units)) for end in range(6)]
        self.scopes.append((mode, tuple(self.units), family))

    def set_epoch(self, epoch, progress=1.):
        self.epoch = epoch

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        engine, end = self.samples[index]
        rng = np.random.default_rng(7919 * int(self.units[engine]) + end)
        x = torch.tensor(rng.normal(0., .02, (12, 24)), dtype=torch.float32)
        labels = torch.zeros((12, 21), dtype=torch.bool)
        if self.family not in ("clean", "outage"):
            # Faulted sensors share a visible domain signature; ordinary cells
            # keep the clean scale. A real GRU detector must learn this mapping.
            x[:, 3:6] += 6.
            labels[:, :3] = True
        if self.family in ("outage", "mixed"):
            x[:4, 8] = float("nan")
            labels[:4, 5] = True
        valid = torch.ones(12, dtype=torch.bool)
        valid[0] = False
        return {"x": x, "valid": valid, "altered": labels,
                "unit": self.units[engine], "cycle": end}


class DetectorRefinement(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(41)
        self.config = {**quick_config({**DEFAULT_CONFIG, "device": "cpu"}),
                       "sequence_length": 12, "local_length": 6, "dropout": 0.,
                       "cnn_channels": 4, "gru_hidden": 8,
                       "detector_refinement_epochs": 6,
                       "detector_refinement_learning_rate": .03,
                       "detector_refinement_patience": 6,
                       "maximum_batches": 3, "maximum_windows": None}
        self.fit = pd.DataFrame({"unit": [1, 2, 3, 4], "cycle": [1] * 4})
        self.validation = pd.DataFrame({"unit": [5, 6], "cycle": [1, 1]})
        self.scaler = SimpleNamespace(state={"fit_units": [1, 2, 3, 4]})
        self.model = GraceRUL(self.config).eval()
        self.model.repair_enabled.fill_(False)
        TinyFaultWindows.scopes = []

    def run_refinement(self, **options):
        with patch("grace_rul.detection.WindowDataset", TinyFaultWindows):
            return refine_bypassed_detector(self.model, self.fit, self.validation,
                                            self.scaler, self.config, 17, **options)

    def test_learns_detector_without_mutating_rul_modules_or_predictions(self):
        self.model.train()
        self.model.generator.eval()
        next(self.model.predictor.parameters()).requires_grad_(False)
        before = deepcopy(self.model.state_dict())
        modes = [child.training for child in self.model.modules()]
        flags = [parameter.requires_grad for parameter in self.model.parameters()]
        x = torch.randn(2, 12, 24)
        x[0, 3:5, 5] = float("nan")
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(x)["prediction"].clone()
        for child, mode in zip(self.model.modules(), modes):
            child.training = mode
        rng = torch.get_rng_state().clone()
        history = self.run_refinement()
        self.assertTrue(history.attrs["audit"]["accepted"])
        self.assertGreater(history.attrs["audit"]["selected"]["score"], 0.)
        self.assertTrue(any(not torch.equal(value, self.model.state_dict()[name])
                            for name, value in before.items() if name.startswith("detector.")))
        for name, value in before.items():
            if not name.startswith("detector."):
                torch.testing.assert_close(value, self.model.state_dict()[name], rtol=0, atol=0)
        self.assertEqual(modes, [child.training for child in self.model.modules()])
        self.assertEqual(flags, [parameter.requires_grad for parameter in self.model.parameters()])
        torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
        self.model.eval()
        with torch.no_grad():
            torch.testing.assert_close(prediction, self.model(x)["prediction"], rtol=0, atol=0)

    def test_active_recovery_is_not_refined_or_validated(self):
        self.model.repair_enabled.fill_(True)
        before = deepcopy(self.model.state_dict())
        with patch("grace_rul.detection.WindowDataset", side_effect=AssertionError("must skip")):
            history = refine_bypassed_detector(self.model, None, None, None, self.config, 17)
        self.assertEqual(history.attrs["audit"]["status"], "skipped_active_recovery")
        for name, value in before.items():
            torch.testing.assert_close(value, self.model.state_dict()[name], rtol=0, atol=0)

    def test_finite_loss_excludes_missing_and_padded_labels(self):
        logits = torch.zeros((1, 3, 2), requires_grad=True)
        available = torch.tensor([[[False, False], [True, False], [True, True]]])
        labels = torch.ones_like(available)
        balanced_finite_fault_loss(logits, labels, available).backward()
        self.assertEqual(float(logits.grad[~available].abs().sum()), 0.)
        self.assertGreater(float(logits.grad[available].abs().sum()), 0.)

    def test_unchanged_baseline_restored_when_candidates_do_not_improve(self):
        self.config["detector_refinement_epochs"] = 2
        before = deepcopy(self.model.detector.state_dict())
        def metric(*args, **kwargs):
            return [.5] * 21, {"score": .8, "clean_false_positive_rate": 0.,
                                "max_context_false_positive_rate": 0., "contexts": {}}
        with patch("grace_rul.detection.select_detector_thresholds", side_effect=metric):
            history = self.run_refinement()
        self.assertFalse(history.attrs["audit"]["accepted"])
        self.assertFalse(history.accepted.any())
        for name, value in before.items():
            torch.testing.assert_close(value, self.model.detector.state_dict()[name], rtol=0, atol=0)

    def test_improved_f1_is_rejected_if_any_context_exceeds_fpr_guard(self):
        self.config["detector_refinement_epochs"] = 1
        before = deepcopy(self.model.detector.state_dict())
        baseline = {"score": .1, "clean_false_positive_rate": 0.,
                    "max_context_false_positive_rate": 0., "contexts": {}}
        candidate = {**baseline, "score": .9, "max_context_false_positive_rate": .4}
        with patch("grace_rul.detection.select_detector_thresholds",
                   side_effect=[([.5] * 21, baseline), ([.5] * 21, candidate)]):
            history = self.run_refinement()
        self.assertFalse(history.attrs["audit"]["accepted"])
        for name, value in before.items():
            torch.testing.assert_close(value, self.model.detector.state_dict()[name], rtol=0, atol=0)

    def test_threshold_guard_covers_intact_cells_during_stuck_faults(self):
        clean = (np.full(20, .1, np.float32), np.zeros(20, bool))
        stuck = (np.r_[np.full(20, .9), np.full(4, .95)].astype(np.float32),
                 np.r_[np.zeros(20, bool), np.ones(4, bool)])
        contexts = {"clean": [clean] * 21, "stuck": [stuck] * 21}
        thresholds = _select_thresholds(contexts, .05)
        audit = _summarize(contexts, thresholds)
        self.assertGreater(min(thresholds), .9)
        self.assertLessEqual(audit["max_context_false_positive_rate"], .05)
        self.assertEqual(audit["contexts"]["stuck"]["recall"], 1.)

    def test_only_fit_engines_train_and_validation_engines_select(self):
        self.config["detector_refinement_epochs"] = 1
        self.run_refinement()
        for mode, units, _ in TinyFaultWindows.scopes:
            self.assertEqual(units, (1, 2, 3, 4) if mode == "train" else (5, 6))
        for mode in ("train", "validation"):
            families = {family for called_mode, _, family in TinyFaultWindows.scopes if called_mode == mode}
            self.assertEqual(families, set(DETECTOR_FAMILIES))

    def test_overlapping_engines_and_contaminated_scaler_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "disjoint"):
            refine_bypassed_detector(self.model, self.fit, self.fit, self.scaler, self.config, 17)
        self.scaler.state["fit_units"].append(5)
        with self.assertRaisesRegex(ValueError, "scaler fitted"):
            self.run_refinement()

    def test_completed_checkpoint_reuse_is_deterministic_and_checks_identity(self):
        initial = deepcopy(self.model.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "detector.pt"
            first = self.run_refinement(checkpoint_path=path)
            selected = deepcopy(self.model.detector.state_dict())
            self.model.load_state_dict(initial)
            with patch("grace_rul.detection.WindowDataset", side_effect=AssertionError("must reuse")):
                second = refine_bypassed_detector(self.model, self.fit, self.validation,
                                                  self.scaler, self.config, 17, checkpoint_path=path)
            self.assertTrue(second.attrs["audit"]["checkpoint_reused"])
            pd.testing.assert_frame_equal(first, second)
            for name, value in selected.items():
                torch.testing.assert_close(value, self.model.detector.state_dict()[name], rtol=0, atol=0)
            # A second invocation on the already-selected detector is also valid.
            self.run_refinement(checkpoint_path=path)
            changed = self.validation.copy(); changed.loc[0, "cycle"] += 1
            with self.assertRaisesRegex(ValueError, "does not match"):
                refine_bypassed_detector(self.model, self.fit, changed, self.scaler,
                                         self.config, 17, checkpoint_path=path)

    def test_interrupted_checkpoint_resumes_identically_at_epoch_boundary(self):
        initial = deepcopy(self.model.state_dict())
        expected = self.run_refinement()
        selected = deepcopy(self.model.detector.state_dict())
        self.model.load_state_dict(initial)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "interrupted.pt"
            calls = 0
            def interrupted(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise RuntimeError("fixture interruption after one saved epoch")
                return select_detector_thresholds(*args, **kwargs)
            with patch("grace_rul.detection.select_detector_thresholds", side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, "fixture interruption"):
                    self.run_refinement(checkpoint_path=path)
            self.model.load_state_dict(initial)
            resumed = self.run_refinement(checkpoint_path=path)
        pd.testing.assert_frame_equal(expected, resumed)
        self.assertEqual(expected.attrs["audit"], resumed.attrs["audit"])
        for name, value in selected.items():
            torch.testing.assert_close(value, self.model.detector.state_dict()[name], rtol=0, atol=0)

    def test_threshold_audit_uses_all_validation_contexts(self):
        with patch("grace_rul.detection.WindowDataset", TinyFaultWindows):
            thresholds, audit = select_detector_thresholds(self.model, self.validation,
                                                           self.scaler, self.config, 17, return_audit=True)
        self.assertEqual(len(thresholds), 21)
        self.assertEqual(set(audit["contexts"]), set(DETECTOR_FAMILIES))
        self.assertEqual(audit["contexts"]["outage"]["positive_cells"], 0)
        self.assertGreater(audit["contexts"]["mixed"]["positive_cells"], 0)


if __name__ == "__main__":
    unittest.main()
