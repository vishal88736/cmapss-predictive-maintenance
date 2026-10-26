"""Finite-sample, engine grouping, and target-contract calibration safeguards."""

import json
import math
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from grace_rul.calibration import (calibrate_endpoint_intervals, conformal_rank,
                                   interval_bounds, split_conformal_quantile)


class TransformOnlyScaler:
    def __init__(self, fitting_units=(999,)):
        self.state = {"fit_units": list(fitting_units)}

    def fit(self, *args, **kwargs):
        raise AssertionError("Calibration refit preprocessing")

    def transform(self, frame):
        return np.zeros((len(frame), 24), dtype=np.float32), np.zeros(len(frame), dtype=int)


def trajectories(lengths):
    return pd.DataFrame([{"unit": unit, "cycle": cycle}
                         for unit, length in enumerate(lengths, start=1)
                         for cycle in range(1, length + 1)])


def records_for(dataset, prediction):
    rows = []
    for engine_index, end in dataset.samples:
        engine = dataset.engines[engine_index]
        raw = float(engine["targets"][end])
        rows.append({"unit": engine["unit"], "cycle": int(engine["cycles"][end]),
                     "prediction": float(prediction(engine, end)), "raw_target": raw,
                     "target": min(dataset.config["train_rul_cap"], raw), "altered_entries": 0})
    return pd.DataFrame(rows)


class CalibrationProtocol(unittest.TestCase):
    def setUp(self):
        self.config = {"interval_alpha": .1, "train_rul_cap": 125., "maximum_windows": 1,
                       "maximum_batches": 1, "window_stride": 7}

    def calibrate(self, frame=None, config=None, prediction=None, evaluator=None, seed=42):
        frame = trajectories([3] * 10) if frame is None else frame
        prediction = prediction or (lambda engine, end: 0.)
        if evaluator is None:
            def evaluator(model, dataset, settings):
                self.assertEqual(settings["window_stride"], 1)
                self.assertIsNone(settings["maximum_windows"])
                self.assertIsNone(settings["maximum_batches"])
                self.assertEqual(dataset.family, "clean")
                return {}, records_for(dataset, prediction)
        return calibrate_endpoint_intervals(Mock(), frame, TransformOnlyScaler(),
                                             config or self.config, seed, evaluator=evaluator)

    def test_order_statistic_uses_engine_count_and_finite_sample_rank(self):
        scores = np.arange(1., 11.)
        self.assertEqual(conformal_rank(10, .1), 10)
        self.assertEqual(split_conformal_quantile(scores, .1), 10.)
        self.assertEqual(conformal_rank(21, .1), 20)
        self.assertEqual(conformal_rank(20, .1), 19)
        self.assertEqual(conformal_rank(9, .1), 9)

    def test_small_fleet_requires_unbounded_interval_and_json_nulls(self):
        calibration, _ = self.calibrate(trajectories([25, 26]))
        self.assertEqual(calibration["engines"], 2)
        self.assertEqual(calibration["rank"], 3)
        self.assertTrue(calibration["unbounded"])
        self.assertIsNone(calibration["radius"])
        self.assertIsNone(calibration["lower_offset"])
        json.dumps(calibration, allow_nan=False)
        lower, upper = interval_bounds([0., 40.], calibration)
        np.testing.assert_array_equal(lower, [0., 0.])
        self.assertTrue(np.isposinf(upper).all())
        self.assertTrue(math.isinf(split_conformal_quantile([1., 2.], .1)))

    def test_ties_do_not_interpolate_or_shrink(self):
        self.assertEqual(split_conformal_quantile([1.] * 8 + [9., 9.], .2), 9.)
        self.assertEqual(split_conformal_quantile([0.] * 10, .1), 0.)

    def test_all_prefixes_are_scored_once_and_grouped_by_independent_engine(self):
        frame = trajectories([3] * 9 + [20])
        calibration, records = self.calibrate(frame, prediction=lambda engine, end: float(engine["unit"]))
        self.assertEqual(calibration["engines"], 10)
        self.assertEqual(calibration["endpoints"], len(frame))
        self.assertEqual(len(records), len(frame))
        self.assertEqual(calibration["policy"], "engine_prefix_max")
        self.assertEqual(calibration["radius"], 10.)
        self.assertEqual(len(calibration["protocols"]["capped_125"]["engine_scores"]), 10)
        self.assertFalse(records.duplicated(["unit", "cycle"]).any())
        self.assertEqual(self.config["maximum_windows"], 1)
        self.assertEqual(self.config["window_stride"], 7)

    def test_longer_engine_does_not_count_as_many_calibration_units(self):
        calibration, _ = self.calibrate(trajectories([2] * 9 + [100]),
                                      prediction=lambda engine, end: float(engine["targets"][end]) + engine["unit"])
        self.assertEqual(calibration["n"], 10)
        self.assertEqual(calibration["rank"], 10)
        self.assertEqual(calibration["radius"], 10.)

    def test_capped_and_raw_targets_have_separate_contracts(self):
        calibration, _ = self.calibrate(trajectories([200] * 10),
                                      prediction=lambda engine, end: min(125., float(engine["targets"][end])))
        self.assertEqual(calibration["primary_protocol"], "capped_125")
        self.assertEqual(calibration["protocols"]["capped_125"]["radius"], 0.)
        self.assertEqual(calibration["protocols"]["official_uncapped"]["radius"], 74.)
        np.testing.assert_array_equal(interval_bounds([125.], calibration)[1], [125.])
        np.testing.assert_array_equal(interval_bounds([125.], calibration, "official_uncapped")[1], [199.])
        with self.assertRaisesRegex(ValueError, "No interval calibration"):
            interval_bounds([125.], calibration, "another_target")

    def test_complete_failure_endpoints_are_not_mistaken_for_test_censoring(self):
        calibration, records = self.calibrate(trajectories([40] * 10))
        self.assertGreater(float(records.raw_target.max()), 0.)
        self.assertEqual(calibration["endpoints"], 400)
        self.assertIn("exchangeable complete engine trajectories", calibration["scope"])
        with self.assertRaisesRegex(ValueError, "Unknown calibration policy"):
            self.calibrate(config={**self.config, "calibration_policy": "last_observed"})

    def test_random_prefix_policy_is_explicit_and_seed_reproducible(self):
        config = {**self.config, "calibration_policy": "random_prefix"}
        a, first = self.calibrate(trajectories([80] * 10), config)
        _, second = self.calibrate(trajectories([80] * 10), config)
        self.assertEqual(a["endpoints"], 10)
        self.assertEqual(first.unit.nunique(), 10)
        self.assertEqual(first.cycle.tolist(), second.cycle.tolist())
        self.assertIn("same prefix-sampling policy", a["scope"])
        self.assertEqual(a["coverage_unit"], "engine_random_prefix_marginal")

    def test_neither_fits_nor_opens_validation_or_official_test_files(self):
        frame = trajectories([3] * 10)
        captured = {}

        def evaluator(model, dataset, config):
            captured["units"] = {engine["unit"] for engine in dataset.engines}
            return {}, records_for(dataset, lambda engine, end: 0.)

        with patch("pandas.read_csv", side_effect=AssertionError("Unexpected data read")), \
                patch("numpy.loadtxt", side_effect=AssertionError("Unexpected label read")):
            self.calibrate(frame, evaluator=evaluator)
        self.assertEqual(captured["units"], set(frame.unit))

    def test_fitting_engine_overlap_is_rejected_before_evaluation(self):
        evaluator = Mock(side_effect=AssertionError("Evaluation should not execute"))
        with self.assertRaisesRegex(ValueError, "overlap"):
            calibrate_endpoint_intervals(Mock(), trajectories([3] * 10),
                                         TransformOnlyScaler([1]), self.config, 42, evaluator=evaluator)
        evaluator.assert_not_called()

    def test_missing_engine_or_prefix_cannot_silently_reduce_calibration_set(self):
        def truncated(model, dataset, config):
            return {}, records_for(dataset, lambda engine, end: 0.).iloc[:-1]
        with self.assertRaisesRegex(ValueError, "every declared prefix"):
            self.calibrate(evaluator=truncated)

    def test_duplicate_prefix_or_changed_target_is_rejected(self):
        for mutation, message in [
                (lambda records: pd.concat([records, records.iloc[[0]]]), "more than once"),
                (lambda records: records.assign(target=999.), "target convention")]:
            def incorrect(model, dataset, config):
                return {}, mutation(records_for(dataset, lambda engine, end: 0.))
            with self.assertRaisesRegex(ValueError, message):
                self.calibrate(evaluator=incorrect)

    def test_legacy_offsets_keep_their_original_behavior(self):
        old = {"lower_offset": -30., "upper_offset": -10.}
        lo, hi = interval_bounds([0., 20., 100.], old)
        np.testing.assert_array_equal(lo, [0., 0., 70.])
        np.testing.assert_array_equal(hi, [0., 10., 90.])
        with self.assertRaisesRegex(ValueError, "no uncapped"):
            interval_bounds([100.], old, "official_uncapped")

    def test_invalid_scores_alpha_and_bounds_fail_clearly(self):
        for scores, alpha in [([], .1), ([np.nan], .1), ([-1.], .1), ([[1.]], .1),
                              ([1.], 0.), ([1.], 1.), ([1.], True)]:
            with self.assertRaises(ValueError):
                split_conformal_quantile(scores, alpha)
        with self.assertRaises(ValueError):
            interval_bounds([np.nan], {"lower_offset": -1., "upper_offset": 1.})


if __name__ == "__main__":
    unittest.main()
