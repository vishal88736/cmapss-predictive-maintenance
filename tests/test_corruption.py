"""Regression checks for causal, persistent synthetic sensor faults."""

from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from grace_rul.config import COLUMNS, DEFAULT_CONFIG, FEATURE_COLUMNS
from grace_rul.data import FAULTS, WindowDataset, corrupt_trajectory, select_background_indices, window_view


class CausalCorruption(unittest.TestCase):
    def test_block_faults_recur_in_late_observed_history(self):
        reference = np.ones((600, 24), dtype=np.float32)
        for seed in range(12):
            with self.subTest(seed=seed):
                damaged, labels = corrupt_trajectory(reference, np.random.default_rng(seed), "block", .2,
                                                     start_horizon=0)
                self.assertTrue(labels[-70:].any(), "Late endpoints must not all become clean after an early block.")
                self.assertTrue(np.isfinite(damaged[150:, 3:]).all(1).any(), "Temporary blocks must leave intact intervals.")
                short, short_labels = corrupt_trajectory(reference[:350], np.random.default_rng(seed), "block", .2,
                                                         start_horizon=0)
                np.testing.assert_array_equal(short, damaged[:350])
                np.testing.assert_array_equal(short_labels, labels[:350])

    def test_unseen_future_does_not_change_corrupted_prefix(self):
        reference = np.random.default_rng(8).normal(size=(240, 24)).astype(np.float32)
        for family in [*FAULTS, "mixed", None, "clean"]:
            for seed in range(40):
                full, full_labels = corrupt_trajectory(reference, np.random.default_rng(seed), family, .3)
                for length in [1, 8, 30, 60, 100, 180]:
                    with self.subTest(family=family, seed=seed, length=length):
                        short, labels = corrupt_trajectory(reference[:length], np.random.default_rng(seed), family, .3)
                        np.testing.assert_array_equal(short, full[:length])
                        np.testing.assert_array_equal(labels, full_labels[:length])

    def test_drift_reaches_its_bound_after_fixed_elapsed_cycles(self):
        reference = np.ones((80, 24), dtype=np.float32)
        damaged, labels = corrupt_trajectory(reference, np.random.default_rng(17), "drift", .3,
                                             start_horizon=0, drift_ramp_cycles=10)
        channels = np.flatnonzero(labels.any(0))
        self.assertTrue(len(channels))
        self.assertFalse(labels[0].any())
        final_change = damaged[10, 3 + channels] - 1.
        np.testing.assert_allclose(damaged[5, 3 + channels] - 1., .5 * final_change, rtol=0, atol=1e-7)
        np.testing.assert_array_equal(damaged[10:, 3 + channels], np.tile(damaged[10, 3 + channels], (70, 1)))
        self.assertTrue((np.abs(final_change) <= .3 + 1e-7).all())

    def test_onset_after_observed_history_does_not_move_to_its_end(self):
        reference = np.ones((3, 24), dtype=np.float32)
        for family in ["outage", "block", "bias", "drift"]:
            with self.subTest(family=family):
                damaged, labels = corrupt_trajectory(reference, np.random.default_rng(42), family,
                                                     start_horizon=1_000_000)
                np.testing.assert_array_equal(damaged, reference)
                self.assertFalse(labels.any())

    def test_changed_entry_labels_and_settings_are_preserved(self):
        reference = np.ones((80, 24), dtype=np.float32)
        reference[:, 3::2] = 0.
        for family in [*FAULTS, "mixed", "clean"]:
            with self.subTest(family=family):
                damaged, labels = corrupt_trajectory(reference, np.random.default_rng(42), family,
                                                     start_horizon=0)
                np.testing.assert_array_equal(damaged[:, :3], reference[:, :3])
                np.testing.assert_array_equal(labels, damaged[:, 3:] != reference[:, 3:])
                np.testing.assert_array_equal(damaged[:, 3:][~labels], reference[:, 3:][~labels])
                if family == "clean":
                    self.assertFalse(labels.any())
                elif family == "drift":
                    self.assertFalse(labels[:, ::2].any())
                else:
                    self.assertTrue(labels.any())

    def test_dataset_prefix_inputs_do_not_depend_on_future_rul(self):
        frame = pd.DataFrame(np.ones((120, len(COLUMNS))), columns=COLUMNS)
        frame["unit"] = 1
        frame["cycle"] = np.arange(1, 121)
        scaler = SimpleNamespace(transform=lambda group: (group[FEATURE_COLUMNS].to_numpy(np.float32), None))
        config = {**DEFAULT_CONFIG, "fault_start_horizon": 0, "drift_ramp_cycles": 10}
        for family in [*FAULTS, "mixed", None]:
            with self.subTest(family=family):
                short = WindowDataset(frame.iloc[:60], scaler, config, family=family)
                long = WindowDataset(frame, scaler, config, family=family)
                for epoch in [0, 1]:
                    short.set_epoch(epoch); long.set_epoch(epoch)
                    a, b = short[29], long[29]
                    np.testing.assert_array_equal(a["x"].numpy(), b["x"].numpy())
                    np.testing.assert_array_equal(a["altered"].numpy(), b["altered"].numpy())
                    self.assertNotEqual(float(a["raw_target"].numpy()), float(b["raw_target"].numpy()))

    def test_overlapping_windows_keep_identical_fault_history(self):
        frame = pd.DataFrame(np.ones((120, len(COLUMNS))), columns=COLUMNS)
        frame["unit"] = 1
        frame["cycle"] = np.arange(1, 121)
        scaler = SimpleNamespace(transform=lambda group: (group[FEATURE_COLUMNS].to_numpy(np.float32), None))
        config = {**DEFAULT_CONFIG, "fault_start_horizon": 0}
        for family in [*FAULTS, "mixed", None]:
            with self.subTest(family=family):
                data = WindowDataset(frame, scaler, config, family=family)
                for epoch in [0, 1]:
                    data.set_epoch(epoch)
                    a, b = data[80], data[81]
                    np.testing.assert_array_equal(a["x"].numpy()[1:], b["x"].numpy()[:-1])
                    np.testing.assert_array_equal(a["altered"].numpy()[1:], b["altered"].numpy()[:-1])

    def test_invalid_timing_parameters_are_rejected(self):
        reference = np.ones((30, 24), dtype=np.float32)
        for values in [{"start_horizon": -1}, {"start_horizon": True}, {"start_horizon": 1.5},
                          {"drift_ramp_cycles": 0}, {"drift_ramp_cycles": True}, {"drift_ramp_cycles": 1.5},
                          {"block_gap_cycles": 0}, {"block_gap_cycles": True}, {"block_gap_cycles": 1.5}]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                corrupt_trajectory(reference, np.random.default_rng(42), "drift", **values)

    def test_background_selection_spreads_engines_regimes_and_life_stages(self):
        frames = []
        for unit in range(1, 21):
            frame = pd.DataFrame(np.ones((180, len(COLUMNS))), columns=COLUMNS)
            frame["unit"], frame["cycle"] = unit, np.arange(1, 181)
            frames.append(frame)
        scaler = SimpleNamespace(transform=lambda group: (group[FEATURE_COLUMNS].to_numpy(np.float32),
            ((group.cycle.to_numpy() // 20 + int(group.unit.iloc[0])) % 3)))
        data = WindowDataset(pd.concat(frames), scaler, DEFAULT_CONFIG, mode="correction", family="clean")
        indices, metadata = select_background_indices(data, 12, 42)
        self.assertEqual((indices, metadata), select_background_indices(data, 12, 42))
        self.assertEqual(len(set(indices)), 12)
        self.assertEqual(len({row["unit"] for row in metadata}), 12)
        self.assertEqual({row["regime"] for row in metadata}, {0, 1, 2})
        self.assertGreaterEqual(len({row["life_stage"] for row in metadata}), 3)
        self.assertTrue({row["unit"] for row in metadata} <= set(data.engines[i]["unit"] for i, _ in data.samples))

    def test_comparator_contexts_share_endpoint_and_fault_realization(self):
        frame = pd.DataFrame(np.ones((220, len(COLUMNS))), columns=COLUMNS)
        frame["unit"], frame["cycle"] = 1, np.arange(1, 221)
        scaler = SimpleNamespace(transform=lambda group: (group[FEATURE_COLUMNS].to_numpy(np.float32), None))
        original = WindowDataset(frame, scaler, DEFAULT_CONFIG, mode="test", family="block")
        shorter = window_view(original, 30)
        self.assertIs(window_view(original, 70), original)
        self.assertIs(original.cache, shorter.cache)
        first, second = original[0], shorter[0]
        self.assertEqual((first["unit"], first["cycle"]), (second["unit"], second["cycle"]))
        np.testing.assert_array_equal(first["x"].numpy()[-30:], second["x"].numpy())
        np.testing.assert_array_equal(first["altered"].numpy()[-30:], second["altered"].numpy())
        self.assertEqual(original.config["sequence_length"], 70)


if __name__ == "__main__":
    unittest.main()
