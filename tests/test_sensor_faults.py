"""Mechanism/physics regression checks; these require NumPy, not PyTorch."""

import unittest

import numpy as np

from grace_rul.corruption import FAULTS, MIXED_SCENARIOS, SENSOR_DOMAINS, corrupt_trajectory


class StructuredSensorFaults(unittest.TestCase):
    def simulate(self, reference, family, seed=17, **kwargs):
        return corrupt_trajectory(reference, np.random.default_rng(seed), family,
                                  start_horizon=0, return_metadata=True, **kwargs)

    @staticmethod
    def profile():
        return {"centers": [[0., 0., 0.], [2., 0., 0.]],
                "sensor_mean": [np.arange(21).tolist(), (100 + np.arange(21)).tolist()],
                "sensor_std": [np.ones(21).tolist(), (np.ones(21) * 10).tolist()],
                "sensor_fault_scale": (np.ones(21) * 2).tolist()}

    def test_faults_remain_in_related_domains_and_preserve_commands(self):
        reference = np.random.default_rng(2).normal(size=(160, 24)).astype(np.float32)
        for family in [*FAULTS, "mixed"]:
            for seed in range(30):
                with self.subTest(family=family, seed=seed):
                    damaged, altered, meta = self.simulate(reference, family, seed, severity=.3)
                    selected = {int(name.split('_')[1]) - 1 for name in meta['channels']}
                    self.assertTrue(selected <= set(SENSOR_DOMAINS[meta['domain']]))
                    self.assertTrue(set(np.flatnonzero(altered.any(0))) <= selected)
                    np.testing.assert_array_equal(damaged[:, [0, 1, 2, 20, 21]], reference[:, [0, 1, 2, 20, 21]])
                    np.testing.assert_array_equal(altered, damaged[:, 3:] != reference[:, 3:])

    def test_noise_is_temporally_correlated_with_a_shared_error_source(self):
        reference = np.zeros((4000, 24), np.float32)
        for seed in range(30):
            damaged, altered, meta = self.simulate(reference, 'noise', seed, severity=.5)
            channels = np.flatnonzero(altered.any(0))
            if len(channels) >= 2:
                error = damaged[200:, 3 + channels]
                self.assertGreater(np.corrcoef(error[:-1, 0], error[1:, 0])[0, 1], .8)
                self.assertGreater(np.corrcoef(error[:, 0], error[:, 1])[0, 1], .65)
                self.assertLess(np.corrcoef(error[:, 0], error[:, 1])[0, 1], .98)
                break
        else:
            self.fail('No multi-channel domain sampled.')

    def test_drift_alters_zero_readings_and_has_a_fixed_additive_bound(self):
        reference = np.zeros((150, 24), np.float32)
        damaged, altered, meta = self.simulate(reference, 'drift', severity=.3, drift_ramp_cycles=20)
        channels = np.flatnonzero(altered.any(0))
        self.assertTrue(len(channels))
        self.assertFalse(altered[0].any())
        np.testing.assert_allclose(damaged[10, 3 + channels], .5 * damaged[20, 3 + channels])
        np.testing.assert_array_equal(damaged[20:, 3 + channels], np.tile(damaged[20, 3 + channels], (130, 1)))
        self.assertTrue((np.abs(damaged[:, 3:]) <= 2.5 * .3).all())

    def test_intermittent_loss_is_group_coupled_and_bursty(self):
        reference = np.zeros((1500, 24), np.float32)
        _, altered, _ = self.simulate(reference, 'intermittent', severity=.5)
        channels = np.flatnonzero(altered.any(0))
        for channel in channels:
            np.testing.assert_array_equal(altered[:, channel], altered[:, channels[0]])
        down = altered[:, channels[0]]
        starts = np.flatnonzero(np.diff(np.r_[False, down, False].astype(int)) == 1)
        ends = np.flatnonzero(np.diff(np.r_[False, down, False].astype(int)) == -1)
        self.assertGreater(len(starts), 5)
        self.assertTrue(((ends - starts)[:-1] >= 2).all())
        self.assertTrue((~down).any())

    def test_bias_is_constant_in_physical_units_across_regimes(self):
        reference = np.zeros((120, 24), np.float32)
        regimes = np.arange(120) % 2
        profile = self.profile()
        damaged, altered, meta = self.simulate(reference, 'bias', profile=profile, regimes=regimes)
        channels = np.flatnonzero(altered.any(0))
        raw_error = damaged[:, 3 + channels] * np.asarray(profile['sensor_std'])[regimes][:, channels]
        np.testing.assert_allclose(raw_error, np.tile(raw_error[0], (120, 1)), atol=1e-7)
        np.testing.assert_allclose(raw_error[0], meta['mechanisms'][0]['raw_offset_bound'], rtol=1e-6)

    def test_stuck_reading_is_held_in_physical_units_across_regimes(self):
        reference = np.zeros((120, 24), np.float32)
        regimes = np.arange(120) % 2
        profile = self.profile()
        damaged, _, meta = self.simulate(reference, 'stuck', profile=profile, regimes=regimes)
        channels = [int(name.split('_')[1]) - 1 for name in meta['channels']]
        mu, sd = np.asarray(profile['sensor_mean'])[regimes], np.asarray(profile['sensor_std'])[regimes]
        raw = damaged[:, 3 + np.asarray(channels)] * sd[:, channels] + mu[:, channels]
        np.testing.assert_allclose(raw, np.tile(raw[0], (120, 1)), atol=3e-6)
        self.assertTrue(np.isfinite(damaged).all())

    def test_noise_amplitude_responds_only_to_observed_operating_transitions(self):
        stable = np.zeros((1000, 24), np.float32)
        changing = stable.copy()
        changing[500:, 0] = np.arange(500) % 2 * 3.
        baseline, _, _ = self.simulate(stable, 'noise')
        stressed, _, _ = self.simulate(changing, 'noise')
        np.testing.assert_array_equal(stressed[:500], baseline[:500])
        self.assertGreater(np.linalg.norm(stressed[550:, 3:]), 1.8 * np.linalg.norm(baseline[550:, 3:]))

    def test_mixed_faults_have_one_related_source(self):
        reference = np.zeros((150, 24), np.float32)
        seen = set()
        for seed in range(60):
            _, _, meta = self.simulate(reference, 'mixed', seed)
            families = tuple(row['family'] for row in meta['mechanisms'])
            self.assertEqual(families, MIXED_SCENARIOS[meta['scenario']])
            seen.add(meta['scenario'])
        self.assertEqual(seen, set(MIXED_SCENARIOS))

    def test_severity_sweeps_keep_the_same_source_onset_and_nested_channels(self):
        reference = np.zeros((200, 24), np.float32)
        for family in [*FAULTS, 'mixed']:
            for seed in range(15):
                previous, previous_meta = None, None
                for severity in [.1, .2, .3, .5]:
                    damaged, _, meta = corrupt_trajectory(reference, np.random.default_rng(seed), family, severity,
                                                          return_metadata=True)
                    if previous_meta is not None:
                        self.assertEqual(meta['domain'], previous_meta['domain'])
                        self.assertEqual(meta['onset_observed_offset'], previous_meta['onset_observed_offset'])
                        self.assertTrue(set(previous_meta['channels']) <= set(meta['channels']))
                        if family in ['noise', 'bias', 'drift']:
                            common = [2 + int(name.split('_')[1]) for name in previous_meta['channels']]
                            np.testing.assert_allclose(damaged[:, common] / severity,
                                                       previous[:, common] / previous_meta['severity'], atol=1e-6)
                    previous, previous_meta = damaged, meta

    def test_all_profiles_and_mechanisms_preserve_observed_prefixes(self):
        reference = np.random.default_rng(8).normal(size=(220, 24)).astype(np.float32)
        regimes = np.arange(220) % 2
        for family in [*FAULTS, 'mixed', None, 'clean']:
            for seed in range(20):
                options = dict(profile=self.profile(), regimes=regimes, return_metadata=True)
                full, labels, _ = corrupt_trajectory(reference, np.random.default_rng(seed), family, .3, **options)
                for length in [1, 8, 30, 100, 180]:
                    with self.subTest(family=family, seed=seed, length=length):
                        short, short_labels, _ = corrupt_trajectory(reference[:length], np.random.default_rng(seed), family, .3,
                                                                   **{**options, 'regimes': regimes[:length]})
                        np.testing.assert_array_equal(short, full[:length])
                        np.testing.assert_array_equal(short_labels, labels[:length])

    def test_invalid_profiles_and_mechanism_parameters_fail_explicitly(self):
        reference = np.zeros((20, 24), np.float32)
        for overrides in [{'severity': 0}, {'severity': .6}, {'noise_autocorrelation': 1.},
                          {'transition_gain': -1.}, {'intermittent_mean_burst': 1},
                          {'profile': {}}, {'profile': self.profile(), 'regimes': np.zeros(20) }]:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.simulate(reference, 'noise', **overrides)


if __name__ == '__main__':
    unittest.main()
