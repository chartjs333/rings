"""Behavioral checks for the experimental sparse phase-gated dynamics."""

import unittest

import numpy as np
from scipy import sparse

from rings_fly.dynamics import DynamicsConfig, GateSummary, RingFlyDynamics, StepResult


def always_open(**changes):
    values = dict(carrier_hz=0.0, phase_sensitivity=0.0, coupling=1.0, half_width=np.pi)
    values.update(changes)
    return DynamicsConfig(**values)


class DynamicsTests(unittest.TestCase):
    def test_orientation_and_synchronous_one_edge_per_tick(self):
        # 0 -> 1 -> 2; a synchronous update cannot reach 2 on the first hop.
        weights = sparse.csr_matrix(([2.0, 3.0], ([1, 2], [0, 1])), shape=(3, 3))
        network = RingFlyDynamics(weights, always_open())
        np.testing.assert_array_equal(network.step([1, 0, 0]).activity, [1, 0, 0])
        np.testing.assert_array_equal(network.step().activity, [0, 2, 0])
        np.testing.assert_array_equal(network.step().activity, [0, 0, 6])

    def test_signed_inhibition_and_nonnegative_activity(self):
        weights = sparse.csr_matrix(([-0.75], ([1], [0])), shape=(2, 2))
        network = RingFlyDynamics(weights, always_open())
        network.step([2, 0])
        np.testing.assert_allclose(network.step([0, 2]).activity, [0, 0.5])
        network.reset()
        network.step([2, 0])
        np.testing.assert_array_equal(network.step([0, 1]).activity, [0, 0])

    def test_silencing_clears_old_and_new_activity_without_delay(self):
        weights = sparse.csr_matrix(([1.0], ([1], [0])), shape=(2, 2))
        network = RingFlyDynamics(weights, always_open())
        network.step([3, 0])
        network.set_silenced([0])
        np.testing.assert_array_equal(network.activity, [0, 0])
        result = network.step([5, 0])
        np.testing.assert_array_equal(result.activity, [0, 0])
        np.testing.assert_array_equal(result.gates, [True, True])
        network.reset()
        self.assertTrue(network.silenced[0])
        network.set_silenced([])
        np.testing.assert_array_equal(network.step([1, 0]).activity, [1, 0])

    def test_gate_is_not_inferred_from_output(self):
        network = RingFlyDynamics(sparse.csr_matrix((2, 2)),
                                  DynamicsConfig(carrier_hz=0, phase_sensitivity=0),
                                  initial_phase=[0, np.pi])
        result = network.step([0, 1])
        np.testing.assert_array_equal(result.activity, [0, 0])
        np.testing.assert_array_equal(result.gates, [True, False])

    def test_fixed_gate_baseline_and_per_node_parameters(self):
        network = RingFlyDynamics(sparse.csr_matrix((2, 2)),
                                  DynamicsConfig(dt_s=0.1, carrier_hz=[0, 5],
                                                 phase_sensitivity=0, half_width=[0, 0]))
        first = network.step([100, 100])
        np.testing.assert_array_equal(first.gates, [True, False])
        np.testing.assert_allclose(first.phase, [0, -np.pi])
        second = network.step([0, 1000])
        np.testing.assert_array_equal(second.gates, [True, True])

    def test_input_changes_dynamic_phase(self):
        network = RingFlyDynamics(sparse.csr_matrix((2, 2)),
                                  DynamicsConfig(dt_s=0.1, carrier_hz=0,
                                                 phase_sensitivity=[0, 2], half_width=0.1))
        result = network.step([1, 1])
        np.testing.assert_allclose(result.phase, [0, 0.2])
        np.testing.assert_array_equal(result.gates, [True, False])

    def test_reset_and_owned_inputs(self):
        weights = sparse.csr_matrix(([1.0], ([1], [0])), shape=(2, 2))
        phases = np.array([0.2, -0.4])
        network = RingFlyDynamics(weights, always_open(), phases)
        phases[:] = 2
        weights.data[:] = 99
        network.step([2, 0])
        np.testing.assert_allclose(network.step().activity, [0, 2])
        network.reset()
        np.testing.assert_allclose(network.phase, [0.2, -0.4])
        np.testing.assert_array_equal(network.activity, [0, 0])
        network.reset(initial_phase=0)
        np.testing.assert_array_equal(network.phase, [0, 0])
        network.reset()
        np.testing.assert_allclose(network.phase, [0.2, -0.4])

    def test_snapshots_cannot_change_live_state(self):
        network = RingFlyDynamics(sparse.csr_matrix((1, 1)), always_open())
        result = network.step(1)
        with self.assertRaises(ValueError):
            result.activity[0] = 10
        snapshot = network.activity
        snapshot.flags.writeable = True
        snapshot[0] = 20
        self.assertEqual(network.activity[0], 1)

    def test_rejects_bad_configuration_and_inputs(self):
        weights = sparse.csr_matrix((2, 2))
        for changes in [dict(dt_s=0), dict(dt_s=np.nan), dict(carrier_hz=-1),
                        dict(phase_sensitivity=np.inf), dict(half_width=np.pi + 0.1),
                        dict(coupling=-0.1), dict(carrier_hz=[1, 2, 3])]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                RingFlyDynamics(weights, DynamicsConfig(**changes))
        with self.assertRaises(TypeError):
            RingFlyDynamics(np.zeros((2, 2)))
        with self.assertRaises(ValueError):
            RingFlyDynamics(sparse.csr_matrix((2, 3)))
        with self.assertRaises(ValueError):
            RingFlyDynamics(sparse.csr_matrix(([np.nan], ([0], [1])), shape=(2, 2)))
        network = RingFlyDynamics(weights, always_open())
        for value in ([1, np.nan], [1, np.inf], [1, 2, 3], [1j, 0]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                network.step(value)
        for indices in ([-1], [2], [0.5], [True], [[0]]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                network.set_silenced(indices)

    def test_overflow_rejects_without_partial_state_change(self):
        weights = sparse.csr_matrix(([1e308], ([0], [0])), shape=(1, 1))
        network = RingFlyDynamics(weights, always_open())
        network.step(2)
        before = network.activity
        with self.assertRaises(ValueError):
            network.step()
        np.testing.assert_array_equal(network.activity, before)

    def test_large_node_indices_are_not_uint16(self):
        n = 70_000
        network = RingFlyDynamics(sparse.csr_matrix((n, n)), always_open())
        network.set_silenced([65_536])
        result = network.step(1)
        self.assertEqual(result.activity.shape, (n,))
        self.assertEqual(result.activity[65_536], 0)
        self.assertEqual(result.activity[0], 1)
        self.assertTrue(np.all(result.gates))
        self.assertEqual(np.count_nonzero(result.activity), n - 1)


class GateSummaryTests(unittest.TestCase):
    @staticmethod
    def sample(gates, activity=None):
        gates = np.asarray(gates, dtype=bool)
        activity = gates.astype(float) if activity is None else np.asarray(activity, dtype=float)
        return StepResult(activity, gates, np.zeros(gates.size))

    def test_pooled_transitions_exclude_episode_boundary(self):
        summary = GateSummary(2)
        # Node 0: episode one 0,1,1,0; episode two 1,0.
        # Including the episode boundary would add a forbidden extra 0->1.
        # Check exact counts, since that extra transition would not change p01.
        for row in ([False, True], [True, True], [True, True], [False, True]):
            summary.update(self.sample(row))
        summary.end_episode()
        for row in ([True, True], [False, True]):
            summary.update(self.sample(row))
        stats = summary.result()
        self.assertEqual(stats.steps, 6)
        self.assertEqual(stats.transitions, 4)
        np.testing.assert_array_equal(stats.n00, [0, 0])
        np.testing.assert_array_equal(stats.n01, [1, 0])
        np.testing.assert_array_equal(stats.n10, [2, 0])
        np.testing.assert_array_equal(stats.n11, [1, 4])
        np.testing.assert_allclose(stats.p01[0], 1)
        np.testing.assert_allclose(stats.p10, [2 / 3, 0])
        self.assertTrue(np.isnan(stats.p01[1]))
        np.testing.assert_array_equal(stats.valid01, [True, False])
        np.testing.assert_array_equal(stats.valid10, [True, True])
        np.testing.assert_allclose(stats.mean_activity, [0.5, 1])
        np.testing.assert_allclose(stats.duty_cycle, [0.5, 1])
        np.testing.assert_array_equal(stats.n00 + stats.n01 + stats.n10 + stats.n11, [4, 4])

    def test_no_samples_and_reset_have_explicit_unsupported_probabilities(self):
        summary = GateSummary(1)
        empty = summary.result()
        self.assertEqual(empty.steps, 0)
        self.assertTrue(np.isnan(empty.mean_activity[0]))
        self.assertTrue(np.isnan(empty.p01[0]))
        self.assertFalse(empty.valid01[0])
        summary.update(self.sample([True]))
        self.assertEqual(summary.result().transitions, 0)
        summary.reset()
        summary.update(self.sample([False]))
        stats = summary.result()
        self.assertEqual(stats.steps, 1)
        self.assertEqual(stats.transitions, 0)
        self.assertFalse(stats.valid01[0])
        self.assertFalse(stats.valid10[0])

    def test_summary_rejects_nonfinite_activity_before_changing_counts(self):
        summary = GateSummary(1)
        with self.assertRaises(ValueError):
            summary.update(self.sample([True], [np.nan]))
        self.assertEqual(summary.steps, 0)
        self.assertEqual(summary.transitions, 0)


if __name__ == "__main__":
    unittest.main()
