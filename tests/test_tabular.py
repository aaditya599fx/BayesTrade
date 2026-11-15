import itertools
import unittest

import numpy as np

from bayestrade.tabular import (Beta, Dirichlet, backward_induction, belief_update, condition,
                                hyperstate_planner, pomdp_exact, probabilities, q_learning)


class TabularTests(unittest.TestCase):
    def test_probability_validation(self):
        for invalid in ([], [.5, .6], [1.1, -.1], [np.nan, .9], [np.inf, 0]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                probabilities(invalid)

    def test_belief_hand_computation(self):
        np.testing.assert_allclose(belief_update([.6, .4], np.eye(2), [.8, .3]), [.8, .2])

    def test_nonidentity_transition_and_uninformative_emission(self):
        transition = np.array([[.7, .3], [.4, .6]])
        np.testing.assert_allclose(
            belief_update([.25, .75], transition, [.3, .3]), [.475, .525])

    def test_impossible_evidence(self):
        with self.assertRaises(ValueError):
            condition([1, 0], [0, 1])

    def test_mdp_matches_brute_force(self):
        transition = np.eye(2)[[[0, 1], [1, 0]]]
        costs = np.array([[[2, 1], [3, 0]], [[1, 4], [0, 3]], [[1, 3], [2, 0]]])
        terminal = np.array([3, 0])
        values, actions = backward_induction(transition, costs, terminal)
        for start in range(2):
            brute = []
            for route in itertools.product(range(2), repeat=3):
                state = start
                cost = 0
                for step, action in enumerate(route):
                    cost += costs[step, state, action]
                    state = transition[state, action].argmax()
                brute.append((cost + terminal[state], route[0]))
            self.assertEqual((values[0, start], actions[0, start]), min(brute))

    def test_action_mask(self):
        _, policy = backward_induction([[[1], [1]]], [[[10, 0]]], [0], [[True, False]])
        self.assertEqual(policy[0, 0], 0)

    def test_q_learning_recovers_deterministic_policy(self):
        transition = np.array([[[1], [1]]])
        costs = np.array([[[2, 1]], [[1, 3]]])
        _, actual = q_learning(transition, costs, [0], 1500, 42)
        _, expected = backward_induction(transition, costs, [0])
        np.testing.assert_array_equal(actual, expected)

    def test_exact_pomdp_values_information(self):
        transition = np.broadcast_to(np.eye(2)[:, None, :], (2, 3, 2))
        observations = np.array([[[1, 0], [1, 0]], [[1, 0], [1, 0]], [[1, 0], [0, 1]]])
        costs = [[0, 10, 3], [10, 0, 3]]
        self.assertEqual(pomdp_exact(transition, observations, costs, [0, 0], [.5, .5], 2),
                         (3.0, 2))

    def test_conjugate_updates(self):
        self.assertAlmostEqual(Beta(2, 2).observe(True).mean, .6)
        self.assertLess(Beta(2, 2).observe(True).observe(False).variance, Beta(2, 2).variance)
        np.testing.assert_allclose(Dirichlet((1, 2, 3)).observe(1).mean, [1/7, 3/7, 3/7])

    def test_exact_hyperstate_value(self):
        def branches(state, action):
            inventory, prior = state
            if action == "market":
                yield 1, 2, (0, prior)
            else:
                for filled, prob in enumerate((1 - prior.mean, prior.mean)):
                    yield prob, 0, (inventory - filled, prior.observe(filled))

        planner = hyperstate_planner(lambda s: ("market", "post") if s[0] else ("done",),
                                    lambda s, a: [(1, 0, s)] if a == "done" else branches(s, a),
                                    lambda s: 6 * s[0])
        value, action = planner(2, (1, Beta()))
        self.assertAlmostEqual(value, 1.0)
        self.assertEqual(action, "post")
