import csv
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from bayestrade.market import Fill, Order, SCENARIOS, execute, make_tape, shortfall
from bayestrade.policies import Baseline, BeliefPolicy, ObservedPolicy
from bayestrade.study import paired_interval, run


class Spy:
    def __init__(self, action):
        self.action = action
        self.views = []

    def act(self, view):
        self.views.append(view)
        return self.action(view)

    def observe(self, order, filled):
        pass


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.scenario = SCENARIOS["switching"]
        self.tape = make_tape(self.scenario, 8, 12)

    def test_ledger_shortfall_by_hand(self):
        fills = [Fill(0, "market", 2, 100.02, .60), Fill(8, "cleanup", 1, 100.09, .30)]
        self.assertAlmostEqual(shortfall(fills, 3), 13.9 / 3)

    def test_wait_gets_single_cleanup(self):
        result = execute(Spy(lambda _: Order("wait")), self.tape, 6, self.scenario)
        self.assertEqual([(x.step, x.kind, x.lots) for x in result["ledger"]], [(8, "cleanup", 6)])
        self.assertEqual(result["inventory_risk"], 1)
        self.assertAlmostEqual(result["fees_dollars"], 1.8)

    def test_queue_and_price(self):
        tape = replace(self.tape, flow=(.99,) * 8, arrival=(0., 1.) * 4)
        result = execute(Spy(lambda _: Order("post", 1)), tape, 6, self.scenario)
        self.assertEqual([row[4] for row in result["events"]], [1, 0] * 4)
        for step, _, kind, _, filled, _, queue, sell in result["events"]:
            self.assertEqual(filled, int(sell > queue))
        self.assertLess(result["ledger"][0].price, 100)

    def test_market_respects_depth(self):
        result = execute(Baseline("immediate", 6), self.tape, 6, self.scenario)
        self.assertEqual(sum(fill.lots for fill in result["ledger"]), 6)
        for step, remaining, _, request, filled, depth, *_ in result["events"]:
            self.assertLessEqual(filled, min(request, depth, remaining))

    def test_info_boundary(self):
        spy = Spy(lambda _: Order("wait"))
        execute(spy, self.tape, 6, self.scenario)
        for view in spy.views:
            self.assertIsNone(view.regime)
            self.assertFalse(hasattr(view, "fill"))
            self.assertFalse(hasattr(view, "future_volume"))

    def test_uninformative_signal_holds_predictive(self):
        from bayestrade.market import View
        policy = BeliefPolicy(8, 6, 21, signal=(.5, .5))
        policy.act(View(0, 8, 6, True, 800))
        self.assertAlmostEqual(policy.belief, .4)
        policy.observe(Order("wait"), 0)
        policy.act(View(1, 8, 6, False, 800))
        self.assertAlmostEqual(policy.belief, .6 * .13 + .4 * .80)

    def test_bamdp_updates_only_observed_post_regime(self):
        from bayestrade.market import View
        learner = ObservedPolicy(2, True, 3)
        prior = learner.posterior
        learner.act(View(0, 2, 2, True, 600, 1))
        learner.observe(Order("market", 1), 1)
        self.assertEqual(learner.posterior, prior)
        learner.observe(Order("post", 1), 0)
        self.assertEqual(learner.posterior, (prior[0], prior[1].observe(False)))

    def test_seed_repeatability(self):
        self.assertEqual(self.tape, make_tape(self.scenario, 8, 12))
        first = execute(BeliefPolicy(8, 6, 21), self.tape, 6, self.scenario)
        second = execute(BeliefPolicy(8, 6, 21), self.tape, 6, self.scenario)
        self.assertEqual(first, second)

    def test_vwap_and_twap_differ_without_lookahead(self):
        from bayestrade.market import View
        view = View(1, 8, 5, False, 500)
        self.assertEqual(Baseline("vwap", 6).act(view), Order("market", 1))
        self.assertEqual(Baseline("vwap", 6).act(replace(view, volume=900)), Order("market", 1))
        initial = View(0, 8, 2, False, 500)
        self.assertEqual(Baseline("vwap", 2).act(initial), Order("wait"))
        self.assertEqual(Baseline("twap", 2).act(initial), Order("market", 1))

    def test_paired_constant_interval(self):
        self.assertEqual(paired_interval([-2] * 10, 101, 7), (-2., -2., -2.))

    def test_full_workflow_small_and_paired(self):
        config = json.loads((Path(__file__).resolve().parents[1] / "configs/study.json").read_text())
        config.update(horizon=3, target_lots=2, grid_points=11, train_episodes=3,
                      test_seeds=[56], episodes_per_seed=4, bootstrap_draws=30,
                      scenarios=["switching", "unknown_fill"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run(config, root / "a")
            run(config, root / "b")
            for name in ("episodes.csv", "paired.csv", "beliefs.csv.gz", "manifest.json"):
                self.assertEqual((root / "a" / name).read_bytes(), (root / "b" / name).read_bytes())
            with (root / "a" / "episodes.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 56)


if __name__ == "__main__":
    unittest.main()
