"""Causal baselines, belief-grid execution, and exact beta-Bernoulli planning."""
from dataclasses import dataclass
from functools import cache
from math import ceil

import numpy as np

from .market import (DEPTH, FILL, LOT, HALF_SPREAD, MAKER, PERMANENT, SIGNAL, TAKER,
                     TEMPORARY, TRANSITION, Order, profile)
from .tabular import Beta, hyperstate_planner

RISK = .23
CLEANUP = 1.5
PRIOR_BAD = .40


def emission(wide, signal=SIGNAL):
    rates = np.asarray(signal)
    return rates if wide else 1 - rates


def market_edge(lots, depth, wide):
    return lots * (HALF_SPREAD[wide] + TAKER + TEMPORARY * lots / depth)


def depths(regime):
    return (DEPTH[regime] - 1, DEPTH[regime], DEPTH[regime] + 1)


def market_likelihood(request, filled):
    return np.array([sum(min(request, d) == filled for d in depths(z)) / 3
                     for z in range(2)])


def stage(fill, inventory, target, edge):
    """Expected local edge, permanent impact on future fills, risk preference."""
    remaining = inventory - fill
    return edge + PERMANENT * fill * remaining + RISK * remaining**2 / target


@dataclass
class Baseline:
    name: str
    target: int
    parameter: float = 1.0

    def act(self, view):
        q = view.remaining
        left = view.horizon - view.step
        if not q:
            return Order("wait")
        if self.name == "immediate":
            quantity = q
        elif self.name == "twap":
            quantity = ceil(q / left)
        elif self.name == "vwap":
            forecast = profile(view.horizon)
            scheduled = round(self.target * forecast[:view.step + 1].sum() / forecast.sum())
            quantity = max(0, scheduled - (self.target - q))
        elif self.name == "pov":
            quantity = max(ceil(q / left) if left <= 2 else 0,
                           round(self.parameter * view.volume / LOT))
        elif self.name == "reactive":
            if view.wide and q <= left * self.parameter and left > 1:
                return Order("post", 1)
            quantity = ceil(q / left)
        else:
            raise ValueError(f"Unknown baseline: {self.name}")
        return Order("market", min(q, quantity)) if quantity else Order("wait")

    def observe(self, order, filled):
        pass


@cache
def grid_plan(horizon, target, points, transition, signal):
    """Approximate DP on [remaining time, inventory, quote class, P(bad)]."""
    grid = np.linspace(0, 1, points)
    belief = np.column_stack((1 - grid, grid))
    transition = np.asarray(transition)
    value = np.zeros((horizon + 1, target + 1, 2, points))
    scores = np.full((*value.shape, target + 2), np.inf)
    for q in range(1, target + 1):
        for wide in range(2):
            value[0, q, wide] = (belief @ np.asarray(
                [np.mean([market_edge(q, d, wide) for d in depths(z)])
                 for z in range(2)])) + CLEANUP * q

    def forward(previous, joint):
        expected = np.zeros(points)
        predicted = joint @ transition
        for symbol in range(2):
            weights = predicted * emission(symbol, signal)
            mass = weights.sum(axis=1)
            posterior = np.divide(weights[:, 1], mass, out=np.zeros(points), where=mass > 0)
            expected += mass * np.interp(posterior, grid, previous[symbol])
        return expected

    for left in range(1, horizon + 1):
        scores[left, 0, :, :, 0] = 0
        for q in range(1, target + 1):
            for wide in range(2):
                current = scores[left, q, wide]
                previous = value[left - 1]
                for request in range(q + 1):
                    cost = np.zeros(points)
                    for z in range(2):
                        for depth in depths(z):
                            filled = min(request, depth)
                            edge = market_edge(filled, depth, wide)
                            cost += belief[:, z] * stage(filled, q, target, edge) / 3
                    for filled in range(request + 1):
                        joint = belief * market_likelihood(request, filled)
                        cost += forward(previous[q - filled], joint)
                    current[:, request] = cost
                cost = np.zeros(points)
                for filled, likelihood in enumerate((1 - np.array(FILL), np.array(FILL))):
                    edge = filled * (-HALF_SPREAD[wide] + MAKER)
                    joint = belief * likelihood
                    cost += joint.sum(axis=1) * stage(filled, q, target, edge)
                    cost += forward(previous[q - filled], joint)
                current[:, -1] = cost
                value[left, q, wide] = current.min(axis=1)
    grid.setflags(write=False)
    scores.setflags(write=False)
    return grid, scores


class BeliefPolicy:
    def __init__(self, horizon, target, points, *, memory=True, transition=TRANSITION,
                 signal=SIGNAL, prior_bad=PRIOR_BAD):
        self.target = target
        self.memory = memory
        self.transition = np.asarray(transition)
        self.signal = signal
        self.predictive = np.array([1 - prior_bad, prior_bad])
        self.prior_bad = prior_bad
        self.belief = None
        self.grid, self.scores = grid_plan(horizon, target, points, transition, signal)

    def act(self, view):
        likelihood = emission(view.wide, self.signal)
        weights = self.predictive * likelihood
        self.posterior = weights / weights.sum()
        self.belief = float(self.posterior[1])
        if not view.remaining:
            return Order("wait")
        left = view.horizon - view.step
        scores = self.scores[left, view.remaining, int(view.wide)]
        valid = [*range(view.remaining + 1), self.target + 1]
        choice = min(valid, key=lambda action: np.interp(self.belief, self.grid,
                                                        scores[:, action]))
        if choice == self.target + 1:
            return Order("post", 1)
        return Order("market", choice) if choice else Order("wait")

    def observe(self, order, filled):
        if not self.memory:
            self.predictive = np.array([1 - self.prior_bad, self.prior_bad])
            return
        posterior = self.posterior
        if order.kind == "post":
            likelihood = np.array(FILL) if filled else 1 - np.array(FILL)
            posterior = posterior * likelihood
            posterior /= posterior.sum()
        if order.kind == "market":
            weights = posterior * market_likelihood(order.lots, filled)
            # A stress fill may be impossible under the nominal depth model.
            if weights.sum() > 0:
                posterior = weights / weights.sum()
        self.predictive = posterior @ self.transition


@cache
def observed_plan(target, learn, concentration):
    """Exact DP for the proxy with revealed regime/quote and unknown fill rates."""
    prior = tuple(Beta(p * concentration, (1 - p) * concentration) for p in FILL)

    def actions(state):
        q, *_ = state
        return (*range(q + 1), -1) if q else (0,)

    def terminal(state):
        q, z, wide, _ = state
        return sum(market_edge(q, d, wide) for d in depths(z)) / 3 + CLEANUP * q

    def branches(state, action):
        q, z, wide, posterior = state
        if q == 0:
            yield 1.0, 0.0, state
            return
        if action == -1:
            mean = posterior[z].mean
            outcomes = ((0, 1 - mean, 1), (1, mean, 1))
        else:
            outcomes = ((0, 1, 1),) if action == 0 else (
                (min(action, depth), 1 / 3, depth) for depth in depths(z))
        for filled, probability, depth in outcomes:
            updated = posterior
            if action == -1 and learn:
                counts = list(posterior)
                counts[z] = counts[z].observe(filled)
                updated = tuple(counts)
            edge = (filled * (-HALF_SPREAD[wide] + MAKER) if action == -1 else
                    market_edge(filled, depth, wide))
            cost = stage(filled, q, target, edge)
            for next_z, transition in enumerate(TRANSITION[z]):
                for next_wide in range(2):
                    quote = SIGNAL[next_z] if next_wide else 1 - SIGNAL[next_z]
                    yield (probability * transition * quote, cost,
                           (q - filled, next_z, next_wide, updated))

    return prior, hyperstate_planner(actions, branches, terminal)


class ObservedPolicy:
    def __init__(self, target, learn, concentration):
        self.learn = learn
        self.posterior, self.plan = observed_plan(target, learn, concentration)

    def act(self, view):
        if view.regime is None:
            raise ValueError("This policy requires the observed-regime study")
        self.regime = view.regime
        state = (view.remaining, view.regime, int(view.wide), self.posterior)
        _, action = self.plan(view.horizon - view.step, state)
        if action == -1:
            return Order("post", 1)
        return Order("market", action) if action else Order("wait")

    def observe(self, order, filled):
        if order.kind == "post" and self.learn:
            counts = list(self.posterior)
            counts[self.regime] = counts[self.regime].observe(filled)
            self.posterior = tuple(counts)
