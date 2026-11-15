"""Small finite decision processes. Arrays use state/action/next-state order."""
from dataclasses import dataclass
from functools import cache

import numpy as np


def probabilities(values, axis=-1):
    """Validate a batch of categorical distributions without renormalizing."""
    array = np.asarray(values, dtype=float)
    if (not array.size or not np.isfinite(array).all() or (array < 0).any()
            or not np.allclose(array.sum(axis=axis), 1, rtol=0, atol=1e-10)):
        raise ValueError("Expected finite, nonnegative categorical probabilities")
    return array


def condition(prior, likelihood):
    """Return posterior and observation probability, rejecting impossible evidence."""
    prior = probabilities(prior)
    likelihood = np.asarray(likelihood, dtype=float)
    if (prior.ndim != 1 or likelihood.shape != prior.shape
            or not np.isfinite(likelihood).all()
            or (likelihood < 0).any() or (likelihood > 1).any()):
        raise ValueError("Expected a likelihood in [0, 1] for each state")
    weights = prior * likelihood
    evidence = weights.sum()
    if evidence <= 0:
        raise ValueError("Observation has zero likelihood")
    return weights / evidence, float(evidence)


def belief_update(belief, transition, likelihood):
    return condition(probabilities(belief) @ probabilities(transition), likelihood)[0]


def backward_induction(transition, costs, terminal, legal=None):
    """Solve a finite, undiscounted MDP with costs indexed [time,state,action]."""
    transition = probabilities(transition)
    costs = np.asarray(costs, dtype=float)
    terminal = np.asarray(terminal, dtype=float)
    if costs.ndim != 3:
        raise ValueError("Costs require time, state and action axes")
    horizon, states, actions = costs.shape
    if (transition.shape != (states, actions, states) or terminal.shape != (states,)
            or not np.isfinite(costs).all() or not np.isfinite(terminal).all()):
        raise ValueError("Inconsistent MDP shapes or nonfinite costs")
    legal = np.ones(costs.shape, dtype=bool) if legal is None else np.broadcast_to(
        np.asarray(legal, dtype=bool), costs.shape)
    if not legal.any(axis=-1).all():
        raise ValueError("Every state and time needs a legal action")
    values = np.empty((horizon + 1, states))
    policy = np.empty((horizon, states), dtype=int)
    values[-1] = terminal
    for step in reversed(range(horizon)):
        scores = np.where(legal[step], costs[step] + transition @ values[step + 1], np.inf)
        policy[step] = scores.argmin(axis=1)
        values[step] = scores.min(axis=1)
    return values, policy


def q_learning(transition, costs, terminal, episodes, seed, epsilon=0.2):
    """Educational cost-minimizing Q-learning with uniform initial-state sampling."""
    backward_induction(transition, costs, terminal)
    transition, costs, terminal = map(np.asarray, (transition, costs, terminal))
    horizon, states, actions = costs.shape
    if episodes < 1 or not 0 <= epsilon <= 1:
        raise ValueError("Expected positive episodes and epsilon in [0, 1]")
    rng = np.random.default_rng(seed)
    q = np.zeros_like(costs, dtype=float)
    visits = np.zeros(costs.shape, dtype=int)
    for _ in range(episodes):
        state = rng.integers(states)
        for step in range(horizon):
            action = rng.integers(actions) if rng.random() < epsilon else q[step, state].argmin()
            next_state = rng.choice(states, p=transition[state, action])
            future = terminal[next_state] if step + 1 == horizon else q[step + 1, next_state].min()
            index = (step, state, action)
            visits[index] += 1
            q[index] += visits[index] ** -0.65 * (costs[index] + future - q[index])
            state = next_state
    return q, q.argmin(axis=-1)


def pomdp_exact(transition, observation, costs, terminal, belief, horizon):
    """Exact observation-tree recursion for tiny stationary POMDPs.

    The observation tensor has axes [action, next_state, symbol].
    Returns expected cost and first action (None at horizon zero).
    """
    transition, observation = probabilities(transition), probabilities(observation)
    belief, costs, terminal = probabilities(belief), np.asarray(costs), np.asarray(terminal)
    if costs.ndim != 2:
        raise ValueError("Expected costs indexed by state and action")
    states, actions = costs.shape
    if (transition.shape != (states, actions, states) or observation.ndim != 3
            or observation.shape[:2] != (actions, states) or terminal.shape != (states,)
            or belief.shape != (states,) or horizon < 0
            or not np.isfinite(costs).all() or not np.isfinite(terminal).all()):
        raise ValueError("Inconsistent POMDP inputs")

    @cache
    def solve(remaining, weights):
        current = np.asarray(weights)
        if remaining == 0:
            return float(current @ terminal), None
        scores = current @ costs.astype(float)
        for action in range(actions):
            predicted = current @ transition[:, action, :]
            for likelihood in observation[action].T:
                joint = predicted * likelihood
                mass = joint.sum()
                if mass > 0:
                    scores[action] += mass * solve(remaining - 1, tuple(joint / mass))[0]
        choice = int(scores.argmin())
        return float(scores[choice]), choice

    return solve(horizon, tuple(belief))


@dataclass(frozen=True)
class Beta:
    alpha: float = 1
    beta: float = 1

    def __post_init__(self):
        if not np.isfinite([self.alpha, self.beta]).all() or min(self.alpha, self.beta) <= 0:
            raise ValueError("Beta parameters must be finite and positive")

    @property
    def mean(self):
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self):
        return self.mean * (1 - self.mean) / (self.alpha + self.beta + 1)

    def observe(self, success):
        if success not in (0, 1):
            raise ValueError("Expected a Bernoulli outcome")
        return Beta(self.alpha + int(success), self.beta + 1 - int(success))


@dataclass(frozen=True)
class Dirichlet:
    concentration: tuple[float, ...]

    def __post_init__(self):
        concentration = tuple(self.concentration)
        if not concentration or not np.isfinite(concentration).all() or min(concentration) <= 0:
            raise ValueError("Dirichlet concentrations must be finite and positive")
        object.__setattr__(self, "concentration", concentration)

    @property
    def mean(self):
        return np.asarray(self.concentration) / sum(self.concentration)

    def observe(self, symbol):
        if not 0 <= symbol < len(self.concentration):
            raise ValueError("Invalid categorical outcome")
        return Dirichlet(tuple(count + (index == symbol)
                               for index, count in enumerate(self.concentration)))


def hyperstate_planner(actions, branches, terminal):
    """Return a memoized exact finite-horizon planner for a finite reachable tree.

    Each branch is (probability, immediate cost, next hyperstate). Hyperstates
    must be hashable. The caller specifies posterior updates in branches.
    """
    @cache
    def solve(remaining, state):
        if remaining <= 0:
            return float(terminal(state)), None
        scores = []
        for action in actions(state):
            outcomes = tuple(branches(state, action))
            weights = probabilities([weight for weight, _, _ in outcomes])
            expected = sum(weight * (cost + solve(remaining - 1, successor)[0])
                           for weight, (_, cost, successor) in zip(weights, outcomes)
                           if weight > 0)
            scores.append((expected, action))
        return min(scores, key=lambda item: item[0])

    return solve
