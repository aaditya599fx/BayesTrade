"""Discrete-event synthetic buy execution. Policies receive public views only."""
from dataclasses import dataclass
from typing import Protocol

import numpy as np

TRANSITION = ((.87, .13), (.20, .80))
SIGNAL = (.18, .79)  # P(wide public signal | regime)
FILL = (.73, .30)
HALF_SPREAD = (1.3, 4.5)
DEPTH = (5, 2)
VOLUME_PROFILE = (1.40, 1.15, .90, .78, .85, 1.34)
MAKER = .10
TAKER = .30
TEMPORARY = 2.8
PERMANENT = .22
SELECTION = .85
MID = 100.0
LOT = 100


@dataclass(frozen=True)
class Scenario:
    name: str
    transition: tuple = TRANSITION
    signal: tuple = SIGNAL
    fill: tuple = FILL
    volatility: float = 1.0
    stress: bool = False
    unknown_fill: bool = False
    prior_bad: float = .40
    fill_concentration: float = 3.0


SCENARIOS = {
    s.name: s for s in (
        Scenario("stable", transition=((.98, .02), (.04, .96))),
        Scenario("switching"),
        Scenario("weak_signal", signal=(.43, .56)),
        Scenario("no_information", signal=(.5, .5)),
        Scenario("transition_shift", transition=((.40, .60), (.65, .35))),
        Scenario("fill_shift", fill=(.36, .12)),
        Scenario("rare_stress", volatility=1.65, stress=True),
        Scenario("unknown_fill", unknown_fill=True),
    )
}


@dataclass(frozen=True)
class Tape:
    regime: tuple
    wide: tuple
    volume: tuple
    depth: tuple
    flow: tuple
    arrival: tuple
    noise: tuple
    stress_start: int
    fill: tuple


def profile(horizon):
    return np.interp(np.linspace(0, 1, horizon), np.linspace(0, 1, len(VOLUME_PROFILE)),
                     VOLUME_PROFILE)


def make_tape(scenario, horizon, seed):
    rng = np.random.default_rng(seed)
    regime = [int(rng.random() < scenario.prior_bad)]
    for chance in rng.random(horizon):
        regime.append(int(chance < scenario.transition[regime[-1]][1]))
    stress_start = horizon + 1
    if scenario.stress and rng.random() < .18:
        stress_start = int(rng.integers(max(1, horizon // 2), horizon))
        regime[stress_start:] = [1] * (horizon + 1 - stress_start)
    state = np.asarray(regime)
    wide = rng.random(horizon + 1) < np.asarray(scenario.signal)[state]
    # Public volume and displayed quote depend on the public spread category.
    # This makes the modeled category sufficient for the pre-order emission.
    volume = 800 * profile(horizon) * np.where(wide[:-1], .75, 1.25)
    volume *= np.exp(.32 * rng.normal(size=horizon) - .32**2 / 2)
    depth = np.maximum(1, np.asarray(DEPTH)[state] + rng.integers(-1, 2, horizon + 1))
    fill = scenario.fill
    if scenario.unknown_fill:
        concentration = scenario.fill_concentration
        fill = tuple(rng.beta(p * concentration, (1 - p) * concentration) for p in fill)
    return Tape(tuple(regime), tuple(wide), tuple(volume), tuple(depth),
                tuple(rng.random(horizon)), tuple(rng.random(horizon)),
                tuple(rng.normal(size=horizon)), stress_start, fill)


@dataclass(frozen=True)
class View:
    step: int
    horizon: int
    remaining: int
    wide: bool
    volume: float
    regime: int | None = None


@dataclass(frozen=True)
class Order:
    kind: str
    lots: int = 0


@dataclass(frozen=True)
class Fill:
    step: int
    kind: str
    lots: int
    price: float
    fees: float


class Policy(Protocol):
    def act(self, view: View) -> Order: ...
    def observe(self, order: Order, filled: int) -> None: ...


def shortfall(fills, target, arrival=MID):
    dollars = sum(fill.lots * LOT * (fill.price - arrival) + fill.fees for fill in fills)
    return 1e4 * dollars / (target * LOT * arrival)


def execute(policy, tape, target, scenario, *, observed=False):
    """Use a fixed tape, one-interval queues and a mandatory sweep at deadline."""
    horizon = len(tape.flow)
    remaining = target
    mid_bps = 0.0
    ledger, trace, events = [], [], []
    exposure = 0
    completed_at = horizon + 1

    def book(step, kind, lots, edge, fee):
        if lots:
            ledger.append(Fill(step, kind, lots, MID * (1 + (mid_bps + edge) / 1e4),
                               LOT * lots * MID * fee / 1e4))

    for step in range(horizon):
        regime = tape.regime[step]
        stressed = step >= tape.stress_start
        spread = HALF_SPREAD[int(tape.wide[step])] * (2.1 if stressed else 1)
        depth = 1 if stressed else tape.depth[step]
        view = View(step, horizon, remaining, tape.wide[step], tape.volume[step],
                    regime if observed else None)
        order = policy.act(view)
        if (order.kind not in ("wait", "market", "post")
                or not isinstance(order.lots, (int, np.integer))
                or not 0 <= order.lots <= remaining
                or (order.kind == "wait" and order.lots != 0)
                or (order.kind == "post" and order.lots != 1)):
            raise ValueError(f"Invalid order for public inventory: {order}")
        belief = getattr(policy, "belief", None)
        if belief is not None:
            trace.append((step, regime, int(view.wide), float(belief)))
        filled = 0
        if order.kind == "market":
            filled = min(order.lots, depth)
            book(step, "market", filled, spread + TEMPORARY * filled / depth, TAKER)
        toxicity = 0.0
        queue_ahead = sell_flow = None
        if order.kind == "post":
            # A standardized one-lot child joins the visible bid for this interval.
            # Sale size is independent of queue ahead conditional on the regime.
            # Symmetric survival thresholds give a marginal fill probability p.
            p = tape.fill[regime] * (.55 if stressed else 1)
            queue_ahead = 1 + int(tape.flow[step] * 3)
            spread_mass = .45 * min(p, 1 - p)
            draw = tape.arrival[step]
            sell_flow = sum(draw < threshold for threshold in
                            (min(1, p + 2 * spread_mass), p + spread_mass,
                             p, p - spread_mass))
            filled = int(sell_flow >= queue_ahead + 1)
            book(step, "post", filled, -spread, MAKER)
            toxicity = -SELECTION * filled
        policy.observe(order, filled)
        events.append((step, remaining, order.kind, order.lots, filled, depth,
                       queue_ahead, sell_flow))
        remaining -= filled
        if remaining == 0 and completed_at == horizon + 1:
            completed_at = step + 1
        exposure += remaining**2
        sigma = (1.5, 4.0)[regime] * scenario.volatility * (1.6 if stressed else 1)
        mid_bps += sigma * tape.noise[step] + PERMANENT * filled + toxicity
    cleanup = remaining
    if cleanup:
        depth = 1 if horizon >= tape.stress_start else tape.depth[-1]
        spread = HALF_SPREAD[int(tape.wide[-1])] * (2.1 if horizon >= tape.stress_start else 1)
        book(horizon, "cleanup", cleanup,
             spread + TEMPORARY * cleanup / depth + 1.5, TAKER)
    return {
        "shortfall_bps": shortfall(ledger, target),
        "fees_dollars": sum(item.fees for item in ledger),
        "passive_lots": sum(item.lots for item in ledger if item.kind == "post"),
        "post_attempts": sum(event[2] == "post" for event in events),
        "cleanup_lots": cleanup,
        "completed_early": int(cleanup == 0),
        "completion_step": completed_at,
        "inventory_risk": exposure / (horizon * target**2),
        "ledger": ledger,
        "trace": trace,
        "events": events,
    }
