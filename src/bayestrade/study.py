"""Deterministic experiment driver; wall-clock timings are recorded separately."""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import numpy as np

from . import __version__
from .market import SCENARIOS, SIGNAL, execute, make_tape
from .policies import Baseline, BeliefPolicy, ObservedPolicy, grid_plan, observed_plan

BASELINES = ("immediate", "twap", "vwap", "pov", "reactive")
HIDDEN = (*BASELINES, "snapshot", "pomdp")
KNOWN = (*BASELINES, "frozen", "bamdp")


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def paired_interval(deltas, draws, seed):
    """Percentile bootstrap of paired episode means."""
    delta = np.asarray(deltas)
    rng = np.random.default_rng(seed)
    samples = np.concatenate([delta[rng.integers(len(delta), size=(min(256, draws - start),
                                                                   len(delta)))].mean(axis=1)
                              for start in range(0, draws, 256)])
    return float(delta.mean()), *np.quantile(samples, [.025, .975])


def tune(cfg):
    scenario = SCENARIOS["switching"]
    horizon, target = cfg["horizon"], cfg["target_lots"]
    tapes = [make_tape(scenario, horizon, np.random.SeedSequence([cfg["train_seed"], i]))
             for i in range(cfg["train_episodes"])]
    scores = {}
    for name in ("pov", "reactive"):
        scores[name] = [
            [parameter, float(np.mean([
                execute(Baseline(name, target, parameter), tape, target, scenario)["shortfall_bps"]
                for tape in tapes]))]
            for parameter in cfg[f"{name}_candidates"]
        ]
    return {name: min(rows, key=lambda item: item[1])[0] for name, rows in scores.items()}, scores


def policy_for(name, cfg, scenario, selected):
    horizon, target = cfg["horizon"], cfg["target_lots"]
    if name in BASELINES:
        return Baseline(name, target, selected.get(name, 1))
    if name in ("snapshot", "pomdp"):
        signal = (.5, .5) if scenario.name == "no_information" else SIGNAL
        return BeliefPolicy(horizon, target, cfg["grid_points"],
                            memory=name == "pomdp", signal=signal)
    return ObservedPolicy(target, name == "bamdp", scenario.fill_concentration)


def aggregate(rows):
    costs = np.array([row["shortfall_bps"] for row in rows])
    tail = max(1, int(np.ceil(.05 * len(costs))))
    answer = dict(mean=costs.mean(), median=np.median(costs), sd=costs.std(ddof=1),
                  p90=np.quantile(costs, .90), p95=np.quantile(costs, .95),
                  es95=np.sort(costs)[-tail:].mean())
    for key in ("fees_dollars", "passive_lots", "post_attempts", "cleanup_lots",
                "completed_early", "completion_step", "inventory_risk"):
        answer[key] = np.mean([row[key] for row in rows])
    return answer


def validate_config(cfg):
    if (min(cfg[key] for key in ("horizon", "target_lots", "train_episodes",
                                   "episodes_per_seed", "bootstrap_draws")) < 1
            or cfg["grid_points"] < 2 or not cfg["test_seeds"]
            or cfg["episodes_per_seed"] * len(cfg["test_seeds"]) < 2
            or len(set(cfg["test_seeds"])) != len(cfg["test_seeds"])
            or cfg["train_seed"] in cfg["test_seeds"]
            or len(set(cfg["scenarios"])) != len(cfg["scenarios"])
            or not set(cfg["scenarios"]) <= SCENARIOS.keys()
            or not cfg["scenarios"]
            or ("rare_stress" in cfg["scenarios"] and cfg["horizon"] < 2)):
        raise ValueError("Invalid study dimensions, scenarios or overlapping seeds")


def run(cfg, output):
    validate_config(cfg)
    output.mkdir(parents=True, exist_ok=True)
    grid_plan.cache_clear()
    observed_plan.cache_clear()
    selected, tuning = tune(cfg)
    rows, summaries, paired, strata, beliefs, timings = [], [], [], [], [], []
    example, example_ledger, example_fills, fill_diagnostics = [], [], [], []
    for name in cfg["scenarios"]:
        scenario = SCENARIOS[name]
        policy_names = KNOWN if scenario.unknown_fill else HIDDEN
        batches = {policy: [] for policy in policy_names}
        elapsed = dict.fromkeys(policy_names, 0.0)
        for seed in cfg["test_seeds"]:
            for episode in range(cfg["episodes_per_seed"]):
                tape = make_tape(scenario, cfg["horizon"], np.random.SeedSequence([seed, episode]))
                fraction_bad = np.mean(tape.regime[:-1])
                illiquid = fraction_bad >= .5
                for label in policy_names:
                    start = perf_counter()
                    policy = policy_for(label, cfg, scenario, selected)
                    outcome = execute(policy, tape, cfg["target_lots"], scenario,
                                      observed=label in ("bamdp", "frozen"))
                    elapsed[label] += perf_counter() - start
                    ledger, trace, events = (outcome.pop(key) for key in ("ledger", "trace", "events"))
                    if label in ("bamdp", "frozen"):
                        for z, posterior in enumerate(policy.posterior):
                            fill_diagnostics.append(dict(seed=seed, episode=episode, policy=label,
                                                         regime=z, true_fill=tape.fill[z],
                                                         mean=posterior.mean,
                                                         variance=posterior.variance,
                                                         observations=round(posterior.alpha
                                                                            + posterior.beta
                                                                            - scenario.fill_concentration)))
                    row = dict(scenario=name, seed=seed, episode=episode, policy=label,
                               illiquid=int(illiquid), stressed=int(tape.stress_start < cfg["horizon"]),
                               regime_bad_fraction=fraction_bad,
                               volatility_scale=scenario.volatility, **outcome)
                    rows.append(row)
                    batches[label].append(row)
                    for step, true, wide, belief in trace:
                        entry = dict(scenario=name, seed=seed, episode=episode, policy=label,
                                     step=step, regime=true, wide=wide, belief=belief)
                        beliefs.append(entry)
                    if (name == "switching" and label == "pomdp" and not example
                            and 0 < fraction_bad < 1):
                        example.extend(beliefs[-len(trace):])
                        example_fills.extend(asdict(fill) for fill in ledger)
                        example_ledger.extend(dict(step=step, remaining=q, action=kind,
                                                   requested=requested, filled=filled, depth=depth,
                                                   queue_ahead=queue, sell_flow=flow)
                                              for step, q, kind, requested, filled, depth, queue, flow
                                              in events)
                        example_ledger.extend(dict(step=fill.step, remaining="", action=fill.kind,
                                                   requested="", filled=fill.lots, depth="",
                                                   queue_ahead="", sell_flow="")
                                              for fill in ledger if fill.kind == "cleanup")
        for label, batch in batches.items():
            summaries.append(dict(scenario=name, policy=label, n=len(batch), **aggregate(batch)))
            timings.append(dict(scenario=name, policy=label, runtime_seconds=elapsed[label],
                                milliseconds_per_episode=1000 * elapsed[label] / len(batch)))
            for column in ("illiquid", "stressed"):
                for flag in (0, 1):
                    subset = [row for row in batch if row[column] == flag]
                    if len(subset) >= 2:
                        strata.append(dict(scenario=name, policy=label, stratum=column,
                                           flag=flag, n=len(subset), **aggregate(subset)))
        comparisons = (("bamdp", "frozen"), ("bamdp", "twap")) if scenario.unknown_fill else (
            ("pomdp", "snapshot"), ("pomdp", "reactive"), ("pomdp", "twap"),
            ("pomdp", "vwap"), ("pomdp", "pov"))
        for model, control in comparisons:
            delta = [a["shortfall_bps"] - b["shortfall_bps"]
                     for a, b in zip(batches[model], batches[control], strict=True)]
            mean, low, high = paired_interval(delta, cfg["bootstrap_draws"], cfg["bootstrap_seed"])
            paired.append(dict(scenario=name, model=model, control=control, n=len(delta),
                               mean_delta_bps=mean, ci_low=low, ci_high=high,
                               share_cheaper=np.mean(np.asarray(delta) < 0)))
        print(f"{name}: {len(next(iter(batches.values())))} paired episodes", flush=True)

    calibration = []
    for name in cfg["scenarios"]:
        for label in ("snapshot", "pomdp"):
            relevant = [row for row in beliefs if row["scenario"] == name and row["policy"] == label]
            if relevant:
                pred = np.clip([r["belief"] for r in relevant], 1e-12, 1 - 1e-12)
                truth = np.array([r["regime"] for r in relevant])
                calibration.append(dict(scenario=name, policy=label, n=len(truth),
                                        brier=np.mean((pred - truth)**2),
                                        log_loss=-np.mean(truth * np.log(pred) +
                                                          (1 - truth) * np.log1p(-pred))))
    for filename, table in (("episodes", rows), ("summary", summaries), ("paired", paired),
                            ("strata", strata), ("calibration", calibration),
                            ("trace", example), ("trace_orders", example_ledger),
                            ("trace_fills", example_fills), ("fill_posteriors", fill_diagnostics),
                            ("timing", timings)):
        write_csv(output / f"{filename}.csv", table)
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
                             capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git = None
    source = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        source.update(path.name.encode())
        source.update(path.read_bytes())
    metadata = dict(config=cfg, scenario_parameters={name: asdict(SCENARIOS[name])
                    for name in cfg["scenarios"]}, selected=selected, tuning=tuning,
                    package=__version__, python=platform.python_version(), numpy=np.__version__,
                    git_commit=git, code_sha256=source.hexdigest())
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    # Retain episode/step granularity for reproducible calibration diagnostics.
    if beliefs:
        import gzip
        import io
        with (output / "beliefs.csv.gz").open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with io.TextIOWrapper(compressed, newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(beliefs[0]))
                    writer.writeheader()
                    writer.writerows(beliefs)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--config", type=Path, default=Path("configs/study.json"))
    cli.add_argument("--output", type=Path, default=Path("reports/results"))
    cli.add_argument("--quick", action="store_true", help="Smoke study with reduced sample")
    args = cli.parse_args()
    config = json.loads(args.config.read_text())
    if args.quick:
        config.update(train_episodes=10, test_seeds=config["test_seeds"][:1],
                      episodes_per_seed=8, bootstrap_draws=200)
    run(config, args.output)


if __name__ == "__main__":
    main()
