# BayesTrade
**Bayesian adaptive execution under latent market regimes**

**[Read the research paper (PDF)](reports/research-paper.pdf)** · [Study protocol](reports/protocol.md) · [Tutorial notebook](examples/bayestrade_tutorial.ipynb)

BayesTrade is a Python library for tabular MDPs, POMDPs, and BAMDPs, with a
seeded market execution simulator for testing Bayesian policies against
execution benchmarks. The study uses synthetic data and requires no API keys.

## Research findings

Each comparison uses 600 paired episodes in its scenario. Differences are
model minus comparator in basis points (bps) of implementation shortfall.
Negative values favor the model. The intervals are paired 95% bootstrap intervals.

| Scenario | Comparison | Mean difference | 95% interval |
|---|---|---:|---:|
| Nominal regime switching | POMDP vs. TWAP execution benchmark | −5.28 bps | [−5.52, −5.04] |
| Nominal regime switching | POMDP vs. the same planner using the current quote only | +0.04 bps | [−0.02, +0.09] |
| Unknown fill rates with observed regimes | BAMDP vs. the same planner with frozen fill priors | −0.09 bps | [−0.17, −0.01] |

The TWAP comparison captures the effect of changing the overall execution
policy. The same-planner comparison found little incremental cost benefit
from regime memory, and fill-rate learning produced a small gain. On the 94
tapes with a triggered liquidity shock, POMDP mean cost was 3.62 bps versus
5.32 bps for TWAP, while its 95th-percentile cost was 23.99 bps versus 14.36 bps.

[Paired results](reports/results/paired.csv) · [Stratified results](reports/results/strata.csv) · [Seeds and provenance](reports/results/manifest.json)

## Reproduce

Python 3.11+ is supported. The reference numerical run used Python 3.13.7
and NumPy 2.4.2. Run these commands from the repository root.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m unittest discover -s tests -v
bayestrade --quick --output /tmp/bayestrade-smoke
bayestrade --config configs/study.json --output reports/results
```

For PDF regeneration install the reporting dependencies, then run the report
builder. Use the same dependency versions for byte-level comparisons of the
CSV and compressed belief outputs. Wall-clock timings vary by machine.

```sh
python -m pip install -r requirements-report.txt
python reports/make_figures.py
python reports/build_report.py
```

On Windows activate with `.venv\Scripts\Activate.ps1` and choose a local
directory for the smoke output. The test suite uses Python's standard
`unittest` runner. GitHub Actions tests Python 3.11 and 3.13 and runs a smoke
benchmark on every push or pull request.

## Library

| Module | Interfaces |
|---|---|
| `tabular` | Probability validation, Bayes update, finite-horizon MDP backward induction with action masks, Q-learning, exact finite POMDP observation-tree search, conjugate Beta and Dirichlet models, exact reachable-hyperstate planning |
| `market` | Immutable seeded market tape, public observation and order interfaces, discrete queue and depth execution, fees, impact, terminal sweep, ledger and shortfall |
| `policies` | Immediate, TWAP, causal profile-VWAP, POV, spread-reactive, belief-grid POMDP, same-planner snapshot control, observed-regime exact BAMDP proxy and fixed-prior control |
| `study` | Disjoint tuning episodes, paired experiments, bootstrap comparisons, risk and calibration diagnostics and versioned outputs |

```python
import numpy as np
from bayestrade.tabular import belief_update

posterior = belief_update([0.6, 0.4], np.eye(2), [0.8, 0.3])
assert np.allclose(posterior, [0.8, 0.2])
```

Policies implement `act(view)` and `observe(order, filled)`. The tape is an
environment input. The hidden track exposes the noisy quote class and current
volume. Only the two observed-regime diagnostic policies receive the regime.
Midprice dynamics are used to book fills, and fill feedback to the supplied
policies consists of executed quantity. Immediate orders retry any depth-limited
remainder. The terminal sweep uses a priced depth curve and fills the remainder.

## Result files

- `episodes.csv` contains one policy outcome per paired tape and scenario.
  Join on `(scenario, seed, episode)`. Seed indices are reused across scenario
  definitions for comparability, so avoid resampling pooled scenarios as independent.
- `summary.csv` and `strata.csv` contain risk, cost and completion statistics.
  Cleanup is included in cost. `completed_early` means completed before the
  terminal sweep. `completion_step` is one-based and takes value horizon + 1
  when the terminal sweep is required.
- `paired.csv` gives policy-minus-control differences in basis points and
  pointwise 95% paired intervals. Negative differences favor the model.
- `calibration.csv`, `beliefs.csv.gz`, and `fill_posteriors.csv` provide
  regime-calibration and unknown-fill diagnostics. True latent values are
  evaluator fields.
- `trace*.csv` contains an illustrative belief trace, order events and its
  priced fill ledger. `timing.csv` measures total policy construction,
  planning and simulation by label. The first use of a cached planner pays
  the build cost.
- `manifest.json` contains the exact inputs, realized baseline settings,
  dependency versions and fingerprint of the source package. After a Git commit
  from a source checkout, the manifest also records its commit hash.

The library supports small finite problems. Exact POMDP tree search and exact
BAMDP hyperstate planning grow rapidly with horizon. The execution POMDP uses
a two-state belief grid to manage that cost. The study keeps hidden-state
filtering and observed-state parameter learning separate.

## Research roadmap

1. **Risk-sensitive execution.** Add a policy that optimizes a mean–CVaR
   objective for terminal implementation shortfall. Benchmark the cost–tail
   trade-off against the current POMDP and TWAP on paired tapes, including a
   held-out liquidity-shock stratum.
2. **Historical order-book calibration.** Estimate spreads, depth, volatility,
   and order-arrival patterns from timestamped order-book data. Split calibration
   and validation by session and quantify sensitivity to assumed passive fills
   and market impact, which cannot be identified for hypothetical orders from
   historical replay alone.
3. **Joint regime and parameter uncertainty.** Maintain a joint posterior over
   latent liquidity and uncertain fill rates. Verify approximate inference against
   exact enumeration on a small model before testing execution outcomes.

An IPOMDP remains a separate extension. Covering the full CaCTüS Project 8
scope would require confirming the project's intended IPOMDP formalism.

## License

Released under the [MIT License](LICENSE).
