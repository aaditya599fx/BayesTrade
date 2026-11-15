# BayesTrade
**Bayesian adaptive execution under latent market regimes**

BayesTrade contains a reusable tabular decision library, a seeded event simulator
for buying a parent order, and two controlled tests of whether Bayesian
information changes execution outcomes. The experience is synthetic and uses
no API keys or market data.

The [research paper](reports/research-paper.pdf) gives the model and findings.
The [study protocol](reports/protocol.md) defines comparisons and metrics.
The [result manifest](reports/results/manifest.json) records source and seed
provenance, and the [notebook](examples/bayestrade_tutorial.ipynb) walks through
the library and an auditable execution.

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

To publish, create an empty GitHub repository under your account, then commit
this directory and push it to the repository URL. The supplied project also
runs locally without a GitHub account.
