# Study protocol
## Research questions and comparisons

1. Does a filtered belief with history improve mean cost over current-quote
   inference? Compare POMDP against Snapshot on nominal switching, with
   identical action space, costs, lookahead grid and observations. Snapshot
   resets to the fixed prior before each quote. This comparison changes policy
   decisions and, therefore, subsequent feedback.
2. Does exact posterior updating in the proxy improve realized cost with
   persistent unknown fill probability? Compare BAMDP against Frozen on the
   unknown-fill scenario. Both see the regime and use the same prior mean,
   quote, depth law, objective, actions and horizon. Frozen carries the
   original priors through every branch.
3. Measure external comparisons against immediate, TWAP, forecast VWAP,
   POV and spread-reactive policies. Those comparisons also change sizing,
   order types or scheduling. Use them for overall outcomes.

The primary outcome is arrival-price implementation shortfall, normalized to
bps of the parent order. The primary contrasts are POMDP minus Snapshot in
nominal switching and BAMDP minus Frozen in unknown fill. A 0.25 bps
difference is a descriptive economic yardstick, selected for the scale of the
toy costs without external execution calibration. Secondary outcomes include
standard deviation, p90, p95, worst-5% average (ES95), fees, pre-sweep
completion, cleanup volume and mean squared inventory exposure. A paired
bootstrap samples whole episodes within a scenario and reports pointwise
percentile intervals. Inference describes the specified generator. The
study plan is documented with the experiment, with no external preregistration.
Intervals across stresses and baselines have no multiplicity adjustment.

## Inputs and shifts

| Scenario | Difference from nominal switching |
|---|---|
| Stable | Both regimes persist longer |
| Switching | Nominal T, emissions and per-regime fills |
| Weak signal | True quote likelihoods .43 and .56. Planner retains nominal emissions |
| No information | Equal quote likelihoods, announced to planner. Executed quantities can still be informative |
| Transition shift | True good-to-bad .60 and bad-to-good .65. Planner retains nominal T |
| Fill shift | True fill probabilities reduced to .36 and .12. Planner retains nominal rates |
| Rare stress | Higher volatility, and an 18% episode probability of a late illiquid stress state |
| Unknown fill | Episode-static regime-specific Beta draws, with observed-regime comparison |

See `configs/study.json` for episode counts, seeds and bootstrap budget.
Each evaluation policy sees the same fixed exogenous tape within each episode;
endogenous prices, inventory, actions and fills may diverge. Policies are
deterministic, so the three independent seeds index tape generation. POV and
reactive candidates minimize mean cost on disjoint nominal tuning tapes.
Planner risk and action-grid settings are fixed. VWAP uses a fixed public
volume profile and its cumulative lot target. During a development run, its
original rounding rule generated the same orders as TWAP at the default lot
size. The final benchmark uses cumulative rounding and was rerun in full.

Strata split by whether at least half of decision intervals were illiquid
and by the stress flag. Regimes and true fill parameters remain on the
evaluation side of the interface. Regime calibration uses pre-order
predictions at each interval, including after early completion, and reports
descriptive Brier and log loss. Bin frequencies are pooled for the reliability
chart. Post-run fill posterior error is evaluated across all episodes.
