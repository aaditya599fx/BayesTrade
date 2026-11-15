"""Build the research paper from the saved benchmark and prepared figures."""
import argparse
import csv
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Table, TableStyle

ROOT = Path(__file__).resolve().parent
INK = colors.HexColor("#182b40")
SECONDARY = colors.HexColor("#52667b")
ACCENT = colors.HexColor("#12697c")
PALE = colors.HexColor("#edf5f6")
LINE = colors.HexColor("#d7e3e7")
WIDTH, HEIGHT = A4
CONTENT = WIDTH - 102


def style(name, font, size, leading, color=INK, after=8, before=0):
    return ParagraphStyle(name, fontName=font, fontSize=size, leading=leading,
                          textColor=color, spaceAfter=after, spaceBefore=before,
                          alignment=TA_LEFT, allowWidows=0, allowOrphans=0)


STYLES = {
    "label": style("label", "Helvetica-Bold", 9, 12, ACCENT, 12),
    "title": style("title", "Times-Bold", 31, 33, INK, 14),
    "subtitle": style("subtitle", "Helvetica", 12, 17, SECONDARY, 24),
    "heading": style("heading", "Times-Bold", 27, 30, INK, 17),
    "sub": style("sub", "Helvetica-Bold", 11, 15, ACCENT, 5, 9),
    "text": style("text", "Times-Roman", 11.5, 15, INK, 10),
    "eq": style("eq", "Courier", 9, 13, ACCENT, 10),
    "caption": style("caption", "Helvetica", 8.7, 12, SECONDARY, 11, 5),
    "table_head": style("table_head", "Helvetica-Bold", 8.3, 11, colors.white, 0),
    "table": style("table", "Helvetica", 9, 12, INK, 0),
}


def records(name):
    with (ROOT / "results" / name).open() as stream:
        return list(csv.DictReader(stream))


def fmt(number, places=2):
    return f"{float(number):.{places}f}"


def contrast(row):
    return f"{float(row['mean_delta_bps']):+.2f} [{float(row['ci_low']):+.2f}, {float(row['ci_high']):+.2f}]"


def furniture(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(LINE)
    canvas.line(51, HEIGHT - 42, WIDTH - 51, HEIGHT - 42)
    canvas.line(51, 43, WIDTH - 51, 43)
    canvas.setFillColor(SECONDARY)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.drawString(51, HEIGHT - 31, "BAYESTRADE  /  RESEARCH PAPER")
    canvas.drawRightString(WIDTH - 51, HEIGHT - 31, "SYNTHETIC EXECUTION STUDY")
    canvas.setFont("Helvetica", 8)
    canvas.drawString(51, 29, "Research release 1.0  |  November 2026")
    canvas.drawRightString(WIDTH - 51, 29, f"{doc.page:02d}")
    canvas.restoreState()


def build(output):
    document = SimpleDocTemplate(str(output), pagesize=A4, topMargin=59, bottomMargin=56,
                                 leftMargin=45, rightMargin=45,
                                 title="BayesTrade: Bayesian Adaptive Execution",
                                 subject="Paired synthetic experiments on regime filtering and fill learning",
                                 author="BayesTrade research project")
    contents = []
    pairs = {(r["scenario"], r["control"]): r for r in records("paired.csv")}
    summaries = {(r["scenario"], r["policy"]): r for r in records("summary.csv")}
    calibration = {(r["scenario"], r["policy"]): r for r in records("calibration.csv")}
    stress = {(r["flag"], r["policy"]): r for r in records("strata.csv")
              if r["scenario"] == "rare_stress" and r["stratum"] == "stressed"}
    metadata = json.loads((ROOT / "results" / "manifest.json").read_text())
    cfg = metadata["config"]
    episodes = cfg["episodes_per_seed"] * len(cfg["test_seeds"])
    observations = records("fill_posteriors.csv")
    timing = {(r["scenario"], r["policy"]): r for r in records("timing.csv")}

    def p(value, kind="text"):
        contents.append(Paragraph(value, STYLES[kind]))

    def chapter(label, title):
        contents.append(PageBreak())
        p(label, "label")
        p(title, "heading")

    def table(headers, rows, widths):
        matrix = [[Paragraph(text, STYLES["table_head"]) for text in headers]]
        matrix.extend([Paragraph(str(text), STYLES["table"]) for text in row] for row in rows)
        item = Table(matrix, colWidths=widths, hAlign="LEFT", repeatRows=1)
        item.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LINEBELOW", (0, -1), (-1, -1), .6, LINE),
        ]))
        contents.append(item)

    def figure(name, caption):
        path = str(ROOT / "figures" / name)
        width, height = ImageReader(path).getSize()
        img = Image(path, width=CONTENT, height=CONTENT * height / width)
        img.hAlign = "LEFT"
        contents.append(img)
        p(caption, "caption")

    p("BAYESIAN DECISION-MAKING  /  MARKET MICROSTRUCTURE", "label")
    p("Bayesian adaptive execution<br/>under latent market regimes", "title")
    p("Paired simulation evidence on the value of regime memory and fill-rate learning", "subtitle")
    p("Abstract", "sub")
    p(f"We study a deadline-constrained purchase in a seeded market with latent liquidity, "
      f"queue-dependent passive fills, depth-limited market orders and impact. We compare a "
      f"belief-grid POMDP against current-quote inference under the same planner, and exact "
      f"Beta-Bernoulli planning against frozen priors in a separate observed-regime setting. "
      f"Across eight scenarios with {episodes:,} paired episodes each, the nominal POMDP "
      f"lowered shortfall versus TWAP by {fmt(-float(pairs['switching', 'twap']['mean_delta_bps']))} bps. "
      f"The paired POMDP-minus-Snapshot mean was {contrast(pairs['switching', 'snapshot'])} bps. "
      f"The BAMDP-minus-Frozen mean was {contrast(pairs['unknown_fill', 'frozen'])} bps. "
      f"Brackets give pointwise 95% intervals. Cost differences from the controlled comparisons "
      f"were small relative to those from changing the execution policy more broadly.")
    p("Research questions", "sub")
    p("Does remembering quote and execution feedback improve implementation shortfall over a "
      "fresh inference from the current quote? Does planning over a learned fill-rate posterior "
      "improve outcomes when the regime is observable and each regime's fill rate persists for "
      "an episode? We use equal-information controls to study each question.")
    p("Relation to prior work", "sub")
    p("Almgren and Chriss [1] formalize the trade-off between impact and exposure in optimal "
      "execution. We use a small discrete action model with market and passive decisions. "
      "Kaelbling, Littman and Cassandra [2] give the belief-MDP framework for partial "
      "observability. Ross, Chaib-draa and Pineau [3] model uncertainty over dynamics through "
      "Bayes-adaptive state. Our learning experiment uses exact conjugacy in a fully observed "
      "regime model. It studies parameter learning separately from latent-state filtering.")

    chapter("01  /  MODELS", "Decision problem")
    p("At decision step t, the order has q lots remaining. The hidden regime z is liquid (0) "
      "or illiquid (1). An exogenous Markov chain has nominal rows (0.87, 0.13) and "
      "(0.20, 0.80), with initial illiquid probability 0.40. A public wide-quote category "
      "has probability 0.18 or 0.79, conditional on regime. The agent sees current public "
      "volume and inventory, sends an order, then observes the number of lots filled. "
      "Wait, request 1 to q market lots, or post one passive lot are available.")
    p("Finite-horizon MDP", "sub")
    p("For a fully observed finite state s and stage cost c, terminal value J is fixed at the "
      "deadline. Backward induction enumerates legal actions and takes their minimum. "
      "The package also includes a time-indexed tabular Q-learning implementation.")
    p("V[t,s] = min_a { c[t,s,a] + sum_s' T[s,a,s'] V[t+1,s'] }", "eq")
    p("Hidden regime and belief-space backup", "sub")
    p("Let b denote the pre-order posterior and L<sub>a</sub>(f | z) the likelihood of the "
      "executed quantity f. Multiply b by this likelihood, propagate the joint weights "
      "through T, then multiply by the emission for the next quote class o. Summing gives "
      "the joint probability of f and o. Normalizing gives the next pre-order belief.")
    p("w(z') = O(o | z') sum_z b(z) L_a(f | z) T(z' | z)<br/>"
      "P(f,o | b,a) = sum_z' w(z');   b'(z') = w(z') / P(f,o | b,a)", "eq")
    p("The Bellman backup adds expected local cost to the continuation value averaged over "
      f"feedback and next quote. Inventory becomes q - f. On the {cfg['grid_points']}-point binary belief grid, continuation "
      "values are linearly interpolated; the runtime choice minimizes interpolated action "
      "scores. Zero-probability branches contribute zero. If realized depth feedback is "
      "impossible under the nominal model, the execution filter skips that feedback. "
      "Generic exact observation-tree search is provided for tiny POMDPs.")
    p("Bayes-adaptive hyperstate", "sub")
    p("In the observed-regime track, each unknown p<sub>z</sub> has independent Beta "
      "parameters (alpha<sub>z</sub>, beta<sub>z</sub>). A standardized passive child yields "
      "Bernoulli result f. Add f to alpha and 1 - f to beta for that observed regime. "
      "Predictive success is alpha / (alpha + beta). Backward induction enumerates "
      "fill, regime and quote branches while carrying posterior counts in the state. "
      "Memoization gives an exact optimum in the reachable finite planning proxy.")

    chapter("02  /  SIMULATOR", "Execution and accounting")
    p("Each episode buys six lots of 100 shares from an arrival mid of $100 over eight "
      "decision intervals. The same exogenous tape supplies quote categories, volume, "
      "regimes, depth, queues, sell-flow uniforms and price innovations to each policy. "
      "The evaluator computes policy-dependent inventory and impacted midpoint.")
    table(["Component", "Event rule"],
          [
              ("Quote and volume", "Half-spread is 1.3 or 4.5 bps by public class. Volume is 800 shares times a fixed intraday profile, a quote multiplier (.75 wide, 1.25 tight) and mean-one lognormal noise."),
              ("Aggressive order", "Depth is uniform on {4, 5, 6} or {1, 2, 3} lots by regime. Executed size f is capped at depth d. Ask price adds 2.8 f/d bps of temporary impact. Taker fee is 0.30 bps."),
              ("Passive order", "A one-lot bid joins a queue with 1, 2 or 3 lots ahead uniformly. Same-interval sell quantity is independent of queue conditional on regime. Its survival at 2, 3 and 4 lots is p+d, p and p-d with d=.45 min(p,1-p)."),
              ("Fill and expiry", "A passive fill occurs only if sell flow clears the queue and child. Nominal p is .73 in liquid and .30 in illiquid, exactly the marginal fill rate. Unfilled bids expire at interval end. Maker fee is 0.10 bps."),
              ("Mid and deadline", "Mid change in bps is zero-mean Gaussian innovation with regime SD 1.5 or 4.0, plus .22 per executed lot and minus .85 per passive fill. A final market sweep has the same impact curve and 1.5 bps surcharge."),
          ], [104, CONTENT - 104])
    p("The rare-stress shock forces illiquidity, depth one, 2.1x spread, .55x passive "
      "fill probability and 1.6x per-step volatility on top of the scenario-wide 1.65x "
      "scale. The terminal sweep books one fill for all outstanding lots at the sweep "
      "price. It uses the depth curve beyond displayed capacity.")
    p("Implementation shortfall", "sub")
    p("IS[bps] = 10,000 [sum_j x_j(P_j - P_0) + fees] / (Q P_0)", "eq")
    p("The execution ledger records shares, prices and fees at every fill, including the "
      "sweep. Pre-sweep completion flags reliance on that sweep. Mean squared remaining "
      "inventory, scaled by squared target, measures exposure. The planner objective "
      "adds a .23 risk coefficient times remaining lots squared divided by target lots, "
      "and a term for permanent impact on future executions. The planning proxy "
      "omits price innovations and the simulator's passive selection drift.")

    chapter("03  /  EXPERIMENTS", "Controls and paired design")
    p("The matched switching scenario tests filtering against Snapshot, which resets "
      "the prior before every quote and uses identical planning scores. The unknown-fill "
      "scenario draws episode-static probabilities from independent Beta distributions "
      "concentrated at three pseudo-observations with nominal means. Frozen shares the "
      "same regime, prior, objective and lookahead, holding its parameters fixed.")
    table(["Policy", "Decision rule"],
          [
              ("Immediate", "Request all remaining lots at each eligible interval."),
              ("TWAP", "Round remaining lots divided by remaining intervals up."),
              ("Causal VWAP", "Trade up to the rounded cumulative target from a fixed public volume profile."),
              ("POV", "Target a tuned fraction of current observed volume; apply a late catch-up floor."),
              ("Reactive", "Post one lot on wide quotes if behind-schedule risk is sufficiently low; otherwise use TWAP."),
              ("POMDP / Snapshot", "Use the same policy grid with different treatment of historical feedback."),
              ("BAMDP / Frozen", "Use revealed regime and identical planning costs, with posterior updating or fixed priors."),
          ], [116, CONTENT - 116])
    p(f"POV and reactive parameters were selected by mean cost on {cfg['train_episodes']} "
      f"disjoint switching tapes. The chosen values were {metadata['selected']['pov']:.2f} "
      f"and {metadata['selected']['reactive']:.2f}. The final evaluation used "
      f"{len(cfg['test_seeds'])} tape seeds and {episodes} paired episodes per scenario. "
      f"The VWAP rounding rule changed after a development run exposed duplicate TWAP "
      f"actions; the full final study was rerun using cumulative rounding.")
    p("Analysis", "sub")
    p(f"The primary metric is mean paired shortfall difference in bps, with "
      f"{cfg['bootstrap_draws']:,} whole-episode bootstrap draws and pointwise 95% "
      f"percentile intervals. Negative differences favor the model. The illustrative "
      f"0.25 bps economic yardstick reflects the scale of the simulation. Tails, "
      f"fees and completion are secondary. Reported intervals are unadjusted for "
      f"multiple comparisons. The plan is documented with the experiment and has "
      f"no external preregistration.")
    p("In the no-information test the planner knows that quote likelihoods are identical. "
      "Quote observations then leave relative beliefs unchanged. Fill quantities can still "
      "reveal liquidity. The weak-signal, transition and fill-shift tests leave the "
      "planner's nominal model in place.")

    chapter("04  /  RESULTS", "Little incremental value from history")
    hidden = [("Stable", "stable"), ("Nominal switching", "switching"), ("Weak signal", "weak_signal"),
              ("No information", "no_information"), ("Transition shift", "transition_shift"),
              ("Fill shift", "fill_shift"), ("Rare stress", "rare_stress")]
    table(["Scenario", "POMDP - Snapshot", "POMDP - TWAP"],
          [(label, contrast(pairs[key, "snapshot"]), contrast(pairs[key, "twap"]))
           for label, key in hidden], [148, 171, CONTENT - 319])
    p("Table 1. Paired mean shortfall differences in bps with 95% bootstrap intervals.", "caption")
    figure("paired.png", "Figure 1. Same-planner contrasts. The shaded area spans +/-0.25 bps. "
           "The final row compares BAMDP with Frozen in the observed-regime track.")
    p(f"In nominal switching, POMDP minus causal VWAP was {contrast(pairs['switching', 'vwap'])} bps "
      f"and POMDP minus Reactive was {contrast(pairs['switching', 'reactive'])} bps. "
      "The same-planner contrast is close to zero, so these larger differences provide "
      "little evidence for a cost benefit from memory alone. Under weak signal and "
      "transition misspecification, its pointwise intervals favor Snapshot, with "
      "effects still inside the descriptive economic band.")
    p("In the observed-regime study, exact planning with learning delivered "
      f"{contrast(pairs['unknown_fill', 'frozen'])} bps versus Frozen. Both policies "
      "used the same information except for how they incorporated feedback. The "
      "estimated gain is below the illustrative economic yardstick.")

    chapter("05  /  RISK AND DIAGNOSTICS", "Cost tails and belief quality")
    rows = []
    for scenario, policy in (("switching", "twap"), ("switching", "vwap"),
                              ("switching", "snapshot"), ("switching", "pomdp"),
                              ("rare_stress", "twap"), ("rare_stress", "pomdp"),
                              ("unknown_fill", "frozen"), ("unknown_fill", "bamdp")):
        row = summaries[scenario, policy]
        rows.append((scenario.replace("_", " ").capitalize(), policy.upper(),
                     *(fmt(row[key]) for key in ("mean", "p95", "es95")),
                     f"{100 * float(row['completed_early']):.1f}%"))
    table(["Scenario", "Policy", "Mean", "P95", "ES95", "Pre-sweep"],
          rows, [119, 83, 63, 63, 63, CONTENT - 391])
    p("Table 2. Mean, P95 and ES95 in bps. Pre-sweep is the share completing before cleanup.", "caption")
    figure("shortfall.png", "Figure 2. Shortfall distributions. In rare stress, POMDP shifts much of the "
           "distribution left while worsening the upper tail relative to TWAP.")
    p(f"Among {stress['1', 'pomdp']['n']} tapes with a triggered stress shock, POMDP mean "
      f"shortfall was {fmt(stress['1', 'pomdp']['mean'])} bps versus "
      f"{fmt(stress['1', 'twap']['mean'])} for TWAP. P95 rose to "
      f"{fmt(stress['1', 'pomdp']['p95'])} bps versus {fmt(stress['1', 'twap']['p95'])}. "
      f"In nominal switching, POMDP cut mean fees from "
      f"${fmt(summaries['switching', 'twap']['fees_dollars'])} to "
      f"${fmt(summaries['switching', 'pomdp']['fees_dollars'])} and increased normalized "
      f"inventory exposure from {fmt(summaries['switching', 'twap']['inventory_risk'])} "
      f"to {fmt(summaries['switching', 'pomdp']['inventory_risk'])}.")
    p("Regime calibration", "sub")
    p(f"Nominal Brier scores were {fmt(calibration['switching', 'pomdp']['brier'], 3)} for "
      f"POMDP and {fmt(calibration['switching', 'snapshot']['brier'], 3)} for Snapshot. "
      f"Under transition shift they were {fmt(calibration['transition_shift', 'pomdp']['brier'], 3)} "
      f"and {fmt(calibration['transition_shift', 'snapshot']['brier'], 3)}. "
      "Calibration and control performance can move differently because only actions "
      "near a decision boundary change as beliefs move.")

    chapter("06  /  REPRODUCIBILITY", "Model checks and limitations")
    figure("calibration.png", "Figure 3. Left, illustrative tape with regime changes; low and high red marks "
           "denote tight and wide quotes. Right, equal-width reliability bins with at least 15 "
           "observations. Marker area reflects relative counts within each policy.")
    by_policy = {}
    for policy in ("frozen", "bamdp"):
        by_policy[policy] = []
        for regime in ("0", "1"):
            subset = [r for r in observations if r["policy"] == policy and r["regime"] == regime]
            by_policy[policy].append(sum(abs(float(r["mean"]) - float(r["true_fill"]))
                                         for r in subset) / len(subset))
    p(f"Learning reduced absolute fill-probability error from {by_policy['frozen'][0]:.3f} "
      f"to {by_policy['bamdp'][0]:.3f} in liquid regimes and from {by_policy['frozen'][1]:.3f} "
      f"to {by_policy['bamdp'][1]:.3f} in illiquid regimes. Measured mean end-to-end time "
      f"was {fmt(timing['unknown_fill', 'bamdp']['milliseconds_per_episode'])} ms per "
      f"BAMDP episode versus {fmt(timing['unknown_fill', 'frozen']['milliseconds_per_episode'])} "
      "ms for Frozen. Timing includes initial cache construction and is hardware-dependent.")
    p("Scope", "sub")
    p("The simulator uses one asset and direction, short-lived queues, stylized adverse "
      "selection and an unconstrained terminal sweep price. Depth feedback is coarsened "
      "to executed quantity and quote classes. The policy baselines use different child "
      "types and schedules; same-planner comparisons more directly test inference. "
      "Historical order-book validation and joint hidden-regime parameter learning "
      "are outside this study. The observed-regime track supplies a diagnostic upper "
      "level of information and its economics are specific to this synthetic generator.")
    p("Reproduction", "sub")
    p("Run <font name='Courier'>bayestrade --config configs/study.json</font> after an "
      "editable install. The repository includes episode-level results, manifests, "
      "source and tests of mathematical, accounting and information-boundary invariants. "
      "Generate graphics with <font name='Courier'>reports/make_figures.py</font> and "
      "the PDF with <font name='Courier'>reports/build_report.py</font>.")
    p("References", "sub")
    p("[1] Almgren &amp; Chriss (2000). Optimal execution of portfolio transactions. "
      "<i>Journal of Risk</i>, 3(2), 5-39.<br/>"
      "[2] Kaelbling, Littman &amp; Cassandra (1998). Planning and acting in partially "
      "observable stochastic domains. <i>Artificial Intelligence</i>, 101, 99-134.<br/>"
      "[3] Ross, Chaib-draa &amp; Pineau (2007). Bayes-Adaptive POMDPs. <i>NeurIPS 20</i>.",
      "caption")
    document.build(contents, onFirstPage=furniture, onLaterPages=furniture)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "research-paper.pdf")
    build(parser.parse_args().output)
