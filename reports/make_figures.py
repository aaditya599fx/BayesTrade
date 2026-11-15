"""Read saved evaluation outputs and render the paper's figures."""
import csv
import gzip
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)
INK, BLUE, RED, GRAY = "#17293f", "#136c80", "#b8583c", "#65768b"
plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "xtick.color": GRAY, "ytick.color": GRAY, "axes.axisbelow": True,
                     "figure.facecolor": "white"})


def read(filename):
    with (RESULTS / filename).open() as stream:
        return list(csv.DictReader(stream))


def save(figure, name):
    figure.savefig(FIGURES / name, dpi=180, bbox_inches="tight", pad_inches=.10)
    plt.close(figure)


def paired():
    rows = [row for row in read("paired.csv") if row["control"] in ("snapshot", "frozen")]
    names = [row["scenario"].replace("_", " ").capitalize() for row in rows]
    y = np.arange(len(rows))
    mean, lo, hi = [np.array([float(row[key]) for row in rows])
                    for key in ("mean_delta_bps", "ci_low", "ci_high")]
    fig, ax = plt.subplots(figsize=(6.7, 2.8))
    ax.axvspan(-.25, .25, color="#eaf3f3")
    ax.axvline(0, color=GRAY, linewidth=.9)
    for i, tag in enumerate(names):
        color = RED if rows[i]["model"] == "bamdp" else BLUE
        ax.errorbar(mean[i], y[i], xerr=[[mean[i] - lo[i]], [hi[i] - mean[i]]],
                    fmt="o", color=color, capsize=3, markersize=4)
    ax.set(yticks=y, yticklabels=names, xlabel="Model minus same-planner control (bps)",
           xlim=(-.31, .31))
    ax.invert_yaxis()
    ax.grid(axis="y", color="#edf0f3")
    save(fig, "paired.png")


def calibration():
    trace = read("trace.csv")
    with gzip.open(RESULTS / "beliefs.csv.gz", "rt") as stream:
        records = [row for row in csv.DictReader(stream) if row["scenario"] == "switching"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(7, 2.65), layout="constrained")
    steps = [int(row["step"]) for row in trace]
    left.step(steps, [int(row["regime"]) for row in trace], where="mid",
              linewidth=1.2, color=GRAY, label="True regime")
    left.plot(steps, [float(row["belief"]) for row in trace],
              color=BLUE, marker="o", markersize=3, label="Filtered belief")
    left.scatter(steps, [.055 if row["wide"] == "1" else .012 for row in trace],
                 color=RED, marker="|", label="Quote category")
    left.set(xlabel="Decision interval", ylabel="P(illiquid)", ylim=(-.05, 1.07),
             title="Belief trace")
    left.legend(frameon=False, fontsize=7, loc="upper center", ncols=3,
                bbox_to_anchor=(.5, -.23), columnspacing=.9, handlelength=1.5)
    right.plot([0, 1], [0, 1], linestyle=":", color=GRAY)
    for label, color in (("pomdp", BLUE), ("snapshot", RED)):
        rows = [row for row in records if row["policy"] == label]
        pred, truth = [np.array([float(row[key]) for row in rows]) for key in ("belief", "regime")]
        group = np.minimum((pred * 10).astype(int), 9)
        bins = [group == index for index in range(10)]
        points = [(pred[mask].mean(), truth[mask].mean(), int(mask.sum()))
                  for mask in bins if mask.sum() >= 15]
        x, y, count = np.asarray(points).T
        right.scatter(x, y, s=9 + 45 * count / count.max(), color=color, alpha=.85,
                      label=label.capitalize())
    right.set(xlabel="Mean predicted probability", ylabel="Observed frequency",
              xlim=(0, 1), ylim=(0, 1), title="Nominal reliability")
    right.legend(frameon=False, fontsize=7)
    save(fig, "calibration.png")


def cost_distribution():
    rows = read("episodes.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.5), layout="constrained", sharey=True)
    for ax, scenario in zip(axes, ("switching", "rare_stress"), strict=True):
        for policy, color in (("twap", GRAY), ("vwap", RED), ("pomdp", BLUE)):
            values = np.sort([float(row["shortfall_bps"]) for row in rows
                              if row["scenario"] == scenario and row["policy"] == policy])
            ax.step(values, np.arange(1, len(values) + 1) / len(values), where="post",
                    label=policy.upper(), color=color, linewidth=1.1)
        ax.set(xlabel="Shortfall (bps)", title=scenario.replace("_", " ").capitalize())
        ax.grid(axis="y", color="#edf0f3")
    axes[0].set_ylabel("Empirical cumulative probability")
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, "shortfall.png")


if __name__ == "__main__":
    paired()
    calibration()
    cost_distribution()
