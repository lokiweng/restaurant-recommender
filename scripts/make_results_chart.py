"""
scripts/make_results_chart.py

Regenerates docs_assets/results_chart.png — Figure 2 in the documentation.

Run from the project root:

    python scripts/make_results_chart.py

WHY THIS RE-RUNS THE EVALUATION RATHER THAN PLOTTING STORED NUMBERS
-------------------------------------------------------------------
Hard-coding the figures into the plotting script is how a report ends up with a
chart that no longer matches its own table. Running the evaluation here means
the figure cannot drift from the code: change a model, re-run this, and the
figure in the report is correct by construction. It costs about a minute, which
is the right trade for a file that is regenerated a handful of times.

WHY matplotlib HERE AND altair IN THE APP
-----------------------------------------
Different medium, different tool. The app needs interactive SVG that inherits
the page's styling; a Word document needs a fixed-size raster at print
resolution. The two share the same palette and the same design decisions, so
the figure looks like it belongs to the same project.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")   # no display in a script; must be set before pyplot
import matplotlib.pyplot as plt

from core.collaborative import CollaborativeRecommender
from core.content_based import ContentBasedRecommender
from core.data import load_dataset
from core.evaluation import DEFAULT_K, evaluate_all
from core.hybrid import HybridRecommender
from core.popularity import PopularityRecommender

# The same two colours the app uses, for the same reason: the only distinction
# worth encoding is personalised versus non-personalised.
PERSONALISED = "#E8564F"
BASELINE = "#1B7FA8"
INK = "#222222"
MUTED = "#717171"
GRID = "#DDDDDD"

BASELINE_NAME = "Popularity baseline"

# Short names, because a print figure has no tooltip to fall back on.
SHORT = {
    "Popularity baseline": "Popularity\n(baseline)",
    "Content-based": "Content-based",
    "Collaborative filtering": "Collaborative",
    "Hybrid": "Hybrid",
}

# Marker shapes for the scatter. A second, non-colour channel for identity:
# it cannot collide the way text labels do, it survives greyscale printing, and
# it keeps the chart readable without relying on colour alone.
MARKERS = {
    "Popularity baseline": "o",
    "Content-based": "s",
    "Collaborative filtering": "^",
    "Hybrid": "D",
}


def style_axes(ax) -> None:
    """Recessive axes: the data should be the darkest thing on the chart."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.set_axisbelow(True)


def main() -> None:
    print("Loading data…")
    data = load_dataset()

    print("Running the evaluation (about a minute)…")
    results = evaluate_all(
        [PopularityRecommender(), ContentBasedRecommender(),
         CollaborativeRecommender(), HybridRecommender(alpha=0.5)],
        data, k=DEFAULT_K,
    )
    results["colour"] = results["model"].apply(
        lambda name: BASELINE if name == BASELINE_NAME else PERSONALISED
    )

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.4))

    # -- left: rating-prediction error ------------------------------------
    ordered = results.sort_values("rmse", ascending=True)
    labels = [SHORT.get(name, name) for name in ordered["model"]]
    positions = range(len(ordered))

    left.barh(list(positions), ordered["rmse"], height=0.58,
              color=ordered["colour"], zorder=3)
    left.set_yticks(list(positions))
    left.set_yticklabels(labels, fontsize=9.5, color=INK)
    left.invert_yaxis()                       # best at the top
    left.set_xlabel("RMSE (stars) — lower is better", fontsize=9.5, color=MUTED)
    left.set_title("Rating prediction", fontsize=11.5, fontweight="600",
                   color=INK, loc="left", pad=12)
    left.set_xlim(0, max(ordered["rmse"]) * 1.22)
    left.xaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
    style_axes(left)

    for y, value in zip(positions, ordered["rmse"]):
        left.text(value + max(ordered["rmse"]) * 0.02, y, f"{value:.3f}",
                  va="center", fontsize=9.5, color=INK)

    # -- right: the trade-off ---------------------------------------------
    #
    # Identity comes from marker SHAPE plus a legend, not from a label beside
    # each point. Three of the four models sit almost on top of one another at
    # high coverage, and no amount of offsetting keeps four text labels apart in
    # that space -- successive attempts just moved which pair collided. Shape is
    # a secondary encoding that cannot overlap, survives greyscale printing, and
    # stays readable for a colourblind reader.
    #
    # Colour still carries the one distinction the chart is about: personalised
    # against the non-personalised baseline.
    for _, row in results.iterrows():
        right.scatter(
            row["coverage"], row["f1_at_k"],
            s=210, c=row["colour"], marker=MARKERS.get(row["model"], "o"),
            edgecolors="white", linewidths=1.6, zorder=3,
            label=row["model"],
        )

    legend = right.legend(
        loc="upper right", frameon=False, fontsize=8.6,
        labelcolor=INK, handletextpad=0.5, borderaxespad=0.2,
    )
    for handle in legend.legend_handles:
        handle.set_sizes([70])

    # One annotation, naming the cluster rather than its members. The finding is
    # about the group -- three personalised models trading accuracy for coverage
    # -- and the individual figures are in the table beside this figure.
    personalised = results[results["model"] != BASELINE_NAME]
    right.annotate(
        "three personalised\nmodels",
        (personalised["coverage"].mean(), personalised["f1_at_k"].mean()),
        textcoords="offset points", xytext=(-16, -34),
        ha="right", fontsize=8.6, color=MUTED, style="italic",
    )

    right.set_xlabel("Catalogue coverage — share ever recommended",
                     fontsize=9.5, color=MUTED)
    right.set_ylabel(f"F1@{DEFAULT_K} — higher is better", fontsize=9.5, color=MUTED)
    right.set_title("Accuracy against coverage", fontsize=11.5, fontweight="600",
                    color=INK, loc="left", pad=12)
    right.set_xlim(-0.06, 1.12)
    right.set_ylim(0, max(results["f1_at_k"]) * 1.35)
    right.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    right.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    right.grid(True, color=GRID, linewidth=0.7, zorder=0)
    style_axes(right)

    fig.suptitle(
        f"Evaluation results — 80/20 per-user split, real Yelp data ({data.city}, "
        f"{data.n_reviews:,} ratings)",
        fontsize=12, fontweight="600", color=INK, x=0.02, ha="left", y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out_dir = Path(__file__).resolve().parent.parent / "docs_assets"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "results_chart.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")

    print(f"\nsaved figure -> {out_path}")
    print("\nNumbers in this figure:")
    for _, row in results.iterrows():
        print(f"  {row['model']:<24} RMSE {row['rmse']:.3f}  MSE {row['mse']:.3f}  "
              f"P@{DEFAULT_K} {row['precision_at_k']:.3f}  R@{DEFAULT_K} {row['recall_at_k']:.3f}  "
              f"F1@{DEFAULT_K} {row['f1_at_k']:.3f}  coverage {row['coverage']:.1%}")


if __name__ == "__main__":
    main()
