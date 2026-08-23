"""
scripts/make_flowchart.py

Regenerates docs_assets/flowchart.png — Figure 1 in the documentation.

Run from the project root:

    python scripts/make_flowchart.py

Every box below corresponds to a real file in this project, and the figure is
kept in step with the prose it illustrates: the validation stage, all four
models, and both halves of the evaluation appear because all of them are
described in the methodology. A figure that omits a stage its own caption
describes is worse than no figure at all.

A NOTE ON THE LAYOUT
--------------------
Positions are derived from two constants — ROW_GAP and the box half-extent —
rather than typed in one at a time. The first version of this script used
hand-picked y-values, and because FancyBboxPatch adds its padding *outside* the
height you give it, several boxes silently overlapped their neighbours and hid
the arrows between them. Deriving the rows means the spacing cannot drift out
of step with the box size again.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # no display in a script; must be set before pyplot
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Palette mirrors the application's, so the report and the prototype read as one
# piece of work.
DATA_FILL = "#E8EEF9"       # source data and preprocessing
PIPELINE_FILL = "#DCE8DC"   # things this project builds
MODEL_FILL = "#FCE8D5"      # the two personalised models
HYBRID_FILL = "#FCE0DE"     # their combination
BASELINE_FILL = "#DCEAF2"   # the non-personalised comparison
EVAL_FILL = "#EDE3F5"       # measurement
EDGE = "#333333"

PAD = 0.25          # FancyBboxPatch padding, added OUTSIDE the given height
BOX_H = 0.95        # standard box height
TALL_H = 1.55       # the two evaluation boxes carry more text
ROW_GAP = 2.15      # centre-to-centre spacing between rows

# Half the visual extent of a standard box, padding included. Every arrow starts
# and ends on these edges rather than at a guessed coordinate.
HALF = BOX_H / 2 + PAD
TALL_HALF = TALL_H / 2 + PAD

TOP = 20.7
row = lambda n: TOP - n * ROW_GAP        # noqa: E731 - a table of y-positions

fig, ax = plt.subplots(figsize=(9.6, 14.6))
ax.set_xlim(0, 10)
ax.set_ylim(0.0, 22.6)
ax.axis("off")


def box(cx, cy, w, h, text, facecolor, fontsize=10.0):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad={PAD},rounding_size=0.16",
        linewidth=1.3, edgecolor=EDGE, facecolor=facecolor, zorder=2,
    ))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            color="black", zorder=3, linespacing=1.45)


def arrow(x1, y1, x2, y2, style="arc3,rad=0.0"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=17,
        linewidth=1.4, color=EDGE, connectionstyle=style,
        shrinkA=0, shrinkB=0, zorder=1,
    ))


ax.text(5, 21.9, "Restaurant Recommender — System Architecture",
        ha="center", va="center", fontsize=13.5, fontweight="bold")

# ---- the linear pipeline -------------------------------------------------
box(5, row(0), 6.8, BOX_H, "Yelp Open Dataset\n(business + review JSON)", DATA_FILL)
arrow(5, row(0) - HALF, 5, row(1) + HALF)

box(5, row(1), 8.2, BOX_H,
    "Preprocessing (offline)\nOne city · restaurants only · 5-core density filter",
    DATA_FILL, fontsize=9.4)
arrow(5, row(1) - HALF, 5, row(2) + HALF)

box(5, row(2), 7.8, BOX_H,
    "data/   businesses.csv  ·  users.csv  ·  reviews.csv",
    PIPELINE_FILL, fontsize=9.6)
arrow(5, row(2) - HALF, 5, row(3) + HALF)

box(5, row(3), 8.6, BOX_H,
    "core/validation.py\nSchema · rating ranges · duplicates · orphan IDs — before any model is built",
    PIPELINE_FILL, fontsize=8.7)
arrow(5, row(3) - HALF, 5, row(4) + HALF)

box(5, row(4), 8.6, BOX_H,
    "Feature engineering\nTF-IDF over category tags + price   |   user × item rating matrix",
    PIPELINE_FILL, fontsize=9.0)

# ---- branch into the two personalised models ----------------------------
arrow(3.6, row(4) - HALF, 2.6, row(5) + HALF, "arc3,rad=-0.16")
arrow(6.4, row(4) - HALF, 7.4, row(5) + HALF, "arc3,rad=0.16")

box(2.6, row(5), 4.0, BOX_H,
    "core/content_based.py\nContent-based — cosine\nsimilarity on TF-IDF", MODEL_FILL, fontsize=8.7)
box(7.4, row(5), 4.0, BOX_H,
    "core/collaborative.py\nCollaborative filtering\nitem-based cosine similarity", MODEL_FILL, fontsize=8.7)

# ---- they combine into the hybrid; the baseline sits alongside -----------
arrow(3.4, row(5) - HALF, 5.4, row(6) + HALF, "arc3,rad=-0.16")
arrow(7.4, row(5) - HALF, 7.0, row(6) + HALF, "arc3,rad=0.16")

box(1.65, row(6), 2.7, BOX_H,
    "core/popularity.py\nBaseline\n(non-personalised)", BASELINE_FILL, fontsize=8.4)
box(6.3, row(6), 5.0, BOX_H,
    "core/hybrid.py\nHybrid — weighted blend (α)", HYBRID_FILL, fontsize=9.3)

arrow(6.3, row(6) - HALF, 5.6, row(7) + HALF, "arc3,rad=0.10")
arrow(1.65, row(6) - HALF, 3.4, row(7) + HALF, "arc3,rad=-0.14")

# ---- the application -----------------------------------------------------
box(5, row(7), 8.8, BOX_H,
    "Streamlit app  —  app.py · pages/ · ui/\nDiscover · Browse · Rate · Your picks · Evaluation · How it works",
    PIPELINE_FILL, fontsize=8.8)

# ---- both halves of the evaluation --------------------------------------
arrow(3.6, row(7) - HALF, 2.6, row(8) + TALL_HALF, "arc3,rad=-0.16")
arrow(6.4, row(7) - HALF, 7.4, row(8) + TALL_HALF, "arc3,rad=0.16")

box(2.6, row(8), 4.0, TALL_H,
    "core/evaluation.py\nQuantitative\n80/20 per-user split\nRMSE · MSE · Precision@K\nRecall@K · F1@K · coverage",
    EVAL_FILL, fontsize=8.1)
box(7.4, row(8), 4.0, TALL_H,
    "core/satisfaction.py\nQualitative\n3-item questionnaire\nrelevance · discovery · intent\n→ data/satisfaction.csv",
    EVAL_FILL, fontsize=8.1)

arrow(2.6, row(8) - TALL_HALF, 4.2, row(9) + HALF, "arc3,rad=0.14")
arrow(7.4, row(8) - TALL_HALF, 5.8, row(9) + HALF, "arc3,rad=-0.14")

box(5, row(9), 6.8, BOX_H,
    "Reported results\nscripts/run_evaluation.py  ·  scripts/make_results_chart.py",
    PIPELINE_FILL, fontsize=8.8)

ax.text(5, row(9) - HALF - 0.55,
        "Boxes naming a file correspond to that file in the submitted project.  "
        "`python -m pytest tests/` exercises the shaded stages.",
        ha="center", va="center", fontsize=8.4, color="#555555", style="italic")

fig.tight_layout()

out_dir = Path(__file__).resolve().parent.parent / "docs_assets"
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "flowchart.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
print(f"saved flowchart -> {out_path}")
