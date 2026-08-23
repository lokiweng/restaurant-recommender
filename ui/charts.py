"""
ui/charts.py

The charts on the evaluation screen.

WHY ALTAIR AND NOT MATPLOTLIB
-----------------------------
Streamlit renders Altair natively as interactive SVG, so every mark gets a
hover tooltip for free and the type stays crisp at any zoom. Matplotlib would
produce a flat PNG whose default styling — heavy axes, a boxed frame, its own
font — visibly disagrees with the rest of the interface.

matplotlib is still used, in scripts/, for the figures that go into the Word
document. Different medium, different tool: print figures need a fixed size at
print resolution, screen figures need to be interactive.

CHART DESIGN RULES APPLIED HERE
-------------------------------
  * Thin marks, rounded data-ends, and a gap between adjacent bars, so the bars
    read as objects rather than as a solid block of ink.
  * Recessive grid and axes. The data is the darkest thing on the chart.
  * Two colours only, and they encode the one distinction the evaluation is
    actually about: personalised models versus the non-personalised baseline.
    Everything else is identified by a direct label, so nothing depends on
    colour alone.
  * A tooltip on every mark. An SVG chart in a browser is interactive; not
    wiring that up wastes it.
"""

import altair as alt
import pandas as pd

from ui.theme import CHART, FONT_STACK

BASELINE_NAME = "Popularity baseline"

# The questionnaire's answer scale. Pinned rather than fitted to the responses:
# an axis that started at the lowest score received would make a small spread
# look enormous, and the reader needs to see where these sit on the scale people
# were actually offered.
SCALE_FLOOR, SCALE_NEUTRAL, SCALE_CEILING = 1, 3, 5

# Altair's defaults are built for exploratory analysis, not for embedding in a
# designed page. Registering a theme once means every chart in the app inherits
# the same typography, the same recessive axes, and the same absence of a
# boxed frame -- rather than each chart re-specifying it.
def _app_theme():
    return {
        "config": {
            "font": FONT_STACK,
            "view": {"stroke": "transparent"},          # no box around the plot
            "background": "transparent",
            "axis": {
                "labelColor": CHART["axis"],
                "titleColor": CHART["axis"],
                "labelFontSize": 12,
                "titleFontSize": 12,
                "titleFontWeight": "normal",
                "domainColor": CHART["grid"],
                "tickColor": CHART["grid"],
                "gridColor": CHART["grid"],
                "gridWidth": 1,
                "labelPadding": 6,
            },
            "legend": {
                "labelColor": CHART["axis"],
                "titleColor": CHART["axis"],
                "labelFontSize": 12,
                "titleFontSize": 12,
                "titleFontWeight": "normal",
                "symbolType": "circle",
                "orient": "top",
                "direction": "horizontal",
                "offset": 8,
            },
            "title": {
                "color": CHART["label"],
                "fontSize": 14,
                "fontWeight": 600,
                "anchor": "start",
                "offset": 10,
            },
        }
    }


alt.theme.register("cleveland_eats", enable=True)(_app_theme)


def _with_group(results: pd.DataFrame) -> pd.DataFrame:
    """Tag each row as baseline or personalised — the chart's one colour split."""
    frame = results.copy()
    frame["group"] = frame["model"].apply(
        lambda name: "Baseline" if name == BASELINE_NAME else "Personalised"
    )
    return frame


# The colour scale is declared once and reused, so the baseline is the same
# colour on every chart. Colour follows the entity, never its position in a
# sorted list.
_GROUP_SCALE = alt.Scale(
    domain=["Personalised", "Baseline"],
    range=[CHART["personalised"], CHART["baseline"]],
)


def rmse_chart(results: pd.DataFrame) -> alt.Chart:
    """Rating-prediction error by model. Lower is better.

    A horizontal bar chart: the job is comparing one magnitude across four named
    categories, and horizontal bars give the model names room to sit as readable
    text rather than as rotated axis labels.
    """
    # Sorted here, in the data, rather than through Vega-Lite's sort channel.
    # Both sort="x" and an explicit EncodingSortField were silently ignored
    # inside the layered chart below, leaving the bars in alphabetical order --
    # which looks deliberate, because tidy alphabetical order is indistinguishable
    # from tidy sorted order at a glance. Sorting the frame and pinning
    # sort=None ("use data order") cannot fail quietly in the same way.
    frame = _with_group(results).sort_values("rmse", ascending=True)

    bars = (
        alt.Chart(frame)
        .mark_bar(
            height=22,               # thin marks; the gap between them is the point
            cornerRadiusEnd=4,       # rounded at the data end, square at the baseline
        )
        .encode(
            y=alt.Y("model:N", title=None, sort=None,
                    # Model names are long; without a generous label limit
                    # Altair truncates them to "Collaborative filt...".
                    axis=alt.Axis(labelLimit=220)),
            x=alt.X("rmse:Q", title="RMSE (stars — lower is better)",
                    scale=alt.Scale(zero=True)),
            color=alt.Color("group:N", scale=_GROUP_SCALE, title=None),
            tooltip=[
                alt.Tooltip("model:N", title="Model"),
                alt.Tooltip("rmse:Q", title="RMSE", format=".3f"),
                alt.Tooltip("mse:Q", title="MSE", format=".3f"),
                alt.Tooltip("n_predictions:Q", title="Predictions", format=","),
            ],
        )
    )

    # Direct labels on the bars, in ink rather than in the series colour, so the
    # value is readable without chasing an axis.
    labels = bars.mark_text(
        align="left", dx=6, fontSize=12, color=CHART["label"],
    ).encode(x="rmse:Q", text=alt.Text("rmse:Q", format=".3f"), color=alt.value(CHART["label"]))

    return (bars + labels).properties(height=alt.Step(34))


def tradeoff_chart(results: pd.DataFrame) -> alt.Chart:
    """Ranking quality against catalogue coverage — the central finding.

    A scatter, because the job here is a *relationship* between two measures,
    not a magnitude. Reading the two as separate bar charts would show both
    numbers and hide the thing that matters: that the model scoring best on
    accuracy is the one recommending almost nothing.

    Every point is directly labelled, so identity never rests on colour.
    """
    frame = _with_group(results)

    # Labels are placed away from the edge they are nearest, and nudged apart
    # where two points sit almost on top of each other. Both are necessary here
    # rather than fussy: the three personalised models cluster tightly at high
    # coverage, so a single fixed offset either clipped the longest name off the
    # right-hand edge or stacked two labels on top of one another.
    frame = frame.sort_values("f1_at_k").reset_index(drop=True)

    # Vertical nudge: walk up the y-axis and push a point's label clear of the
    # previous one whenever they are within a tenth of the axis range.
    span = float(frame["f1_at_k"].max() - frame["f1_at_k"].min()) or 1.0
    offsets, previous_y, direction = [], None, -1
    for value in frame["f1_at_k"]:
        if previous_y is not None and (value - previous_y) / span < 0.10:
            direction = -direction          # alternate above/below
            offsets.append(-13 if direction < 0 else 13)
        else:
            direction = -1
            offsets.append(-13)
        previous_y = value
    frame["label_dy"] = offsets

    # Horizontal side: points past the midpoint get their label on the left, so
    # a long name grows into the empty middle of the chart instead of off the edge.
    frame["label_side"] = frame["coverage"].apply(lambda c: "right" if c > 0.5 else "left")

    base = alt.Chart(frame).encode(
        x=alt.X("coverage:Q", title="Catalogue coverage (share of restaurants ever recommended)",
                axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("f1_at_k:Q", title="F1@10 (higher is better)", scale=alt.Scale(zero=True)),
    )

    points = base.mark_circle(
        size=260, opacity=0.9, stroke="white", strokeWidth=2,
    ).encode(
        color=alt.Color("group:N", scale=_GROUP_SCALE, title=None),
        tooltip=[
            alt.Tooltip("model:N", title="Model"),
            alt.Tooltip("f1_at_k:Q", title="F1@10", format=".3f"),
            alt.Tooltip("precision_at_k:Q", title="Precision@10", format=".3f"),
            alt.Tooltip("recall_at_k:Q", title="Recall@10", format=".3f"),
            alt.Tooltip("coverage:Q", title="Coverage", format=".1%"),
        ],
    )

    # One text layer per (side, nudge) combination, because align and dy are
    # mark properties in Vega-Lite and cannot vary row by row within a layer.
    layers = [points]
    for side in ("left", "right"):
        for nudge in sorted(frame["label_dy"].unique()):
            subset = frame[(frame["label_side"] == side) & (frame["label_dy"] == nudge)]
            if subset.empty:
                continue
            layers.append(
                alt.Chart(subset)
                .mark_text(
                    align=side,
                    dx=12 if side == "left" else -12,
                    dy=int(nudge),
                    fontSize=12,
                    fontWeight=500,
                )
                .encode(
                    x=alt.X("coverage:Q", scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("f1_at_k:Q", scale=alt.Scale(zero=True)),
                    text="model:N",
                    color=alt.value(CHART["label"]),
                )
            )

    return alt.layer(*layers).properties(height=340)


def ranking_metrics_chart(results: pd.DataFrame) -> alt.Chart:
    """Precision, recall and F1 side by side, one small multiple per metric.

    Faceting rather than crowding three measures into one grouped bar chart:
    recall runs about five times larger than precision here, so on a shared
    axis precision would be squashed into invisibility. Separate panels let
    each metric use the scale that suits it, and the comparison between models
    — which is what the reader is here for — stays intact within each panel.
    """
    frame = _with_group(results)

    tidy = frame.melt(
        id_vars=["model", "group"],
        value_vars=["precision_at_k", "recall_at_k", "f1_at_k"],
        var_name="metric",
        value_name="value",
    )
    tidy["metric"] = tidy["metric"].map({
        "precision_at_k": "Precision@10",
        "recall_at_k": "Recall@10",
        "f1_at_k": "F1@10",
    })

    # One order for every panel, taken from F1 -- the metric that balances the
    # other two. Letting each panel sort itself would put the models in a
    # different order in each one, which makes them impossible to compare.
    model_order = (
        results.sort_values("f1_at_k", ascending=False)["model"].tolist()
    )

    return (
        alt.Chart(tidy)
        .mark_bar(height=16, cornerRadiusEnd=4)
        .encode(
            y=alt.Y("model:N", title=None, sort=model_order, axis=alt.Axis(labelLimit=220)),
            x=alt.X("value:Q", title=None, scale=alt.Scale(zero=True)),
            color=alt.Color("group:N", scale=_GROUP_SCALE, title=None),
            tooltip=[
                alt.Tooltip("model:N", title="Model"),
                alt.Tooltip("metric:N", title="Metric"),
                alt.Tooltip("value:Q", title="Value", format=".3f"),
            ],
        )
        .properties(height=alt.Step(26), width=200)
        .facet(
            column=alt.Column("metric:N", title=None, sort=["Precision@10", "Recall@10", "F1@10"]),
        )
        .resolve_scale(x="independent")   # each metric gets the scale it needs
    )


def satisfaction_chart(frame: pd.DataFrame) -> alt.Chart:
    """Mean score for each questionnaire item, on the 1-5 scale it was asked on.

    WHY DOTS AND NOT BARS
    ---------------------
    The scale starts at 1, not 0 — nobody can score below 1, so an axis running
    from 0 would waste a fifth of its width on impossible values. But a *bar* on
    an axis that starts at 1 is dishonest: a bar communicates through its length,
    and once the baseline moves off zero its length no longer means anything. A
    4.5 and a 2.5 would look four times apart instead of twice.

    A dot has no length to misread. Its position on a fixed 1-5 axis is the whole
    message, and a faint line back to the scale floor gives the eye something to
    follow without implying magnitude. The first version of this chart was bars,
    and Vega-Lite quietly clipped them against the domain — which is what made
    the problem visible.

    A dashed reference at 3 marks the neutral midpoint, so a reader can see at a
    glance whether a score is positive or merely not-negative.
    """
    plot = frame.copy()
    plot["floor"] = SCALE_FLOOR   # where the connector line starts

    # The scale object is built once and handed to every layer. Reaching for it
    # through an already-built channel (shared_x.scale) returns Altair's property
    # setter rather than the Scale itself, which fails schema validation at
    # render time -- so it is defined standalone here.
    x_scale = alt.Scale(domain=[SCALE_FLOOR, SCALE_CEILING], nice=False)

    shared_y = alt.Y("short:N", title=None, sort=None, axis=alt.Axis(labelLimit=220))
    shared_x = alt.X("mean:Q", title="Mean score (1–5)", scale=x_scale)
    shared_tooltip = [
        alt.Tooltip("question:N", title="Question"),
        alt.Tooltip("mean:Q", title="Mean", format=".2f"),
        alt.Tooltip("std:Q", title="Std dev", format=".2f"),
    ]

    connector = (
        alt.Chart(plot)
        .mark_rule(strokeWidth=2, color=CHART["grid"])
        .encode(y=shared_y, x=alt.X("floor:Q", title=None, scale=x_scale), x2="mean:Q")
    )

    neutral = (
        alt.Chart(pd.DataFrame({"x": [SCALE_NEUTRAL]}))
        .mark_rule(strokeDash=[4, 4], strokeWidth=1, color=CHART["axis"], opacity=0.55)
        .encode(x=alt.X("x:Q", title=None, scale=x_scale))
    )

    dots = (
        alt.Chart(plot)
        .mark_circle(size=200, color=CHART["personalised"], opacity=1,
                     stroke="white", strokeWidth=2)
        .encode(y=shared_y, x=shared_x, tooltip=shared_tooltip)
    )

    labels = (
        alt.Chart(plot)
        .mark_text(align="left", dx=13, fontSize=12, fontWeight=500)
        .encode(y=shared_y, x=shared_x, text=alt.Text("mean:Q", format=".2f"),
                color=alt.value(CHART["label"]))
    )

    return alt.layer(neutral, connector, dots, labels).properties(height=alt.Step(44))
