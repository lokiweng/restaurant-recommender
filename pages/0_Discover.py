"""
pages/0_Discover.py

The landing screen.

WHAT THIS SCREEN IS FOR
-----------------------
Two jobs, both of which the previous version failed at. First: say what the
app does and what data stands behind it, so a visitor is not guessing. Second:
make the path obvious. The old interface opened on a wall of controls with no
indication of what to do first — this one names the three steps before
anything else and gives exactly one primary button.

Nothing here is personalised, because at this point the app knows nothing
about the visitor. That is the cold-start problem, and the honest answer is to
show the highest-rated restaurants and label them as such, rather than
presenting a popularity list as if it were a recommendation.
"""

import streamlit as st

from core.explain import add_reasons
from ui.components import divider, eyebrow, lede, render_card_grid, stat_row
from ui.state import RATINGS_FOR_GOOD_RESULTS, boot, my_ratings

data, models, warnings = boot()
rated_count = len(my_ratings())

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
eyebrow(st, "Find somewhere to eat")
st.markdown("# Good food in Cleveland, chosen for you")
lede(
    st,
    f"Built on {data.n_reviews:,} real Yelp ratings from {data.n_users:,} diners "
    f"across {data.n_businesses:,} restaurants. Rate a few places you already "
    "know and the recommendations adapt to your taste.",
)

if warnings:
    with st.expander(f"⚠️ {len(warnings)} data quality note(s)", expanded=False):
        for warning in warnings:
            st.write(f"- {warning}")

st.write("")

# ---------------------------------------------------------------------------
# The three-step journey.
#
# This is the direct answer to "no clear flow from input to results". The
# steps are stated before any control appears, and the one the visitor is
# currently on is marked, so progress is visible rather than implied.
# ---------------------------------------------------------------------------
steps = [
    ("Step 1", "Rate a few restaurants", "Tell us about places you've been."),
    ("Step 2", "See your recommendations", "Built from what you rated."),
    ("Step 3", "Tell us how we did", "A short questionnaire on the results."),
]
current_step = 1 if rated_count == 0 else 2

for column, (index, (number, title, description)) in zip(st.columns(3), enumerate(steps, start=1)):
    with column:
        eyebrow(st, f"{number}{'  ·  you are here' if index == current_step else ''}")
        st.markdown(f"**{title}**")
        st.caption(description)

st.write("")

if rated_count == 0:
    if st.button("Start rating →", type="primary"):
        st.switch_page("pages/2_Rate.py")
else:
    left, right = st.columns([1, 3])
    with left:
        if st.button("See your recommendations →", type="primary"):
            st.switch_page("pages/3_Recommendations.py")
    with right:
        remaining = max(0, RATINGS_FOR_GOOD_RESULTS - rated_count)
        st.caption(
            f"{rated_count} rated so far."
            + (f" {remaining} more sharpens the results." if remaining else " That's plenty.")
        )

divider(st)

# ---------------------------------------------------------------------------
# Top rated — the honest cold-start state
# ---------------------------------------------------------------------------
if rated_count == 0:
    eyebrow(st, "Before you rate anything")
    st.markdown("## Highest rated in Cleveland")
    lede(
        st,
        "Ranked by a weighted rating that discounts restaurants with very few "
        "reviews. Identical for every visitor — this is the baseline the "
        "personalised models are measured against.",
    )
    picks = models["popularity"].recommend("", top_n=8, exclude_rated=False)
else:
    eyebrow(st, "Based on what you've rated")
    st.markdown("## A few you might like")
    lede(st, "From the hybrid model. The full list, and how each approach ranked it, is on the recommendations screen.")
    picks = models["hybrid"].recommend_from_ratings(my_ratings(), top_n=8)
    # Explanations only exist once there is something to explain them with.
    # Before any rating, the block above shows the popularity baseline, which
    # is the same list for everybody and has nothing personal to point at.
    picks = add_reasons(picks, my_ratings(), data.businesses)

render_card_grid(st, picks, score_column="score", score_label="score",
                 reason_column="reason" if rated_count else None,
                 empty_message="No restaurants to show.")

divider(st)

# ---------------------------------------------------------------------------
# Dataset summary
# ---------------------------------------------------------------------------
eyebrow(st, "The data behind it")
stat_row(
    st,
    [
        ("Restaurants", f"{data.n_businesses:,}"),
        ("Diners", f"{data.n_users:,}"),
        ("Ratings", f"{data.n_reviews:,}"),
        ("City", data.city),
    ],
)
st.caption(
    "Yelp Open Dataset, filtered to Cleveland restaurants and reduced to a 5-core "
    "(users and restaurants with at least five ratings each)."
)
