"""
pages/3_Recommendations.py

Step 2: the payoff screen.

This is what the whole application exists to produce, so it leads with the
answer — the recommendations themselves — and puts the machinery underneath,
where anyone curious can open it without it getting in the way of everyone
else.

THE COMPARISON SECTION
----------------------
Below the picks, the same session ratings are run through all four models side
by side. That section is doing real work for the assignment: it is the visible
evidence that three distinct approaches were implemented and that they behave
differently, which is otherwise only visible in the evaluation table.

It is collapsed by default because a diner does not care which algorithm chose
their dinner, and expanded by anyone assessing the project.
"""

import streamlit as st

from ui.components import divider, eyebrow, lede, render_card_grid
from ui.state import (RATINGS_FOR_GOOD_RESULTS, boot, my_ratings)

data, models, _ = boot()
ratings = my_ratings()

eyebrow(st, "Step 2 of 3")
st.markdown("# Your picks")

# ---------------------------------------------------------------------------
# Cold start.
#
# Rather than hiding this state or filling it with fake personalisation, it is
# named. The cold-start problem is a genuine, well-documented limitation of
# collaborative filtering and showing it honestly is more defensible than
# disguising a popularity list as a recommendation.
# ---------------------------------------------------------------------------
if not ratings:
    lede(
        st,
        "You haven't rated anything yet, so there is nothing personal to go on. "
        "These are the highest-rated restaurants overall — the same list every "
        "visitor sees.",
    )

    st.info(
        "**This is the cold-start problem.** A recommender cannot personalise for "
        "someone it knows nothing about. Rating even three restaurants is enough "
        "for the models to produce something genuinely yours.",
        icon=":material/info:",
    )

    if st.button("Rate some restaurants →", type="primary"):
        st.switch_page("pages/2_Rate.py")

    divider(st)
    render_card_grid(st, models["popularity"].recommend("", top_n=9, exclude_rated=False))
    st.stop()

# ---------------------------------------------------------------------------
# Personalised results
# ---------------------------------------------------------------------------
lede(
    st,
    f"Built from the {len(ratings)} restaurant{'s' if len(ratings) != 1 else ''} you rated, "
    "blending both approaches. Nothing was retrained — your ratings are scored "
    "against models that were already fitted.",
)

if len(ratings) < RATINGS_FOR_GOOD_RESULTS:
    st.caption(
        f"⚠️ Only {len(ratings)} rating so far, so these are still thin. "
        f"{RATINGS_FOR_GOOD_RESULTS - len(ratings)} more will change them noticeably."
    )

hybrid = models["hybrid"]
picks = hybrid.recommend_from_ratings(ratings, top_n=9)

render_card_grid(
    st, picks, score_column="score", score_label="match",
    empty_message="No recommendations could be produced from these ratings.",
)

action_left, action_right = st.columns([1, 3])
with action_left:
    if st.button("Rate more →", type="secondary"):
        st.switch_page("pages/2_Rate.py")
with action_right:
    st.caption("The match score is the blended model output — higher means a closer fit to your ratings.")

divider(st)

# ---------------------------------------------------------------------------
# How each approach sees it
# ---------------------------------------------------------------------------
with st.expander("How did each approach rank these?", expanded=False):
    st.caption(
        "The same ratings, scored by each model on its own. The differences are "
        "the point: content-based matches cuisine and price, collaborative "
        "filtering matches other diners' behaviour, and the popularity baseline "
        "ignores you entirely."
    )

    # The alpha slider makes the hybrid's central parameter tangible: sliding
    # it to either end reproduces one of the two models it blends.
    alpha = st.slider(
        "Hybrid blend (α) — 0 is pure content-based, 1 is pure collaborative",
        min_value=0.0, max_value=1.0, value=hybrid.alpha, step=0.1,
    )
    hybrid.alpha = alpha

    comparison = [
        ("Content-based", models["content"], "Matches cuisine and price to what you rated highly."),
        ("Collaborative", models["collaborative"], "Matches diners whose ratings look like yours."),
        (f"Hybrid (α = {alpha:g})", hybrid, "A weighted blend of the two."),
        ("Popularity", models["popularity"], "Ignores you completely — the baseline."),
    ]

    for column, (label, model, explanation) in zip(st.columns(4), comparison):
        with column:
            st.markdown(f"**{label}**")
            st.caption(explanation)
            results = model.recommend_from_ratings(ratings, top_n=5)
            if results.empty:
                st.caption("_No signal from these ratings._")
                continue
            for rank, (_, row) in enumerate(results.iterrows(), start=1):
                st.markdown(f"{rank}. {row['name']}")
                st.caption(f"{row['primary_category']} · ⭐ {row['avg_rating']:.1f}")

divider(st)

step_left, step_right = st.columns([1, 3])
with step_left:
    if st.button("Step 3 — tell us how we did →", type="primary"):
        st.switch_page("pages/4_Evaluation.py")
with step_right:
    st.caption("The questionnaire is the qualitative half of the evaluation the assignment requires.")
