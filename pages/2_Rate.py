"""
pages/2_Rate.py

Step 1 of the journey: the visitor tells the app what they already like.

WHY THIS SCREEN EXISTS AT ALL
-----------------------------
Every personalised recommendation in this project traces back to what is
collected here. Without it there is no user to recommend for, and the app can
only ever show the popularity baseline.

HOW THIS DIFFERS FROM BROWSE, NOW THAT BROWSE CAN ALSO RATE
-----------------------------------------------------------
Once cards on the catalogue screen became rateable in place, these two pages
briefly did the same job through two different interfaces, which is worse than
either alone: a visitor cannot predict which one to open.

They now split by what the visitor already knows. Browse is for finding
something -- filter by cuisine, price and rating, recognise a place, rate it
there. This screen is for a visitor who already has a name in mind and does
not want to filter their way to it, and it is where the ratings collected from
anywhere in the app are reviewed, annotated and removed. Searching is
incidental here; the list below is the point.

The ratings are held in st.session_state and are never written back to the
CSV files. That is a deliberate boundary: the dataset the models were fitted
on stays untouched, and personalisation happens through the models'
recommend_from_ratings() path, which needs no retraining at all. Someone can
rate five restaurants and get personalised results in the same second.
"""

import streamlit as st

from ui.components import divider, eyebrow, lede, price_label, stars
from ui.state import (RATINGS_FOR_GOOD_RESULTS, boot, clear_ratings, my_notes,
                      my_ratings, remove_rating, set_rating)

data, models, _ = boot()
ratings = my_ratings()

eyebrow(st, "Step 1 of 2")
st.markdown("# Your ratings")
lede(
    st,
    "Add a place by name, and review everything you have rated so far. "
    "Three or four ratings is enough to see the recommendations change.",
)

if not ratings:
    st.caption(
        "Know the name already? Add it below. Otherwise **Browse restaurants** "
        "lets you filter by cuisine and price and rate the cards as you go."
    )

# ---------------------------------------------------------------------------
# Progress.
#
# A count on its own ("2 rated") tells the visitor nothing about whether that
# is enough. A progress bar against a stated target answers the question they
# actually have, which is "can I stop yet?".
# ---------------------------------------------------------------------------
progress = min(1.0, len(ratings) / RATINGS_FOR_GOOD_RESULTS)
st.progress(progress)

if len(ratings) == 0:
    st.caption(f"Rate {RATINGS_FOR_GOOD_RESULTS} restaurants for good results.")
elif len(ratings) < RATINGS_FOR_GOOD_RESULTS:
    remaining = RATINGS_FOR_GOOD_RESULTS - len(ratings)
    st.caption(f"{len(ratings)} rated — {remaining} more for good results.")
else:
    st.caption(f"{len(ratings)} rated — enough for personalised picks. More will sharpen them.")

divider(st)

# ---------------------------------------------------------------------------
# The rating form
# ---------------------------------------------------------------------------
# Restaurants are offered most-reviewed first, because a visitor is far more
# likely to recognise a place with 400 reviews than one with 5. Recognition is
# the bottleneck on this screen, not choice.
options = (
    data.businesses.sort_values("review_count", ascending=False)["business_id"].tolist()
)
labels = {
    row["business_id"]: f"{row['name']} · {row['primary_category']} · {price_label(row['price_range'])}"
    for _, row in data.businesses.iterrows()
}

with st.form("rate_restaurant", clear_on_submit=True):
    st.markdown("**Add a rating**")

    chosen = st.selectbox(
        "Restaurant",
        options,
        format_func=lambda b: labels.get(b, b),
        index=None,
        placeholder="Search for a restaurant by name…",
    )

    form_columns = st.columns([1, 2])
    with form_columns[0]:
        # A radio rather than a slider, with no default. An untouched slider
        # sitting at 3 looks like an answer without being one.
        score = st.radio(
            "Your rating", [1, 2, 3, 4, 5],
            index=None, horizontal=True,
            format_func=lambda v: f"{v}★",
        )
    with form_columns[1]:
        note = st.text_input("A note, if you like (optional)", placeholder="What did you think?")

    submitted = st.form_submit_button("Save rating", type="primary")

if submitted:
    if chosen is None:
        st.warning("Choose a restaurant first.")
    elif score is None:
        st.warning("Give it a rating from 1 to 5.")
    else:
        set_rating(chosen, score, note)
        st.success(f"Saved {score}★ for **{labels.get(chosen, chosen)}**.")
        # Re-read rather than st.rerun(). A rerun would restart the script from
        # the top and wipe the confirmation above before anyone could read it;
        # re-reading picks up the rating just saved so the list below is current
        # in this same pass.
        ratings = my_ratings()

# ---------------------------------------------------------------------------
# What has been rated so far
# ---------------------------------------------------------------------------
if ratings:
    divider(st)
    st.markdown(f"### Your ratings ({len(ratings)})")

    notes = my_notes()
    for business_id, score in list(ratings.items()):
        row_left, row_right = st.columns([6, 1])
        with row_left:
            st.markdown(
                f'<span class="rr-stars">{stars(score)}</span> '
                f'&nbsp;{labels.get(business_id, business_id)}',
                unsafe_allow_html=True,
            )
            if business_id in notes:
                st.caption(f"“{notes[business_id]}”")
        with row_right:
            # Keyed by business_id so Streamlit can tell the buttons apart --
            # without a unique key every remove button would be the same widget.
            if st.button("Remove", key=f"remove_{business_id}", type="secondary"):
                remove_rating(business_id)
                st.rerun()

    divider(st)

    action_left, action_right = st.columns([1, 3])
    with action_left:
        if st.button("See your recommendations →", type="primary"):
            st.switch_page("pages/3_Recommendations.py")
    with action_right:
        if st.button("Clear all ratings", type="secondary"):
            clear_ratings()
            st.rerun()
