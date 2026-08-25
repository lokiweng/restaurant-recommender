"""
pages/1_Browse.py

The full catalogue, with filters -- and, now, the ability to rate anything in
it without leaving the page.

DESIGN NOTE: WHY THE FILTERS ARE NOT IN THE SIDEBAR
---------------------------------------------------
Streamlit's convention is to put controls in the sidebar, and that is where
the previous version had them. It is the wrong place here for two reasons:
the sidebar is already the app's navigation, so mixing "where am I going" with
"what am I filtering" muddles both; and on a narrow screen the sidebar
collapses, hiding the filters entirely.

Putting them in a row above the results keeps the controls and the thing they
control on the same screen, which is how every real restaurant site does it.

DESIGN NOTE: WHY THE CARDS ARE RATEABLE HERE
--------------------------------------------
Until now this screen was a dead end. A visitor could search 817 restaurants,
recognise one they had actually been to, and then had no way to say so -- the
only rating control lived on another page behind a dropdown, so acting on a
recognition meant retyping the name they were already looking at.

That is backwards. Recognition is the scarce thing in this app: a visitor can
only rate places they have been, and browsing is how they find them. The
rating control belongs where the recognition happens.

It also removes the app's worst piece of duplicated work. Search-by-name
exists twice -- once as this page's filter, once as the dropdown on the Rate
screen -- and the two do the same job with different interfaces. This page now
does the job properly, and the Rate screen keeps its dropdown for the visitor
who arrives already knowing the name they want.
"""

import pandas as pd
import streamlit as st

from ui.components import (divider, eyebrow, lede, render_rateable_grid,
                           RATEABLE_COLUMNS)
from ui.state import (RATINGS_FOR_GOOD_RESULTS, boot, my_ratings,
                      remove_rating, set_rating)

# Four rows of three. A page that ends mid-row looks broken, so the page size
# is a multiple of the column count rather than a round number.
PAGE_SIZE = RATEABLE_COLUMNS * 4

data, models, _ = boot()
ratings = my_ratings()

eyebrow(st, "The whole catalogue")
st.markdown("# Browse restaurants")
lede(st, f"All {data.n_businesses:,} restaurants in the dataset. Rate any of them here — "
         "the recommendations update from whatever you rate.")

# ---------------------------------------------------------------------------
# Progress.
#
# Repeated from the Rate screen on purpose. Someone who starts on this page
# and never visits that one still needs to know how many ratings make the
# recommendations worth looking at.
# ---------------------------------------------------------------------------
if ratings:
    rated_count = len(ratings)
    remaining = max(0, RATINGS_FOR_GOOD_RESULTS - rated_count)

    progress_left, progress_right = st.columns([3, 1])
    with progress_left:
        st.caption(
            f"**{rated_count} rated.** "
            + (f"{remaining} more for good results."
               if remaining else "That's enough for personalised picks.")
        )
    with progress_right:
        if st.button("See your recommendations →", type="primary", use_container_width=True):
            st.switch_page("pages/3_Recommendations.py")

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
businesses = data.businesses

cuisines = sorted(businesses["primary_category"].dropna().unique().tolist())

filter_columns = st.columns([2, 1.6, 1.2, 1.2])

with filter_columns[0]:
    search = st.text_input("Search by name", placeholder="e.g. Bistro, Taco, Noodle")

with filter_columns[1]:
    cuisine = st.selectbox("Cuisine", ["All cuisines"] + cuisines)

with filter_columns[2]:
    price = st.selectbox("Price", ["Any price", "$", "$$", "$$$", "$$$$"])

with filter_columns[3]:
    min_rating = st.selectbox("Minimum rating", ["Any rating", "3.0+", "3.5+", "4.0+", "4.5+"])

# Apply each filter in turn. Written as separate steps rather than one long
# boolean expression so that each condition is readable on its own line and
# can be explained independently.
results = businesses.copy()

if search.strip():
    # case=False makes the search case-insensitive; na=False stops a missing
    # name from raising rather than simply not matching.
    results = results[results["name"].str.contains(search.strip(), case=False, na=False)]

if cuisine != "All cuisines":
    results = results[results["primary_category"] == cuisine]

if price != "Any price":
    results = results[pd.to_numeric(results["price_range"], errors="coerce") == len(price)]

if min_rating != "Any rating":
    threshold = float(min_rating.rstrip("+"))
    results = results[pd.to_numeric(results["avg_rating"], errors="coerce") >= threshold]

# Sorted by rating then review count, so that a single five-star review cannot
# outrank a restaurant with hundreds of consistently good ones.
results = results.sort_values(["avg_rating", "review_count"], ascending=False)

divider(st)

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
total = len(results)

if total == 0:
    st.markdown(
        '<div class="rr-empty">No restaurants match those filters.<br>'
        'Try widening the search — or clear one of them.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)

header_left, header_right = st.columns([3, 1])
with header_left:
    st.markdown(f"**{total:,}** restaurant{'s' if total != 1 else ''} found")
with header_right:
    page = st.number_input(
        "Page", min_value=1, max_value=total_pages, value=1, step=1,
        label_visibility="collapsed",
        help=f"{total_pages} page{'s' if total_pages != 1 else ''} of results",
    ) if total_pages > 1 else 1

start = (int(page) - 1) * PAGE_SIZE
page_of_results = results.iloc[start:start + PAGE_SIZE]

render_rateable_grid(
    st,
    page_of_results,
    ratings=ratings,
    on_rate=set_rating,
    on_remove=remove_rating,
    # Scoped to this page so the widget keys cannot collide with a rateable
    # grid on any other screen.
    key_prefix="browse",
)

if total_pages > 1:
    st.caption(f"Page {int(page)} of {total_pages}")
