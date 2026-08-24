"""
app.py

Entry point. This file is a router, not a screen — every screen lives in
pages/ and this decides which one is showing.

Run with:  streamlit run app.py
      or:  python -m streamlit run app.py     (if 'streamlit' is not on PATH)

WHY st.navigation RATHER THAN STREAMLIT'S AUTOMATIC PAGES
---------------------------------------------------------
Dropping files into a pages/ folder gives you navigation for free, but the
labels come from the filenames: the entry script shows up in the sidebar as
"app", and every page is stuck with whatever the file is called. Declaring the
pages explicitly costs a few lines and buys real titles, a chosen order, and
one place where the whole structure of the application is visible at a glance.

It also means set_page_config() and the stylesheet are applied exactly once,
here, rather than being repeated at the top of all six screens.

WHY THE PAGES ARE GROUPED
-------------------------
Six pages listed flat read as six equal choices, but they serve two different
readers. Four of them are the product — a diner arrives, looks around, rates a
few places, gets recommendations. The other two explain and measure the system
for anyone assessing it. Passing a dict rather than a list turns those groups
into sidebar headings, so a visitor can tell at a glance which screens are the
journey and which are the supporting material.
"""

import streamlit as st

from ui.theme import apply_theme

# Must be the first Streamlit call in the script.
st.set_page_config(
    page_title="Cleveland Eats",
    page_icon="🍽️",
    layout="wide",
    # "auto" lets Streamlit collapse the sidebar itself on narrow screens.
    # Forcing "expanded" keeps it open on a phone, where it covers the page.
    initial_sidebar_state="auto",
)

apply_theme(st)

# The application's structure, in the order a visitor should meet it. The dict
# keys become section headings in the sidebar.
#
# The step numbers that used to be in these titles ("Rate — step 1") have been
# dropped. They contradicted the sidebar: the page labelled step 1 sat third in
# the list, so a reader had to stop and work out whether the number or the
# position was wrong. Ordering already carries the sequence, and each screen
# points to the next one in its own body text.
pages = {
    "Find a restaurant": [
        st.Page(
            "pages/0_Discover.py",
            title="Discover",
            icon=":material/home:",
            default=True,
        ),
        st.Page(
            "pages/1_Browse.py",
            title="Browse restaurants",
            icon=":material/grid_view:",
        ),
        st.Page(
            "pages/2_Rate.py",
            title="Rate a few places",
            icon=":material/star:",
        ),
        st.Page(
            "pages/3_Recommendations.py",
            title="Your picks",
            icon=":material/recommend:",
        ),
    ],
    "Behind the system": [
        st.Page(
            "pages/4_Evaluation.py",
            title="How well does it work?",
            icon=":material/insights:",
        ),
        st.Page(
            "pages/5_About.py",
            title="How it works",
            icon=":material/info:",
        ),
    ],
}

navigation = st.navigation(pages)

with st.sidebar:
    st.markdown("### 🍽️ Cleveland Eats")
    st.caption("Restaurant recommendations from real Yelp data")

navigation.run()
