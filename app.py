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

# The application's structure, in the order a visitor should meet it: find
# something, look around, then the three steps of the journey, then the
# supporting material.
pages = [
    st.Page("pages/0_Discover.py", title="Discover", icon=":material/home:", default=True),
    st.Page("pages/1_Browse.py", title="Browse restaurants", icon=":material/grid_view:"),
    st.Page("pages/2_Rate.py", title="Rate — step 1", icon=":material/star:"),
    st.Page("pages/3_Recommendations.py", title="Your picks — step 2", icon=":material/recommend:"),
    st.Page("pages/4_Evaluation.py", title="How well does it work?", icon=":material/insights:"),
    st.Page("pages/5_About.py", title="How it works", icon=":material/info:"),
]

navigation = st.navigation(pages)

with st.sidebar:
    st.markdown("### 🍽️ Cleveland Eats")
    st.caption("Restaurant recommendations from real Yelp data")

navigation.run()
