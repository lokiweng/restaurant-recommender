"""
ui/state.py

Everything the screens share: the loaded dataset, the fitted models, and the
visitor's own ratings for this session.

WHY THIS FILE MATTERS MORE THAN IT LOOKS
----------------------------------------
Streamlit re-runs the entire script from the top every time anyone touches a
widget. Without caching, changing a filter dropdown would re-read three CSV
files and re-fit four models — several seconds of work, on every keystroke.

Two different caches are used, and the distinction is deliberate:

  @st.cache_data     for the dataset. Streamlit copies the cached value on
                     each access, so one page cannot corrupt another page's
                     view of the data. Right for plain values.

  @st.cache_resource for the fitted models. Returns the *same* object to every
                     caller rather than a copy — which is what we want, since
                     re-copying four fitted similarity matrices on every rerun
                     would defeat the point entirely.

Fitting the models this way also means content-based and collaborative are
each fitted exactly once and then *shared* with the hybrid, rather than the
hybrid fitting its own second copy of both.
"""

import streamlit as st

from core.collaborative import CollaborativeRecommender
from core.content_based import ContentBasedRecommender
from core.data import Dataset, DatasetError, load_dataset
from core.hybrid import HybridRecommender
from core.popularity import PopularityRecommender
from core.validation import DataValidationError, validate

# Session-state keys, named once here so a typo in one screen cannot silently
# create a second, empty ratings dictionary.
RATINGS_KEY = "my_ratings"      # {business_id: 1-5}
NOTES_KEY = "my_notes"          # {business_id: "free text"}
SURVEY_KEY = "survey_done"      # set once the questionnaire is submitted


@st.cache_data(show_spinner="Loading restaurants…")
def get_dataset() -> Dataset:
    return load_dataset()


@st.cache_data(show_spinner=False)
def get_data_warnings(_data: Dataset) -> list[str]:
    """Non-fatal data problems, surfaced in the interface rather than hidden.

    The leading underscore tells Streamlit not to try to hash the Dataset when
    deciding whether the cache is still valid — DataFrames are expensive to
    hash and this value only ever changes when the CSVs do.
    """
    return validate(_data)


@st.cache_resource(show_spinner="Fitting the recommendation models…")
def get_models(_data: Dataset) -> dict:
    """Fit all four models once per session and return them by key.

    Order matters here: content-based and collaborative are fitted first, then
    handed to the hybrid already fitted. Constructing the hybrid without them
    would make it fit its own private copies — doubling the work and doubling
    the memory for no benefit.
    """
    content = ContentBasedRecommender().fit(_data)
    collaborative = CollaborativeRecommender().fit(_data)
    hybrid = HybridRecommender(alpha=0.5, content=content, collaborative=collaborative).fit(_data)
    popularity = PopularityRecommender().fit(_data)

    return {
        "hybrid": hybrid,
        "content": content,
        "collaborative": collaborative,
        "popularity": popularity,
    }


def boot() -> tuple[Dataset, dict, list[str]]:
    """Load data, fit models, initialise session keys.

    Called at the top of every screen. Returns (dataset, models, warnings), or
    stops the page with a readable message if the data cannot be loaded at all.
    """
    try:
        data = get_dataset()
    except DatasetError as exc:
        st.error(f"**Could not load the dataset.**\n\n{exc}")
        st.stop()

    try:
        warnings = get_data_warnings(data)
    except DataValidationError as exc:
        # A fatal validation error means a model would produce wrong answers
        # rather than crash, so the app refuses to start instead.
        st.error(f"**The dataset failed validation and cannot be used.**\n\n{exc}")
        st.stop()

    models = get_models(data)

    st.session_state.setdefault(RATINGS_KEY, {})
    st.session_state.setdefault(NOTES_KEY, {})
    st.session_state.setdefault(SURVEY_KEY, False)

    return data, models, warnings


# ---------------------------------------------------------------------------
# The visitor's own ratings.
#
# These live in st.session_state and are never written back to the CSV files.
# The dataset the models were fitted on stays exactly as it was — a live
# session personalises through recommend_from_ratings(), which needs no
# retraining at all.
# ---------------------------------------------------------------------------

def my_ratings() -> dict:
    return st.session_state.get(RATINGS_KEY, {})


def my_notes() -> dict:
    return st.session_state.get(NOTES_KEY, {})


def set_rating(business_id: str, rating: int, note: str = "") -> None:
    st.session_state[RATINGS_KEY][business_id] = int(rating)
    cleaned = " ".join(str(note or "").split())[:500]
    if cleaned:
        st.session_state[NOTES_KEY][business_id] = cleaned
    else:
        st.session_state[NOTES_KEY].pop(business_id, None)


def remove_rating(business_id: str) -> None:
    st.session_state[RATINGS_KEY].pop(business_id, None)
    st.session_state[NOTES_KEY].pop(business_id, None)


def clear_ratings() -> None:
    st.session_state[RATINGS_KEY] = {}
    st.session_state[NOTES_KEY] = {}
    st.session_state[SURVEY_KEY] = False


# How many ratings before the recommendations are worth showing off. Below
# this the picks are technically personalised but visibly thin, so the
# interface says so rather than pretending otherwise.
RATINGS_FOR_GOOD_RESULTS = 3
