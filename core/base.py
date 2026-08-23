"""
core/base.py

The contract every recommender in this project implements.

WHY THIS FILE EXISTS
--------------------
The previous version of this system had no shared interface. Each model
exposed slightly different method names, returned scores on different scales,
and the evaluation code had to carry a `model_type` string and branch on it:

    if model_type == "content":
        pred = 1 + model.predict_score(u, b) * 4   # rescale 0-1 to 1-5
    elif model_type == "collab":
        pred = model.predict_score(u, b)           # already 1-5

That is a design smell. The evaluation harness had to know the internals of
every model, which meant adding a fourth model required editing the evaluator,
and a scale conversion lived in the wrong file entirely.

Here, every model promises the same two things:

    predict(user_id, business_id) -> a rating on the SAME 1-5 scale
    score_all(user_id)            -> a score for every restaurant, for ranking

Everything else -- turning scores into a sorted top-N list, excluding places
the user has already rated, attaching restaurant names -- is written once in
this base class and inherited. Adding a new model means implementing two
methods and nothing else.

PREDICTING VERSUS RANKING
-------------------------
These are genuinely different tasks and the assignment measures both:

  * predict() answers "what would this user rate this restaurant?" -- graded
    by RMSE and MSE, which compare against held-out real ratings.
  * score_all() answers "in what order should these restaurants be shown?" --
    graded by Precision/Recall/F1@K.

A model can be good at one and poor at the other. Keeping them as separate
methods means neither is quietly forced to stand in for the other.
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from core.data import Dataset

# Ratings in this dataset are Yelp stars: whole numbers from 1 to 5.
RATING_MIN, RATING_MAX = 1.0, 5.0

# The columns every recommend() call returns, in this order. Fixed so the UI
# can render any model's output with the same component.
RESULT_COLUMNS = ["business_id", "name", "primary_category", "price_range",
                  "avg_rating", "review_count", "score"]

# How many extra candidates to pull before collapsing same-named branches, so
# that removing duplicates still leaves a full-length list.
NAME_BUFFER_FACTOR = 3


class NotFittedError(RuntimeError):
    """Raised when a model is asked for predictions before fit() was called."""


class Recommender(ABC):
    """Base class for all four recommenders.

    Subclasses implement fit(), predict() and score_all(). The ranking helpers
    below are inherited and should not be overridden.
    """

    #: Human-readable name, used in the UI and in evaluation tables.
    name: str = "Recommender"

    #: One sentence the About screen shows to explain this model.
    description: str = ""

    def __init__(self) -> None:
        self._fitted = False
        self._businesses: pd.DataFrame | None = None
        self._ratings: pd.DataFrame | None = None

    # -- to be implemented by each model -----------------------------------

    @abstractmethod
    def fit(self, data: Dataset, ratings: pd.DataFrame | None = None) -> "Recommender":
        """Learn from the data.

        `ratings` exists so the evaluation harness can fit on a training split
        only. When it is None the model uses every rating in the dataset.
        Returns self, so `model = Model().fit(data)` reads naturally.
        """

    @abstractmethod
    def predict(self, user_id: str, business_id: str) -> float:
        """Predicted rating on a 1-5 scale.

        Returns NaN when the model genuinely has no basis for a prediction --
        an unknown user, or a restaurant with no usable signal. NaN is the
        honest answer; a made-up 3.0 would quietly flatter the RMSE.
        """

    @abstractmethod
    def score_all(self, user_id: str) -> pd.Series:
        """A ranking score for every restaurant, indexed by business_id.

        Higher is better. The scale does not have to match predict() -- these
        values are only ever compared against each other, never against a real
        rating.
        """

    # -- optional: live "guest" scoring, overridden where it makes sense ----

    def score_all_from_ratings(self, ratings: dict[str, float]) -> pd.Series:
        """Score every restaurant for someone who is not in the dataset.

        `ratings` maps business_id to a 1-5 rating collected in this browser
        session. The default falls back to the model's non-personalised
        ordering; models that can genuinely personalise from a handful of live
        ratings override this.
        """
        return self._cold_start_scores()

    # -- shared plumbing, written once -------------------------------------

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise NotFittedError(f"{self.name} must be fitted before use — call .fit(data) first.")

    def _cold_start_scores(self) -> pd.Series:
        """The fallback ordering when there is nothing personal to go on.

        Rating and review count, which is the same thing every "Top Rated"
        list does. Deliberately not disguised as a recommendation.
        """
        self._require_fitted()
        ordered = self._businesses.sort_values(["avg_rating", "review_count"], ascending=False)
        # Descending integers, so the resulting Series sorts the same way.
        return pd.Series(
            np.arange(len(ordered), 0, -1, dtype=float),
            index=ordered["business_id"].to_numpy(),
        )

    def _rated_by(self, user_id: str) -> set:
        """Restaurants this user already rated in the fitted data."""
        if self._ratings is None:
            return set()
        return set(self._ratings.loc[self._ratings["user_id"] == user_id, "business_id"])

    def _to_results(self, scores: pd.Series, top_n: int, exclude: set | None = None) -> pd.DataFrame:
        """Turn a score-per-restaurant Series into a display-ready table.

        Shared by every model so that all four return identically shaped
        output and the UI needs exactly one rendering path.
        """
        self._require_fitted()

        if scores is None or scores.empty:
            return pd.DataFrame(columns=RESULT_COLUMNS)

        if exclude:
            scores = scores[~scores.index.isin(exclude)]

        # NaN means "no opinion" -- such restaurants must not be ranked at all,
        # rather than being sorted to the bottom as if they scored badly.
        scores = scores.dropna().sort_values(ascending=False)
        if scores.empty:
            return pd.DataFrame(columns=RESULT_COLUMNS)

        # Take a buffer rather than exactly top_n, because collapsing branches
        # below will remove some rows and the list still has to come out full.
        scores = scores.head(max(top_n * NAME_BUFFER_FACTOR, top_n))

        ranked = pd.DataFrame({"business_id": scores.index, "score": scores.to_numpy()})
        available = [c for c in RESULT_COLUMNS if c in self._businesses.columns and c != "score"]
        ranked = ranked.merge(self._businesses[available], on="business_id", how="left")

        for column in RESULT_COLUMNS:
            if column not in ranked.columns:
                ranked[column] = np.nan

        ranked = ranked.sort_values("score", ascending=False)

        # Collapse chains and branches to one entry each.
        #
        # 70 rows in this dataset share a name with another row -- five separate
        # Applebee's, two Al's Delis, and so on. They are genuinely different
        # businesses with different ids and different ratings, so the data is
        # correct; but a top-10 list containing the same name three times reads
        # as a bug, and with no address field in the dataset there is no way for
        # a reader to tell the branches apart anyway.
        #
        # The highest-scoring branch is kept, since that is the one the model
        # actually recommends. This applies to RECOMMENDATIONS only -- browsing
        # the catalogue still shows every branch, which is correct there.
        if "name" in ranked.columns:
            ranked = ranked.drop_duplicates(subset="name", keep="first")

        return ranked.head(top_n)[RESULT_COLUMNS].reset_index(drop=True)

    def recommend(self, user_id: str, top_n: int = 10, exclude_rated: bool = True) -> pd.DataFrame:
        """Top-N restaurants for a user already present in the dataset."""
        self._require_fitted()
        exclude = self._rated_by(user_id) if exclude_rated else set()
        return self._to_results(self.score_all(user_id), top_n, exclude)

    def recommend_from_ratings(self, ratings: dict[str, float], top_n: int = 10) -> pd.DataFrame:
        """Top-N restaurants for a live session, from ratings given just now.

        Anything the visitor rated is excluded: recommending a restaurant back
        to the person who just told you what they thought of it is the most
        obvious failure a recommender can make.

        Session ratings come from browser state rather than from the validated
        CSVs, so they are cleaned here -- see clean_session_ratings() for why
        this guard is at this particular layer.
        """
        self._require_fitted()
        known_ids = set(self._businesses["business_id"])
        clean = clean_session_ratings(ratings, known_ids)
        if not clean:
            return self._to_results(self._cold_start_scores(), top_n)
        return self._to_results(self.score_all_from_ratings(clean), top_n, set(clean))


def clean_session_ratings(ratings: dict, known_ids: set) -> dict[str, float]:
    """Drop anything from a live session's ratings that is not a usable star.

    WHY THIS EXISTS, AND WHY HERE
    -----------------------------
    core/validation.py guards the CSVs, which are the data the models are
    fitted on. Session ratings are a different input entirely: they arrive from
    st.session_state, which is browser state, and they reach the models without
    passing through any of that. The two need separate guards because they fail
    in different ways.

    Before this function existed the four models disagreed about what to do
    with a malformed rating, which is the worst possible outcome: the
    collaborative model quietly returned an empty list, the content-based model
    raised TypeError, and the page showed either "no recommendations" with no
    reason or a red Streamlit traceback. Same bad input, three different
    behaviours, none of them explained.

    Cleaning centrally means all four models see the same ratings and behave
    identically. Anything unusable is dropped rather than repaired: a rating of
    0 or 6 stars is not a near-miss to be clipped, it is evidence that
    something upstream is wrong, and inventing a value would hide that. If
    nothing survives, the caller falls back to the cold-start ordering, which
    is exactly what a visitor who has rated nothing already sees.
    """
    clean: dict[str, float] = {}
    for business_id, value in ratings.items():
        if business_id not in known_ids:
            continue                      # stale id from an older session
        try:
            rating = float(value)
        except (TypeError, ValueError):
            continue                      # None, a string, an object
        if not np.isfinite(rating):
            continue                      # NaN or infinity
        if not (RATING_MIN <= rating <= RATING_MAX):
            continue                      # outside the scale the UI can produce
        clean[business_id] = rating
    return clean


def clip_rating(value: float) -> float:
    """Hold a predicted rating inside the 1-5 range the scale actually allows.

    Weighted averages can drift slightly outside the bounds through rounding,
    and a predicted 5.02 stars is not wrong so much as meaningless.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("nan")
    return float(min(RATING_MAX, max(RATING_MIN, value)))
