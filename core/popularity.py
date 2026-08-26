"""
core/popularity.py

The non-personalised baseline: a "Top Rated" list, done properly.

WHY A BASELINE MATTERS
----------------------
Reporting that a recommender scores 0.9 on some metric means nothing on its
own. The question a marker (or a reviewer) will ask is: better than what?
A model that cannot beat "just show everyone the most popular restaurants"
has not earned the complexity it costs.

So this model implements the same interface as the three real recommenders
and is evaluated alongside them. That comparison is the point of the file.

THE FORMULA
-----------
A naive "sort by average rating" list is dominated by restaurants with a
single five-star review. The standard fix -- used by IMDb's Top 250 and known
generally as Bayesian averaging -- pulls each restaurant's average toward the
global average, with the pull weakening as it accumulates reviews:

    weighted = (v / (v + m)) * R  +  (m / (v + m)) * C

    v = this restaurant's review count
    R = this restaurant's own average rating
    C = the mean rating across all restaurants  (the prior)
    m = the review count at which a restaurant is trusted half on its own
        average and half on the prior

With v = 0 the score is exactly C (no evidence, no opinion); as v grows the
score converges on R (plenty of evidence, trust it). m is set to the 90th
percentile of review counts, so only the most-reviewed tenth of the catalogue
is judged almost entirely on its own average.

This is the same shrinkage idea that appears again in collaborative.py, where
the evidence being weighed is similarity rather than review count.
"""

import numpy as np
import pandas as pd

from core.base import Recommender, clip_rating
from core.data import Dataset


class PopularityRecommender(Recommender):
    """Ranks every restaurant identically for every user."""

    name = "Popularity baseline"
    description = (
        "Ranks restaurants by a Bayesian-weighted rating that discounts places "
        "with very few reviews. Identical for every user — the honest starting "
        "point before any personalisation exists."
    )

    def __init__(self, quantile: float = 0.90) -> None:
        super().__init__()
        # Which slice of the catalogue counts as "well reviewed enough to be
        # judged on its own average". Exposed as a parameter so the choice is
        # visible and arguable rather than buried as a magic number.
        self.quantile = quantile
        self.prior_mean_: float = np.nan   # C
        self.min_reviews_: float = np.nan  # m
        self.scores_: pd.Series | None = None

    def fit(self, data: Dataset, ratings: pd.DataFrame | None = None) -> "PopularityRecommender":
        self._businesses = data.businesses.reset_index(drop=True)
        self._ratings = data.reviews if ratings is None else ratings

        # R and v are derived from the ratings this model was actually given,
        # NOT from the precomputed avg_rating and review_count columns in
        # businesses.csv.
        #
        # This is a test-set leak, and it is worth naming precisely because the
        # leaking version looks completely reasonable. Those two columns are
        # computed from every review in the dataset. Under a train/test split
        # the model is handed only the training ratings, but reading the
        # columns would let it score restaurants using statistics that already
        # reflect the held-out ratings it is about to be graded on.
        #
        # That mattered more here than it usually would, because the headline
        # finding of this project is that this non-personalised baseline beats
        # all three personalised models. The baseline was the model reading
        # those columns, so the finding has to be re-established with the leak
        # closed before it can be trusted.
        #
        # When fitted on the full dataset — which is what the running
        # application does — this computes the same numbers as before, so the
        # interface and the app's behaviour are unchanged.
        ids = pd.Index(self._businesses["business_id"])
        observed = self._ratings.groupby("business_id")["rating"].agg(["mean", "count"])

        counts = observed["count"].reindex(ids).fillna(0.0).astype(float)
        averages = observed["mean"].reindex(ids).astype(float)

        # C: the mean of the per-restaurant averages over restaurants with at
        # least one observed rating. A restaurant with none contributes no
        # evidence and must not drag the prior around.
        self.prior_mean_ = float(averages.mean())
        averages = averages.fillna(self.prior_mean_)

        counts = counts.to_numpy()
        averages = averages.to_numpy()

        self.min_reviews_ = float(pd.Series(counts).quantile(self.quantile))

        denominator = counts + self.min_reviews_
        # Guard the degenerate case where a restaurant has no reviews and m is
        # also zero: fall back to the restaurant's own average.
        weighted = np.where(
            denominator > 0,
            (counts / denominator) * averages + (self.min_reviews_ / denominator) * self.prior_mean_,
            averages,
        )

        self.scores_ = pd.Series(weighted, index=self._businesses["business_id"].to_numpy())
        self._fitted = True
        return self

    def predict(self, user_id: str, business_id: str) -> float:
        """The weighted rating, which is already on the 1-5 scale.

        The user_id is ignored -- that is the definition of a non-personalised
        model, and it is exactly why this baseline is worth beating.
        """
        self._require_fitted()
        if business_id not in self.scores_.index:
            return float("nan")
        return clip_rating(float(self.scores_.loc[business_id]))

    def score_all(self, user_id: str) -> pd.Series:
        self._require_fitted()
        return self.scores_.copy()

    def score_all_from_ratings(self, ratings: dict[str, float]) -> pd.Series:
        # Nothing a visitor tells us changes a non-personalised ranking.
        return self.score_all(user_id="")
