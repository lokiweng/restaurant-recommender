"""
core/collaborative.py

Item-based collaborative filtering: recommend restaurants that the same people
tend to rate the same way, ignoring what the restaurants actually are.

THE IDEA
--------
Build a matrix of who rated what. Two restaurants are "similar" if the same
diners gave them similar ratings -- regardless of cuisine, price or anything
else on the menu. A steakhouse and a wine bar can come out similar if the same
people love both, which is precisely the kind of connection content-based
filtering can never make, because nothing in their tags says so.

To predict what a user would give an unseen restaurant, take a weighted
average of that user's existing ratings, weighting each by how similar the
rated restaurant is to the unseen one.

WHY ITEM-BASED RATHER THAN USER-BASED
-------------------------------------
The alternative is to find similar *users* and average their ratings. Both are
valid; item-based is chosen here for two concrete reasons:

  1. Stability. Restaurant-to-restaurant similarity barely moves — the
     catalogue changes far more slowly than people's rating histories — so the
     similarity matrix can be computed once and reused. A user-based matrix
     goes stale every time anyone rates anything.
  2. Size. There are 817 restaurants and 2,112 users here, so the item-item
     matrix is roughly a sixth the size of the user-user one. That gap widens
     on any real deployment, where users vastly outnumber items.

THE SPARSITY PROBLEM, AND THE SHRINKAGE THAT ANSWERS IT
-------------------------------------------------------
This dataset is about 1.5% dense: the average restaurant pair shares very few
raters. A naive weighted average divides by whatever similarity happens to
exist, so a restaurant sharing 0.01 of similarity with one five-star rating
comes out at a confident 5.00 — a maximal score resting on almost no evidence.

The same Bayesian shrinkage used in popularity.py fixes it, with summed
similarity standing in for review count:

    prediction = (d / (d + k)) * weighted_average  +  (k / (d + k)) * prior

where d is the total similarity supporting the prediction and k is the typical
value of d. Well-supported predictions are left alone; thin ones collapse
toward the prior instead of masquerading as certainty.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from core.base import Recommender, clip_rating
from core.data import Dataset

# How many ratings' worth of "the average diner" to mix into a user's own mean.
# A user with 5 ratings is judged half on their own average and half on the
# global one; a user with 50 is judged almost entirely on their own.
#
# This matters more than it looks. Without it, someone who has rated exactly
# one restaurant has a "personal mean" equal to that single rating, so shrinking
# a prediction toward their mean shrinks it toward the very number it already
# is -- the shrinkage cancels out, every restaurant scores identically, and the
# recommendation order becomes arbitrary. Mixing in the global mean gives thin
# and thick evidence different answers again, which is the entire point.
USER_MEAN_PRIOR_WEIGHT = 5.0


class CollaborativeRecommender(Recommender):
    """Item-based collaborative filtering with shrinkage toward the user mean."""

    name = "Collaborative filtering"
    description = (
        "Finds restaurants that the same diners tend to rate alike, then "
        "predicts your rating from how you scored the ones most similar to it. "
        "Ignores cuisine and price entirely — it only looks at behaviour."
    )

    def __init__(self) -> None:
        super().__init__()
        self.matrix_: pd.DataFrame | None = None          # users x items
        self.item_similarity_: pd.DataFrame | None = None  # items x items
        self.predictions_: pd.DataFrame | None = None      # users x items
        self.global_mean_: float = np.nan
        self.shrinkage_k_: float = np.nan
        self.density_: float = np.nan

    def fit(self, data: Dataset, ratings: pd.DataFrame | None = None) -> "CollaborativeRecommender":
        self._businesses = data.businesses.reset_index(drop=True)
        self._ratings = data.reviews if ratings is None else ratings

        business_ids = self._businesses["business_id"].tolist()

        # Rows are users, columns are restaurants, values are ratings, NaN where
        # unrated. Reindexed onto the full catalogue so that restaurants nobody
        # rated in this split still get a column.
        self.matrix_ = (
            self._ratings
            .pivot_table(index="user_id", columns="business_id", values="rating")
            .reindex(columns=business_ids)
        )

        R = self.matrix_.fillna(0.0).to_numpy()
        mask = self.matrix_.notna().to_numpy().astype(float)

        self.density_ = float(mask.mean())
        self.global_mean_ = float(self._ratings["rating"].mean())

        # Cosine similarity between COLUMNS. Unrated cells are treated as 0,
        # which is the standard simplification: it reads "no opinion" as
        # "no contribution" rather than as a bad rating.
        similarity = cosine_similarity(R.T)
        np.fill_diagonal(similarity, 0.0)  # a restaurant must not predict itself
        self.item_similarity_ = pd.DataFrame(similarity, index=business_ids, columns=business_ids)

        self._build_predictions(R, mask, similarity)
        self._fitted = True
        return self

    def _build_predictions(self, R: np.ndarray, mask: np.ndarray, similarity: np.ndarray) -> None:
        """Every user-restaurant prediction, as two matrix products.

        Written this way rather than as a loop over (user, restaurant) pairs
        because the loop version does not finish in reasonable time at this
        size: 2,112 x 817 is 1.7 million cells, and each one would need its own
        pandas lookups. As matrix algebra it is two multiplications.
        """
        numerator = R @ similarity
        denominator = mask @ similarity

        with np.errstate(invalid="ignore", divide="ignore"):
            raw = np.divide(numerator, denominator,
                            out=np.zeros_like(numerator), where=denominator > 1e-9)

        # Prior: the user's own average, itself shrunk toward the global average
        # in proportion to how many ratings back it. Someone who rates everything
        # 5 should not be dragged all the way to 3.8 -- but someone who has rated
        # one thing has not yet demonstrated a personal average at all.
        rated_counts = mask.sum(axis=1)
        user_means = (R.sum(axis=1) + USER_MEAN_PRIOR_WEIGHT * self.global_mean_) / \
                     (rated_counts + USER_MEAN_PRIOR_WEIGHT)

        supported = denominator[denominator > 1e-9]
        self.shrinkage_k_ = float(np.median(supported)) if supported.size else 1.0

        weight = denominator / (denominator + self.shrinkage_k_)
        shrunk = weight * raw + (1.0 - weight) * user_means[:, None]

        self.predictions_ = pd.DataFrame(
            np.clip(shrunk, 1.0, 5.0), index=self.matrix_.index, columns=self.matrix_.columns
        )

    def predict(self, user_id: str, business_id: str) -> float:
        self._require_fitted()
        if user_id not in self.predictions_.index or business_id not in self.predictions_.columns:
            return float("nan")
        value = self.predictions_.at[user_id, business_id]
        return float("nan") if pd.isna(value) else clip_rating(float(value))

    def score_all(self, user_id: str) -> pd.Series:
        """Rank by predicted rating.

        Unlike content-based filtering, this model's ranking signal and its
        rating prediction are the same quantity, so there is nothing to
        separate here.
        """
        self._require_fitted()
        if user_id not in self.predictions_.index:
            return self._cold_start_scores()
        return self.predictions_.loc[user_id].copy()

    def score_all_from_ratings(self, ratings: dict[str, float]) -> pd.Series:
        """Score for a live visitor straight from the similarity matrix.

        The visitor never enters the fitted rating matrix and nothing is
        retrained: their ratings are simply run through the same weighted
        average, with the same shrinkage applied.
        """
        self._require_fitted()

        known = {b: r for b, r in ratings.items() if b in self.item_similarity_.index}
        if not known:
            return self._cold_start_scores()

        rated_ids = list(known)
        values = np.array([known[b] for b in rated_ids], dtype=float)
        similarity = self.item_similarity_.loc[rated_ids].to_numpy()   # rated x all

        numerator = values @ similarity
        denominator = similarity.sum(axis=0)

        with np.errstate(invalid="ignore", divide="ignore"):
            raw = np.divide(numerator, denominator,
                            out=np.zeros_like(numerator), where=denominator > 1e-9)

        # With only a handful of live ratings the evidence is thinner than in
        # the fitted case, so shrinkage matters more here, not less -- and the
        # prior must be shrunk the same way. Using the raw session mean would
        # make a one-rating session score every restaurant identically.
        prior = float(
            (values.sum() + USER_MEAN_PRIOR_WEIGHT * self.global_mean_)
            / (len(values) + USER_MEAN_PRIOR_WEIGHT)
        )
        supported = denominator[denominator > 1e-9]
        k = float(np.median(supported)) if supported.size else 1.0

        weight = denominator / (denominator + k)
        shrunk = weight * raw + (1.0 - weight) * prior

        return pd.Series(np.clip(shrunk, 1.0, 5.0), index=self.item_similarity_.columns)
