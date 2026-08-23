"""
core/content_based.py

Content-based filtering: recommend restaurants that *resemble* the ones a user
already liked, judged by the restaurant's own attributes rather than by what
anyone else thought of it.

THE FEATURES
------------
Each restaurant becomes a vector of:

  * its category tags, weighted by TF-IDF
  * its price range, scaled to 0-1

TF-IDF rather than a plain tag count matters here. Every row in this dataset
carries the tag "Restaurants", so a raw count would treat that tag as
informative when it distinguishes nothing. TF-IDF automatically discounts
terms that appear everywhere and promotes rare, discriminating ones -- so
"Ethiopian" ends up carrying far more weight than "Restaurants" without
anyone hand-writing a stop-word list.

TWO JOBS, TWO METHODS
---------------------
This model answers the project's two questions differently, on purpose:

  score_all()  builds one "taste vector" for the user (the average feature
               vector of everything they rated highly) and ranks every
               restaurant by cosine similarity to it. Good at ranking.

  predict()    estimates an actual star rating as a similarity-weighted
               average of that user's own past ratings, using content
               similarity to decide which of their ratings are relevant.

The previous version used a single method for both, converting a cosine
similarity into a rating with `1 + similarity * 4`. That mapping is the reason
its RMSE was poor: cosine similarities on sparse tag vectors cluster in the
0.2-0.6 range, so every prediction landed between 1.8 and 3.4 while real
ratings average about 3.8. The model was not bad at judging restaurants -- it
was being asked to read its output on a scale it was never on. Predicting from
the user's own ratings fixes that at the root.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

from core.base import Recommender, clip_rating
from core.data import Dataset

# A rating at or above this counts as "the user liked it" when building the
# taste vector. 4 of 5 is the conventional cut-off for implicit approval and
# is the same threshold the evaluation uses to define a relevant item.
LIKED_THRESHOLD = 4.0

# Ratings' worth of "the average diner" mixed into each user's own mean before
# it is used as a prediction prior. See the same constant in collaborative.py
# for the failure it prevents.
USER_MEAN_PRIOR_WEIGHT = 5.0


class ContentBasedRecommender(Recommender):
    """Cosine similarity over TF-IDF category vectors plus price."""

    name = "Content-based"
    description = (
        "Compares restaurants by their own attributes — category tags weighted "
        "with TF-IDF, plus price range — and recommends the ones most similar "
        "to what you already rated highly."
    )

    def __init__(self, liked_threshold: float = LIKED_THRESHOLD) -> None:
        super().__init__()
        self.liked_threshold = liked_threshold
        self.features_: np.ndarray | None = None       # items x features
        self.item_similarity_: pd.DataFrame | None = None
        self.predictions_: pd.DataFrame | None = None  # users x items
        self.taste_scores_: pd.DataFrame | None = None  # users x items
        self.business_ids_: list[str] = []
        self.vocabulary_size_: int = 0

    # -- feature construction ----------------------------------------------

    @staticmethod
    def _normalise_categories(value) -> str:
        """Turn "Chinese, Restaurants, Noodles" into "chinese restaurants noodles".

        Multi-word tags are joined with an underscore so the vectoriser treats
        "Ice Cream" as one term rather than as "ice" plus "cream" -- otherwise
        an ice cream parlour and a creamery look artificially alike.
        """
        if not isinstance(value, str) or not value.strip():
            return "uncategorised"
        tags = [t.strip().lower().replace(" ", "_") for t in value.split(",") if t.strip()]
        return " ".join(tags) if tags else "uncategorised"

    def _build_features(self) -> None:
        documents = self._businesses["categories"].apply(self._normalise_categories)

        self.vectoriser_ = TfidfVectorizer()
        category_matrix = self.vectoriser_.fit_transform(documents).toarray()
        self.vocabulary_size_ = category_matrix.shape[1]

        # Price sits on a 1-4 scale; the TF-IDF columns are already 0-1. Scaling
        # price to the same range stops it from dominating the distance purely
        # because its raw numbers are larger.
        price = self._businesses[["price_range"]].astype(float)
        price = price.fillna(price.median())
        price_scaled = MinMaxScaler().fit_transform(price)

        self.features_ = np.hstack([category_matrix, price_scaled])
        self.business_ids_ = self._businesses["business_id"].tolist()

    # -- fitting -----------------------------------------------------------

    def fit(self, data: Dataset, ratings: pd.DataFrame | None = None) -> "ContentBasedRecommender":
        self._businesses = data.businesses.reset_index(drop=True)
        self._ratings = data.reviews if ratings is None else ratings

        self._build_features()

        # Item-to-item similarity from content alone. 817 x 817 is small enough
        # to hold in memory, and computing it once here makes every later
        # lookup a constant-time array read.
        similarity = cosine_similarity(self.features_)
        self.item_similarity_ = pd.DataFrame(
            similarity, index=self.business_ids_, columns=self.business_ids_
        )

        self._build_taste_scores()
        self._build_predictions(similarity)

        self._fitted = True
        return self

    def _build_taste_scores(self) -> None:
        """One taste vector per user, scored against every restaurant at once.

        Scoring users one at a time in a Python loop is the obvious way to
        write this and is far too slow at 2,112 users. Stacking every taste
        vector into a matrix turns the whole job into a single BLAS call.
        """
        liked = self._ratings[self._ratings["rating"] >= self.liked_threshold]
        index_of = {bid: i for i, bid in enumerate(self.business_ids_)}

        vectors, users = [], []
        for user_id, group in liked.groupby("user_id"):
            rows = [index_of[b] for b in group["business_id"] if b in index_of]
            if rows:
                vectors.append(self.features_[rows].mean(axis=0))
                users.append(user_id)

        if not vectors:
            self.taste_scores_ = pd.DataFrame(columns=self.business_ids_)
            return

        scores = cosine_similarity(np.vstack(vectors), self.features_)
        self.taste_scores_ = pd.DataFrame(scores, index=users, columns=self.business_ids_)

    def _build_predictions(self, similarity: np.ndarray) -> None:
        """Predict ratings as a content-similarity-weighted average of the
        user's own ratings -- content-based k-nearest-neighbours.

        Expressed as two matrix products rather than a per-user loop:

            numerator   = R    @ S   -- each user's ratings, spread over items
                                        in proportion to content similarity
            denominator = mask @ S   -- how much similarity backed each of those

        Dividing gives a weighted average. Thin evidence is then shrunk toward
        the user's own mean rating, for the same reason popularity.py shrinks
        toward the global mean: a prediction resting on almost no similarity
        should not be stated with full confidence.
        """
        matrix = self._ratings.pivot_table(index="user_id", columns="business_id", values="rating")
        # Restaurants present in the catalogue but unrated in this split still
        # need columns, or the matrix product below has the wrong shape.
        matrix = matrix.reindex(columns=self.business_ids_)

        R = matrix.fillna(0.0).to_numpy()
        mask = matrix.notna().to_numpy().astype(float)

        numerator = R @ similarity
        denominator = mask @ similarity

        with np.errstate(invalid="ignore", divide="ignore"):
            raw = np.divide(numerator, denominator,
                            out=np.zeros_like(numerator), where=denominator > 1e-9)

        # The user's own average, shrunk toward the global average by how many
        # ratings support it -- the same correction collaborative.py applies,
        # and for the same reason: a "personal average" computed from one
        # rating is not yet personal.
        global_mean = float(self._ratings["rating"].mean())
        rated_counts = mask.sum(axis=1)
        user_means = (R.sum(axis=1) + USER_MEAN_PRIOR_WEIGHT * global_mean) / \
                     (rated_counts + USER_MEAN_PRIOR_WEIGHT)

        # k = the typical amount of similarity backing a prediction. A cell with
        # that much support is weighted half on its own estimate, half on the
        # user's mean.
        k = float(np.median(denominator[denominator > 1e-9])) if (denominator > 1e-9).any() else 1.0
        weight = denominator / (denominator + k)
        shrunk = weight * raw + (1.0 - weight) * user_means[:, None]

        self.predictions_ = pd.DataFrame(
            np.clip(shrunk, 1.0, 5.0), index=matrix.index, columns=self.business_ids_
        )

    # -- the interface -----------------------------------------------------

    def predict(self, user_id: str, business_id: str) -> float:
        self._require_fitted()
        if self.predictions_ is None:
            return float("nan")
        if user_id not in self.predictions_.index or business_id not in self.predictions_.columns:
            return float("nan")
        return clip_rating(float(self.predictions_.at[user_id, business_id]))

    def score_all(self, user_id: str) -> pd.Series:
        self._require_fitted()
        if self.taste_scores_ is None or user_id not in self.taste_scores_.index:
            # No liked restaurants yet, so there is no taste to match against.
            return self._cold_start_scores()
        return self.taste_scores_.loc[user_id].copy()

    def score_all_from_ratings(self, ratings: dict[str, float]) -> pd.Series:
        """Rank for a live visitor by building their taste vector on the spot.

        No retraining is involved: the feature matrix is already built, so this
        is one average and one cosine call.
        """
        self._require_fitted()
        index_of = {bid: i for i, bid in enumerate(self.business_ids_)}
        liked_rows = [index_of[b] for b, r in ratings.items()
                      if r >= self.liked_threshold and b in index_of]

        if not liked_rows:
            # They rated things, but liked none of them. Ranking by "most like
            # the restaurants you disliked" would be actively unhelpful.
            return self._cold_start_scores()

        taste = self.features_[liked_rows].mean(axis=0, keepdims=True)
        similarity = cosine_similarity(taste, self.features_)[0]
        return pd.Series(similarity, index=self.business_ids_)

    def similar_to(self, business_id: str, top_n: int = 4) -> pd.DataFrame:
        """Restaurants most like a given one. Powers "you might also like".

        This is the one place content-based filtering is used without any user
        involved at all -- pure item-to-item similarity.
        """
        self._require_fitted()
        if business_id not in self.item_similarity_.index:
            return pd.DataFrame()
        scores = self.item_similarity_.loc[business_id].drop(labels=[business_id], errors="ignore")
        return self._to_results(scores, top_n)
