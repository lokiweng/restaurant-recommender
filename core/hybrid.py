"""
core/hybrid.py

A weighted hybrid: combine the content-based and collaborative models so that
each covers the other's blind spot.

WHY COMBINE THEM AT ALL
-----------------------
The two models fail in opposite situations, which is exactly the condition
under which blending helps:

  * Collaborative filtering cannot say anything about a restaurant nobody has
    rated yet — the cold-start problem. Content-based filtering can, because a
    brand-new Thai restaurant still has category tags.
  * Content-based filtering can only ever recommend more of what you already
    like; it has no way to discover that people with your taste also love a
    place with completely unrelated tags. Collaborative filtering finds those.

Burke's 2002 survey of hybrid recommenders calls this the *weighted* strategy:
the simplest of the seven he catalogues, and the easiest to explain and defend.

THE BLEND
---------
    hybrid = α × collaborative + (1 − α) × content-based

α = 0 is pure content-based, α = 1 is pure collaborative, α = 0.5 weighs them
equally. Because base.py requires both models to predict on the same 1-5 scale,
this is a straight weighted average of two comparable numbers.

That was not true of the previous version. Content-based returned a 0-1 cosine
similarity and collaborative returned a 1-5 rating, so the old hybrid had to
rescale one of them mid-calculation before averaging — a conversion buried in
the blending code, where nobody would think to look for it. Fixing the scales
at the interface makes this file almost trivial, which is the point.

HANDLING A MISSING OPINION
--------------------------
Either model can legitimately return NaN. If both do, the hybrid returns NaN —
no opinion, honestly reported. If exactly one does, the hybrid returns the
other one's answer at full weight rather than discarding a usable prediction.
"""

import numpy as np
import pandas as pd

from core.base import Recommender, clip_rating
from core.collaborative import CollaborativeRecommender
from core.content_based import ContentBasedRecommender
from core.data import Dataset

DEFAULT_ALPHA = 0.5


class HybridRecommender(Recommender):
    """Weighted combination of the content-based and collaborative models."""

    name = "Hybrid"
    description = (
        "Blends the two approaches: α of the collaborative prediction plus "
        "(1 − α) of the content-based one. Covers new restaurants that "
        "collaborative filtering cannot see, and taste connections that "
        "content-based filtering cannot make."
    )

    def __init__(self, alpha: float = DEFAULT_ALPHA,
                 content: ContentBasedRecommender | None = None,
                 collaborative: CollaborativeRecommender | None = None) -> None:
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
        self.alpha = alpha
        # Accepting already-fitted sub-models is what lets the app fit each one
        # once and share them, rather than paying to fit content-based and
        # collaborative twice over.
        self.content = content or ContentBasedRecommender()
        self.collaborative = collaborative or CollaborativeRecommender()
        self._owns_submodels = content is None or collaborative is None

    def fit(self, data: Dataset, ratings: pd.DataFrame | None = None) -> "HybridRecommender":
        self._businesses = data.businesses.reset_index(drop=True)
        self._ratings = data.reviews if ratings is None else ratings

        # Only fit sub-models that were not handed in already fitted.
        if not getattr(self.content, "_fitted", False):
            self.content.fit(data, ratings)
        if not getattr(self.collaborative, "_fitted", False):
            self.collaborative.fit(data, ratings)

        self._fitted = True
        return self

    @staticmethod
    def _blend(collab_value: float, content_value: float, alpha: float) -> float:
        """Weighted average of two predictions, tolerating a missing one."""
        collab_ok = collab_value is not None and not np.isnan(collab_value)
        content_ok = content_value is not None and not np.isnan(content_value)

        if collab_ok and content_ok:
            return alpha * collab_value + (1.0 - alpha) * content_value
        if collab_ok:
            return collab_value          # content had nothing to say
        if content_ok:
            return content_value         # collaborative had nothing to say
        return float("nan")              # neither did — say so

    def predict(self, user_id: str, business_id: str) -> float:
        self._require_fitted()
        blended = self._blend(
            self.collaborative.predict(user_id, business_id),
            self.content.predict(user_id, business_id),
            self.alpha,
        )
        return clip_rating(blended)

    def _blend_series(self, collab: pd.Series, content: pd.Series) -> pd.Series:
        """Blend two full score vectors, aligning them on business_id first.

        The two models can return scores on different ranges — collaborative
        predicts 1-5 while content-based ranking is a 0-1 cosine — so for
        RANKING (unlike predict()) each vector is min-max normalised before
        blending. Without that, whichever series happens to span a wider
        numeric range would dominate the ordering for no principled reason.
        """
        combined = pd.DataFrame({"collab": collab, "content": content})

        def normalise(series: pd.Series) -> pd.Series:
            valid = series.dropna()
            if valid.empty:
                return series
            low, high = float(valid.min()), float(valid.max())
            if high - low < 1e-12:
                return series * 0.0 + 0.5   # perfectly flat: no ranking signal
            return (series - low) / (high - low)

        collab_n = normalise(combined["collab"])
        content_n = normalise(combined["content"])

        blended = self.alpha * collab_n + (1.0 - self.alpha) * content_n
        # Where exactly one model has an opinion, use it rather than dropping
        # the restaurant out of the ranking entirely.
        blended = blended.fillna(collab_n).fillna(content_n)
        return blended

    def score_all(self, user_id: str) -> pd.Series:
        self._require_fitted()
        return self._blend_series(
            self.collaborative.score_all(user_id),
            self.content.score_all(user_id),
        )

    def score_all_from_ratings(self, ratings: dict[str, float]) -> pd.Series:
        self._require_fitted()
        return self._blend_series(
            self.collaborative.score_all_from_ratings(ratings),
            self.content.score_all_from_ratings(ratings),
        )
