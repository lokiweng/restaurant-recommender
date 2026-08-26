"""
core/evaluation.py

Measures how good each recommender actually is.

THE ASSIGNMENT ASKS FOR THREE KINDS OF EVALUATION
-------------------------------------------------
Two of them are computed here from held-out data; the third (a user
satisfaction questionnaire) cannot be computed at all and is collected from
real people in core/satisfaction.py.

  1. RATING PREDICTION — RMSE and MSE.
     "If this user rated this restaurant, how close was our guess?"
     Lower is better. RMSE is in stars, so 1.0 means the average prediction is
     off by roughly one star; MSE is its square, included because the brief
     names both, and because squaring punishes large misses disproportionately.

  2. RANKING QUALITY — Precision@K, Recall@K, F1@K.
     "Of the ten restaurants we put in front of this user, how many did they
     actually go on to like?" Higher is better.

WHY BOTH, WHEN THEY MEASURE THE SAME MODEL
------------------------------------------
They disagree, and the disagreement is informative. A model can predict star
ratings tightly while ranking badly — if it predicts 3.9 for everything, its
RMSE looks respectable and its top-10 list is meaningless. The reverse also
happens. Reporting only one metric hides half the picture.

HOW THE DATA IS SPLIT
---------------------
Per user, not at random across the whole table. A purely random 80/20 split
would leave some users with no training ratings at all, and a model cannot be
blamed for failing a user it was never shown. Holding out 20% of *each* user's
ratings guarantees every evaluated user has a history to learn from — which is
what makes the resulting numbers a fair test rather than a mixture of a fair
test and a cold-start test.
"""

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from core.base import Recommender
from core.data import Dataset

DEFAULT_K = 10
RELEVANCE_THRESHOLD = 4.0   # a held-out rating this high means the user liked it
DEFAULT_TEST_SIZE = 0.2
DEFAULT_SEED = 42


@dataclass
class EvaluationResult:
    """Every metric for one model, in one object."""

    model: str
    rmse: float | None
    mse: float | None
    n_predictions: int
    precision_at_k: float | None
    recall_at_k: float | None
    f1_at_k: float | None
    hit_rate_at_k: float | None
    ndcg_at_k: float | None
    n_users_evaluated: int
    k: int
    coverage: float | None
    personalisation: float | None
    n_personalised_users: int | None
    n_fallback_users: int | None

    def as_row(self) -> dict:
        """Flatten for display in a table."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def train_test_split_per_user(
    reviews: pd.DataFrame,
    test_size: float = DEFAULT_TEST_SIZE,
    seed: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out a fraction of each user's ratings.

    Users with only one rating contribute it entirely to training: holding out
    their single rating would leave the model nothing to personalise from, so
    the test case would measure cold-start behaviour rather than accuracy.

    The seed is fixed so the same split — and therefore the same reported
    numbers — comes out every time the evaluation is run. Results that shift
    between runs cannot be quoted in a report.
    """
    if reviews.empty:
        raise ValueError("Cannot split an empty ratings table.")
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")

    rng = np.random.default_rng(seed)
    test_indices: list = []

    for _, group in reviews.groupby("user_id"):
        if len(group) < 2:
            continue   # keep this user's only rating in training
        n_test = max(1, int(round(len(group) * test_size)))
        n_test = min(n_test, len(group) - 1)   # always leave at least one to train on
        chosen = rng.choice(group.index.to_numpy(), size=n_test, replace=False)
        test_indices.extend(chosen.tolist())

    test = reviews.loc[test_indices]
    train = reviews.drop(index=test_indices)
    return train, test


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def rmse(actual, predicted) -> float:
    """Root mean squared error, in stars."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mse(actual, predicted) -> float:
    """Mean squared error — RMSE before the square root."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.mean((actual - predicted) ** 2))


def evaluate_rating_prediction(model: Recommender, test: pd.DataFrame) -> tuple[float | None, float | None, int]:
    """Compare predicted ratings against held-out real ones.

    Predictions that come back NaN are skipped rather than substituted. A model
    saying "I don't know" is different from a model guessing the average, and
    quietly filling in the mean would flatter every metric here.
    """
    actual, predicted = [], []

    for row in test.itertuples(index=False):
        estimate = model.predict(row.user_id, row.business_id)
        if estimate is None or np.isnan(estimate):
            continue
        predicted.append(estimate)
        actual.append(float(row.rating))

    if not predicted:
        return None, None, 0

    return rmse(actual, predicted), mse(actual, predicted), len(predicted)


def ndcg_at_k(ranked_ids: list, relevant: set, k: int) -> float:
    """Normalised discounted cumulative gain over a binary-relevance ranking.

    Precision@K treats a hit in position one and a hit in position ten as
    equally good. They are not: a user reads a recommendation list from the
    top, so where a hit lands matters. NDCG applies a logarithmic discount by
    rank and then divides by the best score the list could possibly have
    achieved given how many relevant items exist, which keeps it comparable
    across users who have different numbers of relevant items held out.

    Binary relevance is used because the "relevant" decision is already a
    threshold on the held-out rating, so there are no graded gains to weight.
    """
    gains = [1.0 / np.log2(rank + 1)
             for rank, item in enumerate(ranked_ids[:k], start=1)
             if item in relevant]
    dcg = float(sum(gains))

    # The ideal list puts every relevant item it can at the top.
    ideal_hits = min(k, len(relevant))
    idcg = float(sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1)))

    return dcg / idcg if idcg > 0 else 0.0


def personalisation_index(lists: list[set], seed: int = DEFAULT_SEED,
                          max_pairs: int = 5000) -> float | None:
    """How different users' recommendation lists are from one another.

    One minus the mean Jaccard overlap between pairs of users' top-K lists. A
    model handing everybody the same ten restaurants scores 0; a model whose
    users share nothing scores 1.

    This exists because catalogue coverage cannot do the job it is often asked
    to do. Coverage counts how much of the catalogue is in circulation across
    all users, so a model could rotate through the catalogue while still giving
    any two individuals identical lists. Overlap between lists measures
    personalisation directly, and is the metric that actually distinguishes a
    personalised model from a non-personalised one.

    Pairs are sampled rather than enumerated: 1,700 users is 1.4 million pairs
    per model, and a seeded sample of 5,000 gives a stable estimate for a small
    fraction of the work.
    """
    if len(lists) < 2:
        return None

    rng = np.random.default_rng(seed)
    n = len(lists)
    total_pairs = n * (n - 1) // 2

    if total_pairs <= max_pairs:
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        left = rng.integers(0, n, size=max_pairs * 2)
        right = rng.integers(0, n, size=max_pairs * 2)
        pairs = [(int(a), int(b)) for a, b in zip(left, right) if a != b][:max_pairs]

    overlaps = []
    for i, j in pairs:
        a, b = lists[i], lists[j]
        union = len(a | b)
        overlaps.append(len(a & b) / union if union else 0.0)

    return float(1.0 - np.mean(overlaps)) if overlaps else None


def count_personalised_users(model: Recommender, train: pd.DataFrame,
                             threshold: float = RELEVANCE_THRESHOLD) -> tuple[int, int]:
    """How many users the model can personalise for, and how many it cannot.

    The content-based model builds a user's taste vector from restaurants they
    rated at or above the liked threshold. A user with no such rating in the
    training split has nothing to build from, so the model returns the
    popularity ordering instead — meaning that for those users the
    content-based row of the results table is measuring the baseline, not
    content filtering. Reporting the split is what makes that readable.

    Models without a liked-threshold fallback report every training user as
    personalised, which is accurate: collaborative filtering needs ratings, not
    *high* ratings, and the popularity baseline personalises for nobody by
    design.
    """
    all_users = train["user_id"].nunique()

    if not hasattr(model, "liked_threshold"):
        return all_users, 0

    liked = train[train["rating"] >= model.liked_threshold]
    personalised = liked["user_id"].nunique()
    return personalised, all_users - personalised


def evaluate_ranking(
    model: Recommender,
    train: pd.DataFrame,
    test: pd.DataFrame,
    k: int = DEFAULT_K,
    threshold: float = RELEVANCE_THRESHOLD,
) -> dict:
    """Precision@K, Recall@K, F1@K and catalogue coverage.

    An item is "relevant" if the user's held-out rating for it was at least
    `threshold`. For each user we ask the model for its top K and count how
    many of those relevant items it found.

    Precision = hits / K            -- of what we showed, how much was good
    Recall    = hits / relevant     -- of what was good, how much did we show
    F1        = harmonic mean       -- one number balancing the two

    Precision@10 is bounded above by relevant/10 whenever a user has fewer than
    ten relevant items held out, which is almost always. Values look small in
    absolute terms as a result; what matters is the comparison between models,
    all measured the same way.

    Coverage is reported alongside them: the share of the catalogue that
    appears in anyone's top-K at all. A model with excellent precision that
    only ever recommends the same twenty restaurants is not doing its job, and
    coverage is the metric that exposes it — though only in that direction.
    High coverage does not evidence personalisation, because a model could
    spread recommendations widely while still giving any two users the same
    list; `personalisation` measures that directly.

    Hit rate and NDCG are computed from the same pass. Hit rate asks the
    user-centric question Precision@K does not — was this user helped at all?
    — and NDCG asks where in the list the help landed.
    """
    users_in_train = set(train["user_id"])
    relevant_test = test[test["rating"] >= threshold]

    precisions, recalls, hits_flags, ndcgs = [], [], [], []
    recommended_items: set = set()
    per_user_lists: list[set] = []

    for user_id, group in relevant_test.groupby("user_id"):
        if user_id not in users_in_train:
            continue   # no training history: this would measure cold start

        relevant = set(group["business_id"])
        results = model.recommend(user_id, top_n=k)
        if results.empty:
            continue

        # Order matters for NDCG, so the ranked list is kept as a list; the set
        # is only used for the order-insensitive counts.
        ranked = results["business_id"].tolist()
        recommended = set(ranked)
        recommended_items |= recommended
        per_user_lists.append(recommended)

        hits = len(recommended & relevant)
        precisions.append(hits / k)
        recalls.append(hits / len(relevant))
        hits_flags.append(1.0 if hits > 0 else 0.0)
        ndcgs.append(ndcg_at_k(ranked, relevant, k))

    if not precisions:
        return {"precision_at_k": None, "recall_at_k": None, "f1_at_k": None,
                "hit_rate_at_k": None, "ndcg_at_k": None, "n_users_evaluated": 0,
                "coverage": None, "personalisation": None}

    precision = float(np.mean(precisions))
    recall = float(np.mean(recalls))
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    catalogue_size = len(model._businesses) if model._businesses is not None else 0
    coverage = (len(recommended_items) / catalogue_size) if catalogue_size else None

    return {
        "precision_at_k": precision,
        "recall_at_k": recall,
        "f1_at_k": f1,
        "hit_rate_at_k": float(np.mean(hits_flags)),
        "ndcg_at_k": float(np.mean(ndcgs)),
        "n_users_evaluated": len(precisions),
        "coverage": coverage,
        "personalisation": personalisation_index(per_user_lists),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def evaluate_model(
    model: Recommender,
    data: Dataset,
    train: pd.DataFrame,
    test: pd.DataFrame,
    k: int = DEFAULT_K,
) -> EvaluationResult:
    """Fit one model on the training split and measure it on the test split.

    The model is fitted here rather than by the caller so that it cannot
    accidentally be evaluated while still holding the full dataset — which
    would let it predict ratings it had already seen, and produce excellent,
    meaningless numbers.
    """
    model.fit(data, ratings=train)

    error_rmse, error_mse, n_predictions = evaluate_rating_prediction(model, test)
    ranking = evaluate_ranking(model, train, test, k)
    n_personalised, n_fallback = count_personalised_users(model, train)

    return EvaluationResult(
        model=model.name,
        rmse=error_rmse,
        mse=error_mse,
        n_predictions=n_predictions,
        precision_at_k=ranking["precision_at_k"],
        recall_at_k=ranking["recall_at_k"],
        f1_at_k=ranking["f1_at_k"],
        hit_rate_at_k=ranking["hit_rate_at_k"],
        ndcg_at_k=ranking["ndcg_at_k"],
        n_users_evaluated=ranking["n_users_evaluated"],
        k=k,
        coverage=ranking["coverage"],
        personalisation=ranking["personalisation"],
        n_personalised_users=n_personalised,
        n_fallback_users=n_fallback,
    )


def evaluate_all(
    models: list[Recommender],
    data: Dataset,
    k: int = DEFAULT_K,
    test_size: float = DEFAULT_TEST_SIZE,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Evaluate every model on one identical split and return a results table.

    Splitting once, outside the loop, is what makes the comparison fair: every
    model sees exactly the same training ratings and is tested on exactly the
    same held-out ones.
    """
    train, test = train_test_split_per_user(data.reviews, test_size=test_size, seed=seed)
    rows = [evaluate_model(model, data, train, test, k).as_row() for model in models]
    return pd.DataFrame(rows)
