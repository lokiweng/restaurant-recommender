"""
scripts/popularity_stratification.py

Tests the popularity-bias explanation instead of asserting it.

WHY THIS SCRIPT EXISTS
----------------------
The central finding of this project is that a non-personalised popularity
baseline outperforms all three personalised models on every accuracy metric.
The explanation offered for that is popularity bias: because ratings are
missing not at random, frequently-rated restaurants are over-represented in
held-out data, so recommending them scores well whoever the user is.

That is a plausible account, and plausible is all it is until it makes a
prediction that could fail. This one does. If the baseline wins *because* of
popularity concentration, its advantage should be concentrated among
frequently-rated restaurants and should shrink — or reverse — on the long
tail. If the advantage is instead uniform across the popularity range, the
explanation is wrong and something else is going on.

Run:  python scripts/popularity_stratification.py

WHAT IT MEASURES
----------------
Restaurants are ranked by training-split review count and cut into deciles;
decile 1 is the least-reviewed tenth of the catalogue, decile 10 the most.
Each held-out relevant item is assigned to the decile of its restaurant, and
recall is computed per decile per model: of the relevant items sitting in this
popularity band, what share did each model actually surface?

Recall is used rather than precision because precision is a property of a
list, not of an item, and a top-10 list spans several deciles. Recall
decomposes cleanly: every relevant held-out item belongs to exactly one
decile, so the per-decile figures partition the whole.

The decile boundaries come from the TRAINING split only, for the same reason
the models' statistics do — cutting the catalogue on counts that include the
held-out ratings would leak the test set into the analysis of the test set.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.collaborative import CollaborativeRecommender          # noqa: E402
from core.content_based import ContentBasedRecommender           # noqa: E402
from core.data import load_dataset                               # noqa: E402
from core.evaluation import (DEFAULT_K, DEFAULT_SEED,            # noqa: E402
                             DEFAULT_TEST_SIZE, RELEVANCE_THRESHOLD,
                             train_test_split_per_user)
from core.hybrid import DEFAULT_ALPHA, HybridRecommender         # noqa: E402
from core.popularity import PopularityRecommender                # noqa: E402

N_DECILES = 10


def popularity_deciles(train: pd.DataFrame, businesses: pd.DataFrame) -> pd.Series:
    """Map each restaurant to a popularity decile, 1 = least reviewed.

    Restaurants with no training ratings still need a decile, and they belong
    in the lowest one: a restaurant nobody rated is the extreme case of
    unpopular, not a missing value.

    qcut with duplicates="drop" is used because review counts are heavily tied
    at the low end -- dozens of restaurants share a count of five -- so exact
    decile boundaries do not exist and forcing ten equal bins would fail.
    """
    counts = train.groupby("business_id").size()
    counts = counts.reindex(businesses["business_id"]).fillna(0).astype(int)

    ranks = counts.rank(method="first")
    deciles = pd.qcut(ranks, q=N_DECILES, labels=False, duplicates="drop") + 1
    return deciles.astype(int)


def recall_by_decile(model, train: pd.DataFrame, test: pd.DataFrame,
                     deciles: pd.Series, k: int) -> dict[int, float]:
    """Per-decile recall: of the relevant items in this band, how many surfaced?

    Accumulated as hits and totals across all users first, then divided, rather
    than averaging per-user recalls. A per-user average would weight a user
    with one relevant item in a decile equally with a user who has six, and the
    question here is about items, not users.
    """
    users_in_train = set(train["user_id"])
    relevant_test = test[test["rating"] >= RELEVANCE_THRESHOLD]

    hits = {d: 0 for d in range(1, N_DECILES + 1)}
    totals = {d: 0 for d in range(1, N_DECILES + 1)}

    for user_id, group in relevant_test.groupby("user_id"):
        if user_id not in users_in_train:
            continue
        results = model.recommend(user_id, top_n=k)
        if results.empty:
            continue
        recommended = set(results["business_id"])

        for business_id in group["business_id"]:
            decile = deciles.get(business_id)
            if decile is None or pd.isna(decile):
                continue
            totals[int(decile)] += 1
            if business_id in recommended:
                hits[int(decile)] += 1

    return {d: (hits[d] / totals[d] if totals[d] else float("nan"))
            for d in range(1, N_DECILES + 1)}, totals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--csv", type=Path, default=None,
                        help="optional path to write the table to")
    args = parser.parse_args()

    data = load_dataset()
    train, test = train_test_split_per_user(
        data.reviews, test_size=args.test_size, seed=args.seed)

    deciles = popularity_deciles(train, data.businesses)
    deciles.index = data.businesses["business_id"]

    models = [
        PopularityRecommender(),
        ContentBasedRecommender(),
        CollaborativeRecommender(),
        HybridRecommender(alpha=DEFAULT_ALPHA),
    ]

    rows, totals_seen = {}, None
    for model in models:
        model.fit(data, ratings=train)
        recalls, totals = recall_by_decile(model, train, test, deciles, args.k)
        rows[model.name] = recalls
        totals_seen = totals

    table = pd.DataFrame(rows).T
    table.columns = [f"D{d}" for d in table.columns]

    print(f"\nRecall@{args.k} by restaurant popularity decile "
          f"(D1 = least reviewed, D{N_DECILES} = most)\n")
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))

    print("\nRelevant held-out items per decile:")
    print("  " + "  ".join(f"D{d}:{totals_seen[d]}" for d in range(1, N_DECILES + 1)))

    # The prediction the popularity-bias account makes, stated as a number.
    baseline = table.loc["Popularity baseline"]
    best_personalised = table.drop(index="Popularity baseline").max()
    advantage = baseline - best_personalised

    print("\nBaseline advantage over the best personalised model, per decile:")
    print(advantage.to_string(float_format=lambda v: f"{v:+.4f}"))

    low = advantage[[f"D{d}" for d in (1, 2, 3)]].mean()
    high = advantage[[f"D{d}" for d in (8, 9, 10)]].mean()
    print(f"\n  mean advantage, bottom three deciles: {low:+.4f}")
    print(f"  mean advantage, top three deciles:    {high:+.4f}")
    print("\n  The popularity-bias explanation predicts the second number is")
    print("  substantially larger than the first. If it is not, the explanation")
    print("  does not hold and should not be offered in the report.")

    if args.csv:
        table.to_csv(args.csv)
        print(f"\nWritten to {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
