"""
scripts/run_evaluation.py

Runs the full evaluation from the command line and prints the numbers that go
into the report.

Run from the project root:

    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --k 5 --alpha 0.7

Kept separate from the Streamlit app on purpose. The figures quoted in a report
should come from a script anyone can re-run to reproduce them, not from
whatever happened to be on screen when a screenshot was taken.
"""

import argparse
import sys
import time
from pathlib import Path

# Allow running this file directly from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from core.collaborative import CollaborativeRecommender
from core.content_based import ContentBasedRecommender
from core.data import load_dataset
from core.evaluation import evaluate_all
from core.hybrid import HybridRecommender
from core.popularity import PopularityRecommender
from core.validation import validate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate every recommender on a held-out split.")
    parser.add_argument("--k", type=int, default=10, help="Top-K size for ranking metrics (default 10)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Hybrid blend weight toward collaborative filtering (default 0.5)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction of each user's ratings held out (default 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the split (default 42)")
    args = parser.parse_args()

    print("Loading data…")
    data = load_dataset()
    print(f"  {data.n_businesses:,} restaurants · {data.n_users:,} users · {data.n_reviews:,} ratings")

    warnings = validate(data)
    if warnings:
        print(f"\n{len(warnings)} data warning(s):")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("  data validation: clean")

    models = [
        PopularityRecommender(),
        ContentBasedRecommender(),
        CollaborativeRecommender(),
        HybridRecommender(alpha=args.alpha),
    ]

    print(f"\nEvaluating {len(models)} models "
          f"(k={args.k}, test_size={args.test_size}, seed={args.seed})…")
    started = time.time()
    results = evaluate_all(models, data, k=args.k, test_size=args.test_size, seed=args.seed)
    print(f"  done in {time.time() - started:.1f}s\n")

    # Rating-prediction table
    print("RATING PREDICTION  (lower is better)")
    print("-" * 62)
    prediction = results[["model", "rmse", "mse", "n_predictions"]].copy()
    prediction.columns = ["Model", "RMSE", "MSE", "Predictions"]
    print(prediction.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # Ranking table
    print(f"\nRANKING QUALITY @ {args.k}  (higher is better)")
    print("-" * 62)
    ranking = results[["model", "precision_at_k", "recall_at_k", "f1_at_k",
                       "coverage", "n_users_evaluated"]].copy()
    ranking.columns = [f"Model", f"Precision@{args.k}", f"Recall@{args.k}",
                       f"F1@{args.k}", "Coverage", "Users"]
    print(ranking.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\nNotes for the report:")
    print(f"  · Split: {int((1 - args.test_size) * 100)}/{int(args.test_size * 100)} per user, seed {args.seed}")
    print(f"  · An item counts as relevant if its held-out rating was >= 4")
    print(f"  · Coverage is the share of the {data.n_businesses:,}-restaurant catalogue "
          f"appearing in any user's top {args.k}")


if __name__ == "__main__":
    main()
