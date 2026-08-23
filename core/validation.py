"""
core/validation.py

Checks the dataset before any model is built.

WHY VALIDATE AT ALL
-------------------
Every model in core/ assumes things about its input: that ratings sit between
1 and 5, that business_id is unique, that reviews point at restaurants that
actually exist. When one of those assumptions quietly fails, the failure does
not look like a failure. A rating of 50 slipping into the matrix does not
raise an exception — it silently drags one user's mean upward and produces a
recommendation list that is wrong in a way nobody notices.

So the checks run once, up front, and split into two kinds:

  ERRORS   break an assumption a model depends on. Raise immediately, with a
           message naming the file and the problem, rather than letting a
           pandas traceback surface three layers down.

  WARNINGS are suspicious but survivable. Collected and returned so the
           interface can surface them, without blocking the app from running.

That distinction is the whole design. Treating everything as fatal makes the
app fragile; treating everything as a warning means real corruption reaches
the models.
"""

import pandas as pd

from core.data import Dataset

# The columns each table must have for the models to work at all.
REQUIRED_BUSINESS_COLUMNS = {"business_id", "name", "primary_category", "categories",
                             "price_range", "avg_rating", "review_count"}
REQUIRED_USER_COLUMNS = {"user_id"}
REQUIRED_REVIEW_COLUMNS = {"review_id", "user_id", "business_id", "rating"}

RATING_MIN, RATING_MAX = 1, 5
PRICE_MIN, PRICE_MAX = 1, 4


class DataValidationError(ValueError):
    """Raised when the data breaks an assumption the models rely on."""


def _require_columns(frame: pd.DataFrame, required: set, table: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise DataValidationError(
            f"{table}.csv is missing required column(s): {sorted(missing)}"
        )


def validate_businesses(businesses: pd.DataFrame) -> list[str]:
    warnings: list[str] = []
    _require_columns(businesses, REQUIRED_BUSINESS_COLUMNS, "businesses")

    if businesses.empty:
        raise DataValidationError("businesses.csv contains no rows.")

    # A duplicate business_id is fatal: the pivot tables in the models are
    # keyed on it, so duplicates would silently merge two restaurants into one.
    duplicates = int(businesses["business_id"].duplicated().sum())
    if duplicates:
        raise DataValidationError(f"businesses.csv has {duplicates} duplicate business_id value(s).")

    if businesses["business_id"].isna().any():
        raise DataValidationError("businesses.csv has null business_id values.")

    if businesses["name"].isna().any():
        warnings.append(f"{int(businesses['name'].isna().sum())} restaurant(s) have no name.")

    ratings = pd.to_numeric(businesses["avg_rating"], errors="coerce")
    out_of_range = int((~ratings.between(RATING_MIN, RATING_MAX) & ratings.notna()).sum())
    if out_of_range:
        raise DataValidationError(
            f"businesses.csv has {out_of_range} avg_rating value(s) outside {RATING_MIN}-{RATING_MAX}."
        )

    prices = pd.to_numeric(businesses["price_range"], errors="coerce")
    bad_prices = int((~prices.between(PRICE_MIN, PRICE_MAX) & prices.notna()).sum())
    if bad_prices:
        warnings.append(
            f"{bad_prices} restaurant(s) have a price_range outside {PRICE_MIN}-{PRICE_MAX}; "
            "they will be treated as missing and filled with the median."
        )

    if businesses["categories"].isna().all():
        warnings.append(
            "No restaurant has any category tags — content-based recommendations "
            "will carry no signal."
        )

    return warnings


def validate_users(users: pd.DataFrame) -> list[str]:
    warnings: list[str] = []
    _require_columns(users, REQUIRED_USER_COLUMNS, "users")

    if users.empty:
        raise DataValidationError("users.csv contains no rows.")

    duplicates = int(users["user_id"].duplicated().sum())
    if duplicates:
        # Survivable: the models key off the reviews table, not this one.
        warnings.append(f"users.csv has {duplicates} duplicate user_id value(s).")

    return warnings


def validate_reviews(reviews: pd.DataFrame, business_ids: set, user_ids: set) -> list[str]:
    warnings: list[str] = []
    _require_columns(reviews, REQUIRED_REVIEW_COLUMNS, "reviews")

    if reviews.empty:
        raise DataValidationError("reviews.csv contains no rows — there is nothing to learn from.")

    ratings = pd.to_numeric(reviews["rating"], errors="coerce")

    if ratings.isna().any():
        raise DataValidationError(
            f"reviews.csv has {int(ratings.isna().sum())} rating(s) that are not numbers."
        )

    # Fatal, and the most important check in this file: an out-of-range rating
    # corrupts every average, every similarity and every prediction downstream,
    # without ever raising an error of its own.
    bad = ratings[~ratings.between(RATING_MIN, RATING_MAX)]
    if not bad.empty:
        raise DataValidationError(
            f"reviews.csv has {len(bad)} rating(s) outside {RATING_MIN}-{RATING_MAX}: "
            f"{sorted(bad.unique().tolist())[:5]}"
        )

    orphan_businesses = int((~reviews["business_id"].isin(business_ids)).sum())
    if orphan_businesses:
        warnings.append(
            f"{orphan_businesses} review(s) reference a restaurant that is not in businesses.csv; "
            "they contribute nothing and are ignored."
        )

    orphan_users = int((~reviews["user_id"].isin(user_ids)).sum())
    if orphan_users:
        warnings.append(f"{orphan_users} review(s) reference a user not listed in users.csv.")

    duplicate_reviews = int(reviews["review_id"].duplicated().sum())
    if duplicate_reviews:
        warnings.append(f"reviews.csv has {duplicate_reviews} duplicate review_id value(s).")

    # The same person rating the same restaurant twice makes the pivot table
    # silently average them. Worth surfacing, not worth refusing to start over.
    repeated = int(reviews.duplicated(subset=["user_id", "business_id"]).sum())
    if repeated:
        warnings.append(
            f"{repeated} user/restaurant pair(s) appear more than once; "
            "duplicate ratings are averaged together."
        )

    return warnings


def validate(data: Dataset) -> list[str]:
    """Run every check.

    Raises DataValidationError on anything that would break a model. Returns a
    list of human-readable warnings for anything merely suspicious — an empty
    list means the data is completely clean.
    """
    warnings: list[str] = []
    warnings += validate_businesses(data.businesses)
    warnings += validate_users(data.users)
    warnings += validate_reviews(
        data.reviews,
        business_ids=set(data.businesses["business_id"]),
        user_ids=set(data.users["user_id"]),
    )
    return warnings
