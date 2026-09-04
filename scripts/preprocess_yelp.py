"""
preprocess_yelp.py

Turns the raw Yelp Open Dataset into the three CSV files in data/.

Run with:  python scripts/preprocess_yelp.py --raw path/to/yelp_dataset
     or:   python scripts/preprocess_yelp.py --raw path/to/yelp_dataset --verify

Run it from the repository root, like the other scripts. It writes to data/
next to app.py, resolved from this file rather than from the working
directory, so it does not matter what folder you happen to be standing in.

WHY THIS FILE IS HERE BUT CANNOT BE RUN FROM A CLEAN CHECKOUT
-------------------------------------------------------------
The raw export is 4.35 GB compressed and 8.65 GB uncompressed, so it is not
redistributed with this submission. What is redistributed is the three derived
CSVs and this script, which is the record of exactly how they were produced:
which businesses counted as restaurants, how the city filter was applied, how
the 5-core density filter was performed, and which columns were computed
rather than copied.

Point --raw at a folder containing the two files this project uses:

    yelp_academic_dataset_business.json
    yelp_academic_dataset_review.json

Both are newline-delimited JSON (one object per line), not JSON arrays, so
they are streamed a line at a time rather than loaded whole — the review file
has 6,990,280 lines and will not fit in memory on a laptop.

WHAT --verify DOES
------------------
Regenerates the three tables and compares them cell-by-cell against the CSVs
already committed in data/, without overwriting them. That is the check that
this script really is the provenance of the shipped data rather than a
plausible-looking reconstruction of it. Use it before trusting anything here.

THE PIPELINE, IN ORDER
----------------------
  1. business.json  ->  keep city == "Cleveland" AND "Restaurants" in categories
  2. review.json    ->  keep reviews pointing at a surviving restaurant
  3. 5-core density filter, ONE PASS (see the long note on that function)
  4. recompute avg_rating and review_count from the surviving reviews
  5. derive primary_category as a display label
  6. write businesses.csv, users.csv, reviews.csv
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Filter parameters. Every one of these is a decision the report has to
# justify, so they are named constants rather than literals buried in the code.
# ---------------------------------------------------------------------------

#: The single metropolitan area this project covers. Matched exactly against
#: Yelp's `city` field — Cleveland is mid-sized enough to retain a usable
#: rating core after 5-core filtering, while staying small enough that the
#: full evaluation runs in under a minute.
CITY = "Cleveland"

#: A business counts as a restaurant if this token appears in its comma
#: separated `categories` string. Yelp applies it to every food-serving venue,
#: so it is the category test rather than any list of cuisines.
RESTAURANT_TAG = "Restaurants"

#: Minimum ratings a user and a restaurant must each have to be retained.
K_CORE = 5

#: Yelp files consumed. Only these two of the five in the export are used.
BUSINESS_FILE = "yelp_academic_dataset_business.json"
REVIEW_FILE = "yelp_academic_dataset_review.json"

#: Column order of the written files. Fixed, because core/validation.py checks
#: for these names and the models index into them positionally nowhere but
#: still read them by name.
BUSINESS_COLUMNS = ["business_id", "name", "primary_category", "categories",
                    "city", "price_range", "avg_rating", "review_count"]
REVIEW_COLUMNS = ["review_id", "user_id", "business_id", "rating", "date"]
USER_COLUMNS = ["user_id"]


# ---------------------------------------------------------------------------
# Stage 1 — the catalogue
# ---------------------------------------------------------------------------

def load_restaurants(raw_dir: Path) -> pd.DataFrame:
    """Businesses in CITY whose categories include RESTAURANT_TAG.

    Two fields need care.

    `categories` is None for a small number of businesses in the export. A
    None cannot be searched for the tag, so those rows are skipped rather than
    coerced to an empty string — a business with no categories at all is not
    evidence of a restaurant.

    `attributes.RestaurantsPriceRange2` is Yelp's 1-4 price tier and is
    missing for a substantial minority of listings. It is NOT a reason to drop
    a business: doing so would remove restaurants that have perfectly good
    rating data, and would mean the catalogue reduction reported in the write
    up could no longer be attributed to the density filter alone. The gaps are
    filled with the median observed tier after loading, which is recorded here
    as an imputation rather than left to look like observed data.
    """
    path = raw_dir / BUSINESS_FILE
    if not path.is_file():
        raise SystemExit(f"Missing {BUSINESS_FILE} in {raw_dir}")

    rows = []
    for line in path.open(encoding="utf-8"):
        record = json.loads(line)

        if record.get("city") != CITY:
            continue

        categories = record.get("categories")
        if not categories or RESTAURANT_TAG not in categories:
            continue

        attributes = record.get("attributes") or {}
        price = attributes.get("RestaurantsPriceRange2")
        try:
            price = int(price)
        except (TypeError, ValueError):
            price = None                     # missing, "None", or unparseable

        rows.append({
            "business_id": record["business_id"],
            "name": record["name"],
            "categories": categories,
            "city": record["city"],
            "price_range": price,
        })

    businesses = pd.DataFrame(rows)
    if businesses.empty:
        raise SystemExit(f"No businesses matched city={CITY!r} with tag {RESTAURANT_TAG!r}")

    n_missing = int(businesses["price_range"].isna().sum())
    median_price = businesses["price_range"].median()
    businesses["price_range"] = (
        businesses["price_range"].fillna(median_price).astype(int)
    )
    if n_missing:
        print(f"  price_range imputed for {n_missing} businesses "
              f"(median tier {int(median_price)})")

    return businesses


def primary_category(categories: str) -> str:
    """The single cuisine label shown on a restaurant card.

    Yelp lists categories in no meaningful order and every restaurant in this
    dataset carries the generic "Restaurants" tag, so taking the literal first
    element would label a fifth of the catalogue "Restaurants". Dropping that
    one token and taking the next is what produces a usable label for the
    Browse screen's cuisine filter.

    The handful of businesses whose ONLY tag is "Restaurants" have nothing left
    after the drop; they are labelled with the singular form so the filter has
    a value to show rather than a blank.

    This column is presentational and is read only by the interface. The
    content-based model builds its TF-IDF features from the full `categories`
    string (see core/content_based.py), so changing this rule cannot move any
    number reported in the evaluation.
    """
    tags = [t.strip() for t in str(categories).split(",") if t.strip()]
    for tag in tags:
        if tag != RESTAURANT_TAG:
            return tag
    return "Restaurant"


# ---------------------------------------------------------------------------
# Stage 2 — the ratings
# ---------------------------------------------------------------------------

def load_reviews(raw_dir: Path, business_ids: set) -> pd.DataFrame:
    """Every review pointing at one of the retained restaurants.

    Streamed line by line: the review file holds 6,990,280 records and reading
    it with pd.read_json would exhaust memory long before the filter runs.

    Yelp's `date` is a full timestamp ("2013-11-19 20:31:49"). Only the day is
    kept, because nothing downstream uses the time and a shorter field keeps
    the CSV readable. The timestamps that remain are what would make the
    temporal split identified as future work directly implementable.
    """
    path = raw_dir / REVIEW_FILE
    if not path.is_file():
        raise SystemExit(f"Missing {REVIEW_FILE} in {raw_dir}")

    rows = []
    for line in path.open(encoding="utf-8"):
        record = json.loads(line)
        if record["business_id"] not in business_ids:
            continue
        rows.append({
            "review_id": record["review_id"],
            "user_id": record["user_id"],
            "business_id": record["business_id"],
            "rating": int(record["stars"]),
            "date": str(record["date"])[:10],
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stage 3 — the density filter
# ---------------------------------------------------------------------------

def five_core_single_pass(reviews: pd.DataFrame, k: int = K_CORE) -> pd.DataFrame:
    """Drop sparse users and sparse restaurants in ONE simultaneous pass.

    WHAT THIS DOES, PRECISELY
    -------------------------
    Rating counts are taken once, on the unfiltered set. Users with at least k
    and restaurants with at least k are identified from those counts, and a
    review survives only if BOTH its endpoints did.

    WHY IT IS NOT A TRUE k-CORE, AND WHY THAT IS DISCLOSED
    -----------------------------------------------------
    Because both filters read the same pre-filter counts, removing a sparse
    user can push a restaurant below the threshold and vice versa. A single
    pass therefore does not guarantee the property it appears to claim: the
    result is APPROXIMATELY k-core, not exactly so.

    On this dataset the residue is one user left with four ratings and ten
    restaurants left with four. Iterating to convergence takes four passes and
    removes a further 68 ratings, 0.3% of the total — see
    report_convergence_cost() below, which quantifies it rather than asserting
    it. The single pass is what produced the committed data and is kept so the
    reported figures reproduce; the shortfall is stated in the write-up rather
    than papered over.
    """
    user_counts = reviews["user_id"].value_counts()
    business_counts = reviews["business_id"].value_counts()

    keep_users = set(user_counts[user_counts >= k].index)
    keep_businesses = set(business_counts[business_counts >= k].index)

    return reviews[
        reviews["user_id"].isin(keep_users)
        & reviews["business_id"].isin(keep_businesses)
    ].reset_index(drop=True)


def report_convergence_cost(reviews: pd.DataFrame, k: int = K_CORE) -> None:
    """Print what iterating the filter to convergence WOULD cost.

    Nothing here changes the output. It exists so the claim made in the report
    — that full convergence costs four passes and 68 further ratings — is
    produced by the code rather than quoted from memory.
    """
    current = reviews
    passes = 0
    while True:
        reduced = five_core_single_pass(current, k)
        passes += 1
        if len(reduced) == len(current):
            break
        current = reduced

    removed = len(reviews) - len(current)
    share = removed / len(reviews) * 100 if len(reviews) else 0.0
    print(f"  iterating to convergence would take {passes} passes and remove "
          f"a further {removed:,} ratings ({share:.1f}% of the filtered set)")


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the whole pipeline and return the three tables."""
    print(f"Reading {BUSINESS_FILE}…")
    businesses = load_restaurants(raw_dir)
    print(f"  {len(businesses):,} restaurants in {CITY}")

    print(f"Reading {REVIEW_FILE}… (this takes a few minutes)")
    reviews = load_reviews(raw_dir, set(businesses["business_id"]))
    print(f"  {len(reviews):,} reviews from {reviews['user_id'].nunique():,} users")
    print(f"  rating-matrix density before filtering: "
          f"{len(reviews) / (len(businesses) * reviews['user_id'].nunique()) * 100:.2f}%")

    print(f"Applying the {K_CORE}-core density filter (single pass)…")
    reviews = five_core_single_pass(reviews, K_CORE)
    report_convergence_cost(reviews, K_CORE)

    # The catalogue is cut back to restaurants that still have ratings. A
    # restaurant with none cannot be scored by any model and would only pad
    # the coverage denominator.
    businesses = businesses[
        businesses["business_id"].isin(set(reviews["business_id"]))
    ].reset_index(drop=True)

    # avg_rating and review_count are COMPUTED from the surviving reviews, not
    # copied from Yelp's `stars` and `review_count` fields. Those fields count
    # every review Yelp holds, including ones for other cities' visitors and
    # ones this pipeline discarded, so carrying them over would put statistics
    # into businesses.csv that do not match the ratings the models are fitted
    # on. (core/popularity.py additionally recomputes both from the TRAINING
    # split at evaluation time, which is a separate concern — see the note on
    # test-set leakage in that file.)
    aggregates = reviews.groupby("business_id")["rating"].agg(["mean", "count"])
    businesses = businesses.join(aggregates, on="business_id")
    businesses = businesses.rename(columns={"mean": "avg_rating", "count": "review_count"})
    businesses["review_count"] = businesses["review_count"].astype(int)

    businesses["primary_category"] = businesses["categories"].apply(primary_category)

    users = pd.DataFrame({"user_id": sorted(set(reviews["user_id"]))})

    density = len(reviews) / (len(businesses) * len(users)) * 100
    print(f"  {len(businesses):,} restaurants · {len(users):,} users · "
          f"{len(reviews):,} ratings · density {density:.2f}% · "
          f"{len(reviews) / len(users):.1f} ratings per user")

    return (businesses[BUSINESS_COLUMNS],
            users[USER_COLUMNS],
            reviews[REVIEW_COLUMNS])


def write(tables, out_dir: Path) -> None:
    businesses, users, reviews = tables
    out_dir.mkdir(parents=True, exist_ok=True)
    businesses.to_csv(out_dir / "businesses.csv", index=False)
    users.to_csv(out_dir / "users.csv", index=False)
    reviews.to_csv(out_dir / "reviews.csv", index=False)
    print(f"Wrote three CSVs to {out_dir}")


def verify(tables, out_dir: Path) -> int:
    """Compare freshly built tables against the committed CSVs.

    Compares as SETS of rows keyed on the natural identifier, so a difference
    in row order — which depends on the order records happen to appear in the
    raw export — is not reported as a mismatch, while a difference in any
    value is.

    Returns a process exit code: 0 if identical, 1 otherwise.
    """
    built = dict(zip(("businesses", "users", "reviews"), tables))
    keys = {"businesses": "business_id", "users": "user_id", "reviews": "review_id"}
    problems = 0

    for name, fresh in built.items():
        path = out_dir / f"{name}.csv"
        if not path.is_file():
            print(f"  {name}.csv: not present, nothing to compare")
            problems += 1
            continue

        committed = pd.read_csv(path)
        key = keys[name]

        if set(fresh.columns) != set(committed.columns):
            print(f"  {name}.csv: COLUMN MISMATCH "
                  f"{sorted(set(fresh.columns) ^ set(committed.columns))}")
            problems += 1
            continue

        a = fresh.set_index(key).sort_index()
        b = committed[list(fresh.columns)].set_index(key).sort_index()

        if len(a) != len(b):
            print(f"  {name}.csv: ROW COUNT {len(a):,} rebuilt vs {len(b):,} committed")
            problems += 1
            continue
        if not a.index.equals(b.index):
            print(f"  {name}.csv: {len(a.index.difference(b.index)):,} ids differ")
            problems += 1
            continue

        differing = []
        for column in a.columns:
            left, right = a[column], b[column]
            if pd.api.types.is_float_dtype(left) or pd.api.types.is_float_dtype(right):
                same = ((left - right).abs() < 1e-6) | (left.isna() & right.isna())
            else:
                same = (left == right) | (left.isna() & right.isna())
            if not same.all():
                differing.append(f"{column} ({int((~same).sum())} rows)")

        if differing:
            print(f"  {name}.csv: VALUE MISMATCH in {', '.join(differing)}")
            problems += 1
        else:
            print(f"  {name}.csv: identical ({len(a):,} rows, {len(a.columns)} columns)")

    print()
    if problems:
        print("VERIFY FAILED — the committed CSVs are not what this script produces.")
    else:
        print("VERIFY PASSED — this script reproduces the committed CSVs exactly.")
    return 1 if problems else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build data/{businesses,users,reviews}.csv from the raw Yelp Open Dataset."
    )
    parser.add_argument("--raw", required=True, type=Path,
                        help="folder holding the raw yelp_academic_dataset_*.json files")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parent.parent / "data",
                        help="where to write the three CSVs (default: the repo's data/ folder)")
    parser.add_argument("--verify", action="store_true",
                        help="compare against the committed CSVs instead of overwriting them")
    args = parser.parse_args()

    if not args.raw.is_dir():
        raise SystemExit(f"--raw folder not found: {args.raw}")

    tables = build(args.raw)

    if args.verify:
        print("\nVerifying against the committed CSVs…")
        return verify(tables, args.out)

    write(tables, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
