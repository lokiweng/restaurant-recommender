"""
core/explain.py

Why a restaurant was recommended, in a sentence a diner can read.

WHAT THIS IS FOR
----------------
A recommender that outputs "match 0.87" has told the user nothing. The number
is the model's internal confidence, on a scale nobody outside the code knows,
and a visitor cannot tell whether 0.87 is remarkable or ordinary. Worse, they
have no way to judge whether the recommendation is any good, so their only
options are to trust it blindly or ignore it.

This module turns the same recommendation into a claim that can be checked:

    Shares Indian, Curry with The Pub, which you rated 5★

The reader can immediately agree or disagree, and either outcome is useful --
agreement builds trust in the next recommendation, disagreement tells them the
system has misread their taste. Neither is possible with a bare number.

In the literature this is *explainability*, and it is a recognised research
area in recommender systems rather than a cosmetic addition. Tintarev and
Masthoff's work on explanations in recommender systems is the standard
starting point, and the usual finding is that explanations improve a user's
trust and their ability to judge a recommendation, even when the underlying
accuracy is unchanged.

WHAT THIS IS NOT
----------------
These explanations are *post hoc*. They are reconstructed from the data after
the model has ranked, not extracted from the model's own reasoning, and this
module never changes what is recommended or in what order.

That distinction is worth stating plainly, because overstating it would be a
real methodological error. For the content-based model the reconstruction is
close to faithful: it ranks on shared category tags and price, which is what
the explanation reports. For collaborative filtering it is not faithful at
all -- that model has no notion of cuisine, and any tag overlap the
explanation finds is a description of the result, not the cause. The wording
below is chosen to stay honest about that: a shared-tag explanation says the
restaurants *share* tags, never that the system chose it *because* of them.
"""

from __future__ import annotations

import pandas as pd

#: A rating at or above this counts as "you liked it", and is eligible to be
#: cited in an explanation. Four matches the threshold the evaluation uses for
#: a relevant item, so the app and its metrics agree on what liking means.
LIKED_THRESHOLD = 4

#: At most this many tags are named in one explanation. Three is enough to be
#: specific; beyond that the sentence stops being readable at card size.
MAX_TAGS_SHOWN = 3

#: Tags that describe a venue, a business function or a logistics category
#: rather than a kind of food.
#:
#: Every restaurant in the dataset carries "Restaurants", and most carry
#: "Food", so without this filter every single explanation would proudly
#: report that two restaurants share the tag "Restaurants" -- true, useless,
#: and worse than no explanation at all, because it looks like the system has
#: found something when it has not.
#:
#: Defined here rather than imported so this module stands alone: it needs no
#: model, no fitted state and no other core module, which is what makes it
#: testable on a plain DataFrame.
GENERIC_TAGS = {
    "Restaurants", "Food", "Nightlife", "Arts & Entertainment", "Shopping",
    "Event Planning & Services", "Venues & Event Spaces", "Party & Event Planning",
    "Local Flavor", "Active Life", "Hotels & Travel", "Hotels", "Grocery",
    "Caterers", "Specialty Food", "Beer, Wine & Spirits", "Wholesale Stores",
    "Convenience Stores", "Professional Services", "Social Clubs",
}

PRICE_SYMBOLS = {1: "$", 2: "$$", 3: "$$$", 4: "$$$$"}


def parse_tags(categories) -> list[str]:
    """Split a Yelp category string into meaningful tags, in original order.

    "Chinese, Restaurants, Noodles" -> ["Chinese", "Noodles"]
    """
    if not isinstance(categories, str) or not categories.strip():
        return []
    return [
        tag.strip() for tag in categories.split(",")
        if tag.strip() and tag.strip() not in GENERIC_TAGS
    ]


def shared_tags(left, right) -> list[str]:
    """Tags two restaurants have in common, in the order the first lists them.

    Order is taken from `left` rather than being sorted alphabetically because
    Yelp lists a restaurant's most specific tags first, so the first shared tag
    is usually the most informative one to name.
    """
    right_set = set(parse_tags(right))
    return [tag for tag in parse_tags(left) if tag in right_set]


def _price(value) -> int | None:
    try:
        level = int(value)
    except (TypeError, ValueError):
        return None
    return level if 1 <= level <= 4 else None


def _liked_frame(ratings: dict, businesses: pd.DataFrame) -> pd.DataFrame:
    """The restaurants this visitor rated at or above the liked threshold.

    Returned with a `user_rating` column attached, sorted best first, so any
    caller looking for something to cite can take the first match it finds.
    """
    if not ratings or businesses is None or businesses.empty:
        return businesses.iloc[0:0] if businesses is not None else pd.DataFrame()

    liked_ids = [b for b, score in ratings.items() if score >= LIKED_THRESHOLD]
    if not liked_ids:
        return businesses.iloc[0:0]

    liked = businesses[businesses["business_id"].isin(liked_ids)].copy()
    liked["user_rating"] = liked["business_id"].map(ratings)
    return liked.sort_values("user_rating", ascending=False)


def explain(row: pd.Series, ratings: dict, businesses: pd.DataFrame) -> str:
    """One sentence explaining a single recommendation. "" when there is none.

    Four cases, tried in order of how much they actually tell the reader:

    1. Shared cuisine tags with something they rated highly -- the most
       specific claim available, and the one a content-based model is really
       acting on.
    2. Same price bracket as something they rated highly. Weaker, but still a
       real property of their taste.
    3. Nothing in common that this module can see. This happens when the
       ranking came from collaborative filtering, which matches behaviour
       rather than attributes, so the explanation says exactly that instead of
       inventing an attribute-based reason.
    4. No ratings at all -- the cold start. Returns "", and the caller shows
       no explanation rather than a hollow one.
    """
    liked = _liked_frame(ratings, businesses)
    if liked.empty:
        return ""

    candidate_categories = row.get("categories")

    # --- 1. shared cuisine ------------------------------------------------
    best_tags: list[str] = []
    best_name = ""
    best_score = 0

    for _, liked_row in liked.iterrows():
        if liked_row.get("business_id") == row.get("business_id"):
            continue                      # never explain a place by itself
        overlap = shared_tags(candidate_categories, liked_row.get("categories"))
        if len(overlap) > best_score:
            best_score = len(overlap)
            best_tags = overlap
            best_name = str(liked_row.get("name", ""))

    if best_tags:
        named = ", ".join(best_tags[:MAX_TAGS_SHOWN])
        rating = int(liked.loc[liked["name"] == best_name, "user_rating"].iloc[0])
        return f"Shares {named} with {best_name}, which you rated {rating}★"

    # --- 2. shared price bracket -----------------------------------------
    candidate_price = _price(row.get("price_range"))
    if candidate_price is not None:
        for _, liked_row in liked.iterrows():
            # Same guard as the tag loop above. A recommender normally excludes
            # what the visitor has already rated, but nothing here should
            # depend on that: "Same $$ price range as Cafe Tandoor" printed on
            # Cafe Tandoor's own card is the kind of error that undoes a
            # reader's trust in every other explanation on the page.
            if liked_row.get("business_id") == row.get("business_id"):
                continue
            if _price(liked_row.get("price_range")) == candidate_price:
                symbol = PRICE_SYMBOLS[candidate_price]
                return (f"Same {symbol} price range as {liked_row.get('name', '')}, "
                        f"which you rated {int(liked_row['user_rating'])}★")

    # --- 3. behavioural signal only --------------------------------------
    return "Diners who rated the same places rated this highly"


def add_reasons(frame: pd.DataFrame, ratings: dict, businesses: pd.DataFrame,
                column: str = "reason") -> pd.DataFrame:
    """Attach an explanation column to a frame of recommendations.

    A copy is returned rather than the frame being modified in place: the
    recommenders' output is cached in places, and quietly writing a column
    into a cached object is how one screen ends up changing what another one
    sees.

    The recommenders return a fixed set of columns which does not include
    `categories`, so it is merged back in from `businesses` first -- the
    explanation needs the full tag list, not the single display cuisine.
    """
    if frame is None or frame.empty:
        return frame

    working = frame.copy()

    if "categories" not in working.columns and businesses is not None:
        working = working.merge(
            businesses[["business_id", "categories"]],
            on="business_id", how="left",
        )

    working[column] = [
        explain(row, ratings, businesses) for _, row in working.iterrows()
    ]
    return working
