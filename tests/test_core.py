"""
tests/test_core.py

Tests for the recommendation engine.

Run from the project root:   python -m pytest tests/ -v

WHAT IS BEING TESTED, AND WHY THESE THINGS
------------------------------------------
A recommender is unusually easy to get subtly wrong, because almost any bug
still produces a plausible-looking list of restaurants. Nothing crashes; the
output is simply worse than it should be, in a way no amount of clicking
around will reveal. These tests target the failures that hide:

  * recommending a restaurant back to the person who just rated it
  * predictions drifting outside the 1-5 scale
  * a model quietly returning nothing for an unknown user instead of falling
    back to something sensible
  * the evaluation harness leaking test ratings into training, which would
    make every metric look excellent and mean nothing

Most tests run against a small hand-built fixture rather than the real 26,096
ratings, so the whole suite finishes in seconds and each failure points at one
specific behaviour. Two tests at the end use the real data, because some
properties only appear at scale.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.base import NotFittedError
from core.collaborative import CollaborativeRecommender
from core.content_based import ContentBasedRecommender
from core.data import Dataset, DatasetError, load_dataset
from core.evaluation import (evaluate_all, evaluate_model, mse, rmse,
                             train_test_split_per_user)
from core.hybrid import HybridRecommender
from core.popularity import PopularityRecommender
from core.validation import DataValidationError, validate


# ---------------------------------------------------------------------------
# Fixtures: a tiny, fully understood dataset
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny() -> Dataset:
    """Six restaurants, four users, deliberately structured.

    Users u1 and u2 both like Italian and dislike Sushi; u3 is the mirror
    image. That pattern is strong enough that a working collaborative filter
    must pick it up, which makes the assertions below meaningful rather than
    just "it returned something".
    """
    businesses = pd.DataFrame([
        {"business_id": "b1", "name": "Pasta Place",  "primary_category": "Italian",
         "categories": "Italian, Restaurants",  "price_range": 2, "avg_rating": 4.5, "review_count": 100},
        {"business_id": "b2", "name": "Pizza Corner", "primary_category": "Italian",
         "categories": "Italian, Pizza, Restaurants", "price_range": 1, "avg_rating": 4.2, "review_count": 80},
        {"business_id": "b3", "name": "Sushi Bar",    "primary_category": "Japanese",
         "categories": "Japanese, Sushi, Restaurants", "price_range": 3, "avg_rating": 4.8, "review_count": 60},
        {"business_id": "b4", "name": "Ramen House",  "primary_category": "Japanese",
         "categories": "Japanese, Ramen, Restaurants", "price_range": 2, "avg_rating": 4.0, "review_count": 40},
        {"business_id": "b5", "name": "Taco Stand",   "primary_category": "Mexican",
         "categories": "Mexican, Restaurants", "price_range": 1, "avg_rating": 3.9, "review_count": 20},
        {"business_id": "b6", "name": "New Trattoria", "primary_category": "Italian",
         "categories": "Italian, Restaurants", "price_range": 2, "avg_rating": 5.0, "review_count": 2},
    ])
    users = pd.DataFrame({"user_id": ["u1", "u2", "u3", "u4"]})
    reviews = pd.DataFrame([
        {"review_id": "r1", "user_id": "u1", "business_id": "b1", "rating": 5},
        {"review_id": "r2", "user_id": "u1", "business_id": "b2", "rating": 5},
        {"review_id": "r3", "user_id": "u1", "business_id": "b3", "rating": 2},
        {"review_id": "r4", "user_id": "u2", "business_id": "b1", "rating": 5},
        {"review_id": "r5", "user_id": "u2", "business_id": "b2", "rating": 4},
        {"review_id": "r6", "user_id": "u2", "business_id": "b3", "rating": 1},
        {"review_id": "r7", "user_id": "u3", "business_id": "b3", "rating": 5},
        {"review_id": "r8", "user_id": "u3", "business_id": "b4", "rating": 5},
        {"review_id": "r9", "user_id": "u3", "business_id": "b1", "rating": 2},
        {"review_id": "r10", "user_id": "u4", "business_id": "b5", "rating": 4},
        {"review_id": "r11", "user_id": "u4", "business_id": "b2", "rating": 4},
        # b6 exists to test Bayesian shrinkage: a perfect average from almost
        # no evidence. These two ratings are what make it so -- the avg_rating
        # and review_count columns above are display metadata and are
        # deliberately NOT read by any model, because under a train/test split
        # they carry the held-out ratings the model is about to be graded on.
        {"review_id": "r12", "user_id": "u1", "business_id": "b6", "rating": 5},
        {"review_id": "r13", "user_id": "u2", "business_id": "b6", "rating": 5},
    ])
    return Dataset(businesses=businesses, users=users, reviews=reviews)


ALL_MODELS = [PopularityRecommender, ContentBasedRecommender,
              CollaborativeRecommender, HybridRecommender]


# ---------------------------------------------------------------------------
# The shared contract -- every model must satisfy all of these
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_class", ALL_MODELS)
def test_predictions_stay_on_the_rating_scale(model_class, tiny):
    """No model may predict outside 1-5. A 5.3-star prediction is meaningless,
    and would corrupt RMSE in a way that looks like poor accuracy."""
    model = model_class().fit(tiny)
    for user in tiny.users["user_id"]:
        for business in tiny.businesses["business_id"]:
            value = model.predict(user, business)
            if not np.isnan(value):
                assert 1.0 <= value <= 5.0, f"{model.name} predicted {value}"


@pytest.mark.parametrize("model_class", ALL_MODELS)
def test_never_recommends_something_already_rated(model_class, tiny):
    """The most visible possible failure: telling a user to try the restaurant
    they just reviewed."""
    model = model_class().fit(tiny)
    for user in ["u1", "u2", "u3"]:
        already_rated = set(tiny.reviews.loc[tiny.reviews["user_id"] == user, "business_id"])
        recommended = set(model.recommend(user, top_n=6)["business_id"])
        assert not (recommended & already_rated), f"{model.name} re-recommended to {user}"


@pytest.mark.parametrize("model_class", ALL_MODELS)
def test_unknown_user_falls_back_instead_of_failing(model_class, tiny):
    """A visitor nobody has seen before must still get a sensible list, not an
    empty one and not an exception. This is the cold-start path."""
    model = model_class().fit(tiny)
    results = model.recommend("nobody-has-this-id", top_n=3)
    assert not results.empty
    assert len(results) == 3


@pytest.mark.parametrize("model_class", ALL_MODELS)
def test_results_have_the_agreed_shape(model_class, tiny):
    """Every model returns the same columns, so the interface can render any
    of them with one component."""
    model = model_class().fit(tiny)
    results = model.recommend("u1", top_n=3)
    for column in ["business_id", "name", "primary_category", "avg_rating", "score"]:
        assert column in results.columns, f"{model.name} is missing '{column}'"
    # Sorted best-first.
    assert results["score"].is_monotonic_decreasing


@pytest.mark.parametrize("model_class", ALL_MODELS)
def test_using_a_model_before_fitting_is_an_error(model_class):
    """Silently returning nothing would be worse: it looks like 'no results'
    rather than 'you forgot a step'."""
    model = model_class()
    with pytest.raises(NotFittedError):
        model.recommend("u1")


@pytest.mark.parametrize("model_class", ALL_MODELS)
def test_live_session_ratings_work_without_refitting(model_class, tiny):
    """A guest rates two restaurants and immediately gets picks, with neither
    of the rated ones echoed back."""
    model = model_class().fit(tiny)
    session = {"b1": 5, "b2": 4}
    results = model.recommend_from_ratings(session, top_n=3)
    assert not results.empty
    assert not (set(results["business_id"]) & set(session))


# ---------------------------------------------------------------------------
# Model-specific behaviour
# ---------------------------------------------------------------------------

def test_content_based_matches_on_cuisine(tiny):
    """u1 rated both Italian places 5 and the sushi place 2, so the unrated
    Italian restaurant should out-rank the unrated Japanese one."""
    model = ContentBasedRecommender().fit(tiny)
    scores = model.score_all("u1")
    assert scores["b6"] > scores["b4"], "Italian should beat Japanese for an Italian-lover"


def test_content_based_finds_similar_items_without_a_user(tiny):
    """Item-to-item similarity powers 'you might also like', which involves no
    user at all."""
    model = ContentBasedRecommender().fit(tiny)
    similar = model.similar_to("b1", top_n=2)
    assert "b1" not in set(similar["business_id"]), "a restaurant must not be similar to itself"
    assert similar.iloc[0]["primary_category"] == "Italian"


def test_collaborative_uses_behaviour_not_attributes(tiny):
    """u1 and u2 rate alike. u2 rated nothing u1 hasn't, so the signal must
    come through item similarity: b4 (liked by sushi-loving u3) should not
    out-rank b6 for u1, whose taste opposes u3's."""
    model = CollaborativeRecommender().fit(tiny)
    prediction = model.predict("u1", "b4")
    assert not np.isnan(prediction)
    assert 1.0 <= prediction <= 5.0


def test_collaborative_shrinks_thin_evidence(tiny):
    """With almost no supporting similarity, a prediction must not come back
    as a confident 5.0 — it should be pulled toward the user's own mean.

    This is the exact bug the shrinkage term exists to prevent.
    """
    model = CollaborativeRecommender().fit(tiny)
    session = {"b1": 5}                      # a single five-star rating
    scores = model.score_all_from_ratings(session)
    # Nothing should reach a perfect 5.0 on the strength of one rating.
    assert scores.max() < 5.0, "a lone rating produced a maximum-confidence score"


def test_popularity_ignores_who_is_asking(tiny):
    """The definition of a non-personalised baseline."""
    model = PopularityRecommender().fit(tiny)
    assert model.predict("u1", "b1") == model.predict("u3", "b1")


def test_popularity_discounts_a_restaurant_with_almost_no_reviews(tiny):
    """b4 and b6 both average a perfect 5.0 — b4 from one rating, b6 from two.

    Holding the average fixed and varying only the evidence behind it isolates
    the Bayesian term exactly: with nothing else to separate them, the one
    supported by more ratings must score higher. A model ranking on raw
    averages would tie them.
    """
    model = PopularityRecommender().fit(tiny)
    assert model.predict("u1", "b6") > model.predict("u1", "b4")


def test_popularity_ignores_the_precomputed_catalogue_columns(tiny):
    """The regression test for the test-set leak.

    businesses.csv carries avg_rating and review_count computed from every
    review in the dataset. If the baseline reads them, then under a train/test
    split it scores restaurants using statistics that already reflect the
    held-out ratings it is about to be measured on — and since the headline
    finding of this project is that this baseline beats all three personalised
    models, that leak sits directly beneath the central claim.

    Rewriting those columns to nonsense must therefore change nothing. If this
    test fails, the leak is back.
    """
    import dataclasses

    tampered = tiny.businesses.copy()
    tampered["avg_rating"] = 1.0
    tampered["review_count"] = 0
    poisoned = dataclasses.replace(tiny, businesses=tampered)

    honest = PopularityRecommender().fit(tiny)
    tampered_model = PopularityRecommender().fit(poisoned)

    for business_id in tiny.businesses["business_id"]:
        assert honest.predict("u1", business_id) == pytest.approx(
            tampered_model.predict("u1", business_id)
        ), f"{business_id}: the baseline is reading the catalogue columns again"


def test_hybrid_alpha_controls_the_blend(tiny):
    """alpha=0 must reproduce content-based, alpha=1 must reproduce
    collaborative. If it doesn't, the blend is wired up backwards."""
    content = ContentBasedRecommender().fit(tiny)
    collaborative = CollaborativeRecommender().fit(tiny)

    pure_content = HybridRecommender(alpha=0.0, content=content, collaborative=collaborative).fit(tiny)
    pure_collab = HybridRecommender(alpha=1.0, content=content, collaborative=collaborative).fit(tiny)

    assert pure_content.predict("u1", "b4") == pytest.approx(content.predict("u1", "b4"))
    assert pure_collab.predict("u1", "b4") == pytest.approx(collaborative.predict("u1", "b4"))


def test_hybrid_rejects_an_impossible_alpha():
    with pytest.raises(ValueError):
        HybridRecommender(alpha=1.5)


# ---------------------------------------------------------------------------
# Evaluation harness
# ---------------------------------------------------------------------------

def test_rmse_and_mse_are_zero_for_perfect_predictions():
    assert rmse([3, 4, 5], [3, 4, 5]) == 0.0
    assert mse([3, 4, 5], [3, 4, 5]) == 0.0


def test_rmse_is_the_square_root_of_mse():
    actual, predicted = [1, 2, 3, 4], [2, 2, 4, 3]
    assert rmse(actual, predicted) == pytest.approx(np.sqrt(mse(actual, predicted)))


def test_split_never_leaks_a_rating_into_both_sides(tiny):
    """The single most damaging bug an evaluation harness can have: if a test
    rating is also in training, the model has already seen the answer and
    every metric becomes meaningless."""
    train, test = train_test_split_per_user(tiny.reviews, test_size=0.34, seed=1)
    assert set(train.index) & set(test.index) == set()
    assert len(train) + len(test) == len(tiny.reviews)


def test_split_leaves_every_tested_user_with_training_history(tiny):
    """Otherwise the metrics measure cold start, not accuracy."""
    train, test = train_test_split_per_user(tiny.reviews, test_size=0.34, seed=1)
    for user in test["user_id"].unique():
        assert user in set(train["user_id"]), f"{user} was tested with no training data"


def test_split_is_reproducible(tiny):
    """A report cannot quote numbers that change between runs."""
    a_train, a_test = train_test_split_per_user(tiny.reviews, seed=7)
    b_train, b_test = train_test_split_per_user(tiny.reviews, seed=7)
    pd.testing.assert_frame_equal(a_train, b_train)
    pd.testing.assert_frame_equal(a_test, b_test)


def test_evaluation_produces_every_required_metric(tiny):
    """The assignment names RMSE, MSE, Precision, Recall and F1 explicitly."""
    train, test = train_test_split_per_user(tiny.reviews, test_size=0.34, seed=3)
    result = evaluate_model(PopularityRecommender(), tiny, train, test, k=3)
    assert result.model
    assert result.rmse is not None and result.mse is not None
    assert result.n_predictions > 0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_clean_data_produces_no_warnings(tiny):
    assert validate(tiny) == []


def test_an_out_of_range_rating_is_fatal(tiny):
    """This is the check that matters most: a rating of 50 would silently
    corrupt every average and similarity downstream without raising anything
    on its own."""
    broken = tiny.reviews.copy()
    broken.loc[0, "rating"] = 50
    with pytest.raises(DataValidationError, match="outside"):
        validate(Dataset(tiny.businesses, tiny.users, broken))


def test_duplicate_restaurant_ids_are_fatal(tiny):
    broken = pd.concat([tiny.businesses, tiny.businesses.head(1)], ignore_index=True)
    with pytest.raises(DataValidationError, match="duplicate"):
        validate(Dataset(broken, tiny.users, tiny.reviews))


def test_a_missing_column_is_reported_by_name(tiny):
    broken = tiny.businesses.drop(columns=["price_range"])
    with pytest.raises(DataValidationError, match="price_range"):
        validate(Dataset(broken, tiny.users, tiny.reviews))


def test_orphan_reviews_warn_but_do_not_stop_the_app(tiny):
    """A review pointing at a restaurant that isn't in the catalogue is
    suspicious, but the other 26,000 ratings are still perfectly usable."""
    extra = pd.DataFrame([{"review_id": "rX", "user_id": "u1",
                           "business_id": "does-not-exist", "rating": 4}])
    warnings = validate(Dataset(tiny.businesses, tiny.users,
                                pd.concat([tiny.reviews, extra], ignore_index=True)))
    assert any("not in businesses.csv" in w for w in warnings)


def test_missing_data_directory_raises_a_readable_error():
    with pytest.raises(DatasetError, match="Data folder not found"):
        load_dataset("/tmp/definitely-not-a-real-data-directory")


# ---------------------------------------------------------------------------
# Against the real dataset -- slower, and only for properties that need scale
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real() -> Dataset:
    try:
        return load_dataset()
    except DatasetError as exc:
        pytest.skip(f"real dataset unavailable: {exc}")


def test_the_shipped_dataset_is_valid(real):
    """The data actually in data/ must pass its own validation."""
    validate(real)   # raises on anything fatal


def test_models_fit_the_real_dataset_quickly(real):
    """Guards against an accidental return to per-pair Python loops, which is
    the difference between two seconds and several minutes at this size."""
    import time

    started = time.time()
    CollaborativeRecommender().fit(real)
    elapsed = time.time() - started
    assert elapsed < 60, f"collaborative filtering took {elapsed:.1f}s to fit — check for a loop"


def test_recommendations_do_not_repeat_a_restaurant_name(real):
    """The dataset contains chains — five Applebee's, two Al's Delis — as
    separate businesses with separate ids. That is correct data, but a top-10
    list showing the same name three times reads as a bug, and with no address
    field there is no way to tell the branches apart.

    Browsing the catalogue still shows every branch; only recommendations
    collapse them.
    """
    from core.popularity import PopularityRecommender as _P

    model = _P().fit(real)
    for user in list(real.users["user_id"])[:5]:
        names = model.recommend(user, top_n=10)["name"].tolist()
        assert len(names) == len(set(names)), f"duplicate name in results for {user}: {names}"


def test_recommendations_still_return_a_full_list_after_collapsing(real):
    """Collapsing branches must not quietly shorten the list."""
    from core.content_based import ContentBasedRecommender as _C

    model = _C().fit(real)
    assert len(model.recommend(real.users["user_id"].iloc[0], top_n=10)) == 10


# ---------------------------------------------------------------------------
# Session ratings arrive from browser state, not from the validated CSVs
# ---------------------------------------------------------------------------

def test_malformed_session_ratings_are_cleaned_not_crashed(tiny):
    """st.session_state is not validated data, and the four models used to
    disagree about what a malformed entry meant.

    Given the same bad rating the collaborative model returned an empty list,
    the content-based model raised TypeError, and the page showed either "no
    recommendations" with no explanation or a red traceback. Cleaning centrally
    in recommend_from_ratings() means all four now behave identically."""
    from core.base import clean_session_ratings

    models = [
        PopularityRecommender().fit(tiny),
        ContentBasedRecommender().fit(tiny),
        CollaborativeRecommender().fit(tiny),
        HybridRecommender(alpha=0.5).fit(tiny),
    ]
    known = tiny.businesses["business_id"].tolist()

    junk = {
        known[0]: "4",              # a string that happens to parse
        known[1]: None,             # nothing at all
        known[2]: float("nan"),     # a cleared widget
        known[3]: 0.0,              # below the scale
        known[4]: 9.0,              # above the scale
        "NOT_A_REAL_ID": 5.0,       # stale id from an older session
    }

    cleaned = clean_session_ratings(junk, set(known))
    assert cleaned == {known[0]: 4.0}, "only the parseable in-range rating survives"

    for model in models:
        result = model.recommend_from_ratings(junk, top_n=3)
        assert not result.empty, f"{model.name} returned nothing for a cleanable session"
        assert known[0] not in set(result["business_id"]), \
            f"{model.name} recommended a restaurant the visitor had just rated"


def test_a_session_of_pure_junk_falls_back_to_cold_start(tiny):
    """If nothing survives cleaning the visitor is, in effect, someone who has
    rated nothing — so they should see what that person sees, not an empty
    page and not an error."""
    models = [
        PopularityRecommender().fit(tiny),
        ContentBasedRecommender().fit(tiny),
        CollaborativeRecommender().fit(tiny),
        HybridRecommender(alpha=0.5).fit(tiny),
    ]
    junk = {"GHOST_ID": 5.0, tiny.businesses["business_id"].iloc[0]: None}

    for model in models:
        from_junk = model.recommend_from_ratings(junk, top_n=3)
        from_nothing = model.recommend_from_ratings({}, top_n=3)
        assert not from_junk.empty, f"{model.name} returned an empty list"
        assert from_junk["business_id"].tolist() == from_nothing["business_id"].tolist(), \
            f"{model.name} did not fall back to the cold-start ordering"
