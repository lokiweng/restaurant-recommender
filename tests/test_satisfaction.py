"""
tests/test_satisfaction.py

Tests for the user-satisfaction questionnaire.

Run from the project root:   python -m pytest tests/ -v

WHY THIS NEEDS TESTS AT ALL
---------------------------
It is a form that writes a CSV — the least glamorous code in the project. It is
also the only place where a silent failure would be invisible *and* unrecoverable:
a rejected response is a person's answer gone for good, and a mis-stored one
shifts a number that ends up quoted in a report. Neither announces itself.

Every test below writes to a temporary directory, so the real
data/satisfaction.csv is never touched by the suite.
"""

import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.satisfaction import (QUESTIONS, RESPONSE_COLUMNS, SatisfactionError,
                               aggregate, as_frame, load_responses,
                               record_response)


@pytest.fixture
def path(tmp_path) -> str:
    """A CSV path inside a directory that does not exist yet.

    Deliberately nested: recording a response has to create the folder, which
    is what happens on a fresh checkout where data/ has no satisfaction file.
    """
    return str(tmp_path / "nested" / "satisfaction.csv")


# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------

def test_no_file_yet_reads_as_empty_not_an_error(path):
    """"No responses yet" is a normal state on every fresh install."""
    frame = load_responses(path)
    assert frame.empty
    assert list(frame.columns) == RESPONSE_COLUMNS


def test_aggregate_returns_none_rather_than_zero(path):
    """A satisfaction score of 0.0 and "nobody has answered" are very different
    claims. Returning None forces the caller to say which."""
    assert aggregate(path=path) is None


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

def test_recording_creates_the_directory_and_the_file(path):
    record_response(5, 4, 5, n_ratings_given=3, path=path)
    assert os.path.exists(path)


def test_header_is_written_exactly_once(path):
    """Responses are appended. A header repeated mid-file would be read back as
    a data row and silently corrupt every mean."""
    for _ in range(3):
        record_response(4, 4, 4, path=path)
    assert open(path, encoding="utf-8").read().count("response_id") == 1


def test_each_response_gets_a_unique_id(path):
    ids = {record_response(4, 4, 4, path=path)["response_id"] for _ in range(5)}
    assert len(ids) == 5


def test_comment_whitespace_is_collapsed(path):
    """Newlines inside a comment would break the one-response-per-line shape of
    the CSV."""
    row = record_response(4, 4, 4, comment="  messy\n\ntext   here ", path=path)
    assert row["comment"] == "messy text here"


def test_an_overlong_comment_is_truncated_not_rejected(path):
    """Someone pasting an essay should not lose their scores over it."""
    row = record_response(4, 4, 4, comment="x" * 2000, path=path)
    assert len(row["comment"]) == 500


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [0, 6, -1, 99])
def test_scores_outside_the_scale_are_rejected(bad, path):
    with pytest.raises(SatisfactionError):
        record_response(bad, 3, 3, path=path)


@pytest.mark.parametrize("bad", ["", "x", None, [3]])
def test_non_numeric_scores_are_rejected(bad, path):
    with pytest.raises(SatisfactionError):
        record_response(bad, 3, 3, path=path)


def test_a_fractional_score_is_rejected_not_truncated(path):
    """The subtle one. int(2.7) == 2, so without an explicit check a 2.7 would
    be stored as a *different valid answer* rather than refused — corruption
    that never raises and quietly shifts the mean."""
    with pytest.raises(SatisfactionError):
        record_response(2.7, 3, 3, path=path)


def test_a_rejected_response_writes_nothing(path):
    """Validation must happen before the write, not after."""
    record_response(4, 4, 4, path=path)
    with pytest.raises(SatisfactionError):
        record_response(9, 4, 4, path=path)
    assert len(pd.read_csv(path)) == 1


def test_a_malformed_file_is_reported_by_name(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame([{"unexpected": 1}]).to_csv(path, index=False)
    with pytest.raises(SatisfactionError, match="missing expected column"):
        load_responses(path)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def test_means_are_computed_per_question(path):
    record_response(5, 1, 3, path=path)
    record_response(3, 3, 3, path=path)
    summary = aggregate(path=path)
    assert summary["relevance_mean"] == pytest.approx(4.0)
    assert summary["discovery_mean"] == pytest.approx(2.0)
    assert summary["intent_mean"] == pytest.approx(3.0)


def test_overall_is_the_mean_of_the_three_item_means(path):
    """Stated explicitly so it cannot be mistaken for a fourth measurement."""
    record_response(5, 2, 4, path=path)
    summary = aggregate(path=path)
    expected = sum(summary[f"{f}_mean"] for f in QUESTIONS) / len(QUESTIONS)
    assert summary["overall_mean"] == pytest.approx(expected)


def test_a_single_response_reports_zero_deviation_not_nan(path):
    """pandas gives NaN for the standard deviation of one value, which would
    render on screen as the literal text "nan"."""
    record_response(5, 5, 5, path=path)
    assert aggregate(path=path)["relevance_std"] == 0.0


def test_blank_comments_are_excluded(path):
    record_response(4, 4, 4, comment="useful", path=path)
    record_response(4, 4, 4, comment="", path=path)
    record_response(4, 4, 4, comment="   ", path=path)
    assert aggregate(path=path)["comments"] == ["useful"]


def test_corrupt_rows_are_discarded_and_counted_not_fatal(path):
    """One bad line, appended by hand or by a crash mid-write, must not take
    the whole page down — but the reader has to be told it happened."""
    record_response(5, 5, 5, path=path)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("bad1,2026-08-23T10:00:00,3,notanumber,4,4,junk\n")
        handle.write("bad2,2026-08-23T10:00:00,3,99,4,4,out of range\n")

    summary = aggregate(path=path)
    assert summary["n_responses"] == 1
    assert summary["n_discarded"] == 2


def test_median_ratings_given_is_tracked(path):
    """Stored so the write-up can weight a response from someone who rated
    eight restaurants above one from someone who rated one."""
    for n in [1, 3, 9]:
        record_response(4, 4, 4, n_ratings_given=n, path=path)
    assert aggregate(path=path)["median_ratings_given"] == 3.0


def test_summary_frame_has_one_row_per_question(path):
    record_response(5, 4, 3, path=path)
    frame = as_frame(aggregate(path=path))
    assert len(frame) == len(QUESTIONS)
    assert set(frame.columns) == {"question", "short", "mean", "std"}


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

@pytest.fixture
def results() -> pd.DataFrame:
    """A stand-in evaluation table, so chart tests do not pay the one-minute
    cost of a real evaluation run."""
    return pd.DataFrame([
        {"model": "Popularity baseline", "rmse": 1.074, "mse": 1.154, "n_predictions": 5130,
         "precision_at_k": 0.020, "recall_at_k": 0.102, "f1_at_k": 0.033,
         "coverage": 0.033, "n_users_evaluated": 1699, "k": 10},
        {"model": "Content-based", "rmse": 1.129, "mse": 1.274, "n_predictions": 5130,
         "precision_at_k": 0.005, "recall_at_k": 0.026, "f1_at_k": 0.008,
         "coverage": 0.897, "n_users_evaluated": 1699, "k": 10},
        {"model": "Collaborative filtering", "rmse": 1.115, "mse": 1.242, "n_predictions": 5130,
         "precision_at_k": 0.002, "recall_at_k": 0.017, "f1_at_k": 0.004,
         "coverage": 0.996, "n_users_evaluated": 1699, "k": 10},
        {"model": "Hybrid", "rmse": 1.115, "mse": 1.243, "n_predictions": 5130,
         "precision_at_k": 0.005, "recall_at_k": 0.031, "f1_at_k": 0.009,
         "coverage": 0.947, "n_users_evaluated": 1699, "k": 10},
    ])


def test_every_chart_builds_a_valid_spec(results, path):
    """A malformed encoding raises here rather than rendering as a blank
    rectangle in the middle of a demonstration."""
    from ui.charts import (ranking_metrics_chart, rmse_chart,
                           satisfaction_chart, tradeoff_chart)

    for builder in (rmse_chart, ranking_metrics_chart, tradeoff_chart):
        spec = builder(results).to_dict()
        assert isinstance(spec, dict) and spec

    record_response(5, 4, 4, path=path)
    assert satisfaction_chart(as_frame(aggregate(path=path))).to_dict()


def test_the_baseline_keeps_its_own_colour(results):
    """Colour follows the entity, not its position in a sorted list — the
    baseline must be the same colour on every chart, whatever it scores."""
    from ui.charts import BASELINE_NAME, _with_group

    tagged = _with_group(results)
    assert tagged.loc[tagged["model"] == BASELINE_NAME, "group"].tolist() == ["Baseline"]
    assert set(tagged.loc[tagged["model"] != BASELINE_NAME, "group"]) == {"Personalised"}


def test_the_rmse_chart_is_actually_sorted(results):
    """Vega-Lite silently ignored the sort channel inside a layered chart,
    leaving the bars alphabetical — which is indistinguishable from sorted at a
    glance, so nothing looked wrong. The order is now fixed in the data, and
    this asserts it stays that way."""
    from ui.charts import rmse_chart

    spec = rmse_chart(results).to_dict()
    layer = spec["layer"][0]
    rows = layer["data"]["values"] if "data" in layer else spec["datasets"][
        list(spec["datasets"])[0]
    ]
    order = [row["rmse"] for row in rows]
    assert order == sorted(order), f"RMSE bars are not in ascending order: {order}"


def test_a_stray_text_row_does_not_break_the_summary(path):
    """A duplicated header row in the middle of the file used to crash
    aggregate() with "cannot perform reduction 'median' with string dtype".

    This is not a contrived input. It is what the file looks like after
    somebody opens it in Excel and re-saves it, or after two collectors' copies
    are concatenated. The three scored columns were coerced to numbers on load
    and so survived it; n_ratings_given was not, and one string in that column
    was enough to take the whole Evaluation page down.

    The junk row must be discarded and counted, not crash the page and not
    silently shift the means."""
    header = ",".join(RESPONSE_COLUMNS)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(header + "\n")
        handle.write("id1,2026-08-23T10:00:00,4,5,4,5,\n")
        handle.write(header + "\n")                      # the stray row
        handle.write("id2,2026-08-23T11:00:00,6,3,3,3,\n")

    summary = aggregate(path=path)

    assert summary is not None
    assert summary["n_responses"] == 2                   # the two real answers
    assert summary["n_discarded"] == 1                   # the stray row, counted
    assert summary["median_ratings_given"] == 5.0        # (4 + 6) / 2, a number


def test_median_ratings_given_is_never_nan(path):
    """The page formats this with :.0f, so a NaN would print the word "nan" in
    a sentence about how many restaurants respondents rated."""
    header = ",".join(RESPONSE_COLUMNS)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(header + "\n")
        handle.write("id1,2026-08-23T10:00:00,,5,4,5,\n")   # no count recorded

    summary = aggregate(path=path)

    assert summary is not None
    assert summary["median_ratings_given"] == 0.0
