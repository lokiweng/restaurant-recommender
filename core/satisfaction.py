"""
core/satisfaction.py

The user-satisfaction questionnaire — the third evaluation component the
assignment requires, alongside rating accuracy (RMSE/MSE) and ranking quality
(Precision/Recall/F1).

WHY THIS ONE IS DIFFERENT
-------------------------
The other two metrics are computed. This one cannot be: no amount of arithmetic
on held-out ratings can tell you whether a person looked at their recommendations
and thought "yes, that's me". It has to be asked.

Collecting it *through the prototype* rather than on a separate form matters for
the same reason the evaluation script exists: every figure quoted in the report
should trace back to a row someone can go and look at. Responses land in
data/satisfaction.csv and are aggregated here.

WHY THREE QUESTIONS AND NOT ONE
-------------------------------
"Was this good?" conflates things a recommender can succeed and fail at
independently. It can be accurate but obvious — every suggestion is somewhere
you'd already thought of. It can be surprising but useless. Asking separately
means the results can say *which*, which is the difference between a finding and
a number.

  relevance  — did the picks match my taste?
  discovery  — did they show me something I wouldn't have found?
  intent     — would I actually use this?

WHY THE SCALE IS ANCHORED
-------------------------
A bare 1-5 means different things to different people. Labelling the endpoints
and the midpoint ("1 — not at all", "3 — somewhat", "5 — very much") makes the
scale mean the same thing to everyone, which shows up as less noise in the
aggregate.
"""

import os
import uuid
from datetime import datetime

import pandas as pd

DEFAULT_PATH = os.path.join("data", "satisfaction.csv")

RESPONSE_COLUMNS = [
    "response_id", "timestamp", "n_ratings_given",
    "relevance", "discovery", "intent", "comment",
]

# One source of truth for the questionnaire. The interface renders these and the
# report quotes them, so the wording cannot drift between what was asked and
# what is written up.
QUESTIONS = {
    "relevance": "How well did these recommendations match your taste?",
    "discovery": "Did they show you somewhere you wouldn't have found on your own?",
    "intent": "How likely would you be to use this to choose where to eat?",
}

SCALE_LABELS = {1: "1 — not at all", 2: "2", 3: "3 — somewhat", 4: "4", 5: "5 — very much"}

SCALE_MIN, SCALE_MAX = 1, 5
MAX_COMMENT_LENGTH = 500


class SatisfactionError(ValueError):
    """Raised when a response does not meet the contract this module promises."""


def _validate_score(value, field: str) -> int:
    """Every Likert answer must be a whole number from 1 to 5.

    Note the explicit check for a fractional value. int(2.7) truncates to 2
    rather than failing, so without it a 2.7 would be silently stored as a
    *different valid answer* instead of being rejected — a corruption that
    never announces itself and quietly shifts every mean computed afterwards.
    A response off the scale is a data problem, not something to round.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise SatisfactionError(f"{field} must be a whole number from 1 to 5, got {value!r}")

    if not number.is_integer():
        raise SatisfactionError(f"{field} must be a whole number from 1 to 5, got {value!r}")

    number = int(number)
    if not SCALE_MIN <= number <= SCALE_MAX:
        raise SatisfactionError(f"{field} must be between {SCALE_MIN} and {SCALE_MAX}, got {number}")
    return number


def record_response(relevance, discovery, intent, comment: str = "",
                    n_ratings_given: int = 0, path: str = DEFAULT_PATH) -> dict:
    """Append one response and return the row that was written.

    n_ratings_given is stored alongside the scores because a response from
    somebody who rated eight restaurants is better-informed than one from
    somebody who rated a single restaurant. Keeping it means the write-up can
    weight or filter on it rather than treating every response as equal.
    """
    row = {
        "response_id": uuid.uuid4().hex[:12],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_ratings_given": max(0, int(n_ratings_given)),
        "relevance": _validate_score(relevance, "relevance"),
        "discovery": _validate_score(discovery, "discovery"),
        "intent": _validate_score(intent, "intent"),
        # Newlines would break the one-response-per-line shape of the CSV and a
        # runaway paste would bloat it. Both are handled at the boundary, once.
        "comment": " ".join(str(comment or "").split())[:MAX_COMMENT_LENGTH],
    }

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    pd.DataFrame([row], columns=RESPONSE_COLUMNS).to_csv(
        path, mode="a", header=write_header, index=False, encoding="utf-8"
    )
    return row


def load_responses(path: str = DEFAULT_PATH) -> pd.DataFrame:
    """Every response collected so far.

    Returns an empty but correctly-shaped frame when nothing has been collected,
    so callers can treat "no file yet" and "no rows yet" identically instead of
    branching on which it is.
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame(columns=RESPONSE_COLUMNS)

    responses = pd.read_csv(path)

    missing = set(RESPONSE_COLUMNS) - set(responses.columns)
    if missing:
        raise SatisfactionError(f"{path} is missing expected column(s): {sorted(missing)}")

    # Every numeric column is coerced here, at the boundary, so that nothing
    # downstream has to wonder whether a column holds numbers or text.
    #
    # n_ratings_given belongs in this list even though no question is asked
    # about it. Leaving it out was a real defect: a file with a stray text row
    # in it -- which is what you get if the CSV is opened in Excel and re-saved,
    # or if two copies are concatenated header and all -- left that one column
    # as strings, and aggregate() then failed on .median() with "cannot perform
    # reduction with string dtype". The three scored columns were already
    # protected; this one was not, and the questionnaire file is precisely the
    # file a human being is most likely to open and re-save by hand.
    for field in (*QUESTIONS, "n_ratings_given"):
        responses[field] = pd.to_numeric(responses[field], errors="coerce")
    return responses


def _safe_median(values: pd.Series) -> float:
    """Median that returns 0.0 rather than NaN when there is nothing to take."""
    median = pd.to_numeric(values, errors="coerce").median()
    return 0.0 if pd.isna(median) else float(median)


def aggregate(responses: pd.DataFrame | None = None, path: str = DEFAULT_PATH) -> dict | None:
    """Summarise the questionnaire for reporting.

    Returns None when there is nothing usable yet — matching core/evaluation.py,
    where a metric that cannot be computed returns None rather than a misleading
    zero. A satisfaction score of 0.0 and "nobody has answered" are very
    different claims.
    """
    if responses is None:
        responses = load_responses(path)

    if responses.empty:
        return None

    # Drop anything unscoreable rather than letting one corrupt row shift the
    # means; how many were dropped is reported, not hidden.
    usable = responses.dropna(subset=list(QUESTIONS))
    for field in QUESTIONS:
        usable = usable[usable[field].between(SCALE_MIN, SCALE_MAX)]

    if usable.empty:
        return None

    summary = {
        "n_responses": int(len(usable)),
        "n_discarded": int(len(responses) - len(usable)),
        # median() skips NaN, but returns NaN itself if every value is missing.
        # The page formats this with :.0f, so an unguarded NaN would print the
        # literal word "nan" in a sentence about how many restaurants people
        # rated. Zero is the honest reading of "no counts recorded".
        "median_ratings_given": _safe_median(usable["n_ratings_given"]),
        "comments": [c for c in usable["comment"].fillna("").tolist() if str(c).strip()],
    }

    for field in QUESTIONS:
        summary[f"{field}_mean"] = float(usable[field].mean())
        # Standard deviation needs at least two responses; pandas returns NaN
        # for one, which would render as "nan" on screen.
        summary[f"{field}_std"] = float(usable[field].std()) if len(usable) > 1 else 0.0

    # A single headline figure for the results table, kept explicitly as the
    # mean of the three item means so it cannot be mistaken for an independent
    # fourth measurement.
    summary["overall_mean"] = float(
        sum(summary[f"{field}_mean"] for field in QUESTIONS) / len(QUESTIONS)
    )
    return summary


def as_frame(summary: dict) -> pd.DataFrame:
    """The per-question summary as a tidy table, for display and for charting."""
    return pd.DataFrame([
        {
            "question": QUESTIONS[field],
            "short": field.capitalize(),
            "mean": round(summary[f"{field}_mean"], 2),
            "std": round(summary[f"{field}_std"], 2),
        }
        for field in QUESTIONS
    ])
