"""
core/data.py

Loads the three CSV files that make up the dataset and bundles them into one
object the rest of the application passes around.

WHY THIS FILE HAS NO STREAMLIT IMPORTS
--------------------------------------
Everything under core/ is plain Python: it can be imported by the Streamlit
app, by the command-line evaluation script, and by the tests without any of
them needing a running web server. Caching is a *presentation* concern (it
depends on how often a page re-runs), so the @st.cache_data wrapper lives in
the UI layer instead. Keeping the boundary that way is what lets tests/ run
in a fraction of a second.

WHERE THE DATA CAME FROM
------------------------
The Yelp Open Dataset (business + review JSON), filtered down to restaurants
in Cleveland, Ohio, then reduced to a "5-core": only users and restaurants
with at least 5 ratings each are kept. Real review data is extremely sparse
-- most Yelp users have reviewed one or two places in their lifetime -- and
5-core filtering is the standard preprocessing step that leaves enough signal
per user to actually measure whether a recommendation is any good.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# The project's data directory, resolved relative to this file rather than to
# whatever directory the user happened to launch the app from. Hard-coding an
# absolute path here is the classic bug that makes a project run on one
# machine and fail on every other.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

BUSINESSES_CSV = "businesses.csv"
USERS_CSV = "users.csv"
REVIEWS_CSV = "reviews.csv"


@dataclass(frozen=True)
class Dataset:
    """The three tables, loaded once and passed around together.

    Frozen (immutable) on purpose: a recommender that silently mutates its own
    training data as a side effect of rendering a page is impossible to reason
    about. Session ratings from the live user are kept separately, in memory,
    and never written back into these frames.
    """

    businesses: pd.DataFrame   # one row per restaurant
    users: pd.DataFrame        # one row per reviewer (just the id)
    reviews: pd.DataFrame      # one row per rating: who rated what, and how

    # -- Convenience accessors, so pages don't reach into the frames directly --

    @property
    def n_businesses(self) -> int:
        return len(self.businesses)

    @property
    def n_users(self) -> int:
        return len(self.users)

    @property
    def n_reviews(self) -> int:
        return len(self.reviews)

    @property
    def city(self) -> str:
        """The single city this dataset covers (all rows share one value)."""
        if "city" in self.businesses.columns and not self.businesses.empty:
            return str(self.businesses["city"].iloc[0])
        return "Unknown"

    def business_by_id(self, business_id: str) -> pd.Series | None:
        """One restaurant's row, or None if the id isn't in the catalogue."""
        match = self.businesses.loc[self.businesses["business_id"] == business_id]
        return None if match.empty else match.iloc[0]


class DatasetError(RuntimeError):
    """Raised when the data directory is missing or a file cannot be read.

    A distinct exception type (rather than a bare FileNotFoundError) lets the
    app catch *this* and show the user a helpful message, while genuine
    programming errors still surface as crashes during development.
    """


def load_dataset(data_dir: Path | str = DATA_DIR) -> Dataset:
    """Read the three CSVs from disk and return them as one Dataset.

    Raises DatasetError with a readable message if anything is missing, rather
    than letting a pandas stack trace reach the screen.
    """
    directory = Path(data_dir)

    if not directory.is_dir():
        raise DatasetError(
            f"Data folder not found at {directory}. "
            "The three CSV files must sit in a 'data' folder next to the app."
        )

    frames = {}
    for key, filename in (
        ("businesses", BUSINESSES_CSV),
        ("users", USERS_CSV),
        ("reviews", REVIEWS_CSV),
    ):
        path = directory / filename
        if not path.is_file():
            raise DatasetError(f"Missing required data file: {path.name} (looked in {directory})")
        try:
            frames[key] = pd.read_csv(path)
        except Exception as exc:  # malformed CSV, encoding problem, empty file
            raise DatasetError(f"Could not read {path.name}: {exc}") from exc

    return Dataset(
        businesses=frames["businesses"],
        users=frames["users"],
        reviews=frames["reviews"],
    )
