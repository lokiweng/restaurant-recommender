"""
pages/5_About.py

How the whole thing works, in plain language.

This screen has two audiences and serves both with the same text: a visitor
wondering where the recommendations come from, and the person who has to
explain every part of this project out loud in a viva. Everything stated here
is checkable against the code in core/.
"""

import streamlit as st

from ui.components import divider, eyebrow, lede, stat_row
from ui.state import boot

data, models, warnings = boot()

eyebrow(st, "Under the hood")
st.markdown("# How it works")
lede(
    st,
    "Four recommendation approaches, three of them personalised, all fitted on "
    "real Yelp data and measured against ratings they were never shown.",
)

divider(st)

# ---------------------------------------------------------------------------
# The data
# ---------------------------------------------------------------------------
st.markdown("### The data")

stat_row(
    st,
    [
        ("Restaurants", f"{data.n_businesses:,}"),
        ("Diners", f"{data.n_users:,}"),
        ("Ratings", f"{data.n_reviews:,}"),
        ("City", data.city),
    ],
)

st.markdown(
    """
Everything comes from the **Yelp Open Dataset**, a public release of real
business and review records. Two filters were applied to it:

1. **One city, restaurants only.** The full dataset spans several gigabytes
   across many cities. Narrowing to Cleveland restaurants keeps it small enough
   to work with while leaving the ratings genuinely dense enough to learn from.

2. **A 5-core filter.** Only users and restaurants with at least five ratings
   each are kept. Real review data is extremely sparse — most people review one
   or two places in their lifetime — and a user with a single rating offers no
   signal to personalise from and no way to evaluate whether a recommendation
   was any good. Reducing to a 5-core is the standard preprocessing step for
   exactly this reason.

No ratings were invented, and nothing on any screen is simulated. Ratings you
give during a session live only in your browser and are never written back to
the source files.
"""
)

if warnings:
    with st.expander(f"{len(warnings)} data quality note(s) found by validation"):
        for warning in warnings:
            st.write(f"- {warning}")
else:
    st.caption("✓ The dataset currently passes every validation check.")

divider(st)

# ---------------------------------------------------------------------------
# The models
# ---------------------------------------------------------------------------
st.markdown("### The four approaches")
st.caption("Each is a real implementation in `core/`, and each is evaluated on identical data.")

for key in ["popularity", "content", "collaborative", "hybrid"]:
    model = models[key]
    with st.container(border=True):
        st.markdown(f"**{model.name}**")
        st.caption(model.description)

st.markdown(
    """
The two personalised models fail in **opposite** situations, which is the whole
reason a hybrid is worth building:

- Collaborative filtering cannot say anything about a restaurant nobody has
  rated yet. Content-based filtering can, because a brand-new Thai place still
  has category tags.
- Content-based filtering can only ever offer more of what you already like. It
  has no way to discover that people with your taste also love somewhere with
  completely unrelated tags. Collaborative filtering finds exactly that.
"""
)

divider(st)

# ---------------------------------------------------------------------------
# How it is measured
# ---------------------------------------------------------------------------
st.markdown("### How it is measured")

st.markdown(
    """
Three kinds of evaluation, because they answer different questions:

| Question | Metric |
|---|---|
| How close is a predicted star rating to the real one? | RMSE, MSE |
| Are the right restaurants in the top ten? | Precision@10, Recall@10, F1@10 |
| How much of the catalogue ever gets recommended? | Coverage |
| Did people actually like the results? | Satisfaction questionnaire |

The first three are computed on a held-out split: each user's ratings are cut
80/20, every model learns from the 80% and is scored on the 20% it never saw.
The split is per user rather than at random across the whole table, so no user
is ever tested without a history to learn from.

The last one cannot be computed at all. It comes from real people using the
app, which is why the questionnaire exists.
"""
)

divider(st)

# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------
st.markdown("### The pipeline, end to end")

st.markdown(
    """
```
Yelp Open Dataset (business + review JSON)
        ↓  filter to one city, restaurants only, 5-core
data/businesses.csv · users.csv · reviews.csv
        ↓  schema, rating ranges, duplicates, orphan IDs
core/validation.py
        ↓  TF-IDF over category tags + price  |  user × item rating matrix
core/content_based.py · core/collaborative.py
        ↓  weighted blend
core/hybrid.py                       core/popularity.py  (baseline)
        ↓
this app  →  core/evaluation.py (quantitative)
          →  the questionnaire  (qualitative)
```
"""
)

st.caption(
    "Every stage above corresponds to a real file. `python scripts/run_evaluation.py` "
    "reproduces the reported numbers from the command line, and "
    "`python -m pytest tests/` runs the test suite."
)
