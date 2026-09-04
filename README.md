# Cleveland Eats — Restaurant Recommender System

A restaurant recommender built for the BMCS2074 Artificial Intelligence
assignment. Four recommendation approaches, three of them personalised, fitted
on **real Yelp Open Dataset data** and measured against ratings they were never
shown.

**817 restaurants · 2,112 diners · 26,096 ratings · Cleveland, Ohio**

---

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

If `streamlit` is not on your PATH (common on Windows, where pip's `Scripts\`
folder often isn't), use the module form instead — it works either way:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app opens at <http://localhost:8501>. On first launch Streamlit asks for an
email address; press Enter to skip.

Two other entry points, neither of which needs a browser:

```bash
python scripts/run_evaluation.py      # prints the evaluation table
python -m pytest tests/               # runs the test suite
```

---

## What it does

The interface is built around one two-step journey, because the most common
complaint about a recommender demo is not knowing what to do first:

| Step | Screen | What happens |
|---|---|---|
| 1 | **Browse restaurants** / **Your ratings** | You rate restaurants you already know — in place on the catalogue cards, or by name. |
| 2 | **Recommended for you** | Recommendations built from those ratings, plus a side-by-side comparison of all four models. |

Two further screens sit outside that journey, under *Behind the system* in the
sidebar. **How well does it work?** carries the measured results and the
satisfaction questionnaire; **How it works** explains the four models. They are
addressed to whoever is assessing the project rather than to someone deciding
where to eat, which is why neither is numbered as a third step.

Before you rate anything the app shows the popularity baseline and says so.
That is the **cold-start problem**, shown honestly rather than disguised as
personalisation.

---

## The four approaches

| Model | File | How it decides |
|---|---|---|
| **Popularity baseline** | `core/popularity.py` | Bayesian-weighted rating that discounts restaurants with few reviews. Identical for everyone — the thing the other three have to beat. |
| **Content-based** | `core/content_based.py` | TF-IDF over category tags plus price; recommends restaurants similar to what you rated highly. |
| **Collaborative filtering** | `core/collaborative.py` | Item-based cosine similarity on the user × item rating matrix; ignores cuisine entirely and looks only at behaviour. |
| **Hybrid** | `core/hybrid.py` | Weighted blend: `α × collaborative + (1 − α) × content-based`. |

All four implement the same interface (`core/base.py`): `predict()` returns a
rating on the **same 1–5 scale** for every model, and `score_all()` returns a
ranking. That contract is what lets one evaluation harness measure all four
without knowing anything about their internals.

---

## Project layout

Three layers that never reach across each other — `core/` knows nothing about
Streamlit, `ui/` knows nothing about algorithms, `pages/` composes the two.
That separation is why the test suite runs in seconds without starting a server.

```
app.py                  entry point / router
pages/                  one file per screen
  0_Discover.py  1_Browse.py  2_Rate.py
  3_Recommendations.py  4_Evaluation.py  5_About.py
core/                   pure logic — no Streamlit imports anywhere
  data.py               loading the three CSVs
  validation.py         schema, ranges, duplicates, orphan IDs
  base.py               the shared Recommender contract
  content_based.py  collaborative.py  hybrid.py  popularity.py
  evaluation.py         RMSE · MSE · Precision/Recall/F1@K · coverage
  satisfaction.py       the questionnaire
ui/                     presentation only
  theme.py              every design token, one stylesheet
  components.py         cards, stars, stat tiles
  charts.py             Altair chart specs
  state.py              cached data + fitted models, session ratings
data/                   the dataset (see note below)
scripts/                run_evaluation.py · popularity_stratification.py
                        make_flowchart.py · make_results_chart.py
tests/                  test_core.py · test_satisfaction.py · test_theme.py
```

---

## The data

`data/businesses.csv`, `users.csv` and `reviews.csv` come from the **Yelp Open
Dataset**, filtered by `preprocess_yelp.py` to restaurants in one city and then
reduced to a **5-core** — only users and restaurants with at least five ratings
each. Real review data is extremely sparse; without that filter most users have
one or two ratings, which is too little to personalise from or to evaluate
against.

> **These three files cannot be regenerated from what is in this folder.** The
> multi-gigabyte raw Yelp JSON is not included, so `preprocess_yelp.py` is
> shipped for inspection rather than for re-running: it documents exactly how
> the city filter, the `Restaurants` category test and the 5-core pass were
> applied. Treat the three CSVs as source data.

Ratings you give while using the app live in the browser session only and are
never written back. `data/satisfaction.csv` is the one file the app writes, and
it is created on the first questionnaire response.

---

## How it is evaluated

Three kinds, because they answer different questions:

| Question | Metric |
|---|---|
| How close is a predicted star rating to the real one? | RMSE, MSE |
| Are the right restaurants in the top ten? | Precision@10, Recall@10, F1@10 |
| Did any user get at least one good pick, and how high up? | Hit Rate@10, NDCG@10 |
| How much of the catalogue ever gets recommended? | Coverage |
| Do two users actually get different lists? | Personalisation index |
| Did people actually like the results? | Satisfaction questionnaire |

The first three use a **per-user 80/20 split**: each user's ratings are cut
individually, so no user is ever tested without a history to learn from. A
purely random split across the whole table would leave some users with no
training data at all, and a model cannot fairly be blamed for failing a user it
was never shown. The seed is fixed, so the numbers reproduce exactly.

### The main finding

The non-personalised baseline scores **higher on every accuracy metric** than
all three personalised models — while recommending **3.3% of the catalogue**,
the same handful of restaurants to everybody. The personalised models cover
**89–100%**.

That is not a bug. The most-reviewed 3% of restaurants hold almost a fifth of
all ratings, so popular places are disproportionately likely to appear in
anyone's held-out set. A model that recommends whatever is popular therefore
scores well on hit-rate metrics while doing nothing a diner would recognise as
a recommendation.

This is the accuracy-versus-coverage trade-off, and it is why coverage is
reported alongside precision rather than instead of it.

---

## Regenerating the report figures

```bash
python scripts/make_flowchart.py        # Figure 1 — system architecture
python scripts/make_results_chart.py    # Figure 2 — evaluation results
```

`make_results_chart.py` re-runs the evaluation rather than plotting stored
numbers, so the figure in the report cannot drift out of step with the code.
