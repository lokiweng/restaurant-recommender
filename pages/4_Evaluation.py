"""
pages/4_Evaluation.py

Step 3, and the answer to "does any of this actually work?"

Three kinds of evaluation live on this screen, which is the set the assignment
asks for:

  * rating accuracy      — RMSE and MSE, computed on held-out ratings
  * ranking quality      — Precision@10, Recall@10, F1@10, plus coverage
  * user satisfaction    — a questionnaire, because it cannot be computed

The first two are measured against data no model was trained on. The third is
collected from whoever is using the app, which is why it sits here rather than
in core/evaluation.py.

WHY THE COMPUTED EVALUATION SITS BEHIND A BUTTON
------------------------------------------------
Running it re-fits four models on a training split and produces a top-10 list
for roughly 1,700 users — about a minute of work. Doing that on page load would
mean a minute of blank screen every time anyone opened this page. The result is
cached once computed, so it is instant for the rest of the session, including
part-way through a live demonstration.
"""

import pandas as pd
import streamlit as st

from core.collaborative import CollaborativeRecommender
from core.content_based import ContentBasedRecommender
from core.evaluation import DEFAULT_K, evaluate_all
from core.hybrid import DEFAULT_ALPHA, HybridRecommender
from core.popularity import PopularityRecommender
from core.satisfaction import (QUESTIONS, SCALE_LABELS, SatisfactionError,
                               aggregate, as_frame, load_responses,
                               record_response)
from ui.charts import (ranking_metrics_chart, rmse_chart, satisfaction_chart,
                       tradeoff_chart)
from ui.components import divider, eyebrow, lede, stat_row
from ui.state import SURVEY_KEY, boot, my_ratings

data, models, _ = boot()
ratings = my_ratings()

eyebrow(st, "Evaluation")
st.markdown("# How well does it work?")
lede(
    st,
    "Every model is fitted on 80% of each user's ratings and then asked about "
    "the 20% it never saw. Nothing below is measured on data a model was "
    "trained on.",
)

divider(st)

# ---------------------------------------------------------------------------
# Step 3 — the questionnaire
#
# Offered only once there are recommendations worth judging, so an answer always
# refers to picks the respondent actually looked at. Answers about an empty
# screen would be noise dressed up as data.
# ---------------------------------------------------------------------------
eyebrow(st, "Step 3 of 3")
st.markdown("### Tell us how we did")

if not ratings:
    st.info(
        "Rate a few restaurants first — the questionnaire asks about your "
        "recommendations, so it needs you to have some.",
        icon=":material/info:",
    )
    if st.button("Go and rate some →", type="primary"):
        st.switch_page("pages/2_Rate.py")

elif st.session_state.get(SURVEY_KEY):
    st.success("Thanks — your response was saved. It appears in the results below.")

else:
    st.caption(
        "Three questions, about the picks you just saw. This is the qualitative "
        "half of the evaluation: RMSE and F1 measure whether the numbers are "
        "right, not whether the recommendations felt right."
    )

    with st.form("satisfaction"):
        answers = {}
        for field, question in QUESTIONS.items():
            answers[field] = st.radio(
                question,
                [1, 2, 3, 4, 5],
                # No default. An untouched control sitting at 3 looks like an
                # answer without being one, and would quietly drag every mean
                # toward the middle.
                index=None,
                horizontal=True,
                format_func=lambda value: SCALE_LABELS.get(value, str(value)),
                key=f"survey_{field}",
            )

        written = st.text_area(
            "Anything that felt off, or missing? (optional)",
            placeholder="Free text — it appears alongside the scores.",
        )
        sent = st.form_submit_button("Submit feedback", type="primary")

    if sent:
        if any(value is None for value in answers.values()):
            st.warning("Please answer all three questions before submitting.")
        else:
            try:
                record_response(
                    relevance=answers["relevance"],
                    discovery=answers["discovery"],
                    intent=answers["intent"],
                    comment=written,
                    n_ratings_given=len(ratings),
                )
                st.session_state[SURVEY_KEY] = True
                st.rerun()
            except (SatisfactionError, OSError) as exc:
                st.error(f"Couldn't save your response: {exc}")

divider(st)


# ---------------------------------------------------------------------------
# The computed evaluation
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_evaluation(_data, k: int) -> pd.DataFrame:
    """Fit and score all four models on one identical split.

    Fresh model instances are built here rather than reusing the fitted ones
    from ui.state: those were fitted on the *entire* dataset, so asking them
    about held-out ratings would be asking about ratings they had already seen.
    That single mistake is the most common way an evaluation ends up reporting
    excellent and meaningless numbers.
    """
    contenders = [
        PopularityRecommender(),
        ContentBasedRecommender(),
        CollaborativeRecommender(),
        HybridRecommender(alpha=DEFAULT_ALPHA),
    ]
    return evaluate_all(contenders, _data, k=k)


st.markdown("### The measured results")

if "evaluation_results" not in st.session_state:
    st.info(
        "The evaluation re-fits every model on a training split and scores it "
        "against held-out ratings. It takes about a minute, and the result is "
        "kept for the rest of this session.",
        icon=":material/timer:",
    )
    if st.button("Run the evaluation", type="primary"):
        with st.spinner("Fitting four models and scoring ~1,700 users…"):
            st.session_state["evaluation_results"] = run_evaluation(data, DEFAULT_K)
        st.rerun()
else:
    results = st.session_state["evaluation_results"]

    best_rmse = results.loc[results["rmse"].idxmin()]
    best_f1 = results.loc[results["f1_at_k"].idxmax()]
    best_coverage = results.loc[results["coverage"].idxmax()]
    best_hit = results.loc[results["hit_rate_at_k"].idxmax()]
    best_personal = results.loc[results["personalisation"].idxmax()]

    # The value stays short and the model name moves into the label. Putting
    # both in the value ("1.074 · Popularity baseline") wraps the headline
    # number onto two lines and loses exactly the glanceability a stat tile is
    # for.
    stat_row(
        st,
        [
            (f"Lowest RMSE — {best_rmse['model']}", f"{best_rmse['rmse']:.3f}"),
            (f"Best F1@{DEFAULT_K} — {best_f1['model']}", f"{best_f1['f1_at_k']:.3f}"),
            (f"Best hit rate — {best_hit['model']}", f"{best_hit['hit_rate_at_k']:.1%}"),
            (f"Most personalised — {best_personal['model']}", f"{best_personal['personalisation']:.2f}"),
        ],
    )

    st.write("")

    # -- rating accuracy ---------------------------------------------------
    st.markdown("#### Rating prediction")
    st.caption(
        "How close each model's predicted star rating came to the real one. "
        "RMSE is in stars, so 1.10 means the average prediction is off by about "
        "one star. Lower is better."
    )
    st.altair_chart(rmse_chart(results), use_container_width=True)

    # -- ranking quality ---------------------------------------------------
    st.markdown(f"#### Ranking quality (top {DEFAULT_K})")
    st.caption(
        f"Of the {DEFAULT_K} restaurants each model recommended, how many the user "
        "actually rated 4★ or above in the held-out data. Precision is the share "
        "of the list that was right; recall is the share of their favourites we "
        "found; F1 balances the two. Each panel has its own scale — recall runs "
        "several times larger than precision, and a shared axis would flatten "
        "precision to nothing."
    )
    st.altair_chart(ranking_metrics_chart(results), use_container_width=True)

    # -- hit rate, NDCG and personalisation --------------------------------
    st.markdown(f"#### Did it help anyone, and how much of the catalogue moved?")
    st.caption(
        f"**Hit rate** is the share of users with at least one relevant restaurant "
        f"in their top {DEFAULT_K} — the user-centric question Precision@{DEFAULT_K} "
        f"cannot answer, and far easier to read than an F1 of 0.03. "
        f"**NDCG@{DEFAULT_K}** asks where in the list the hit landed, since "
        f"precision treats position 1 and position 10 alike. "
        f"**Personalisation** is one minus the average overlap between two users' "
        f"lists: a model giving everybody the same ten restaurants scores 0. "
        f"That is the measure coverage is often mistaken for — coverage counts how "
        f"much of the catalogue circulates in total, which a model could achieve "
        f"while still handing any two people identical lists."
    )

    extra = results[["model", "hit_rate_at_k", "ndcg_at_k", "coverage", "personalisation"]].copy()
    extra.columns = ["Model", f"Hit rate@{DEFAULT_K}", f"NDCG@{DEFAULT_K}",
                     "Coverage", "Personalisation"]
    st.dataframe(
        extra.style.format({
            f"Hit rate@{DEFAULT_K}": "{:.1%}", f"NDCG@{DEFAULT_K}": "{:.4f}",
            "Coverage": "{:.1%}", "Personalisation": "{:.3f}",
        }),
        hide_index=True, width="stretch",
    )

    # -- who could actually be personalised for ----------------------------
    fallback = results[results["n_fallback_users"] > 0]
    if not fallback.empty:
        lines = []
        for _, row in fallback.iterrows():
            total = int(row["n_personalised_users"] + row["n_fallback_users"])
            share = row["n_personalised_users"] / total
            lines.append(
                f"**{row['model']}** personalised for {int(row['n_personalised_users']):,} "
                f"of {total:,} users ({share:.1%}); the remaining "
                f"{int(row['n_fallback_users']):,} had no training rating at 4★ or above "
                f"and received the popularity ordering instead."
            )
        st.info(
            "  \n".join(lines)
            + "  \n\nFor those users the model's row above is measuring the baseline, "
              "not content filtering — which is why the split is reported rather "
              "than left implicit.",
            icon=":material/group:",
        )

    # -- the finding -------------------------------------------------------
    st.markdown("#### Accuracy against coverage")
    st.caption(
        "The two measures plotted together. This is the chart that carries the "
        "central finding, and the reason it is a scatter rather than two more "
        "bar charts: read separately, the numbers hide the relationship between "
        "them."
    )
    st.altair_chart(tradeoff_chart(results), use_container_width=True)

    popularity = results[results["model"] == "Popularity baseline"].iloc[0]
    personalised = results[results["model"] != "Popularity baseline"]

    if popularity["f1_at_k"] >= personalised["f1_at_k"].max():
        st.markdown(
            f"""
The non-personalised baseline scores **higher on every accuracy metric** than
all three personalised models — while recommending just
**{popularity['coverage']:.1%} of the catalogue**, the same handful of
restaurants to everybody. The personalised models cover
**{personalised['coverage'].min():.0%}–{personalised['coverage'].max():.0%}**.

That is not a bug. In this dataset the most-reviewed 3% of restaurants hold
almost a fifth of all ratings, so popular places are disproportionately likely
to appear in anyone's held-out set. A model that simply recommends whatever is
popular therefore scores well on hit-rate metrics while doing nothing a diner
would recognise as a recommendation.

This is the accuracy-versus-coverage trade-off, and it is the main result of
this evaluation: precision alone is the wrong way to judge a recommender on
sparse, popularity-skewed data.
"""
        )

    # -- the underlying table ----------------------------------------------
    with st.expander("The numbers behind the charts"):
        table = results[["model", "rmse", "mse", "precision_at_k", "recall_at_k",
                         "f1_at_k", "hit_rate_at_k", "ndcg_at_k", "coverage",
                         "personalisation", "n_predictions", "n_users_evaluated"]].copy()
        table.columns = ["Model", "RMSE", "MSE", f"Precision@{DEFAULT_K}",
                         f"Recall@{DEFAULT_K}", f"F1@{DEFAULT_K}",
                         f"Hit rate@{DEFAULT_K}", f"NDCG@{DEFAULT_K}", "Coverage",
                         "Personalisation", "Predictions", "Users"]
        st.dataframe(
            table.style.format({
                "RMSE": "{:.3f}", "MSE": "{:.3f}",
                f"Precision@{DEFAULT_K}": "{:.4f}", f"Recall@{DEFAULT_K}": "{:.4f}",
                f"F1@{DEFAULT_K}": "{:.4f}", f"Hit rate@{DEFAULT_K}": "{:.1%}",
                f"NDCG@{DEFAULT_K}": "{:.4f}", "Coverage": "{:.1%}",
                "Personalisation": "{:.3f}",
                "Predictions": "{:,.0f}", "Users": "{:,.0f}",
            }),
            hide_index=True, width="stretch",
        )
        st.caption(
            "Split: 80/20 per user, fixed seed so these reproduce exactly. "
            "An item counts as relevant if its held-out rating was 4★ or above. "
            "`python scripts/run_evaluation.py` prints the same table."
        )

divider(st)

# ---------------------------------------------------------------------------
# Questionnaire results
# ---------------------------------------------------------------------------
st.markdown("### What people said")

try:
    summary = aggregate(load_responses())
except SatisfactionError as exc:
    st.error(f"Couldn't read the questionnaire responses: {exc}")
    summary = None

if summary is None:
    st.markdown(
        '<div class="rr-empty">No questionnaire responses yet.<br>'
        'The three questions above are the qualitative half of the evaluation — '
        'they need real people to answer them.</div>',
        unsafe_allow_html=True,
    )
else:
    stat_row(
        st,
        [
            ("Responses", f"{summary['n_responses']:,}"),
            ("Overall", f"{summary['overall_mean']:.2f} / 5"),
            ("Relevance", f"{summary['relevance_mean']:.2f}"),
            ("Discovery", f"{summary['discovery_mean']:.2f}"),
        ],
    )

    st.altair_chart(satisfaction_chart(as_frame(summary)), use_container_width=True)

    footnote = (
        f"Respondents had rated a median of {summary['median_ratings_given']:.0f} "
        "restaurant(s) before answering."
    )
    if summary["n_discarded"]:
        footnote += f" {summary['n_discarded']} unusable row(s) discarded."
    if summary["n_responses"] < 5:
        footnote += " With this few responses the means are indicative, not conclusive."
    st.caption(footnote)

    if summary["comments"]:
        with st.expander(f"{len(summary['comments'])} written comment(s)"):
            for comment in summary["comments"]:
                st.markdown(f"> {comment}")

    # Download the raw responses.
    #
    # This exists because of where the app may be running rather than because
    # the evaluation needs it. On a hosted free tier the container's filesystem
    # is temporary: data/satisfaction.csv is written correctly, and then thrown
    # away the next time the app sleeps or redeploys. Every response collected
    # from someone who is not sitting at this machine would be lost with it.
    #
    # A download button is the smallest thing that makes that recoverable —
    # collect responses, press this, keep the file. It costs one button and
    # saves the only data in this project that cannot be regenerated by
    # re-running a script.
    try:
        raw = load_responses()
        if not raw.empty:
            st.download_button(
                "Download responses (.csv)",
                data=raw.to_csv(index=False).encode("utf-8"),
                file_name="satisfaction.csv",
                mime="text/csv",
                help="Save the raw questionnaire responses. If this app is hosted "
                     "rather than running on your own machine, download them before "
                     "it sleeps — the hosted filesystem does not keep them.",
            )
    except SatisfactionError:
        pass          # the error above already told the reader what went wrong
