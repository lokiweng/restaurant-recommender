"""
ui/components.py

The reusable visual pieces of the app. Every screen builds from these, which
is what makes six separate pages read as one product rather than six
different assignments stapled together.

WHY THE CARD IS RAW HTML
------------------------
Streamlit's st.columns() cannot reflow: st.columns(3) renders three columns
on a 27-inch monitor and three squeezed columns on a phone. Emitting one
block of HTML into a CSS grid (defined in theme.py) gives a layout that
collapses 3 -> 2 -> 1 properly, which is the "responsive layout" requirement.

The cost is that every value interpolated into that HTML must be escaped --
a restaurant called "Tony & Sons <Pizzeria>" would otherwise break the page.
html.escape() below is doing real work, not ceremony.
"""

import html

import pandas as pd

from ui.theme import COLORS

# Rating out of 5, drawn as filled/half/empty stars.
FULL_STAR, HALF_STAR, EMPTY_STAR = "★", "½", "☆"


def _escape(value) -> str:
    """Make any value safe to drop into HTML. Missing values become ''."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return html.escape(str(value))


def _compact(markup: str) -> str:
    """Flatten generated HTML onto a single line before handing it to Streamlit.

    This is not cosmetic. st.markdown() renders Markdown, and Markdown treats
    any line indented by four or more spaces as a *code block*. Readable,
    indented HTML therefore renders as visible source code on the page instead
    of as a card -- which is exactly what happened the first time this grid was
    rendered. Flattening here means the HTML below can stay indented and
    legible in the source without breaking the output.
    """
    return "".join(line.strip() for line in markup.splitlines())


def stars(rating: float) -> str:
    """Render a 0-5 rating as star glyphs.

    Rounded to the nearest half so the visual matches the number shown beside
    it -- a 4.3 that draws four and a half stars looks like a bug to anyone
    reading carefully.
    """
    if rating is None or pd.isna(rating):
        return EMPTY_STAR * 5

    rounded = round(float(rating) * 2) / 2
    full = int(rounded)
    half = 1 if rounded - full >= 0.5 else 0
    return FULL_STAR * full + HALF_STAR * half + EMPTY_STAR * (5 - full - half)


def price_label(price_range) -> str:
    """Yelp stores price as 1-4. Show it the way diners actually read it."""
    try:
        level = int(price_range)
    except (TypeError, ValueError):
        return "$$"
    return "$" * max(1, min(4, level))


def restaurant_card_html(row: pd.Series, score: float | None = None,
                         score_label: str = "match") -> str:
    """One restaurant, as a card.

    row must contain: name, primary_category, avg_rating, review_count.
    price_range is optional and falls back to a sensible default.

    `score` is shown only when a model actually produced one -- browsing the
    catalogue shows no score, because there is no model opinion to report.
    Displaying a meaningless 0.00 there would be worse than showing nothing.
    """
    name = _escape(row.get("name", "Unnamed restaurant"))
    cuisine = _escape(row.get("primary_category", ""))
    price = price_label(row.get("price_range", 2))

    rating_value = row.get("avg_rating")
    rating_text = "—" if pd.isna(rating_value) else f"{float(rating_value):.1f}"

    review_count = row.get("review_count", 0)
    try:
        review_count = int(review_count)
    except (TypeError, ValueError):
        review_count = 0
    review_text = f"{review_count:,} review{'s' if review_count != 1 else ''}"

    pills = f'<span class="rr-pill">{cuisine}</span>' if cuisine else ""
    pills += f'<span class="rr-pill">{price}</span>'

    score_html = ""
    if score is not None and not pd.isna(score):
        score_html = f'<span class="rr-score">{_escape(score_label)} {float(score):.2f}</span>'

    return _compact(f"""
    <div class="rr-card">
        <div class="rr-card-name">{name}</div>
        <div class="rr-pills">{pills}</div>
        <div class="rr-meta">
            <span class="rr-stars">{stars(rating_value)}</span>
            <span>{rating_text}</span>
            <span>·</span>
            <span>{review_text}</span>
            {score_html}
        </div>
    </div>
    """)


def render_card_grid(st, frame: pd.DataFrame, score_column: str | None = None,
                     score_label: str = "match", empty_message: str = "Nothing to show yet.") -> None:
    """Render a DataFrame of restaurants as a responsive grid of cards.

    The whole grid is emitted as a single markdown call. Rendering one card
    per call would work, but Streamlit inserts a wrapper element around each
    one, which breaks the CSS grid -- a good example of why the HTML is built
    up first and written out once.
    """
    if frame is None or frame.empty:
        st.markdown(f'<div class="rr-empty">{_escape(empty_message)}</div>', unsafe_allow_html=True)
        return

    cards = []
    for _, row in frame.iterrows():
        score = row.get(score_column) if score_column and score_column in frame.columns else None
        cards.append(restaurant_card_html(row, score=score, score_label=score_label))

    st.markdown(f'<div class="rr-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def eyebrow(st, text: str) -> None:
    """A small uppercase label above a heading. Used to name a section's role
    ("STEP 1", "BASELINE") where the heading itself names its content."""
    st.markdown(f'<div class="rr-eyebrow">{_escape(text)}</div>', unsafe_allow_html=True)


def lede(st, text: str) -> None:
    """A single line of supporting copy under a heading, held to ~60
    characters per line so it stays readable on a wide screen."""
    st.markdown(f'<div class="rr-lede">{_escape(text)}</div>', unsafe_allow_html=True)


def divider(st) -> None:
    """A hairline rule. Uses the border token so it matches every card edge."""
    st.markdown('<hr class="rr-divider" />', unsafe_allow_html=True)


def stat_row(st, stats: list[tuple[str, str]]) -> None:
    """A row of headline numbers -- value large, label small beneath.

    Takes (label, value) pairs. Used on the About and Evaluation screens where
    the summary should be readable before any detail.
    """
    if not stats:
        return

    blocks = []
    for label, value in stats:
        blocks.append(_compact(f"""
            <div style="flex:1 1 130px;">
                <div style="font-size:27px;font-weight:600;color:{COLORS['text']};letter-spacing:-0.02em;font-variant-numeric:tabular-nums;">{_escape(value)}</div>
                <div style="font-size:13px;color:{COLORS['muted']};margin-top:2px;">{_escape(label)}</div>
            </div>
        """))
    st.markdown(
        f'<div style="display:flex;flex-wrap:wrap;gap:24px;margin:8px 0 4px;">{"".join(blocks)}</div>',
        unsafe_allow_html=True,
    )
