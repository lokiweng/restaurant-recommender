"""
ui/components.py

The reusable visual pieces of the app. Every screen builds from these, which
is what makes six separate pages read as one product rather than six
different assignments stapled together.

WHY THE CARD IS RAW HTML
------------------------
Emitting one block of HTML into a CSS grid (defined in theme.py) gives a
layout that collapses 3 -> 2 -> 1 as the viewport narrows. st.columns() only
reflows at one breakpoint -- Streamlit stacks columns vertically below 640px
and leaves them side by side above it -- so a three-column row stays three
columns at 700px, where 260px cards would already have dropped to two. The
CSS grid is the better default for display-only listings.

The cost is that every value interpolated into that HTML must be escaped --
a restaurant called "Tony & Sons <Pizzeria>" would otherwise break the page.
html.escape() below is doing real work, not ceremony.

WHY THERE ARE TWO GRIDS
-----------------------
Streamlit widgets cannot live inside injected HTML, so a card in the CSS grid
can never be clicked. That is fine for a screen that only displays results,
and not fine for Browse, where the obvious action -- "I have been here, let me
rate it" -- was impossible without retyping the restaurant's name into a
dropdown on another page.

render_card_grid()     display only, one markdown call, fully reflowing.
render_rateable_grid() one Streamlit column per card, so each card can carry
                       a real rating control. Reflows only at 640px.

The trade is deliberate: interactivity is worth one breakpoint on the screen
whose entire purpose is to let someone act on what they are looking at.
"""

import html

import pandas as pd

from ui.theme import COLORS, SPACE

# Rating out of 5, drawn as filled/half/empty stars.
#
# The half star is a full star glyph with its right-hand side clipped in CSS
# (see .rr-half in theme.py), not the fraction character "½". A fraction sits
# on a different baseline and at a different width to "★", so a row reading
# ★★★★½ visibly stumbled at the last character and looked like a rendering
# fault rather than a rating. Every caller of stars() already writes into an
# HTML context, so returning markup here costs nothing.
FULL_STAR, EMPTY_STAR = "★", "☆"
HALF_STAR = '<span class="rr-half">★</span>' 

# How many cards sit side by side in the rateable grid. Three keeps each card
# above the ~260px the display grid treats as a card's minimum comfortable
# width on a 1000px content area.
RATEABLE_COLUMNS = 3


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
    """Yelp stores price as 1-4. Show it the way diners actually read it.

    Falls back to "$$" for a missing value, which is right for a text label in
    a dropdown -- something has to sit between the two dots. It is *not* right
    on a card, where it reads as a fact about the restaurant. See price_pill().
    """
    try:
        level = int(price_range)
    except (TypeError, ValueError):
        return "$$"
    return "$" * max(1, min(4, level))


def price_pill(price_range) -> str:
    """The price as it should appear on a card, or "" when it is unknown.

    Inventing "$$" for a restaurant with no recorded price range tells the
    visitor something the dataset does not know. That is the same mistake as
    printing a 0.00 match score on an unscored card, and it is avoided the same
    way: show nothing rather than something made up.
    """
    try:
        level = int(price_range)
    except (TypeError, ValueError):
        return ""
    return "$" * max(1, min(4, level))


def restaurant_card_html(row: pd.Series, score: float | None = None,
                         score_label: str = "match", reason: str | None = None) -> str:
    """One restaurant, as a card.

    row must contain: name, primary_category, avg_rating, review_count.
    price_range is optional and is simply omitted when absent.

    `score` is shown only when a model actually produced one -- browsing the
    catalogue shows no score, because there is no model opinion to report.
    Displaying a meaningless 0.00 there would be worse than showing nothing.
    """
    name = _escape(row.get("name", "Unnamed restaurant"))
    cuisine = _escape(row.get("primary_category", ""))
    price = price_pill(row.get("price_range"))

    rating_value = row.get("avg_rating")
    rating_text = "—" if pd.isna(rating_value) else f"{float(rating_value):.1f}"

    review_count = row.get("review_count", 0)
    try:
        review_count = int(review_count)
    except (TypeError, ValueError):
        review_count = 0
    review_text = f"{review_count:,} review{'s' if review_count != 1 else ''}"

    pills = f'<span class="rr-pill">{cuisine}</span>' if cuisine else ""
    if price:
        pills += f'<span class="rr-pill">{price}</span>'

    score_html = ""
    if score is not None and not pd.isna(score):
        score_html = f'<span class="rr-score">{_escape(score_label)} {float(score):.2f}</span>'

    # The explanation, when there is one. Omitted entirely rather than shown
    # empty: a card with a blank reason line looks like the text failed to
    # load, and on the Browse screen -- where no model has an opinion -- there
    # is genuinely nothing to say.
    reason_html = ""
    if reason:
        reason_html = f'<div class="rr-reason">{_escape(reason)}</div>'

    return _compact(f"""
    <div class="rr-card">
        <div class="rr-card-name">{name}</div>
        <div class="rr-pills">{pills}</div>
        {reason_html}
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
                     score_label: str = "match", empty_message: str = "Nothing to show yet.",
                     reason_column: str | None = None) -> None:
    """Render a DataFrame of restaurants as a responsive grid of cards.

    Display only -- nothing here can be clicked. Use render_rateable_grid()
    on a screen where acting on a card is the point.

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
        reason = row.get(reason_column) if reason_column and reason_column in frame.columns else None
        cards.append(restaurant_card_html(row, score=score, score_label=score_label, reason=reason))

    st.markdown(f'<div class="rr-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def _rating_control(st, row: pd.Series, ratings: dict, on_rate, on_remove,
                    key_prefix: str) -> None:
    """The five-star control that sits under one card.

    It lives inside a popover rather than on the card face for a reason worth
    stating: a star row on every card, twelve cards to a page, is sixty
    controls competing with the restaurant names for attention. Collapsed, the
    page still reads as a catalogue; expanded, the action is one click away.

    The current rating is shown in the popover's own label, so a visitor can
    see what they have already rated without opening anything.
    """
    business_id = row.get("business_id")
    if business_id is None or on_rate is None:
        return

    current = (ratings or {}).get(business_id)
    label = f"Rated {current}★ — change" if current else "Rate this"

    # Keyed by business_id so Streamlit can tell the widgets apart -- without a
    # unique key every card on the page would share one widget.
    stars_key = f"{key_prefix}_stars_{business_id}"

    # Seed the widget with the existing rating the first time it is built, so
    # reopening the popover shows what was already chosen rather than a blank
    # row. Only on first build: overwriting it on every run would fight the
    # visitor's click, because the click lands in exactly this slot.
    if current and stars_key not in st.session_state:
        st.session_state[stars_key] = current - 1

    with st.popover(label, width="stretch"):
        st.caption(str(row.get("name", "")))

        # st.feedback rather than five st.buttons in five st.columns.
        #
        # The buttons were the obvious construction and the wrong one. This
        # popover sits inside a one-third-width card column, so splitting it
        # five ways left each button about 40px wide -- narrower than the
        # label "1★", which then wrapped onto two lines. st.feedback draws one
        # compact row of star glyphs instead of five bordered boxes, so it
        # fits the width that is actually available.
        #
        # It is 0-indexed: the first star returns 0, so everything crossing
        # into our 1-5 scale is offset by one.
        selected = st.feedback("stars", key=stars_key)

        if selected is not None and (selected + 1) != current:
            on_rate(business_id, selected + 1)
            st.rerun()

        if current and on_remove is not None:
            if st.button("Remove this rating", key=f"{key_prefix}_del_{business_id}",
                         width="stretch"):
                on_remove(business_id)
                # Drop the widget's own state too. Without this the stars stay
                # lit after the rating is gone, and the next rerun would read
                # them back and silently re-add it.
                st.session_state.pop(stars_key, None)
                st.rerun()


def render_rateable_grid(st, frame: pd.DataFrame, *, ratings: dict | None = None,
                         on_rate=None, on_remove=None,
                         columns: int = RATEABLE_COLUMNS,
                         score_column: str | None = None, score_label: str = "match",
                         empty_message: str = "Nothing to show yet.",
                         reason_column: str | None = None,
                         key_prefix: str = "card") -> None:
    """The card grid, with a working rating control under every card.

    Built from st.columns rather than the CSS grid because Streamlit widgets
    cannot be placed inside injected HTML. Each card is still the same
    restaurant_card_html(), so the two grids stay visually identical.

    on_rate and on_remove are passed in rather than imported so this module
    keeps its one useful property: it imports neither streamlit nor anything
    that touches session state, which is why it can be read, reasoned about,
    and tested without starting a server.
    """
    if frame is None or frame.empty:
        st.markdown(f'<div class="rr-empty">{_escape(empty_message)}</div>', unsafe_allow_html=True)
        return

    rows = list(frame.iterrows())

    for start in range(0, len(rows), columns):
        chunk = rows[start:start + columns]
        cells = st.columns(columns, gap="medium")

        for cell, (_, row) in zip(cells, chunk):
            with cell:
                score = row.get(score_column) if score_column and score_column in frame.columns else None
                reason = row.get(reason_column) if reason_column and reason_column in frame.columns else None
                st.markdown(
                    restaurant_card_html(row, score=score, score_label=score_label, reason=reason),
                    unsafe_allow_html=True,
                )
                _rating_control(st, row, ratings, on_rate, on_remove, key_prefix)

        # The CSS grid gets its row gap from theme.py; this one has to say so.
        st.write("")


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


# Type scale for the stat tiles. Previously these three numbers were written
# inline as 27px / 13px / 2px, which made stat_row the one component in the app
# improvising its own values while theme.py insisted nothing was improvised.
STAT_VALUE_SIZE = 27
STAT_LABEL_SIZE = 13


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
                <div style="font-size:{STAT_VALUE_SIZE}px;font-weight:600;color:{COLORS['text']};letter-spacing:-0.02em;font-variant-numeric:tabular-nums;">{_escape(value)}</div>
                <div style="font-size:{STAT_LABEL_SIZE}px;color:{COLORS['muted']};margin-top:{SPACE['xs'] // 2}px;">{_escape(label)}</div>
            </div>
        """))
    st.markdown(
        f'<div style="display:flex;flex-wrap:wrap;gap:{SPACE["lg"]}px;'
        f'margin:{SPACE["sm"]}px 0 {SPACE["xs"]}px;">{"".join(blocks)}</div>',
        unsafe_allow_html=True,
    )
