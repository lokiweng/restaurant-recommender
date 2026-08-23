"""
ui/theme.py

The single source of truth for how the application looks.

WHY A FILE LIKE THIS EXISTS
---------------------------
Streamlit gives you working widgets with a default appearance. Left alone, an
app built from those defaults looks like every other Streamlit app: full-width
text on a wide monitor, inconsistent gaps, and the framework's own toolbar
sitting on top of your work.

Rather than sprinkling one-off styling through every page, all of it is
defined once here as design tokens and injected as a single stylesheet. Two
consequences worth stating in a viva: changing the accent colour is a
one-line edit that updates the entire app, and no page file contains any
styling of its own, so the pages stay readable as *logic*.

THE DESIGN DIRECTION
--------------------
Clean and minimal. Near-white ground, near-black text, generous whitespace,
and exactly one accent colour reserved for actions the user can take. Star
ratings get a separate amber so a rating never visually competes with a
button -- if everything is emphasised, nothing is.

The app commits to a light theme rather than following the operating system.
That is a deliberate choice, not an omission: the palette below is tuned for
a light ground, and a half-working dark mode looks worse in a demo than a
confident light one.
"""

# ---------------------------------------------------------------------------
# DESIGN TOKENS
# Every colour, size and spacing value in the app comes from here. Nothing
# else in the codebase should contain a raw hex code.
# ---------------------------------------------------------------------------

COLORS = {
    "ground":    "#FFFFFF",   # page background
    "surface":   "#F7F7F7",   # cards, inset panels
    "text":      "#222222",   # primary text -- near-black, not pure black
    "muted":     "#717171",   # secondary text, labels, metadata
    "border":    "#DDDDDD",   # hairlines and card edges
    "accent":    "#E8564F",   # ACTIONS ONLY -- buttons, active states, links
    "accent_dk": "#C9443E",   # accent, pressed/hover
    "accent_sf": "#FDEDEC",   # accent at 8% -- tinted backgrounds
    "rating":    "#F5A623",   # stars only, never anything else
    "positive":  "#1B6340",   # "good" states in the evaluation screen
}

# A 4px base scale. Every gap in the app is one of these numbers; nothing is
# improvised. This is what makes spacing look deliberate rather than drifting.
SPACE = {"xs": 4, "sm": 8, "md": 12, "base": 16, "lg": 24, "xl": 32, "xxl": 48}

# Chart colours.
#
# Only two hues are needed, because the charts encode one distinction that
# actually matters: personalised models against the non-personalised baseline.
# Everything else in a chart is identified by a direct label, never by colour
# alone, so no further hues are required.
#
# The pair was checked with a colour-vision validator rather than chosen by eye:
# against a white surface it passes the lightness band, the chroma floor, and
# contrast, and the two are separated by ΔE 13.3 under protanopia and 29.4 under
# normal vision — comfortably above the ΔE 8 threshold at which a red/blue pair
# starts to merge for a colourblind reader.
CHART = {
    "personalised": "#E8564F",   # the app accent — the three real recommenders
    "baseline":     "#1B7FA8",   # the popularity baseline
    "grid":         "#EDEDED",   # recessive — the data should carry the chart
    "axis":         "#717171",
    "label":        "#222222",
}

# A system font stack rather than a web font. Deliberate: it renders instantly,
# identically, and *offline* -- there is no network fetch that can fail halfway
# through a live demonstration on a lab machine.
FONT_STACK = (
    '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, '
    '"Helvetica Neue", Arial, sans-serif'
)

RADIUS = {"card": 12, "input": 8, "pill": 999}

# Content is capped so running text never stretches the full width of a wide
# monitor. Long line lengths are the single most common reason a data app
# feels unpleasant to read.
MAX_CONTENT_WIDTH = 1120


def _css() -> str:
    """Build the stylesheet from the tokens above.

    Written as one f-string so that a change to COLORS or SPACE propagates
    everywhere automatically -- the tokens are the interface, this function is
    just the renderer.
    """
    c, s = COLORS, SPACE

    return f"""
    <style>
    /* ---------------------------------------------------------------
       1. Hide Streamlit's own chrome.
       The hamburger menu, "Deploy" button, coloured top ribbon and
       "Made with Streamlit" footer all announce the framework rather
       than the product. None of them serve the user here.
       --------------------------------------------------------------- */
    #MainMenu, footer, header [data-testid="stToolbar"], [data-testid="stDecoration"] {{
        display: none !important;
    }}
    [data-testid="stHeader"] {{
        background: transparent;
        height: 0;
    }}

    /* ---------------------------------------------------------------
       2. Page frame -- background, typography, content width.
       --------------------------------------------------------------- */
    .stApp {{
        background: {c['ground']};
        font-family: {FONT_STACK};
        color: {c['text']};
    }}

    .block-container, [data-testid="stMainBlockContainer"] {{
        max-width: {MAX_CONTENT_WIDTH}px;
        padding-top: {s['xl']}px;
        padding-bottom: {s['xxl']}px;
    }}

    /* Type scale: 32 / 24 / 19 / 16 / 14 / 12.5.
       Headings sit at 600 rather than 700 -- at this size, bolder reads
       as shouting against so much whitespace. */
    h1, h2, h3, h4, h5 {{
        font-family: {FONT_STACK};
        color: {c['text']};
        font-weight: 600;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }}
    h1 {{ font-size: 32px !important; }}
    h2 {{ font-size: 24px !important; padding-top: {s['lg']}px; }}
    h3 {{ font-size: 19px !important; }}

    p, li, .stMarkdown {{
        font-size: 16px;
        line-height: 1.6;
        color: {c['text']};
    }}

    /* ---------------------------------------------------------------
       3. Sidebar -- this is the app's navigation, so it earns real
          styling rather than being left as a grey slab.
       --------------------------------------------------------------- */
    [data-testid="stSidebar"] {{
        background: {c['ground']};
        border-right: 1px solid {c['border']};
    }}
    [data-testid="stSidebar"] .block-container {{
        padding-top: {s['lg']}px;
    }}
    [data-testid="stSidebarNav"] a {{
        border-radius: {RADIUS['input']}px;
        font-size: 15px;
    }}

    /* ---------------------------------------------------------------
       4. Controls. One accent colour, used only for things the user
          can act on.
       --------------------------------------------------------------- */
    .stButton > button {{
        background: {c['accent']};
        color: #FFFFFF;
        border: none;
        border-radius: {RADIUS['input']}px;
        padding: 10px 20px;
        font-size: 15px;
        font-weight: 600;
        font-family: {FONT_STACK};
        transition: background 140ms ease, transform 140ms ease;
    }}
    .stButton > button:hover {{
        background: {c['accent_dk']};
        color: #FFFFFF;
        transform: translateY(-1px);
    }}
    .stButton > button:focus-visible {{
        outline: 2px solid {c['accent']};
        outline-offset: 2px;
    }}

    /* Secondary buttons: quiet by default, accent only on interaction. */
    .stButton > button[kind="secondary"] {{
        background: {c['ground']};
        color: {c['text']};
        border: 1px solid {c['border']};
    }}
    .stButton > button[kind="secondary"]:hover {{
        border-color: {c['text']};
        background: {c['ground']};
        color: {c['text']};
    }}

    .stTextInput input, .stSelectbox div[data-baseweb="select"] > div {{
        border-radius: {RADIUS['input']}px !important;
        border-color: {c['border']} !important;
        font-size: 15px;
    }}
    .stTextInput input:focus {{
        border-color: {c['text']} !important;
        box-shadow: none !important;
    }}

    /* ---------------------------------------------------------------
       5. The restaurant card grid.
       Rendered as real CSS grid rather than Streamlit columns, because
       columns cannot reflow responsively -- st.columns(3) stays three
       columns on a phone. This collapses 3 -> 2 -> 1 properly.
       --------------------------------------------------------------- */
    .rr-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
        gap: {s['base']}px;
        margin: {s['base']}px 0 {s['lg']}px;
    }}

    .rr-card {{
        background: {c['ground']};
        border: 1px solid {c['border']};
        border-radius: {RADIUS['card']}px;
        padding: {s['base']}px;
        display: flex;
        flex-direction: column;
        gap: {s['sm']}px;
        transition: box-shadow 160ms ease, transform 160ms ease, border-color 160ms ease;
    }}
    .rr-card:hover {{
        border-color: transparent;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.10);
        transform: translateY(-2px);
    }}

    .rr-card-name {{
        font-size: 16.5px;
        font-weight: 600;
        color: {c['text']};
        line-height: 1.3;
        /* Names vary wildly in length; clamping to two lines keeps every
           card in a row the same height without truncating most of them. */
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }}

    .rr-pills {{ display: flex; flex-wrap: wrap; gap: {s['xs']}px; }}
    .rr-pill {{
        font-size: 12.5px;
        color: {c['muted']};
        background: {c['surface']};
        border-radius: {RADIUS['pill']}px;
        padding: 3px 10px;
        white-space: nowrap;
    }}

    .rr-meta {{
        display: flex;
        align-items: center;
        gap: {s['sm']}px;
        font-size: 13.5px;
        color: {c['muted']};
        margin-top: auto;   /* pins the rating row to the bottom of the card */
        padding-top: {s['sm']}px;
    }}
    .rr-stars {{ color: {c['rating']}; font-weight: 600; letter-spacing: -0.5px; }}
    .rr-score {{
        font-size: 12px;
        color: {c['accent']};
        background: {c['accent_sf']};
        border-radius: {RADIUS['pill']}px;
        padding: 3px 9px;
        font-weight: 600;
        margin-left: auto;
        white-space: nowrap;
    }}

    /* ---------------------------------------------------------------
       6. Small shared pieces.
       --------------------------------------------------------------- */
    .rr-eyebrow {{
        font-size: 12.5px;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: {c['muted']};
        margin-bottom: {s['xs']}px;
    }}
    .rr-lede {{
        font-size: 17px;
        color: {c['muted']};
        max-width: 60ch;
        line-height: 1.55;
    }}
    .rr-divider {{
        height: 1px;
        background: {c['border']};
        border: none;
        margin: {s['xl']}px 0;
    }}
    .rr-empty {{
        border: 1px dashed {c['border']};
        border-radius: {RADIUS['card']}px;
        padding: {s['xl']}px;
        text-align: center;
        color: {c['muted']};
        font-size: 15px;
    }}

    /* Respect a reader who has asked the OS to reduce motion. */
    @media (prefers-reduced-motion: reduce) {{
        .rr-card, .stButton > button {{ transition: none; }}
        .rr-card:hover, .stButton > button:hover {{ transform: none; }}
    }}
    </style>
    """


def apply_theme(st) -> None:
    """Inject the stylesheet into the current page.

    Call this once, immediately after st.set_page_config(), on every page.
    `st` is passed in as an argument rather than imported at module level so
    this file stays importable (and testable) without Streamlit running.
    """
    st.markdown(_css(), unsafe_allow_html=True)
