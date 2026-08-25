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

A NOTE ON HIDING FRAMEWORK CHROME
---------------------------------
Section 1 below removes Streamlit's toolbar, menu and footer. That is safe.
Removing the *header* is not, and the difference is worth understanding
before anyone edits it: the control that reopens a collapsed sidebar lives
inside the header, so flattening the header to zero height takes the sidebar
with it -- one click on the collapse arrow and the navigation is gone until
the page is reloaded. The header is therefore made invisible rather than
absent. See the comment on stHeader for the specific values.
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

# Height reserved for Streamlit's header bar.
#
# Not a free choice. The header is visually empty here -- its toolbar and menu
# are hidden in section 1 -- so the temptation is to set this to 0 and reclaim
# the space. Doing that clips the button that reopens a collapsed sidebar,
# which is rendered inside the header, and the sidebar then cannot be brought
# back. 2rem is the smallest value that still leaves that button clickable.
HEADER_HEIGHT = "2rem"

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

    /* The header stays, at a reduced height, and only its background is
       removed. It is not set to height: 0.

       On a narrow screen the sidebar still collapses (see the rule below), and
       the button that brings it back is rendered inside this header. With the
       header flattened that button has nowhere to sit and is clipped, so the
       navigation cannot be recovered without reloading the page. */
    [data-testid="stHeader"] {{
        background: transparent;
        height: {HEADER_HEIGHT};
    }}

    /* Remove the collapse control on anything wider than a phone.

       The sidebar is this app's navigation, not a panel of optional settings,
       so there is nothing to gain by hiding it on a desktop -- and the arrow
       that hides it has been the single most reliable way for a visitor to
       get stuck. Taking the control away is a smaller loss than the state it
       leads to.

       Below 640px the rule does not apply, deliberately: there the sidebar
       covers the whole page, so being able to close it is the difference
       between a usable app and an unusable one. Streamlit renames this element
       between versions -- collapsedControl before 1.38, stSidebarCollapseButton
       and stExpandSidebarButton after -- so all three are listed. Naming an id
       that does not exist in the running version is harmless; missing the one
       that does is not. */
    @media (min-width: 640px) {{
        [data-testid="collapsedControl"],
        [data-testid="stSidebarCollapseButton"],
        [data-testid="stExpandSidebarButton"] {{
            display: none !important;
        }}
    }}

    /* ---------------------------------------------------------------
       2. Page frame -- background, typography, content width.
       --------------------------------------------------------------- */
    .stApp {{
        background: {c['ground']};
        font-family: {FONT_STACK};
        color: {c['text']};
    }}

    /* padding-top is 'md' rather than 'xl' because the header above now
       contributes its own 2rem. Together they come to roughly the space the
       old xl padding gave on its own. */
    .block-container, [data-testid="stMainBlockContainer"] {{
        max-width: {MAX_CONTENT_WIDTH}px;
        padding-top: {s['md']}px;
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
        /* Never break a label across two lines. Streamlit wraps button text at
           any character when the container is narrow, which turned "1★" into a
           digit stacked above a star inside the rating popover. A button label
           is a name, not a paragraph: if it does not fit, the container is
           wrong, and wrapping only hides that. */
        white-space: nowrap;
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

    /* The popover that carries the star rating on a browse card. Its trigger
       is a button, so without this it inherits the solid accent fill above and
       every card in the grid shouts. Quiet by default, like a secondary. */
    [data-testid="stPopover"] > button {{
        background: {c['ground']};
        color: {c['text']};
        border: 1px solid {c['border']};
        font-weight: 500;
    }}
    [data-testid="stPopover"] > button:hover {{
        background: {c['ground']};
        color: {c['text']};
        border-color: {c['text']};
    }}

    /* Buttons *inside* a popover sit five to a row in a panel that is itself
       only a third of the page wide, so the 20px side padding above leaves
       almost no room for the label. nowrap alone would just overflow; the
       padding has to come down with it. */
    [data-testid="stPopoverBody"] .stButton > button {{
        padding: 8px 4px;
        min-width: 0;
    }}

    /* ---------------------------------------------------------------
       5. The restaurant card grid.

       Rendered as real CSS grid rather than Streamlit columns because it
       reflows at every width: auto-fill with a 260px minimum collapses
       3 -> 2 -> 1 as the viewport narrows. st.columns only reflows once --
       Streamlit stacks columns vertically below 640px and leaves them side
       by side above it -- so a three-column row is still three columns at
       700px, where 260px cards would already have dropped to two.

       The Browse screen uses st.columns anyway, and accepts that, because
       Streamlit widgets cannot be placed inside injected HTML and a card
       there has to carry a working rating control.
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

    /* The explanation line -- why this restaurant was recommended.
       Sits between the cuisine pills and the rating row, indented behind a
       rule so it reads as commentary on the card rather than as another
       attribute of the restaurant. Clamped to three lines: an explanation
       that names a long restaurant name and three tags must not be allowed
       to push the cards in a row out of alignment. */
    .rr-reason {{
        font-size: 13px;
        line-height: 1.45;
        color: {c['muted']};
        border-left: 2px solid {c['accent_sf']};
        padding-left: {s['sm']}px;
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
        overflow: hidden;
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
