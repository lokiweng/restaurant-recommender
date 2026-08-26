"""
tests/test_theme.py

Guards the one place this project keeps the same value in two files.

WHY THIS FILE EXISTS
--------------------
The palette is defined twice, and it has to be. ui/theme.py holds it as Python
so the stylesheet can be built from tokens; .streamlit/config.toml holds it
because Streamlit reads that file to colour the parts of its own chrome that
injected CSS cannot reach — widget internals, the sidebar frame, focus rings.
Neither file can read the other: config.toml is parsed by Streamlit before any
of this code runs.

Duplication that cannot be removed can still be defended. The comment in
config.toml used to say "if you change a colour, change it in both places",
which is a note asking a human to remember something — exactly the kind of
instruction that is followed until the one time it is not, and then produces a
theme that is subtly wrong in a way nobody notices until a demonstration.

These tests turn that note into a failure. Change one file and forget the
other, and the suite says so by name.
"""

import re
import tomllib
from pathlib import Path

import pytest

from ui.theme import COLORS

CONFIG_PATH = Path(__file__).resolve().parent.parent / ".streamlit" / "config.toml"

#: Which Streamlit theme key mirrors which token in ui/theme.py.
MIRRORED = {
    "primaryColor": "accent",
    "backgroundColor": "ground",
    "secondaryBackgroundColor": "surface",
    "textColor": "text",
}


@pytest.fixture(scope="module")
def config() -> dict:
    if not CONFIG_PATH.exists():
        pytest.skip(f"no config at {CONFIG_PATH}")
    with CONFIG_PATH.open("rb") as handle:
        return tomllib.load(handle)


@pytest.mark.parametrize("streamlit_key,token", MIRRORED.items())
def test_config_matches_the_palette(config, streamlit_key, token):
    """Every colour Streamlit is told about must match the token it mirrors."""
    declared = config["theme"][streamlit_key].upper()
    expected = COLORS[token].upper()
    assert declared == expected, (
        f".streamlit/config.toml sets {streamlit_key} = {declared}, but "
        f"ui/theme.py has COLORS[{token!r}] = {expected}. "
        f"Change both, or the framework's own widgets will not match the app."
    )


def test_the_app_stays_pinned_to_a_light_theme(config):
    """base = "light" is a decision, not a default.

    The palette is tuned for a light ground and there is no dark variant of it.
    Letting the theme follow the operating system would hand roughly half of
    all visitors a half-working dark mode built from light-mode colours.
    """
    assert config["theme"]["base"] == "light"


def test_every_colour_token_is_a_valid_hex_code():
    """A malformed token fails silently in CSS — the rule is simply ignored,
    so the element keeps the browser default and nothing announces the typo."""
    for name, value in COLORS.items():
        assert re.fullmatch(r"#[0-9A-Fa-f]{6}", value), \
            f"COLORS[{name!r}] = {value!r} is not a six-digit hex colour"
