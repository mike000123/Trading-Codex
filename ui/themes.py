"""
ui/themes.py
────────────
Predefined UI themes injected via st.markdown CSS.

Usage:
    from ui.themes import apply_theme, THEME_NAMES
    apply_theme(st.session_state.get("theme", "Dark"))
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

# ─── Theme definitions ────────────────────────────────────────────────────────
# Each theme is a dict of CSS variable overrides applied to :root.
# Extend freely; the apply_theme() function picks the active one.

_THEMES: dict[str, dict[str, str]] = {
    "MRMI Gold": {
        "--bg-primary":     "#0c0d14",
        "--bg-secondary":   "#13141d",
        "--bg-card":        "#181a25",
        "--text-primary":   "#f0e8d0",
        "--text-secondary": "#a89c80",
        "--accent":         "#d4af37",
        "--accent-hover":   "#f5c93b",
        "--positive":       "#e2c25b",
        "--negative":       "#c64242",
        "--border":         "#3a2f1a",
        "--chart-grid":     "#1c1d28",
        "--font":           "'Inter', sans-serif",
        "--brand-font":     "'Cinzel', 'Playfair Display', serif",
        "--radius":         "10px",
    },
    "MRMI Silver": {
        "--bg-primary":     "#0c0d14",
        "--bg-secondary":   "#13141d",
        "--bg-card":        "#181a25",
        "--text-primary":   "#eef0f5",
        "--text-secondary": "#9aa5b3",
        "--accent":         "#c0c0c8",
        "--accent-hover":   "#e8e8ee",
        "--positive":       "#c8d0d8",
        "--negative":       "#c64242",
        "--border":         "#2a2d3a",
        "--chart-grid":     "#1c1d28",
        "--font":           "'Inter', sans-serif",
        "--brand-font":     "'Cinzel', 'Playfair Display', serif",
        "--radius":         "10px",
    },
    "Dark": {
        "--bg-primary": "#0e1117",
        "--bg-secondary": "#1a1d27",
        "--bg-card": "#1e2130",
        "--text-primary": "#e8eaf6",
        "--text-secondary": "#9e9eb8",
        "--accent": "#7c83fd",
        "--accent-hover": "#6268e8",
        "--positive": "#26a69a",
        "--negative": "#ef5350",
        "--border": "#2a2d3e",
        "--chart-grid": "#252837",
        "--font": "'Inter', sans-serif",
        "--radius": "8px",
    },
    "Midnight Blue": {
        "--bg-primary": "#050d1f",
        "--bg-secondary": "#071428",
        "--bg-card": "#0a1a35",
        "--text-primary": "#c9d8f5",
        "--text-secondary": "#7a9ac0",
        "--accent": "#4a9eff",
        "--accent-hover": "#2e8be8",
        "--positive": "#43e97b",
        "--negative": "#fa5252",
        "--border": "#112244",
        "--chart-grid": "#0d1e3a",
        "--font": "'Inter', sans-serif",
        "--radius": "10px",
    },
}

THEME_NAMES: list[str] = list(_THEMES.keys())


def current_theme_name(default: str = "Dark") -> str:
    """Return the active theme name from session state."""
    return st.session_state.get("theme_selector") or st.session_state.get("theme") or default


def themed_dataframe_style(
    df: pd.DataFrame | None,
    *,
    theme_name: str | None = None,
    hide_index: bool = False,
):
    """Return a reusable dataframe styler aligned with the active app theme."""
    active_name = theme_name or current_theme_name()
    theme = _THEMES.get(active_name, _THEMES["Dark"])
    safe_df = (df if df is not None else pd.DataFrame()).copy()
    bg_primary = theme.get("--bg-primary", "#0e1117")
    bg_secondary = theme.get("--bg-secondary", "#1a1d27")
    bg_card = theme.get("--bg-card", "#1e2130")
    text_primary = theme.get("--text-primary", "#e8eaf6")
    text_secondary = theme.get("--text-secondary", "#9e9eb8")
    accent = theme.get("--accent", "#7c83fd")
    border = theme.get("--border", "#2a2d3e")
    row_even = bg_card
    row_odd = bg_secondary if bg_secondary != bg_card else bg_primary

    styler = safe_df.style.format(na_rep="—")

    def _row_style(row):
        bg = row_even if int(getattr(row, "name", 0)) % 2 == 0 else row_odd
        return [
            f"background-color: {bg}; color: {text_primary}; border-color: {border};"
            for _ in row
        ]

    styler = styler.apply(_row_style, axis=1).set_table_styles(
        [
            {
                "selector": "table",
                "props": [
                    ("background-color", bg_card),
                    ("color", text_primary),
                    ("border", f"1px solid {border}"),
                    ("border-collapse", "collapse"),
                ],
            },
            {
                "selector": "thead th",
                "props": [
                    ("background-color", bg_secondary),
                    ("color", accent),
                    ("border", f"1px solid {border}"),
                    ("font-weight", "600"),
                ],
            },
            {
                "selector": "tbody td",
                "props": [
                    ("border", f"1px solid {border}"),
                    ("color", text_primary),
                ],
            },
            {
                "selector": "tbody th",
                "props": [
                    ("background-color", bg_secondary),
                    ("color", text_secondary),
                    ("border", f"1px solid {border}"),
                ],
            },
        ],
        overwrite=False,
    )
    if hide_index:
        try:
            styler = styler.hide(axis="index")
        except Exception:
            pass
    return styler


def apply_theme(theme_name: str) -> None:
    """Inject CSS variables for the selected theme into the Streamlit page."""
    theme = _THEMES.get(theme_name, _THEMES["Dark"])

    vars_css = "\n".join(f"    {k}: {v};" for k, v in theme.items())

    # MRMI Gold gets a few extras on top of the variable system: serif
    # branding font, gold metric values, gold-tinted card borders, a soft
    # metallic gradient on h1/h2 headings, and a Cinzel font import.
    is_mrmi = theme_name in ("MRMI Gold", "MRMI Silver")
    mrmi_extras = ""
    if is_mrmi:
        mrmi_extras = """
/* Brand wordmark — gold metallic gradient on h1 / h2 / page title */
h1, h2,
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stHeader"] h1 {
    font-family: var(--brand-font) !important;
    background: linear-gradient(180deg, #f5e6a8 0%, #d4af37 45%, #b9871e 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    color: transparent !important;
    letter-spacing: 0.06em;
    font-weight: 700 !important;
    text-shadow: 0 1px 0 rgba(0,0,0,0.55);
}

/* Subtler gold for h3 / h4 — keep them readable, not flashy */
h3, h4,
[data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] h4 {
    color: #e8c566 !important;
    font-family: var(--brand-font) !important;
    letter-spacing: 0.03em;
}

/* Sidebar nav title */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--accent) !important;
    font-family: var(--brand-font) !important;
}

/* Metric values — gold; labels stay muted */
[data-testid="stMetricValue"] {
    color: var(--accent) !important;
    font-family: var(--brand-font) !important;
    font-weight: 600 !important;
}

/* Card borders gain a faint gold inner tint */
[data-testid="stExpander"],
[data-testid="stForm"],
[data-testid="stMetric"],
div[data-baseweb="card"] {
    border: 1px solid rgba(212, 175, 55, 0.22) !important;
    box-shadow: 0 0 0 1px rgba(212, 175, 55, 0.05) inset;
}

/* Primary buttons — gold on charcoal */
.stButton > button[kind="primary"] {
    background: linear-gradient(180deg, #e8c566 0%, #d4af37 60%, #b9871e 100%) !important;
    color: #1a1306 !important;
    border: 1px solid #b9871e !important;
    font-weight: 700 !important;
    letter-spacing: 0.03em;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(180deg, #f5d77a 0%, #f5c93b 60%, #d4af37 100%) !important;
}

/* Tabs — active tab underline in gold */
button[role="tab"][aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* Tooltip ⓘ stays gold here too */
[data-testid="stTooltipIcon"] svg,
[data-testid="stTooltipHoverTarget"] svg {
    color: #c9a227 !important;
    fill:  #c9a227 !important;
}

/* Force the main app background dark — Streamlit sometimes leaves the
   outer container at the default theme colour. */
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > section,
section.main,
.main {
    background-color: #0c0d14 !important;
}

/* Top header — keep dark with a thin gold underline. */
[data-testid="stHeader"] {
    background: #0c0d14 !important;
    border-bottom: 1px solid #3a2f1a !important;
}

/* Sidebar nav radio bullets — replace bright white with gold. */
[data-testid="stSidebar"] [data-baseweb="radio"] [aria-checked="true"] [data-baseweb="circle"],
[data-testid="stSidebar"] [data-baseweb="radio"] div[role="radio"][aria-checked="true"] svg,
[data-testid="stSidebar"] [role="radiogroup"] [data-baseweb="radio"] [aria-checked="true"] *,
[data-testid="stSidebar"] [role="radio"][aria-checked="true"] {
    background: #d4af37 !important;
    border-color: #d4af37 !important;
    color: #d4af37 !important;
}
[data-testid="stSidebar"] [data-baseweb="radio"] [data-baseweb="circle"] {
    border: 1.5px solid rgba(212,175,55,0.6) !important;
    background: transparent !important;
}
[data-testid="stSidebar"] [role="radiogroup"] label {
    color: #f0e8d0 !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    color: #f5c93b !important;
}

/* Info / warning / success / error alert boxes — dark with gold accent. */
[data-testid="stAlert"],
div[role="alert"],
[data-baseweb="notification"] {
    background-color: #14161e !important;
    border: 1px solid #3a2f1a !important;
    border-left: 4px solid #d4af37 !important;
    color: #f0e8d0 !important;
    border-radius: 8px !important;
}
[data-testid="stAlert"] *,
div[role="alert"] *,
[data-baseweb="notification"] * {
    color: #f0e8d0 !important;
}
[data-testid="stAlert"] strong,
div[role="alert"] strong,
[data-baseweb="notification"] strong {
    color: #e8c566 !important;
}

/* Per-kind left borders. */
[data-testid="stAlert"][kind="warning"] { border-left-color: #f5c93b !important; }
[data-testid="stAlert"][kind="error"]   { border-left-color: #c64242 !important; }
[data-testid="stAlert"][kind="success"] { border-left-color: #2faa6a !important; }

/* Kill switch + other secondary buttons */
.stButton > button[kind="secondary"],
.stButton > button:not([kind="primary"]) {
    background: #14161e !important;
    color: #f0e8d0 !important;
    border: 1px solid rgba(212,175,55,0.35) !important;
}
.stButton > button[kind="secondary"]:hover,
.stButton > button:not([kind="primary"]):hover {
    border-color: #d4af37 !important;
    color: #f5c93b !important;
}

/* Checkbox toggle in gold when checked */
[data-baseweb="checkbox"] [aria-checked="true"] {
    background: #d4af37 !important;
    border-color: #d4af37 !important;
}
[data-baseweb="checkbox"] [aria-checked="true"] svg {
    fill: #1a1306 !important;
}

/* Selectbox / number-input inner backgrounds — keep them dark. */
[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background-color: #14161e !important;
    color: #f0e8d0 !important;
    border-color: rgba(212,175,55,0.25) !important;
}

/* Code blocks / inline code — warm amber on charcoal */
code, pre {
    background-color: #14161e !important;
    color: #e8c566 !important;
    border: 1px solid rgba(212,175,55,0.18) !important;
}

/* Dividers in gold */
hr { border-color: rgba(212,175,55,0.30) !important; }

/* Dataframe / table — header bar AND row body, so big tables don't
   render as a white card on the dark page. Streamlit's data-grid uses a
   shadow DOM in some versions; we paint both the wrapper iframe and the
   inner cells. */
[data-testid="stDataFrame"],
[data-testid="stDataFrame"] iframe,
[data-testid="stDataFrame"] [data-testid="stDataFrameContainer"] {
    background-color: #181a25 !important;
    border: 1px solid rgba(212,175,55,0.22) !important;
    border-radius: 8px !important;
}
[data-testid="stDataFrame"] [role="columnheader"],
[data-testid="stDataFrame"] th {
    background-color: #14161e !important;
    color: #e8c566 !important;
    border-bottom: 1px solid rgba(212,175,55,0.30) !important;
}
[data-testid="stDataFrame"] [role="row"],
[data-testid="stDataFrame"] [role="cell"],
[data-testid="stDataFrame"] [role="gridcell"],
[data-testid="stDataFrame"] td,
[data-testid="stDataFrame"] tr {
    background-color: #181a25 !important;
    color: #f0e8d0 !important;
    border-color: rgba(212,175,55,0.10) !important;
}
[data-testid="stDataFrame"] [role="row"]:hover,
[data-testid="stDataFrame"] tr:hover {
    background-color: #1f2230 !important;
}
/* ───────────────────────────────────────────────────────────────────────
   SLIDER — silver track + thumb instead of Streamlit's default red.
   The user explicitly asked for a silver look; we paint the filled
   portion of the track in a brushed-silver gradient and the thumb in a
   brighter silver so it reads cleanly on the dark background.
   ─────────────────────────────────────────────────────────────────── */
[data-baseweb="slider"] > div > div {
    background: rgba(168,156,128,0.30) !important;   /* unfilled track */
}
[data-baseweb="slider"] > div > div > div {
    background: linear-gradient(90deg, #c0c0c0 0%, #e0e0e0 50%, #c0c0c0 100%) !important;
}
[data-baseweb="slider"] [role="slider"] {
    background: linear-gradient(180deg, #f5f5f5 0%, #c0c0c0 100%) !important;
    border: 1px solid #d4af37 !important;
    box-shadow: 0 0 0 2px rgba(212,175,55,0.25) !important;
}
[data-baseweb="slider"] [data-testid="stThumbValue"],
[data-baseweb="slider"] [data-testid="stTickBarMin"],
[data-baseweb="slider"] [data-testid="stTickBarMax"] {
    color: #c0c0c0 !important;
    font-weight: 600 !important;
}

/* ───────────────────────────────────────────────────────────────────────
   FORM LABELS inside expanders — Streamlit greys the field labels.
   Force them to readable warm-white so the "Symbol / Warm-up bars /
   Leverage / …" labels in Add-Symbol expanders are visible.
   ─────────────────────────────────────────────────────────────────── */
[data-testid="stExpander"] label,
[data-testid="stExpander"] label p,
[data-testid="stExpander"] label span,
[data-testid="stForm"] label,
[data-testid="stForm"] label p,
[data-testid="stExpander"] [data-testid="stWidgetLabel"],
[data-testid="stExpander"] [data-testid="stWidgetLabel"] p {
    color: #f0e8d0 !important;
    opacity: 1 !important;
    font-weight: 500 !important;
}
/* Section sub-headers ("Risk Manager", "Transaction Costs", "Execution
   logic") rendered as bare st.markdown bold lines — make them legible. */
[data-testid="stMarkdownContainer"] strong,
[data-testid="stMarkdownContainer"] b {
    color: #e8c566 !important;
}

/* ───────────────────────────────────────────────────────────────────────
   BUTTONS — Kill switch / Flatten all / Refresh All / Clear All.
   The user reported the secondary buttons still rendered with white
   backgrounds. Force a strong, near-black background so the button
   text is readable on the dark page. Primary buttons keep the gold
   gradient set elsewhere; this rule is intentionally last so it wins
   the cascade.
   ─────────────────────────────────────────────────────────────────── */
.stButton > button,
button[kind="secondary"],
button[kind="secondaryFormSubmit"] {
    background-color: #14161e !important;
    color: #f0e8d0 !important;
    border: 1px solid rgba(212,175,55,0.45) !important;
    font-weight: 600 !important;
}
.stButton > button:hover,
button[kind="secondary"]:hover {
    color: #f5c93b !important;
    border-color: #d4af37 !important;
    background-color: #1a1d27 !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(180deg, #e8c566 0%, #d4af37 60%, #b9871e 100%) !important;
    color: #1a1306 !important;
    border: 1px solid #b9871e !important;
}

/* ───────────────────────────────────────────────────────────────────────
   DATA-GRID — Streamlit's modern dataframe is rendered by the Glide
   Data Grid component which has its own canvas + white-on-light
   defaults. Streamlit's base="dark" theme (set in .streamlit/config.toml)
   already darkens it; these rules belt-and-brace the wrapper, scroll
   container, and the iframe element used in some Streamlit versions.
   ─────────────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"],
[data-testid="stDataFrame"] > div,
[data-testid="stDataFrame"] > div > div,
[data-testid="stDataFrame"] iframe,
[data-testid="stDataFrameContainer"],
[data-testid="stDataFrameContainer"] > div {
    background-color: #181a25 !important;
}
[data-testid="stDataFrame"] canvas {
    background-color: #181a25 !important;
}

/* Older Streamlit styler-rendered HTML tables (st.table / df.style…) */
[data-testid="stTable"] table,
[data-testid="stTable"] th,
[data-testid="stTable"] td {
    background-color: #181a25 !important;
    color: #f0e8d0 !important;
    border-color: rgba(212,175,55,0.20) !important;
}
[data-testid="stTable"] th {
    background-color: #14161e !important;
    color: #e8c566 !important;
}

/* Hide Streamlit's auto-generated multipage nav (the "page_xxx" file list).
   We provide our own custom nav radio under the logo. */
[data-testid="stSidebarNav"],
[data-testid="stSidebarNavItems"],
section[data-testid="stSidebar"] > div:first-child > div > div > nav {
    display: none !important;
}

/* Tab labels — readable warm-white when inactive, gold when active. */
button[role="tab"] {
    color: #d6c89a !important;
    font-weight: 500 !important;
}
button[role="tab"][aria-selected="false"] {
    color: #d6c89a !important;
    opacity: 0.78;
}
button[role="tab"][aria-selected="true"] {
    color: #e8c566 !important;
    opacity: 1.0;
    border-bottom: 2px solid #d4af37 !important;
}

/* Radio + checkbox: nuke the default red 'selected' indicator with gold.
   Streamlit ships these as base-web widgets — cover every variant. */
[data-baseweb="radio"] input[type="radio"]:checked + div,
[data-baseweb="radio"] input[type="radio"]:checked + span,
[data-baseweb="radio"] [aria-checked="true"] {
    background-color: #d4af37 !important;
    border-color: #d4af37 !important;
}
/* The actual filled disk inside the outer radio circle */
[data-baseweb="radio"] [aria-checked="true"] > div:first-child,
[data-baseweb="radio"] [aria-checked="true"] [data-baseweb="circle"] {
    background-color: #d4af37 !important;
    border-color: #d4af37 !important;
}
/* Inactive state — silver-ish ring */
[data-baseweb="radio"] [aria-checked="false"] [data-baseweb="circle"] {
    border-color: rgba(212,175,55,0.45) !important;
    background-color: transparent !important;
}
[data-baseweb="radio"] [aria-checked="false"] {
    color: #b8b39a !important;
}

/* Streamlit also renders some radios as <svg circle> — colour those too. */
[data-baseweb="radio"] svg circle[fill] {
    fill: #d4af37 !important;
}
[data-baseweb="radio"] svg [stroke] {
    stroke: #d4af37 !important;
}

/* Checkbox — same treatment for the "checked" red square. */
[data-baseweb="checkbox"] [aria-checked="true"],
[data-baseweb="checkbox"] [aria-checked="mixed"] {
    background-color: #d4af37 !important;
    border-color: #d4af37 !important;
}
[data-baseweb="checkbox"] [aria-checked="true"] svg,
[data-baseweb="checkbox"] [aria-checked="mixed"] svg {
    fill: #1a1306 !important;
    stroke: #1a1306 !important;
    color: #1a1306 !important;
}
[data-baseweb="checkbox"] [aria-checked="false"] {
    border-color: rgba(212,175,55,0.45) !important;
    background-color: transparent !important;
}

/* Mode badge — gold variant for paper, restrained colours for live/backtest. */
.mode-paper {
    background: rgba(212,175,55,0.12) !important;
    color: #e8c566 !important;
    border: 1px solid #d4af37 !important;
}
.mode-live {
    background: rgba(198,66,66,0.15) !important;
    color: #ff8a7a !important;
    border: 1px solid #c64242 !important;
}
.mode-backtest {
    background: rgba(168,156,128,0.15) !important;
    color: #d6c89a !important;
    border: 1px solid #a89c80 !important;
}

/* Altair chart container — give it a charcoal background instead of the
   default white. The Altair view fill is also set in code (ui/charts.py),
   but this catches the iframe/container wrapper too. */
[data-testid="stVegaLiteChart"],
[data-testid="stArrowVegaLiteChart"],
.vega-embed {
    background: transparent !important;
}
.vega-embed canvas {
    background: #181a25 !important;
    border-radius: 8px;
}

/* Kill-switch and any other small "white pill" custom buttons rendered as
   pure HTML get nuked here too. */
.stButton > button {
    background-color: #14161e !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(180deg, #e8c566 0%, #d4af37 60%, #b9871e 100%) !important;
}

/* ───────────────────────────────────────────────────────────────────────
   Sidebar radio nav: force GOLD selected dot regardless of how Streamlit
   decides to render the indicator. Streamlit's default red comes from
   the primary-color CSS variable and is occasionally re-injected as an
   inline style — we cover (a) the data-checked attribute, (b) the
   nested div bullet, (c) any inner SVG with a fill, and (d) the
   primaryColor variable itself.
   ─────────────────────────────────────────────────────────────────── */
:root,
[data-testid="stSidebar"] {
    --primary-color: #d4af37 !important;
    --primaryColor: #d4af37 !important;
}
[data-testid="stSidebar"] [role="radiogroup"] [data-baseweb="radio"] [aria-checked="true"],
[data-testid="stSidebar"] [role="radiogroup"] [data-baseweb="radio"] [data-checked="true"],
[data-testid="stSidebar"] [role="radiogroup"] [role="radio"][aria-checked="true"] {
    background-color: #d4af37 !important;
    border-color: #d4af37 !important;
    color: #d4af37 !important;
}
[data-testid="stSidebar"] [role="radiogroup"] [data-baseweb="radio"] [aria-checked="true"] *,
[data-testid="stSidebar"] [role="radiogroup"] [data-baseweb="radio"] [aria-checked="true"] {
    background-color: #d4af37 !important;
    border-color: #d4af37 !important;
}
[data-testid="stSidebar"] [role="radiogroup"] [data-baseweb="radio"] [aria-checked="true"] svg,
[data-testid="stSidebar"] [role="radiogroup"] [data-baseweb="radio"] [aria-checked="true"] svg * {
    fill: #d4af37 !important;
    stroke: #d4af37 !important;
}
[data-testid="stSidebar"] [role="radiogroup"] [data-baseweb="radio"] [aria-checked="true"] [class^="st-emotion"] {
    background-color: #d4af37 !important;
    border-color: #d4af37 !important;
}
/* The little inner dot itself rendered as a div::after pseudo or nested div */
[data-testid="stSidebar"] [role="radiogroup"] [data-baseweb="radio"] [aria-checked="true"] > div > div {
    background-color: #d4af37 !important;
    border-color: #d4af37 !important;
}
/* Inactive — silver-gold ring */
[data-testid="stSidebar"] [role="radiogroup"] [data-baseweb="radio"] [aria-checked="false"] {
    border-color: rgba(212,175,55,0.40) !important;
    background-color: transparent !important;
}

/* ───────────────────────────────────────────────────────────────────────
   Expander headers: gold-textured wordmark when collapsed AND when
   expanded. The previous CSS only set the text colour; the open-state
   sometimes flipped it to default again. Force serif gold on both
   states + add a faint gold gradient to the header strip itself for
   the "metallic plate" effect from the mockup.
   ─────────────────────────────────────────────────────────────────── */
[data-testid="stExpander"] > details > summary {
    background: linear-gradient(180deg, rgba(212,175,55,0.08) 0%, rgba(212,175,55,0.02) 100%) !important;
    border-left: 3px solid rgba(212,175,55,0.35) !important;
    border-radius: 8px !important;
    padding: 0.55rem 0.85rem !important;
}
/* Apply the gradient text-clip to ONE element only — Streamlit wraps the
   summary label in both a <p> and an inner <span>, so styling both
   produces two overlapping copies. Targeting summary p alone keeps the
   serif gold treatment without the ghost text. */
[data-testid="stExpander"] > details > summary p,
[data-testid="stExpander"] > details[open] > summary p {
    font-family: 'Cinzel', serif !important;
    background: linear-gradient(180deg, #f5e6a8 0%, #d4af37 50%, #b9871e 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    color: transparent !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
}
[data-testid="stExpander"] summary svg {
    color: #d4af37 !important;
    fill:  #d4af37 !important;
}

/* ───────────────────────────────────────────────────────────────────────
   Checkbox labels — make them readable. Streamlit's default greys-out
   labels at low opacity in dark mode; force readable warm-white.
   ─────────────────────────────────────────────────────────────────── */
[data-baseweb="checkbox"] label,
[data-testid="stCheckbox"] label,
[data-testid="stCheckbox"] label p {
    color: #f0e8d0 !important;
    opacity: 1 !important;
    font-weight: 500 !important;
}
[data-baseweb="checkbox"]:hover label {
    color: #ffffff !important;
}
"""

    if theme_name == "MRMI Silver" and mrmi_extras:
        # Silver palette mirrors the gold gradient family. Pairs are
        # ordered longest-first so #f5e6a8 doesn't accidentally match
        # inside a longer string.
        _silver_swaps = [
            # rgba(212,175,55,X) wrappers -> silver triple
            ("212,175,55,0.40", "192,192,200,0.40"),
            ("212,175,55,0.45", "192,192,200,0.45"),
            ("212, 175, 55, 0.22", "192, 192, 200, 0.22"),
            ("212,175,55,0.35", "192,192,200,0.35"),
            ("212,175,55,0.30", "192,192,200,0.30"),
            ("212,175,55,0.25", "192,192,200,0.25"),
            ("212,175,55,0.22", "192,192,200,0.22"),
            ("212,175,55,0.18", "192,192,200,0.18"),
            ("212,175,55,0.12", "192,192,200,0.12"),
            ("212,175,55,0.10", "192,192,200,0.10"),
            ("212,175,55,0.08", "192,192,200,0.08"),
            ("212,175,55,0.05", "192,192,200,0.05"),
            ("212,175,55,0.02", "192,192,200,0.02"),
            ("212,175,55", "192,192,200"),  # any uncovered rgb()
            # Hex-coded gold values (longest first to avoid prefix collisions)
            ("#f5d77a", "#f5f5fa"),    # bright gold hover -> bright silver
            ("#f5e6a8", "#f0f0f5"),    # light gold -> light silver
            ("#f5c93b", "#e8e8ee"),    # bright accent gold -> bright silver
            ("#e8c566", "#cfcfd6"),    # warm head gold -> cool silver
            ("#d4af37", "#c0c0c8"),    # gold mid -> silver mid (accent)
            ("#c9a227", "#a8a8b0"),    # tooltip gold
            ("#b9871e", "#8a8a98"),    # dark gold -> dark silver
            ("#1a1306", "#0d0e14"),    # primary-button text on gold -> on silver
        ]
        for old, new in _silver_swaps:
            mrmi_extras = mrmi_extras.replace(old, new)

    css = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Fira+Code&family=Cinzel:wght@500;600;700&family=Playfair+Display:wght@600;700&display=swap');

:root {{
{vars_css}
}}

/* ── Global ── */
html, body, [class*="css"] {{
    font-family: var(--font) !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}}

/* ── Main content area ── */
.main .block-container {{
    background-color: var(--bg-primary);
    padding-top: 1.5rem;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background-color: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}}
[data-testid="stSidebar"] * {{
    color: var(--text-primary) !important;
}}

/* ── Cards / containers ── */
[data-testid="stExpander"],
[data-testid="stForm"],
div[data-baseweb="card"] {{
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}}

/* ── Metric widgets ── */
[data-testid="stMetric"] {{
    background-color: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 12px 16px;
}}
[data-testid="stMetricLabel"] {{ color: var(--text-secondary) !important; font-size: 0.78rem; }}
[data-testid="stMetricValue"] {{ color: var(--text-primary) !important; font-size: 1.4rem; font-weight: 600; }}
[data-testid="stMetricDelta"] svg {{ display: none; }}

/* ── Buttons ── */
.stButton > button {{
    background-color: var(--accent) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-weight: 600 !important;
    transition: background-color 0.2s ease;
}}
.stButton > button:hover {{
    background-color: var(--accent-hover) !important;
}}

/* ── Inputs ── */
input, textarea, select {{
    background-color: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}}

/* ── DataFrames / tables ── */
[data-testid="stDataFrame"] {{
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}}

/* ── Dividers ── */
hr {{ border-color: var(--border) !important; }}

/* ── Status colours ── */
.positive-value {{ color: var(--positive) !important; font-weight: 600; }}
.negative-value {{ color: var(--negative) !important; font-weight: 600; }}

/* ── Mode badge ── */
.mode-badge {{
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}}
.mode-paper   {{ background: #1a3a4a; color: #4dd0e1; border: 1px solid #4dd0e1; }}
.mode-live    {{ background: #4a1a1a; color: #ff7043; border: 1px solid #ff7043; }}
.mode-backtest{{ background: #2a2a4a; color: #ba68c8; border: 1px solid #ba68c8; }}
{mrmi_extras}
</style>
"""
    st.markdown(css, unsafe_allow_html=True)


def mode_badge(mode: str) -> str:
    """Return an HTML mode badge string for use with st.markdown(..., unsafe_allow_html=True)."""
    label = mode.upper()
    css_class = f"mode-{mode.lower()}"
    return f'<span class="mode-badge {css_class}">{label}</span>'
