"""
ui/themes.py
────────────
Predefined UI themes injected via st.markdown CSS.

Usage:
    from ui.themes import apply_theme, THEME_NAMES
    apply_theme(st.session_state.get("theme", "Dark"))
"""
from __future__ import annotations

import streamlit as st

# ─── Theme definitions ────────────────────────────────────────────────────────
# Each theme is a dict of CSS variable overrides applied to :root.
# Extend freely; the apply_theme() function picks the active one.

_THEMES: dict[str, dict[str, str]] = {
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
    "Light": {
        "--bg-primary": "#f5f7ff",
        "--bg-secondary": "#ffffff",
        "--bg-card": "#ffffff",
        "--text-primary": "#1a1a2e",
        "--text-secondary": "#555577",
        "--accent": "#5056d6",
        "--accent-hover": "#3a40bb",
        "--positive": "#00897b",
        "--negative": "#c62828",
        "--border": "#dde1f0",
        "--chart-grid": "#eef0fa",
        "--font": "'Inter', sans-serif",
        "--radius": "8px",
    },
    "Terminal": {
        "--bg-primary": "#0d0d0d",
        "--bg-secondary": "#111111",
        "--bg-card": "#161616",
        "--text-primary": "#00ff88",
        "--text-secondary": "#00cc66",
        "--accent": "#00ff88",
        "--accent-hover": "#00dd77",
        "--positive": "#00ff88",
        "--negative": "#ff4444",
        "--border": "#1a1a1a",
        "--chart-grid": "#1a1a1a",
        "--font": "'Fira Code', 'Courier New', monospace",
        "--radius": "2px",
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


def apply_theme(theme_name: str) -> None:
    """Inject CSS variables for the selected theme into the Streamlit page."""
    theme = _THEMES.get(theme_name, _THEMES["Dark"])

    vars_css = "\n".join(f"    {k}: {v};" for k, v in theme.items())

    css = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Fira+Code&display=swap');

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
</style>
"""
    st.markdown(css, unsafe_allow_html=True)


def mode_badge(mode: str) -> str:
    """Return an HTML mode badge string for use with st.markdown(..., unsafe_allow_html=True)."""
    label = mode.upper()
    css_class = f"mode-{mode.lower()}"
    return f'<span class="mode-badge {css_class}">{label}</span>'
