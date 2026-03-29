"""
app.py  ←  streamlit run app.py
────────────────────────────────
Multi-page Streamlit trading platform.
Sidebar navigation drives all pages.
"""
import streamlit as st

st.set_page_config(
    page_title="AlgoTrader Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme must be applied before any other st.* calls ──────────────────────
from ui.themes import apply_theme, THEME_NAMES

if "theme" not in st.session_state:
    st.session_state["theme"] = "Dark"

# Theme selector lives at the very top of the sidebar
with st.sidebar:
    st.markdown("### 🎨 Theme")
    chosen_theme = st.selectbox(
        "Theme", THEME_NAMES,
        index=THEME_NAMES.index(st.session_state["theme"]),
        key="theme_selector",
        label_visibility="collapsed",
    )
    st.session_state["theme"] = chosen_theme

apply_theme(st.session_state["theme"])

# ── Page registry ──────────────────────────────────────────────────────────
from pages import (
    page_simulator,
    page_strategy_lab,
    page_backtest,
    page_paper_trading,
    page_portfolio,
    page_settings,
)

PAGES = {
    "📊 Historical Simulator":  page_simulator,
    "🔬 Strategy Lab":           page_strategy_lab,
    "⏪ Backtester":             page_backtest,
    "📝 Paper Trading":          page_paper_trading,
    "💼 Portfolio":              page_portfolio,
    "⚙️ Settings":               page_settings,
}

with st.sidebar:
    st.markdown("---")
    st.markdown("### 📈 AlgoTrader Pro")
    page_name = st.radio(
        "Navigation",
        list(PAGES.keys()),
        key="nav",
        label_visibility="collapsed",
    )
    st.markdown("---")

# ── Render selected page ───────────────────────────────────────────────────
PAGES[page_name].render()
