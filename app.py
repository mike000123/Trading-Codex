"""
app.py  ←  streamlit run app.py
────────────────────────────────
Multi-page Streamlit trading platform.
Sidebar navigation drives all pages.
"""
import logging
import streamlit as st

# Suppress Tornado WebSocket closed errors — these are harmless and appear
# when long-running backtests cause the browser WebSocket to time out.
# Results are always computed and stored correctly despite these messages.
logging.getLogger("tornado.websocket").setLevel(logging.CRITICAL)
logging.getLogger("tornado.iostream").setLevel(logging.CRITICAL)

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
    page_forward_test,
    page_paper_trading,
    page_shadow_compare,
    page_portfolio,
    page_settings,
)

PAGES = {
    "📊 Historical Simulator":  page_simulator,
    "🔬 Strategy Lab":           page_strategy_lab,
    "⏪ Backtester":             page_backtest,
    "🔭 Forward Test":            page_forward_test,
    "📝 Paper Trading":          page_paper_trading,
    "🔁 Shadow Compare":         page_shadow_compare,
    "💼 Portfolio":              page_portfolio,
    "⚙️ Settings":               page_settings,
}

PAGE_SLUGS = {
    "📊 Historical Simulator": "simulator",
    "🔬 Strategy Lab": "strategy_lab",
    "⏪ Backtester": "backtester",
    "🔭 Forward Test": "forward_test",
    "📝 Paper Trading": "paper_trading",
    "🔁 Shadow Compare": "shadow_compare",
    "💼 Portfolio": "portfolio",
    "⚙️ Settings": "settings",
}
SLUG_TO_PAGE = {slug: label for label, slug in PAGE_SLUGS.items()}


def _get_page_from_query() -> str | None:
    try:
        page_slug = st.query_params.get("page")
    except Exception:
        return None
    if isinstance(page_slug, list):
        page_slug = page_slug[0] if page_slug else None
    if not page_slug:
        return None
    return SLUG_TO_PAGE.get(str(page_slug))


def _set_page_query(page_name: str) -> None:
    slug = PAGE_SLUGS.get(page_name)
    if not slug:
        return
    try:
        current = st.query_params.get("page")
        if isinstance(current, list):
            current = current[0] if current else None
        if current != slug:
            st.query_params["page"] = slug
    except Exception:
        pass

with st.sidebar:
    st.markdown("---")
    st.markdown("### 📈 AlgoTrader Pro")
    page_keys = list(PAGES.keys())
    # Allow portfolio nav buttons to redirect here
    nav_target = st.session_state.pop("nav_target", None)
    if nav_target and nav_target in page_keys:
        default_idx = page_keys.index(nav_target)
    else:
        default_page = _get_page_from_query() or "⏪ Backtester"
        default_idx  = page_keys.index(default_page) if default_page in page_keys else 0
    page_name = st.radio(
        "Navigation",
        page_keys,
        index=default_idx,
        key="nav",
        label_visibility="collapsed",
    )
    _set_page_query(page_name)
    st.markdown("---")

    # ── Diagnostic: session-state snapshot ────────────────────────────────
    # Lets us confirm whether in-memory state actually survives page switches.
    # Uses the same explicit keys each page writes to (pt_*/ft_*/bt_*).
    with st.expander("🔍 State", expanded=False):
        _ss          = st.session_state
        _pt_runs     = _ss.get("pt_active_runs") or {}
        _ft_runs     = _ss.get("ft_active_runs") or {}
        _bt_cached   = "bt_result" in _ss
        _bt_symbol   = _ss.get("bt_symbol", "—") if _bt_cached else "—"
        _pt_signals  = len(_ss.get("pt_all_signals") or [])
        _ft_signals  = len(_ss.get("ft_all_signals") or [])
        _total_keys  = len(list(_ss.keys()))

        st.caption(
            f"Paper runs: **{len(_pt_runs)}**"
            + (f"  ·  {', '.join(list(_pt_runs.keys())[:6])}" if _pt_runs else "")
        )
        st.caption(
            f"Forward runs: **{len(_ft_runs)}**"
            + (f"  ·  {', '.join(list(_ft_runs.keys())[:6])}" if _ft_runs else "")
        )
        st.caption(
            f"Backtest result cached: **{'yes' if _bt_cached else 'no'}**"
            + (f"  ·  {_bt_symbol}" if _bt_cached else "")
        )
        st.caption(
            f"Signals buffered · paper: **{_pt_signals}**  ·  forward: **{_ft_signals}**"
        )
        st.caption(f"Total session keys: **{_total_keys}**")
    st.markdown("---")

# ── Render selected page ───────────────────────────────────────────────────
PAGES[page_name].render()
