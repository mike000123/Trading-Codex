"""
app.py  ←  streamlit run app.py
────────────────────────────────
Multi-page Streamlit trading platform.
Sidebar navigation drives all pages.
"""
import importlib
import logging
import streamlit as st

# Suppress Tornado WebSocket closed errors — these are harmless and appear
# when long-running backtests cause the browser WebSocket to time out.
# Results are always computed and stored correctly despite these messages.
logging.getLogger("tornado.websocket").setLevel(logging.CRITICAL)
logging.getLogger("tornado.iostream").setLevel(logging.CRITICAL)
# Streamlit fragments used for page-local auto-refresh can legitimately be
# removed during a full-app rerun when the user navigates to another page.
# Streamlit logs that as INFO from app_session, but it's expected and noisy.
logging.getLogger("streamlit.runtime.app_session").setLevel(logging.WARNING)

st.set_page_config(
    page_title="MRMI Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme must be applied before any other st.* calls ──────────────────────
from ui.themes import apply_theme, THEME_NAMES
from core.startup_preload import maybe_run_startup_preload

# The selectbox below uses key="theme_selector". Streamlit syncs that key
# to st.session_state at the START of every rerun (before any script line
# runs), so reading it here gives us the user's NEW pick on the very same
# rerun the dropdown change triggered. Reading st.session_state["theme"]
# here would return the OLD value because we only assign it AFTER the
# selectbox has run later in the sidebar — that's what caused the
# "needs two clicks for the theme to apply" bug.
_current_theme = st.session_state.get("theme_selector") or st.session_state.get("theme") or "MRMI Gold"
st.session_state["theme"] = _current_theme

# Theme applied immediately so subsequent st.* calls inherit the styling.
# The selector itself renders later in the sidebar (under the logo).
apply_theme(_current_theme)

if maybe_run_startup_preload():
    st.stop()

from config.settings import settings
from db.database import Database

# ── Server-side paper-trading worker ──────────────────────────────────────
# Launches a daemon thread that ticks every 60s independent of any browser
# session, so phone-locked / refreshed pages keep getting bars processed.
# Idempotent: only the first call per process actually starts a thread.
try:
    from execution import paper_worker as _paper_worker
    _paper_worker.start()
except Exception as _exc:  # noqa: BLE001
    # Never let worker boot kill the app; it's a backstop, not a hard dep.
    import logging as _logging
    _logging.getLogger(__name__).warning(
        f"paper_worker failed to start: {_exc}"
    )

PAGES = {
    "Historical Simulator": "pages.page_simulator",
    "Strategy Lab":         "pages.page_strategy_lab",
    "Backtester":           "pages.page_backtest",
    "Forward Test":         "pages.page_forward_test",
    "Paper Trading":        "pages.page_paper_trading",
    "Shadow Compare":       "pages.page_shadow_compare",
    "Portfolio":            "pages.page_portfolio",
    "Settings":             "pages.page_settings",
}

PAGE_SLUGS = {
    "Historical Simulator": "simulator",
    "Strategy Lab":         "strategy_lab",
    "Backtester":           "backtester",
    "Forward Test":         "forward_test",
    "Paper Trading":        "paper_trading",
    "Shadow Compare":       "shadow_compare",
    "Portfolio":            "portfolio",
    "Settings":             "settings",
}
SLUG_TO_PAGE = {slug: label for label, slug in PAGE_SLUGS.items()}
_APP_PAGE_CFG_KEY = "app_last_page_v1"
_CURRENT_PAGE_KEY = "current_page"
_NAV_WIDGET_KEY = "nav_radio"


def _db() -> Database:
    return Database(settings.db_path)


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


def _get_page_from_config() -> str | None:
    try:
        payload = _db().load_config(_APP_PAGE_CFG_KEY) or {}
    except Exception:
        return None
    page_slug = payload.get("page")
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


def _persist_page_config(page_name: str) -> None:
    slug = PAGE_SLUGS.get(page_name)
    if not slug:
        return
    try:
        _db().save_config(_APP_PAGE_CFG_KEY, {"page": slug})
    except Exception:
        pass


def _resolve_desired_page(page_keys: list[str]) -> str:
    nav_target = st.session_state.pop("nav_target", None)
    if nav_target and nav_target in page_keys:
        return nav_target
    desired = (
        _get_page_from_query()
        or _get_page_from_config()
        or st.session_state.get(_CURRENT_PAGE_KEY)
        or "Backtester"
    )
    if desired not in page_keys:
        desired = page_keys[0]
    return desired

with st.sidebar:
    # 1. Logo block — pick a gold or silver variant based on the active
    #    theme. User-supplied PNG (mrmi_logo.png) overrides everything;
    #    otherwise the SVG matching the brand colour family wins.
    import base64
    from pathlib import Path as _P
    _is_silver = _current_theme == "MRMI Silver"
    _logo_png = _P(__file__).parent / "assets" / "mrmi_logo.png"
    _logo_svg_silver = _P(__file__).parent / "assets" / "mrmi_logo_silver.svg"
    _logo_svg_gold = _P(__file__).parent / "assets" / "mrmi_logo.svg"
    _logo_svg = _logo_svg_silver if (_is_silver and _logo_svg_silver.exists()) else _logo_svg_gold
    if _logo_png.exists():
        st.image(str(_logo_png), width=140)
    elif _logo_svg.exists():
        try:
            _svg_b64 = base64.b64encode(_logo_svg.read_bytes()).decode("ascii")
            st.markdown(
                f'<div style="text-align:center;margin:0.4rem 0 0.4rem 0;">'
                f'<img src="data:image/svg+xml;base64,{_svg_b64}" '
                f'width="120" alt="MRMI Platform"/></div>',
                unsafe_allow_html=True,
            )
        except Exception:
            pass
    # Wordmark gradient also follows the theme family.
    _wordmark_gradient = (
        "linear-gradient(180deg,#f5f5fa 0%,#c0c0c8 50%,#7a7a86 100%)"
        if _is_silver else
        "linear-gradient(180deg,#f5e6a8 0%,#d4af37 50%,#b9871e 100%)"
    )
    st.markdown(
        f"<div style=\"text-align:center;font-family:\'Cinzel\',serif;"
        f"font-weight:700;letter-spacing:0.18em;font-size:1.05rem;"
        f"background:{_wordmark_gradient};"
        f"-webkit-background-clip:text;-webkit-text-fill-color:transparent;"
        f"background-clip:text;color:transparent;"
        f"margin:0.1rem 0 0.8rem 0;\">MRMI PLATFORM</div>",
        unsafe_allow_html=True,
    )

    # 2. Theme dropdown directly under the logo. The widget's `key`
    #    auto-syncs to st.session_state["theme_selector"] before each
    #    rerun, which is read at the top of this file to call
    #    apply_theme() on THIS rerun (no second click needed).
    if "theme_selector" not in st.session_state:
        st.session_state["theme_selector"] = _current_theme
    st.selectbox(
        "Theme", THEME_NAMES,
        key="theme_selector",
    )
    st.markdown("---")

    # 3. Page navigation.
    page_keys = list(PAGES.keys())
    desired_page = _resolve_desired_page(page_keys)
    page_name = st.radio(
        "Navigation",
        page_keys,
        index=page_keys.index(desired_page),
        key=_NAV_WIDGET_KEY,
        label_visibility="collapsed",
    )
    st.session_state[_CURRENT_PAGE_KEY] = page_name
    _set_page_query(page_name)
    _persist_page_config(page_name)
    st.markdown("---")

# ── Render selected page ───────────────────────────────────────────────────
page_module = importlib.import_module(PAGES[page_name])
page_module.render()
