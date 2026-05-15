"""
Startup warmup orchestration for the Streamlit app shell.

This module coordinates an initial preload pass before the normal page shell
renders so users see one clear loading screen instead of several page-local
waits. The warmup only fetches and prepares data; it does not execute trading
logic or submit broker orders.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import streamlit as st

from config.settings import settings
from core.logger import log
from db.database import Database


_DONE_KEY = "_startup_preload_done_v1"
_SUMMARY_KEY = "_startup_preload_summary_v1"
_WORKER_DELAY_PENDING_KEY = "_startup_worker_delay_pending_v1"
_APP_PAGE_CFG_KEY = "app_last_page_v1"
_PAGE_SLUG_TO_NAME = {
    "simulator": "Historical Simulator",
    "strategy_lab": "Strategy Lab",
    "backtester": "Backtester",
    "forward_test": "Forward Test",
    "paper_trading": "Paper Trading",
    "shadow_compare": "Shadow Compare",
    "portfolio": "Portfolio",
    "settings": "Settings",
}


def _db() -> Database:
    return Database(settings.db_path)


@dataclass(frozen=True)
class _PreloadTask:
    label: str
    runner: Callable[[Callable[[str], None] | None], dict]


def _has_backtest_payload() -> bool:
    try:
        from pages import page_backtest

        payload = _db().load_config(page_backtest._BT_RESULT_CFG_KEY) or {}
        portfolio_payload = _db().load_config(page_backtest._BT_PORTFOLIO_CFG_KEY) or {}
    except Exception:
        return False
    return bool((payload.get("selection") or {}).get("symbol")) or bool(portfolio_payload)


def _has_forward_payload() -> bool:
    try:
        from pages import page_forward_test

        payload = _db().load_config(page_forward_test._STATE_CFG_KEY) or {}
        auto_cfg = _db().load_config(page_forward_test._EARNINGS_AUTO_CFG_KEY) or {}
    except Exception:
        return False
    runs = payload.get("runs") or {}
    return (isinstance(runs, dict) and bool(runs)) or bool(auto_cfg.get("enabled"))


def _has_paper_payload() -> bool:
    try:
        from pages import page_paper_trading

        payload = _db().load_config(page_paper_trading._RUNS_CFG_KEY) or {}
        auto_cfg = _db().load_config(page_paper_trading._EARNINGS_AUTO_CFG_KEY) or {}
    except Exception:
        return False
    runs = payload.get("runs") or {}
    return (isinstance(runs, dict) and bool(runs)) or bool(auto_cfg.get("enabled"))


def _needs_earnings_calendar_refresh() -> bool:
    try:
        from data.earnings_calendar import daily_cache_is_stale, load_earnings_universe_config

        universe_id = load_earnings_universe_config().get("universe_id")
        return bool(daily_cache_is_stale(universe_id=universe_id))
    except Exception:
        return False


def _refresh_earnings_calendar(status_cb: Callable[[str], None] | None = None) -> dict:
    from data.earnings_calendar import (
        available_earnings_universe_labels,
        load_earnings_universe_config,
        refresh_daily_earnings_cache,
    )

    universe_id = load_earnings_universe_config().get("universe_id")
    universe_label = available_earnings_universe_labels().get(universe_id, universe_id)
    if status_cb is not None:
        status_cb(f"Earnings calendar: refreshing {universe_label} for today's startup window")
    return refresh_daily_earnings_cache(status_cb=status_cb, universe_id=universe_id)


def _resolve_target_page() -> str:
    try:
        page_slug = st.query_params.get("page")
        if isinstance(page_slug, list):
            page_slug = page_slug[0] if page_slug else None
        if page_slug:
            resolved = _PAGE_SLUG_TO_NAME.get(str(page_slug))
            if resolved:
                return resolved
    except Exception:
        pass

    try:
        payload = _db().load_config(_APP_PAGE_CFG_KEY) or {}
        page_slug = payload.get("page")
        if page_slug:
            resolved = _PAGE_SLUG_TO_NAME.get(str(page_slug))
            if resolved:
                return resolved
    except Exception:
        pass

    return "Backtester"


def _build_plan() -> list[_PreloadTask]:
    tasks: list[_PreloadTask] = []
    target_page = _resolve_target_page()

    if _needs_earnings_calendar_refresh():
        tasks.append(
            _PreloadTask(
                label="Refreshing the shared daily earnings calendar",
                runner=_refresh_earnings_calendar,
            )
        )

    if target_page == "Backtester" and _has_backtest_payload():
        from pages import page_backtest

        tasks.append(
            _PreloadTask(
                label="Loading the last Backtester dataset and its companion context",
                runner=page_backtest.startup_preload,
            )
        )

    if target_page == "Forward Test" and _has_forward_payload():
        from pages import page_forward_test

        tasks.append(
            _PreloadTask(
                label="Warming Forward Test symbols and syncing today's earnings auto-load",
                runner=page_forward_test.startup_preload,
            )
        )

    if target_page == "Paper Trading" and _has_paper_payload():
        from pages import page_paper_trading

        tasks.append(
            _PreloadTask(
                label="Priming Paper Trading symbols and syncing today's earnings auto-load",
                runner=page_paper_trading.startup_preload,
            )
        )

    return tasks


def _render_shell() -> tuple:
    st.title("Preparing MRMI Platform")
    st.caption(
        "We’re warming the last saved datasets and live-mode caches so the main pages "
        "open with less waiting afterward."
    )
    progress = st.progress(0.0, text="Starting startup warmup…")
    step_box = st.empty()
    note_box = st.empty()
    return progress, step_box, note_box


def maybe_run_startup_preload() -> bool:
    """
    Run the one-time startup warmup for the current browser session.

    Returns True when a loading screen was rendered and a rerun was triggered,
    which means the caller should stop further app rendering for this pass.
    """
    if st.session_state.get(_DONE_KEY):
        return False

    plan = _build_plan()
    if not plan:
        log.info("startup_preload: no startup tasks needed")
        st.session_state[_DONE_KEY] = True
        return False

    log.info(f"startup_preload: running {len(plan)} task(s)")
    progress, step_box, note_box = _render_shell()
    summaries: list[dict] = []
    total = len(plan)

    for idx, task in enumerate(plan, start=1):
        step_box.markdown(f"**Step {idx} of {total}**  \n{task.label}")
        log.info(f"startup_preload: step {idx}/{total} — {task.label}")

        def _update(message: str) -> None:
            note_box.info(message)
            try:
                log.info(f"startup_preload: {message}")
            except Exception:
                pass

        try:
            result = task.runner(_update) or {}
            summaries.append({"label": task.label, **result})
            log.info(f"startup_preload: completed — {task.label}")
        except Exception as exc:  # noqa: BLE001
            summaries.append({"label": task.label, "error": str(exc)})
            note_box.warning(f"{task.label} skipped with a recoverable error: {exc}")
            log.warning(f"startup_preload: {task.label} failed — {exc}")

        progress.progress(idx / total, text=f"Warmup progress: {idx}/{total}")

    st.session_state[_SUMMARY_KEY] = summaries
    st.session_state[_DONE_KEY] = True
    if _has_paper_payload():
        st.session_state[_WORKER_DELAY_PENDING_KEY] = True
    note_box.success("Startup warmup finished. Opening the app…")
    log.info("startup_preload: finished")
    st.rerun()
    return True


def pop_startup_preload_summary() -> list[dict]:
    try:
        summary = st.session_state.pop(_SUMMARY_KEY, [])
    except Exception:
        summary = []
    return list(summary or [])
