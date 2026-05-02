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
from db.database import Database


_DONE_KEY = "_startup_preload_done_v1"
_SUMMARY_KEY = "_startup_preload_summary_v1"


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
    except Exception:
        return False
    return bool((payload.get("selection") or {}).get("symbol"))


def _has_forward_payload() -> bool:
    try:
        from pages import page_forward_test

        payload = _db().load_config(page_forward_test._STATE_CFG_KEY) or {}
    except Exception:
        return False
    runs = payload.get("runs") or {}
    return isinstance(runs, dict) and bool(runs)


def _has_paper_payload() -> bool:
    try:
        from pages import page_paper_trading

        payload = _db().load_config(page_paper_trading._RUNS_CFG_KEY) or {}
    except Exception:
        return False
    runs = payload.get("runs") or {}
    return isinstance(runs, dict) and bool(runs)


def _build_plan() -> list[_PreloadTask]:
    tasks: list[_PreloadTask] = []

    if _has_backtest_payload():
        from pages import page_backtest

        tasks.append(
            _PreloadTask(
                label="Loading the last Backtester dataset and its companion context",
                runner=page_backtest.startup_preload,
            )
        )

    if _has_forward_payload():
        from pages import page_forward_test

        tasks.append(
            _PreloadTask(
                label="Warming Forward Test symbols and their strategy context",
                runner=page_forward_test.startup_preload,
            )
        )

    if _has_paper_payload():
        from pages import page_paper_trading

        tasks.append(
            _PreloadTask(
                label="Priming Paper Trading symbols and their chart/signal caches",
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
        st.session_state[_DONE_KEY] = True
        return False

    progress, step_box, note_box = _render_shell()
    summaries: list[dict] = []
    total = len(plan)

    for idx, task in enumerate(plan, start=1):
        step_box.markdown(f"**Step {idx} of {total}**  \n{task.label}")

        def _update(message: str) -> None:
            note_box.info(message)

        try:
            result = task.runner(_update) or {}
            summaries.append({"label": task.label, **result})
        except Exception as exc:  # noqa: BLE001
            summaries.append({"label": task.label, "error": str(exc)})
            note_box.warning(f"{task.label} skipped with a recoverable error: {exc}")

        progress.progress(idx / total, text=f"Warmup progress: {idx}/{total}")

    st.session_state[_SUMMARY_KEY] = summaries
    st.session_state[_DONE_KEY] = True
    note_box.success("Startup warmup finished. Opening the app…")
    st.rerun()
    return True

