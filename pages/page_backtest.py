"""
pages/page_backtest.py
Walk-forward backtester.
"""
from __future__ import annotations

import altair as alt
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import pandas as pd
from pathlib import Path
import pickle
import re
import streamlit as st
import time
from typing import Callable

from config.symbol_profiles import resolve_execution_default
from config.strategy_presets.bollinger_rsi.gld_candidates import get_candidate
from config.settings import settings
from core import runtime_cache
from core.logger import log
from core.models import Direction, TradeOutcome, TradeRecord
from data.fair_value import compute_gld_fair_value_diagnostics, fair_value_cache_fingerprint
from data.ingestion import (
    load_forward_blended_data,
    load_from_ticker,
    prepare_strategy_data,
)
from db.database import Database
from reporting.backtest import BacktestEngine, BacktestResult
from reporting.managed_portfolio_backtest import (
    ManagedPortfolioBacktestEngine,
    ManagedPortfolioBacktestResult,
    ManagedPortfolioSymbolInput,
)
from risk.manager import RiskManager
from strategies import get_strategy, list_strategies
from execution.entry_policy_base import available_policies
# Importing the concrete policies registers them with the factory so the
# dropdown sees both entries.
import execution.entry_policy_classic  # noqa: F401
import execution.entry_policy_alpaca   # noqa: F401
from ui.charts import (
    CHART_WINDOW_OPTIONS,
    chart_window_label,
    clip_frame_to_price_window,
    filter_chart_window,
    pnl_distribution,
)
from ui.components import (
    _companion_strategy,
    auto_select_strategy_name_for_state,
    ordered_strategy_items,
    recommended_primary_strategy_id,
    render_data_source_selector,
    render_metrics_row,
    render_mode_banner,
    render_strategy_selection_help,
    render_strategy_params,
    strategy_display_name_from_id,
    strategy_option_label_from_name,
)


def _theme_chart_color(key: str = "primary") -> str:
    """Active-theme palette shim — reads ui.charts._palette() so this page's
    chart colours follow the user's theme selection. Falls back to gold."""
    try:
        from ui.charts import _palette
        return _palette().get(key, "#d4af37")
    except Exception:
        return "#d4af37"


_GREEN = "#2faa6a"
_RED = "#c64242"
_BLUE = "#d4af37"  # (gold default — chart funcs override via _theme_chart_color)
_GOLD = "#ffd54f"
_ORANGE = "#ff9800"
_PURPLE = "#ab47bc"
_BT_RUNTIME_CACHE_NS = "backtest_loaded_frames_v1"
_BT_RUNTIME_CACHE_KEY = "current"
_BT_MANAGED_FRAME_CACHE_NS = "backtest_managed_frames_v1"
_BT_MANAGED_PREPARED_CACHE_NS = "backtest_managed_prepared_frames_v1"
_BT_MANAGED_PREPARED_CACHE_VERSION = "v2"
_BT_MANAGED_PREPARED_DISK_CACHE_DIR = (
    Path(__file__).resolve().parents[1] / "artifacts" / "cache" / "managed_prepared_frames"
)
_EARNINGS_UNIFIED_STRATEGY_NAME = "Earnings Event Hybrid (Research)"
_LEGACY_EARNINGS_STRATEGY_NAMES = {
    "Earnings Overshoot Hybrid (Research)",
    "Earnings Negative Rebound (Research)",
    "Earnings Negative Hybrid (Research)",
}
_LEGACY_EARNINGS_STRATEGY_IDS = {
    "earnings_overshoot_hybrid",
    "earnings_negative_rebound",
    "earnings_negative_hybrid",
}


def _axis_cfg() -> dict:
    return dict(
        gridColor=_theme_chart_color("axis_grid"),
        labelColor=_theme_chart_color("axis_label"),
        titleColor=_theme_chart_color("axis_title"),
        labelFontSize=12,
        titleFontSize=13,
    )


def _title_cfg() -> dict:
    return dict(color=_theme_chart_color("title"), fontSize=14, fontWeight="bold")


def _render_stage_progress(progress_widget, progress: float, label: str) -> None:
    pct = max(0.0, min(1.0, float(progress)))
    pct_label = int(round(pct * 100))
    progress_widget.progress(pct, text=f"{pct_label}% · {label}")


def _map_legacy_earnings_strategy_name(name: str | None, strategy_names: list[str]) -> str | None:
    if name in _LEGACY_EARNINGS_STRATEGY_NAMES and _EARNINGS_UNIFIED_STRATEGY_NAME in strategy_names:
        return _EARNINGS_UNIFIED_STRATEGY_NAME
    return name


def _progress_time_label(ts) -> str:
    try:
        stamp = pd.Timestamp(ts)
    except Exception:
        return "current bar"
    if stamp.hour == 0 and stamp.minute == 0 and stamp.second == 0:
        return stamp.strftime("%Y-%m-%d")
    return stamp.strftime("%Y-%m-%d %H:%M")


_MAX_CHART_PTS = 5_000
_BT_RESULT_CFG_KEY = "backtester_last_result_v1"
_BT_PORTFOLIO_CFG_KEY = "backtester_last_managed_portfolio_v1"
_BT_PENDING_TARGET_LOAD_KEY = "bt_pending_target_load_v1"
_BT_PENDING_TARGET_EXECUTE_KEY = "bt_pending_target_execute_v1"
_BT_PENDING_STRATEGY_NAME_KEY = "bt_pending_strategy_name_v1"
_BT_RUN_AFTER_LOAD_KEY = "bt_run_after_target_load_v1"
_BT_TARGET_STATUS_KEY = "bt_target_status_msg_v1"
_BT_STAGE2_PENDING_KEY = "bt_stage2_pending_v1"
_BT_PORTFOLIO_RESULT_KEY = "bt_managed_portfolio_result_v1"
_BT_PORT_PENDING_RUN_KEY = "bt_port_pending_run_v1"
_BT_MODE_KEY = "bt_mode_v1"
_BT_PENDING_MODE_KEY = "bt_mode_pending_v1"
_BT_LAST_RENDERED_MODE_KEY = "bt_last_rendered_mode_v1"
_BT_MODE_SINGLE = "Single Ticker"
_BT_MODE_PORTFOLIO = "Portfolio"
_BT_DEFAULT_START = pd.Timestamp("2024-04-04").date()
_BT_DEFAULT_END = pd.Timestamp("2026-04-23").date()


def _db() -> Database:
    return Database(settings.db_path)


def _backtest_result_stats() -> dict[str, int]:
    result = st.session_state.get("bt_result")
    if result is None:
        return {"trades": 0, "equity_rows": 0}
    try:
        return {
            "trades": int(len(getattr(result, "trades", []) or [])),
            "equity_rows": int(len(getattr(result, "equity_curve", pd.DataFrame()) or [])),
        }
    except Exception:
        return {"trades": 0, "equity_rows": 0}


def _managed_result_stats(payload: dict | None = None) -> dict[str, int]:
    payload = payload if isinstance(payload, dict) else st.session_state.get(_BT_PORTFOLIO_RESULT_KEY)
    if not isinstance(payload, dict):
        return {"trades": 0, "equity_rows": 0, "symbols": 0}
    result = payload.get("result")
    if result is None:
        return {"trades": 0, "equity_rows": 0, "symbols": 0}
    try:
        return {
            "trades": int(len(getattr(result, "trades", []) or [])),
            "equity_rows": int(len(getattr(result, "equity_curve", pd.DataFrame()) or [])),
            "symbols": int(len(payload.get("symbols") or [])),
        }
    except Exception:
        return {"trades": 0, "equity_rows": 0, "symbols": 0}


def _log_backtester_mode_timing(
    *,
    previous_mode: str | None,
    mode: str,
    mode_changed: bool,
    total_elapsed: float,
    clear_elapsed: float,
    branch_elapsed: float,
    restored_snapshot: bool,
    restored_managed_snapshot: bool,
    cleared_single_stats: dict[str, int] | None = None,
    cleared_portfolio_stats: dict[str, int] | None = None,
) -> None:
    if not mode_changed and total_elapsed < 0.35:
        return
    detail_parts = [
        f"mode={mode}",
        f"prev={previous_mode or 'none'}",
        f"changed={mode_changed}",
        f"elapsed={total_elapsed:.3f}s",
        f"clear={clear_elapsed:.3f}s",
        f"branch={branch_elapsed:.3f}s",
        f"restored_single={restored_snapshot}",
        f"restored_portfolio={restored_managed_snapshot}",
    ]
    if cleared_single_stats:
        detail_parts.append(
            "cleared_single="
            f"{cleared_single_stats.get('trades', 0)}t/"
            f"{cleared_single_stats.get('equity_rows', 0)}eq"
        )
    if cleared_portfolio_stats:
        detail_parts.append(
            "cleared_portfolio="
            f"{cleared_portfolio_stats.get('symbols', 0)}sym/"
            f"{cleared_portfolio_stats.get('trades', 0)}t/"
            f"{cleared_portfolio_stats.get('equity_rows', 0)}eq"
        )
    log.info("backtester_mode_switch: " + " | ".join(detail_parts))


def _clear_backtest_result_state() -> None:
    for key in (
        "bt_result",
        "bt_symbol",
        "bt_bar_label",
        "bt_selected_id",
        "bt_params",
        "bt_starting_equity",
        "bt_cost_settings",
        "bt_execution_logic",
        "bt_execution_label",
        "bt_db_msg",
        "bt_restored_msg",
    ):
        st.session_state.pop(key, None)


def _clear_managed_portfolio_result_state() -> None:
    for key in (
        _BT_PORTFOLIO_RESULT_KEY,
        "bt_port_restored_msg",
    ):
        st.session_state.pop(key, None)


def _iso(v) -> str | None:
    if v is None:
        return None
    return pd.Timestamp(v).isoformat()


def _selection_snapshot(symbol: str) -> dict:
    return {
        "symbol": symbol,
        "source": st.session_state.get("bt_source_live") or st.session_state.get("loaded_source"),
        "interval": st.session_state.get("bt_interval_live") or st.session_state.get("loaded_interval"),
        "start": _iso(st.session_state.get("bt_start_live") or st.session_state.get("loaded_start")),
        "end": _iso(st.session_state.get("bt_end_live") or st.session_state.get("loaded_end")),
    }


def _managed_frame_cache_key(
    *,
    source: str,
    symbol: str,
    interval: str,
    start,
    end,
) -> str:
    return "|".join(
        [
            str(source or "").strip().lower(),
            str(symbol or "").strip().upper(),
            str(interval or "").strip().lower(),
            pd.Timestamp(start).isoformat(),
            pd.Timestamp(end).isoformat(),
        ]
    )


def _load_managed_cached_frame(
    *,
    source: str,
    symbol: str,
    interval: str,
    start,
    end,
    local_cache: dict | None = None,
) -> pd.DataFrame:
    cache_key = _managed_frame_cache_key(
        source=source,
        symbol=symbol,
        interval=interval,
        start=start,
        end=end,
    )
    if local_cache is not None and cache_key in local_cache:
        return local_cache[cache_key].copy()

    cached = runtime_cache.get(_BT_MANAGED_FRAME_CACHE_NS, cache_key)
    if cached is not None and not getattr(cached, "empty", True):
        if local_cache is not None:
            local_cache[cache_key] = cached.copy()
        return cached.copy()

    if source == "forward_blend":
        frame = load_forward_blended_data(symbol, interval, pd.Timestamp(start), pd.Timestamp(end))
    elif source == "yfinance":
        frame = load_from_ticker(symbol, interval, pd.Timestamp(start), pd.Timestamp(end))
    else:
        raise ValueError(f"Unsupported managed cache source: {source}")

    if local_cache is not None and frame is not None and not frame.empty:
        local_cache[cache_key] = frame.copy()
    if frame is not None and not frame.empty:
        runtime_cache.put(_BT_MANAGED_FRAME_CACHE_NS, cache_key, frame.copy())
    return frame


def _managed_prepared_cache_key(
    *,
    source: str,
    symbol: str,
    strategy_id: str,
    interval: str,
    start,
    end,
) -> str:
    return "|".join(
        [
            _BT_MANAGED_PREPARED_CACHE_VERSION,
            str(source or "").strip().lower(),
            str(symbol or "").strip().upper(),
            str(strategy_id or "").strip().lower(),
            str(interval or "").strip().lower(),
            pd.Timestamp(start).isoformat(),
            pd.Timestamp(end).isoformat(),
        ]
    )


def _managed_prepared_disk_cache_path(cache_key: str) -> Path:
    digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
    return _BT_MANAGED_PREPARED_DISK_CACHE_DIR / f"{digest}.pkl"


def _load_managed_prepared_frame_from_disk(cache_key: str) -> pd.DataFrame | None:
    path = _managed_prepared_disk_cache_path(cache_key)
    if not path.exists():
        return None
    try:
        with path.open("rb") as fh:
            frame = pickle.load(fh)
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            return frame.copy()
    except Exception:
        return None
    return None


def _save_managed_prepared_frame_to_disk(cache_key: str, frame: pd.DataFrame) -> None:
    try:
        _BT_MANAGED_PREPARED_DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = _managed_prepared_disk_cache_path(cache_key)
        tmp_path = path.with_suffix(".tmp")
        with tmp_path.open("wb") as fh:
            pickle.dump(frame, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(path)
    except Exception:
        return


def _prepare_managed_symbol_input(
    *,
    idx: int,
    symbol: str,
    interval: str,
    start,
    end,
    strategies: list[dict],
    raw_frame_cache: dict[str, pd.DataFrame] | None = None,
    companion_frame_cache: dict | None = None,
    prepared_frame_cache: dict[str, pd.DataFrame] | None = None,
) -> dict:
    symbol = str(symbol or "").strip().upper()
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    strategy_id = recommended_primary_strategy_id(symbol, start=start_ts, end=end_ts)
    strategy_cls = get_strategy(strategy_id)
    strategy = strategy_cls(params={})
    prepared_cache_key = _managed_prepared_cache_key(
        source="forward_blend",
        symbol=symbol,
        strategy_id=strategy_id,
        interval=interval,
        start=start_ts,
        end=end_ts,
    )
    prepared_prices = None
    if prepared_frame_cache is not None:
        prepared_prices = prepared_frame_cache.get(prepared_cache_key)
        if isinstance(prepared_prices, pd.DataFrame) and not prepared_prices.empty:
            prepared_prices = prepared_prices.copy()
        else:
            prepared_prices = None
    if prepared_prices is None:
        cached_prepared = runtime_cache.get(_BT_MANAGED_PREPARED_CACHE_NS, prepared_cache_key)
        if isinstance(cached_prepared, pd.DataFrame) and not cached_prepared.empty:
            prepared_prices = cached_prepared.copy()
    if prepared_prices is None:
        prepared_prices = _load_managed_prepared_frame_from_disk(prepared_cache_key)
    if prepared_prices is None:
        raw_prices = _load_managed_cached_frame(
            source="forward_blend",
            symbol=symbol,
            interval=interval,
            start=start_ts,
            end=end_ts,
            local_cache=raw_frame_cache,
        )
        if raw_prices is None or raw_prices.empty:
            return {"idx": idx, "symbol": symbol, "skipped": True}

        if companion_frame_cache is not None:
            companion_frame_cache[
                (
                    "forward_blend",
                    symbol,
                    interval.strip().lower(),
                    start_ts,
                    end_ts,
                )
            ] = raw_prices.copy()
        prepared_prices = prepare_strategy_data(
            raw_prices,
            strategy,
            primary_symbol=symbol,
            source="forward_blend",
            interval=interval,
            start=start_ts,
            end=end_ts,
            frame_cache=companion_frame_cache,
        )
        if prepared_prices is not None and not prepared_prices.empty:
            runtime_cache.put(_BT_MANAGED_PREPARED_CACHE_NS, prepared_cache_key, prepared_prices.copy())
            _save_managed_prepared_frame_to_disk(prepared_cache_key, prepared_prices.copy())
    elif prepared_prices is not None and not prepared_prices.empty:
        runtime_cache.put(_BT_MANAGED_PREPARED_CACHE_NS, prepared_cache_key, prepared_prices.copy())
    if prepared_prices is not None and not prepared_prices.empty and prepared_frame_cache is not None:
        prepared_frame_cache[prepared_cache_key] = prepared_prices.copy()
    if prepared_prices is None or prepared_prices.empty:
        return {"idx": idx, "symbol": symbol, "skipped": True}

    strategy_name = strategy_display_name_from_id(strategy_id, symbol=symbol, strategies=strategies)
    symbol_input = ManagedPortfolioSymbolInput(
        symbol=symbol,
        strategy_id=strategy_id,
        strategy_name=strategy_name,
        strategy=strategy,
        data=prepared_prices,
    )
    assignment = {
        "Ticker": symbol,
        "Strategy": strategy_name,
        "Bars": f"{len(prepared_prices):,}",
    }
    return {
        "idx": idx,
        "symbol": symbol,
        "skipped": False,
        "symbol_input": symbol_input,
        "assignment": assignment,
    }


def _serialize_trade(t: TradeRecord) -> dict:
    return {
        "id": t.id,
        "symbol": t.symbol,
        "direction": t.direction.value if hasattr(t.direction, "value") else str(t.direction),
        "entry_price": t.entry_price,
        "take_profit": t.take_profit,
        "stop_loss": t.stop_loss,
        "leverage": t.leverage,
        "capital_allocated": t.capital_allocated,
        "entry_time": _iso(t.entry_time),
        "mode": t.mode,
        "strategy_id": t.strategy_id,
        "exit_price": t.exit_price,
        "exit_time": _iso(t.exit_time),
        "outcome": t.outcome.value if hasattr(t.outcome, "value") and t.outcome is not None else t.outcome,
        "leveraged_return_pct": t.leveraged_return_pct,
        "pnl": t.pnl,
        "notes": t.notes,
        "broker_order_id": getattr(t, "broker_order_id", None),
        "broker_status": getattr(t, "broker_status", None),
        "broker_submitted_at": _iso(getattr(t, "broker_submitted_at", None)),
        "filled_qty": getattr(t, "filled_qty", None),
        "filled_avg_price": getattr(t, "filled_avg_price", None),
        "filled_at": _iso(getattr(t, "filled_at", None)),
        "last_synced_at": _iso(getattr(t, "last_synced_at", None)),
    }


def _deserialize_trade(raw: dict) -> TradeRecord:
    return TradeRecord(
        id=raw["id"],
        symbol=raw["symbol"],
        direction=Direction(raw["direction"]),
        entry_price=float(raw["entry_price"]),
        take_profit=raw.get("take_profit"),
        stop_loss=raw.get("stop_loss"),
        leverage=float(raw["leverage"]),
        capital_allocated=float(raw["capital_allocated"]),
        entry_time=pd.Timestamp(raw["entry_time"]).to_pydatetime(),
        mode=raw["mode"],
        strategy_id=raw["strategy_id"],
        exit_price=raw.get("exit_price"),
        exit_time=pd.Timestamp(raw["exit_time"]).to_pydatetime() if raw.get("exit_time") else None,
        outcome=TradeOutcome(raw["outcome"]) if raw.get("outcome") else None,
        leveraged_return_pct=raw.get("leveraged_return_pct"),
        pnl=raw.get("pnl"),
        notes=raw.get("notes", ""),
        broker_order_id=raw.get("broker_order_id"),
        broker_status=raw.get("broker_status"),
        broker_submitted_at=pd.Timestamp(raw["broker_submitted_at"]).to_pydatetime() if raw.get("broker_submitted_at") else None,
        filled_qty=raw.get("filled_qty"),
        filled_avg_price=raw.get("filled_avg_price"),
        filled_at=pd.Timestamp(raw["filled_at"]).to_pydatetime() if raw.get("filled_at") else None,
        last_synced_at=pd.Timestamp(raw["last_synced_at"]).to_pydatetime() if raw.get("last_synced_at") else None,
    )


def _serialize_backtest_result(
    result: BacktestResult,
    *,
    symbol: str,
    bar_label: str,
    selected_id: str,
    params: dict,
    starting_equity: float,
    costs: dict,
    execution_logic: str,
    execution_label: str,
) -> dict:
    eq = result.equity_curve.copy()
    if not eq.empty:
        eq = eq.copy()
        eq["date"] = pd.to_datetime(eq["date"], errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        eq["date"] = eq["date"].fillna("")
    return {
        "selection": _selection_snapshot(symbol),
        "symbol": symbol,
        "bar_label": bar_label,
        "selected_id": selected_id,
        "params": params,
        "starting_equity": float(starting_equity),
        "costs": costs,
        "execution_logic": execution_logic,
        "execution_label": execution_label,
        "result": {
            "trades": [_serialize_trade(t) for t in result.trades],
            "equity_curve": eq.to_dict("records"),
            "total_return_pct": result.total_return_pct,
            "win_rate_pct": result.win_rate_pct,
            "max_drawdown_pct": result.max_drawdown_pct,
            "sharpe_ratio": result.sharpe_ratio,
            "total_trades": result.total_trades,
            "winning_trades": result.winning_trades,
            "losing_trades": result.losing_trades,
            "avg_win_pct": result.avg_win_pct,
            "avg_loss_pct": result.avg_loss_pct,
        },
    }


def _deserialize_backtest_result(payload: dict) -> BacktestResult:
    raw = payload["result"]
    eq_df = pd.DataFrame(raw.get("equity_curve", []))
    if not eq_df.empty and "date" in eq_df.columns:
        eq_df["date"] = pd.to_datetime(eq_df["date"], errors="coerce")
    return BacktestResult(
        trades=[_deserialize_trade(t) for t in raw.get("trades", [])],
        equity_curve=eq_df,
        total_return_pct=float(raw.get("total_return_pct", 0.0)),
        win_rate_pct=float(raw.get("win_rate_pct", 0.0)),
        max_drawdown_pct=float(raw.get("max_drawdown_pct", 0.0)),
        sharpe_ratio=float(raw.get("sharpe_ratio", 0.0)),
        total_trades=int(raw.get("total_trades", 0)),
        winning_trades=int(raw.get("winning_trades", 0)),
        losing_trades=int(raw.get("losing_trades", 0)),
        avg_win_pct=float(raw.get("avg_win_pct", 0.0)),
        avg_loss_pct=float(raw.get("avg_loss_pct", 0.0)),
    )


def _serialize_managed_portfolio_snapshot(payload: dict) -> dict:
    result = payload["result"]
    eq = result.equity_curve.copy()
    if not eq.empty:
        eq = eq.copy()
        eq["date"] = pd.to_datetime(eq["date"], errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        eq["date"] = eq["date"].fillna("")
    return {
        "symbols": list(payload.get("symbols") or []),
        "interval": payload.get("interval"),
        "start": _iso(payload.get("start")),
        "end": _iso(payload.get("end")),
        "starting_equity": float(payload.get("starting_equity", 0.0)),
        "capital_per_trade": float(payload.get("capital_per_trade", 0.0)),
        "leverage": float(payload.get("leverage", 1.0)),
        "max_open_positions": int(payload.get("max_open_positions", 1)),
        "execution_label": payload.get("execution_label"),
        "execution_logic": payload.get("execution_logic"),
        "enforce_ssr": bool(payload.get("enforce_ssr", True)),
        "dynamic_sizing": bool(payload.get("dynamic_sizing", False)),
        "assignments": list(payload.get("assignments") or []),
        "skipped_symbols": list(payload.get("skipped_symbols") or []),
        "result": {
            "trades": [_serialize_trade(t) for t in result.trades],
            "equity_curve": eq.to_dict("records"),
            "total_return_pct": result.total_return_pct,
            "win_rate_pct": result.win_rate_pct,
            "max_drawdown_pct": result.max_drawdown_pct,
            "sharpe_ratio": result.sharpe_ratio,
            "total_trades": result.total_trades,
            "winning_trades": result.winning_trades,
            "losing_trades": result.losing_trades,
            "avg_win_pct": result.avg_win_pct,
            "avg_loss_pct": result.avg_loss_pct,
            "symbol_strategy_map": dict(result.symbol_strategy_map or {}),
            "candidate_entries": int(result.candidate_entries),
            "skipped_entries": int(result.skipped_entries),
            "replaced_positions": int(result.replaced_positions),
            "max_concurrent_positions_seen": int(result.max_concurrent_positions_seen),
        },
    }


def _deserialize_managed_portfolio_snapshot(payload: dict) -> dict:
    raw = payload["result"]
    eq_df = pd.DataFrame(raw.get("equity_curve", []))
    if not eq_df.empty and "date" in eq_df.columns:
        eq_df["date"] = pd.to_datetime(eq_df["date"], errors="coerce")
    result = ManagedPortfolioBacktestResult(
        trades=[_deserialize_trade(t) for t in raw.get("trades", [])],
        equity_curve=eq_df,
        total_return_pct=float(raw.get("total_return_pct", 0.0)),
        win_rate_pct=float(raw.get("win_rate_pct", 0.0)),
        max_drawdown_pct=float(raw.get("max_drawdown_pct", 0.0)),
        sharpe_ratio=float(raw.get("sharpe_ratio", 0.0)),
        total_trades=int(raw.get("total_trades", 0)),
        winning_trades=int(raw.get("winning_trades", 0)),
        losing_trades=int(raw.get("losing_trades", 0)),
        avg_win_pct=float(raw.get("avg_win_pct", 0.0)),
        avg_loss_pct=float(raw.get("avg_loss_pct", 0.0)),
        symbol_strategy_map=dict(raw.get("symbol_strategy_map") or {}),
        candidate_entries=int(raw.get("candidate_entries", 0)),
        skipped_entries=int(raw.get("skipped_entries", 0)),
        replaced_positions=int(raw.get("replaced_positions", 0)),
        max_concurrent_positions_seen=int(raw.get("max_concurrent_positions_seen", 0)),
    )
    return {
        "result": result,
        "symbols": list(payload.get("symbols") or []),
        "interval": payload.get("interval"),
        "start": pd.Timestamp(payload["start"]) if payload.get("start") else None,
        "end": pd.Timestamp(payload["end"]) if payload.get("end") else None,
        "starting_equity": float(payload.get("starting_equity", 0.0)),
        "capital_per_trade": float(payload.get("capital_per_trade", 0.0)),
        "leverage": float(payload.get("leverage", 1.0)),
        "max_open_positions": int(payload.get("max_open_positions", 1)),
        "execution_label": payload.get("execution_label"),
        "execution_logic": payload.get("execution_logic"),
        "enforce_ssr": bool(payload.get("enforce_ssr", True)),
        "dynamic_sizing": bool(payload.get("dynamic_sizing", False)),
        "assignments": list(payload.get("assignments") or []),
        "skipped_symbols": list(payload.get("skipped_symbols") or []),
    }


def _persist_backtest_snapshot(payload: dict) -> None:
    try:
        _db().save_config(_BT_RESULT_CFG_KEY, payload)
    except Exception:
        pass


def _strategy_name_from_id(strategy_id: str | None) -> str | None:
    if not strategy_id:
        return None
    try:
        for item in list_strategies():
            if item.get("id") == strategy_id:
                return item.get("name")
    except Exception:
        return None
    return None


def _restore_backtest_snapshot() -> bool:
    if "bt_result" in st.session_state:
        return True
    try:
        payload = _db().load_config(_BT_RESULT_CFG_KEY) or {}
    except Exception:
        payload = {}
    if not payload:
        return False
    current_sel = _selection_snapshot(str(st.session_state.get("loaded_symbol", "")))
    saved_sel = payload.get("selection", {})
    if current_sel != saved_sel:
        return False
    try:
        st.session_state["bt_result"] = _deserialize_backtest_result(payload)
        st.session_state["bt_symbol"] = payload.get("symbol", current_sel.get("symbol", "DATA"))
        st.session_state["bt_bar_label"] = payload.get("bar_label", "bars")
        st.session_state["bt_selected_id"] = payload.get("selected_id", "")
        restored_strategy_name = _strategy_name_from_id(payload.get("selected_id"))
        if restored_strategy_name:
            st.session_state["bt_strategy"] = restored_strategy_name
        st.session_state["bt_params"] = payload.get("params", {})
        st.session_state["bt_starting_equity"] = float(payload.get("starting_equity", 1000.0))
        st.session_state["bt_cost_settings"] = payload.get("costs", {})
        st.session_state["bt_execution_logic"] = payload.get("execution_logic", "alpaca")
        st.session_state["bt_execution_label"] = payload.get("execution_label", "Alpaca-realistic")
        st.session_state["bt_restored_msg"] = "Restored the last backtest result for this dataset."
        return True
    except Exception:
        return False


def _persist_managed_portfolio_snapshot(payload: dict) -> None:
    try:
        _db().save_config(_BT_PORTFOLIO_CFG_KEY, _serialize_managed_portfolio_snapshot(payload))
    except Exception:
        pass


def _restore_managed_portfolio_snapshot() -> bool:
    if _BT_PORTFOLIO_RESULT_KEY in st.session_state:
        return True
    try:
        payload = _db().load_config(_BT_PORTFOLIO_CFG_KEY) or {}
    except Exception:
        payload = {}
    if not payload:
        return False
    try:
        st.session_state[_BT_PORTFOLIO_RESULT_KEY] = _deserialize_managed_portfolio_snapshot(payload)
        st.session_state["bt_port_restored_msg"] = "Restored the last managed portfolio backtest result."
        return True
    except Exception:
        return False


def _resolve_gld_candidate_for_leverage(leverage: float) -> tuple[str | None, dict]:
    lev = round(float(leverage), 4)
    if lev == 1.0:
        name = "gld_best_1x_sweep_20260419"
    elif lev == 5.0:
        name = "gld_best_leverage_5x_tuned_20260419"
    else:
        return None, {}
    return name, get_candidate(name)


def _set_backtest_dataset(
    prices: pd.DataFrame,
    *,
    symbol: str,
    source: str | None,
    interval: str | None,
    start,
    end,
) -> None:
    runtime_cache.put(_BT_RUNTIME_CACHE_NS, _BT_RUNTIME_CACHE_KEY, prices)
    st.session_state.pop("bt_prices_live", None)
    st.session_state["bt_symbol_live"] = str(symbol or "DATA").upper()
    st.session_state["bt_source_live"] = source
    st.session_state["bt_interval_live"] = interval
    st.session_state["bt_start_live"] = pd.Timestamp(start) if start is not None else None
    st.session_state["bt_end_live"] = pd.Timestamp(end) if end is not None else None


def _get_backtest_dataset() -> pd.DataFrame | None:
    prices = st.session_state.get("bt_prices_live")
    if prices is not None:
        return prices
    cached = runtime_cache.get(_BT_RUNTIME_CACHE_NS, _BT_RUNTIME_CACHE_KEY)
    if isinstance(cached, pd.DataFrame):
        return cached
    return None


def _is_backtest_intraday_interval(interval: str | None) -> bool:
    if not interval:
        return False
    key = str(interval).lower()
    return any(token in key for token in ("m", "h", "min", "hour")) and "day" not in key and "wk" not in key


def _load_backtest_target(
    *,
    symbol: str,
    interval: str | None,
    start_ts: pd.Timestamp | None,
    end_ts: pd.Timestamp | None,
    strategy_id: str | None,
    status_cb: Callable[[str], None] | None = None,
    progress_cb: Callable[[float, str], None] | None = None,
) -> tuple[pd.DataFrame, str, str | None, str | None, pd.Timestamp | None, pd.Timestamp | None]:
    sym = str(symbol or "").upper().strip()
    if not sym:
        raise ValueError("Ticker cannot be empty.")
    if start_ts is None or end_ts is None:
        raise ValueError("Start and end dates are required.")
    if end_ts < start_ts:
        raise ValueError("End date must be on or after start date.")

    end_exclusive = pd.Timestamp(end_ts) + pd.Timedelta(days=1)

    if not interval:
        raise ValueError("Interval is required.")

    if _is_backtest_intraday_interval(interval):
        if progress_cb is not None:
            progress_cb(0.12, f"Requesting intraday {interval} bars for {sym}")
        if status_cb is not None:
            status_cb(f"Fetching intraday bars for {sym} via the blended loader…")
        data = load_forward_blended_data(
            sym,
            interval,
            pd.Timestamp(start_ts),
            end_exclusive,
            lookback=None,
        )
        source_key = "forward_blend"
    else:
        if progress_cb is not None:
            progress_cb(0.12, f"Requesting {interval} bars for {sym}")
        if status_cb is not None:
            status_cb(f"Fetching historical bars for {sym} from Yahoo/cache…")
        data = load_from_ticker(sym, interval, pd.Timestamp(start_ts), end_exclusive)
        source_key = "yfinance"

    if progress_cb is not None:
        progress_cb(0.55, f"Loaded {len(data):,} raw bars for {sym}")
    if status_cb is not None:
        status_cb(
            f"Loaded {len(data):,} raw bars. "
            "Stage 2 will prepare the strategy context and companion data."
        )
    if progress_cb is not None:
        progress_cb(1.0, f"Loaded {len(data):,} raw bars for {sym}")
    actual_end = pd.Timestamp(data["date"].max()) if not data.empty else end_exclusive
    return data, sym, source_key, interval, pd.Timestamp(start_ts), actual_end


def _apply_backtest_target(
    *,
    symbol: str,
    interval: str | None,
    start_ts: pd.Timestamp | None,
    end_ts: pd.Timestamp | None,
    strategy_id: str | None,
    status_cb: Callable[[str], None] | None = None,
    progress_cb: Callable[[float, str], None] | None = None,
) -> tuple[bool, str]:
    data, actual_symbol, actual_source, actual_interval, actual_start, actual_end = _load_backtest_target(
        symbol=symbol,
        interval=interval,
        start_ts=start_ts,
        end_ts=end_ts,
        strategy_id=strategy_id,
        status_cb=status_cb,
        progress_cb=progress_cb,
    )
    _set_backtest_dataset(
        data,
        symbol=actual_symbol,
        source=actual_source,
        interval=actual_interval,
        start=actual_start,
        end=actual_end,
    )
    return (
        True,
        f"Loaded backtest target: {actual_symbol} · {len(data):,} bars"
        + (f" · {actual_interval}" if actual_interval else ""),
    )


def startup_preload(status_cb=None) -> dict:
    """
    Warm the last saved Backtester target into session state.

    This is intentionally data-only: it restores the target dataset so the
    Backtester page opens warm, but it does not run a backtest automatically.
    """
    try:
        payload = _db().load_config(_BT_RESULT_CFG_KEY) or {}
        portfolio_payload = _db().load_config(_BT_PORTFOLIO_CFG_KEY) or {}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    selection = payload.get("selection") or {}
    symbol = str(selection.get("symbol") or "").upper().strip()
    interval = selection.get("interval")
    start_raw = selection.get("start")
    end_raw = selection.get("end")
    strategy_id = payload.get("selected_id")

    if not symbol or not interval or not start_raw or not end_raw:
        if portfolio_payload:
            _restore_managed_portfolio_snapshot()
            warm_summary = _startup_warm_managed_portfolio_states(portfolio_payload, status_cb=status_cb)
            message = "Restored the last managed portfolio snapshot."
            warmed = int(warm_summary.get("warmed", 0) or 0)
            if warmed > 0:
                message += f" Primed {warmed} managed ticker state(s)."
            if status_cb is not None:
                status_cb(message)
            return {"ok": True, "loaded": 0, "warmed": warmed, "message": message}
        return {"ok": True, "loaded": 0, "message": "No saved Backtester target to preload."}

    start_ts = pd.Timestamp(start_raw)
    end_ts = pd.Timestamp(end_raw)
    if status_cb is not None:
        status_cb(f"Backtester: loading {symbol} {interval} from {start_ts.date()} to {end_ts.date()}")

    data, actual_symbol, actual_source, actual_interval, actual_start, actual_end = _load_backtest_target(
        symbol=symbol,
        interval=interval,
        start_ts=start_ts,
        end_ts=end_ts,
        strategy_id=strategy_id,
    )
    _set_backtest_dataset(
        data,
        symbol=actual_symbol,
        source=actual_source,
        interval=actual_interval,
        start=actual_start,
        end=actual_end,
    )
    _restore_backtest_snapshot()
    _restore_managed_portfolio_snapshot()
    warm_summary = _startup_warm_managed_portfolio_states(portfolio_payload, status_cb=status_cb)
    return {
        "ok": True,
        "loaded": int(len(data)),
        "symbol": actual_symbol,
        "interval": actual_interval,
        "warmed": int(warm_summary.get("warmed", 0) or 0),
    }


def _startup_warm_managed_portfolio_states(portfolio_payload: dict | None, status_cb=None) -> dict:
    payload = portfolio_payload or {}
    symbols = [
        str(sym or "").strip().upper()
        for sym in (payload.get("symbols") or [])
        if str(sym or "").strip()
    ]
    interval = str(payload.get("interval") or "").strip()
    start_raw = payload.get("start")
    end_raw = payload.get("end")
    if not symbols or not interval or not start_raw or not end_raw:
        return {"ok": True, "warmed": 0, "message": "No managed portfolio state to warm."}

    start_ts = pd.Timestamp(start_raw)
    end_ts = pd.Timestamp(end_raw)
    strategies = list_strategies()
    raw_frame_cache: dict[str, pd.DataFrame] = {}
    companion_frame_cache: dict = {}
    prepared_frame_cache: dict[str, pd.DataFrame] = {}
    total_symbols = len(symbols)
    max_workers = min(8, total_symbols)
    if status_cb is not None:
        status_cb(
            f"Backtester: priming managed portfolio states for {total_symbols} ticker(s) "
            f"({interval} · {start_ts.date()} → {end_ts.date()})"
        )

    prepared_results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="bt-port-startup") as executor:
        future_map = {
            executor.submit(
                _prepare_managed_symbol_input,
                idx=idx,
                symbol=sym,
                interval=interval,
                start=start_ts,
                end=end_ts,
                strategies=strategies,
                raw_frame_cache=raw_frame_cache,
                companion_frame_cache=companion_frame_cache,
                prepared_frame_cache=prepared_frame_cache,
            ): sym
            for idx, sym in enumerate(symbols)
        }
        completed = 0
        for future in as_completed(future_map):
            sym = future_map[future]
            completed += 1
            try:
                prepared = future.result()
            except Exception:
                prepared = {"symbol": sym, "skipped": True}
            prepared_results.append(prepared)
            if status_cb is not None and (
                completed >= total_symbols
                or completed % max(1, total_symbols // 6) == 0
            ):
                status_cb(f"Backtester: prepared managed ticker {completed}/{total_symbols}")

    prepared_results.sort(key=lambda item: int(item.get("idx", 0)))
    symbol_inputs: list[ManagedPortfolioSymbolInput] = []
    for prepared in prepared_results:
        if prepared.get("skipped"):
            continue
        symbol_input = prepared.get("symbol_input")
        if isinstance(symbol_input, ManagedPortfolioSymbolInput):
            symbol_inputs.append(symbol_input)

    if not symbol_inputs:
        return {"ok": True, "warmed": 0, "message": "No managed ticker states could be prepared."}

    engine = ManagedPortfolioBacktestEngine()
    warmed = 0
    total_prepared = len(symbol_inputs)
    for idx, symbol_input in enumerate(symbol_inputs, start=1):
        try:
            engine._prepare_symbol_state(symbol_input)
            warmed += 1
            if status_cb is not None and (
                idx >= total_prepared
                or idx % max(1, total_prepared // 6) == 0
            ):
                status_cb(f"Backtester: warmed managed state {idx}/{total_prepared}")
        except Exception as exc:
            if status_cb is not None:
                status_cb(f"Backtester: skipped managed state warm for {symbol_input.symbol} ({exc})")

    return {
        "ok": True,
        "warmed": warmed,
        "message": f"Warmed {warmed} managed ticker state(s).",
    }


def _render_backtest_target_loading_shell(payload: dict) -> None:
    symbol = str(payload.get("symbol") or "").upper()
    interval = str(payload.get("interval") or "")
    start_ts = pd.Timestamp(payload.get("start_ts")) if payload.get("start_ts") is not None else None
    end_ts = pd.Timestamp(payload.get("end_ts")) if payload.get("end_ts") is not None else None

    st.subheader("Single-Ticker Setup")
    st.caption(
        "Preparing the requested target dataset before the strategy run starts."
    )
    details: list[str] = []
    if symbol:
        details.append(f"**Ticker:** {symbol}")
    if interval:
        details.append(f"**Interval:** {interval}")
    if start_ts is not None and end_ts is not None:
        details.append(
            f"**Window:** {start_ts.date()} → {end_ts.date()}"
        )
    if details:
        st.markdown(" · ".join(details))


def _render_backtest_target_selector() -> dict:
    st.subheader("Single-Ticker Setup")
    st.caption(
        "Choose the exact ticker, interval, and dates for the backtest here. "
        "The loader automatically reuses cache, pulls the most recent intraday bars from Yahoo when useful, "
        "and fills older intraday history from Alpaca when needed."
    )

    pending_strategy_name = st.session_state.pop(_BT_PENDING_STRATEGY_NAME_KEY, None)
    if pending_strategy_name:
        st.session_state["bt_strategy"] = pending_strategy_name

    current_symbol = (
        st.session_state.get("bt_symbol_live")
        or st.session_state.get("loaded_symbol")
        or "UVXY"
    )
    current_start = st.session_state.get("bt_start_live") or st.session_state.get("loaded_start")
    current_end = st.session_state.get("bt_end_live") or st.session_state.get("loaded_end")

    strategies = ordered_strategy_items(list_strategies())
    strategy_names = [item["name"] for item in strategies]
    name_to_id = {item["name"]: item["id"] for item in strategies}
    default_strategy_name = st.session_state.get("bt_strategy") or "Bollinger + RSI (Spike-Aware)"
    default_strategy_name = _map_legacy_earnings_strategy_name(default_strategy_name, strategy_names)
    if default_strategy_name not in strategy_names and strategy_names:
        default_strategy_name = strategy_names[0]
    strategy_symbol = str(current_symbol).upper()
    auto_select_strategy_name_for_state(
        "bt_strategy",
        symbol=strategy_symbol,
        strategies=strategies,
        start=current_start,
        end=current_end,
    )
    selected_target_strategy_name = str(
        st.selectbox(
            "Strategy",
            strategy_names,
            key="bt_strategy",
            format_func=lambda name: strategy_option_label_from_name(name, strategies),
        )
    )
    selected_target_strategy_id = name_to_id.get(selected_target_strategy_name)
    render_strategy_selection_help(selected_target_strategy_id, symbol=strategy_symbol, start=current_start, end=current_end)

    action = {"run_requested": False, "submitted": False, "payload": None}
    with st.form("bt_target_form", clear_on_submit=False):
        tcol1, tcol2, tcol3, tcol4 = st.columns(4)
        with tcol1:
            bt_symbol = st.text_input(
                "Ticker",
                value=str(current_symbol).upper(),
                key="bt_target_symbol",
            )
        with tcol2:
            interval_options = ["1m", "5m", "15m", "30m", "1h", "1d", "1wk"]
            current_interval = st.session_state.get("bt_interval_live") or st.session_state.get("loaded_interval") or "1m"
            default_interval = current_interval if current_interval in interval_options else "1m"
            bt_interval = st.selectbox(
                "Interval",
                interval_options,
                index=interval_options.index(default_interval),
                key="bt_target_interval",
            )
        with tcol3:
            bt_start = st.date_input(
                "Start",
                value=_BT_DEFAULT_START,
                key="bt_target_start",
            )
        with tcol4:
            bt_end = st.date_input(
                "End",
                value=_BT_DEFAULT_END,
                key="bt_target_end",
            )

        st.caption(
            "Target fields stay grouped so you can edit them without triggering a full page rerun on every keystroke. "
            "Sidebar fetches are still useful for warming cache or working offline. This button loads the target and runs the backtest in one step."
        )
        st.caption(
            "On submit, the selected ticker will automatically switch to its recommended strategy before the backtest runs."
        )
        run_target_clicked = st.form_submit_button("Run Backtest", type="primary")

    if run_target_clicked:
        submitted_symbol = str(bt_symbol or "").strip().upper() or str(current_symbol).upper()
        submitted_start = pd.Timestamp(bt_start)
        submitted_end = pd.Timestamp(bt_end)
        recommended_id = recommended_primary_strategy_id(
            submitted_symbol,
            start=submitted_start,
            end=submitted_end,
        )
        recommended_name = strategy_display_name_from_id(recommended_id) or selected_target_strategy_name
        recommended_name = _map_legacy_earnings_strategy_name(recommended_name, strategy_names)
        if recommended_name not in strategy_names:
            recommended_name = selected_target_strategy_name
        recommended_id = name_to_id.get(recommended_name, selected_target_strategy_id)
        st.session_state[_BT_PENDING_STRATEGY_NAME_KEY] = recommended_name
        action["submitted"] = True
        action["payload"] = {
            "symbol": submitted_symbol,
            "interval": bt_interval,
            "start_ts": submitted_start,
            "end_ts": submitted_end,
            "strategy_name": recommended_name,
            "strategy_id": recommended_id,
        }

    return action


def _downsample(df: pd.DataFrame, max_pts: int = _MAX_CHART_PTS) -> pd.DataFrame:
    if len(df) <= max_pts:
        return df
    step = max(1, len(df) // max_pts)
    return df.iloc[::step].reset_index(drop=True)


def _is_signal_exit(outcome: str) -> bool:
    return any(k in outcome for k in ("overbought", "oversold", "Counter"))


def _calc_rsi(series: pd.Series, period: int) -> pd.Series:
    d = series.diff()
    g = d.clip(lower=0)
    l = (-d).clip(lower=0)
    ag = g.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    al = l.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return 100 - (100 / (1 + ag / al.replace(0, float("nan"))))


def _parse_levels(raw) -> list[float]:
    if isinstance(raw, (int, float)):
        return [float(raw)]
    s = str(raw).strip().lower()
    if s in ("", "none", "off", "-"):
        return []
    try:
        return sorted(float(p.strip()) for p in s.replace(";", ",").split(",") if p.strip())
    except ValueError:
        return []


def _bar_label(prices: pd.DataFrame) -> str:
    if len(prices) < 2:
        return "bars"

    # Use the dominant spacing, not the first two rows. Intraday data can start
    # with a missing/opening gap, which made 1-min caches display as "4-min".
    dates = pd.to_datetime(prices["date"], errors="coerce").dropna().sort_values()
    if len(dates) < 2:
        return "bars"
    diffs = dates.diff().dropna().dt.total_seconds()
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return "bars"

    mode = diffs.mode()
    delta = int(round(float(mode.iloc[0] if not mode.empty else diffs.median())))
    return {
        60: "1-min",
        300: "5-min",
        900: "15-min",
        1800: "30-min",
        3600: "1-hour",
        86400: "1-day",
    }.get(delta, f"{max(1, int(round(delta / 60)))}-min")


def _trade_regime_from_notes(notes: str) -> str:
    raw = notes or ""
    if "regime=" not in raw:
        return "unknown"
    return raw.split("regime=")[1].split(" | ")[0]


def _trade_regime(trade) -> str:
    return _trade_regime_from_notes(getattr(trade, "notes", "") or "")


def _trade_events(trades) -> tuple[pd.DataFrame, pd.DataFrame]:
    entries, exits = [], []
    for i, t in enumerate(trades):
        direction = t.direction.value if hasattr(t.direction, "value") else str(t.direction)
        outcome = t.outcome.value if hasattr(t.outcome, "value") else str(t.outcome)
        ret = t.leveraged_return_pct
        label = f"T{i + 1}"
        entries.append(
            {
                "date": pd.Timestamp(t.entry_time),
                "price": t.entry_price,
                "direction": direction,
                "outcome": outcome,
                "return_pct": ret,
                "trade_n": label,
            }
        )
        if t.exit_time and t.exit_price is not None:
            exits.append(
                {
                    "date": pd.Timestamp(t.exit_time),
                    "price": t.exit_price,
                    "direction": direction,
                    "outcome": outcome,
                    "return_pct": ret,
                    "trade_n": label,
                    "win": (ret or 0) > 0,
                }
            )
    return pd.DataFrame(entries), pd.DataFrame(exits)


def _price_chart(prices, trades, symbol, show_long, show_short, show_tp, show_sl, show_trail, show_sig):
    base = alt.Chart(prices).mark_line(color=_theme_chart_color("primary"), strokeWidth=1.2).encode(
        x=alt.X("date:T", title="Date / Time", axis=alt.Axis(**_axis_cfg())),
        y=alt.Y("close:Q", title="Price", scale=alt.Scale(zero=False), axis=alt.Axis(**_axis_cfg())),
        tooltip=["date:T", alt.Tooltip("close:Q", format=".4f")],
    )
    layers = [base]
    if not trades:
        return (
            alt.layer(*layers)
            .properties(title=alt.TitleParams(f"{symbol} – Price", **_title_cfg()), height=320)
            .configure(background="#0c0d14").configure_view(fill="#181a25", strokeOpacity=0)
            .configure_axis(**_axis_cfg())
            .configure_title(**_title_cfg())
        )
    entry_df, exit_df = _trade_events(trades)
    entry_df = clip_frame_to_price_window(entry_df, prices)
    exit_df = clip_frame_to_price_window(exit_df, prices)
    tt_e = [
        "date:T",
        "trade_n:N",
        "direction:N",
        alt.Tooltip("price:Q", format=".4f", title="Entry"),
        "outcome:N",
        alt.Tooltip("return_pct:Q", format=".2f", title="Return %"),
    ]
    if not entry_df.empty:
        long_e = entry_df[entry_df["direction"] == "Long"].copy()
        short_e = entry_df[entry_df["direction"] == "Short"].copy()
        if show_long and not long_e.empty:
            long_e["y"] = long_e["price"] * 0.997
            layers.append(
                alt.Chart(long_e)
                .mark_point(shape="triangle-up", size=120, filled=True, color=_GREEN)
                .encode(x="date:T", y="y:Q", tooltip=tt_e)
            )
        if show_short and not short_e.empty:
            short_e["y"] = short_e["price"] * 1.003
            layers.append(
                alt.Chart(short_e)
                .mark_point(shape="triangle-down", size=120, filled=True, color=_RED)
                .encode(x="date:T", y="y:Q", tooltip=tt_e)
            )
    if not exit_df.empty:
        tt_x = [
            "date:T",
            "trade_n:N",
            "direction:N",
            alt.Tooltip("price:Q", format=".4f", title="Exit"),
            "outcome:N",
            alt.Tooltip("return_pct:Q", format=".2f", title="Return %"),
        ]
        tp_ex = exit_df[exit_df["outcome"] == "TP hit"]
        sl_ex = exit_df[exit_df["outcome"] == "SL hit"]
        trail_ex = exit_df[exit_df["outcome"] == "Trail stop"]
        sig_ex = exit_df[exit_df["outcome"].apply(_is_signal_exit)]
        if show_tp and not tp_ex.empty:
            layers.append(
                alt.Chart(tp_ex)
                .mark_point(shape="cross", size=110, strokeWidth=2.5, color=_GREEN)
                .encode(x="date:T", y="price:Q", tooltip=tt_x)
            )
        if show_sl and not sl_ex.empty:
            layers.append(
                alt.Chart(sl_ex)
                .mark_point(shape="cross", size=110, strokeWidth=2.5, color=_RED)
                .encode(x="date:T", y="price:Q", tooltip=tt_x)
            )
        if show_trail and not trail_ex.empty:
            layers.append(
                alt.Chart(trail_ex)
                .mark_point(shape="diamond", size=100, filled=True, color=_PURPLE)
                .encode(x="date:T", y="price:Q", tooltip=tt_x)
            )
        if show_sig and not sig_ex.empty:
            layers.append(
                alt.Chart(sig_ex)
                .mark_point(shape="cross", size=110, strokeWidth=2.5, color=_ORANGE)
                .encode(x="date:T", y="price:Q", tooltip=tt_x)
            )
    return (
        alt.layer(*layers)
        .properties(title=alt.TitleParams(f"{symbol} – Price  ▲ Long  ▼ Short  ✕ Exit", **_title_cfg()), height=320)
        .configure(background="#0c0d14").configure_view(fill="#181a25", strokeOpacity=0)
        .configure_axis(**_axis_cfg())
        .configure_title(**_title_cfg())
    )


def _rsi_chart(prices, trades, period, buy_levels, sell_levels, symbol, show_long, show_short, show_tp, show_sl, show_trail, show_sig):
    rsi_s = _calc_rsi(prices["close"], period).rename("rsi")
    df = pd.concat([prices[["date"]], rsi_s], axis=1).dropna()
    rsi_line = alt.Chart(df).mark_line(color=_theme_chart_color("secondary"), strokeWidth=1.8).encode(
        x=alt.X("date:T", title="Date / Time", axis=alt.Axis(**_axis_cfg())),
        y=alt.Y("rsi:Q", title="RSI", scale=alt.Scale(domain=[0, 100]), axis=alt.Axis(**_axis_cfg())),
        tooltip=["date:T", alt.Tooltip("rsi:Q", format=".2f")],
    )
    layers = [rsi_line]
    for lvl in buy_levels:
        ldf = pd.DataFrame({"y": [lvl], "label": [f"OS {lvl:.0f}"]})
        layers += [
            alt.Chart(ldf).mark_rule(color=_GREEN, strokeDash=[5, 3], strokeWidth=1.5).encode(y="y:Q"),
            alt.Chart(ldf).mark_text(align="left", dx=4, dy=-7, fontSize=12, color=_GREEN, fontWeight="bold").encode(y="y:Q", x=alt.value(4), text="label:N"),
        ]
    for lvl in sell_levels:
        ldf = pd.DataFrame({"y": [lvl], "label": [f"OB {lvl:.0f}"]})
        layers += [
            alt.Chart(ldf).mark_rule(color=_RED, strokeDash=[5, 3], strokeWidth=1.5).encode(y="y:Q"),
            alt.Chart(ldf).mark_text(align="left", dx=4, dy=-7, fontSize=12, color=_RED, fontWeight="bold").encode(y="y:Q", x=alt.value(4), text="label:N"),
        ]
    if buy_levels:
        layers.append(alt.Chart(pd.DataFrame({"y1": [0], "y2": [min(buy_levels)]})).mark_rect(color=_GREEN, opacity=0.07).encode(y="y1:Q", y2="y2:Q"))
    if sell_levels:
        layers.append(alt.Chart(pd.DataFrame({"y1": [max(sell_levels)], "y2": [100]})).mark_rect(color=_RED, opacity=0.07).encode(y="y1:Q", y2="y2:Q"))

    def _snap(ts):
        idx = min(df["date"].searchsorted(pd.Timestamp(ts)), len(df) - 1)
        row = df.iloc[idx]
        return (float(row["rsi"]) if not pd.isna(row["rsi"]) else 50.0), row["date"]

    if trades:
        entry_df, exit_df = _trade_events(trades)
        entry_df = clip_frame_to_price_window(entry_df, prices)
        exit_df = clip_frame_to_price_window(exit_df, prices)
        tt = ["date:T", "trade_n:N", "direction:N", alt.Tooltip("rsi_val:Q", format=".1f", title="RSI"), "outcome:N", alt.Tooltip("return_pct:Q", format=".2f", title="Return %")]
        if not entry_df.empty:
            entry_df[["rsi_val", "date"]] = pd.DataFrame([_snap(r) for r in entry_df["date"]], columns=["rsi_val", "date"])
            long_e = entry_df[entry_df["direction"] == "Long"].copy()
            short_e = entry_df[entry_df["direction"] == "Short"].copy()
            if show_long and not long_e.empty:
                long_e["y"] = long_e["rsi_val"] - 5
                layers.append(alt.Chart(long_e).mark_point(shape="triangle-up", size=90, filled=True, color=_GREEN).encode(x="date:T", y="y:Q", tooltip=tt))
            if show_short and not short_e.empty:
                short_e["y"] = short_e["rsi_val"] + 5
                layers.append(alt.Chart(short_e).mark_point(shape="triangle-down", size=90, filled=True, color=_RED).encode(x="date:T", y="y:Q", tooltip=tt))
        if not exit_df.empty:
            exit_df[["rsi_val", "date"]] = pd.DataFrame([_snap(r) for r in exit_df["date"]], columns=["rsi_val", "date"])
            tp_ex = exit_df[exit_df["outcome"] == "TP hit"]
            sl_ex = exit_df[exit_df["outcome"] == "SL hit"]
            trail_ex = exit_df[exit_df["outcome"] == "Trail stop"]
            sig_ex = exit_df[exit_df["outcome"].apply(_is_signal_exit)]
            if show_tp and not tp_ex.empty:
                layers.append(alt.Chart(tp_ex).mark_point(shape="cross", size=90, strokeWidth=2.5, color=_GREEN).encode(x="date:T", y="rsi_val:Q", tooltip=tt))
            if show_sl and not sl_ex.empty:
                layers.append(alt.Chart(sl_ex).mark_point(shape="cross", size=90, strokeWidth=2.5, color=_RED).encode(x="date:T", y="rsi_val:Q", tooltip=tt))
            if show_trail and not trail_ex.empty:
                layers.append(alt.Chart(trail_ex).mark_point(shape="diamond", size=80, filled=True, color=_PURPLE).encode(x="date:T", y="rsi_val:Q", tooltip=tt))
            if show_sig and not sig_ex.empty:
                layers.append(alt.Chart(sig_ex).mark_point(shape="cross", size=90, strokeWidth=2.5, color=_ORANGE).encode(x="date:T", y="rsi_val:Q", tooltip=tt))
    return (
        alt.layer(*layers)
        .properties(title=alt.TitleParams(f"{symbol} – RSI ({period})  Buy≤{buy_levels}  Sell≥{sell_levels}", **_title_cfg()), height=300)
        .configure(background="#0c0d14").configure_view(fill="#181a25", strokeOpacity=0)
        .configure_axis(**_axis_cfg())
        .configure_title(**_title_cfg())
    )


def _equity_chart(equity_curve: pd.DataFrame, symbol: str, window: str = "All"):
    if equity_curve is None or equity_curve.empty:
        return alt.Chart(pd.DataFrame()).mark_line()
    eq_df = equity_curve.copy()
    eq_df["date"] = pd.to_datetime(eq_df["date"])
    eq_df = filter_chart_window(eq_df, window)
    line = alt.Chart(eq_df).mark_line(color=_theme_chart_color("primary"), strokeWidth=2).encode(
        x=alt.X("date:T", title="Date / Time", axis=alt.Axis(**_axis_cfg())),
        y=alt.Y("equity:Q", title="Equity ($)", scale=alt.Scale(zero=False), axis=alt.Axis(**_axis_cfg())),
        tooltip=["date:T", alt.Tooltip("equity:Q", format="$,.2f", title="Portfolio Value")],
    )
    return alt.layer(line).properties(title=alt.TitleParams(f"{symbol} – Portfolio Equity", **_title_cfg()), height=300).configure(background="#0c0d14").configure_view(fill="#181a25", strokeOpacity=0).configure_axis(**_axis_cfg()).configure_title(**_title_cfg())


@st.cache_data(show_spinner=False)
def _cached_gld_fair_value(start: str | None, end: str | None, cache_fingerprint: str):
    try:
        diagnostics = compute_gld_fair_value_diagnostics(start=start, end=end)
    except FileNotFoundError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    if diagnostics is None:
        return None
    return {
        "frame": diagnostics.frame,
        "stats": diagnostics.stats,
        "model": diagnostics.model,
        "symbol": diagnostics.symbol,
    }


def _fair_value_chart(frame: pd.DataFrame, symbol: str):
    plot_df = frame.copy()
    value_cols = ["actual", "fair_value"]
    if "structural_fair_value" in plot_df.columns and plot_df["structural_fair_value"].notna().any():
        value_cols.append("structural_fair_value")
    value_df = plot_df.melt(
        id_vars=["date"],
        value_vars=value_cols,
        var_name="series",
        value_name="price",
    )
    value_df["series"] = value_df["series"].map(
        {
            "actual": f"{symbol} actual",
            "fair_value": f"{symbol} fair value",
            "structural_fair_value": f"{symbol} structural fair value",
        }
    )
    return (
        alt.Chart(value_df)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("date:T", title="Date", axis=alt.Axis(**_axis_cfg())),
            y=alt.Y("price:Q", title="Price", scale=alt.Scale(zero=False), axis=alt.Axis(**_axis_cfg())),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(
                    domain=[f"{symbol} actual", f"{symbol} fair value", f"{symbol} structural fair value"],
                    range=[_theme_chart_color("primary"), _theme_chart_color("secondary"), _ORANGE],
                ),
                legend=alt.Legend(title=None, labelColor=_theme_chart_color("axis_label")),
            ),
            tooltip=["date:T", "series:N", alt.Tooltip("price:Q", format=".2f")],
        )
        .properties(title=alt.TitleParams(f"{symbol} – Actual vs Fair Value (optimized slow macro fit)"  
                                          f" - fair value = structural fair value + market adjustment", **_title_cfg()), height=320)
        .configure(background="#0c0d14").configure_view(fill="#181a25", strokeOpacity=0)
        .configure_axis(**_axis_cfg())
        .configure_title(**_title_cfg())
    )


def _fair_gap_chart(frame: pd.DataFrame, symbol: str):
    plot_df = frame.copy()
    plot_df["gap_sign"] = plot_df["fair_gap_pct"].apply(lambda x: "Undervalued" if x > 0 else "Overvalued")
    bars = (
        alt.Chart(plot_df)
        .mark_bar(opacity=0.75)
        .encode(
            x=alt.X("date:T", title="Date", axis=alt.Axis(**_axis_cfg())),
            y=alt.Y("fair_gap_pct:Q", title="Fair Gap %", axis=alt.Axis(**_axis_cfg())),
            color=alt.Color(
                "gap_sign:N",
                scale=alt.Scale(domain=["Undervalued", "Overvalued"], range=[_GREEN, _RED]),
                legend=alt.Legend(title=None, labelColor=_theme_chart_color("axis_label")),
            ),
            tooltip=[
                "date:T",
                alt.Tooltip("fair_gap_pct:Q", format=".2f", title="Gap %"),
                alt.Tooltip("actual:Q", format=".2f", title=f"{symbol} actual"),
                alt.Tooltip("fair_value:Q", format=".2f", title="Fair value"),
            ],
        )
    )
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#cfd8dc", opacity=0.5).encode(y="y:Q")
    return (
        alt.layer(bars, zero)
        .properties(title=alt.TitleParams(f"{symbol} – Fair Value Gap", **_title_cfg()), height=180)
        .configure(background="#0c0d14").configure_view(fill="#181a25", strokeOpacity=0)
        .configure_axis(**_axis_cfg())
        .configure_title(**_title_cfg())
    )


def render() -> None:
    render_started_at = time.perf_counter()
    render_mode_banner()
    st.title("Backtester")
    st.session_state.setdefault("bt_strategy", "Bollinger + RSI (Spike-Aware)")
    previous_mode = st.session_state.get(_BT_LAST_RENDERED_MODE_KEY)
    restored_snapshot = False
    restored_managed_snapshot = False
    pending_mode = st.session_state.pop(_BT_PENDING_MODE_KEY, None)
    if pending_mode in (_BT_MODE_SINGLE, _BT_MODE_PORTFOLIO):
        st.session_state[_BT_MODE_KEY] = pending_mode
    if _BT_MODE_KEY not in st.session_state:
        st.session_state[_BT_MODE_KEY] = (
            _BT_MODE_SINGLE
            if st.session_state.get(_BT_PENDING_TARGET_LOAD_KEY)
            else (
                _BT_MODE_PORTFOLIO
                if st.session_state.get(_BT_PORT_PENDING_RUN_KEY) or st.session_state.get(_BT_PORTFOLIO_RESULT_KEY)
                else _BT_MODE_SINGLE
            )
        )

    pending_target = st.session_state.get(_BT_PENDING_TARGET_LOAD_KEY)
    if pending_target:
        pending_strategy_name = pending_target.get("strategy_name")
        if pending_strategy_name:
            st.session_state["bt_strategy"] = pending_strategy_name
        should_execute_pending = bool(st.session_state.pop(_BT_PENDING_TARGET_EXECUTE_KEY, False))
        _render_backtest_target_loading_shell(pending_target)
        if not should_execute_pending:
            with st.status("Stage 1 of 2 — loading target dataset", expanded=True):
                progress_bar = st.progress(0.0, text="0% · Starting target dataset load")
                _render_stage_progress(progress_bar, 0.01, "Preparing target dataset loader")
            st.session_state[_BT_PENDING_TARGET_EXECUTE_KEY] = True
            st.rerun()

        pending_target = st.session_state.pop(_BT_PENDING_TARGET_LOAD_KEY, None)
        with st.status("Stage 1 of 2 — loading target dataset", expanded=True) as status:
            message_slot = st.empty()
            progress_bar = st.progress(0.0, text="0% · Starting target dataset load")
            try:
                def _target_stage(message: str) -> None:
                    message_slot.write(message)

                def _target_progress(progress: float, label: str) -> None:
                    _render_stage_progress(progress_bar, progress, label)

                _target_progress(0.04, "Starting target dataset load")

                _, message = _apply_backtest_target(
                    symbol=pending_target["symbol"],
                    interval=pending_target["interval"],
                    start_ts=pd.Timestamp(pending_target["start_ts"]),
                    end_ts=pd.Timestamp(pending_target["end_ts"]),
                    strategy_id=pending_target.get("strategy_id"),
                    status_cb=_target_stage,
                    progress_cb=_target_progress,
                )
                _target_progress(0.97, "Target dataset ready — handing off to stage 2")
                st.session_state[_BT_TARGET_STATUS_KEY] = message
                st.session_state[_BT_RUN_AFTER_LOAD_KEY] = True
                st.session_state[_BT_STAGE2_PENDING_KEY] = {
                    "symbol": pending_target["symbol"],
                    "interval": pending_target["interval"],
                }
                status.update(
                    label="Stage 1 of 2 — target dataset ready, starting stage 2",
                    state="complete",
                )
            except Exception as exc:
                st.session_state.pop(_BT_RUN_AFTER_LOAD_KEY, None)
                st.session_state[_BT_TARGET_STATUS_KEY] = f"error::{exc}"
                _target_progress(1.0, "Target dataset load failed")
                status.update(
                    label="Stage 1 of 2 — target dataset failed",
                    state="error",
                )
            finally:
                st.session_state.pop(_BT_PENDING_TARGET_EXECUTE_KEY, None)
        st.rerun()

    mode = st.radio(
        "Backtest Mode",
        [_BT_MODE_SINGLE, _BT_MODE_PORTFOLIO],
        horizontal=True,
        key=_BT_MODE_KEY,
    )
    first_render = previous_mode is None
    mode_changed = previous_mode is not None and previous_mode != mode
    clear_started_at = time.perf_counter()
    cleared_single_stats: dict[str, int] | None = None
    cleared_portfolio_stats: dict[str, int] | None = None
    if mode_changed:
        if previous_mode == _BT_MODE_PORTFOLIO:
            cleared_portfolio_stats = _managed_result_stats()
            _clear_managed_portfolio_result_state()
        elif previous_mode == _BT_MODE_SINGLE:
            cleared_single_stats = _backtest_result_stats()
            _clear_backtest_result_state()
    clear_elapsed = time.perf_counter() - clear_started_at
    st.session_state[_BT_LAST_RENDERED_MODE_KEY] = mode
    branch_started_at = time.perf_counter()
    if mode == _BT_MODE_PORTFOLIO:
        if first_render and _BT_PORTFOLIO_RESULT_KEY not in st.session_state:
            restored_managed_snapshot = _restore_managed_portfolio_snapshot()
        st.caption(
            "Run a shared-balance portfolio backtest across multiple tickers with automatic strategy assignment, "
            "ranking, dynamic sizing, and optional replacement."
        )
        if _BT_PORTFOLIO_RESULT_KEY not in st.session_state and not st.session_state.get(_BT_PORT_PENDING_RUN_KEY):
            if st.button("Restore Last Saved Portfolio Result", key="bt_restore_portfolio_result"):
                if _restore_managed_portfolio_snapshot():
                    st.rerun()
                else:
                    st.info("No saved portfolio result is available to restore right now.")
        managed_restored_msg = st.session_state.pop("bt_port_restored_msg", None)
        if managed_restored_msg:
            st.caption(managed_restored_msg)
        _render_managed_portfolio_backtest()
        branch_elapsed = time.perf_counter() - branch_started_at
        total_elapsed = time.perf_counter() - render_started_at
        _log_backtester_mode_timing(
            previous_mode=previous_mode,
            mode=mode,
            mode_changed=mode_changed,
            total_elapsed=total_elapsed,
            clear_elapsed=clear_elapsed,
            branch_elapsed=branch_elapsed,
            restored_snapshot=restored_snapshot,
            restored_managed_snapshot=restored_managed_snapshot,
            cleared_single_stats=cleared_single_stats,
            cleared_portfolio_stats=cleared_portfolio_stats,
        )
        return

    st.caption(
        "Run a detailed single-ticker backtest with full strategy diagnostics, trade review, and chart controls."
    )

    sidebar_prices = render_data_source_selector(restore_cached_frame=False)
    if _get_backtest_dataset() is None and sidebar_prices is not None:
        _set_backtest_dataset(
            sidebar_prices,
            symbol=st.session_state.get("loaded_symbol", "DATA"),
            source=st.session_state.get("loaded_source"),
            interval=st.session_state.get("loaded_interval"),
            start=st.session_state.get("loaded_start"),
            end=st.session_state.get("loaded_end"),
        )

    target_action = _render_backtest_target_selector()
    if target_action.get("submitted") and target_action.get("payload"):
        _clear_backtest_result_state()
        if target_action["payload"].get("strategy_name"):
            st.session_state["bt_strategy"] = target_action["payload"]["strategy_name"]
        st.session_state[_BT_PENDING_MODE_KEY] = _BT_MODE_SINGLE
        st.session_state[_BT_PENDING_TARGET_LOAD_KEY] = target_action["payload"]
        st.rerun()

    target_status = st.session_state.pop(_BT_TARGET_STATUS_KEY, None)
    if target_status:
        if str(target_status).startswith("error::"):
            st.error(str(target_status).split("error::", 1)[1])
        else:
            st.success(str(target_status))

    run_status_slot = st.empty()
    stage2_pending = st.session_state.get(_BT_STAGE2_PENDING_KEY)

    prices = _get_backtest_dataset()
    if prices is None:
        st.info("Load a backtest target above, or use the sidebar to fetch/cache a dataset first.")
        if "bt_result" in st.session_state:
            st.divider()
            _show_results()
        branch_elapsed = time.perf_counter() - branch_started_at
        total_elapsed = time.perf_counter() - render_started_at
        _log_backtester_mode_timing(
            previous_mode=previous_mode,
            mode=mode,
            mode_changed=mode_changed,
            total_elapsed=total_elapsed,
            clear_elapsed=clear_elapsed,
            branch_elapsed=branch_elapsed,
            restored_snapshot=restored_snapshot,
            restored_managed_snapshot=restored_managed_snapshot,
            cleared_single_stats=cleared_single_stats,
            cleared_portfolio_stats=cleared_portfolio_stats,
        )
        return

    if first_render and "bt_result" not in st.session_state:
        restored_snapshot = _restore_backtest_snapshot()
    restored_msg = st.session_state.pop("bt_restored_msg", None)
    if restored_msg:
        st.caption(restored_msg)

    symbol = st.session_state.get("bt_symbol_live", "DATA")
    bar_label = _bar_label(prices)
    st.success(f"**{symbol}** — {len(prices):,} bars · bar size: **{bar_label}**")
    current_result = st.session_state.get("bt_result")
    if current_result is None:
        if st.button("Restore Last Saved Single-Ticker Result", key="bt_restore_single_result"):
            if _restore_backtest_snapshot():
                st.rerun()
            else:
                st.info("No saved single-ticker result matches the currently loaded dataset.")
        current_result = st.session_state.get("bt_result")
    if current_result is not None:
        _show_results_preview(current_result, symbol)
    st.divider()

    strategies = ordered_strategy_items(list_strategies())
    strat_names = {s["name"]: s["id"] for s in strategies}
    strategy_names = list(strat_names.keys())
    default_strategy_name = "Bollinger + RSI (Spike-Aware)"
    if st.session_state.get("bt_strategy") in _LEGACY_EARNINGS_STRATEGY_NAMES and _EARNINGS_UNIFIED_STRATEGY_NAME in strategy_names:
        st.session_state["bt_strategy"] = _EARNINGS_UNIFIED_STRATEGY_NAME
    if st.session_state.get("bt_selected_id") in _LEGACY_EARNINGS_STRATEGY_IDS and _EARNINGS_UNIFIED_STRATEGY_NAME in strategy_names:
        st.session_state["bt_selected_id"] = strat_names[_EARNINGS_UNIFIED_STRATEGY_NAME]
    if st.session_state.get("bt_strategy") not in strategy_names:
        restored_strategy_name = _strategy_name_from_id(st.session_state.get("bt_selected_id"))
        restored_strategy_name = _map_legacy_earnings_strategy_name(restored_strategy_name, strategy_names)
        st.session_state["bt_strategy"] = (
            restored_strategy_name
            if restored_strategy_name in strategy_names
            else (default_strategy_name if default_strategy_name in strategy_names else strategy_names[0])
        )
    symbol_u = str(symbol).upper()
    strategy_window_start = st.session_state.get("bt_start_live") or st.session_state.get("loaded_start")
    strategy_window_end = st.session_state.get("bt_end_live") or st.session_state.get("loaded_end")
    selected_name = str(st.session_state.get("bt_strategy") or default_strategy_name)
    if selected_name not in strat_names:
        selected_name = default_strategy_name if default_strategy_name in strat_names else strategy_names[0]
        st.session_state["bt_strategy"] = selected_name
    selected_id = strat_names[selected_name]
    desired_monday_default = bool(resolve_execution_default(symbol_u, "monday_open_delay", True))
    if st.session_state.get("bt_monday_open_delay_symbol_sig") != symbol_u:
        previous_auto = st.session_state.get("bt_monday_open_delay_last_auto")
        current_value = st.session_state.get("bt_monday_open_delay")
        if "bt_monday_open_delay" not in st.session_state or current_value == previous_auto:
            st.session_state["bt_monday_open_delay"] = desired_monday_default
        st.session_state["bt_monday_open_delay_last_auto"] = desired_monday_default
        st.session_state["bt_monday_open_delay_symbol_sig"] = symbol_u

    auto_run_requested = bool(st.session_state.pop(_BT_RUN_AFTER_LOAD_KEY, False)) or bool(target_action.get("run_requested"))
    if auto_run_requested and stage2_pending:
        with run_status_slot.container():
            with st.status("Stage 2 of 2 — preparing simulation handoff", expanded=True):
                pre_bar = st.progress(0.0, text="0% · Starting stage 2")
                _render_stage_progress(
                    pre_bar,
                    0.03,
                    f"Preparing stage 2 for {str(stage2_pending.get('symbol') or symbol).upper()}",
                )

    with st.form("bt_controls_form", clear_on_submit=False):
        # ── Configuration tabs ──────────────────────────────────────────
        tab_strategy, tab_risk, tab_exec = st.tabs([
            "Strategy & Sizing",
            "Risk & Costs",
            "Execution Rules",
        ])

        with tab_strategy:
            s_col1, s_col2 = st.columns(2)
            with s_col1:
                leverage = st.number_input("Leverage", 1.0, 100.0, 1.0, 0.5, key="bt_lev")
                capital_per_trade = st.number_input("Capital per trade ($)", 100.0, value=1000.0, key="bt_cap")
            with s_col2:
                starting_equity = st.number_input("Starting equity ($)", 1000.0, value=1000.0, key="bt_equity")
                direction_filter = st.selectbox("Direction filter", ["Both", "Long only", "Short only"], key="bt_dir")

            show_gld_candidates = selected_id == "bollinger_rsi" and symbol_u == "GLD"
            candidate_name, candidate_payload = _resolve_gld_candidate_for_leverage(leverage) if show_gld_candidates else (None, {})
            optional_preset_payload: dict[str, object] = {}
            if show_gld_candidates:
                with st.expander("GLD research overlays (optional)", expanded=False):
                    use_rsi_spike_fade_short = st.checkbox(
                        "Use RSI spike-fade short (research)",
                        value=False,
                        key="bt_gld_rsi_spike_fade_short",
                        help="Optional GLD research overlay. This layers the researched RSI spike-fade short rule on top of the current default or leverage-specific GLD preset.",
                    )
                    use_fair_gap_fade_short = st.checkbox(
                        "Use fair-gap fade short (research)",
                        value=False,
                        key="bt_gld_fair_gap_fade_short",
                        help="Optional GLD research overlay. This shorts GLD only when price is materially above fair value, daily RSI is overbought, and the minute chart starts rolling over.",
                    )
                    use_weak_0800_filter = st.checkbox(
                        "Use weak 08:00 ET filter (research)",
                        value=False,
                        key="bt_gld_weak_0800_filter",
                        help="Optional GLD research overlay. This blocks the weak 08:00-08:29 ET shock-reversal short entries identified in the narrower time-of-day study.",
                    )
                    if use_rsi_spike_fade_short:
                        optional_preset_payload.update(get_candidate("gld_rsi_spike_fade_short_20260426"))
                    if use_fair_gap_fade_short:
                        optional_preset_payload.update(get_candidate("gld_fair_gap_fade_short_20260426"))
                    if use_weak_0800_filter:
                        optional_preset_payload.update(get_candidate("gld_weak_0800_shock_reversal_filter_20260426"))
            candidate_param_overrides = {
                k: v for k, v in candidate_payload.items() if k not in {"leverage", "risk_max_loss_pct_of_capital"}
            }
            optional_param_overrides = {
                k: v for k, v in optional_preset_payload.items() if k not in {"leverage", "risk_max_loss_pct_of_capital"}
            }
            combined_param_overrides = {**candidate_param_overrides, **optional_param_overrides}
            if show_gld_candidates and candidate_payload:
                desired_risk_cap = int(round(float(candidate_payload.get("risk_max_loss_pct_of_capital", 50.0))))
                desired_sig = (candidate_name, round(float(leverage), 4), desired_risk_cap)
                if st.session_state.get("bt_gld_auto_risk_sig") != desired_sig:
                    st.session_state["bt_maxloss"] = desired_risk_cap
                    st.session_state["bt_gld_auto_risk_sig"] = desired_sig
            else:
                st.session_state.pop("bt_gld_auto_risk_sig", None)
            if show_gld_candidates:
                if candidate_name == "gld_best_1x_sweep_20260419":
                    st.caption("Auto-applied GLD preset: optimized `1x` research candidate.")
                elif candidate_name == "gld_best_leverage_5x_tuned_20260419":
                    st.caption("Auto-applied GLD preset: optimized `5x` tuned research candidate.")
                else:
                    st.caption("No stored GLD leverage-specific preset for this leverage yet. Current GLD default stays active.")
                if optional_preset_payload:
                    captions: list[str] = []
                    if use_rsi_spike_fade_short:
                        captions.append("`RSI spike-fade short` (`rise >= 0.8%`, `RSI >= 88`)")
                    if use_fair_gap_fade_short:
                        captions.append("`fair-gap fade short` (`gap >= 3%`, daily `RSI >= 80`)")
                    if use_weak_0800_filter:
                        captions.append("`weak 08:00 ET filter` (blocks `shock_reversal_short` only)")
                    st.caption("Optional GLD preset active: adds " + " + ".join(captions) + ".")
            if selected_id == "bollinger_rsi" and symbol_u == "SPY":
                st.caption(
                    "Auto-applied SPY preset: a sparse long trend-bias profile. "
                    "It disables the generic mean-reversion and short stacks, and mainly trades the dedicated trend-bias long leg."
                )

        with tab_risk:
            r_col1, r_col2 = st.columns(2)
            with r_col1:
                st.markdown("**Risk Manager**")
                use_risk = st.checkbox("Apply risk manager", value=True, key="bt_risk")
                max_loss = st.slider("Max loss per trade (% of capital)", 5, 100, 50, key="bt_maxloss")
                counter_signal_exit = st.checkbox(
                    "Counter-signal exit",
                    value=True,
                    key="bt_counter",
                    help="When ON: opposing RSI signal closes the current trade and opens reverse.",
                )
                if show_gld_candidates and candidate_payload:
                    st.caption(
                        f"Preset risk note: leverage `{float(candidate_payload.get('leverage', leverage)):.1f}x`, "
                        f"max capital loss cap `{float(candidate_payload.get('risk_max_loss_pct_of_capital', max_loss)):.0f}%`."
                    )
            with r_col2:
                st.markdown("**Transaction Costs**")
                st.caption("Typical retail-broker round-trip: spread 0.02-0.10%, slippage 0.01-0.05%. UVXY-style ETFs sit near the higher end; large ETFs like GLD or SPY closer to the lower end. Leave at 0 for gross return.")
                spread_pct = st.number_input("Spread % (round-trip)", 0.0, 2.0, 0.06, step=0.01, format="%.2f", key="bt_spread")
                slippage_pct = st.number_input("Slippage % (round-trip)", 0.0, 2.0, 0.02, step=0.01, format="%.2f", key="bt_slip")
                commission = st.number_input("Commission per trade ($)", 0.0, 10.0, 0.0, step=0.10, format="%.2f", key="bt_comm")

        with tab_exec:
            _bt_policy_opts = available_policies()
            _bt_policy_labels = [lbl for _, lbl in _bt_policy_opts]
            _bt_policy_names = [nm for nm, _ in _bt_policy_opts]
            _bt_default_idx = _bt_policy_names.index("alpaca") if "alpaca" in _bt_policy_names else 0
            _bt_chosen_label = st.selectbox(
                "Entry-gate policy",
                _bt_policy_labels,
                index=_bt_default_idx,
                key="bt_exec_logic",
                help=(
                    "Classic = the unconstrained logic we had before Alpaca gates were "
                    "added. Alpaca-realistic = RTH / PDT / SSR / fractional / fill-"
                    "diagnostic applied at entry so backtest numbers track what Alpaca "
                    "would actually fill."
                ),
            )
            execution_logic = _bt_policy_names[_bt_policy_labels.index(_bt_chosen_label)]
            _is_alpaca_bt = execution_logic == "alpaca"
            if _is_alpaca_bt:
                st.markdown("**Alpaca-realistic rules**")
                bt_ac1, bt_ac2, bt_ac3 = st.columns(3)
                with bt_ac1:
                    bt_enforce_rth = st.checkbox(
                        "RTH only", value=True, key="bt_rth",
                        help="Skip entries outside NYSE RTH (09:30-16:00 ET) and on holidays.",
                    )
                    bt_extended_hours = st.checkbox(
                        "Extended hours", value=False, key="bt_ext_hrs",
                        help="Also accept 04:00-20:00 ET on trading days.",
                    )
                with bt_ac2:
                    bt_enforce_pdt = st.checkbox(
                        "PDT (<$25k)", value=True, key="bt_pdt",
                        help="Block 4th day-trade in 5 days when equity < $25k.",
                    )
                    bt_enforce_ssr = st.checkbox(
                        "SSR (shorts)", value=True, key="bt_ssr",
                        help="Skip short entries on ≥10% gap-down vs prior close.",
                    )
                with bt_ac3:
                    bt_enforce_frac = st.checkbox(
                        "Fractional rule", value=True, key="bt_frac",
                        help="Shorts need integer qty ≥ 1 (Alpaca rule).",
                    )
                    bt_fill_diag = st.checkbox(
                        "Fill-timing diag", value=True, key="bt_fill_diag",
                        help="Attach bar H/L/range to each entry's notes.",
                    )
            else:
                st.caption(
                    "Classic mode — no Alpaca gates applied. Useful as an "
                    "unconstrained baseline for comparing vs Alpaca-realistic runs."
                )
                bt_enforce_rth = False
                bt_extended_hours = False
                bt_enforce_pdt = False
                bt_enforce_ssr = False
                bt_enforce_frac = False
                bt_fill_diag = False

            bt_enforce_monday_open_delay = st.checkbox(
                "No Monday trades first 30m",
                key="bt_monday_open_delay",
                help="Block all entries from the NYSE open until 10:00 ET on Mondays, regardless of strategy.",
            )

        st.divider()
        params = render_strategy_params(
            selected_id,
            leverage=leverage,
            max_capital_loss_pct=float(max_loss),
            symbol=symbol,
            source=st.session_state.get("bt_source_live") or st.session_state.get("loaded_source"),
            interval=st.session_state.get("bt_interval_live") or st.session_state.get("loaded_interval"),
            base_overrides=combined_param_overrides if show_gld_candidates and combined_param_overrides else None,
            in_form=True,
        )
        st.caption("Use this button after changing strategy, risk, execution, or parameter controls for the already loaded target dataset.")
        manual_run_clicked = st.form_submit_button("Re-run with Current Controls", type="primary")

    run_clicked = bool(manual_run_clicked) or auto_run_requested

    if "bt_result" in st.session_state and not run_clicked:
        st.divider()
        show_details = st.checkbox(
            "Show detailed charts and trade log",
            value=False,
            key="bt_show_details",
        )
        if show_details:
            _show_results()
        else:
            st.caption("Detailed backtest charts and trade log are hidden to keep the initial page load light.")

    if run_clicked:
        st.session_state.pop(_BT_STAGE2_PENDING_KEY, None)
        cls = get_strategy(selected_id)
        strategy = cls(params=params)
        errors = strategy.validate_params()
        if errors:
            for e in errors:
                st.error(e)
            return
        from config.settings import RiskConfig
        from core.models import Direction as Dir

        risk_cfg = RiskConfig(
            max_capital_per_trade_pct=100.0,
            max_daily_loss_pct=100.0,
            max_open_positions=999,
            default_max_loss_pct_of_capital=float(max_loss),
        )
        rm = RiskManager(risk_cfg) if use_risk else None
        dir_filter = None
        if direction_filter == "Long only":
            dir_filter = Dir.LONG
        if direction_filter == "Short only":
            dir_filter = Dir.SHORT
        engine = BacktestEngine(
            strategy,
            risk_manager=rm,
            direction_filter=dir_filter,
            counter_signal_exit=counter_signal_exit,
            spread_pct=float(spread_pct),
            slippage_pct=float(slippage_pct),
            commission_per_trade=float(commission),
            enforce_rth=bool(bt_enforce_rth),
            extended_hours=bool(bt_extended_hours),
            enforce_pdt=bool(bt_enforce_pdt),
            enforce_ssr=bool(bt_enforce_ssr),
            enforce_fractional=bool(bt_enforce_frac),
            fill_diagnostic=bool(bt_fill_diag),
            enforce_monday_open_delay=bool(bt_enforce_monday_open_delay),
        )
        prepared_bars = 0
        with run_status_slot.container():
            with st.status("Stage 2 of 2 — preparing context and running backtest", expanded=True) as status:
                message_slot = st.empty()
                progress_bar = st.progress(0.0, text=f"0% · Starting stage 2 for {symbol}")
                message_slot.write(f"Preparing strategy context for {symbol} on {len(prices):,} raw bars…")
                _render_stage_progress(progress_bar, 0.06, f"Preparing strategy context for {symbol}")
                prepared_prices = prepare_strategy_data(
                    prices,
                    strategy,
                    primary_symbol=symbol,
                    source=st.session_state.get("bt_source_live") or st.session_state.get("loaded_source"),
                    interval=st.session_state.get("bt_interval_live") or st.session_state.get("loaded_interval"),
                    start=st.session_state.get("bt_start_live") or st.session_state.get("loaded_start"),
                    end=st.session_state.get("bt_end_live") or st.session_state.get("loaded_end"),
                )
                prepared_bars = int(len(prepared_prices)) if prepared_prices is not None else 0
                message_slot.write(f"Prepared {prepared_bars:,} bars. Running strategy simulation…")
                _render_stage_progress(progress_bar, 0.30, f"Prepared {prepared_bars:,} bars")
                _render_stage_progress(progress_bar, 0.52, f"Running simulation for {symbol}")
                last_sim_progress = 0.0

                def _simulation_progress(progress: float, current_ts, closed_trades: int) -> None:
                    nonlocal last_sim_progress
                    clamped = max(0.0, min(1.0, float(progress)))
                    last_sim_progress = max(last_sim_progress, clamped)
                    overall = 0.52 + (0.38 * last_sim_progress)
                    if isinstance(current_ts, str) and current_ts.startswith("phase::"):
                        phase_label = current_ts.split("phase::", 1)[1]
                        label = f"{phase_label} · closed trades {int(closed_trades)}"
                        message = f"{phase_label} · closed trades so far: {int(closed_trades)}"
                    else:
                        label = (
                            f"Simulation at {_progress_time_label(current_ts)}"
                            f" · closed trades {int(closed_trades)}"
                        )
                        message = (
                            f"Simulation progress: {_progress_time_label(current_ts)}"
                            f" · closed trades so far: {int(closed_trades)}"
                        )
                    _render_stage_progress(progress_bar, overall, label)
                    message_slot.write(message)

                result = engine.run(
                    data=prepared_prices,
                    symbol=symbol,
                    leverage=leverage,
                    capital_per_trade=capital_per_trade,
                    starting_equity=starting_equity,
                    progress_cb=_simulation_progress,
                )
                message_slot.write(
                    f"Simulation finished: {result.total_trades} trade(s), "
                    f"{result.total_return_pct:.2f}% total return. Saving results…"
                )
                _render_stage_progress(
                    progress_bar,
                    0.93,
                    f"Saving {result.total_trades} trade(s) and final metrics",
                )
                _render_stage_progress(progress_bar, 1.0, f"Backtest finished for {symbol}")
                status.update(
                    label=(
                        f"Backtest finished — {symbol} · {prepared_bars:,} prepared bars"
                    ),
                    state="complete",
                )
        st.session_state["bt_result"] = result
        st.session_state["bt_symbol"] = symbol
        st.session_state["bt_bar_label"] = bar_label
        st.session_state["bt_selected_id"] = selected_id
        st.session_state["bt_params"] = dict(params)
        st.session_state["bt_starting_equity"] = float(starting_equity)
        st.session_state["bt_cost_settings"] = {
            "spread_pct": float(spread_pct),
            "slippage_pct": float(slippage_pct),
            "commission": float(commission),
        }
        st.session_state["bt_execution_logic"] = execution_logic
        st.session_state["bt_execution_label"] = _bt_chosen_label
        _persist_backtest_snapshot(
            _serialize_backtest_result(
                result,
                symbol=symbol,
                bar_label=bar_label,
                selected_id=selected_id,
                params=dict(params),
                starting_equity=float(starting_equity),
                costs={
                    "spread_pct": float(spread_pct),
                    "slippage_pct": float(slippage_pct),
                    "commission": float(commission),
                },
                execution_logic=execution_logic,
                execution_label=_bt_chosen_label,
            )
        )
        try:
            from db.database import Database

            db = Database(settings.db_path)
            for t in result.trades:
                db.save_trade(t)
            st.session_state["bt_db_msg"] = f"✓ {len(result.trades)} trades saved."
        except Exception as e:
            st.session_state["bt_db_msg"] = f"DB save skipped: {e}"
        _show_results_preview(result, symbol)
        st.divider()
        _show_results()
        branch_elapsed = time.perf_counter() - branch_started_at
        total_elapsed = time.perf_counter() - render_started_at
        _log_backtester_mode_timing(
            previous_mode=previous_mode,
            mode=mode,
            mode_changed=mode_changed,
            total_elapsed=total_elapsed,
            clear_elapsed=clear_elapsed,
            branch_elapsed=branch_elapsed,
            restored_snapshot=restored_snapshot,
            restored_managed_snapshot=restored_managed_snapshot,
            cleared_single_stats=cleared_single_stats,
            cleared_portfolio_stats=cleared_portfolio_stats,
        )
        return

    branch_elapsed = time.perf_counter() - branch_started_at
    total_elapsed = time.perf_counter() - render_started_at
    _log_backtester_mode_timing(
        previous_mode=previous_mode,
        mode=mode,
        mode_changed=mode_changed,
        total_elapsed=total_elapsed,
        clear_elapsed=clear_elapsed,
        branch_elapsed=branch_elapsed,
        restored_snapshot=restored_snapshot,
        restored_managed_snapshot=restored_managed_snapshot,
        cleared_single_stats=cleared_single_stats,
        cleared_portfolio_stats=cleared_portfolio_stats,
    )


def _show_results() -> None:
    if "bt_result" not in st.session_state:
        return
    result = st.session_state["bt_result"]
    symbol_r = st.session_state.get("bt_symbol", "DATA")
    selected_id_r = st.session_state.get("bt_selected_id", "")
    params_r = st.session_state.get("bt_params", {})
    costs_r = st.session_state.get("bt_cost_settings", {})
    prices_r = _get_backtest_dataset()
    closed = [t for t in result.trades if t.leveraged_return_pct is not None]

    st.subheader("Results")
    s = result.summary()
    render_metrics_row(
        {
            "Total Trades": s["Total Trades"],
            "Win Rate": s["Win Rate"],
            "Total Return": s["Total Return"],
            "Max Drawdown": s["Max Drawdown"],
            "Sharpe Ratio": s["Sharpe Ratio"],
            "Avg Win": s["Avg Win"],
            "Avg Loss": s["Avg Loss"],
        }
    )
    if closed:
        from collections import Counter

        outcome_counts = Counter(t.outcome.value if hasattr(t.outcome, "value") else str(t.outcome) for t in closed)
        st.markdown("**Exit breakdown:**")
        cols = st.columns(min(len(outcome_counts), 5))
        for col, (label, cnt) in zip(cols, sorted(outcome_counts.items())):
            col.metric(label, cnt)
        gross_pnl = sum((t.capital_allocated or 0) * ((t.leveraged_return_pct or 0) / 100.0) for t in closed)
        net_pnl = sum((t.pnl or 0) for t in closed)
        deducted_cost = gross_pnl - net_pnl
        spread_used = float(costs_r.get("spread_pct", 0.0) or 0.0)
        slippage_used = float(costs_r.get("slippage_pct", 0.0) or 0.0)
        commission_used = float(costs_r.get("commission", 0.0) or 0.0)
        st.caption(
            f"Applied costs: spread {spread_used:.2f}% + slippage {slippage_used:.2f}% "
            f"+ commission {commission_used:.2f} dollars/trade = approx. "
            f"{deducted_cost:,.2f} dollars deducted from gross closed-trade PnL."
        )
        _exec_label = st.session_state.get(
            "bt_execution_label",
            "Alpaca-realistic (RTH / PDT / SSR / fractional)"
            if st.session_state.get("bt_execution_logic", "alpaca") == "alpaca"
            else "Classic (pre-Alpaca gates)",
        )
        st.caption(f"Execution logic: **{_exec_label}**")
    st.caption("📖 **PnL ($)** = dollar profit/loss · **Return %** = leveraged return on capital · **TP hit** = price target reached · **SL hit** = stop hit · **Trail stop** = trailing stop exit · **RSI exits** = RSI threshold crossed")
    st.divider()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    show_long = c1.checkbox("▲ Long entries", value=True, key="show_long")
    show_short = c2.checkbox("▼ Short entries", value=True, key="show_short")
    show_tp = c3.checkbox("✕ TP exits", value=True, key="show_tp_x")
    show_sl = c4.checkbox("✕ SL exits", value=True, key="show_sl_x")
    show_trail = c5.checkbox("◆ Trail exits", value=True, key="show_trail_x")
    show_sig = c6.checkbox("✕ Signal exits", value=True, key="show_sig_x")
    chart_window = st.radio(
        "Chart focus",
        options=CHART_WINDOW_OPTIONS,
        index=0,
        key=f"bt_chart_window_{symbol_r}",
        horizontal=True,
        format_func=chart_window_label,
    )

    if prices_r is not None:
        prices_window = filter_chart_window(prices_r, chart_window)
        prices_plot = _downsample(prices_window)
        n_bars = len(prices_window)
        total_bars = len(prices_r)
        label_parts = []
        if chart_window != "All":
            label_parts.append(f"window {chart_window_label(chart_window)}")
        if len(prices_plot) < n_bars:
            label_parts.append(f"{len(prices_plot):,} of {n_bars:,} bars shown")
        elif n_bars < total_bars:
            label_parts.append(f"{n_bars:,} of {total_bars:,} bars in view")
        label_extra = f"  ·  *{' · '.join(label_parts)}*" if label_parts else ""
        st.markdown(f"#### Price{label_extra}")
        st.altair_chart(_price_chart(prices_plot, result.trades, symbol_r, show_long, show_short, show_tp, show_sl, show_trail, show_sig), width='stretch')
        if selected_id_r in ("rsi_threshold", "atr_rsi", "vwap_rsi", "bollinger_rsi", "ema_trend_rsi"):
            period = int(params_r.get("rsi_period", 9))
            buy_levels = _parse_levels(params_r.get("buy_levels", "30"))
            sell_levels = _parse_levels(params_r.get("sell_levels", "70"))
            st.markdown(f"#### RSI ({period})")
            st.altair_chart(_rsi_chart(prices_plot, result.trades, period, buy_levels, sell_levels, symbol_r, show_long, show_short, show_tp, show_sl, show_trail, show_sig), width='stretch')
    else:
        st.info("ℹ️ Price chart not available — reload data to see charts.")

    if closed:
        st.markdown("#### Equity Curve")
        st.altair_chart(_equity_chart(result.equity_curve, symbol_r, chart_window), width='stretch')

    if closed:
        trades_df = pd.DataFrame(
            [
                {
                    "symbol": t.symbol,
                    "regime": _trade_regime(t),
                    "direction": t.direction.value,
                    "capital_allocated": t.capital_allocated,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "outcome": t.outcome.value if t.outcome else None,
                    "leveraged_return_pct": t.leveraged_return_pct,
                    "pnl": t.pnl,
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                    "notes": t.notes,
                }
                for t in closed
            ]
        )
        st.markdown("#### Per-Trade Return")
        st.altair_chart(pnl_distribution(trades_df), width='stretch')
        with st.expander("Trade Log", expanded=False):
            st.dataframe(
                trades_df.rename(
                    columns={
                        "capital_allocated": "capital_allocated ($)",
                        "leveraged_return_pct": "return_pct (%)",
                        "pnl": "PnL ($)",
                    }
                ).sort_values("entry_time", ascending=False),
                width='stretch',
            )


def _show_results_preview(result: BacktestResult, symbol: str) -> None:
    st.markdown("**Latest Result Summary**")
    render_metrics_row(
        {
            "Ticker": symbol,
            "Trades": result.total_trades,
            "Return": f"{result.total_return_pct:.2f}%",
            "Max DD": f"{result.max_drawdown_pct:.2f}%",
            "Sharpe": f"{result.sharpe_ratio:.2f}",
        }
    )
    st.caption("Detailed charts, trade log, and diagnostics appear below the parameter panel.")

    if "bt_db_msg" in st.session_state:
        st.caption(st.session_state["bt_db_msg"])

    if symbol.upper() == "GLD":
        st.markdown("#### Macro Fair Value")
        st.caption(
            "This is a slow diagnostic model for GLD only. It fits an optimized fair-value proxy from cached macro and peer series, "
            "so we can judge the macro layer by fit quality before using it for trading bias."
        )
        trigger_key = "bt_gld_fair_value_visible"
        if st.button("Build / Refresh Fair Value Diagnostics", key="bt_gld_fair_value_btn"):
            st.session_state[trigger_key] = True
        if st.session_state.get(trigger_key):
            with st.spinner("Fitting GLD fair-value curve from cached slow macro data…"):
                fair_payload = _cached_gld_fair_value(
                    str(st.session_state.get("loaded_start")) if st.session_state.get("loaded_start") is not None else None,
                    str(st.session_state.get("loaded_end")) if st.session_state.get("loaded_end") is not None else None,
                    fair_value_cache_fingerprint(),
                )
            if fair_payload and fair_payload.get("error"):
                st.warning(f"Could not build GLD fair-value diagnostics: {fair_payload['error']}")
            elif fair_payload:
                fair_stats = fair_payload["stats"]
                model = fair_payload["model"]
                cache_mode = str(model.get("cache_mode", "live"))
                model_tf = str(model.get("data_timeframe", "1M")).upper()
                tf_label = "daily" if model_tf == "1D" else "monthly"
                tf_unit = "days" if model_tf == "1D" else "months"
                render_metrics_row(
                    {
                        "Correlation": f"{fair_stats['corr']:.3f}",
                        "R²": f"{fair_stats['r2']:.3f}",
                        "MAE Gap": f"{fair_stats['mae_pct']:.2f}%",
                        "RMSE Gap": f"{fair_stats['rmse_pct']:.2f}%",
                        "Direction Hit": f"{fair_stats['directional_hit'] * 100:.1f}%",
                    }
                )
                if model.get("model_type") == "two_layer":
                    st.caption(
                        f"{tf_label.capitalize()} slow fair-value proxy optimized as a structural layer plus a market-adjustment layer. "
                        f"Best fit: structural set `{model['structural_set']}`, market set `{model['market_set']}`, "
                        f"z-window `{model['z_window']}` {tf_unit}, structural fit window `{model['structural_fit_window']}` {tf_unit}, "
                        f"market fit window `{model['market_fit_window']}` {tf_unit}, ridge α `{model['ridge_alpha']:.2f}`, "
                        f"smoothing span `{model['smooth_span']}` {tf_unit}, target source `{model.get('target_source', 'unknown')}`."
                    )
                else:
                    st.caption(
                        f"{tf_label.capitalize()} slow fair-value proxy optimized as a blended macro-plus-market fit. "
                        f"Best fit: feature set `{model['feature_set']}`, z-window `{model['z_window']}` {tf_unit}, "
                        f"fit window `{model['fit_window']}` {tf_unit}, ridge α `{model['ridge_alpha']:.2f}`, "
                        f"smoothing span `{model['smooth_span']}` {tf_unit}, target source `{model.get('target_source', 'unknown')}`."
                    )
                if cache_mode == "stale_fallback":
                    st.info(
                        "Using the latest cached GLD fair-value diagnostics because one or more raw slow-data files "
                        "are unavailable in this environment."
                    )
                optional_sources = model.get("optional_sources") or {}
                if optional_sources:
                    pretty_sources = ", ".join(f"`{k}` from `{v}`" for k, v in sorted(optional_sources.items()))
                    st.caption(f"Optional proxy inputs currently available: {pretty_sources}.")
                else:
                    st.caption(
                        "Optional official ETF / central-bank proxy files are not loaded yet. "
                        "Current fit is using only the cached macro and market proxies."
                    )
                st.altair_chart(_fair_value_chart(fair_payload["frame"], symbol), width='stretch')
                st.altair_chart(_fair_gap_chart(fair_payload["frame"], symbol), width='stretch')
            else:
                st.warning("Could not build GLD fair-value diagnostics from the cached slow datasets.")


def _parse_symbol_list(raw: str) -> list[str]:
    tokens: list[str] = []
    for chunk in re.split(r"[\s,;]+", str(raw or "").strip()):
        cleaned = str(chunk).strip().upper()
        if cleaned:
            tokens.append(cleaned)
    seen: set[str] = set()
    ordered: list[str] = []
    for symbol in tokens:
        if symbol not in seen:
            ordered.append(symbol)
            seen.add(symbol)
    return ordered


def _show_managed_portfolio_results(payload: dict, *, show_details: bool = True) -> None:
    result = payload.get("result")
    if not isinstance(result, ManagedPortfolioBacktestResult):
        return

    symbols = payload.get("symbols") or []
    interval = str(payload.get("interval") or "n/a")
    execution_label = str(payload.get("execution_label") or "n/a")
    assignments = list(payload.get("assignments") or [])
    skipped_symbols = list(payload.get("skipped_symbols") or [])

    st.markdown("**Managed Portfolio Summary**")
    render_metrics_row(
        {
            "Tickers": len(symbols),
            "Trades": result.total_trades,
            "Return": f"{result.total_return_pct:.2f}%",
            "Max DD": f"{result.max_drawdown_pct:.2f}%",
            "Sharpe": f"{result.sharpe_ratio:.2f}",
        }
    )
    render_metrics_row(
        {
            "Win Rate": f"{result.win_rate_pct:.1f}%",
            "Candidates": result.candidate_entries,
            "Skipped": result.skipped_entries,
            "Replaced": result.replaced_positions,
            "Peak Open": result.max_concurrent_positions_seen,
        }
    )
    st.caption(
        f"Universe: `{', '.join(symbols)}` · interval `{interval}` · execution `{execution_label}`"
    )

    if assignments:
        st.markdown("#### Symbol Strategy Map")
        st.dataframe(pd.DataFrame(assignments), width="stretch", hide_index=True)
    if skipped_symbols:
        st.warning(
            "Some tickers were skipped because their data or strategy context could not be prepared: "
            + ", ".join(skipped_symbols)
        )

    closed = [t for t in result.trades if t.leveraged_return_pct is not None]
    if show_details and not result.equity_curve.empty:
        chart_window = st.radio(
            "Portfolio chart focus",
            options=CHART_WINDOW_OPTIONS,
            index=0,
            key="bt_portfolio_chart_window",
            horizontal=True,
            format_func=chart_window_label,
        )
        st.markdown("#### Portfolio Equity Curve")
        st.altair_chart(
            _equity_chart(result.equity_curve, "Managed Portfolio", chart_window),
            width="stretch",
        )

    if show_details and closed:
        trades_df = pd.DataFrame(
            [
                {
                    "symbol": t.symbol,
                    "strategy": strategy_display_name_from_id(getattr(t, "strategy_id", None), symbol=t.symbol),
                    "regime": _trade_regime(t),
                    "direction": t.direction.value,
                    "capital_allocated": t.capital_allocated,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "outcome": t.outcome.value if t.outcome else None,
                    "leveraged_return_pct": t.leveraged_return_pct,
                    "pnl": t.pnl,
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                    "notes": t.notes,
                }
                for t in closed
            ]
        )
        st.markdown("#### Per-Trade Return")
        st.altair_chart(pnl_distribution(trades_df), width="stretch")
        with st.expander("Managed Trade Log", expanded=False):
            st.dataframe(
                trades_df.rename(
                    columns={
                        "capital_allocated": "capital_allocated ($)",
                        "leveraged_return_pct": "return_pct (%)",
                        "pnl": "PnL ($)",
                    }
                ).sort_values("entry_time", ascending=False),
                width="stretch",
                hide_index=True,
            )
    elif not show_details:
        st.caption("Detailed managed charts and trade log are hidden to keep the initial page load light.")


def _execute_managed_portfolio_run(
    *,
    run_cfg: dict,
    strategies: list[dict],
) -> bool:
    symbols = list(run_cfg.get("symbols") or [])
    interval = str(run_cfg.get("interval") or "1m")
    start_date = pd.Timestamp(run_cfg.get("start_date")).date()
    end_date = pd.Timestamp(run_cfg.get("end_date")).date()
    execution_label = str(run_cfg.get("execution_label") or "Alpaca-realistic")
    execution_logic = str(run_cfg.get("execution_logic") or "alpaca")
    is_alpaca_logic = execution_logic == "alpaca"
    starting_equity = float(run_cfg.get("starting_equity", 5000.0))
    leverage = float(run_cfg.get("leverage", 1.0))
    capital_per_trade = float(run_cfg.get("capital_per_trade", 1000.0))
    max_open_positions = int(run_cfg.get("max_open_positions", 1))
    direction_filter = str(run_cfg.get("direction_filter") or "Both")
    replace_weaker = bool(run_cfg.get("replace_weaker", True))
    replacement_edge = float(run_cfg.get("replacement_edge", 10.0))
    use_risk = bool(run_cfg.get("use_risk", True))
    counter_signal_exit = bool(run_cfg.get("counter_signal_exit", True))
    monday_open_delay = bool(run_cfg.get("monday_open_delay", False))
    spread_pct = float(run_cfg.get("spread_pct", 0.06))
    slippage_pct = float(run_cfg.get("slippage_pct", 0.02))
    commission = float(run_cfg.get("commission", 0.0))
    enforce_rth = bool(run_cfg.get("enforce_rth", True))
    extended_hours = bool(run_cfg.get("extended_hours", False))
    enforce_pdt = bool(run_cfg.get("enforce_pdt", True))
    enforce_ssr = bool(run_cfg.get("enforce_ssr", True))
    enforce_fractional = bool(run_cfg.get("enforce_fractional", True))
    dynamic_sizing = bool(run_cfg.get("dynamic_sizing", False))

    symbol_inputs: list[ManagedPortfolioSymbolInput] = []
    assignments: list[dict[str, str]] = []
    skipped_symbols: list[str] = []
    raw_frame_cache: dict[str, pd.DataFrame] = {}
    companion_frame_cache: dict = {}
    prepared_frame_cache: dict[str, pd.DataFrame] = {}
    managed_run_started_at = time.perf_counter()
    log.info(
        f"managed_backtest: queued run for {len(symbols)} symbols, interval={interval}, "
        f"window={pd.Timestamp(start_date).date()}→{pd.Timestamp(end_date).date()}"
    )

    with st.status("Preparing managed portfolio backtest", expanded=True) as status:
        message_slot = st.empty()
        progress_bar = st.progress(0.0, text="0% · Starting managed backtest")
        total_symbols = max(len(symbols), 1)
        max_workers = min(8, total_symbols)
        message_slot.write(
            f"Loading and preparing {len(symbols)} ticker(s) using {max_workers} worker(s)…"
        )
        _render_stage_progress(progress_bar, 0.03, f"Preparing {len(symbols)} tickers")
        prepared_results: list[dict] = []
        prep_started_at = time.perf_counter()
        last_prepare_emit = {"count": 0, "ts": prep_started_at}
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="bt-port-prep") as executor:
            future_map = {
                executor.submit(
                    _prepare_managed_symbol_input,
                    idx=idx,
                    symbol=sym,
                    interval=interval,
                    start=start_date,
                    end=end_date,
                    strategies=strategies,
                    raw_frame_cache=raw_frame_cache,
                    companion_frame_cache=companion_frame_cache,
                    prepared_frame_cache=prepared_frame_cache,
                ): (idx, sym)
                for idx, sym in enumerate(symbols)
            }
            completed = 0
            for future in as_completed(future_map):
                idx, sym = future_map[future]
                completed += 1
                load_progress = 0.05 + (0.40 * (completed / total_symbols))
                try:
                    prepared = future.result()
                except Exception:
                    prepared = {"symbol": sym, "idx": idx, "skipped": True}
                prepared_results.append(prepared)
                now = time.perf_counter()
                should_emit = (
                    completed >= total_symbols
                    or (completed - last_prepare_emit["count"]) >= max(1, total_symbols // 10)
                    or (now - last_prepare_emit["ts"]) >= 1.0
                )
                if should_emit:
                    status_label = (
                        f"Prepared {prepared['symbol']}"
                        if not prepared.get("skipped")
                        else f"Skipped {prepared['symbol']}"
                    )
                    message_slot.write(f"{status_label} ({completed}/{len(symbols)})…")
                    _render_stage_progress(progress_bar, load_progress, status_label)
                    last_prepare_emit["count"] = completed
                    last_prepare_emit["ts"] = now

        prepared_results.sort(key=lambda item: int(item.get("idx", 0)))
        for prepared in prepared_results:
            sym = str(prepared.get("symbol") or "").strip().upper()
            if prepared.get("skipped"):
                if sym:
                    skipped_symbols.append(sym)
                continue
            symbol_input = prepared.get("symbol_input")
            assignment = prepared.get("assignment")
            if isinstance(symbol_input, ManagedPortfolioSymbolInput) and isinstance(assignment, dict):
                symbol_inputs.append(symbol_input)
                assignments.append(assignment)
            elif sym:
                skipped_symbols.append(sym)

        if not symbol_inputs:
            _render_stage_progress(progress_bar, 1.0, "Managed backtest failed")
            status.update(label="Managed portfolio backtest failed", state="error")
            st.error("None of the requested tickers could be prepared for the managed portfolio backtest.")
            return False
        prep_elapsed = time.perf_counter() - prep_started_at
        log.info(
            f"managed_backtest: prepared {len(symbol_inputs)}/{len(symbols)} symbols in {prep_elapsed:.2f}s"
        )

        from config.settings import RiskConfig
        from core.models import Direction as Dir

        risk_cfg = RiskConfig(
            max_capital_per_trade_pct=100.0,
            max_daily_loss_pct=100.0,
            max_open_positions=int(max_open_positions),
            default_max_loss_pct_of_capital=50.0,
        )
        risk_manager = RiskManager(risk_cfg) if use_risk else None
        dir_filter = None
        if direction_filter == "Long only":
            dir_filter = Dir.LONG
        elif direction_filter == "Short only":
            dir_filter = Dir.SHORT

        engine = ManagedPortfolioBacktestEngine(
            risk_manager=risk_manager,
            direction_filter=dir_filter,
            counter_signal_exit=bool(counter_signal_exit),
            spread_pct=float(spread_pct),
            slippage_pct=float(slippage_pct),
            commission_per_trade=float(commission),
            enforce_rth=bool(enforce_rth) if is_alpaca_logic else False,
            extended_hours=bool(extended_hours) if is_alpaca_logic else False,
            enforce_pdt=bool(enforce_pdt) if is_alpaca_logic else False,
            enforce_ssr=bool(enforce_ssr) if is_alpaca_logic else False,
            enforce_fractional=bool(enforce_fractional) if is_alpaca_logic else False,
            fill_diagnostic=True if is_alpaca_logic else False,
            enforce_monday_open_delay=bool(monday_open_delay),
            allow_replacement=bool(replace_weaker),
            replacement_score_edge_pct=float(replacement_edge),
            max_open_positions=int(max_open_positions),
            dynamic_sizing=bool(dynamic_sizing),
        )

        message_slot.write(
            f"Prepared {len(symbol_inputs)} ticker(s). Running shared-balance simulation…"
        )
        _render_stage_progress(progress_bar, 0.52, f"Running simulation across {len(symbol_inputs)} tickers")
        sim_started_at = time.perf_counter()
        last_progress_emit = {"overall": -1.0, "ts": 0.0}

        def _portfolio_progress(progress: float, label: str) -> None:
            overall = 0.52 + (0.43 * max(0.0, min(1.0, float(progress))))
            now = time.perf_counter()
            if (
                overall >= 0.999
                or overall - last_progress_emit["overall"] >= 0.05
                or (now - last_progress_emit["ts"]) >= 2.0
            ):
                _render_stage_progress(progress_bar, overall, label)
                message_slot.write(f"Managed simulation: {label}")
                last_progress_emit["overall"] = overall
                last_progress_emit["ts"] = now

        result = engine.run(
            symbol_inputs,
            leverage=float(leverage),
            capital_per_trade=float(capital_per_trade),
            starting_equity=float(starting_equity),
            progress_cb=_portfolio_progress,
        )
        _render_stage_progress(progress_bar, 1.0, "Managed backtest finished")
        status.update(
            label=f"Managed portfolio backtest finished — {len(symbol_inputs)} ticker(s)",
            state="complete",
        )
        sim_elapsed = time.perf_counter() - sim_started_at
        total_elapsed = time.perf_counter() - managed_run_started_at
        log.info(
            f"managed_backtest: simulation {sim_elapsed:.2f}s, total {total_elapsed:.2f}s, "
            f"trades={result.total_trades}, candidates={result.candidate_entries}"
        )
        st.session_state[_BT_PORTFOLIO_RESULT_KEY] = {
            "result": result,
            "symbols": symbols,
            "interval": interval,
            "start": pd.Timestamp(start_date),
            "end": pd.Timestamp(end_date),
            "starting_equity": float(starting_equity),
            "capital_per_trade": float(capital_per_trade),
            "leverage": float(leverage),
            "max_open_positions": int(max_open_positions),
            "execution_label": execution_label,
            "execution_logic": execution_logic,
            "enforce_ssr": bool(enforce_ssr) if is_alpaca_logic else False,
            "dynamic_sizing": bool(dynamic_sizing),
            "assignments": assignments,
            "skipped_symbols": skipped_symbols,
        }
        _persist_managed_portfolio_snapshot(st.session_state[_BT_PORTFOLIO_RESULT_KEY])
    return True


def _render_managed_portfolio_backtest() -> None:
    payload = st.session_state.get(_BT_PORTFOLIO_RESULT_KEY)
    pending_run = st.session_state.get(_BT_PORT_PENDING_RUN_KEY)
    if "bt_port_equity" not in st.session_state:
        st.session_state["bt_port_equity"] = 5000.0
    if "bt_port_cap" not in st.session_state:
        st.session_state["bt_port_cap"] = 1000.0
    if "bt_port_max_open" not in st.session_state:
        equity_seed = float(st.session_state.get("bt_port_equity") or 5000.0)
        cap_seed = max(float(st.session_state.get("bt_port_cap") or 1000.0), 1.0)
        st.session_state["bt_port_max_open"] = max(1, min(20, int(equity_seed // cap_seed)))
    if "bt_port_show_details" not in st.session_state:
        st.session_state["bt_port_show_details"] = False
    if "bt_port_show_details_pref" not in st.session_state:
        st.session_state["bt_port_show_details_pref"] = bool(st.session_state["bt_port_show_details"])
    if "bt_port_dynamic_sizing" not in st.session_state:
        st.session_state["bt_port_dynamic_sizing"] = bool((payload or {}).get("dynamic_sizing", True))
    current_symbol = (
        st.session_state.get("bt_symbol_live")
        or st.session_state.get("loaded_symbol")
        or "APLD"
    )
    current_start = st.session_state.get("bt_start_live") or st.session_state.get("loaded_start")
    current_end = st.session_state.get("bt_end_live") or st.session_state.get("loaded_end")
    default_symbols = st.session_state.get("bt_port_symbols") or str(current_symbol).upper()
    interval_options = ["1m", "5m", "15m", "30m", "1h", "1d"]
    strategies = ordered_strategy_items(list_strategies())
    policy_opts = available_policies()
    policy_labels = [label for _, label in policy_opts]
    policy_names = [name for name, _ in policy_opts]
    default_policy_idx = policy_names.index("alpaca") if "alpaca" in policy_names else 0

    st.subheader("Portfolio Setup")
    st.caption(
        "Backtest multiple tickers under one shared balance. The engine auto-assigns the best-fit strategy per ticker, "
        "ranks simultaneous candidates, and can size or replace positions when capital is tight."
    )
    if pending_run:
        log.info("managed_backtest: pending run detected on rerender; starting execution")
        try:
            _execute_managed_portfolio_run(run_cfg=pending_run, strategies=strategies)
        finally:
            st.session_state.pop(_BT_PORT_PENDING_RUN_KEY, None)
        payload = st.session_state.get(_BT_PORTFOLIO_RESULT_KEY)

    with st.form("bt_managed_portfolio_form", clear_on_submit=False):
        tcol1, tcol2 = st.columns([2.5, 1])
        with tcol1:
            symbols_raw = st.text_area(
                "Ticker list",
                value=default_symbols,
                key="bt_port_symbols",
                help="Comma-, space-, semicolon-, or newline-separated symbols. Strategies are auto-assigned ticker by ticker.",
                height=100,
            )
        with tcol2:
            interval_default = st.session_state.get("bt_port_interval") or (st.session_state.get("bt_interval_live") or st.session_state.get("loaded_interval") or "5m")
            if interval_default not in interval_options:
                interval_default = "5m"
            interval = st.selectbox(
                "Interval",
                interval_options,
                index=interval_options.index(interval_default),
                key="bt_port_interval",
            )
            start_date = st.date_input(
                "Start",
                value=_BT_DEFAULT_START,
                key="bt_port_start",
            )
            end_date = st.date_input(
                "End",
                value=_BT_DEFAULT_END,
                key="bt_port_end",
            )

        s1, s2, s3, s4 = st.columns(4)
        with s1:
            starting_equity = st.number_input("Starting equity ($)", min_value=1000.0, value=5000.0, key="bt_port_equity")
            leverage = st.number_input("Leverage", min_value=1.0, max_value=100.0, value=1.0, step=0.5, key="bt_port_lev")
        with s2:
            capital_per_trade = st.number_input("Capital per position ($)", min_value=100.0, value=1000.0, key="bt_port_cap")
            max_open_positions = st.number_input("Max open positions", min_value=1, max_value=20, value=int(st.session_state.get("bt_port_max_open", 1)), step=1, key="bt_port_max_open")
        with s3:
            direction_filter = st.selectbox("Direction filter", ["Both", "Long only", "Short only"], key="bt_port_dir")
            replace_weaker = st.checkbox("Replace weaker open position", value=True, key="bt_port_replace")
        with s4:
            replacement_edge = st.number_input(
                "Replacement edge (%)",
                min_value=0.0,
                max_value=200.0,
                value=10.0,
                step=1.0,
                key="bt_port_replace_edge",
                help="Required score advantage before a fresh setup can kick out the weakest current position.",
            )
            use_risk = st.checkbox("Apply risk manager", value=True, key="bt_port_risk")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            counter_signal_exit = st.checkbox("Counter-signal exit", value=True, key="bt_port_counter")
            monday_open_delay = st.checkbox("No Monday trades first 30m", value=False, key="bt_port_monday")
        with c2:
            spread_pct = st.number_input("Spread % (round-trip)", 0.0, 2.0, 0.06, step=0.01, format="%.2f", key="bt_port_spread")
            slippage_pct = st.number_input("Slippage % (round-trip)", 0.0, 2.0, 0.02, step=0.01, format="%.2f", key="bt_port_slip")
        with c3:
            commission = st.number_input("Commission per trade ($)", 0.0, 10.0, 0.0, step=0.10, format="%.2f", key="bt_port_comm")
            execution_label = st.selectbox("Entry-gate policy", policy_labels, index=default_policy_idx, key="bt_port_exec_logic")
        with c4:
            enforce_rth = st.checkbox("RTH only", value=True, key="bt_port_rth")
            extended_hours = st.checkbox("Extended hours", value=False, key="bt_port_ext_hrs")
            enforce_pdt = st.checkbox("PDT (<$25k)", value=True, key="bt_port_pdt")
            enforce_ssr = st.checkbox("SSR (shorts)", value=True, key="bt_port_ssr")
            enforce_fractional = st.checkbox("Fractional rule", value=True, key="bt_port_frac")
        dynamic_sizing = st.checkbox(
            "Dynamic sizing (0.5x/1.0x/1.5x/2.0x, higher when cash is abundant)",
            value=bool(st.session_state.get("bt_port_dynamic_sizing", False)),
            key="bt_port_dynamic_sizing",
            help=(
                "Keeps the base position size, but lets a clear standout expand to 1.5x or 2.0x. "
                "When only a few strong setups exist and plenty of cash would otherwise sit idle, "
                "the allocator can scale the best one further. "
                "If extra slots are available and cash is tighter than the number of near-peer setups, "
                "the engine can also downsize to 0.5x to fit more names."
            ),
        )

        st.caption(
            "The portfolio runner uses each ticker's current default or preset strategy behavior automatically. "
            "That keeps the test focused on shared-balance prioritization, sizing, and replacement decisions."
        )
        recommended_slots = max(1, min(20, int(float(starting_equity) // max(float(capital_per_trade), 1.0))))
        if int(max_open_positions) < recommended_slots:
            st.warning(
                f"Current balance could support about {recommended_slots} concurrent positions at ${float(capital_per_trade):,.0f} each, "
                f"but `Max open positions` is set to {int(max_open_positions)}. That will force a much tighter rotating book than the independent single-ticker runs."
            )
        show_details_after_run = st.checkbox(
            "Show portfolio charts and trade log right after the run",
            key="bt_port_show_details_pref",
            help="If enabled, the detailed portfolio charts and trade log open automatically as soon as the simulation finishes.",
        )
        submitted = st.form_submit_button("Run Portfolio Backtest", type="primary")

    if submitted:
        symbols = _parse_symbol_list(symbols_raw)
        if not symbols:
            st.error("Add at least one ticker to run the portfolio backtest.")
            return
        if pd.Timestamp(start_date) > pd.Timestamp(end_date):
            st.error("Start date must be before end date.")
            return

        st.session_state["bt_port_show_details"] = bool(show_details_after_run)
        st.session_state[_BT_PENDING_MODE_KEY] = _BT_MODE_PORTFOLIO
        log.info(
            f"managed_backtest: submit clicked for {len(symbols)} symbols, interval={interval}, "
            f"window={pd.Timestamp(start_date).date()}→{pd.Timestamp(end_date).date()}"
        )
        run_cfg = {
            "symbols": symbols,
            "interval": interval,
            "start_date": pd.Timestamp(start_date),
            "end_date": pd.Timestamp(end_date),
            "starting_equity": float(starting_equity),
            "leverage": float(leverage),
            "capital_per_trade": float(capital_per_trade),
            "max_open_positions": int(max_open_positions),
            "direction_filter": direction_filter,
            "replace_weaker": bool(replace_weaker),
            "replacement_edge": float(replacement_edge),
            "use_risk": bool(use_risk),
            "counter_signal_exit": bool(counter_signal_exit),
            "monday_open_delay": bool(monday_open_delay),
            "spread_pct": float(spread_pct),
            "slippage_pct": float(slippage_pct),
            "commission": float(commission),
            "execution_label": execution_label,
            "execution_logic": policy_names[policy_labels.index(execution_label)],
            "enforce_rth": bool(enforce_rth),
            "extended_hours": bool(extended_hours),
            "enforce_pdt": bool(enforce_pdt),
            "enforce_ssr": bool(enforce_ssr),
            "enforce_fractional": bool(enforce_fractional),
            "dynamic_sizing": bool(dynamic_sizing),
        }
        st.session_state.pop(_BT_PORT_PENDING_RUN_KEY, None)
        log.info("managed_backtest: starting direct execution from submit render")
        _execute_managed_portfolio_run(run_cfg=run_cfg, strategies=strategies)

    payload = st.session_state.get(_BT_PORTFOLIO_RESULT_KEY)
    if payload:
        show_details = bool(st.session_state.get("bt_port_show_details", False))
        st.divider()
        _show_managed_portfolio_results(payload, show_details=show_details)
