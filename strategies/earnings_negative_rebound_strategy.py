"""
Research-only mirrored negative earnings rebound strategy.

This strategy is intentionally narrow:
  - It auto-activates only on locally known negative off-hours earnings
    reaction days from the research artifact table.
  - It looks for a deep downside overshoot below the previous close.
  - It then waits for a confirmed intraday rebound back toward the fair zone
    and takes a single long rebound trade.

The goal is to validate the mirrored negative-earnings family in the normal
Backtester before we decide how to expose earnings-date input or a live feed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.models import Signal, SignalAction
from strategies.base import BaseStrategy, register_strategy


ROOT = Path(__file__).resolve().parents[1]
EVENTS_PATH = ROOT / "artifacts" / "optimization" / "earnings_overshoot_dump_events_labeled.csv"
RTH_OPEN = 9 * 60 + 30
RTH_CLOSE = 16 * 60

_EVENT_TABLE_CACHE: pd.DataFrame | None = None
_EVENTS_BY_SYMBOL: dict[str, pd.DataFrame] = {}


def _safe_pct(current: float, base: float) -> float:
    try:
        cur = float(current)
        ref = float(base)
    except Exception:
        return float("nan")
    if not np.isfinite(cur) or not np.isfinite(ref) or ref == 0.0:
        return float("nan")
    return (cur / ref - 1.0) * 100.0


def _load_event_table() -> pd.DataFrame:
    global _EVENT_TABLE_CACHE
    if _EVENT_TABLE_CACHE is not None:
        return _EVENT_TABLE_CACHE
    if not EVENTS_PATH.exists():
        _EVENT_TABLE_CACHE = pd.DataFrame()
        return _EVENT_TABLE_CACHE

    df = pd.read_csv(EVENTS_PATH)
    if df.empty:
        _EVENT_TABLE_CACHE = pd.DataFrame()
        return _EVENT_TABLE_CACHE

    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["reaction_date"] = pd.to_datetime(df["reaction_date"], errors="coerce").dt.date.astype(str)
    df["timing"] = df["timing"].astype(str).str.lower()
    df = df[pd.to_numeric(df["surprise_pct"], errors="coerce") < 0].copy()
    df = df[df["timing"].isin({"bmo", "amc"})].copy()
    _EVENT_TABLE_CACHE = df.reset_index(drop=True)
    return _EVENT_TABLE_CACHE


def _load_events_for_symbol(symbol: str) -> pd.DataFrame:
    symbol_u = str(symbol).upper()
    if symbol_u in _EVENTS_BY_SYMBOL:
        return _EVENTS_BY_SYMBOL[symbol_u]
    base = _load_event_table()
    filtered = base[base["symbol"] == symbol_u].copy().reset_index(drop=True)
    _EVENTS_BY_SYMBOL[symbol_u] = filtered
    return filtered


def _build_session_arrays(session: pd.DataFrame) -> dict[str, np.ndarray]:
    work = session.copy()
    work["minutes_raw"] = work["date"].dt.hour * 60 + work["date"].dt.minute
    typical = (work["high"] + work["low"] + work["close"]) / 3.0
    vol = pd.to_numeric(work["volume"], errors="coerce").fillna(0.0)
    cum_vol = vol.cumsum().replace(0.0, np.nan)
    anchored_vwap = (typical * vol).cumsum() / cum_vol

    arrays = {
        "date": work["date"].to_numpy(copy=True),
        "orig_index": pd.to_numeric(work["orig_index"], errors="coerce").to_numpy(dtype=np.int64, copy=True),
        "minutes_et": pd.to_numeric(work["minutes_raw"], errors="coerce").to_numpy(dtype=np.int32, copy=True),
        "open": pd.to_numeric(work["open"], errors="coerce").to_numpy(dtype=np.float64, copy=True),
        "high": pd.to_numeric(work["high"], errors="coerce").to_numpy(dtype=np.float64, copy=True),
        "low": pd.to_numeric(work["low"], errors="coerce").to_numpy(dtype=np.float64, copy=True),
        "close": pd.to_numeric(work["close"], errors="coerce").to_numpy(dtype=np.float64, copy=True),
        "anchored_vwap": pd.to_numeric(anchored_vwap, errors="coerce").to_numpy(dtype=np.float64, copy=True),
    }
    arrays["close_from_vwap_pct"] = (arrays["close"] / arrays["anchored_vwap"] - 1.0) * 100.0
    arrays["ret_5m_pct"] = pd.Series(arrays["close"]).pct_change(5).to_numpy(dtype=np.float64, copy=True) * 100.0
    arrays["running_trough_low"] = np.minimum.accumulate(arrays["low"])
    return arrays


def _find_long_rebound_entry_idx(
    event: dict[str, Any],
    arrays: dict[str, np.ndarray],
    *,
    downside_prev_close_min: float,
    start_minute: int,
    max_minute: int,
    rebound_from_trough_min: float,
    vwap_reclaim_min: float,
    rebound_impulse_min: float,
    entry_min_close_from_open_pct: float | None = None,
) -> int | None:
    prev_close = float(event.get("prev_close", np.nan))
    if not np.isfinite(prev_close) or prev_close <= 0.0:
        return None

    minutes = arrays["minutes_et"]
    close = arrays["close"]
    low = arrays["low"]
    open_px = arrays["open"]
    close_from_vwap = arrays["close_from_vwap_pct"]
    ret_5m = arrays["ret_5m_pct"]

    running_trough = np.minimum.accumulate(low)
    running_trough_vs_prev = (running_trough / prev_close - 1.0) * 100.0
    rebound_from_trough = (close / running_trough - 1.0) * 100.0
    close_from_open_pct = (close / open_px - 1.0) * 100.0

    cond = (
        (minutes >= start_minute)
        & (minutes <= max_minute)
        & np.isfinite(close)
        & np.isfinite(close_from_vwap)
        & np.isfinite(ret_5m)
        & np.isfinite(running_trough_vs_prev)
        & np.isfinite(rebound_from_trough)
        & np.isfinite(close_from_open_pct)
        & (running_trough_vs_prev <= -downside_prev_close_min)
        & (rebound_from_trough >= rebound_from_trough_min)
        & (close_from_vwap >= vwap_reclaim_min)
        & (ret_5m >= rebound_impulse_min)
    )
    if entry_min_close_from_open_pct is not None:
        cond &= close_from_open_pct >= float(entry_min_close_from_open_pct)
    hits = np.flatnonzero(cond)
    if hits.size == 0:
        return None
    return int(hits[0])


@register_strategy
class EarningsNegativeReboundStrategy(BaseStrategy):
    strategy_id = "earnings_negative_rebound"
    name = "Earnings Negative Rebound (Research)"
    description = (
        "Research-only stock earnings strategy. It auto-uses the local earnings-event table, "
        "then looks for deep negative off-hours overshoots that confirm into an intraday rebound long."
    )
    ui_hidden = True

    def default_params(self) -> dict[str, Any]:
        return {
            "downside_prev_close_min": 12.5,
            "start_minute": 12 * 60,
            "max_minute": 15 * 60,
            "rebound_from_trough_min": 4.0,
            "vwap_reclaim_min": -0.5,
            "rebound_impulse_min": 0.5,
            "entry_min_close_from_open_pct": 0.0,
            "exit_vwap_touch_buffer": 1.0,
        }

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Signal:
        return Signal(strategy_id=self.strategy_id, symbol=symbol, action=SignalAction.HOLD)

    @staticmethod
    def _empty_meta(n: int) -> tuple[list[SignalAction], list[dict[str, Any]]]:
        return [SignalAction.HOLD] * n, [
            {"suggested_tp": None, "suggested_sl": None, "metadata": {}}
            for _ in range(n)
        ]

    @staticmethod
    def _long_stop(arrays: dict[str, np.ndarray], entry_idx: int) -> float:
        entry_px = float(arrays["close"][entry_idx])
        local_low = float(np.nanmin(arrays["low"][: entry_idx + 1]))
        return min(local_low * 0.994, entry_px * 0.985)

    def generate_signals_bulk(self, data: pd.DataFrame, symbol: str) -> tuple[list, list]:
        n = len(data)
        actions, meta = self._empty_meta(n)
        if data.empty or "date" not in data.columns:
            return actions, meta

        params = self.resolve_params(symbol=symbol)
        symbol_events = _load_events_for_symbol(symbol)
        if symbol_events.empty:
            return actions, meta

        work = data.copy().reset_index().rename(columns={"index": "orig_index"})
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
        work = work.dropna(subset=["date"]).copy()
        if work.empty:
            return actions, meta

        work["session_date"] = work["date"].dt.date.astype(str)
        work["minutes_raw"] = work["date"].dt.hour * 60 + work["date"].dt.minute

        available_dates = set(work["session_date"].astype(str).unique().tolist())
        event_rows = symbol_events[symbol_events["reaction_date"].isin(available_dates)].copy()
        if event_rows.empty:
            return actions, meta

        for event in event_rows.to_dict(orient="records"):
            reaction_date = str(event.get("reaction_date"))
            session = work[
                (work["session_date"] == reaction_date)
                & (work["minutes_raw"] >= RTH_OPEN)
                & (work["minutes_raw"] <= RTH_CLOSE)
            ].copy()
            if session.empty:
                continue
            session = session.sort_values("date").reset_index(drop=True)
            arrays = _build_session_arrays(session)
            entry_idx = _find_long_rebound_entry_idx(
                event,
                arrays,
                downside_prev_close_min=float(params.get("downside_prev_close_min", 12.5)),
                start_minute=int(params.get("start_minute", 12 * 60)),
                max_minute=int(params.get("max_minute", 15 * 60)),
                rebound_from_trough_min=float(params.get("rebound_from_trough_min", 4.0)),
                vwap_reclaim_min=float(params.get("vwap_reclaim_min", -0.5)),
                rebound_impulse_min=float(params.get("rebound_impulse_min", 0.5)),
                entry_min_close_from_open_pct=params.get("entry_min_close_from_open_pct", 0.0),
            )
            if entry_idx is None:
                continue

            close_from_vwap = arrays["close_from_vwap_pct"]
            exit_idx = None
            for idx in range(entry_idx + 1, len(close_from_vwap)):
                if np.isfinite(close_from_vwap[idx]) and close_from_vwap[idx] >= float(
                    params.get("exit_vwap_touch_buffer", 1.0)
                ):
                    exit_idx = int(idx)
                    break
            if exit_idx is None:
                exit_idx = len(close_from_vwap) - 1

            g_entry = int(arrays["orig_index"][entry_idx])
            g_exit = int(arrays["orig_index"][exit_idx])

            actions[g_entry] = SignalAction.BUY
            meta[g_entry] = {
                "suggested_tp": None,
                "suggested_sl": self._long_stop(arrays, entry_idx),
                "metadata": {
                    "regime": "earnings_negative_rebound_long",
                    "session_exit": "eod",
                    "earnings_negative_rebound": True,
                    "event_reaction_date": reaction_date,
                    "verdict_reason": (
                        "Negative off-hours earnings overshot deeply below the prior close, "
                        "then confirmed an intraday rebound back toward the fair zone."
                    ),
                },
            }
            if g_exit > g_entry:
                actions[g_exit] = SignalAction.SELL
                meta[g_exit] = {
                    "suggested_tp": None,
                    "suggested_sl": None,
                    "metadata": {
                        "regime": "earnings_negative_rebound_exit",
                        "cover_only": True,
                        "earnings_negative_rebound": True,
                        "exit_reason": "vwap_touch",
                    },
                }

        return actions, meta
