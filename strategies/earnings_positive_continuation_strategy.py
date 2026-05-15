"""
Research-only positive earnings continuation strategy.

This branch targets a different shape from the main earnings overshoot hybrid:
  - positive off-hours earnings
  - meaningful upside repricing at the open
  - strong enough continuation profile to keep pressing higher
  - no attempt to fade the move intraday

It is intentionally simple:
  - auto-activates only on locally known positive off-hours earnings reaction days
  - enters a long after a configurable opening confirmation window
  - exits at the regular-session close
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.models import Signal, SignalAction
from data.earnings_calendar import merged_events_for_symbol
from strategies.base import BaseStrategy, register_strategy
from strategies.earnings_overshoot_hybrid_strategy import (
    RTH_CLOSE,
    RTH_OPEN,
    _build_session_arrays,
    _event_with_live_metrics,
    _find_continuation_long_entry_idx,
)


@register_strategy
class EarningsPositiveContinuationStrategy(BaseStrategy):
    strategy_id = "earnings_positive_continuation"
    name = "Earnings Positive Continuation (Research)"
    description = (
        "Research-only stock earnings continuation strategy. It auto-uses the local positive off-hours earnings "
        "event table and looks for moderate-to-large upside repricings that keep following through after the open."
    )
    ui_hidden = True

    def default_params(self) -> dict[str, Any]:
        return {
            "gap_pct_min": 4.0,
            "peak_vs_prev_close_min": 12.0,
            "peak_vs_prev_close_max": 30.0,
            "confirm5_min": 0.0,
            "confirm15_min": None,
            "confirm30_min": None,
            "first30_low_from_open_min": None,
            "entry_minute": RTH_OPEN + 30,
            "stop_mode": "pct",
            "stop_pct": 4.0,
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
    def _dynamic_long_stop(arrays: dict[str, np.ndarray], entry_idx: int, stop_pct: float) -> float:
        entry_px = float(arrays["close"][entry_idx])
        local_low = float(np.nanmin(arrays["low"][: entry_idx + 1]))
        return min(local_low * 0.994, entry_px * (1.0 - float(stop_pct) / 100.0))

    def _long_stop(self, arrays: dict[str, np.ndarray], entry_idx: int, params: dict[str, Any]) -> float | None:
        mode = str(params.get("stop_mode", "dynamic")).lower()
        stop_pct = float(params.get("stop_pct", 1.5))
        entry_px = float(arrays["close"][entry_idx])
        if mode == "none":
            return None
        if mode == "pct":
            return entry_px * (1.0 - stop_pct / 100.0)
        return self._dynamic_long_stop(arrays, entry_idx, stop_pct)

    def generate_signals_bulk(self, data: pd.DataFrame, symbol: str) -> tuple[list, list]:
        n = len(data)
        actions, meta = self._empty_meta(n)
        if data.empty or "date" not in data.columns:
            return actions, meta

        params = self.resolve_params(symbol=symbol)
        gap_pct_min = float(params.get("gap_pct_min", 4.0))
        peak_vs_prev_close_min = float(params.get("peak_vs_prev_close_min", 12.0))
        peak_vs_prev_close_max = float(params.get("peak_vs_prev_close_max", 30.0))
        confirm5_min = float(params.get("confirm5_min", 0.0))
        confirm15_min = params.get("confirm15_min")
        confirm30_min = params.get("confirm30_min")
        first30_low_from_open_min = params.get("first30_low_from_open_min")
        entry_minute = int(params.get("entry_minute", RTH_OPEN + 5))

        symbol_events = merged_events_for_symbol(symbol, sign="positive")
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
            event = _event_with_live_metrics(event, work, session, arrays)

            gap_pct = float(event.get("gap_pct", np.nan))
            peak_vs_prev_close = float(event.get("peak_vs_prev_close_pct", np.nan))
            confirm5 = float(event.get("confirm5_close_from_open_pct", np.nan))
            confirm15 = float(event.get("confirm15_close_from_open_pct", np.nan))
            confirm30 = float(event.get("confirm30_close_from_open_pct", np.nan))
            first30_low_from_open = float(event.get("first30_low_from_open_pct", np.nan))

            if not np.isfinite(gap_pct) or gap_pct < gap_pct_min:
                continue
            if not np.isfinite(peak_vs_prev_close):
                continue
            if peak_vs_prev_close < peak_vs_prev_close_min or peak_vs_prev_close > peak_vs_prev_close_max:
                continue
            if not np.isfinite(confirm5) or confirm5 < confirm5_min:
                continue
            if confirm15_min is not None:
                if (not np.isfinite(confirm15)) or confirm15 < float(confirm15_min):
                    continue
            if confirm30_min is not None:
                if (not np.isfinite(confirm30)) or confirm30 < float(confirm30_min):
                    continue
            if first30_low_from_open_min is not None:
                if (not np.isfinite(first30_low_from_open)) or first30_low_from_open < float(first30_low_from_open_min):
                    continue

            entry_idx = _find_continuation_long_entry_idx(arrays, entry_minute=entry_minute)
            if entry_idx is None:
                continue

            g_entry = int(arrays["orig_index"][entry_idx])
            actions[g_entry] = SignalAction.BUY
            meta[g_entry] = {
                "suggested_tp": None,
                "suggested_sl": self._long_stop(arrays, entry_idx, params),
                "metadata": {
                    "regime": "earnings_positive_continuation_long",
                    "session_exit": "eod",
                    "earnings_positive_continuation": True,
                    "event_reaction_date": reaction_date,
                    "verdict_reason": (
                        "Positive off-hours earnings repriced the stock higher, the opening confirmation stayed firm, "
                        "and the strategy followed the continuation path into the close."
                    ),
                },
            }

        return actions, meta
