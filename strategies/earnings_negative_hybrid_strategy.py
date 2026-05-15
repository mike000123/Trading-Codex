"""
Research-only mirrored negative earnings hybrid.

This strategy is intentionally narrow:
  - It auto-activates only on locally known negative off-hours earnings
    reaction days from the research artifact table.
  - Early persistent selloffs can route into a continuation short branch.
  - Most remaining eligible days run a single long rebound branch after a deep panic.
  - Only the deepest panic-and-rebound shapes are allowed to try an optional
    second-leg short after the rebound fails.

The goal is to keep the negative branch general and testable in Backtester
without blindly forcing the second-leg short onto every rebound day.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.models import Signal, SignalAction
from data.earnings_calendar import merged_events_for_symbol
from strategies.base import BaseStrategy, register_strategy


RTH_OPEN = 9 * 60 + 30
RTH_CLOSE = 16 * 60

def _safe_pct(current: float, base: float) -> float:
    try:
        cur = float(current)
        ref = float(base)
    except Exception:
        return float("nan")
    if not np.isfinite(cur) or not np.isfinite(ref) or ref == 0.0:
        return float("nan")
    return (cur / ref - 1.0) * 100.0


def _build_session_arrays(session: pd.DataFrame) -> dict[str, np.ndarray]:
    work = session.copy()
    work["minutes_raw"] = work["date"].dt.hour * 60 + work["date"].dt.minute
    typical = (work["high"] + work["low"] + work["close"]) / 3.0
    vol = pd.to_numeric(work["volume"], errors="coerce").fillna(0.0)
    cum_vol = vol.cumsum().replace(0.0, np.nan)
    anchored_vwap = (typical * vol).cumsum() / cum_vol
    ema_fast = pd.Series(work["close"]).ewm(span=5, adjust=False).mean()
    ema_slow = pd.Series(work["close"]).ewm(span=13, adjust=False).mean()

    arrays = {
        "date": work["date"].to_numpy(copy=True),
        "orig_index": pd.to_numeric(work["orig_index"], errors="coerce").to_numpy(dtype=np.int64, copy=True),
        "minutes_et": pd.to_numeric(work["minutes_raw"], errors="coerce").to_numpy(dtype=np.int32, copy=True),
        "open": pd.to_numeric(work["open"], errors="coerce").to_numpy(dtype=np.float64, copy=True),
        "high": pd.to_numeric(work["high"], errors="coerce").to_numpy(dtype=np.float64, copy=True),
        "low": pd.to_numeric(work["low"], errors="coerce").to_numpy(dtype=np.float64, copy=True),
        "close": pd.to_numeric(work["close"], errors="coerce").to_numpy(dtype=np.float64, copy=True),
        "anchored_vwap": pd.to_numeric(anchored_vwap, errors="coerce").to_numpy(dtype=np.float64, copy=True),
        "ema_fast": ema_fast.to_numpy(dtype=np.float64, copy=True),
        "ema_slow": ema_slow.to_numpy(dtype=np.float64, copy=True),
        "ema_fast_slope": ema_fast.diff().to_numpy(dtype=np.float64, copy=True),
    }
    arrays["close_from_vwap_pct"] = (arrays["close"] / arrays["anchored_vwap"] - 1.0) * 100.0
    arrays["ret_5m_pct"] = pd.Series(arrays["close"]).pct_change(5).to_numpy(dtype=np.float64, copy=True) * 100.0
    arrays["running_trough_low"] = np.minimum.accumulate(arrays["low"])
    return arrays


def _event_with_live_metrics(
    event: dict[str, Any],
    full_frame: pd.DataFrame,
    session: pd.DataFrame,
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    enriched = dict(event or {})
    if session.empty:
        return enriched

    session_open = float(arrays["open"][0]) if len(arrays["open"]) else float("nan")
    session_start = pd.Timestamp(session["date"].iloc[0])
    prior = full_frame[full_frame["date"] < session_start]
    prev_close = float(enriched.get("prev_close", np.nan))
    if (not np.isfinite(prev_close)) and not prior.empty:
        try:
            prev_close = float(pd.to_numeric(prior["close"], errors="coerce").dropna().iloc[-1])
        except Exception:
            prev_close = float("nan")
    enriched["prev_close"] = prev_close

    if "reaction_date" not in enriched or not str(enriched.get("reaction_date") or "").strip():
        enriched["reaction_date"] = str(session["session_date"].iloc[0])

    enriched["gap_pct"] = _safe_pct(session_open, prev_close)

    def _confirm_close_from_open(minute_target: int) -> float:
        hits = np.flatnonzero(arrays["minutes_et"] >= minute_target)
        if hits.size == 0:
            return float("nan")
        idx = int(hits[0])
        return _safe_pct(float(arrays["close"][idx]), session_open)

    enriched["confirm5_close_from_open_pct"] = _confirm_close_from_open(RTH_OPEN + 5)
    enriched["confirm15_close_from_open_pct"] = _confirm_close_from_open(RTH_OPEN + 15)
    enriched["confirm30_close_from_open_pct"] = _confirm_close_from_open(RTH_OPEN + 30)

    try:
        trough_low = float(np.nanmin(arrays["low"]))
    except Exception:
        trough_low = float("nan")
    enriched["trough_vs_prev_close_pct"] = _safe_pct(trough_low, prev_close)
    return enriched


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


def _find_short_after_rebound_idx(
    arrays: dict[str, np.ndarray],
    *,
    start_idx: int,
    max_entry_bars: int,
    pullback_from_peak_min: float,
    vwap_break_max: float,
    downside_impulse_min: float,
) -> tuple[int | None, str | None]:
    close = arrays["close"]
    high = arrays["high"]
    close_from_vwap = arrays["close_from_vwap_pct"]
    ret_5m = arrays["ret_5m_pct"]
    n = len(close)
    if start_idx >= n - 1:
        return None, None
    end_idx = min(start_idx + max_entry_bars, n - 1)
    rebound_peak = float(np.nanmax(high[start_idx:end_idx + 1]))
    if not np.isfinite(rebound_peak):
        return None, None
    pullback_from_peak = (close / rebound_peak - 1.0) * 100.0
    for idx in range(start_idx + 1, end_idx + 1):
        if not np.isfinite(close[idx]) or not np.isfinite(close_from_vwap[idx]) or not np.isfinite(ret_5m[idx]):
            continue
        if (
            np.isfinite(pullback_from_peak[idx])
            and pullback_from_peak[idx] <= -pullback_from_peak_min
            and close_from_vwap[idx] <= vwap_break_max
            and ret_5m[idx] <= -downside_impulse_min
        ):
            return idx, "rebound_failure"
    return None, None


def _find_continuation_short_entry_idx(
    arrays: dict[str, np.ndarray],
    *,
    entry_minute: int,
) -> int | None:
    minutes = arrays["minutes_et"]
    close = arrays["close"]
    hits = np.flatnonzero((minutes >= int(entry_minute)) & np.isfinite(close))
    if hits.size == 0:
        return None
    return int(hits[0])


def _exit_long(
    arrays: dict[str, np.ndarray],
    *,
    entry_idx: int,
    exit_mode: str,
    vwap_touch_buffer: float,
    ema_roll_gain_min: float,
    max_hold_bars: int,
) -> tuple[int, str]:
    close = arrays["close"]
    close_from_vwap = arrays["close_from_vwap_pct"]
    ema_fast = arrays["ema_fast"]
    ema_fast_slope = arrays["ema_fast_slope"]
    n = len(close)

    if exit_mode.startswith("time_"):
        hold_bars = int(exit_mode.split("_", 1)[1])
        return min(entry_idx + hold_bars, n - 1), f"time_{hold_bars}"

    if exit_mode == "vwap_touch":
        for idx in range(entry_idx + 1, min(entry_idx + max_hold_bars + 1, n)):
            if np.isfinite(close_from_vwap[idx]) and close_from_vwap[idx] >= vwap_touch_buffer:
                return idx, "vwap_touch"
        return min(entry_idx + max_hold_bars, n - 1), "vwap_touch_timeout"

    if exit_mode == "ema_roll":
        for idx in range(entry_idx + 1, min(entry_idx + max_hold_bars + 1, n)):
            if not np.isfinite(close[idx]):
                continue
            gain = _safe_pct(float(close[idx]), float(close[entry_idx]))
            if (
                np.isfinite(gain)
                and gain >= ema_roll_gain_min
                and np.isfinite(ema_fast[idx])
                and np.isfinite(ema_fast_slope[idx])
                and close[idx] <= ema_fast[idx]
                and ema_fast_slope[idx] < 0
            ):
                return idx, "ema_roll"
        return min(entry_idx + max_hold_bars, n - 1), "ema_roll_timeout"

    raise ValueError(f"Unknown long exit mode: {exit_mode}")


def _exit_short(
    arrays: dict[str, np.ndarray],
    *,
    entry_idx: int,
    exit_mode: str,
    rebound_exit_pct: float,
    vwap_reclaim_buffer: float,
    max_hold_bars: int,
) -> tuple[int, str]:
    close = arrays["close"]
    close_from_vwap = arrays["close_from_vwap_pct"]
    ema_fast = arrays["ema_fast"]
    ema_slow = arrays["ema_slow"]
    ema_fast_slope = arrays["ema_fast_slope"]
    n = len(close)

    if exit_mode.startswith("time_"):
        hold_bars = int(exit_mode.split("_", 1)[1])
        return min(entry_idx + hold_bars, n - 1), f"time_{hold_bars}"

    if exit_mode == "rebound":
        post_low = float(close[entry_idx])
        for idx in range(entry_idx + 1, min(entry_idx + max_hold_bars + 1, n)):
            if np.isfinite(close[idx]):
                post_low = min(post_low, float(close[idx]))
                rebound = _safe_pct(float(close[idx]), post_low)
                if np.isfinite(rebound) and rebound >= rebound_exit_pct:
                    return idx, "rebound"
        return min(entry_idx + max_hold_bars, n - 1), "rebound_timeout"

    if exit_mode == "vwap_reclaim":
        for idx in range(entry_idx + 1, min(entry_idx + max_hold_bars + 1, n)):
            if np.isfinite(close_from_vwap[idx]) and close_from_vwap[idx] >= vwap_reclaim_buffer:
                return idx, "vwap_reclaim"
        return min(entry_idx + max_hold_bars, n - 1), "vwap_timeout"

    if exit_mode == "ema_turn":
        post_low = float(close[entry_idx])
        for idx in range(entry_idx + 1, min(entry_idx + max_hold_bars + 1, n)):
            if not np.isfinite(close[idx]):
                continue
            post_low = min(post_low, float(close[idx]))
            rebound = _safe_pct(float(close[idx]), post_low)
            if (
                np.isfinite(rebound)
                and rebound >= rebound_exit_pct
                and np.isfinite(ema_fast[idx])
                and np.isfinite(ema_fast_slope[idx])
                and close[idx] >= ema_fast[idx]
                and ema_fast_slope[idx] > 0
            ):
                return idx, "ema_turn"
        return min(entry_idx + max_hold_bars, n - 1), "ema_turn_timeout"

    if exit_mode == "ema_cross":
        post_low = float(close[entry_idx])
        for idx in range(entry_idx + 1, min(entry_idx + max_hold_bars + 1, n)):
            if not np.isfinite(close[idx]):
                continue
            post_low = min(post_low, float(close[idx]))
            rebound = _safe_pct(float(close[idx]), post_low)
            if (
                np.isfinite(rebound)
                and rebound >= rebound_exit_pct
                and np.isfinite(ema_fast[idx])
                and np.isfinite(ema_slow[idx])
                and np.isfinite(ema_fast_slope[idx])
                and ema_fast[idx] >= ema_slow[idx]
                and ema_fast_slope[idx] > 0
            ):
                return idx, "ema_cross"
        return min(entry_idx + max_hold_bars, n - 1), "ema_cross_timeout"

    raise ValueError(f"Unknown short exit mode: {exit_mode}")


@register_strategy
class EarningsNegativeHybridStrategy(BaseStrategy):
    strategy_id = "earnings_negative_hybrid"
    name = "Earnings Negative Hybrid (Research)"
    description = (
        "Research-only stock earnings strategy. It auto-uses the local earnings-event table, "
        "then routes negative off-hours reaction days into continuation-short, rebound-long, "
        "or deep-panic rebound-plus-failure branches."
    )
    ui_hidden = True

    def default_params(self) -> dict[str, Any]:
        return {
            "continuation_selector_enabled": True,
            "continuation_selector_gap_pct_max": -4.0,
            "continuation_selector_trough_vs_prev_close_max": -15.0,
            "continuation_selector_confirm5_max": 0.0,
            "continuation_selector_confirm15_max": 0.0,
            "continuation_selector_confirm30_max": None,
            "continuation_entry_minute": 10 * 60,
            "continuation_stop_mode": "pct",
            "continuation_stop_pct": 5.0,
            "downside_prev_close_min": 12.5,
            "start_minute": 12 * 60,
            "max_minute": 15 * 60,
            "rebound_from_trough_min": 4.0,
            "vwap_reclaim_min": -0.5,
            "rebound_impulse_min": 0.5,
            "entry_min_close_from_open_pct": 0.0,
            "exit_vwap_touch_buffer": 1.0,
            "selector_two_leg_trough_threshold": 20.0,
            "selector_two_leg_confirm15_min": 0.0,
            "selector_two_leg_confirm30_min": 0.0,
            "two_leg_long_exit_mode": "vwap_touch",
            "two_leg_long_vwap_touch_buffer": 1.0,
            "two_leg_long_ema_roll_gain_min": 1.0,
            "two_leg_long_max_hold_bars": 20,
            "two_leg_short_entry_window_bars": 8,
            "two_leg_short_pullback_from_peak_min": 1.0,
            "two_leg_short_vwap_break_max": -0.5,
            "two_leg_short_downside_impulse_min": 1.0,
            "two_leg_short_exit_mode": "time_10",
            "two_leg_short_rebound_exit_pct": 0.5,
            "two_leg_short_vwap_reclaim_buffer": 0.0,
            "two_leg_short_max_hold_bars": 15,
        }

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Signal:
        actions, meta = self.generate_signals_bulk(data, symbol)
        if not actions or not meta:
            return Signal(strategy_id=self.strategy_id, symbol=symbol, action=SignalAction.HOLD)
        last_action = actions[-1]
        last_meta = meta[-1] or {}
        return Signal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            action=last_action if isinstance(last_action, SignalAction) else SignalAction(str(last_action)),
            suggested_tp=last_meta.get("suggested_tp"),
            suggested_sl=last_meta.get("suggested_sl"),
            metadata=dict(last_meta.get("metadata") or {}),
        )

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

    @staticmethod
    def _short_stop(arrays: dict[str, np.ndarray], entry_idx: int, long_start_idx: int) -> float:
        entry_px = float(arrays["close"][entry_idx])
        rebound_peak = float(np.nanmax(arrays["high"][long_start_idx: entry_idx + 1]))
        return max(rebound_peak * 1.006, entry_px * 1.015)

    @staticmethod
    def _continuation_short_stop(arrays: dict[str, np.ndarray], entry_idx: int, params: dict[str, Any]) -> float | None:
        mode = str(params.get("continuation_stop_mode", "pct")).lower()
        stop_pct = float(params.get("continuation_stop_pct", 4.0))
        entry_px = float(arrays["close"][entry_idx])
        if mode == "none":
            return None
        if mode == "pct":
            return entry_px * (1.0 + stop_pct / 100.0)
        local_high = float(np.nanmax(arrays["high"][: entry_idx + 1]))
        return max(local_high * 1.006, entry_px * 1.015)

    @staticmethod
    def _continuation_selector_match(event: dict[str, Any], params: dict[str, Any]) -> bool:
        if not bool(params.get("continuation_selector_enabled", True)):
            return False

        gap_pct = float(event.get("gap_pct", np.nan))
        trough = float(event.get("trough_vs_prev_close_pct", np.nan))
        confirm5 = float(event.get("confirm5_close_from_open_pct", np.nan))
        confirm15 = float(event.get("confirm15_close_from_open_pct", np.nan))
        confirm30 = float(event.get("confirm30_close_from_open_pct", np.nan))

        gap_max = float(params.get("continuation_selector_gap_pct_max", -4.0))
        trough_max = float(params.get("continuation_selector_trough_vs_prev_close_max", -12.0))
        confirm5_max = float(params.get("continuation_selector_confirm5_max", 0.0))
        confirm15_max = params.get("continuation_selector_confirm15_max", 0.0)
        confirm30_max = params.get("continuation_selector_confirm30_max")

        if (not np.isfinite(gap_pct)) or gap_pct > gap_max:
            return False
        if (not np.isfinite(trough)) or trough > trough_max:
            return False
        if (not np.isfinite(confirm5)) or confirm5 > confirm5_max:
            return False
        if confirm15_max is not None:
            if (not np.isfinite(confirm15)) or confirm15 > float(confirm15_max):
                return False
        if confirm30_max is not None:
            if (not np.isfinite(confirm30)) or confirm30 > float(confirm30_max):
                return False
        return True

    def generate_signals_bulk(self, data: pd.DataFrame, symbol: str) -> tuple[list, list]:
        n = len(data)
        actions, meta = self._empty_meta(n)
        if data.empty or "date" not in data.columns:
            return actions, meta

        params = self.resolve_params(symbol=symbol)
        symbol_events = merged_events_for_symbol(symbol, sign="negative")
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

            if self._continuation_selector_match(event, params):
                short_entry_idx = _find_continuation_short_entry_idx(
                    arrays,
                    entry_minute=int(params.get("continuation_entry_minute", RTH_OPEN + 5)),
                )
                if short_entry_idx is None:
                    continue
                g_short_entry = int(arrays["orig_index"][short_entry_idx])
                actions[g_short_entry] = SignalAction.SELL
                meta[g_short_entry] = {
                    "suggested_tp": None,
                    "suggested_sl": self._continuation_short_stop(arrays, short_entry_idx, params),
                    "metadata": {
                        "regime": "earnings_negative_continuation_short",
                        "session_exit": "eod",
                        "earnings_negative_hybrid": True,
                        "earnings_branch": "continuation_short",
                        "event_reaction_date": reaction_date,
                        "verdict_reason": (
                            "Negative off-hours earnings kept pressing lower after the open, "
                            "so the strategy chose the continuation-short branch instead of the rebound family."
                        ),
                    },
                }
                g_short_exit = int(arrays["orig_index"][len(arrays["close"]) - 1])
                if g_short_exit > g_short_entry:
                    actions[g_short_exit] = SignalAction.BUY
                    meta[g_short_exit] = {
                        "suggested_tp": None,
                        "suggested_sl": None,
                        "metadata": {
                            "regime": "earnings_negative_continuation_cover",
                            "cover_only": True,
                            "earnings_negative_hybrid": True,
                            "earnings_branch": "continuation_short",
                            "exit_reason": "session_close",
                        },
                    }
                continue

            long_entry_idx = _find_long_rebound_entry_idx(
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
            if long_entry_idx is None:
                continue

            trough = float(event.get("trough_vs_prev_close_pct", np.nan))
            confirm15 = float(event.get("confirm15_close_from_open_pct", np.nan))
            confirm30 = float(event.get("confirm30_close_from_open_pct", np.nan))
            use_two_leg = (
                np.isfinite(trough)
                and np.isfinite(confirm15)
                and np.isfinite(confirm30)
                and trough <= -float(params.get("selector_two_leg_trough_threshold", 20.0))
                and confirm15 >= float(params.get("selector_two_leg_confirm15_min", 0.0))
                and confirm30 >= float(params.get("selector_two_leg_confirm30_min", 0.0))
            )

            g_long_entry = int(arrays["orig_index"][long_entry_idx])

            actions[g_long_entry] = SignalAction.BUY
            meta[g_long_entry] = {
                "suggested_tp": None,
                "suggested_sl": self._long_stop(arrays, long_entry_idx),
                "metadata": {
                    "regime": "earnings_negative_rebound_long",
                    "session_exit": "eod",
                    "earnings_negative_hybrid": True,
                    "earnings_branch": "rebound_long",
                    "event_reaction_date": reaction_date,
                    "verdict_reason": (
                        "Negative off-hours earnings overshot deeply below the prior close, "
                        "then confirmed an intraday rebound back toward the fair zone."
                    ),
                },
            }

            if not use_two_leg:
                close_from_vwap = arrays["close_from_vwap_pct"]
                long_exit_idx = None
                for idx in range(long_entry_idx + 1, len(close_from_vwap)):
                    if np.isfinite(close_from_vwap[idx]) and close_from_vwap[idx] >= float(
                        params.get("exit_vwap_touch_buffer", 1.0)
                    ):
                        long_exit_idx = int(idx)
                        break
                if long_exit_idx is None:
                    long_exit_idx = len(close_from_vwap) - 1
                g_long_exit = int(arrays["orig_index"][long_exit_idx])
                if g_long_exit > g_long_entry:
                    actions[g_long_exit] = SignalAction.SELL
                    meta[g_long_exit] = {
                        "suggested_tp": None,
                        "suggested_sl": None,
                        "metadata": {
                            "regime": "earnings_negative_rebound_exit",
                            "cover_only": True,
                            "earnings_negative_hybrid": True,
                            "earnings_branch": "rebound_long",
                            "exit_reason": "vwap_touch",
                        },
                    }
                continue

            long_exit_idx, long_exit_reason = _exit_long(
                arrays,
                entry_idx=long_entry_idx,
                exit_mode=str(params.get("two_leg_long_exit_mode", "vwap_touch")),
                vwap_touch_buffer=float(params.get("two_leg_long_vwap_touch_buffer", 1.0)),
                ema_roll_gain_min=float(params.get("two_leg_long_ema_roll_gain_min", 1.0)),
                max_hold_bars=int(params.get("two_leg_long_max_hold_bars", 20)),
            )
            g_long_exit = int(arrays["orig_index"][long_exit_idx])

            short_entry_idx, short_entry_reason = _find_short_after_rebound_idx(
                arrays,
                start_idx=long_exit_idx,
                max_entry_bars=int(params.get("two_leg_short_entry_window_bars", 8)),
                pullback_from_peak_min=float(params.get("two_leg_short_pullback_from_peak_min", 1.0)),
                vwap_break_max=float(params.get("two_leg_short_vwap_break_max", -0.5)),
                downside_impulse_min=float(params.get("two_leg_short_downside_impulse_min", 1.0)),
            )

            if short_entry_idx is None:
                if g_long_exit > g_long_entry:
                    actions[g_long_exit] = SignalAction.SELL
                    meta[g_long_exit] = {
                        "suggested_tp": None,
                        "suggested_sl": None,
                        "metadata": {
                            "regime": "earnings_negative_rebound_exit",
                            "cover_only": True,
                            "earnings_negative_hybrid": True,
                            "earnings_branch": "deep_rebound_long",
                            "exit_reason": long_exit_reason,
                        },
                    }
                continue

            short_exit_idx, short_exit_reason = _exit_short(
                arrays,
                entry_idx=short_entry_idx,
                exit_mode=str(params.get("two_leg_short_exit_mode", "time_10")),
                rebound_exit_pct=float(params.get("two_leg_short_rebound_exit_pct", 0.5)),
                vwap_reclaim_buffer=float(params.get("two_leg_short_vwap_reclaim_buffer", 0.0)),
                max_hold_bars=int(params.get("two_leg_short_max_hold_bars", 15)),
            )

            g_short_entry = int(arrays["orig_index"][short_entry_idx])
            g_short_exit = int(arrays["orig_index"][short_exit_idx])

            if g_long_exit == g_short_entry:
                actions[g_long_exit] = SignalAction.SELL
                meta[g_long_exit] = {
                    "suggested_tp": None,
                    "suggested_sl": self._short_stop(arrays, short_entry_idx, long_entry_idx),
                    "metadata": {
                        "regime": "earnings_negative_short_after_rebound",
                        "earnings_negative_hybrid": True,
                        "earnings_branch": "rebound_then_fail",
                        "long_exit_reason": long_exit_reason,
                        "short_entry_reason": short_entry_reason,
                        "verdict_reason": (
                            "The rebound came off a very deep panic and then failed, so the strategy flipped "
                            "into a second-leg short on the renewed breakdown."
                        ),
                    },
                }
            else:
                actions[g_long_exit] = SignalAction.SELL
                meta[g_long_exit] = {
                    "suggested_tp": None,
                    "suggested_sl": None,
                    "metadata": {
                        "regime": "earnings_negative_rebound_exit",
                        "cover_only": True,
                        "earnings_negative_hybrid": True,
                        "earnings_branch": "deep_rebound_long",
                        "exit_reason": long_exit_reason,
                    },
                }
                actions[g_short_entry] = SignalAction.SELL
                meta[g_short_entry] = {
                    "suggested_tp": None,
                    "suggested_sl": self._short_stop(arrays, short_entry_idx, long_entry_idx),
                    "metadata": {
                        "regime": "earnings_negative_short_after_rebound",
                        "earnings_negative_hybrid": True,
                        "earnings_branch": "rebound_then_fail",
                        "short_entry_reason": short_entry_reason,
                        "verdict_reason": (
                            "After a very deep negative earnings panic rebounded and then failed, "
                            "the strategy entered a second-leg short on the renewed breakdown."
                        ),
                    },
                }

            if g_short_exit > g_short_entry:
                actions[g_short_exit] = SignalAction.BUY
                meta[g_short_exit] = {
                    "suggested_tp": None,
                    "suggested_sl": None,
                    "metadata": {
                        "regime": "earnings_negative_short_cover",
                        "cover_only": True,
                        "earnings_negative_hybrid": True,
                        "earnings_branch": "rebound_then_fail",
                        "exit_reason": short_exit_reason,
                    },
                }

        return actions, meta
