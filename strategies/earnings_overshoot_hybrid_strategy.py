"""
Research-only earnings overshoot hybrid.

This strategy is intentionally narrow:
  - It auto-activates only on locally known positive off-hours earnings
    reaction days from the research artifact table.
  - It routes each eligible event day into a small set of validated shapes:
      1. a weak-open short-only fade for fragile overshoots
      2. a base multi-swing wave branch for neutral overshoots
      3. a robust wave branch for strong-open overshoots
      4. a failed-reclaim short branch for smaller overshoot-dump shapes

The goal is to make the event-day research testable in the normal Backtester
before we decide how to expose earnings-date input or a live earnings feed.
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
        "minutes_raw": pd.to_numeric(work["minutes_raw"], errors="coerce").to_numpy(dtype=np.int32, copy=True),
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
    arrays["high_from_vwap_pct"] = (arrays["high"] / arrays["anchored_vwap"] - 1.0) * 100.0
    arrays["ret_5m_pct"] = pd.Series(arrays["close"]).pct_change(5).to_numpy(dtype=np.float64, copy=True) * 100.0
    arrays["running_peak_high"] = np.maximum.accumulate(arrays["high"])
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
        hits = np.flatnonzero(arrays["minutes_raw"] >= minute_target)
        if hits.size == 0:
            return float("nan")
        idx = int(hits[0])
        return _safe_pct(float(arrays["close"][idx]), session_open)

    enriched["confirm5_close_from_open_pct"] = _confirm_close_from_open(RTH_OPEN + 5)
    enriched["confirm15_close_from_open_pct"] = _confirm_close_from_open(RTH_OPEN + 15)
    enriched["confirm30_close_from_open_pct"] = _confirm_close_from_open(RTH_OPEN + 30)

    first30_mask = arrays["minutes_raw"] <= (RTH_OPEN + 30)
    if np.any(first30_mask):
        try:
            first30_low = float(np.nanmin(arrays["low"][first30_mask]))
        except Exception:
            first30_low = float("nan")
    else:
        first30_low = float("nan")
    enriched["first30_low_from_open_pct"] = _safe_pct(first30_low, session_open)

    try:
        peak_high = float(np.nanmax(arrays["high"]))
    except Exception:
        peak_high = float("nan")
    enriched["peak_vs_prev_close_pct"] = _safe_pct(peak_high, prev_close)
    return enriched


def _find_wave_short_entry_idx(
    event: dict[str, Any],
    arrays: dict[str, np.ndarray],
    *,
    overshoot_prev_close_min: float,
    start_minute: int,
    max_minute: int,
    peak_pullback_min: float,
    vwap_break_min: float,
    breakdown_impulse_min: float,
) -> int | None:
    prev_close = float(event.get("prev_close", np.nan))
    if not np.isfinite(prev_close) or prev_close <= 0.0:
        return None

    minutes = arrays["minutes_raw"]
    close = arrays["close"]
    close_from_vwap = arrays["close_from_vwap_pct"]
    ret_5m = arrays["ret_5m_pct"]
    running_peak = arrays["running_peak_high"]
    running_peak_vs_prev = (running_peak / prev_close - 1.0) * 100.0
    pullback_from_peak = (close / running_peak - 1.0) * 100.0

    cond = (
        (minutes >= start_minute)
        & (minutes <= max_minute)
        & np.isfinite(close)
        & np.isfinite(close_from_vwap)
        & np.isfinite(ret_5m)
        & np.isfinite(running_peak_vs_prev)
        & np.isfinite(pullback_from_peak)
        & (running_peak_vs_prev >= overshoot_prev_close_min)
        & (pullback_from_peak <= -peak_pullback_min)
        & (close_from_vwap <= -vwap_break_min)
        & (ret_5m <= -breakdown_impulse_min)
    )
    hits = np.flatnonzero(cond)
    if hits.size == 0:
        return None
    return int(hits[0])


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


def _find_long_entry_idx(
    arrays: dict[str, np.ndarray],
    *,
    start_idx: int,
    entry_mode: str,
    long_momentum_min: float,
    max_entry_bars: int,
    entry_min_close_from_open_pct: float | None = None,
) -> tuple[int | None, str | None]:
    close = arrays["close"]
    high = arrays["high"]
    open_px = arrays["open"]
    ema_fast = arrays["ema_fast"]
    ema_fast_slope = arrays["ema_fast_slope"]
    ret_5m = arrays["ret_5m_pct"]
    n = len(close)
    end_idx = min(start_idx + max_entry_bars, n - 1)

    if start_idx >= n - 1:
        return None, None

    if entry_mode == "next_bar":
        idx = start_idx + 1
        if idx <= end_idx and np.isfinite(close[idx]):
            if entry_min_close_from_open_pct is not None:
                if not np.isfinite(open_px[idx]):
                    return None, None
                close_from_open_pct = (close[idx] / open_px[idx] - 1.0) * 100.0
                if (not np.isfinite(close_from_open_pct)) or (
                    close_from_open_pct < float(entry_min_close_from_open_pct)
                ):
                    return None, None
            return idx, "next_bar"
        return None, None

    for idx in range(start_idx + 1, end_idx + 1):
        if not np.isfinite(close[idx]):
            continue
        momentum_ok = (not np.isfinite(ret_5m[idx])) or (ret_5m[idx] >= long_momentum_min)
        if not momentum_ok:
            continue
        if entry_min_close_from_open_pct is not None:
            if not np.isfinite(open_px[idx]):
                continue
            close_from_open_pct = (close[idx] / open_px[idx] - 1.0) * 100.0
            if (not np.isfinite(close_from_open_pct)) or (
                close_from_open_pct < float(entry_min_close_from_open_pct)
            ):
                continue
        if entry_mode == "ema_turn":
            if (
                np.isfinite(ema_fast[idx])
                and np.isfinite(ema_fast_slope[idx])
                and close[idx] >= ema_fast[idx]
                and ema_fast_slope[idx] > 0
            ):
                return idx, "ema_turn"
        elif entry_mode == "break_prev_high":
            lookback_start = max(start_idx + 1, idx - 3)
            prev_high = float(np.nanmax(high[lookback_start:idx])) if idx > lookback_start else float("nan")
            if np.isfinite(prev_high) and close[idx] > prev_high:
                return idx, "break_prev_high"
        else:
            raise ValueError(f"Unknown long entry mode: {entry_mode}")

    return None, None


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


def _find_continuation_long_entry_idx(
    arrays: dict[str, np.ndarray],
    *,
    entry_minute: int,
) -> int | None:
    minutes = arrays["minutes_raw"]
    close = arrays["close"]
    hits = np.flatnonzero((minutes >= int(entry_minute)) & np.isfinite(close))
    if hits.size == 0:
        return None
    idx = int(hits[0])
    if idx >= len(close) - 1:
        return None
    return idx


def _find_failed_reclaim_trigger(
    event: dict[str, Any],
    arrays: dict[str, np.ndarray],
    *,
    overshoot_prev_close_min: float,
    start_minute: int,
    max_minute: int,
    vwap_break_min: float,
    rebound_min: float,
    reclaim_window_bars: int,
    reclaim_vwap_buffer: float,
    lower_high_min: float,
    breakdown_impulse_min: float,
) -> int | None:
    prev_close = float(event.get("prev_close", np.nan))
    if not np.isfinite(prev_close) or prev_close <= 0.0:
        return None

    minutes = arrays["minutes_raw"]
    close = arrays["close"]
    high = arrays["high"]
    low = arrays["low"]
    close_from_vwap = arrays["close_from_vwap_pct"]
    high_from_vwap = (arrays["high"] / arrays["anchored_vwap"] - 1.0) * 100.0
    ret_5m = arrays["ret_5m_pct"]
    running_peak = arrays["running_peak_high"]
    running_peak_vs_prev = (running_peak / prev_close - 1.0) * 100.0

    candidate_mask = (
        (minutes >= start_minute)
        & (minutes <= max_minute)
        & np.isfinite(close)
        & np.isfinite(low)
        & np.isfinite(high)
        & np.isfinite(close_from_vwap)
        & np.isfinite(high_from_vwap)
        & np.isfinite(ret_5m)
        & np.isfinite(running_peak_vs_prev)
        & (running_peak_vs_prev >= overshoot_prev_close_min)
        & (close_from_vwap <= -vwap_break_min)
    )
    candidate_idxs = np.flatnonzero(candidate_mask)
    if candidate_idxs.size == 0:
        return None

    for break_idx in candidate_idxs:
        break_low = float(low[break_idx])
        break_close = float(close[break_idx])
        peak_ref = float(running_peak[break_idx])
        rebound_end = min(break_idx + reclaim_window_bars, len(close) - 1)
        if rebound_end <= break_idx:
            continue

        rebound_slice = slice(break_idx + 1, rebound_end + 1)
        rebound_high = high[rebound_slice]
        rebound_high_from_vwap = high_from_vwap[rebound_slice]
        if rebound_high.size == 0:
            continue

        max_rebound_high = float(np.nanmax(rebound_high))
        max_rebound_high_from_vwap = float(np.nanmax(rebound_high_from_vwap))
        rebound_size = _safe_pct(max_rebound_high, break_close)
        lower_high_from_peak = -_safe_pct(max_rebound_high, peak_ref)

        if not np.isfinite(rebound_size) or rebound_size < rebound_min:
            continue
        if not np.isfinite(max_rebound_high_from_vwap) or max_rebound_high_from_vwap > reclaim_vwap_buffer:
            continue
        if not np.isfinite(lower_high_from_peak) or lower_high_from_peak < lower_high_min:
            continue

        for confirm_idx in range(rebound_end + 1, len(close)):
            minute = int(minutes[confirm_idx])
            if minute > max_minute:
                break
            if not np.isfinite(close[confirm_idx]) or not np.isfinite(ret_5m[confirm_idx]):
                continue
            if close[confirm_idx] < break_low and ret_5m[confirm_idx] <= -breakdown_impulse_min:
                return int(confirm_idx)
    return None


@register_strategy
class EarningsOvershootHybridStrategy(BaseStrategy):
    strategy_id = "earnings_overshoot_hybrid"
    name = "Earnings Overshoot Hybrid (Research)"
    description = (
        "Research-only stock earnings strategy. It auto-uses the local earnings-event table, "
        "then routes each positive off-hours reaction day into continuation, weak-open fade, wave, "
        "robust wave, or failed-reclaim branches using broad overshoot and early-tape rules."
    )
    ui_hidden = True

    def default_params(self) -> dict[str, Any]:
        return {
            "selector_peak_vs_prev_close_threshold": 21.0,
            "selector_weak_confirm15_max": -0.5,
            "selector_weak_confirm30_max": -1.0,
            "selector_strong_confirm15_min": 1.0,
            "selector_strong_confirm30_min": 1.0,
            "continuation_selector_enabled": True,
            "continuation_selector_gap_pct_min": 4.0,
            "continuation_selector_peak_vs_prev_close_min": 18.0,
            "continuation_selector_peak_vs_prev_close_max": 28.0,
            "continuation_selector_confirm5_min": 0.0,
            "continuation_selector_confirm15_min": None,
            "continuation_selector_confirm30_min": None,
            "continuation_selector_first30_low_from_open_min": -1.0,
            "wave_best_short_exit_mode": "time_10",
            "wave_best_short_rebound_exit_pct": 0.5,
            "wave_best_short_vwap_reclaim_buffer": 0.0,
            "wave_best_short_max_hold_bars": 20,
            "wave_best_long_entry_mode": "ema_turn",
            "wave_best_long_momentum_min": 0.0,
            "wave_best_long_max_entry_bars": 10,
            "wave_best_long_entry_min_close_from_open_pct": None,
            "wave_best_long_exit_mode": "ema_roll",
            "wave_best_long_vwap_touch_buffer": -0.5,
            "wave_best_long_ema_roll_gain_min": 1.0,
            "wave_best_long_max_hold_bars": 20,
            "wave_robust_short_exit_mode": "rebound",
            "wave_robust_short_rebound_exit_pct": 0.5,
            "wave_robust_short_vwap_reclaim_buffer": 0.0,
            "wave_robust_short_max_hold_bars": 20,
            "wave_robust_long_entry_mode": "next_bar",
            "wave_robust_long_momentum_min": 0.0,
            "wave_robust_long_max_entry_bars": 10,
            "wave_robust_long_entry_min_close_from_open_pct": None,
            "wave_robust_long_exit_mode": "time_20",
            "wave_robust_long_vwap_touch_buffer": -0.5,
            "wave_robust_long_ema_roll_gain_min": 0.5,
            "wave_robust_long_max_hold_bars": 20,
            "short_only_short_exit_mode": "time_20",
            "short_only_short_rebound_exit_pct": 0.5,
            "short_only_short_vwap_reclaim_buffer": 0.0,
            "short_only_short_max_hold_bars": 20,
            "failed_reclaim_short_exit_mode": "close_only",
            "failed_reclaim_short_rebound_exit_pct": 0.5,
            "failed_reclaim_short_vwap_reclaim_buffer": 0.0,
            "failed_reclaim_short_max_hold_bars": 20,
            "continuation_enabled": False,
            "continuation_gap_pct_min": 4.0,
            "continuation_confirm5_min": 0.0,
            "continuation_confirm15_min": None,
            "continuation_confirm30_min": None,
            "continuation_first30_low_from_open_min": None,
            "continuation_entry_minute": RTH_OPEN + 30,
            "continuation_stop_mode": "pct",
            "continuation_stop_pct": 4.0,
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
    def _short_stop(arrays: dict[str, np.ndarray], entry_idx: int) -> float:
        entry_px = float(arrays["close"][entry_idx])
        peak = float(arrays["running_peak_high"][entry_idx])
        return max(peak * 1.006, entry_px * 1.015)

    @staticmethod
    def _long_stop(arrays: dict[str, np.ndarray], entry_idx: int, from_idx: int) -> float:
        entry_px = float(arrays["close"][entry_idx])
        local_low = float(np.nanmin(arrays["low"][from_idx:entry_idx + 1]))
        return min(local_low * 0.994, entry_px * 0.985)

    def _apply_wave_branch_variant(
        self,
        event: dict[str, Any],
        arrays: dict[str, np.ndarray],
        actions: list[SignalAction],
        meta: list[dict[str, Any]],
        *,
        selector_threshold: float,
        branch_label: str,
        regime_prefix: str,
        overshoot_prev_close_min: float,
        start_minute: int,
        max_minute: int,
        peak_pullback_min: float,
        vwap_break_min: float,
        breakdown_impulse_min: float,
        short_exit_mode: str,
        short_rebound_exit_pct: float,
        short_vwap_reclaim_buffer: float,
        short_max_hold_bars: int,
        long_enabled: bool,
        long_entry_mode: str = "ema_turn",
        long_momentum_min: float = 0.0,
        long_max_entry_bars: int = 10,
        long_entry_min_close_from_open_pct: float | None = None,
        long_exit_mode: str = "ema_roll",
        long_vwap_touch_buffer: float = -0.5,
        long_ema_roll_gain_min: float = 1.0,
        long_max_hold_bars: int = 20,
        branch_verdict: str = "",
    ) -> None:
        short_entry_idx = _find_wave_short_entry_idx(
            event,
            arrays,
            overshoot_prev_close_min=overshoot_prev_close_min,
            start_minute=start_minute,
            max_minute=max_minute,
            peak_pullback_min=peak_pullback_min,
            vwap_break_min=vwap_break_min,
            breakdown_impulse_min=breakdown_impulse_min,
        )
        if short_entry_idx is None:
            return

        short_exit_idx, short_exit_reason = _exit_short(
            arrays,
            entry_idx=short_entry_idx,
            exit_mode=short_exit_mode,
            rebound_exit_pct=short_rebound_exit_pct,
            vwap_reclaim_buffer=short_vwap_reclaim_buffer,
            max_hold_bars=short_max_hold_bars,
        )
        long_entry_idx = None
        long_entry_reason = None
        long_exit_idx = None
        long_exit_reason = None
        if long_enabled:
            long_entry_idx, long_entry_reason = _find_long_entry_idx(
                arrays,
                start_idx=short_exit_idx,
                entry_mode=long_entry_mode,
                long_momentum_min=long_momentum_min,
                max_entry_bars=long_max_entry_bars,
                entry_min_close_from_open_pct=long_entry_min_close_from_open_pct,
            )
            if long_entry_idx is not None:
                long_exit_idx, long_exit_reason = _exit_long(
                    arrays,
                    entry_idx=long_entry_idx,
                    exit_mode=long_exit_mode,
                    vwap_touch_buffer=long_vwap_touch_buffer,
                    ema_roll_gain_min=long_ema_roll_gain_min,
                    max_hold_bars=long_max_hold_bars,
                )

        g_short_entry = int(arrays["orig_index"][short_entry_idx])
        actions[g_short_entry] = SignalAction.SELL
        meta[g_short_entry] = {
            "suggested_tp": None,
            "suggested_sl": self._short_stop(arrays, short_entry_idx),
            "metadata": {
                "regime": f"{regime_prefix}_short",
                "session_exit": "eod",
                "earnings_overshoot_hybrid": True,
                "earnings_branch": branch_label,
                "event_reaction_date": str(event.get("reaction_date")),
                "selector_peak_vs_prev_close_threshold": float(selector_threshold),
                "verdict_reason": branch_verdict,
            },
        }

        if long_entry_idx is None:
            g_short_exit = int(arrays["orig_index"][short_exit_idx])
            actions[g_short_exit] = SignalAction.BUY
            meta[g_short_exit] = {
                "suggested_tp": None,
                "suggested_sl": None,
                "metadata": {
                    "regime": f"{regime_prefix}_short_cover",
                    "cover_only": True,
                    "earnings_overshoot_hybrid": True,
                    "earnings_branch": branch_label,
                    "short_exit_reason": short_exit_reason,
                },
            }
            return

        long_stop = self._long_stop(arrays, long_entry_idx, short_exit_idx)
        g_short_exit = int(arrays["orig_index"][short_exit_idx])
        g_long_entry = int(arrays["orig_index"][long_entry_idx])

        if short_exit_idx == long_entry_idx:
            actions[g_short_exit] = SignalAction.BUY
            meta[g_short_exit] = {
                "suggested_tp": None,
                "suggested_sl": long_stop,
                "metadata": {
                    "regime": f"{regime_prefix}_long",
                    "session_exit": "eod",
                    "earnings_overshoot_hybrid": True,
                    "earnings_branch": branch_label,
                    "short_exit_reason": short_exit_reason,
                    "long_entry_reason": long_entry_reason,
                    "verdict_reason": (
                        "Cover the first dump-leg short and reverse long once the rebound confirms."
                    ),
                },
            }
        else:
            actions[g_short_exit] = SignalAction.BUY
            meta[g_short_exit] = {
                "suggested_tp": None,
                "suggested_sl": None,
                "metadata": {
                    "regime": f"{regime_prefix}_short_cover",
                    "cover_only": True,
                    "earnings_overshoot_hybrid": True,
                    "earnings_branch": branch_label,
                    "short_exit_reason": short_exit_reason,
                },
            }
            actions[g_long_entry] = SignalAction.BUY
            meta[g_long_entry] = {
                "suggested_tp": None,
                "suggested_sl": long_stop,
                "metadata": {
                    "regime": f"{regime_prefix}_long",
                    "session_exit": "eod",
                    "earnings_overshoot_hybrid": True,
                    "earnings_branch": branch_label,
                    "long_entry_reason": long_entry_reason,
                    "verdict_reason": (
                        "After the first dump leg exhausted, enter the rebound leg and let the long exit structure manage it."
                    ),
                },
            }

        if long_exit_idx is not None:
            g_long_exit = int(arrays["orig_index"][long_exit_idx])
            actions[g_long_exit] = SignalAction.SELL
            meta[g_long_exit] = {
                "suggested_tp": None,
                "suggested_sl": None,
                "metadata": {
                    "regime": f"{regime_prefix}_long_exit",
                    "cover_only": True,
                    "earnings_overshoot_hybrid": True,
                    "earnings_branch": branch_label,
                    "long_exit_reason": long_exit_reason,
                },
            }

    def _apply_wave_best_branch(
        self,
        event: dict[str, Any],
        arrays: dict[str, np.ndarray],
        actions: list[SignalAction],
        meta: list[dict[str, Any]],
        *,
        selector_threshold: float,
        params: dict[str, Any] | None = None,
    ) -> None:
        params = params or {}
        self._apply_wave_branch_variant(
            event,
            arrays,
            actions,
            meta,
            selector_threshold=selector_threshold,
            branch_label="wave_best",
            regime_prefix="earnings_wave",
            overshoot_prev_close_min=20.0,
            start_minute=13 * 60,
            max_minute=14 * 60,
            peak_pullback_min=3.0,
            vwap_break_min=1.0,
            breakdown_impulse_min=0.5,
            short_exit_mode=str(params.get("wave_best_short_exit_mode", "time_10")),
            short_rebound_exit_pct=float(params.get("wave_best_short_rebound_exit_pct", 0.5)),
            short_vwap_reclaim_buffer=float(params.get("wave_best_short_vwap_reclaim_buffer", 0.0)),
            short_max_hold_bars=int(params.get("wave_best_short_max_hold_bars", 20)),
            long_enabled=True,
            long_entry_mode=str(params.get("wave_best_long_entry_mode", "ema_turn")),
            long_momentum_min=float(params.get("wave_best_long_momentum_min", 0.0)),
            long_max_entry_bars=int(params.get("wave_best_long_max_entry_bars", 10)),
            long_entry_min_close_from_open_pct=params.get("wave_best_long_entry_min_close_from_open_pct"),
            long_exit_mode=str(params.get("wave_best_long_exit_mode", "ema_roll")),
            long_vwap_touch_buffer=float(params.get("wave_best_long_vwap_touch_buffer", -0.5)),
            long_ema_roll_gain_min=float(params.get("wave_best_long_ema_roll_gain_min", 1.0)),
            long_max_hold_bars=int(params.get("wave_best_long_max_hold_bars", 20)),
            branch_verdict=(
                "Positive off-hours earnings overshoot exceeded the wave threshold, "
                "so the strategy entered the base first-dump wave branch."
            ),
        )

    def _apply_wave_robust_branch(
        self,
        event: dict[str, Any],
        arrays: dict[str, np.ndarray],
        actions: list[SignalAction],
        meta: list[dict[str, Any]],
        *,
        selector_threshold: float,
        params: dict[str, Any] | None = None,
    ) -> None:
        params = params or {}
        self._apply_wave_branch_variant(
            event,
            arrays,
            actions,
            meta,
            selector_threshold=selector_threshold,
            branch_label="wave_robust",
            regime_prefix="earnings_wave_robust",
            overshoot_prev_close_min=20.0,
            start_minute=13 * 60,
            max_minute=14 * 60 + 30,
            peak_pullback_min=3.0,
            vwap_break_min=0.5,
            breakdown_impulse_min=0.5,
            short_exit_mode=str(params.get("wave_robust_short_exit_mode", "rebound")),
            short_rebound_exit_pct=float(params.get("wave_robust_short_rebound_exit_pct", 0.5)),
            short_vwap_reclaim_buffer=float(params.get("wave_robust_short_vwap_reclaim_buffer", 0.0)),
            short_max_hold_bars=int(params.get("wave_robust_short_max_hold_bars", 20)),
            long_enabled=True,
            long_entry_mode=str(params.get("wave_robust_long_entry_mode", "next_bar")),
            long_momentum_min=float(params.get("wave_robust_long_momentum_min", 0.0)),
            long_max_entry_bars=int(params.get("wave_robust_long_max_entry_bars", 10)),
            long_entry_min_close_from_open_pct=params.get("wave_robust_long_entry_min_close_from_open_pct"),
            long_exit_mode=str(params.get("wave_robust_long_exit_mode", "time_20")),
            long_vwap_touch_buffer=float(params.get("wave_robust_long_vwap_touch_buffer", -0.5)),
            long_ema_roll_gain_min=float(params.get("wave_robust_long_ema_roll_gain_min", 0.5)),
            long_max_hold_bars=int(params.get("wave_robust_long_max_hold_bars", 20)),
            branch_verdict=(
                "Positive off-hours earnings overshoot stayed strong through the early tape, "
                "so the strategy used the more robust wave sequence."
            ),
        )

    def _apply_short_only_branch(
        self,
        event: dict[str, Any],
        arrays: dict[str, np.ndarray],
        actions: list[SignalAction],
        meta: list[dict[str, Any]],
        *,
        selector_threshold: float,
        params: dict[str, Any] | None = None,
    ) -> None:
        params = params or {}
        self._apply_wave_branch_variant(
            event,
            arrays,
            actions,
            meta,
            selector_threshold=selector_threshold,
            branch_label="short_only",
            regime_prefix="earnings_short_only",
            overshoot_prev_close_min=20.0,
            start_minute=13 * 60,
            max_minute=14 * 60,
            peak_pullback_min=3.0,
            vwap_break_min=1.0,
            breakdown_impulse_min=0.5,
            short_exit_mode=str(params.get("short_only_short_exit_mode", "time_20")),
            short_rebound_exit_pct=float(params.get("short_only_short_rebound_exit_pct", 0.5)),
            short_vwap_reclaim_buffer=float(params.get("short_only_short_vwap_reclaim_buffer", 0.0)),
            short_max_hold_bars=int(params.get("short_only_short_max_hold_bars", 20)),
            long_enabled=False,
            branch_verdict=(
                "Positive off-hours earnings overshoot was large, but the early tape was already weak, "
                "so the strategy kept the setup as a short-only dump fade."
            ),
        )

    def _apply_failed_reclaim_branch(
        self,
        event: dict[str, Any],
        arrays: dict[str, np.ndarray],
        actions: list[SignalAction],
        meta: list[dict[str, Any]],
        *,
        selector_threshold: float,
        params: dict[str, Any] | None = None,
    ) -> None:
        params = params or {}
        entry_idx = _find_failed_reclaim_trigger(
            event,
            arrays,
            overshoot_prev_close_min=15.0,
            start_minute=12 * 60,
            max_minute=14 * 60,
            vwap_break_min=0.0,
            rebound_min=0.25,
            reclaim_window_bars=10,
            reclaim_vwap_buffer=0.0,
            lower_high_min=3.0,
            breakdown_impulse_min=0.5,
        )
        if entry_idx is None:
            return

        exit_mode = str(params.get("failed_reclaim_short_exit_mode", "close_only"))
        exit_idx = None
        exit_reason = None
        if exit_mode != "close_only":
            exit_idx, exit_reason = _exit_short(
                arrays,
                entry_idx=entry_idx,
                exit_mode=exit_mode,
                rebound_exit_pct=float(params.get("failed_reclaim_short_rebound_exit_pct", 0.5)),
                vwap_reclaim_buffer=float(params.get("failed_reclaim_short_vwap_reclaim_buffer", 0.0)),
                max_hold_bars=int(params.get("failed_reclaim_short_max_hold_bars", 20)),
            )

        g_entry = int(arrays["orig_index"][entry_idx])
        actions[g_entry] = SignalAction.SELL
        meta[g_entry] = {
            "suggested_tp": None,
            "suggested_sl": self._short_stop(arrays, entry_idx),
            "metadata": {
                "regime": "earnings_failed_reclaim_short",
                "session_exit": "eod",
                "earnings_overshoot_hybrid": True,
                "earnings_branch": "failed_reclaim",
                "event_reaction_date": str(event.get("reaction_date")),
                "selector_peak_vs_prev_close_threshold": float(selector_threshold),
                "verdict_reason": (
                    "Positive off-hours earnings overshoot stayed below the wave threshold, "
                    "broke under anchored VWAP, rebounded weakly, failed to reclaim, and broke down again."
                ),
            },
        }

        if exit_idx is not None:
            g_exit = int(arrays["orig_index"][exit_idx])
            actions[g_exit] = SignalAction.BUY
            meta[g_exit] = {
                "suggested_tp": None,
                "suggested_sl": None,
                "metadata": {
                    "regime": "earnings_failed_reclaim_short_cover",
                    "cover_only": True,
                    "earnings_overshoot_hybrid": True,
                    "earnings_branch": "failed_reclaim",
                    "short_exit_reason": exit_reason,
                },
            }

    def _apply_continuation_branch(
        self,
        event: dict[str, Any],
        arrays: dict[str, np.ndarray],
        actions: list[SignalAction],
        meta: list[dict[str, Any]],
        *,
        selector_threshold: float,
        params: dict[str, Any] | None = None,
    ) -> None:
        params = params or {}
        gap_pct = float(event.get("gap_pct", np.nan))
        confirm5 = float(event.get("confirm5_close_from_open_pct", np.nan))
        confirm15 = float(event.get("confirm15_close_from_open_pct", np.nan))
        confirm30 = float(event.get("confirm30_close_from_open_pct", np.nan))
        first30_low_from_open = float(event.get("first30_low_from_open_pct", np.nan))

        gap_min = float(params.get("continuation_gap_pct_min", 4.0))
        confirm5_min = float(params.get("continuation_confirm5_min", 0.0))
        confirm15_min = params.get("continuation_confirm15_min")
        confirm30_min = params.get("continuation_confirm30_min")
        first30_low_min = params.get("continuation_first30_low_from_open_min")

        if (not np.isfinite(gap_pct)) or gap_pct < gap_min:
            return
        if (not np.isfinite(confirm5)) or confirm5 < confirm5_min:
            return
        if confirm15_min is not None:
            if (not np.isfinite(confirm15)) or confirm15 < float(confirm15_min):
                return
        if confirm30_min is not None:
            if (not np.isfinite(confirm30)) or confirm30 < float(confirm30_min):
                return
        if first30_low_min is not None:
            if (not np.isfinite(first30_low_from_open)) or first30_low_from_open < float(first30_low_min):
                return

        entry_idx = _find_continuation_long_entry_idx(
            arrays,
            entry_minute=int(params.get("continuation_entry_minute", RTH_OPEN + 5)),
        )
        if entry_idx is None:
            return

        g_entry = int(arrays["orig_index"][entry_idx])
        stop_mode = str(params.get("continuation_stop_mode", "pct")).lower()
        stop_pct = float(params.get("continuation_stop_pct", 4.0))
        if stop_mode == "none":
            suggested_sl = None
        elif stop_mode == "pct":
            suggested_sl = float(arrays["close"][entry_idx]) * (1.0 - stop_pct / 100.0)
        else:
            suggested_sl = self._long_stop(arrays, entry_idx, 0)
        actions[g_entry] = SignalAction.BUY
        meta[g_entry] = {
            "suggested_tp": None,
            "suggested_sl": suggested_sl,
            "metadata": {
                "regime": "earnings_continuation_long",
                "session_exit": "eod",
                "earnings_overshoot_hybrid": True,
                "earnings_branch": "continuation",
                "event_reaction_date": str(event.get("reaction_date")),
                "selector_peak_vs_prev_close_threshold": float(selector_threshold),
                "verdict_reason": (
                    "Positive off-hours earnings repriced the stock into a controlled follow-through shape, "
                    "so the strategy chose the continuation branch instead of the overshoot-dump branches."
                ),
            },
        }

    @staticmethod
    def _continuation_selector_match(event: dict[str, Any], params: dict[str, Any]) -> bool:
        if not bool(params.get("continuation_selector_enabled", True)):
            return False

        gap_pct = float(event.get("gap_pct", np.nan))
        peak_vs_prev_close = float(event.get("peak_vs_prev_close_pct", np.nan))
        confirm5 = float(event.get("confirm5_close_from_open_pct", np.nan))
        confirm15 = float(event.get("confirm15_close_from_open_pct", np.nan))
        confirm30 = float(event.get("confirm30_close_from_open_pct", np.nan))
        first30_low_from_open = float(event.get("first30_low_from_open_pct", np.nan))

        gap_min = float(params.get("continuation_selector_gap_pct_min", 4.0))
        peak_min = float(params.get("continuation_selector_peak_vs_prev_close_min", 18.0))
        peak_max = float(params.get("continuation_selector_peak_vs_prev_close_max", 28.0))
        confirm5_min = float(params.get("continuation_selector_confirm5_min", 0.0))
        confirm15_min = params.get("continuation_selector_confirm15_min")
        confirm30_min = params.get("continuation_selector_confirm30_min")
        first30_low_min = params.get("continuation_selector_first30_low_from_open_min")

        if (not np.isfinite(gap_pct)) or gap_pct < gap_min:
            return False
        if (not np.isfinite(peak_vs_prev_close)) or peak_vs_prev_close < peak_min or peak_vs_prev_close > peak_max:
            return False
        if (not np.isfinite(confirm5)) or confirm5 < confirm5_min:
            return False
        if confirm15_min is not None:
            if (not np.isfinite(confirm15)) or confirm15 < float(confirm15_min):
                return False
        if confirm30_min is not None:
            if (not np.isfinite(confirm30)) or confirm30 < float(confirm30_min):
                return False
        if first30_low_min is not None:
            if (not np.isfinite(first30_low_from_open)) or first30_low_from_open < float(first30_low_min):
                return False
        return True

    def generate_signals_bulk(self, data: pd.DataFrame, symbol: str) -> tuple[list, list]:
        n = len(data)
        actions, meta = self._empty_meta(n)
        if data.empty or "date" not in data.columns:
            return actions, meta

        params = self.resolve_params(symbol=symbol)
        selector_threshold = float(params.get("selector_peak_vs_prev_close_threshold", 21.0))
        weak_confirm15_max = float(params.get("selector_weak_confirm15_max", -0.5))
        weak_confirm30_max = float(params.get("selector_weak_confirm30_max", -1.0))
        strong_confirm15_min = float(params.get("selector_strong_confirm15_min", 1.0))
        strong_confirm30_min = float(params.get("selector_strong_confirm30_min", 1.0))

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
            session_orig_indices = pd.to_numeric(session["orig_index"], errors="coerce").dropna().astype(int).tolist()
            existing_action_count = sum(
                1 for idx in session_orig_indices if 0 <= idx < len(actions) and actions[idx] != SignalAction.HOLD
            )
            if self._continuation_selector_match(event, params):
                self._apply_continuation_branch(
                    event,
                    arrays,
                    actions,
                    meta,
                    selector_threshold=selector_threshold,
                    params={
                        "continuation_gap_pct_min": params.get("continuation_selector_gap_pct_min", 4.0),
                        "continuation_confirm5_min": params.get("continuation_selector_confirm5_min", 0.0),
                        "continuation_confirm15_min": params.get("continuation_selector_confirm15_min"),
                        "continuation_confirm30_min": params.get("continuation_selector_confirm30_min"),
                        "continuation_first30_low_from_open_min": params.get(
                            "continuation_selector_first30_low_from_open_min", -1.0
                        ),
                        "continuation_entry_minute": params.get("continuation_entry_minute", RTH_OPEN + 30),
                        "continuation_stop_mode": params.get("continuation_stop_mode", "pct"),
                        "continuation_stop_pct": params.get("continuation_stop_pct", 4.0),
                    },
                )
                continue
            peak_vs_prev_close = float(event.get("peak_vs_prev_close_pct", np.nan))
            confirm15 = float(event.get("confirm15_close_from_open_pct", np.nan))
            confirm30 = float(event.get("confirm30_close_from_open_pct", np.nan))

            if np.isfinite(peak_vs_prev_close) and peak_vs_prev_close >= selector_threshold:
                weak_open = (
                    np.isfinite(confirm15)
                    and np.isfinite(confirm30)
                    and confirm15 <= weak_confirm15_max
                    and confirm30 <= weak_confirm30_max
                )
                strong_open = (
                    np.isfinite(confirm15)
                    and np.isfinite(confirm30)
                    and confirm15 >= strong_confirm15_min
                    and confirm30 >= strong_confirm30_min
                )
                if weak_open:
                    self._apply_short_only_branch(
                        event,
                        arrays,
                        actions,
                        meta,
                        selector_threshold=selector_threshold,
                        params=params,
                    )
                elif strong_open:
                    self._apply_wave_robust_branch(
                        event,
                        arrays,
                        actions,
                        meta,
                        selector_threshold=selector_threshold,
                        params=params,
                    )
                else:
                    self._apply_wave_best_branch(
                        event,
                        arrays,
                        actions,
                        meta,
                        selector_threshold=selector_threshold,
                        params=params,
                    )
            else:
                self._apply_failed_reclaim_branch(
                    event,
                    arrays,
                    actions,
                    meta,
                    selector_threshold=selector_threshold,
                    params=params,
                )

            updated_action_count = sum(
                1 for idx in session_orig_indices if 0 <= idx < len(actions) and actions[idx] != SignalAction.HOLD
            )
            if bool(params.get("continuation_enabled", True)) and updated_action_count == existing_action_count:
                self._apply_continuation_branch(
                    event,
                    arrays,
                    actions,
                    meta,
                    selector_threshold=selector_threshold,
                    params=params,
                )

        return actions, meta
