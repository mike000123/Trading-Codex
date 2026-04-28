from __future__ import annotations

import math


def _is_finite(value: float) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def macro_bear_continuation_short_ready(
    *,
    watch_active: bool,
    rebound_seen: bool,
    macro_bear_permission: bool,
    trend_bearish: bool,
    price_below_slow_ema: bool,
    break_low: float,
    rebound_rsi: float,
    current_rsi: float,
    min_current_rsi: float,
    max_current_rsi: float,
    min_rebound_rsi_fade_pts: float,
    current_price: float,
    prev_price: float,
    rebreak_pct: float,
) -> bool:
    """
    Confirm a bearish-macro continuation short after:
    1. macro backdrop has already turned bearish
    2. price breaks lower inside a downtrend
    3. a rebound fails beneath slow-trend resistance
    4. price breaks back through the first breakdown low

    This is a continuation pattern, not a top-picking reversal.

    NaN-safe: any non-finite numeric input (RSI, prices, break_low, etc.)
    fails closed rather than silently passing-through Python's NaN
    comparison semantics.
    """
    numeric_inputs = (
        break_low,
        rebound_rsi,
        current_rsi,
        min_current_rsi,
        max_current_rsi,
        min_rebound_rsi_fade_pts,
        current_price,
        prev_price,
        rebreak_pct,
    )
    if not all(_is_finite(v) for v in numeric_inputs):
        return False
    return (
        macro_bear_permission
        and watch_active
        and rebound_seen
        and trend_bearish
        and price_below_slow_ema
        and break_low > 0.0
        and rebound_rsi > 0.0
        and current_rsi >= min_current_rsi
        and current_rsi <= max_current_rsi
        and current_rsi <= rebound_rsi - min_rebound_rsi_fade_pts
        and current_price <= break_low * (1 - rebreak_pct / 100.0)
        and current_price < prev_price
    )
