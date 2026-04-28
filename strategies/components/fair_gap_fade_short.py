from __future__ import annotations

import math


def _is_finite(value: float) -> bool:
    """Reject NaN / inf so that comparisons don't silently fail closed."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def fair_gap_fade_short_ready(
    *,
    fair_gap_pct: float,
    fair_daily_rsi: float,
    gap_threshold_pct: float,
    daily_rsi_trigger: float,
    current_bar_pct: float,
    min_bar_drop_pct: float,
    price_below_fast_ema: bool,
    current_price: float,
    prev_price: float,
) -> bool:
    """
    GLD fair-gap fade short.

    Use the slow fair-value model as a setup detector rather than a broad
    regime gate:
    1. actual price is materially above fair value
    2. daily RSI is already overbought
    3. the minute bar starts rolling over locally

    This keeps fair value as context and still requires price action to
    confirm the fade.

    NaN handling: any missing fair-value input (gap or daily RSI) causes a
    fail-closed result. We guard explicitly so the caller can log a clear
    "fair_value_inputs_missing" verdict instead of silently dropping the
    signal.
    """
    if not (_is_finite(fair_gap_pct) and _is_finite(fair_daily_rsi)):
        return False
    if not (_is_finite(current_bar_pct) and _is_finite(current_price) and _is_finite(prev_price)):
        return False
    return (
        fair_gap_pct <= -abs(gap_threshold_pct)
        and fair_daily_rsi >= daily_rsi_trigger
        and current_bar_pct <= -abs(min_bar_drop_pct)
        and price_below_fast_ema
        and current_price < prev_price
    )


def fair_gap_fade_short_inputs_missing(
    *, fair_gap_pct: float, fair_daily_rsi: float
) -> bool:
    """Helper for the strategy loop to log a clear verdict when the
    fair-value inputs themselves are NaN, separate from the price-action
    veto. The strategy can record this on the signal so the UI shows
    'fair_value_inputs_missing' instead of silently dropping the trigger."""
    return not (_is_finite(fair_gap_pct) and _is_finite(fair_daily_rsi))
