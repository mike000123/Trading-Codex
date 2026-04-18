from __future__ import annotations


def trend_bias_long_ready(
    *,
    trend_up: bool,
    context_ok: bool,
    atr_pct: float,
    min_atr_pct: float,
    rsi_value: float,
    min_rsi: float,
    max_rsi: float,
    retrace_from_recent_high_pct: float,
    min_retrace_pct: float,
    reclaim_fast_ema: bool,
    prior_near_fast_ema: bool,
    up_bar: bool,
) -> bool:
    """
    Confirm a long-bias trend-continuation entry after a controlled pullback.

    The pattern we want is:
    - broader trend already up
    - macro/peer context not hostile
    - the instrument pulled back some measurable amount from a recent high
    - price is now reclaiming the fast trend EMA
    - the current bar is constructive rather than still fading
    """
    return (
        trend_up
        and context_ok
        and atr_pct >= min_atr_pct
        and min_rsi <= rsi_value <= max_rsi
        and retrace_from_recent_high_pct >= min_retrace_pct
        and reclaim_fast_ema
        and prior_near_fast_ema
        and up_bar
    )
