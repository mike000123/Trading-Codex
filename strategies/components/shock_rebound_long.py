from __future__ import annotations


def shock_rebound_long_ready(
    *,
    episode_phase: str,
    active_spike: bool,
    allow_active_spike: bool,
    recent_min_rsi: float,
    prev_rsi: float,
    current_rsi: float,
    rsi_trigger: float,
    min_rsi_rebound_points: float,
    max_current_rsi: float,
    rebound_from_recent_low_pct: float,
    min_rebound_pct: float,
    current_bar_pct: float,
    min_bar_rise_pct: float,
    recent_lower_band_rejection: bool,
    price_above_fast_ema: bool,
    up_bar: bool,
    atr_pct: float,
    min_atr_pct: float,
) -> bool:
    """
    Rare oversold snapback long.

    This mirrors the short-side shock/pullback logic: it does not buy simply
    because RSI is oversold. We require evidence that price has already turned
    up from the low and is reclaiming strength.
    """
    phase_ok = episode_phase == "idle" or (allow_active_spike and active_spike and episode_phase == "spike")
    return (
        phase_ok
        and recent_min_rsi <= rsi_trigger
        and current_rsi >= prev_rsi
        and (current_rsi - recent_min_rsi) >= min_rsi_rebound_points
        and current_rsi <= max_current_rsi
        and rebound_from_recent_low_pct >= min_rebound_pct
        and current_bar_pct >= min_bar_rise_pct
        and recent_lower_band_rejection
        and price_above_fast_ema
        and up_bar
        and atr_pct >= min_atr_pct
    )
