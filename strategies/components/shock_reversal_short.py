from __future__ import annotations


def shock_reversal_short_ready(
    *,
    episode_phase: str,
    active_spike: bool,
    breakout_still_valid: bool,
    prev_rsi: float,
    recent_max_rsi: float,
    current_rsi: float,
    rsi_trigger: float,
    max_current_rsi: float,
    current_bar_pct: float,
    min_bar_drop_pct: float,
    drop_from_recent_high_pct: float,
    min_drop_from_high_pct: float,
    recent_upper_band_rejection: bool,
    price_below_fast_ema: bool,
) -> bool:
    """
    Rare blow-off-top reversal short.

    Unlike the slower pullback short, this is meant for one-bar or two-bar
    exhaustion breaks where price collapses immediately after an extreme RSI
    reading while the spike episode is still active.
    """
    return (
        active_spike
        and episode_phase == "spike"
        and not breakout_still_valid
        and recent_upper_band_rejection
        and price_below_fast_ema
        and prev_rsi >= rsi_trigger
        and recent_max_rsi >= rsi_trigger
        and current_rsi <= max_current_rsi
        and current_bar_pct <= -min_bar_drop_pct
        and drop_from_recent_high_pct >= min_drop_from_high_pct
    )
