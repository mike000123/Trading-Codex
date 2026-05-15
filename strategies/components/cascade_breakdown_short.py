from __future__ import annotations


def cascade_breakdown_short_ready(
    *,
    episode_phase: str,
    active_spike: bool,
    watch_active: bool,
    rebound_seen: bool,
    breakout_still_valid: bool,
    peak_rsi: float,
    rebound_rsi: float,
    current_rsi: float,
    rsi_trigger: float,
    min_rebound_rsi_fade_pts: float,
    rebound_high: float,
    current_price: float,
    prev_price: float,
    breakdown_drop_pct: float,
) -> bool:
    """
    Confirm a second-leg breakdown after an overbought spike has already:
    1. broken lower once
    2. attempted a weaker rebound
    3. started failing again

    This targets cascading rollovers rather than one-bar shock reversals.
    """
    return (
        active_spike
        and episode_phase == "spike"
        and watch_active
        and rebound_seen
        and not breakout_still_valid
        and peak_rsi >= rsi_trigger
        and rebound_rsi > 0
        and rebound_rsi <= peak_rsi - min_rebound_rsi_fade_pts
        and current_rsi <= rebound_rsi
        and rebound_high > 0
        and current_price <= rebound_high * (1 - breakdown_drop_pct / 100)
        and current_price < prev_price
    )
