from __future__ import annotations


def intraday_pullback_short_ready(
    *,
    episode_phase: str,
    active_spike: bool,
    allow_active_spike: bool,
    drawdown_from_peak_pct: float,
    min_spike_drawdown_pct: float,
    breakout_still_valid: bool,
    recent_max_rsi: float,
    current_rsi: float,
    rsi_trigger: float,
    min_rsi_fade_points: float,
    drop_from_recent_high_pct: float,
    min_drop_pct: float,
    recent_upper_band_rejection: bool,
    price_below_fast_ema: bool,
    down_bar: bool,
    atr_pct: float,
    min_atr_pct: float,
) -> bool:
    """
    Confirm a short after an intraday overbought burst has already started to fade.

    This is intentionally a pullback/fade rule, not a naked "RSI > X" short.
    We want:
    - a recent overbought impulse
    - RSI to have started fading from that overbought impulse
    - measurable pullback from the recent high
    - evidence that price was recently stretched into the upper band
    - some downside confirmation on the current bar
    - enough volatility to justify the trade
    """
    phase_ok = (
        (episode_phase == "idle" and not active_spike)
        or (
            allow_active_spike
            and active_spike
            and drawdown_from_peak_pct >= min_spike_drawdown_pct
            and not breakout_still_valid
        )
    )
    return (
        phase_ok
        and recent_max_rsi >= rsi_trigger
        and (recent_max_rsi - current_rsi) >= min_rsi_fade_points
        and drop_from_recent_high_pct >= min_drop_pct
        and recent_upper_band_rejection
        and price_below_fast_ema
        and down_bar
        and atr_pct >= min_atr_pct
    )
