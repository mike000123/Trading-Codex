from __future__ import annotations


def spike_momentum_long_ready(
    *,
    active_now: bool,
    peak_excess_pct: float,
    min_peak_pct: float,
    rsi_value: float,
    min_rsi: float,
    atr_value: float,
    price: float,
    min_atr_pct: float,
) -> bool:
    if not active_now or price <= 0:
        return False
    atr_pct = (atr_value / max(price, 1e-9)) * 100
    return (
        peak_excess_pct >= min_peak_pct
        and rsi_value >= min_rsi
        and atr_pct >= min_atr_pct
    )


def spike_breakout_long_ready(
    *,
    active_now: bool,
    episode_phase: str,
    in_spike_lockout: bool,
    atr_value: float,
    atr_ma_value: float,
    atr_mult: float,
    momentum_pct: float,
    min_momentum_pct: float,
    peak_excess_pct: float,
    min_peak_pct: float,
    rsi_value: float,
    min_rsi: float,
    max_rsi: float,
    price: float,
    min_atr_pct: float,
) -> bool:
    if not active_now or episode_phase != "spike" or in_spike_lockout or price <= 0:
        return False
    atr_pct = (atr_value / max(price, 1e-9)) * 100
    return (
        atr_ma_value > 0
        and atr_value >= atr_ma_value * atr_mult
        and momentum_pct >= min_momentum_pct
        and peak_excess_pct >= min_peak_pct
        and min_rsi <= rsi_value <= max_rsi
        and atr_pct >= min_atr_pct
    )
