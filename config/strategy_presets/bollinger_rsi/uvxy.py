"""UVXY baseline preset for the Bollinger + RSI spike-aware strategy."""
from __future__ import annotations


# UVXY remains the baseline instrument, but some UVXY-only refinements are kept
# here so the core strategy defaults can stay more generic for reuse on other
# symbols.
PRESET: dict[str, object] = {
    "intraday_pullback_short_enabled": True,
    "intraday_pullback_rsi_trigger": 80.0,
    "intraday_pullback_rsi_fade_pts": 12.0,
    "intraday_pullback_lookback_bars": 60,
    "intraday_pullback_drop_pct": 2.0,
    "intraday_pullback_min_atr_pct": 0.4,
    "intraday_pullback_sl_pct": 2.0,
    "intraday_pullback_tp_pct": 3.0,
    "intraday_pullback_trail_pct": 1.6,
    "intraday_pullback_cooldown": 60,
}
