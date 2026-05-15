"""VXX preset for the Bollinger + RSI spike-aware strategy."""
from __future__ import annotations


# VXX is liquid enough for 1-min testing, but smoother than UVXY. Keep the
# up-leg/event legs and avoid the UVXY-style multi-entry decay bounce shorts.
PRESET: dict[str, object] = {
    "decay_bounce_max": 0,
    "spike_momentum_max": 4,
    "spike_momo_min_peak_pct": 1.0,
    "spike_momo_min_atr_pct": 0.25,
    "spike_momo_trail_pct": 5.0,
    "spike_momo_cooldown": 195,
    "event_target_min_peak_pct": 40.0,
    "event_target_confirm_drop_pct": 10.0,
    "event_target_persistent_confirm_drop_pct": 15.0,
    "low_price_chop_price": 80.0,
    "low_price_chop_bandwidth_pct": 2.5,
}
