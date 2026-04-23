"""VXZ preset for the Bollinger + RSI spike-aware strategy."""
from __future__ import annotations


# VXZ is a slower, smoother vol ETN than UVXY, so it needs gentler spike
# thresholds and broader event windows.
PRESET: dict[str, object] = {
    "min_atr_pct": 0.2,
    "spike_gap_pct": 5.0,
    "grad_spike_pct": 12.0,
    "rise_pct": 2.0,
    "spike_long_min_peak_pct": 6.0,
    "spike_long_min_atr_pct": 0.8,
    "spike_momentum_max": 3,
    "spike_momo_min_peak_pct": 1.0,
    "spike_momo_min_atr_pct": 0.2,
    "event_target_min_peak_pct": 40.0,
    "event_target_max_rise_bars": 30000,
}

