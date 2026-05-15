"""SPY preset for the Bollinger + RSI spike-aware strategy.

SPY is far cleaner than the volatility products and more directional than the
commodity/metal ETFs we tuned earlier. On the current engine, the best SPY
behavior came from staying sparse:

  - disable the generic mean-reversion entries
  - disable the short-side reversal stack
  - let a single long trend-bias leg do the work

The trend-bias thresholds below were tuned on the local 1-minute cache over
2024-04-04 -> 2026-04-03 with current-engine costs
(`spread 0.06% + slippage 0.02%`).
"""
from __future__ import annotations


PRESET: dict[str, object] = {
    "normal_long_enabled": False,
    "normal_short_enabled": False,

    "trend_bias_long_enabled": True,
    "trend_bias_fast_ema": 156,
    "trend_bias_slow_ema": 780,
    "trend_bias_lookback_bars": 90,
    "trend_bias_min_retrace_pct": 0.5,
    "trend_bias_min_momentum_120": 0.5,
    "trend_bias_min_atr_pct": 0.02,
    "trend_bias_min_rsi": 47.0,
    "trend_bias_max_rsi": 72.0,
    "trend_bias_trail_pct": 3.2,
    "trend_bias_sl_pct": 0.9,
    "trend_bias_cooldown": 120,
    "trend_bias_no_higher_reentry": False,
    "trend_context_score_enabled": False,

    "intraday_pullback_short_enabled": False,
    "shock_reversal_short_enabled": False,
    "cascade_breakdown_short_enabled": False,
    "macro_bear_continuation_short_enabled": False,
    "event_target_short_enabled": False,

    "shock_rebound_long_enabled": False,
    "rsi_flush_rebound_long_enabled": False,

    "spike_momentum_max": 0,
    "spike_long_max": 0,
    "spike_max_entries": 0,
    "psshort_max": 0,
    "decay_bounce_max": 0,
    "decay_max_entries": 0,
}
