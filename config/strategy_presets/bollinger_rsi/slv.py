"""SLV preset for the Bollinger + RSI spike-aware strategy.

SLV earned its own sleeve because it behaved much better as a sparse,
long-only trend participant than as a GLD clone or a generic mean-reversion
ticker. The current-engine search favored:

  - a permissive normal long so silver can re-engage after shallow pullbacks
  - a stronger trend-bias continuation leg with a slow structural EMA
  - all short-side spike / decay modules disabled

Tuned on the local 1-minute Alpaca cache over
2024-04-04 -> 2026-04-23 with current-engine costs
(`spread 0.06% + slippage 0.02%`).
"""
from __future__ import annotations


PRESET: dict[str, object] = {
    "normal_long_enabled": True,
    "normal_short_enabled": False,
    "trend_bias_long_enabled": True,
    "trend_bias_fast_ema": 156,
    "trend_bias_slow_ema": 975,
    "trend_bias_lookback_bars": 120,
    "trend_bias_min_retrace_pct": 0.6,
    "trend_bias_min_momentum_120": 0.7,
    "trend_bias_min_atr_pct": 0.04,
    "trend_bias_min_rsi": 51.0,
    "trend_bias_max_rsi": 72.0,
    "trend_bias_trail_pct": 4.8,
    "trend_bias_sl_pct": 1.5,
    "trend_bias_cooldown": 120,
    "trend_bias_no_higher_reentry": True,
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
