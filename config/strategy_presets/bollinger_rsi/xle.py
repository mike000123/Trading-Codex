"""XLE preset for the Bollinger + RSI spike-aware strategy.

XLE earned promotion as an energy-equity sleeve because the current engine
found a much stronger fit in a sparse trend profile than in the noisier
commodity-style or volatility-style families. The kept profile:

  - stays long-only on the main trend leg
  - uses a faster structural trend than QQQ/SPY to re-engage sooner
  - keeps higher-price re-entry protection on
  - adds the calmer shock-rebound long module to monetize sharp washouts

Tuned on the local 1-minute Alpaca cache over
2024-04-04 -> 2026-04-23 with current-engine costs
(`spread 0.06% + slippage 0.02%`).
"""
from __future__ import annotations


PRESET: dict[str, object] = {
    "normal_long_enabled": False,
    "normal_short_enabled": False,

    "trend_bias_long_enabled": True,
    "trend_bias_fast_ema": 195,
    "trend_bias_slow_ema": 780,
    "trend_bias_lookback_bars": 60,
    "trend_bias_min_retrace_pct": 0.5,
    "trend_bias_min_momentum_120": 0.4,
    "trend_bias_min_atr_pct": 0.02,
    "trend_bias_min_rsi": 47.0,
    "trend_bias_max_rsi": 70.0,
    "trend_bias_trail_pct": 3.2,
    "trend_bias_sl_pct": 1.3,
    "trend_bias_cooldown": 120,
    "trend_bias_no_higher_reentry": True,
    "trend_context_score_enabled": False,

    "intraday_pullback_short_enabled": False,
    "shock_reversal_short_enabled": False,
    "cascade_breakdown_short_enabled": False,
    "macro_bear_continuation_short_enabled": False,
    "event_target_short_enabled": False,

    "shock_rebound_long_enabled": True,
    "rsi_flush_rebound_long_enabled": False,

    "spike_momentum_max": 0,
    "spike_long_max": 0,
    "spike_max_entries": 0,
    "psshort_max": 0,
    "decay_bounce_max": 0,
    "decay_max_entries": 0,
}
