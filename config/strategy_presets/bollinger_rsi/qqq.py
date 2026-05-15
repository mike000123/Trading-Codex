"""QQQ preset for the Bollinger + RSI spike-aware strategy.

QQQ responded well to the same sparse trend-bias family we used for SPY, but
it needed a more selective continuation profile:

  - keep the strategy long-only and sparse
  - require stronger medium-horizon momentum before rejoining the trend
  - demand a slightly healthier RSI posture
  - give winners more room with a wider trend trail and stop
  - prevent higher-price re-entries from churning the trend leg

Second pass:
  - widening the trend trail from 4.0% to 4.8% improved the full-window fit
    on the same validation sample

Walk-forward check:
  - the broader out-of-sample fold check favored the original 4.0% trail over
    4.8%, so we keep the more robust setting instead of the prettier
    in-sample one

Tuned on the local 1-minute Alpaca cache over
2024-04-04 -> 2026-04-02 with current-engine costs
(`spread 0.06% + slippage 0.02%`).
"""
from __future__ import annotations


PRESET: dict[str, object] = {
    "normal_long_enabled": False,
    "normal_short_enabled": False,

    "trend_bias_long_enabled": True,
    "trend_bias_fast_ema": 156,
    "trend_bias_slow_ema": 975,
    "trend_bias_lookback_bars": 90,
    "trend_bias_min_retrace_pct": 0.5,
    "trend_bias_min_momentum_120": 0.7,
    "trend_bias_min_atr_pct": 0.02,
    "trend_bias_min_rsi": 49.0,
    "trend_bias_max_rsi": 72.0,
    "trend_bias_trail_pct": 4.0,
    "trend_bias_sl_pct": 1.3,
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
