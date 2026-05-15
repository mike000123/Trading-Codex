"""APLD research-only preset for the Bollinger + RSI strategy.

This preset intentionally disables the broad Bollinger/RSI entry families and
turns on only the BTC-aware opening overlay we validated in the research
scripts. The idea is to keep APLD inside the existing strategy framework while
still respecting that its best edge so far looks like an opening dislocation
trade driven by overnight BTC context.

Current research state:
  - short overlay: BTC-up + relative gap-up + early failure, fixed 2.5% stop
  - long overlay: BTC-down + gap-down + early reclaim, fixed 3.0% stop
  - added weak-QQQ conditional filter: when early QQQ tape is negative, only
    allow the long if the rebound-from-trough strength is at least 1.5%
  - both close by session end if the stop is not hit first

This is intentionally a research preset rather than a fully promoted live
default. It is available in the normal backtest flow so we can validate it in
the same engine and UI as the rest of the strategy database.
"""
from __future__ import annotations


PRESET: dict[str, object] = {
    "normal_long_enabled": False,
    "normal_short_enabled": False,

    "trend_bias_long_enabled": False,
    "trend_context_score_enabled": False,

    "intraday_pullback_short_enabled": False,
    "shock_reversal_short_enabled": False,
    "cascade_breakdown_short_enabled": False,
    "macro_bear_continuation_short_enabled": False,
    "event_target_short_enabled": False,
    "fair_gap_fade_short_enabled": False,

    "shock_rebound_long_enabled": False,
    "rsi_flush_rebound_long_enabled": False,
    "rsi_spike_fade_short_enabled": False,

    "spike_momentum_max": 0,
    "spike_long_max": 0,
    "spike_max_entries": 0,
    "psshort_max": 0,
    "decay_bounce_max": 0,
    "decay_max_entries": 0,

    "apld_btc_overlay_enabled": True,
    "apld_btc_short_enabled": True,
    "apld_btc_long_enabled": True,

    "apld_btc_short_btc_threshold": 1.0,
    "apld_btc_short_gap_threshold": 0.5,
    "apld_btc_short_entry_offset_min": 15,
    "apld_btc_short_peak_threshold": 0.5,
    "apld_btc_short_pullback_threshold": 0.25,
    "apld_btc_short_confirm_close_max_pct": -0.25,
    "apld_btc_short_stop_loss_pct": 2.5,

    "apld_btc_long_btc_threshold": 1.0,
    "apld_btc_long_gap_threshold": 0.5,
    "apld_btc_long_entry_offset_min": 5,
    "apld_btc_long_trough_threshold": 0.5,
    "apld_btc_long_rebound_threshold": 0.25,
    "apld_btc_long_confirm_close_min_pct": -1.0,
    "apld_btc_long_stop_loss_pct": 3.0,
    "apld_btc_long_qqq_weak_filter_enabled": True,
    "apld_btc_long_qqq_weak_close_max_pct": 0.0,
    "apld_btc_long_qqq_weak_rebound_min_pct": 1.5,
}
