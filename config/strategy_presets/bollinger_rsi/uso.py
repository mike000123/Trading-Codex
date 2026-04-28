"""USO preset for the Bollinger + RSI spike-aware strategy.

Crude-oil ETF (USO) preset. USO is the most liquid pure-crude ETF, has
consistent shortability on Alpaca, and behaves intraday like a higher-vol
spike-and-fade instrument:

  • Per-bar ATR roughly 0.10-0.15% vs ~0.04% for GLD.
  • Daily ATR roughly 1.5-2.0% vs ~0.45% for GLD.
  • EIA petroleum status report (Wed 10:30 ET) routinely produces 1.5-3%
    intraday peak-to-trough moves. The strategy's existing shock_reversal,
    cascade_breakdown, and event_target modules are designed for exactly
    this shock-and-fade pattern — no separate news-window gate is added.

Front-month-roll decay is a multi-month phenomenon and does not affect
intraday trades that flatten within the session, so USO is fine despite
its long-term contango drag.

Tuning history
──────────────
v1 (initial port from gld.py): a 2023-04 → 2026-04 backtest produced
  -25.50% return / 64 trades / 40.6% win rate / -1.823 Sharpe. Per-regime
  breakdown showed spike_momentum_long alone was responsible for the
  entire drawdown (26 trades, 0% TP rate, -$470 PnL); shorts (shock /
  cascade / event_target) never triggered because thresholds were sized
  too high for crude's realised peak distribution; trend-following
  capacity was missing entirely, so the +73% Feb-Apr 2026 rally cost
  $107 in repeated spike_momentum trail-stops with no offsetting trend
  capture, and the symmetric $140 → $120 fade went unshorted.

v2 (this revision): six targeted changes, USO-only.
  1. spike_momentum_long disabled (spike_momentum_max=0). UVXY-flavoured
     "rolling spike continuation" doesn't translate to crude.
  2. Short-side triggers lowered so the modules actually fire on
     crude-sized peaks (event_target peak 12→7%, confirm 6→3%; shock
     reversal RSI 82→78, drop 2.0→1.2%; cascade RSI 82→78, initial
     drawdown 2.0→1.2%; intraday pullback drop 4→2.5%).
  3. shock_rebound_long made more selective at entry and looser on the
     trail (RSI trigger 25→20, rebound 1.5→2.0%, trail 2.2→3.0%) so the
     21-of-27 trail-stop ratio drops.
  4. rsi_flush_rebound_long left untouched — the only v1 module already
     pulling its weight (87.5% TP rate).
  5. trend_bias_long enabled to ride sustained multi-week rallies the
     spike-only logic systematically misses.
  6. event_target_anchor_lookback extended to 11700 bars (~40 trading
     days) so 6+ week rallies still anchor to their actual base when
     the peak rolls over.


v3 (this revision): targets the post-Dec 2025 damage uncovered by the v2
  re-run.  v2 cleared most of the v1 drawdown but still printed -7.74%
  because (a) trend_bias_long, after catching the meat of the Feb-Mar
  2026 rally with a +$107 winner, kept re-entering at progressively
  higher prices and got chopped 7 times for ~-$200, (b) shock_rebound
  bled -$52 catching falling knives during the Dec 2025-Feb 2026 slow
  decline, and (c) intraday_pullback_short faded mid-rally three times
  for -$34.

  Six knob changes plus one structural rule:
    • trend_bias_cooldown 120 -> 3900 (~3 days), eliminating chase-the-top
      re-entries within hours of the prior fill.
    • trend_bias_min_momentum_120 0.6 -> 1.5 and trend_bias_max_rsi
      74 -> 68 to filter weak / late entries.
    • intraday_pullback_drop_pct 2.5 -> 3.5 to stop firing on small
      pullbacks inside vertical advances.
    • shock_rebound_rebound_pct 2.0 -> 3.0 and max_current_rsi 65 -> 55
      so we only catch deep oversold rebounds, not slow drift bounces.
    • Structural opt-in gate (trend_bias_no_higher_reentry=True): once
      a trend_bias_long fills, future entries are blocked while price
      stays above that fill price, until the fast EMA crosses below the
      slow EMA at least once (a real trend break / reset).  Implemented
      in strategies/components/trend.py behind a knob so GLD and UVXY
      keep their existing behavior.

Capital allocation is configured per Paper-Trading run and is left to the
caller; this file only carries strategy-parameter overrides.
"""
from __future__ import annotations


PRESET: dict[str, object] = {
    # ── Mean-reversion bands: off by default. USO does mean-revert intraday,
    # but the spike-aware modules below capture those patterns more cleanly
    # than band-touch entries do. Enable via sweep if a sub-grid wins.
    "normal_long_enabled": False,
    "normal_short_enabled": False,

    # ── Trend-bias long: ON in v2. The Feb-Apr 2026 +73% rally exposed the
    # absence of any "ride a sustained trend" module — spike_momentum_long
    # repeatedly entered and trail-stopped on intraday wiggles instead. The
    # 5% trail / 3% SL on this module is wide enough to survive normal
    # intraday noise inside a multi-week up-trend.
    "trend_bias_long_enabled": True,
    "trend_bias_fast_ema": 156,
    "trend_bias_slow_ema": 780,
    "trend_bias_lookback_bars": 90,
    "trend_bias_min_retrace_pct": 1.5,
    "trend_bias_min_momentum_120": 1.5,    # v3: 0.6 -> 1.5; require stronger trend.
    "trend_bias_min_atr_pct": 0.10,
    "trend_bias_min_rsi": 46.0,
    "trend_bias_max_rsi": 68.0,            # v3: 74 -> 68; don't chase extended price.
    "trend_bias_trail_pct": 5.0,
    "trend_bias_sl_pct": 3.0,
    "trend_bias_cooldown": 3900,           # v3: 120 -> 3900 (~3 trading days).
    # v3 structural gate: USO-only "no re-entry above last entry until a fresh
    # trend break" rule. Implemented in strategies/components/trend.py and
    # gated by this knob so GLD/UVXY behavior is unaffected. A "trend break"
    # is fast_ema crossing below slow_ema since the last entry; once that
    # happens, the price-above-last-entry block resets.
    "trend_bias_no_higher_reentry": True,
    "trend_context_score_enabled": False,
    "trend_context_min_score": 1.1,
    "trend_peer_strength_weight": 0.0,
    "trend_miners_strength_weight": 0.0,
    "trend_riskoff_strength_weight": 0.0,

    # ── Intraday pullback short — drop threshold lowered (4.0→2.5) so the
    # module fires on routine post-spike fades, not just rare 4% drops.
    "intraday_pullback_short_enabled": True,
    "intraday_pullback_rsi_trigger": 80.0,
    "intraday_pullback_rsi_fade_pts": 8.0,
    "intraday_pullback_lookback_bars": 90,
    "intraday_pullback_drop_pct": 3.5,    # v3: 2.5 -> 3.5; less false fades inside rallies.
    "intraday_pullback_min_atr_pct": 0.10,
    "intraday_pullback_allow_active_spike": True,
    "intraday_pullback_spike_drawdown_pct": 8.0,
    "intraday_pullback_sl_pct": 2.0,
    "intraday_pullback_tp_pct": 3.0,
    "intraday_pullback_trail_pct": 1.6,
    "intraday_pullback_cooldown": 180,

    # ── Shock-reversal short. RSI trigger and per-bar drop relaxed so EIA-
    # print fades trigger more reliably (v1: 0 fires across 3 years).
    "shock_reversal_short_enabled": True,
    "shock_reversal_rsi_trigger": 78.0,
    "shock_reversal_max_current_rsi": 70.0,
    "shock_reversal_bar_drop_pct": 1.0,
    "shock_reversal_drop_pct": 1.2,
    "shock_reversal_sl_pct": 1.8,
    "shock_reversal_tp_pct": 2.5,
    "shock_reversal_trail_pct": 1.4,
    "shock_reversal_cooldown": 180,

    # ── Cascade-breakdown short. Same relaxation pattern as shock_reversal.
    "cascade_breakdown_short_enabled": True,
    "cascade_breakdown_rsi_trigger": 78.0,
    "cascade_breakdown_initial_drawdown_pct": 1.2,
    "cascade_breakdown_rebound_min_pct": 2.5,
    "cascade_breakdown_peak_reclaim_pct": 2.5,
    "cascade_breakdown_rebound_rsi_fade_pts": 5.0,
    "cascade_breakdown_break_pct": 1.0,
    "cascade_breakdown_sl_pct": 2.0,
    "cascade_breakdown_tp_pct": 2.5,
    "cascade_breakdown_trail_pct": 1.4,
    "cascade_breakdown_cooldown": 180,

    # Gold-specific bear-continuation short stays off by design.
    "macro_bear_continuation_short_enabled": False,

    # ── Shock-rebound long — tightened entry filter (deeper oversold +
    # stronger initial bounce) so we get fewer but better-quality entries,
    # plus a wider trail so the runners aren't stopped prematurely. v1 hit
    # 21 trail-stops vs 6 TP across 27 trades; this dial-in targets that.
    "shock_rebound_long_enabled": True,
    "shock_rebound_rsi_trigger": 20.0,
    "shock_rebound_rsi_reclaim_pts": 6.0,
    "shock_rebound_max_current_rsi": 55.0,   # v3: 65 -> 55; require still-oversold at entry.
    "shock_rebound_lookback_bars": 90,
    "shock_rebound_rebound_pct": 3.0,        # v3: 2.0 -> 3.0; deeper initial bounce required.
    "shock_rebound_bar_rise_pct": 0.7,
    "shock_rebound_min_atr_pct": 0.10,
    "shock_rebound_allow_active_spike": True,
    "shock_rebound_sl_pct": 2.0,
    "shock_rebound_tp_pct": 4.5,
    "shock_rebound_trail_pct": 3.0,
    "shock_rebound_cooldown": 180,

    # ── RSI-flush rebound long. Untouched — the only v1 module that earned
    # its slot (87.5% TP rate, +$116 PnL across 8 trades).
    "rsi_flush_rebound_long_enabled": True,
    "rsi_flush_drop_pct": 1.5,
    "rsi_flush_rsi_trigger": 22.0,
    "rsi_flush_sl_pct": 2.5,
    "rsi_flush_tp_pct": 1.8,
    "rsi_flush_cooldown": 0,
    "rsi_flush_require_green_rebound_bar": False,
    "rsi_flush_rebound_confirm_bars": 3,
    "rsi_flush_trend_filter_bars": 780,

    # ── Spike-long / persistent-spike-short slots. Off — UVXY-only logic.
    "spike_long_max": 0,
    "spike_max_entries": 0,
    "psshort_max": 0,

    # ── Event-target short. Peak / confirm-drop thresholds lowered (12→7,
    # 6→3) and anchor lookback extended to 11700 bars (~40 trading days)
    # so multi-week peaks anchor to their actual base — the v1 default
    # (~13 days) couldn't see the start of the Feb-Apr 2026 rally when the
    # peak finally rolled.
    "event_target_short_enabled": True,
    "event_target_anchor_lookback": 11700,
    "event_target_min_peak_pct": 7.0,
    "event_target_confirm_drop_pct": 3.0,
    "event_target_persistent_confirm_drop_pct": 4.0,
    "event_target_completion_pct": 50.0,
    "event_target_sl_pct": 3.5,

    # Decay-bounce slots stay off (UVXY-specific).
    "decay_bounce_max": 0,
    "decay_max_entries": 0,

    # ── Spike-classification thresholds — oil-scaled. spike_momentum_max
    # is now 0: the module bled $470 across 26 trades with 0 TP hits in v1.
    # The other knobs are left at v1 values so spike CLASSIFICATION (used
    # by other modules' allow_active_spike gates) still works correctly.
    "spike_gap_pct": 5.0,
    "grad_spike_lookback": 1170,
    "grad_spike_pct": 5.0,
    "grad_spike_rearm_bars": 1950,
    "psshort_window": 1950,
    "spike_lockout_bars": 390,
    "spike_profile_shock_gap_pct": 5.0,
    "spike_profile_shock_peak_pct": 10.0,
    "spike_momentum_max": 0,           # v2: disabled — see tuning history.
    "spike_momo_atr_mult": 1.0,
    "spike_momo_momentum_pct": 0.25,
    "spike_momo_min_peak_pct": 2.0,
    "spike_momo_min_atr_pct": 0.10,
    "spike_momo_max_rsi": 90.0,
    "spike_momo_trail_pct": 7.0,
    "spike_momo_sl_pct": 3.0,
    "spike_momo_cooldown": 120,

    # ── Macro context blockers — neutralised. The dollar/rates correlation
    # of crude is real but opposite-signed to gold and the existing
    # block-thresholds are gold-shaped. Park them at wide values so they
    # never fire; revisit in a follow-up sweep with an oil-specific overlay.
    "dollar_strength_block_ret_30": 999.0,
    "dollar_strength_block_ret_120": 999.0,
    "rates_weakness_block_ret_30": -999.0,
    "rates_weakness_block_ret_120": -999.0,

    # ── Every gold-specific gate stays off, by design. These knobs use
    # gold's fair-value cache and gold's peer/miners/riskoff context and
    # have no equivalent for crude.
    "gold_macro_score_enabled": False,
    "gold_macro_block_score": 1.0,
    "dollar_strength_weight": 0.0,
    "rates_weakness_weight": 0.0,
    "long_rates_weakness_weight": 0.0,
    "gold_macro_regime_enabled": False,
    "gold_macro_regime_fast_bars": 390,
    "gold_macro_regime_slow_bars": 1950,
    "gold_macro_regime_bullish_score": 1.0,
    "gold_macro_regime_bearish_score": -1.0,
    "gold_fair_value_regime_enabled": False,
    "gold_fair_value_confidence_min": 0.65,
    "gold_fair_value_bullish_slope_min": 1.0,
    "gold_fair_value_bearish_slope_max": -1.0,
    "gold_fair_value_undervalued_gap_pct": 2.5,
    "gold_fair_value_overvalued_gap_pct": 2.5,
    "gold_regime_dollar_weight": 0.0,
    "gold_regime_rates_weight": 0.0,
    "gold_regime_long_rates_weight": 0.0,
    "gold_regime_peer_weight": 0.0,
    "gold_regime_miners_weight": 0.0,
    "gold_regime_riskoff_weight": 0.0,
    "gold_context_assist_enabled": False,
    "gold_context_assist_min_score": 0.5,
    "gold_peer_strength_weight": 0.0,
    "gold_miners_strength_weight": 0.0,
    "gold_riskoff_strength_weight": 0.0,
}
