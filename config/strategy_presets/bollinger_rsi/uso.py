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


v4 (attempt, reverted): added an opt-in trail-exit mode to
  rsi_flush_rebound_long via a new rsi_flush_trail_pct knob (1.5% on USO).
  Hypothesis was that tracking peak-and-trail would let oversold bounces
  ride past the fixed TP=3% cap.  Result was worse than v3.5 — the 1.5%
  trail tripped on routine intra-bounce retraces (USO bounces give back
  1-2% intraday on the way up), exiting at lower prices than the fixed TP
  would have locked in.  Counter-signal exits, which already extend
  profitable trades naturally when the regime flips, did a better job in
  practice than the trail did. Reverted to v3.5 fixed TP+SL.

  The new rsi_flush_trail_pct knob stays in default_params (default 0.0
  = off) and the strategy branch stays in place; this leaves the door
  open for a future preset to opt in with a wider trail (e.g. 3-4%) or
  for a hybrid "fixed TP first, then trail" exit if anyone wants to
  experiment.


v5 (this revision): targets two gaps surfaced after v3.5 stabilised the
  drawdown — slow-grind trends went uncaptured (Jul-Oct 2023 +33% move,
  zero trend_bias entries) and only ~47% of the Mar-Apr 2026 +75% rally
  was captured because the no-higher-reentry gate could only unlock on a
  full slow-EMA inversion, which doesn't happen until the trend has
  fully ended. Two changes:

    1. trend_bias_min_momentum_120 1.5 -> 0.6. v3 raised this from 0.6
       to 1.5 to fight v2's chase-the-top, but the structural gate
       added later does that job; the high momentum bar was now just
       blocking slow legitimate trends. 0.6 sits between GLD's 0.3 and
       the pure-vol-scaled equivalent of ~0.9.
    2. New trend_bias_reentry_retrace_pct knob (default 0.0 = off in
       default_params, opt-in here at 4.0). Gives the no-higher-reentry
       gate a softer secondary unlock: after a fill, if price retraces
       from its post-fill peak by >= 4% AND is back above fast EMA, the
       gate unlocks for the next qualifying signal. Catches healthy
       intra-trend pullbacks (the kind that retest the fast EMA without
       breaking the trend structure) so we can engage on continuation
       legs without chasing a still-rising price.


v5 (attempt, reverted): two changes targeting the Jul-Oct 2023 slow grind
  (which never produced a trend_bias entry) and the Mar-Apr 2026 rally
  (which captured ~47% of the move).
    - trend_bias_min_momentum_120 1.5 -> 0.6
    - trend_bias_reentry_retrace_pct 0.0 -> 4.0 (new opt-in knob)
  Result: total return dropped from +30% to +8%. Per-regime audit showed
  trend_bias_long went from 2 trades / +$140 to 6 trades / -$118. The
  Mar 06 +$107 home run was lost — the lower momentum threshold admitted
  an earlier setup at $87 (Mar 2) that took a small +$27 profit and
  consumed the gate-lock, blocking the Mar 06 EMA-pullback trade. The
  retrace-unlock then fired three times during the rally's topping
  phase ($122 / $137 / $126 entries), all stopped at -3%, which is
  exactly the chase-the-top failure mode the strict gate was designed
  to prevent. Slow-grind trends (Jul-Oct 2023, May 2025) also generated
  losers, not winners — by the time momentum_120 ticks above 0.6, the
  EMA-pullback structure is too late.

  Reverted to v3.5: trend_bias_min_momentum_120=1.5, trend_bias_reentry_retrace_pct=0.0.
  The new knob stays in default_params (default 0.0 = off) and the
  strategy-code branch stays gated; future experiments could try a
  stronger filter (e.g. require an actual EMA touch from above, not
  just price-above-EMA) before opting back in.

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
    "trend_bias_min_momentum_120": 1.5,    # v3.5: keeps strict; v5 lowering to 0.6 broke the Mar 06 home run.
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
    # trend_bias_reentry_retrace_pct intentionally NOT overridden (defaults
    # to 0.0 = strict v3 behavior). v5 tested it at 4.0 and the unlock fired
    # on rolling tops, not healthy retraces — three losing re-entries during
    # the Mar-Apr 2026 rally cost more than the early-trend capture earned.
    # The strategy-code branch stays in default_params; future experiments
    # could try a stronger filter (e.g. require an actual EMA touch from
    # above, not just price-above-EMA) before opting back in.
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
    "rsi_flush_sl_pct": 3.0,           # v3.5 retune: 2.5 -> 3.0 to balance reward/risk at 1:1.
    "rsi_flush_tp_pct": 3.0,           # v3.5 retune: 1.8 -> 3.0 to capture more of each oversold bounce.
    # v6 opt-in: when an rsi_flush trade is already up >= 2.0%, ignore opposing
    # regime signals and let the trade ride to TP/SL/trail/EOD instead. Targets
    # the v3.5 pattern where 8 of 12 rsi_flush exits closed via counter-signal
    # at +0.7% to +2.5% — the ones that exited at +2.0%+ were close to TP and
    # likely would have hit it. Smaller-profit trades still get counter-signal
    # protection so we don't expose ourselves to reversal risk on weak setups.
    "rsi_flush_counter_signal_min_profit_pct": 2.0,
    # rsi_flush_trail_pct intentionally NOT overridden here (defaults to 0.0).
    # v4 attempted a 1.5% trail on this module to ride bounces past the fixed
    # TP, but the run was worse than v3.5 — oversold bounces are inherently
    # choppy so a tight trail exits during normal intra-bounce retraces. Every
    # other preset (GLD/UVXY/VXX/VXZ) also keeps rsi_flush at fixed TP+SL by
    # design; this module is a quick scalp, not a ride-the-trend module.
    # Trail-style exits are handled by the OTHER modules already enabled
    # below (intraday_pullback_short, shock_reversal_short, cascade_breakdown
    # _short, shock_rebound_long, trend_bias_long).
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
