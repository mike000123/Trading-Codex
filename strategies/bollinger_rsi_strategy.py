"""
strategies/bollinger_rsi_strategy.py
──────────────────────────────────────
Bollinger + RSI with Four-Regime Classification (stable version)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FOUR REGIMES:
  1. NORMAL      — Bollinger mean reversion, both longs and shorts
  2. SPIKE       — fast (ATR) OR gradual (price > 5% above 3-day low): NO shorts
  3. POST-SPIKE  — 3d-high > 1.5× 5d-EMA AND dropped 8%: AGGRESSIVE SHORTS
  4. DRIFT       — ATR < min_atr_pct% of price: ALL paused

POST-SPIKE EXIT LOGIC (dynamic):
  Rather than a fixed take-profit %, post-spike shorts exit when the declining
  phase shows signs of exhaustion.  Four independent signals vote; any ONE
  triggers an exit close (unless the hard floor TP is already hit first):

  A) REGIME EXIT   — post_spike flag itself flips False (price recovered
                     relative to the rolling high / long EMA anchor).
  B) RSI REVERSAL  — RSI rises ≥ rsi_rev_rise pts from a trough that was
                     ≤ rsi_rev_floor (momentum exhaustion confirmed).
                     Set dyn_rsi_rev_rise=0 to disable (default).
  C) EMA CROSS     — fast EMA crosses above slow EMA inside the short.
                     Set dyn_ema_fast=0 to disable (default).
                     WARNING: on 5-min data spans < 50/130 fire every ~26 bars.
  D) ATR COLLAPSE  — ATR drops below atr_collapse_pct × ATR-at-entry
                     (volatility compression = move is done).

  A hard floor TP (reversion_tp_pct) still acts as the maximum profit
  target so the trade closes even if none of the above fires first.
  The SL (reversion_sl_pct) is unchanged.

  All four thresholds are exposed as strategy params so they can be
  optimised in the Strategy Lab without touching this file.
"""
from __future__ import annotations
from typing import Any, Optional
import numpy as np
import pandas as pd
from core.models import Signal, SignalAction
from strategies.base import BaseStrategy, register_strategy


def _calc_rsi(series: pd.Series, period: int) -> pd.Series:
    d  = series.astype(float).diff()
    g  = d.clip(lower=0.0)
    l  = (-d).clip(lower=0.0)
    ag = g.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    al = l.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    return 100.0 - (100.0 / (1.0 + ag / al.replace(0.0, float("nan"))))


def _calc_bollinger(series: pd.Series, period: int, std_dev: float):
    sma   = series.rolling(period, min_periods=period).mean()
    std   = series.rolling(period, min_periods=period).std(ddof=1)
    return sma + std_dev * std, sma, sma - std_dev * std, std


def _calc_atr(data: pd.DataFrame, period: int) -> pd.Series:
    hi   = data["high"].astype(float)
    lo   = data["low"].astype(float)
    cl   = data["close"].astype(float)
    prev = cl.shift(1)
    tr   = pd.concat([hi-lo, (hi-prev).abs(), (lo-prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()


@register_strategy
class BollingerRSIStrategy(BaseStrategy):
    strategy_id = "bollinger_rsi"
    name        = "Bollinger + RSI (4-Regime)"
    description = (
        "Four-regime UVXY strategy: Normal mean-reversion, "
        "Spike (no shorts, catches gradual rises), "
        "Post-spike reversion (aggressive shorts, independent detection), "
        "Drift (flat)."
    )

    def default_params(self) -> dict[str, Any]:
        return {
            "bb_period":           20,
            "bb_std":              2.0,
            "rsi_period":          9,
            "rsi_oversold":        30,
            "rsi_overbought":      70,
            "sl_band_mult":        0.2,
            "require_cross":       True,
            "min_band_width_pct":  2.0,
            "min_rr_ratio":        1.5,
            "cooldown_bars":       5,
            "min_atr_pct":         0.3,
            "spike_atr_mult":      2.0,
            "rise_lookback":       1170,
            "rise_pct":            5.0,
            "spike_high_window":   1170,
            "spike_ema_mult":      1.5,
            "spike_ema_span":      1950,
            "peak_drop_pct":       8.0,
            "reversion_tp_pct":    15.0,   # hard-floor TP (max profit target, % below entry)
            "reversion_sl_pct":    5.0,    # stop-loss (% above entry)
            "reversion_rsi_min":   40,     # min RSI to enter a post-spike short
            # ── Dynamic exit parameters ──────────────────────────────────────
            # A) Regime exit: fires automatically when post_spike turns False
            "dyn_use_regime_exit": True,
            # B) RSI reversal: set dyn_rsi_rev_rise=0 to disable (default OFF).
            #    On 5-min UVXY the floor must be deep (≤25) and rise large (≥20)
            #    to avoid hair-triggers.
            "dyn_rsi_rev_floor":   25,     # trough must be ≤ this value
            "dyn_rsi_rev_rise":    0,      # 0 = disabled; set e.g. 20 to enable
            # C) EMA cross: set dyn_ema_fast=0 to disable (default OFF).
            #    On 5-min data EMA(5/13) crosses every ~26 bars — far too noisy.
            #    Use spans ≥ 50/130 if enabling (e.g. 1hr / 2.5hr equivalent).
            "dyn_ema_fast":        0,      # 0 = disabled
            "dyn_ema_slow":        130,    # only used when dyn_ema_fast > 0
            # D) ATR collapse: ATR falls to this fraction of ATR at entry
            "dyn_atr_collapse":    0.45,   # 0 = disabled; 0.45 = 45% of entry ATR
        }

    def validate_params(self) -> list[str]:
        p = {**self.default_params(), **self.params}
        errors = []
        if float(p["bb_std"]) <= 0:
            errors.append("bb_std must be > 0.")
        if float(p["rsi_oversold"]) >= float(p["rsi_overbought"]):
            errors.append("rsi_oversold must be < rsi_overbought.")
        return errors

    def _compute_regimes_bulk(self, close: pd.Series, data: pd.DataFrame, p: dict):
        atr_s  = _calc_atr(data, 14)
        atr_ma = atr_s.rolling(20, min_periods=5).mean()
        atr_spike   = atr_s > float(p["spike_atr_mult"]) * atr_ma
        rise_lb     = int(p["rise_lookback"])
        rise_pct    = float(p["rise_pct"])
        roll_min    = close.rolling(rise_lb, min_periods=1).min()
        rising      = (close / roll_min.replace(0, np.nan) - 1) * 100 > rise_pct
        in_spike    = atr_spike | rising
        hw          = int(p["spike_high_window"])
        ema_span    = int(p["spike_ema_span"])
        ema_mult    = float(p["spike_ema_mult"])
        drop_pct    = float(p["peak_drop_pct"])
        high_nd     = close.rolling(hw, min_periods=1).max()
        long_ema    = close.ewm(span=ema_span, adjust=False).mean()
        spike_occ   = high_nd > long_ema * ema_mult
        post_spike  = spike_occ & (close < high_nd * (1 - drop_pct / 100))
        atr_pct_s   = atr_s / close.replace(0, np.nan) * 100
        is_drift    = atr_pct_s < float(p["min_atr_pct"])
        return in_spike, post_spike, is_drift, atr_s

    def _compute_dynamic_exit_series(
        self,
        close: pd.Series,
        rsi: pd.Series,
        atr_s: pd.Series,
        post_spike: pd.Series,
        p: dict,
    ) -> pd.Series:
        """Return a boolean Series that is True on bars where an open
        post-spike short should be closed dynamically (before the hard TP).

        Signals B and C only — signal A (regime flip) and D (ATR collapse)
        are handled inline in the loop because they need per-trade context.

        B) RSI reversal  — disabled when dyn_rsi_rev_rise=0
        C) EMA cross     — disabled when dyn_ema_fast=0
        """
        exit_sig = pd.Series(False, index=close.index)

        # B) RSI reversal — disabled by default (rise=0)
        floor = float(p.get("dyn_rsi_rev_floor", 25))
        rise  = float(p.get("dyn_rsi_rev_rise",  0))
        if rise > 0:
            rsi_trough    = rsi.rolling(3, min_periods=1).min().shift(1)
            rsi_was_low   = rsi_trough <= floor
            rsi_rebounded = (rsi - rsi.shift(3).fillna(rsi)) >= rise
            exit_sig = exit_sig | (rsi_was_low & rsi_rebounded)

        # C) EMA cross — disabled by default (fast_span=0)
        fast_span = int(p.get("dyn_ema_fast", 0))
        slow_span = int(p.get("dyn_ema_slow", 130))
        if fast_span > 0 and slow_span > fast_span:
            ema_fast = close.ewm(span=fast_span, adjust=False).mean()
            ema_slow = close.ewm(span=slow_span, adjust=False).mean()
            cross_up = (ema_fast.shift(1) <= ema_slow.shift(1)) & (ema_fast > ema_slow)
            exit_sig = exit_sig | cross_up

        return exit_sig

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Signal:
        p = {**self.default_params(), **self.params}
        min_bars = max(int(p["bb_period"]), int(p["rsi_period"]),
                       int(p["spike_high_window"])) + 2
        if len(data) < min_bars:
            return Signal(strategy_id=self.strategy_id, symbol=symbol,
                          action=SignalAction.HOLD,
                          metadata={"reason": "insufficient_data"})
        close              = data["close"].astype(float)
        upper, sma, lower, _ = _calc_bollinger(close, int(p["bb_period"]), float(p["bb_std"]))
        rsi                = _calc_rsi(close, int(p["rsi_period"]))
        in_spike, post_spike, is_drift, atr_s = self._compute_regimes_bulk(close, data, p)
        curr  = close.iloc[-1]
        rsi_v = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
        bw    = float(upper.iloc[-1] - lower.iloc[-1])
        bw_pct = bw / max(curr, 1e-9) * 100
        regime = ("drift"      if bool(is_drift.iloc[-1])   else
                  "post_spike" if bool(post_spike.iloc[-1])  else
                  "spike"      if bool(in_spike.iloc[-1])    else "normal")
        action: SignalAction           = SignalAction.HOLD
        suggested_tp: Optional[float] = None
        suggested_sl: Optional[float] = None
        # ── Dynamic exit check for live/forward use ───────────────────────────
        # Callers signal an open post-spike short by passing
        #   params={"open_rev_entry_atr": <atr_at_entry>}  (non-zero activates checks)
        open_entry_atr = float(self.params.get("open_rev_entry_atr", 0.0))
        if open_entry_atr > 0:
            # A) Regime exit: post_spike has flipped off
            if regime != "post_spike" and bool(p.get("dyn_use_regime_exit", True)):
                return Signal(strategy_id=self.strategy_id, symbol=symbol,
                              action=SignalAction.BUY, confidence=1.0,
                              suggested_tp=None, suggested_sl=None,
                              metadata={"rsi": round(rsi_v, 2), "regime": "post_spike_exit",
                                        "exit_reason": "regime_exit"})
            # B/C) RSI reversal or EMA cross
            dyn_exit = self._compute_dynamic_exit_series(close, rsi, atr_s, post_spike, p)
            if bool(dyn_exit.iloc[-1]):
                return Signal(strategy_id=self.strategy_id, symbol=symbol,
                              action=SignalAction.BUY, confidence=1.0,
                              suggested_tp=None, suggested_sl=None,
                              metadata={"rsi": round(rsi_v, 2), "regime": "post_spike_exit",
                                        "exit_reason": "momentum_exit"})
            # D) ATR collapse
            atr_collapse = float(p.get("dyn_atr_collapse", 0.45))
            curr_atr = float(atr_s.iloc[-1]) if not pd.isna(atr_s.iloc[-1]) else 0.0
            if atr_collapse > 0 and curr_atr < atr_collapse * open_entry_atr:
                return Signal(strategy_id=self.strategy_id, symbol=symbol,
                              action=SignalAction.BUY, confidence=1.0,
                              suggested_tp=None, suggested_sl=None,
                              metadata={"rsi": round(rsi_v, 2), "regime": "post_spike_exit",
                                        "exit_reason": "atr_collapse"})
        # ── Post-spike entry ────────────────────────────────────────────────────
        if regime == "post_spike" and rsi_v >= int(p["reversion_rsi_min"]):
            tp = curr * (1 - float(p["reversion_tp_pct"]) / 100)
            sl = curr * (1 + float(p["reversion_sl_pct"]) / 100)
            action, suggested_tp, suggested_sl = SignalAction.SELL, tp, sl
        elif regime in ("spike", "normal"):
            req  = bool(p["require_cross"])
            prev = float(close.iloc[-2])
            lo_v = float(lower.iloc[-1]); hi_v = float(upper.iloc[-1])
            lo_p = float(lower.iloc[-2]); hi_p = float(upper.iloc[-2])
            lo_c = (prev > lo_p and curr <= lo_v) if req else curr <= lo_v
            hi_c = (prev < hi_p and curr >= hi_v) if req else curr >= hi_v
            if lo_c and rsi_v <= float(p["rsi_overbought"]) and bw_pct >= float(p["min_band_width_pct"]):
                tp = float(sma.iloc[-1]); sl = lo_v - float(p["sl_band_mult"]) * bw
                if (tp - curr) / max(curr - sl, 1e-9) >= float(p["min_rr_ratio"]):
                    action, suggested_tp, suggested_sl = SignalAction.BUY, tp, sl
            elif (hi_c and regime == "normal"
                  and rsi_v >= float(p["rsi_oversold"])
                  and bw_pct >= float(p["min_band_width_pct"])):
                tp = float(sma.iloc[-1]); sl = hi_v + float(p["sl_band_mult"]) * bw
                if (curr - tp) / max(sl - curr, 1e-9) >= float(p["min_rr_ratio"]):
                    action, suggested_tp, suggested_sl = SignalAction.SELL, tp, sl
        return Signal(strategy_id=self.strategy_id, symbol=symbol, action=action,
                      confidence=min(1.0, abs(rsi_v - 50) / 50),
                      suggested_tp=suggested_tp, suggested_sl=suggested_sl,
                      metadata={"rsi": round(rsi_v, 2), "regime": regime, "bw_pct": round(bw_pct, 2)})

    def generate_signals_bulk(self, data: pd.DataFrame, symbol: str):
        p = {**self.default_params(), **self.params}
        bb_period     = int(p["bb_period"]); bb_std = float(p["bb_std"])
        rsi_period    = int(p["rsi_period"]); oversold = float(p["rsi_oversold"])
        overbought    = float(p["rsi_overbought"]); sl_band_mult = float(p["sl_band_mult"])
        require_cross = bool(p["require_cross"]); min_bw_pct = float(p["min_band_width_pct"])
        min_rr        = float(p["min_rr_ratio"]); cooldown = int(p["cooldown_bars"])
        rev_tp_pct    = float(p["reversion_tp_pct"]); rev_sl_pct = float(p["reversion_sl_pct"])
        rev_rsi_min   = float(p["reversion_rsi_min"])
        use_regime_exit = bool(p.get("dyn_use_regime_exit", True))
        atr_collapse    = float(p.get("dyn_atr_collapse", 0.45))
        close              = data["close"].astype(float)
        upper, sma, lower, _ = _calc_bollinger(close, bb_period, bb_std)
        rsi                = _calc_rsi(close, rsi_period)
        band_width         = upper - lower
        bw_pct             = band_width / close.replace(0, np.nan) * 100
        in_spike, post_spike, is_drift, atr_s = self._compute_regimes_bulk(close, data, p)
        # Pre-compute dynamic exit signals B and C (vectorised)
        dyn_exit_sig = self._compute_dynamic_exit_series(close, rsi, atr_s, post_spike, p)
        prev_close = close.shift(1); prev_upper = upper.shift(1); prev_lower = lower.shift(1)
        if require_cross:
            long_bb  = (prev_close > prev_lower) & (close <= lower)
            short_bb = (prev_close < prev_upper) & (close >= upper)
        else:
            long_bb  = close <= lower; short_bb = close >= upper
        long_sl      = lower - sl_band_mult * band_width
        long_tp_d    = (sma - close).clip(lower=0)
        long_sl_d    = (close - long_sl).clip(lower=1e-9)
        long_rr_ok   = (long_tp_d / long_sl_d) >= min_rr
        short_sl     = upper + sl_band_mult * band_width
        short_tp_d   = (close - sma).clip(lower=0)
        short_sl_d   = (short_sl - close).clip(lower=1e-9)
        short_rr_ok  = (short_tp_d / short_sl_d) >= min_rr
        bw_ok        = bw_pct >= min_bw_pct
        atr_ok       = (atr_s / close.replace(0, np.nan) * 100) >= float(p["min_atr_pct"])
        long_sig   = (~is_drift & ~post_spike &
                      long_bb & (rsi <= overbought) & bw_ok & long_rr_ok & atr_ok)
        short_sig  = (~is_drift & ~post_spike & ~in_spike &
                      short_bb & (rsi >= oversold) & bw_ok & short_rr_ok & atr_ok)
        rev_sig    = (post_spike & ~is_drift & (rsi >= rev_rsi_min))
        n       = len(data)
        actions = [SignalAction.HOLD] * n
        metas   = [{"suggested_tp": None, "suggested_sl": None, "metadata": {}}] * n
        last_signal_bar   = -cooldown - 1
        max_rev_entries   = 2
        rev_entries_count = 0
        was_in_post_spike = False
        # Track open post-spike short for dynamic exit
        open_rev_entry_bar: Optional[int] = None
        open_rev_entry_atr: float         = 0.0
        for pos in range(n):
            is_rev   = bool(rev_sig.iloc[pos])
            is_long  = bool(long_sig.iloc[pos])
            is_short = bool(short_sig.iloc[pos])
            currently_post = bool(post_spike.iloc[pos])
            if not currently_post and was_in_post_spike:
                rev_entries_count = 0
            was_in_post_spike = currently_post
            # ── Dynamic exit: check if an open post-spike short should close ──
            if open_rev_entry_bar is not None:
                exit_reason: Optional[str] = None
                # A) Regime exit: post_spike flipped off
                if use_regime_exit and not currently_post:
                    exit_reason = "regime_exit"
                # B/C) RSI reversal or EMA cross (pre-computed, off by default)
                elif bool(dyn_exit_sig.iloc[pos]):
                    exit_reason = "momentum_exit"
                # D) ATR collapse below fraction of entry ATR
                elif (atr_collapse > 0
                      and open_rev_entry_atr > 0
                      and float(atr_s.iloc[pos]) < atr_collapse * open_rev_entry_atr):
                    exit_reason = "atr_collapse"
                if exit_reason is not None:
                    px      = float(close.iloc[pos])
                    rsi_val = float(rsi.iloc[pos]) if not pd.isna(rsi.iloc[pos]) else 50.0
                    actions[pos] = SignalAction.BUY   # BUY = close the short
                    metas[pos]   = {"suggested_tp": None, "suggested_sl": None,
                                     "metadata": {"rsi": round(rsi_val, 2),
                                                   "regime": "post_spike_exit",
                                                   "exit_reason": exit_reason}}
                    open_rev_entry_bar = None
                    open_rev_entry_atr = 0.0
                    last_signal_bar    = pos
                    continue
            # ── Entry logic ───────────────────────────────────────────────────
            if not (is_long or is_short or is_rev):
                continue
            rsi_val = float(rsi.iloc[pos]) if not pd.isna(rsi.iloc[pos]) else 50.0
            px      = float(close.iloc[pos])
            if is_rev:
                if rev_entries_count >= max_rev_entries:
                    continue
                if pos - last_signal_bar < cooldown * 10:
                    continue
                tp = px * (1 - rev_tp_pct / 100); sl = px * (1 + rev_sl_pct / 100)
                actions[pos] = SignalAction.SELL
                metas[pos]   = {"suggested_tp": tp, "suggested_sl": sl,
                                 "metadata": {"rsi": round(rsi_val, 2), "regime": "post_spike",
                                              "entry_n": rev_entries_count + 1}}
                rev_entries_count     += 1
                last_signal_bar        = pos
                open_rev_entry_bar     = pos
                open_rev_entry_atr     = float(atr_s.iloc[pos]) if not pd.isna(atr_s.iloc[pos]) else 0.0
                continue
            if pos - last_signal_bar < cooldown:
                continue
            if is_long and not pd.isna(rsi.iloc[pos]):
                tp = float(sma.iloc[pos]); sl = float(long_sl.iloc[pos])
                actions[pos] = SignalAction.BUY
                metas[pos]   = {"suggested_tp": tp, "suggested_sl": sl,
                                 "metadata": {"rsi": round(rsi_val, 2), "regime": "normal"}}
                last_signal_bar    = pos
                open_rev_entry_bar = None   # normal long cancels any tracked rev short
            elif is_short and not pd.isna(rsi.iloc[pos]):
                tp = float(sma.iloc[pos]); sl = float(short_sl.iloc[pos])
                actions[pos] = SignalAction.SELL
                metas[pos]   = {"suggested_tp": tp, "suggested_sl": sl,
                                 "metadata": {"rsi": round(rsi_val, 2), "regime": "normal"}}
                last_signal_bar = pos
        return actions, metas
