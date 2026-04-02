"""
strategies/bollinger_rsi_strategy.py  — Full UVXY Cycle (6-Regime)

SIX REGIMES (priority order):
  1. DRIFT        — ATR too low, all paused
  2. SPIKE LONG   — ATR-confirmed fast spike + momentum → LONG w/ ATR trail
                    NOTE: gradual-rise flag suppresses shorts only, never longs
  3. POST-SPIKE   — dropped 8% from spike high → AGGRESSIVE SHORTS (15% TP)
  4. DECAY        — EMA declining, price below EMA → SHORTS toward $40
  5. NORMAL       — Bollinger mean reversion

KEY FIX vs previous broken version:
  in_spike_atr   = ATR expansion only  → triggers SPIKE LONG
  suppress_shorts = ATR OR gradual rise → blocks normal/decay shorts
  Previously these were the same variable, causing longs to fire
  during gradual declines (the rising flag is True for most of a
  2-year declining dataset). Now correctly separated.
"""
from __future__ import annotations
from typing import Any, Optional
import numpy as np
import pandas as pd
from core.models import Signal, SignalAction
from strategies.base import BaseStrategy, register_strategy


def _calc_rsi(series, period):
    d = series.astype(float).diff()
    g = d.clip(lower=0.0); l = (-d).clip(lower=0.0)
    ag = g.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    al = l.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    return 100.0 - (100.0 / (1.0 + ag / al.replace(0.0, float("nan"))))


def _calc_bollinger(series, period, std_dev):
    sma = series.rolling(period, min_periods=period).mean()
    std = series.rolling(period, min_periods=period).std(ddof=1)
    return sma + std_dev * std, sma, sma - std_dev * std, std


def _calc_atr(data, period):
    hi = data["high"].astype(float); lo = data["low"].astype(float)
    cl = data["close"].astype(float); prev = cl.shift(1)
    tr = pd.concat([hi-lo, (hi-prev).abs(), (lo-prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()


@register_strategy
class BollingerRSIStrategy(BaseStrategy):
    strategy_id = "bollinger_rsi"
    name        = "Bollinger + RSI (Full Cycle)"
    description = (
        "Full UVXY cycle: Spike Long → Post-spike Short → "
        "Decay Short (toward $40) → Normal mean-reversion. "
        "Auto-detects all 6 regimes — no manual switching needed."
    )

    def default_params(self):
        return {
            # SHARED
            "rsi_period":           9,
            "min_atr_pct":          0.3,
            # REGIME 2: SPIKE LONG (ATR-confirmed only)
            "spike_atr_mult":       2.0,
            "spike_atr_ma_period":  20,
            "spike_atr_exit_mult":  1.3,
            "spike_momentum_bars":  12,
            "spike_momentum_pct":   1.5,
            "spike_rsi_ob":         80,    # raised: spikes can push RSI very high
            "spike_sl_pct":         8.0,   # wider: UVXY spikes have 5-10% intraday swings
            "spike_atr_trail":      3.0,   # wider trail: gives room during volatile spike
            "spike_cooldown":       195,   # ~4 hrs between entries on same spike
            "spike_max_entries":    4,     # more entries: spikes can last 1-3 days
            # GRADUAL RISE — short suppression only, never triggers longs
            "rise_lookback":        1170,
            "rise_pct":             5.0,
            # REGIME 3: POST-SPIKE SHORT
            "spike_high_window":    1170,
            "spike_ema_mult":       1.5,
            "spike_ema_span":       1950,
            "peak_drop_pct":        8.0,
            "reversion_tp_pct":     15.0,
            "reversion_sl_pct":     5.0,
            "reversion_rsi_min":    40,
            "max_rev_entries":      2,
            # Post-spike dynamic exit
            "dyn_use_regime_exit":  True,
            "dyn_rsi_rev_floor":    25,
            "dyn_rsi_rev_rise":     0,
            "dyn_ema_fast":         0,
            "dyn_ema_slow":         130,
            "dyn_atr_collapse":     0.45,
            # REGIME 4: DECAY SHORT
            "decay_ema_period":     1950,
            "decay_slope_lb":       780,
            "decay_slope_min_pct":  0.3,
            "decay_floor":          40.0,
            "decay_floor_buf":      8.0,
            "decay_rsi_os":         32,
            "decay_sl_pct":         12.0,
            "decay_atr_trail":      4.5,
            "decay_atr_trail_min_pct": 3.0,
            "decay_cooldown":       780,    # ~2 days between entries
            "decay_max_entries":    12,
            # Trend confirmation: decay regime must be active for this many
            # consecutive bars before any entry is allowed. Prevents entering
            # during choppy post-spike consolidation before trend is confirmed.
            # 1950 bars = ~5 trading days — one full week of sustained decline.
            "decay_confirm_bars":   1950,
            # REGIME 5: NORMAL
            "bb_period":            20,
            "bb_std":               2.0,
            "rsi_oversold":         30,
            "rsi_overbought":       70,
            "sl_band_mult":         0.2,
            "require_cross":        True,
            "min_band_width_pct":   2.0,
            "min_rr_ratio":         1.5,
            "cooldown_bars":        5,
        }

    def validate_params(self):
        p = {**self.default_params(), **self.params}
        errors = []
        if float(p["bb_std"]) <= 0: errors.append("bb_std must be > 0.")
        if float(p["rsi_oversold"]) >= float(p["rsi_overbought"]): errors.append("rsi_oversold must be < rsi_overbought.")
        if float(p["spike_atr_mult"]) <= float(p["spike_atr_exit_mult"]): errors.append("spike_atr_mult must be > spike_atr_exit_mult.")
        if float(p["decay_floor"]) <= 0: errors.append("decay_floor must be > 0.")
        if int(p["decay_ema_period"]) <= int(p["decay_slope_lb"]): errors.append("decay_ema_period must be > decay_slope_lb.")
        return errors

    def _compute_all_regimes(self, close, data, p):
        atr_s  = _calc_atr(data, 14)
        atr_ma = atr_s.rolling(int(p["spike_atr_ma_period"]), min_periods=5).mean()
        atr_pct    = atr_s / close.replace(0, np.nan) * 100
        is_drift   = atr_pct < float(p["min_atr_pct"])
        # Fast ATR spike — for long entries
        in_spike_atr = atr_s > float(p["spike_atr_mult"]) * atr_ma
        atr_unwind   = atr_s < float(p["spike_atr_exit_mult"]) * atr_ma
        # Gradual rise — short suppression ONLY
        roll_min       = close.rolling(int(p["rise_lookback"]), min_periods=1).min()
        gradual_rising = (close / roll_min.replace(0, np.nan) - 1) * 100 > float(p["rise_pct"])
        suppress_shorts = in_spike_atr | gradual_rising
        # Post-spike
        high_nd    = close.rolling(int(p["spike_high_window"]), min_periods=1).max()
        long_ema   = close.ewm(span=int(p["spike_ema_span"]), adjust=False).mean()
        spike_occ  = high_nd > long_ema * float(p["spike_ema_mult"])
        post_spike = spike_occ & (close < high_nd * (1 - float(p["peak_drop_pct"]) / 100)) & ~in_spike_atr
        # Decay
        d_ema    = close.ewm(span=int(p["decay_ema_period"]), adjust=False).mean()
        ema_prev = d_ema.shift(int(p["decay_slope_lb"]))
        ema_dec  = ((ema_prev - d_ema) / ema_prev.replace(0, np.nan)) >= float(p["decay_slope_min_pct"]) / 100.0
        floor    = float(p["decay_floor"]); buf = float(p["decay_floor_buf"]) / 100.0
        in_decay_raw = ema_dec & (close < d_ema) & (close > floor * (1.0 + buf)) & ~in_spike_atr & ~post_spike & ~is_drift
        # Confirmation: require decay to have been active for N consecutive bars
        # before allowing entries. Prevents shorting into choppy consolidation.
        confirm_bars = int(p.get("decay_confirm_bars", 1950))
        decay_streak = in_decay_raw.astype(int)
        # Rolling sum — if sum over last N bars == N, regime has been on that whole time
        in_decay = in_decay_raw & (decay_streak.rolling(confirm_bars, min_periods=confirm_bars).sum() >= confirm_bars)
        return is_drift, in_spike_atr, atr_unwind, suppress_shorts, post_spike, in_decay, atr_s, atr_ma

    def _compute_dynamic_exit_series(self, close, rsi, atr_s, post_spike, p):
        exit_sig = pd.Series(False, index=close.index)
        rise = float(p.get("dyn_rsi_rev_rise", 0))
        if rise > 0:
            floor = float(p.get("dyn_rsi_rev_floor", 25))
            trough = rsi.rolling(3, min_periods=1).min().shift(1)
            rebound = (rsi - rsi.shift(3).fillna(rsi)) >= rise
            exit_sig = exit_sig | ((trough <= floor) & rebound)
        fast = int(p.get("dyn_ema_fast", 0)); slow = int(p.get("dyn_ema_slow", 130))
        if fast > 0 and slow > fast:
            ef = close.ewm(span=fast, adjust=False).mean()
            es = close.ewm(span=slow, adjust=False).mean()
            exit_sig = exit_sig | ((ef.shift(1) <= es.shift(1)) & (ef > es))
        return exit_sig

    def generate_signals_bulk(self, data, symbol):  # noqa: C901
        p = {**self.default_params(), **self.params}
        n = len(data)
        close = data["close"].astype(float)
        rsi   = _calc_rsi(close, int(p["rsi_period"]))
        (is_drift, in_spike_atr, atr_unwind, suppress_shorts,
         post_spike, in_decay, atr_s, atr_ma) = self._compute_all_regimes(close, data, p)
        upper, sma, lower, _ = _calc_bollinger(close, int(p["bb_period"]), float(p["bb_std"]))
        bw = upper - lower; bw_pct = bw / close.replace(0, np.nan) * 100
        prev_cl = close.shift(1); prev_up = upper.shift(1); prev_lo = lower.shift(1)
        mb = int(p["spike_momentum_bars"])
        momentum = (close / close.shift(mb).replace(0, np.nan) - 1.0) * 100 >= float(p["spike_momentum_pct"])
        dyn_exit = self._compute_dynamic_exit_series(close, rsi, atr_s, post_spike, p)
        req = bool(p["require_cross"])
        long_bb  = ((prev_cl > prev_lo) & (close <= lower)) if req else (close <= lower)
        short_bb = ((prev_cl < prev_up) & (close >= upper)) if req else (close >= upper)
        long_sl_s  = lower  - float(p["sl_band_mult"]) * bw
        short_sl_s = upper  + float(p["sl_band_mult"]) * bw
        bw_ok = bw_pct >= float(p["min_band_width_pct"])
        atr_ok = (atr_s / close.replace(0, np.nan) * 100) >= float(p["min_atr_pct"])
        min_rr = float(p["min_rr_ratio"])
        long_rr_ok  = ((sma-close).clip(lower=0) / (close-long_sl_s).clip(lower=1e-9)) >= min_rr
        short_rr_ok = ((close-sma).clip(lower=0) / (short_sl_s-close).clip(lower=1e-9)) >= min_rr
        # Normal longs: not blocked by rising (longs OK during gradual rise)
        normal_long  = (~is_drift & ~in_spike_atr & ~post_spike & ~in_decay
                        & long_bb & (rsi <= float(p["rsi_overbought"])) & bw_ok & long_rr_ok & atr_ok)
        # Normal shorts: suppressed during ATR spike OR gradual rise
        normal_short = (~is_drift & ~suppress_shorts & ~post_spike & ~in_decay
                        & short_bb & (rsi >= float(p["rsi_oversold"])) & bw_ok & short_rr_ok & atr_ok)
        # Decay shorts also suppressed during any rise
        decay_ok = in_decay & ~suppress_shorts
        rev_sig  = post_spike & ~is_drift & (rsi >= float(p["reversion_rsi_min"]))
        actions = [SignalAction.HOLD] * n
        metas   = [{"suggested_tp": None, "suggested_sl": None, "metadata": {}}] * n
        spike_last = -9999; spike_entries = 0; was_spike = False
        ps_last    = -9999; ps_entries    = 0; was_ps    = False
        decay_last = -9999; decay_entries = 0; was_decay = False
        bb_last    = -9999
        open_rev_bar: Optional[int] = None
        open_rev_atr: float = 0.0
        use_re  = bool(p.get("dyn_use_regime_exit", True))
        atr_col = float(p.get("dyn_atr_collapse", 0.45))
        ps_cd   = int(p["rsi_period"]) * 10
        for pos in range(n):
            spiking = bool(in_spike_atr.iloc[pos])
            unwind  = bool(atr_unwind.iloc[pos])
            ps      = bool(post_spike.iloc[pos])
            decay   = bool(decay_ok.iloc[pos])
            drift   = bool(is_drift.iloc[pos])
            px      = float(close.iloc[pos])
            rsi_v   = float(rsi.iloc[pos]) if not pd.isna(rsi.iloc[pos]) else 50.0
            c_atr   = float(atr_s.iloc[pos]) if not pd.isna(atr_s.iloc[pos]) else 0.0
            if spiking and not was_spike: spike_entries = 0
            if ps      and not was_ps:    ps_entries    = 0
            if decay   and not was_decay: decay_entries = 0
            was_spike = spiking; was_ps = ps; was_decay = decay
            if drift: continue
            # REGIME 2: SPIKE LONG
            if spiking and not unwind:
                if (spike_entries < int(p["spike_max_entries"])
                        and pos - spike_last >= int(p["spike_cooldown"])
                        and bool(momentum.iloc[pos])
                        and rsi_v < float(p["spike_rsi_ob"])):
                    sl = px * (1.0 - float(p["spike_sl_pct"]) / 100.0)
                    actions[pos] = SignalAction.BUY
                    metas[pos]   = {"suggested_tp": None, "suggested_sl": sl,
                                     "metadata": {"regime": "spike", "rsi": round(rsi_v, 2),
                                                   "trailing_atr_mult": float(p["spike_atr_trail"]),
                                                   "atr_period": 14, "trail_direction": "long",
                                                   "entry_n": spike_entries + 1}}
                    spike_entries += 1; spike_last = pos
                continue
            # REGIME 3: POST-SPIKE SHORT
            if ps:
                if open_rev_bar is not None:
                    reason: Optional[str] = None
                    if use_re and not ps: reason = "regime_exit"
                    elif bool(dyn_exit.iloc[pos]): reason = "momentum_exit"
                    elif atr_col > 0 and open_rev_atr > 0 and c_atr < atr_col * open_rev_atr: reason = "atr_collapse"
                    if reason is not None:
                        actions[pos] = SignalAction.BUY
                        metas[pos]   = {"suggested_tp": None, "suggested_sl": None,
                                         "metadata": {"regime": "post_spike_exit",
                                                       "rsi": round(rsi_v, 2), "exit_reason": reason}}
                        open_rev_bar = None; open_rev_atr = 0.0; ps_last = pos; continue
                if (bool(rev_sig.iloc[pos]) and ps_entries < int(p["max_rev_entries"])
                        and pos - ps_last >= ps_cd):
                    tp = px * (1.0 - float(p["reversion_tp_pct"]) / 100.0)
                    sl = px * (1.0 + float(p["reversion_sl_pct"]) / 100.0)
                    actions[pos] = SignalAction.SELL
                    metas[pos]   = {"suggested_tp": tp, "suggested_sl": sl,
                                     "metadata": {"regime": "post_spike", "rsi": round(rsi_v, 2),
                                                   "entry_n": ps_entries + 1}}
                    ps_entries += 1; ps_last = pos; open_rev_bar = pos; open_rev_atr = c_atr
                continue
            # Leaving post-spike — fire regime_exit if trade open
            if open_rev_bar is not None and use_re:
                actions[pos] = SignalAction.BUY
                metas[pos]   = {"suggested_tp": None, "suggested_sl": None,
                                 "metadata": {"regime": "post_spike_exit",
                                               "rsi": round(rsi_v, 2), "exit_reason": "regime_exit"}}
                open_rev_bar = None; open_rev_atr = 0.0; ps_last = pos; continue
            # REGIME 4: DECAY SHORT
            if decay:
                if (decay_entries < int(p["decay_max_entries"])
                        and pos - decay_last >= int(p["decay_cooldown"])
                        and rsi_v > float(p["decay_rsi_os"])):
                    sl = px * (1.0 + float(p["decay_sl_pct"]) / 100.0)
                    actions[pos] = SignalAction.SELL
                    metas[pos]   = {"suggested_tp": float(p["decay_floor"]),
                                     "suggested_sl": sl,
                                     "metadata": {"regime": "decay", "rsi": round(rsi_v, 2),
                                                   "trailing_atr_mult": float(p["decay_atr_trail"]),
                                                   "trailing_atr_min_pct": float(p.get("decay_atr_trail_min_pct", 3.0)),
                                                   "atr_period": 14, "trail_direction": "short",
                                                   "entry_n": decay_entries + 1}}
                    decay_entries += 1; decay_last = pos
                continue
            # REGIME 5: NORMAL
            if pos - bb_last < int(p["cooldown_bars"]): continue
            px_sma = float(sma.iloc[pos])
            if bool(normal_long.iloc[pos]):
                actions[pos] = SignalAction.BUY
                metas[pos]   = {"suggested_tp": px_sma, "suggested_sl": float(long_sl_s.iloc[pos]),
                                 "metadata": {"regime": "normal", "rsi": round(rsi_v, 2)}}
                bb_last = pos
            elif bool(normal_short.iloc[pos]):
                actions[pos] = SignalAction.SELL
                metas[pos]   = {"suggested_tp": px_sma, "suggested_sl": float(short_sl_s.iloc[pos]),
                                 "metadata": {"regime": "normal", "rsi": round(rsi_v, 2)}}
                bb_last = pos
        return actions, metas

    def generate_signal(self, data, symbol):
        p = {**self.default_params(), **self.params}
        min_bars = max(int(p["spike_high_window"]), int(p["spike_ema_span"]),
                       int(p["decay_ema_period"]), int(p["decay_slope_lb"]),
                       int(p["bb_period"]), int(p["rise_lookback"])) + 2
        if len(data) < min_bars:
            return Signal(strategy_id=self.strategy_id, symbol=symbol,
                          action=SignalAction.HOLD, metadata={"reason": "insufficient_data"})
        actions, metas = self.generate_signals_bulk(data, symbol)
        meta = metas[-1]
        return Signal(strategy_id=self.strategy_id, symbol=symbol, action=actions[-1],
                      confidence=1.0, suggested_tp=meta.get("suggested_tp"),
                      suggested_sl=meta.get("suggested_sl"), metadata=meta.get("metadata", {}))
