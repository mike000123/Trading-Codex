"""
strategies/trend_decay_strategy.py
────────────────────────────────────
UVXY Trend Decay Strategy
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PURPOSE:
  Captures the prolonged post-spike contango decay of UVXY toward its
  structural equilibrium (~$40).  The Bollinger+RSI strategy only holds
  post-spike regime for a few days (spike_high_window = 3 days).  This
  strategy detects the sustained weeks/months downtrend that follows and
  keeps shorting it all the way to the floor.

REGIME DETECTION (all conditions must be true):
  1. Medium EMA (decay_ema_period bars) is sloping DOWN
     → EMA[now] < EMA[slope_lookback bars ago]
  2. Price is BELOW that EMA (confirming trend direction)
  3. ATR is above min_atr_pct (not in flat/drift)
  4. Price is ABOVE floor_price + floor_buffer_pct (still room to fall)
  5. NOT in a spike (ATR < spike_atr_mult × ATR_MA)

ENTRY:
  Short when regime is active AND RSI is not oversold (rsi > rsi_os_gate)
  Cooldown between entries to avoid over-trading during choppy descent.

EXIT (combination — whichever fires first):
  A) Floor TP: price reaches floor_price (structural equilibrium target)
  B) ATR trailing SL: SL trails price down at (current_low + atr_trail_mult × ATR)
     — tightens automatically as volatility compresses near the bottom
  C) Regime exit: EMA slope turns flat/up (decay is over)
  D) Fixed SL: hard stop at entry × (1 + decay_sl_pct/100) as insurance

  The trailing SL is passed via metadata["trailing_atr_mult"] so the
  backtest engine can update trade.stop_loss each bar.
"""
from __future__ import annotations

from typing import Any, Optional
import numpy as np
import pandas as pd

from core.models import Signal, SignalAction
from strategies.base import BaseStrategy, register_strategy


# ── Shared indicator helpers ──────────────────────────────────────────────────

def _calc_rsi(series: pd.Series, period: int) -> pd.Series:
    d  = series.astype(float).diff()
    g  = d.clip(lower=0.0)
    l  = (-d).clip(lower=0.0)
    ag = g.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    al = l.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    return 100.0 - (100.0 / (1.0 + ag / al.replace(0.0, float("nan"))))


def _calc_atr(data: pd.DataFrame, period: int) -> pd.Series:
    hi   = data["high"].astype(float)
    lo   = data["low"].astype(float)
    cl   = data["close"].astype(float)
    prev = cl.shift(1)
    tr   = pd.concat([hi-lo, (hi-prev).abs(), (lo-prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()


@register_strategy
class TrendDecayStrategy(BaseStrategy):
    strategy_id = "trend_decay"
    name        = "UVXY Trend Decay"
    description = (
        "Shorts UVXY during prolonged post-spike contango decay toward "
        "structural equilibrium (~$40). Uses EMA slope + ATR trailing SL. "
        "Complements Bollinger+RSI post-spike regime for multi-week declines."
    )

    def default_params(self) -> dict[str, Any]:
        return {
            # ── Regime detection ────────────────────────────────────────────
            # Medium EMA — on 5-min bars: 390 bars = 1 trading day
            "decay_ema_period":    780,    # 2 trading days — smooths intraday noise
            "slope_lookback":      390,    # compare EMA now vs 1 day ago
            "slope_min_pct":       0.5,    # EMA must have fallen ≥ 0.5% over lookback
            "min_atr_pct":         0.2,    # skip if too flat (ATR < 0.2% of price)
            "spike_atr_mult":      2.0,    # suppress entries during spike (ATR > X×ATR_MA)
            # ── Floor / equilibrium ─────────────────────────────────────────
            "floor_price":         40.0,   # structural UVXY equilibrium target
            "floor_buffer_pct":    10.0,   # don't enter if price < floor × 1.10
            # ── Entry ───────────────────────────────────────────────────────
            "rsi_period":          14,
            "rsi_os_gate":         35,     # don't enter if RSI already oversold
            "cooldown_bars":       390,    # ~1 trading day between entries
            "max_entries_per_decay": 6,    # max shorts per continuous decay period
            # ── Exit ────────────────────────────────────────────────────────
            "decay_sl_pct":        6.0,    # hard SL: entry × (1 + X%) — insurance only
            "atr_trail_mult":      2.5,    # trailing SL = low + X × ATR (for shorts:
                                           # SL trails down as price falls)
            "atr_period":          14,     # ATR period for trailing SL
        }

    def validate_params(self) -> list[str]:
        p = {**self.default_params(), **self.params}
        errors = []
        if float(p["floor_price"]) <= 0:
            errors.append("floor_price must be > 0.")
        if float(p["atr_trail_mult"]) <= 0:
            errors.append("atr_trail_mult must be > 0.")
        if int(p["decay_ema_period"]) <= int(p["slope_lookback"]):
            errors.append("decay_ema_period must be > slope_lookback.")
        return errors

    def _compute_regime(self, close: pd.Series, data: pd.DataFrame, p: dict):
        """Vectorised regime detection — returns boolean Series."""
        atr_s    = _calc_atr(data, int(p["atr_period"]))
        atr_ma   = atr_s.rolling(20, min_periods=5).mean()
        in_spike = atr_s > float(p["spike_atr_mult"]) * atr_ma

        ema      = close.ewm(span=int(p["decay_ema_period"]), adjust=False).mean()
        lb       = int(p["slope_lookback"])
        slope_min = float(p["slope_min_pct"]) / 100.0
        ema_prev = ema.shift(lb)
        # EMA slope: fallen by at least slope_min_pct over lookback
        ema_declining = (ema_prev - ema) / ema_prev.replace(0, np.nan) >= slope_min
        price_below_ema = close < ema

        atr_pct  = atr_s / close.replace(0, np.nan) * 100
        not_flat = atr_pct >= float(p["min_atr_pct"])

        floor    = float(p["floor_price"])
        buf      = float(p["floor_buffer_pct"]) / 100.0
        above_floor = close > floor * (1.0 + buf)

        in_decay = (ema_declining & price_below_ema & not_flat
                    & above_floor & ~in_spike)
        return in_decay, atr_s, ema

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Signal:
        p = {**self.default_params(), **self.params}
        min_bars = int(p["decay_ema_period"]) + int(p["slope_lookback"]) + 2
        if len(data) < min_bars:
            return Signal(strategy_id=self.strategy_id, symbol=symbol,
                          action=SignalAction.HOLD,
                          metadata={"reason": "insufficient_data"})

        close           = data["close"].astype(float)
        rsi             = _calc_rsi(close, int(p["rsi_period"]))
        in_decay, atr_s, ema = self._compute_regime(close, data, p)

        curr    = float(close.iloc[-1])
        rsi_v   = float(rsi.iloc[-1])   if not pd.isna(rsi.iloc[-1])   else 50.0
        decay   = bool(in_decay.iloc[-1])
        curr_atr = float(atr_s.iloc[-1]) if not pd.isna(atr_s.iloc[-1]) else 0.0

        if decay and rsi_v > float(p["rsi_os_gate"]):
            floor   = float(p["floor_price"])
            sl      = curr * (1.0 + float(p["decay_sl_pct"]) / 100.0)
            return Signal(
                strategy_id=self.strategy_id, symbol=symbol,
                action=SignalAction.SELL,
                confidence=min(1.0, (curr - floor) / max(curr, 1e-9)),
                suggested_tp=floor,
                suggested_sl=sl,
                metadata={
                    "rsi": round(rsi_v, 2),
                    "regime": "decay",
                    "curr_atr": round(curr_atr, 4),
                    "trailing_atr_mult": float(p["atr_trail_mult"]),
                    "atr_period": int(p["atr_period"]),
                },
            )
        return Signal(strategy_id=self.strategy_id, symbol=symbol,
                      action=SignalAction.HOLD,
                      metadata={"regime": "decay" if decay else "no_decay",
                                "rsi": round(rsi_v, 2)})

    def generate_signals_bulk(self, data: pd.DataFrame, symbol: str):
        p = {**self.default_params(), **self.params}
        min_bars = int(p["decay_ema_period"]) + int(p["slope_lookback"]) + 2
        n        = len(data)

        close            = data["close"].astype(float)
        rsi              = _calc_rsi(close, int(p["rsi_period"]))
        in_decay, atr_s, ema = self._compute_regime(close, data, p)

        floor            = float(p["floor_price"])
        decay_sl_pct     = float(p["decay_sl_pct"])
        rsi_os_gate      = float(p["rsi_os_gate"])
        cooldown         = int(p["cooldown_bars"])
        atr_trail_mult   = float(p["atr_trail_mult"])
        atr_period       = int(p["atr_period"])
        max_entries      = int(p["max_entries_per_decay"])

        actions = [SignalAction.HOLD] * n
        metas   = [{"suggested_tp": None, "suggested_sl": None, "metadata": {}}] * n

        last_signal_bar   = -cooldown - 1
        entry_count       = 0
        was_in_decay      = False

        for pos in range(min_bars, n):
            currently_decay = bool(in_decay.iloc[pos])

            # Reset entry count when decay regime restarts
            if currently_decay and not was_in_decay:
                entry_count = 0
            was_in_decay = currently_decay

            if not currently_decay:
                continue
            if entry_count >= max_entries:
                continue
            if pos - last_signal_bar < cooldown:
                continue

            rsi_v    = float(rsi.iloc[pos]) if not pd.isna(rsi.iloc[pos]) else 50.0
            if rsi_v <= rsi_os_gate:
                continue   # already oversold — wait for bounce

            px       = float(close.iloc[pos])
            curr_atr = float(atr_s.iloc[pos]) if not pd.isna(atr_s.iloc[pos]) else 0.0
            sl       = px * (1.0 + decay_sl_pct / 100.0)

            actions[pos] = SignalAction.SELL
            metas[pos]   = {
                "suggested_tp": floor,
                "suggested_sl": sl,
                "metadata": {
                    "rsi": round(rsi_v, 2),
                    "regime": "decay",
                    "curr_atr": round(curr_atr, 4),
                    "trailing_atr_mult": atr_trail_mult,
                    "atr_period": atr_period,
                    "entry_n": entry_count + 1,
                },
            }
            entry_count     += 1
            last_signal_bar  = pos

        return actions, metas
