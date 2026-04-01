"""
strategies/spike_long_strategy.py
───────────────────────────────────
UVXY Spike Long Strategy
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PURPOSE:
  Captures the violent upward leg of UVXY VIX spikes.  Enters long
  when an ATR expansion confirms a spike is underway, exits via a
  tight ATR trailing stop that locks in profits as the spike matures
  and contracts when volatility starts unwinding.

ENTRY CONDITIONS (all required):
  1. ATR expansion: current ATR > spike_atr_mult × ATR_MA
     (confirms genuine volatility spike, not just noise)
  2. Price momentum: close > close N bars ago by at least momentum_pct%
     (confirms price is actually moving up, not just noisy ATR)
  3. RSI not overbought: RSI < rsi_ob_gate
     (avoids chasing already-exhausted spikes)
  4. Cooldown: min bars since last entry
  5. Max entries per spike: prevents pyramiding into a single event

EXIT (whichever fires first):
  A) ATR trailing stop: SL = highest_high_since_entry - atr_trail_mult × ATR
     — trails up as UVXY rises, locks in profit as spike peaks
  B) ATR contraction: current ATR < atr_exit_mult × ATR_MA
     (spike is unwinding — exit before the reversal)
  C) Hard SL: entry × (1 - spike_sl_pct/100) — maximum loss insurance
  D) Hard TP: entry × (1 + spike_tp_pct/100) — optional cap (0 = disabled)

  The trailing high-water mark and ATR contraction exit are passed via
  metadata so the backtest engine's trailing stop mechanism handles them.

RISK NOTE:
  UVXY spikes can reverse violently. This strategy uses moderate entry
  (ATR alone, no price-break confirmation needed) so SL must be tight.
  spike_sl_pct should be kept at 4-6% maximum.
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


def _calc_atr(data: pd.DataFrame, period: int) -> pd.Series:
    hi   = data["high"].astype(float)
    lo   = data["low"].astype(float)
    cl   = data["close"].astype(float)
    prev = cl.shift(1)
    tr   = pd.concat([hi-lo, (hi-prev).abs(), (lo-prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()


@register_strategy
class SpikeLongStrategy(BaseStrategy):
    strategy_id = "spike_long"
    name        = "UVXY Spike Long"
    description = (
        "Enters long during confirmed UVXY VIX spikes (ATR expansion + "
        "price momentum). Exits via ATR trailing stop and spike contraction "
        "signal. Captures the upward leg that Bollinger+RSI deliberately skips."
    )

    def default_params(self) -> dict[str, Any]:
        return {
            # ── ATR spike detection ─────────────────────────────────────────
            "atr_period":        14,
            "atr_ma_period":     20,      # ATR_MA baseline period
            "spike_atr_mult":    2.0,     # ATR > X × ATR_MA → spike confirmed
            "atr_exit_mult":     1.3,     # ATR < X × ATR_MA → spike unwinding → exit
            # ── Price momentum filter ───────────────────────────────────────
            "momentum_bars":     12,      # compare close vs N bars ago
            "momentum_pct":      1.5,     # price must be ≥ X% above N bars ago
            # ── RSI gate ────────────────────────────────────────────────────
            "rsi_period":        9,
            "rsi_ob_gate":       75,      # don't enter if RSI already overbought
            # ── Entry throttle ──────────────────────────────────────────────
            "cooldown_bars":     78,      # ~2 hours on 5-min bars
            "max_entries_per_spike": 2,   # max longs per continuous spike
            # ── Exit ────────────────────────────────────────────────────────
            "spike_sl_pct":      5.0,     # hard SL: entry × (1 - X%)
            "spike_tp_pct":      0.0,     # hard TP: 0 = disabled, rely on trailing
            "atr_trail_mult":    1.5,     # trailing SL = peak_high - X × ATR
                                          # tighter than decay (spikes reverse fast)
        }

    def validate_params(self) -> list[str]:
        p = {**self.default_params(), **self.params}
        errors = []
        if float(p["spike_atr_mult"]) <= float(p["atr_exit_mult"]):
            errors.append("spike_atr_mult must be > atr_exit_mult.")
        if float(p["spike_sl_pct"]) <= 0:
            errors.append("spike_sl_pct must be > 0.")
        return errors

    def _compute_spike_regime(self, close: pd.Series, data: pd.DataFrame, p: dict):
        """Returns (in_spike, atr_unwinding, atr_s, atr_ma_s)."""
        atr_s    = _calc_atr(data, int(p["atr_period"]))
        atr_ma_s = atr_s.rolling(int(p["atr_ma_period"]), min_periods=5).mean()
        in_spike      = atr_s > float(p["spike_atr_mult"]) * atr_ma_s
        atr_unwinding = atr_s < float(p["atr_exit_mult"])  * atr_ma_s
        return in_spike, atr_unwinding, atr_s, atr_ma_s

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Signal:
        p = {**self.default_params(), **self.params}
        min_bars = max(int(p["atr_period"]), int(p["atr_ma_period"]),
                       int(p["rsi_period"]), int(p["momentum_bars"])) + 2
        if len(data) < min_bars:
            return Signal(strategy_id=self.strategy_id, symbol=symbol,
                          action=SignalAction.HOLD,
                          metadata={"reason": "insufficient_data"})

        close = data["close"].astype(float)
        rsi   = _calc_rsi(close, int(p["rsi_period"]))
        in_spike, atr_unwinding, atr_s, atr_ma_s = self._compute_spike_regime(
            close, data, p)

        curr      = float(close.iloc[-1])
        rsi_v     = float(rsi.iloc[-1])   if not pd.isna(rsi.iloc[-1])   else 50.0
        curr_atr  = float(atr_s.iloc[-1]) if not pd.isna(atr_s.iloc[-1]) else 0.0
        spiking   = bool(in_spike.iloc[-1])
        unwinding = bool(atr_unwinding.iloc[-1])

        # Momentum: price must be ≥ momentum_pct% above N bars ago
        mb        = int(p["momentum_bars"])
        prev_px   = float(close.iloc[-mb]) if len(close) > mb else curr
        momentum_ok = (curr / max(prev_px, 1e-9) - 1.0) * 100 >= float(p["momentum_pct"])

        if spiking and not unwinding and momentum_ok and rsi_v < float(p["rsi_ob_gate"]):
            sl      = curr * (1.0 - float(p["spike_sl_pct"]) / 100.0)
            tp_pct  = float(p["spike_tp_pct"])
            tp      = curr * (1.0 + tp_pct / 100.0) if tp_pct > 0 else None
            return Signal(
                strategy_id=self.strategy_id, symbol=symbol,
                action=SignalAction.BUY,
                confidence=min(1.0, curr_atr / max(float(atr_ma_s.iloc[-1]), 1e-9) / 3.0),
                suggested_tp=tp,
                suggested_sl=sl,
                metadata={
                    "rsi": round(rsi_v, 2),
                    "regime": "spike",
                    "curr_atr": round(curr_atr, 4),
                    "trailing_atr_mult": float(p["atr_trail_mult"]),
                    "atr_period": int(p["atr_period"]),
                    "trail_direction": "long",  # engine trails up for longs
                },
            )
        return Signal(strategy_id=self.strategy_id, symbol=symbol,
                      action=SignalAction.HOLD,
                      metadata={"regime": "spike" if spiking else "no_spike",
                                "rsi": round(rsi_v, 2)})

    def generate_signals_bulk(self, data: pd.DataFrame, symbol: str):
        p = {**self.default_params(), **self.params}
        n = len(data)
        min_bars = max(int(p["atr_period"]), int(p["atr_ma_period"]),
                       int(p["rsi_period"]), int(p["momentum_bars"])) + 2

        close = data["close"].astype(float)
        rsi   = _calc_rsi(close, int(p["rsi_period"]))
        in_spike, atr_unwinding, atr_s, atr_ma_s = self._compute_spike_regime(
            close, data, p)

        spike_sl_pct   = float(p["spike_sl_pct"])
        spike_tp_pct   = float(p["spike_tp_pct"])
        rsi_ob_gate    = float(p["rsi_ob_gate"])
        cooldown       = int(p["cooldown_bars"])
        atr_trail_mult = float(p["atr_trail_mult"])
        atr_period     = int(p["atr_period"])
        max_entries    = int(p["max_entries_per_spike"])
        momentum_bars  = int(p["momentum_bars"])
        momentum_pct   = float(p["momentum_pct"])

        actions = [SignalAction.HOLD] * n
        metas   = [{"suggested_tp": None, "suggested_sl": None, "metadata": {}}] * n

        last_signal_bar  = -cooldown - 1
        entry_count      = 0
        was_spiking      = False

        for pos in range(min_bars, n):
            spiking   = bool(in_spike.iloc[pos])
            unwinding = bool(atr_unwinding.iloc[pos])

            # Reset entry count when a new spike begins
            if spiking and not was_spiking:
                entry_count = 0
            was_spiking = spiking

            if not spiking or unwinding:
                continue
            if entry_count >= max_entries:
                continue
            if pos - last_signal_bar < cooldown:
                continue

            rsi_v = float(rsi.iloc[pos]) if not pd.isna(rsi.iloc[pos]) else 50.0
            if rsi_v >= rsi_ob_gate:
                continue   # already overbought

            # Momentum check
            prev_pos = pos - momentum_bars
            if prev_pos < 0:
                continue
            prev_px = float(close.iloc[prev_pos])
            px      = float(close.iloc[pos])
            if (px / max(prev_px, 1e-9) - 1.0) * 100 < momentum_pct:
                continue   # price not moving up strongly enough

            curr_atr = float(atr_s.iloc[pos]) if not pd.isna(atr_s.iloc[pos]) else 0.0
            sl       = px * (1.0 - spike_sl_pct / 100.0)
            tp       = px * (1.0 + spike_tp_pct / 100.0) if spike_tp_pct > 0 else None

            actions[pos] = SignalAction.BUY
            metas[pos]   = {
                "suggested_tp": tp,
                "suggested_sl": sl,
                "metadata": {
                    "rsi": round(rsi_v, 2),
                    "regime": "spike",
                    "curr_atr": round(curr_atr, 4),
                    "trailing_atr_mult": atr_trail_mult,
                    "atr_period": atr_period,
                    "trail_direction": "long",
                    "entry_n": entry_count + 1,
                },
            }
            entry_count     += 1
            last_signal_bar  = pos

        return actions, metas
