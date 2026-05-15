"""
strategies/ema_trend_rsi_strategy.py
──────────────────────────────────────
EMA Crossover + RSI Confirmation + 200 EMA Trend Filter
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Purpose-built for GC=F (Gold Futures) intraday — 1-min / 5-min bars.

WHY PREVIOUS VERSION GAVE ZERO TRADES:
  Required three simultaneous conditions (9>21 EMA, price>50 EMA, RSI<40)
  which almost never all align on a short data window. Fixed by:
  - Making the 200 EMA filter OPTIONAL (default ON but can disable)
  - Using EMA CROSSOVER as the primary entry signal (not RSI dip)
  - RSI used as a GATE (must be on the right side of 50), not a threshold

WHY RSI ALONE FAILS ON GOLD:
  Gold trends persistently — RSI > 70 for weeks during bull runs.
  Solution: EMA crossover determines WHEN to enter, RSI confirms DIRECTION.
  This is the practitioner consensus for 1-min gold scalping (2024-2026).

STRATEGY LOGIC:
  PRIMARY: 9 EMA crosses above 21 EMA → potential LONG
           9 EMA crosses below 21 EMA → potential SHORT

  FILTER 1: RSI must confirm direction
            Long:  RSI > 50 (momentum is bullish)
            Short: RSI < 50 (momentum is bearish)

  FILTER 2: 200 EMA trend filter (optional, default ON)
            Long:  price > 200 EMA (in broader uptrend)
            Short: price < 200 EMA (in broader downtrend)

  STOPS/TARGETS: ATR-based (1.5× SL, 2.5× TP)

RESEARCH BASIS:
  - 9/21 EMA crossover: most cited gold intraday entry signal
    (scribd gold scalping guide, opofinance 2025, xs.com 2026)
  - 200 EMA as trend filter: TradingView XAUUSD 1-min community strategy
    with highest engagement (Feb 2026) uses exactly this combination
  - RSI > 50 gate (not extreme threshold): avoids missing trades in
    trending conditions while still filtering weak signals

PARAMS:
  fast_ema       – fast EMA (default 9)
  slow_ema       – slow EMA (default 21)
  trend_ema      – trend filter EMA (default 200; set 0 to disable)
  rsi_period     – RSI period (default 9)
  rsi_gate       – RSI must be above (long) / below (short) this (default 50)
  atr_period     – ATR period (default 14)
  atr_sl_mult    – SL = entry ± atr_sl_mult × ATR (default 1.5)
  atr_tp_mult    – TP = entry ± atr_tp_mult × ATR (default 2.5)
"""
from __future__ import annotations

from typing import Any, Optional

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


def _calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.astype(float).ewm(span=period, adjust=False,
                                     min_periods=period).mean()


def _calc_atr(data: pd.DataFrame, period: int) -> pd.Series:
    hi   = data["high"].astype(float)
    lo   = data["low"].astype(float)
    cl   = data["close"].astype(float)
    prev = cl.shift(1)
    tr   = pd.concat([hi - lo, (hi - prev).abs(),
                      (lo - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()


@register_strategy
class EMATrendRSIStrategy(BaseStrategy):
    strategy_id = "ema_trend_rsi"
    name        = "EMA Crossover + RSI + Trend Filter"
    description = (
        "9/21 EMA crossover as primary entry signal, RSI > 50 gate for direction "
        "confirmation, optional 200 EMA trend filter. Purpose-built for GC=F "
        "1-min/5-min intraday. ATR-adaptive stops and targets."
    )

    def default_params(self) -> dict[str, Any]:
        return {
            "fast_ema":       9,
            "slow_ema":       21,
            "trend_ema":      200,   # set to 0 to disable
            "rsi_period":     9,
            "rsi_gate":       50.0,  # RSI > this for longs, < this for shorts
            "atr_period":     14,
            "atr_sl_mult":    1.5,
            "atr_tp_mult":    3.0,   # raised from 2.5 — gold trends run hard
            "atr_min_filter": 0.0,   # skip signal if ATR < this price value
                                     # e.g. 0.5 = skip if volatility < $0.50/bar
                                     # set to 0 to disable (default)
        }

    def validate_params(self) -> list[str]:
        p      = {**self.default_params(), **self.params}
        errors = []
        if int(p["fast_ema"]) >= int(p["slow_ema"]):
            errors.append("fast_ema must be less than slow_ema.")
        if float(p["atr_sl_mult"]) <= 0:
            errors.append("atr_sl_mult must be > 0.")
        if float(p["atr_tp_mult"]) <= float(p["atr_sl_mult"]):
            errors.append("atr_tp_mult should be greater than atr_sl_mult for positive R:R.")
        return errors

    def min_warmup_bars(self, symbol=None, source=None, interval=None) -> int:
        p = {**self.default_params(), **self.params}
        slow = int(p.get("slow_ema", 0))
        trend = int(p.get("trend_ema", 0)) if p.get("use_trend_filter") else 0
        rsi = int(p.get("rsi_period", 0))
        atr = int(p.get("atr_period", 0))
        return max(slow, trend, rsi, atr) + 10

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Signal:
        p              = {**self.default_params(), **self.params}
        fast_period    = int(p["fast_ema"])
        slow_period    = int(p["slow_ema"])
        trend_period   = int(p["trend_ema"])
        rsi_period     = int(p["rsi_period"])
        rsi_gate       = float(p["rsi_gate"])
        atr_period     = int(p["atr_period"])
        sl_mult        = float(p["atr_sl_mult"])
        tp_mult        = float(p["atr_tp_mult"])
        atr_min_filter = float(p["atr_min_filter"])
        use_trend      = trend_period > 0

        # Minimum bars needed
        min_bars = max(slow_period, trend_period if use_trend else 0,
                       rsi_period, atr_period) + 2
        if len(data) < min_bars:
            return Signal(strategy_id=self.strategy_id, symbol=symbol,
                          action=SignalAction.HOLD,
                          metadata={"reason": "insufficient_data",
                                    "have": len(data), "need": min_bars})

        close     = data["close"].astype(float)
        fast      = _calc_ema(close, fast_period)
        slow      = _calc_ema(close, slow_period)
        rsi       = _calc_rsi(close, rsi_period)
        atr       = _calc_atr(data, atr_period)

        curr_close = float(close.iloc[-1])
        curr_fast  = float(fast.iloc[-1])
        prev_fast  = float(fast.iloc[-2])
        curr_slow  = float(slow.iloc[-1])
        prev_slow  = float(slow.iloc[-2])
        curr_rsi   = float(rsi.iloc[-1])
        curr_atr   = float(atr.iloc[-1])

        # ATR minimum filter — skip signal in low-volatility/choppy conditions
        if atr_min_filter > 0 and curr_atr < atr_min_filter:
            return Signal(strategy_id=self.strategy_id, symbol=symbol,
                          action=SignalAction.HOLD,
                          metadata={"reason": "atr_below_filter",
                                    "atr": round(curr_atr, 4),
                                    "min_required": atr_min_filter})

        # EMA crossovers — primary entry signal
        bull_cross = prev_fast <= prev_slow and curr_fast > curr_slow
        bear_cross = prev_fast >= prev_slow and curr_fast < curr_slow

        # RSI gate — confirms momentum direction
        rsi_bullish = curr_rsi > rsi_gate
        rsi_bearish = curr_rsi < rsi_gate

        # 200 EMA trend filter (optional)
        trend_bullish = trend_bearish = True   # default: no filter
        if use_trend:
            trend = _calc_ema(close, trend_period)
            curr_trend    = float(trend.iloc[-1])
            trend_bullish = curr_close > curr_trend
            trend_bearish = curr_close < curr_trend
        else:
            curr_trend = float("nan")

        action: SignalAction           = SignalAction.HOLD
        suggested_tp: Optional[float] = None
        suggested_sl: Optional[float] = None
        trigger                       = ""

        if bull_cross and rsi_bullish and trend_bullish:
            action       = SignalAction.BUY
            suggested_tp = curr_close + tp_mult * curr_atr
            suggested_sl = curr_close - sl_mult * curr_atr
            trigger      = (f"9/21 EMA golden cross | RSI {curr_rsi:.1f} > {rsi_gate}"
                            + (f" | above 200 EMA" if use_trend else ""))

        elif bear_cross and rsi_bearish and trend_bearish:
            action       = SignalAction.SELL
            suggested_tp = curr_close - tp_mult * curr_atr
            suggested_sl = curr_close + sl_mult * curr_atr
            trigger      = (f"9/21 EMA death cross | RSI {curr_rsi:.1f} < {rsi_gate}"
                            + (f" | below 200 EMA" if use_trend else ""))

        return Signal(
            strategy_id  = self.strategy_id,
            symbol       = symbol,
            action       = action,
            confidence   = min(1.0, abs(curr_rsi - 50) / 50),
            suggested_tp = suggested_tp,
            suggested_sl = suggested_sl,
            metadata     = {
                "rsi":           round(curr_rsi, 2),
                "fast_ema":      round(curr_fast, 4),
                "slow_ema":      round(curr_slow, 4),
                "trend_ema":     round(curr_trend, 4) if use_trend else "off",
                "atr":           round(curr_atr, 4),
                "bull_cross":    bull_cross,
                "bear_cross":    bear_cross,
                "trigger":       trigger,
                "sl_distance":   round(sl_mult * curr_atr, 4),
                "tp_distance":   round(tp_mult * curr_atr, 4),
                "rr":            round(tp_mult / sl_mult, 2),
                "atr_min_filter":atr_min_filter,
            },
        )
    def generate_signals_bulk(self, data: pd.DataFrame, symbol: str):
        """Vectorised bulk — computes EMA + RSI + ATR once over full dataset."""
        p              = {**self.default_params(), **self.params}
        fast_period    = int(p["fast_ema"])
        slow_period    = int(p["slow_ema"])
        trend_period   = int(p["trend_ema"])
        rsi_period     = int(p["rsi_period"])
        rsi_gate       = float(p["rsi_gate"])
        atr_period     = int(p["atr_period"])
        sl_mult        = float(p["atr_sl_mult"])
        tp_mult        = float(p["atr_tp_mult"])
        atr_min_filter = float(p["atr_min_filter"])
        use_trend      = trend_period > 0

        close     = data["close"].astype(float)
        fast      = _calc_ema(close, fast_period)
        slow      = _calc_ema(close, slow_period)
        rsi       = _calc_rsi(close, rsi_period)
        atr       = _calc_atr(data, atr_period)
        prev_fast = fast.shift(1)
        prev_slow = slow.shift(1)

        bull_cross = (prev_fast <= prev_slow) & (fast > slow)
        bear_cross = (prev_fast >= prev_slow) & (fast < slow)

        if use_trend:
            trend         = _calc_ema(close, trend_period)
            trend_bullish = close > trend
            trend_bearish = close < trend
        else:
            trend_bullish = pd.Series(True,  index=data.index)
            trend_bearish = pd.Series(True,  index=data.index)

        atr_ok = (atr >= atr_min_filter) if atr_min_filter > 0 else pd.Series(True, index=data.index)

        long_signal  = bull_cross & (rsi > rsi_gate)  & trend_bullish & atr_ok
        short_signal = bear_cross & (rsi < rsi_gate)  & trend_bearish & atr_ok & ~long_signal

        n       = len(data)
        actions = [SignalAction.HOLD] * n
        metas   = [{"suggested_tp": None, "suggested_sl": None, "metadata": {}}] * n

        for i in data.index[long_signal & rsi.notna() & atr.notna()]:
            pos = data.index.get_loc(i)
            px  = float(close.iloc[pos])
            a   = float(atr.iloc[pos])
            actions[pos] = SignalAction.BUY
            metas[pos]   = {"suggested_tp": px + tp_mult * a,
                             "suggested_sl": px - sl_mult * a,
                             "metadata": {"rsi": round(float(rsi.iloc[pos]), 2), "atr": round(a, 4)}}

        for i in data.index[short_signal & rsi.notna() & atr.notna()]:
            pos = data.index.get_loc(i)
            px  = float(close.iloc[pos])
            a   = float(atr.iloc[pos])
            actions[pos] = SignalAction.SELL
            metas[pos]   = {"suggested_tp": px - tp_mult * a,
                             "suggested_sl": px + sl_mult * a,
                             "metadata": {"rsi": round(float(rsi.iloc[pos]), 2), "atr": round(a, 4)}}

        return actions, metas
