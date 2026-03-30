"""
strategies/vwap_rsi_strategy.py
────────────────────────────────
VWAP + RSI Confluence Strategy
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Best suited for: GC=F (Gold Futures), intraday 1-min / 5-min bars

Logic:
  LONG  when price crosses above VWAP  AND  RSI crosses below oversold
  SHORT when price crosses below VWAP  AND  RSI crosses above overbought

Rationale (from research):
  VWAP is the institutional benchmark — price above = institutional bias bullish.
  RSI confirms momentum direction before entry.
  Combining both filters out the noise each indicator generates alone.
  Gold practitioners use VWAP as the primary intraday support/resistance level.

Stop-loss: ATR-based (1.5× ATR from entry) for adaptive volatility sizing.
Take-profit: ATR-based (2.5× ATR from entry) giving ~1.67 R:R ratio.

Params:
  rsi_period    – RSI lookback (default 9 for intraday)
  rsi_oversold  – RSI buy threshold (default 35 — slightly higher than 30
                  because VWAP confluence means we can be less extreme on RSI)
  rsi_overbought– RSI sell threshold (default 65)
  atr_period    – ATR lookback for stop/TP sizing (default 14)
  atr_sl_mult   – SL = entry ± atr_sl_mult × ATR (default 1.5)
  atr_tp_mult   – TP = entry ± atr_tp_mult × ATR (default 2.5)
  vwap_band_pct – % tolerance for "near VWAP" signal (default 0.0 = exact cross)
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from core.models import Signal, SignalAction
from strategies.base import BaseStrategy, register_strategy


def _calc_rsi(series: pd.Series, period: int) -> pd.Series:
    """Wilder RSI — alpha=1/period, adjust=False (industry standard)."""
    d = series.astype(float).diff()
    g = d.clip(lower=0.0)
    l = (-d).clip(lower=0.0)
    ag = g.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    al = l.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    return 100.0 - (100.0 / (1.0 + ag / al.replace(0.0, float("nan"))))


def _calc_vwap(data: pd.DataFrame) -> pd.Series:
    """
    Session VWAP: cumulative (price × volume) / cumulative volume.
    Uses typical price = (high + low + close) / 3.
    Resets at start of each calendar day.
    """
    typical = (data["high"] + data["low"] + data["close"]) / 3.0
    vol     = data["volume"].replace(0, np.nan).ffill().fillna(1.0)

    # Day-boundary reset via groupby on date
    date_group = data["date"].dt.date if hasattr(data["date"].iloc[0], "date") else \
                 pd.to_datetime(data["date"]).dt.date

    cum_pv  = (typical * vol).groupby(date_group).cumsum()
    cum_vol = vol.groupby(date_group).cumsum()
    return cum_pv / cum_vol


def _calc_atr(data: pd.DataFrame, period: int) -> pd.Series:
    """Average True Range — measures bar-by-bar volatility."""
    high  = data["high"].astype(float)
    low   = data["low"].astype(float)
    close = data["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()


@register_strategy
class VWAPRSIStrategy(BaseStrategy):
    strategy_id = "vwap_rsi"
    name        = "VWAP + RSI Confluence"
    description = (
        "Long when price crosses above VWAP and RSI is oversold. "
        "Short when price crosses below VWAP and RSI is overbought. "
        "ATR-based stops and targets. Best for GC=F intraday (1-min / 5-min)."
    )

    def default_params(self) -> dict[str, Any]:
        return {
            "rsi_period":     9,
            "rsi_oversold":   35,    # slightly relaxed — VWAP cross provides extra filter
            "rsi_overbought": 65,
            "atr_period":     14,
            "atr_sl_mult":    1.5,   # stop-loss = 1.5 × ATR from entry
            "atr_tp_mult":    2.5,   # take-profit = 2.5 × ATR from entry (~1.67 R:R)
        }

    def validate_params(self) -> list[str]:
        p      = {**self.default_params(), **self.params}
        errors = []
        if float(p["rsi_oversold"]) >= float(p["rsi_overbought"]):
            errors.append("rsi_oversold must be less than rsi_overbought.")
        if float(p["atr_sl_mult"]) <= 0:
            errors.append("atr_sl_mult must be > 0.")
        if float(p["atr_tp_mult"]) <= 0:
            errors.append("atr_tp_mult must be > 0.")
        return errors

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Signal:
        p           = {**self.default_params(), **self.params}
        rsi_period  = int(p["rsi_period"])
        oversold    = float(p["rsi_oversold"])
        overbought  = float(p["rsi_overbought"])
        atr_period  = int(p["atr_period"])
        sl_mult     = float(p["atr_sl_mult"])
        tp_mult     = float(p["atr_tp_mult"])

        min_bars = max(rsi_period, atr_period) + 2
        if len(data) < min_bars:
            return Signal(strategy_id=self.strategy_id, symbol=symbol,
                          action=SignalAction.HOLD,
                          metadata={"reason": "insufficient_data",
                                    "have": len(data), "need": min_bars})

        rsi    = _calc_rsi(data["close"], rsi_period)
        vwap   = _calc_vwap(data)
        atr    = _calc_atr(data, atr_period)

        curr_rsi   = float(rsi.iloc[-1])
        prev_rsi   = float(rsi.iloc[-2])
        curr_close = float(data["close"].iloc[-1])
        prev_close = float(data["close"].iloc[-2])
        curr_vwap  = float(vwap.iloc[-1])
        prev_vwap  = float(vwap.iloc[-2])
        curr_atr   = float(atr.iloc[-1])

        # VWAP cross direction
        price_crossed_above_vwap = prev_close <= prev_vwap and curr_close > curr_vwap
        price_crossed_below_vwap = prev_close >= prev_vwap and curr_close < curr_vwap

        # Also fire when already above/below VWAP + RSI crosses threshold
        price_above_vwap = curr_close > curr_vwap
        price_below_vwap = curr_close < curr_vwap
        rsi_crossed_oversold   = prev_rsi >= oversold   > curr_rsi
        rsi_crossed_overbought = prev_rsi <= overbought < curr_rsi

        action: SignalAction           = SignalAction.HOLD
        suggested_tp: Optional[float] = None
        suggested_sl: Optional[float] = None
        trigger                       = ""

        # ── LONG signal ──────────────────────────────────────────────────────
        # Condition A: price crosses above VWAP while RSI is oversold
        # Condition B: price above VWAP and RSI crosses into oversold (pullback then resume)
        if (price_crossed_above_vwap and curr_rsi < overbought) or \
           (price_above_vwap and rsi_crossed_oversold):
            action       = SignalAction.BUY
            suggested_tp = curr_close + tp_mult * curr_atr
            suggested_sl = curr_close - sl_mult * curr_atr
            trigger      = ("VWAP cross up" if price_crossed_above_vwap
                            else "RSI oversold above VWAP")

        # ── SHORT signal ─────────────────────────────────────────────────────
        elif (price_crossed_below_vwap and curr_rsi > oversold) or \
             (price_below_vwap and rsi_crossed_overbought):
            action       = SignalAction.SELL
            suggested_tp = curr_close - tp_mult * curr_atr
            suggested_sl = curr_close + sl_mult * curr_atr
            trigger      = ("VWAP cross down" if price_crossed_below_vwap
                            else "RSI overbought below VWAP")

        return Signal(
            strategy_id  = self.strategy_id,
            symbol       = symbol,
            action       = action,
            confidence   = min(1.0, abs(curr_rsi - 50) / 50),
            suggested_tp = suggested_tp,
            suggested_sl = suggested_sl,
            metadata     = {
                "rsi":           round(curr_rsi, 2),
                "vwap":          round(curr_vwap, 4),
                "atr":           round(curr_atr, 4),
                "close":         curr_close,
                "above_vwap":    price_above_vwap,
                "trigger":       trigger,
                "sl_distance":   round(sl_mult * curr_atr, 4),
                "tp_distance":   round(tp_mult * curr_atr, 4),
                "implied_rr":    round(tp_mult / sl_mult, 2),
            },
        )
