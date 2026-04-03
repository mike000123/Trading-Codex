"""
strategies/bollinger_rsi_strategy.py
──────────────────────────────────────
Bollinger Bands + RSI Mean Reversion Strategy
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Best suited for: UVXY (volatility mean reversion) and GC=F range days

Logic:
  LONG  when price touches/crosses lower Bollinger Band  AND  RSI oversold
  SHORT when price touches/crosses upper Bollinger Band  AND  RSI overbought
  Exit target: middle band (20-period SMA = mean reversion target)

Rationale (from research):
  UVXY is structurally mean-reverting due to VIX contango decay.
  After a volatility spike RSI spikes to 80-90; when it starts falling
  back below 75 while price touches upper BB = high-probability short.
  For gold on range days, BB squeeze → expansion is the key entry signal.
  Bollinger + RSI divergence is explicitly cited as a gold strategy by
  multiple practitioner sources (quantvps, cloudzy, capital.com 2025).

Stop-loss: beyond the outer BB (1.05× band distance from SMA).
Take-profit: middle band (SMA) for mean reversion.

Params:
  bb_period     – Bollinger Band SMA period (default 20)
  bb_std        – Standard deviations for bands (default 2.0)
  rsi_period    – RSI period (default 9)
  rsi_oversold  – RSI buy threshold (default 30)
  rsi_overbought– RSI sell threshold (default 70)
  sl_band_mult  – SL beyond outer band by this fraction (default 0.2 extra)
  require_cross – If True, require price to cross INTO band, not just touch
"""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from core.models import Signal, SignalAction
from strategies.base import BaseStrategy, register_strategy


def _calc_rsi(series: pd.Series, period: int) -> pd.Series:
    d = series.astype(float).diff()
    g = d.clip(lower=0.0)
    l = (-d).clip(lower=0.0)
    ag = g.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    al = l.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    return 100.0 - (100.0 / (1.0 + ag / al.replace(0.0, float("nan"))))


def _calc_bollinger(series: pd.Series, period: int, std_dev: float):
    sma   = series.rolling(period, min_periods=period).mean()
    std   = series.rolling(period, min_periods=period).std(ddof=1)
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, sma, lower, std


@register_strategy
class BollingerRSIStrategy(BaseStrategy):
    strategy_id = "bollinger_rsi"
    name        = "Bollinger Bands + RSI Mean Reversion"
    description = (
        "Long on lower-band touch + RSI oversold. Short on upper-band touch + RSI overbought. "
        "Target: middle band (mean reversion). Best for UVXY post-spike shorts and GC=F range days."
    )

    def default_params(self) -> dict[str, Any]:
        return {
            "bb_period":       20,
            "bb_std":          2.0,
            "rsi_period":      9,
            "rsi_oversold":    30,
            "rsi_overbought":  70,
            "sl_band_mult":    0.2,   # SL = outer_band ± 0.2 × band_width
            "require_cross":   False, # False = touch is enough; True = must cross band
        }

    def validate_params(self) -> list[str]:
        p      = {**self.default_params(), **self.params}
        errors = []
        if float(p["bb_std"]) <= 0:
            errors.append("bb_std must be > 0.")
        if float(p["rsi_oversold"]) >= float(p["rsi_overbought"]):
            errors.append("rsi_oversold must be less than rsi_overbought.")
        return errors

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Signal:
        p             = {**self.default_params(), **self.params}
        bb_period     = int(p["bb_period"])
        bb_std        = float(p["bb_std"])
        rsi_period    = int(p["rsi_period"])
        oversold      = float(p["rsi_oversold"])
        overbought    = float(p["rsi_overbought"])
        sl_band_mult  = float(p["sl_band_mult"])
        require_cross = bool(p["require_cross"])

        min_bars = max(bb_period, rsi_period) + 2
        if len(data) < min_bars:
            return Signal(strategy_id=self.strategy_id, symbol=symbol,
                          action=SignalAction.HOLD,
                          metadata={"reason": "insufficient_data"})

        close       = data["close"].astype(float)
        upper, sma, lower, std = _calc_bollinger(close, bb_period, bb_std)
        rsi         = _calc_rsi(close, rsi_period)

        curr_close  = float(close.iloc[-1])
        prev_close  = float(close.iloc[-2])
        curr_upper  = float(upper.iloc[-1])
        curr_lower  = float(lower.iloc[-1])
        curr_sma    = float(sma.iloc[-1])
        prev_upper  = float(upper.iloc[-2])
        prev_lower  = float(lower.iloc[-2])
        curr_rsi    = float(rsi.iloc[-1])
        curr_std    = float(std.iloc[-1])
        band_width  = curr_upper - curr_lower

        # Band position
        at_lower = curr_close <= curr_lower
        at_upper = curr_close >= curr_upper
        crossed_below = prev_close > prev_lower and curr_close <= curr_lower
        crossed_above = prev_close < prev_upper and curr_close >= curr_upper

        lower_condition = crossed_below if require_cross else at_lower
        upper_condition = crossed_above if require_cross else at_upper

        action: SignalAction           = SignalAction.HOLD
        suggested_tp: Optional[float] = None
        suggested_sl: Optional[float] = None
        trigger                       = ""
        pct_b = (curr_close - curr_lower) / (band_width + 1e-9)  # 0=lower, 1=upper

        # ── LONG: lower band + oversold RSI ──────────────────────────────────
        if lower_condition and curr_rsi <= overbought:
            action       = SignalAction.BUY
            suggested_tp = curr_sma                                  # mean reversion target
            suggested_sl = curr_lower - sl_band_mult * band_width    # beyond lower band
            trigger      = f"Lower BB touch (RSI {curr_rsi:.1f})"

        # ── SHORT: upper band + overbought RSI ───────────────────────────────
        elif upper_condition and curr_rsi >= oversold:
            action       = SignalAction.SELL
            suggested_tp = curr_sma                                  # mean reversion target
            suggested_sl = curr_upper + sl_band_mult * band_width   # beyond upper band
            trigger      = f"Upper BB touch (RSI {curr_rsi:.1f})"

        # Squeeze detection (bands contracting — breakout imminent)
        squeeze = band_width < close.rolling(bb_period).mean().iloc[-1] * 0.01

        return Signal(
            strategy_id  = self.strategy_id,
            symbol       = symbol,
            action       = action,
            confidence   = min(1.0, abs(curr_rsi - 50) / 50),
            suggested_tp = suggested_tp,
            suggested_sl = suggested_sl,
            metadata     = {
                "rsi":        round(curr_rsi, 2),
                "upper_band": round(curr_upper, 4),
                "lower_band": round(curr_lower, 4),
                "middle":     round(curr_sma, 4),
                "band_width": round(band_width, 4),
                "pct_b":      round(pct_b, 3),
                "squeeze":    bool(squeeze),
                "trigger":    trigger,
            },
        )
    def generate_signals_bulk(self, data: pd.DataFrame, symbol: str):
        """Vectorised bulk — computes BB + RSI once over full dataset."""
        p             = {**self.default_params(), **self.params}
        bb_period     = int(p["bb_period"])
        bb_std        = float(p["bb_std"])
        rsi_period    = int(p["rsi_period"])
        oversold      = float(p["rsi_oversold"])
        overbought    = float(p["rsi_overbought"])
        sl_band_mult  = float(p["sl_band_mult"])
        require_cross = bool(p["require_cross"])

        close              = data["close"].astype(float)
        upper, sma, lower, std = _calc_bollinger(close, bb_period, bb_std)
        rsi                = _calc_rsi(close, rsi_period)
        prev_close         = close.shift(1)
        prev_upper         = upper.shift(1)
        prev_lower         = lower.shift(1)
        band_width         = upper - lower

        if require_cross:
            long_cond  = (prev_close > prev_lower) & (close <= lower)
            short_cond = (prev_close < prev_upper) & (close >= upper)
        else:
            long_cond  = close <= lower
            short_cond = close >= upper

        long_signal  = long_cond  & (rsi <= overbought)
        short_signal = short_cond & (rsi >= oversold) & ~long_signal

        n       = len(data)
        actions = [SignalAction.HOLD] * n
        metas   = [{"suggested_tp": None, "suggested_sl": None, "metadata": {}}] * n

        for i in data.index[long_signal & rsi.notna()]:
            pos = data.index.get_loc(i)
            tp  = float(sma.iloc[pos])
            sl  = float(lower.iloc[pos]) - sl_band_mult * float(band_width.iloc[pos])
            actions[pos] = SignalAction.BUY
            metas[pos]   = {"suggested_tp": tp, "suggested_sl": sl,
                             "metadata": {"rsi": round(float(rsi.iloc[pos]), 2)}}

        for i in data.index[short_signal & rsi.notna()]:
            pos = data.index.get_loc(i)
            tp  = float(sma.iloc[pos])
            sl  = float(upper.iloc[pos]) + sl_band_mult * float(band_width.iloc[pos])
            actions[pos] = SignalAction.SELL
            metas[pos]   = {"suggested_tp": tp, "suggested_sl": sl,
                             "metadata": {"rsi": round(float(rsi.iloc[pos]), 2)}}

        return actions, metas
