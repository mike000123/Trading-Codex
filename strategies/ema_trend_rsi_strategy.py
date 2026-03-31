"""
strategies/ema_trend_rsi_strategy.py
──────────────────────────────────────
EMA Trend Filter + RSI Pullback Strategy
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Purpose-built for GC=F (Gold Futures) intraday — 1-min / 5-min bars.

WHY RSI ALONE FAILS ON GOLD:
  Gold is a trending instrument. RSI can stay above 70 for weeks during
  strong rallies (e.g. 2023–2025 gold bull run). Treating RSI > 70 as
  "overbought/sell" fights the structural trend and produces consistent
  losers. The fix: use a trend filter FIRST, then RSI only for entry timing.

STRATEGY LOGIC:
  Step 1 — Trend regime (EMA filter):
    Bull regime: fast EMA > slow EMA AND price > trend EMA
    Bear regime: fast EMA < slow EMA AND price < trend EMA

  Step 2 — RSI pullback entry (only trade WITH the regime):
    In Bull regime: LONG when RSI dips below oversold and recovers
                    (pullback to oversold in uptrend = buy the dip)
    In Bear regime: SHORT when RSI spikes above overbought and falls
                    (rally to overbought in downtrend = sell the rip)

  Step 3 — ATR-based stops (adapt to gold's intraday volatility range)

RESEARCH BASIS:
  - 9/21 EMA crossover: industry standard for 1-min/5-min gold intraday
    (TradingView gold strategies, QuantVPS 2025, Capital.com 2025)
  - Trend EMA 50: acts as dynamic support/resistance in commodity trends
  - RSI < 40 in uptrend (not 30): more sensitive — catches pullbacks earlier
  - RSI > 60 in downtrend (not 70): mirrors oversold logic for shorts
  - ATR 1.5×/2.5× stops: validated on GC=F by multiple practitioner sources

PARAMS:
  fast_ema      – fast EMA period (default 9)
  slow_ema      – slow EMA period (default 21)
  trend_ema     – trend filter EMA (default 50; price must be on same side)
  rsi_period    – RSI period (default 9)
  rsi_bull_entry– RSI buy threshold IN uptrend (default 40; pull back to here)
  rsi_bear_entry– RSI sell threshold IN downtrend (default 60; rally to here)
  atr_period    – ATR period for stops (default 14)
  atr_sl_mult   – stop distance = atr_sl_mult × ATR (default 1.5)
  atr_tp_mult   – target distance = atr_tp_mult × ATR (default 2.5)
  require_ema_cross – if True, require fresh EMA cross to open (default False)
                      False = enter on any RSI signal while in the regime
"""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from core.models import Signal, SignalAction
from strategies.base import BaseStrategy, register_strategy


def _calc_rsi(series: pd.Series, period: int) -> pd.Series:
    """Wilder RSI — alpha=1/period, adjust=False."""
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
    tr   = pd.concat([hi - lo, (hi - prev).abs(), (lo - prev).abs()],
                     axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()


@register_strategy
class EMATrendRSIStrategy(BaseStrategy):
    strategy_id = "ema_trend_rsi"
    name        = "EMA Trend + RSI Pullback"
    description = (
        "Trend-following strategy purpose-built for GC=F gold intraday. "
        "Uses 9/21 EMA crossover + 50 EMA trend filter to define regime, "
        "then RSI pullback timing to enter WITH the trend. "
        "ATR-based adaptive stops. NEVER trades against the EMA trend."
    )

    def default_params(self) -> dict[str, Any]:
        return {
            "fast_ema":        9,
            "slow_ema":        21,
            "trend_ema":       50,    # price must be above (bull) / below (bear)
            "rsi_period":      9,
            "rsi_bull_entry":  40,    # buy when RSI dips to 40 in uptrend
            "rsi_bear_entry":  60,    # sell when RSI spikes to 60 in downtrend
            "atr_period":      14,
            "atr_sl_mult":     1.5,
            "atr_tp_mult":     2.5,
            "require_ema_cross": False,
        }

    def validate_params(self) -> list[str]:
        p      = {**self.default_params(), **self.params}
        errors = []
        if int(p["fast_ema"]) >= int(p["slow_ema"]):
            errors.append("fast_ema must be less than slow_ema.")
        if int(p["slow_ema"]) >= int(p["trend_ema"]):
            errors.append("slow_ema must be less than trend_ema.")
        if float(p["rsi_bull_entry"]) >= float(p["rsi_bear_entry"]):
            errors.append("rsi_bull_entry must be less than rsi_bear_entry.")
        if float(p["atr_sl_mult"]) <= 0:
            errors.append("atr_sl_mult must be > 0.")
        return errors

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Signal:
        p               = {**self.default_params(), **self.params}
        fast_period     = int(p["fast_ema"])
        slow_period     = int(p["slow_ema"])
        trend_period    = int(p["trend_ema"])
        rsi_period      = int(p["rsi_period"])
        rsi_bull        = float(p["rsi_bull_entry"])
        rsi_bear        = float(p["rsi_bear_entry"])
        atr_period      = int(p["atr_period"])
        sl_mult         = float(p["atr_sl_mult"])
        tp_mult         = float(p["atr_tp_mult"])
        require_cross   = bool(p["require_ema_cross"])

        min_bars = trend_period + rsi_period + 2
        if len(data) < min_bars:
            return Signal(strategy_id=self.strategy_id, symbol=symbol,
                          action=SignalAction.HOLD,
                          metadata={"reason": "insufficient_data",
                                    "have": len(data), "need": min_bars})

        close   = data["close"].astype(float)
        fast    = _calc_ema(close, fast_period)
        slow    = _calc_ema(close, slow_period)
        trend   = _calc_ema(close, trend_period)
        rsi     = _calc_rsi(close, rsi_period)
        atr     = _calc_atr(data, atr_period)

        curr_close = float(close.iloc[-1])
        curr_fast  = float(fast.iloc[-1])
        curr_slow  = float(slow.iloc[-1])
        prev_fast  = float(fast.iloc[-2])
        prev_slow  = float(slow.iloc[-2])
        curr_trend = float(trend.iloc[-1])
        curr_rsi   = float(rsi.iloc[-1])
        prev_rsi   = float(rsi.iloc[-2])
        curr_atr   = float(atr.iloc[-1])

        # ── Regime detection ───────────────────────────────────────────────
        ema_bull = curr_fast > curr_slow and curr_close > curr_trend
        ema_bear = curr_fast < curr_slow and curr_close < curr_trend

        # Optional: require a fresh EMA cross (stricter)
        if require_cross:
            fresh_bull_cross = prev_fast <= prev_slow and curr_fast > curr_slow
            fresh_bear_cross = prev_fast >= prev_slow and curr_fast < curr_slow
            ema_bull = ema_bull and fresh_bull_cross
            ema_bear = ema_bear and fresh_bear_cross

        # ── Entry signals — RSI pullback WITHIN the regime ─────────────────
        # Bull regime + RSI dips to/below bull entry level (buy the dip)
        rsi_bull_trigger = prev_rsi >= rsi_bull >= curr_rsi   # RSI crossed down to oversold
        # Also fire when RSI was below and now recovering (bottom confirmed)
        rsi_bull_recover = prev_rsi < rsi_bull and curr_rsi >= rsi_bull

        # Bear regime + RSI spikes to/above bear entry level (sell the rip)
        rsi_bear_trigger = prev_rsi <= rsi_bear <= curr_rsi
        rsi_bear_recover = prev_rsi > rsi_bear and curr_rsi <= rsi_bear

        action: SignalAction           = SignalAction.HOLD
        suggested_tp: Optional[float] = None
        suggested_sl: Optional[float] = None
        trigger                       = ""

        if ema_bull and (rsi_bull_trigger or rsi_bull_recover):
            action       = SignalAction.BUY
            suggested_tp = curr_close + tp_mult * curr_atr
            suggested_sl = curr_close - sl_mult * curr_atr
            trigger      = ("RSI dip to " + str(round(curr_rsi, 1))
                            + " in bull regime")

        elif ema_bear and (rsi_bear_trigger or rsi_bear_recover):
            action       = SignalAction.SELL
            suggested_tp = curr_close - tp_mult * curr_atr
            suggested_sl = curr_close + sl_mult * curr_atr
            trigger      = ("RSI spike to " + str(round(curr_rsi, 1))
                            + " in bear regime")

        ema_spread_pct = (curr_fast - curr_slow) / curr_slow * 100

        return Signal(
            strategy_id  = self.strategy_id,
            symbol       = symbol,
            action       = action,
            confidence   = min(1.0, abs(ema_spread_pct) / 2),
            suggested_tp = suggested_tp,
            suggested_sl = suggested_sl,
            metadata     = {
                "rsi":           round(curr_rsi, 2),
                "fast_ema":      round(curr_fast, 4),
                "slow_ema":      round(curr_slow, 4),
                "trend_ema":     round(curr_trend, 4),
                "atr":           round(curr_atr, 4),
                "ema_spread_%":  round(ema_spread_pct, 3),
                "regime":        ("bull" if ema_bull else "bear" if ema_bear else "neutral"),
                "trigger":       trigger,
                "sl_distance":   round(sl_mult * curr_atr, 4),
                "tp_distance":   round(tp_mult * curr_atr, 4),
                "implied_rr":    round(tp_mult / sl_mult, 2),
            },
        )
