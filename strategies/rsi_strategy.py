"""
strategies/rsi_strategy.py
──────────────────────────
RSI Threshold Strategy:
  BUY  when RSI crosses below oversold level  (default 30)
  SELL when RSI crosses above overbought level (default 70)
  HOLD otherwise

Params:
    rsi_period    – lookback window (default 14)
    oversold      – RSI level triggering BUY  (default 30)
    overbought    – RSI level triggering SELL (default 70)
    tp_pct        – take-profit % from entry  (default 5.0)
    sl_pct        – stop-loss % from entry    (default 2.5)
"""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from core.models import Signal, SignalAction
from strategies.base import BaseStrategy, register_strategy


def _calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


@register_strategy
class RSIThresholdStrategy(BaseStrategy):
    strategy_id = "rsi_threshold"
    name = "RSI Threshold"
    description = "Buy on RSI oversold, sell on overbought. Classic momentum-reversal."

    def default_params(self) -> dict[str, Any]:
        return {
            "rsi_period": 14,
            "oversold": 30,
            "overbought": 70,
            "tp_pct": 5.0,
            "sl_pct": 2.5,
        }

    def validate_params(self) -> list[str]:
        errors = []
        p = {**self.default_params(), **self.params}
        if p["oversold"] >= p["overbought"]:
            errors.append("oversold must be less than overbought.")
        if p["rsi_period"] < 2:
            errors.append("rsi_period must be at least 2.")
        return errors

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Signal:
        p = {**self.default_params(), **self.params}
        period: int = int(p["rsi_period"])
        oversold: float = float(p["oversold"])
        overbought: float = float(p["overbought"])
        tp_pct: float = float(p["tp_pct"])
        sl_pct: float = float(p["sl_pct"])

        if len(data) < period + 1:
            return Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                action=SignalAction.HOLD,
                metadata={"reason": "insufficient_data"},
            )

        rsi = _calc_rsi(data["close"], period)
        current_rsi = float(rsi.iloc[-1])
        prev_rsi = float(rsi.iloc[-2])
        last_close = float(data["close"].iloc[-1])

        action = SignalAction.HOLD
        suggested_tp: Optional[float] = None
        suggested_sl: Optional[float] = None

        if prev_rsi >= oversold > current_rsi:
            # RSI crossed down through oversold → BUY signal
            action = SignalAction.BUY
            suggested_tp = last_close * (1 + tp_pct / 100)
            suggested_sl = last_close * (1 - sl_pct / 100)
        elif prev_rsi <= overbought < current_rsi:
            # RSI crossed up through overbought → SELL signal
            action = SignalAction.SELL
            suggested_tp = last_close * (1 - tp_pct / 100)
            suggested_sl = last_close * (1 + sl_pct / 100)

        return Signal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            action=action,
            confidence=min(1.0, abs(current_rsi - 50) / 50),
            suggested_tp=suggested_tp,
            suggested_sl=suggested_sl,
            metadata={
                "rsi": round(current_rsi, 2),
                "prev_rsi": round(prev_rsi, 2),
                "last_close": last_close,
            },
        )
