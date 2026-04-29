"""
strategies/macd_strategy.py
────────────────────────────
MACD Signal-Line Crossover:
  BUY  when MACD line crosses above signal line
  SELL when MACD line crosses below signal line

Params:
    fast_period   – default 12
    slow_period   – default 26
    signal_period – default 9
    tp_pct        – take-profit % (default 4.0)
    sl_pct        – stop-loss %   (default 2.0)
"""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from core.models import Signal, SignalAction
from strategies.base import BaseStrategy, register_strategy


@register_strategy
class MACDStrategy(BaseStrategy):
    strategy_id = "macd_crossover"
    name = "MACD Crossover"
    description = "Trade on MACD line crossing above/below the signal line."

    def default_params(self) -> dict[str, Any]:
        return {
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9,
            "tp_pct": 4.0,
            "sl_pct": 2.0,
        }

    def validate_params(self) -> list[str]:
        p = {**self.default_params(), **self.params}
        errors = []
        if p["fast_period"] >= p["slow_period"]:
            errors.append("fast_period must be less than slow_period.")
        return errors

    def min_warmup_bars(self, symbol=None, source=None, interval=None) -> int:
        p = {**self.default_params(), **self.params}
        return int(p["slow_period"]) + int(p["signal_period"]) + 10

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Signal:
        p = {**self.default_params(), **self.params}
        fast: int = int(p["fast_period"])
        slow: int = int(p["slow_period"])
        signal_period: int = int(p["signal_period"])
        tp_pct: float = float(p["tp_pct"])
        sl_pct: float = float(p["sl_pct"])

        min_bars = slow + signal_period + 1
        if len(data) < min_bars:
            return Signal(
                strategy_id=self.strategy_id, symbol=symbol,
                action=SignalAction.HOLD,
                metadata={"reason": "insufficient_data", "need": min_bars, "have": len(data)},
            )

        close = data["close"]
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line

        curr_macd = float(macd_line.iloc[-1])
        curr_signal = float(signal_line.iloc[-1])
        prev_macd = float(macd_line.iloc[-2])
        prev_signal = float(signal_line.iloc[-2])
        last_close = float(close.iloc[-1])

        action = SignalAction.HOLD
        suggested_tp: Optional[float] = None
        suggested_sl: Optional[float] = None

        if prev_macd <= prev_signal and curr_macd > curr_signal:
            action = SignalAction.BUY
            suggested_tp = last_close * (1 + tp_pct / 100)
            suggested_sl = last_close * (1 - sl_pct / 100)
        elif prev_macd >= prev_signal and curr_macd < curr_signal:
            action = SignalAction.SELL
            suggested_tp = last_close * (1 - tp_pct / 100)
            suggested_sl = last_close * (1 + sl_pct / 100)

        return Signal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            action=action,
            confidence=min(1.0, abs(float(histogram.iloc[-1])) / (abs(curr_signal) + 1e-9)),
            suggested_tp=suggested_tp,
            suggested_sl=suggested_sl,
            metadata={
                "macd": round(curr_macd, 6),
                "signal": round(curr_signal, 6),
                "histogram": round(float(histogram.iloc[-1]), 6),
            },
        )
