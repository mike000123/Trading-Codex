"""
strategies/ma_crossover.py
──────────────────────────
Moving Average Crossover:
  BUY  when fast MA crosses above slow MA (golden cross)
  SELL when fast MA crosses below slow MA (death cross)

Params:
    fast_period   – short MA window (default 10)
    slow_period   – long  MA window (default 30)
    ma_type       – "sma" or "ema"   (default "ema")
    tp_pct        – take-profit %    (default 4.0)
    sl_pct        – stop-loss %      (default 2.0)
"""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from core.models import Signal, SignalAction
from strategies.base import BaseStrategy, register_strategy


@register_strategy
class MACrossoverStrategy(BaseStrategy):
    strategy_id = "ma_crossover"
    name = "MA Crossover"
    description = "Golden/death cross on two moving averages (EMA or SMA)."

    def default_params(self) -> dict[str, Any]:
        return {
            "fast_period": 10,
            "slow_period": 30,
            "ma_type": "ema",
            "tp_pct": 4.0,
            "sl_pct": 2.0,
        }

    def validate_params(self) -> list[str]:
        p = {**self.default_params(), **self.params}
        errors = []
        if p["fast_period"] >= p["slow_period"]:
            errors.append("fast_period must be less than slow_period.")
        if p["ma_type"] not in ("sma", "ema"):
            errors.append("ma_type must be 'sma' or 'ema'.")
        return errors

    def min_warmup_bars(self, symbol=None, source=None, interval=None) -> int:
        p = {**self.default_params(), **self.params}
        return int(p["slow_period"]) + 10

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Signal:
        p = {**self.default_params(), **self.params}
        fast: int = int(p["fast_period"])
        slow: int = int(p["slow_period"])
        ma_type: str = p["ma_type"]
        tp_pct: float = float(p["tp_pct"])
        sl_pct: float = float(p["sl_pct"])

        if len(data) < slow + 1:
            return Signal(
                strategy_id=self.strategy_id, symbol=symbol,
                action=SignalAction.HOLD,
                metadata={"reason": "insufficient_data"},
            )

        close = data["close"]
        if ma_type == "ema":
            fast_ma = close.ewm(span=fast, adjust=False).mean()
            slow_ma = close.ewm(span=slow, adjust=False).mean()
        else:
            fast_ma = close.rolling(fast).mean()
            slow_ma = close.rolling(slow).mean()

        curr_fast, curr_slow = float(fast_ma.iloc[-1]), float(slow_ma.iloc[-1])
        prev_fast, prev_slow = float(fast_ma.iloc[-2]), float(slow_ma.iloc[-2])
        last_close = float(close.iloc[-1])

        action = SignalAction.HOLD
        suggested_tp: Optional[float] = None
        suggested_sl: Optional[float] = None

        if prev_fast <= prev_slow and curr_fast > curr_slow:
            # Golden cross
            action = SignalAction.BUY
            suggested_tp = last_close * (1 + tp_pct / 100)
            suggested_sl = last_close * (1 - sl_pct / 100)
        elif prev_fast >= prev_slow and curr_fast < curr_slow:
            # Death cross
            action = SignalAction.SELL
            suggested_tp = last_close * (1 - tp_pct / 100)
            suggested_sl = last_close * (1 + sl_pct / 100)

        spread_pct = (curr_fast - curr_slow) / curr_slow * 100
        return Signal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            action=action,
            confidence=min(1.0, abs(spread_pct) / 5),
            suggested_tp=suggested_tp,
            suggested_sl=suggested_sl,
            metadata={
                "fast_ma": round(curr_fast, 4),
                "slow_ma": round(curr_slow, 4),
                "spread_pct": round(spread_pct, 3),
                "ma_type": ma_type,
            },
        )
