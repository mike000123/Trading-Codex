"""
strategies/fixed_level_strategy.py
────────────────────────────────────
Fixed TP/SL Strategy:
  Always generates a BUY (Long) or SELL (Short) signal on every bar,
  using manually specified take-profit and stop-loss prices (or % offsets
  from entry). No indicator is computed — the signal fires once on the
  first bar and then HOLDs until the position closes.

This mirrors the Historical Simulator's "Given TP/SL" mode but lets you
run it through the Backtester and Strategy Lab with the same position-
management engine.

Params:
    direction        – "Long" or "Short"         (default "Long")
    tp_pct           – take-profit % from entry  (default 2.0)
    sl_pct           – stop-loss  % from entry   (default 1.0)
    signal_frequency – "first_bar" or "every_bar"
                       first_bar  = open one trade and hold (typical)
                       every_bar  = re-signal after each close (more trades)
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from core.models import Signal, SignalAction
from strategies.base import BaseStrategy, register_strategy


@register_strategy
class FixedLevelStrategy(BaseStrategy):
    strategy_id = "fixed_level"
    name = "Fixed TP/SL Levels"
    description = (
        "Enter Long or Short on every new bar with fixed % TP/SL offsets. "
        "Equivalent to the Historical Simulator but usable in the Backtester."
    )

    def default_params(self) -> dict[str, Any]:
        return {
            "direction": "Long",
            "tp_pct": 2.0,
            "sl_pct": 1.0,
            "signal_frequency": "first_bar",
        }

    def validate_params(self) -> list[str]:
        p = {**self.default_params(), **self.params}
        errors = []
        if p["direction"] not in ("Long", "Short"):
            errors.append("direction must be 'Long' or 'Short'.")
        if float(p["tp_pct"]) <= 0:
            errors.append("tp_pct must be > 0.")
        if float(p["sl_pct"]) <= 0:
            errors.append("sl_pct must be > 0.")
        if p["signal_frequency"] not in ("first_bar", "every_bar"):
            errors.append("signal_frequency must be 'first_bar' or 'every_bar'.")
        return errors

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Signal:
        p = {**self.default_params(), **self.params}
        direction: str = p["direction"]
        tp_pct: float = float(p["tp_pct"])
        sl_pct: float = float(p["sl_pct"])
        freq: str = p["signal_frequency"]

        if data.empty:
            return Signal(
                strategy_id=self.strategy_id, symbol=symbol,
                action=SignalAction.HOLD,
                metadata={"reason": "no_data"},
            )

        last_close = float(data["close"].iloc[-1])
        is_first_bar = len(data) == 1

        # first_bar mode: only signal on the very first bar
        if freq == "first_bar" and not is_first_bar:
            return Signal(
                strategy_id=self.strategy_id, symbol=symbol,
                action=SignalAction.HOLD,
                metadata={"reason": "waiting_for_position_to_close"},
            )

        # Compute TP / SL prices from last close
        if direction == "Long":
            action = SignalAction.BUY
            suggested_tp = last_close * (1 + tp_pct / 100)
            suggested_sl = last_close * (1 - sl_pct / 100)
        else:
            action = SignalAction.SELL
            suggested_tp = last_close * (1 - tp_pct / 100)
            suggested_sl = last_close * (1 + sl_pct / 100)

        return Signal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            action=action,
            confidence=1.0,
            suggested_tp=suggested_tp,
            suggested_sl=suggested_sl,
            metadata={
                "direction": direction,
                "tp_pct": tp_pct,
                "sl_pct": sl_pct,
                "entry_price": last_close,
                "tp_price": suggested_tp,
                "sl_price": suggested_sl,
            },
        )
