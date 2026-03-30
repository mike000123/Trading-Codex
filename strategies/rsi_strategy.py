"""
strategies/rsi_strategy.py
──────────────────────────
RSI Multi-Threshold Strategy.

sl_pct and tp_pct are RAW PRICE MOVE percentages (not % of capital).
The RiskManager then caps the SL so the leveraged capital loss
never exceeds its configured max_loss_pct_of_capital setting.

Example with sl_pct=2, leverage=5:
  Price SL floor  = entry * (1 - 0.02) = entry * 0.98   (2% raw price drop)
  Leveraged loss  = 2% × 5 = 10% of capital
  RiskManager cap = tightens SL further if needed (e.g. max 20% capital loss
                    with lev=5 means max 4% price drop, so cap = entry * 0.96)

  The effective SL used in the backtest is always max(strategy SL, risk cap SL).

tp_pct = 0  → TP disabled. Trade exits on SL only.
"""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from core.models import Signal, SignalAction
from strategies.base import BaseStrategy, register_strategy


def _calc_rsi(series: pd.Series, period: int) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def _parse_levels(raw: Any) -> list[float]:
    if isinstance(raw, (int, float)):
        return [float(raw)]
    if isinstance(raw, (list, tuple)):
        return sorted(float(v) for v in raw)
    parts = str(raw).replace(";", ",").split(",")
    return sorted(float(p.strip()) for p in parts if p.strip())


@register_strategy
class RSIThresholdStrategy(BaseStrategy):
    strategy_id = "rsi_threshold"
    name        = "RSI Threshold"
    description = (
        "Buy when RSI crosses below any oversold level; sell when it crosses above "
        "any overbought level. Supports multiple threshold pairs. TP optional (0=off)."
    )

    def default_params(self) -> dict[str, Any]:
        return {
            "rsi_period":  9,     # 9 bars is more responsive for intraday (1-min/5-min)
            "buy_levels":  "30",
            "sell_levels": "70",
            "tp_pct":      3.0,   # raw price move %; 0 = disabled
            "sl_pct":      2.0,   # raw price move % — RiskManager caps this further
        }

    def validate_params(self) -> list[str]:
        p      = {**self.default_params(), **self.params}
        errors = []
        try:
            _parse_levels(p["buy_levels"])
            _parse_levels(p["sell_levels"])
        except Exception:
            errors.append("buy_levels and sell_levels must be comma-separated numbers.")
            return errors
        if int(p["rsi_period"]) < 2:
            errors.append("rsi_period must be >= 2.")
        if float(p["sl_pct"]) <= 0:
            errors.append("sl_pct must be > 0.")
        return errors

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Signal:
        p           = {**self.default_params(), **self.params}
        period      = int(p["rsi_period"])
        buy_levels  = _parse_levels(p["buy_levels"])
        sell_levels = _parse_levels(p["sell_levels"])
        tp_pct      = float(p["tp_pct"])
        sl_pct      = float(p["sl_pct"])

        if len(data) < period + 1:
            return Signal(strategy_id=self.strategy_id, symbol=symbol,
                          action=SignalAction.HOLD,
                          metadata={"reason": "insufficient_data"})

        rsi        = _calc_rsi(data["close"], period)
        curr_rsi   = float(rsi.iloc[-1])
        prev_rsi   = float(rsi.iloc[-2])
        last_close = float(data["close"].iloc[-1])

        action: SignalAction           = SignalAction.HOLD
        suggested_tp: Optional[float] = None
        suggested_sl: Optional[float] = None
        triggered_level: Optional[float] = None

        # BUY: RSI crossed below any buy_level
        for lvl in sorted(buy_levels, reverse=True):
            if prev_rsi >= lvl > curr_rsi:
                action          = SignalAction.BUY
                triggered_level = lvl
                # sl_pct is raw price drop % (RiskManager will further cap if needed)
                suggested_tp = last_close * (1 + tp_pct / 100) if tp_pct > 0 else None
                suggested_sl = last_close * (1 - sl_pct / 100)
                break

        # SELL: RSI crossed above any sell_level
        if action == SignalAction.HOLD:
            for lvl in sorted(sell_levels):
                if prev_rsi <= lvl < curr_rsi:
                    action          = SignalAction.SELL
                    triggered_level = lvl
                    suggested_tp = last_close * (1 - tp_pct / 100) if tp_pct > 0 else None
                    suggested_sl = last_close * (1 + sl_pct / 100)
                    break

        # Compute what the effective SL price implies in capital loss terms
        sl_price_move_pct = sl_pct  # raw price %; actual capital loss = sl_pct * leverage

        return Signal(
            strategy_id  = self.strategy_id,
            symbol       = symbol,
            action       = action,
            confidence   = min(1.0, abs(curr_rsi - 50) / 50),
            suggested_tp = suggested_tp,
            suggested_sl = suggested_sl,
            metadata     = {
                "rsi":                round(curr_rsi, 2),
                "prev_rsi":           round(prev_rsi, 2),
                "last_close":         last_close,
                "triggered_level":    triggered_level,
                "buy_levels":         buy_levels,
                "sell_levels":        sell_levels,
                "tp_disabled":        tp_pct == 0,
                "sl_price_move_pct":  sl_price_move_pct,
                "suggested_sl_price": round(suggested_sl, 4) if suggested_sl else None,
            },
        )
