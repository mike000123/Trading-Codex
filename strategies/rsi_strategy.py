"""
strategies/rsi_strategy.py
──────────────────────────
RSI Multi-Threshold Strategy — Wilder's original smoothing.

RSI is calculated as a sliding window: every new bar shifts the window
forward by 1, so you get one RSI value per bar after the warm-up period.

buy_levels  – comma-separated RSI levels triggering LONG entry.
              Leave EMPTY (or "none") to disable Long entries entirely.
sell_levels – comma-separated RSI levels triggering SHORT entry.
              Leave EMPTY (or "none") to disable Short entries entirely.
tp_pct      – take-profit as raw price move %. Set 0 to disable.
sl_pct      – stop-loss as raw price move %. Required > 0 when levels are set.

Examples:
  buy_levels="30"         → Long when RSI < 30
  buy_levels="25, 30"     → Long when RSI crosses below either level
  buy_levels=""           → No Long entries (Short-only mode)
  sell_levels=""          → No Short entries (Long-only mode)
  buy_levels="" and sell_levels="" → No trades (disabled)
"""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from core.models import Signal, SignalAction
from strategies.base import BaseStrategy, register_strategy


def _calc_rsi(series: pd.Series, period: int) -> pd.Series:
    """
    Wilder RSI — true recursive EWM (alpha=1/period, adjust=False).
    Matches TradingView / Bloomberg / MetaTrader.
    Sliding window: one new RSI value per new bar after warm-up.
    """
    d        = series.astype(float).diff()
    gain     = d.clip(lower=0.0)
    loss     = (-d).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0.0, float("nan"))
    return 100.0 - (100.0 / (1.0 + rs))


def _parse_levels(raw: Any) -> list[float]:
    """
    Parse comma-separated level string → sorted list of floats.
    Returns empty list if raw is blank / 'none' / 'off' → that direction disabled.
    """
    if raw is None:
        return []
    if isinstance(raw, (int, float)):
        return [float(raw)]
    if isinstance(raw, (list, tuple)):
        return sorted(float(v) for v in raw if str(v).strip())
    s = str(raw).strip().lower()
    if s in ("", "none", "off", "0", "-"):
        return []
    parts = s.replace(";", ",").split(",")
    try:
        return sorted(float(p.strip()) for p in parts if p.strip())
    except ValueError:
        return []


@register_strategy
class RSIThresholdStrategy(BaseStrategy):
    strategy_id = "rsi_threshold"
    name        = "RSI Threshold"
    description = (
        "Long on RSI oversold cross, Short on overbought cross. "
        "Multiple levels supported. Leave buy/sell levels blank to disable that direction. "
        "TP optional (0=off). SL required when trades are enabled."
    )

    def default_params(self) -> dict[str, Any]:
        return {
            "rsi_period":  9,
            "buy_levels":  "30",   # blank / 'none' → no Long trades
            "sell_levels": "70",   # blank / 'none' → no Short trades
            "tp_pct":      3.0,    # 0 = disabled
            "sl_pct":      2.0,    # raw price move %
        }

    def validate_params(self) -> list[str]:
        p           = {**self.default_params(), **self.params}
        buy_levels  = _parse_levels(p["buy_levels"])
        sell_levels = _parse_levels(p["sell_levels"])
        errors      = []

        if not buy_levels and not sell_levels:
            errors.append(
                "Both buy_levels and sell_levels are empty — no trades will be generated. "
                "Set at least one to enable trading."
            )
        if int(p["rsi_period"]) < 2:
            errors.append("rsi_period must be >= 2.")
        if float(p["sl_pct"]) <= 0 and (buy_levels or sell_levels):
            errors.append("sl_pct must be > 0.")
        return errors

    def min_warmup_bars(self, symbol=None, source=None, interval=None) -> int:
        p = {**self.default_params(), **self.params}
        return int(p["rsi_period"]) + 10

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
                          metadata={"reason": "insufficient_data",
                                    "have": len(data), "need": period + 1})

        rsi        = _calc_rsi(data["close"], period)
        curr_rsi   = float(rsi.iloc[-1])
        prev_rsi   = float(rsi.iloc[-2])
        last_close = float(data["close"].iloc[-1])

        action: SignalAction           = SignalAction.HOLD
        suggested_tp: Optional[float] = None
        suggested_sl: Optional[float] = None
        triggered_level: Optional[float] = None

        # ── BUY: RSI crossed below any buy level (Long entry) ──────────────
        if buy_levels:
            for lvl in sorted(buy_levels, reverse=True):   # highest first
                if prev_rsi >= lvl > curr_rsi:
                    action          = SignalAction.BUY
                    triggered_level = lvl
                    suggested_tp    = last_close * (1 + tp_pct / 100) if tp_pct > 0 else None
                    suggested_sl    = last_close * (1 - sl_pct / 100)
                    break

        # ── SELL: RSI crossed above any sell level (Short entry) ────────────
        if action == SignalAction.HOLD and sell_levels:
            for lvl in sorted(sell_levels):                 # lowest first
                if prev_rsi <= lvl < curr_rsi:
                    action          = SignalAction.SELL
                    triggered_level = lvl
                    suggested_tp    = last_close * (1 - tp_pct / 100) if tp_pct > 0 else None
                    suggested_sl    = last_close * (1 + sl_pct / 100)
                    break

        return Signal(
            strategy_id  = self.strategy_id,
            symbol       = symbol,
            action       = action,
            confidence   = min(1.0, abs(curr_rsi - 50) / 50),
            suggested_tp = suggested_tp,
            suggested_sl = suggested_sl,
            metadata     = {
                "rsi":               round(curr_rsi, 2),
                "prev_rsi":          round(prev_rsi, 2),
                "last_close":        last_close,
                "triggered_level":   triggered_level,
                "buy_levels":        buy_levels,
                "sell_levels":       sell_levels,
                "buy_disabled":      not buy_levels,
                "sell_disabled":     not sell_levels,
                "tp_disabled":       tp_pct == 0,
                "sl_price":          round(suggested_sl, 4) if suggested_sl else None,
                "tp_price":          round(suggested_tp, 4) if suggested_tp else None,
            },
        )

    def generate_signals_bulk(self, data: pd.DataFrame, symbol: str):
        """
        Fully vectorised bulk signal generation — ~50x faster than bar-by-bar.
        Pre-computes RSI and crossings on the full dataset in one pandas pass.
        """
        import numpy as np
        p           = {**self.default_params(), **self.params}
        period      = int(p["rsi_period"])
        buy_levels  = _parse_levels(p["buy_levels"])
        sell_levels = _parse_levels(p["sell_levels"])
        tp_pct      = float(p["tp_pct"])
        sl_pct      = float(p["sl_pct"])

        close  = data["close"].astype(float)
        n      = len(data)

        # Compute RSI for all bars at once
        rsi = _calc_rsi(close, period)

        # Shift for "previous bar" comparison
        prev_rsi   = rsi.shift(1)
        prev_close = close.shift(1)

        # Initialise output lists
        actions = [SignalAction.HOLD] * n
        metas   = [{"suggested_tp": None, "suggested_sl": None, "metadata": {}}] * n

        # Vectorised crossings — for each level, find bars where RSI crossed
        # We compute all signals as boolean masks, then combine with priority

        buy_signal  = pd.Series(False, index=data.index)
        sell_signal = pd.Series(False, index=data.index)
        buy_trig    = pd.Series(np.nan, index=data.index)
        sell_trig   = pd.Series(np.nan, index=data.index)

        for lvl in sorted(buy_levels, reverse=True):
            crossed = (prev_rsi >= lvl) & (rsi < lvl)
            # Only update bars not already flagged by a higher-priority level
            new_buy = crossed & ~buy_signal
            buy_signal  = buy_signal  | new_buy
            buy_trig    = buy_trig.where(~new_buy, lvl)

        for lvl in sorted(sell_levels):
            crossed = (prev_rsi <= lvl) & (rsi >= lvl)
            new_sell = crossed & ~sell_signal & ~buy_signal  # buy takes priority
            sell_signal = sell_signal | new_sell
            sell_trig   = sell_trig.where(~new_sell, lvl)

        # Build output lists from masks
        buy_idx  = data.index[buy_signal  & rsi.notna() & prev_rsi.notna()]
        sell_idx = data.index[sell_signal & rsi.notna() & prev_rsi.notna()]

        for idx in buy_idx:
            i  = data.index.get_loc(idx)
            px = float(close.iloc[i])
            tp = px * (1 + tp_pct / 100) if tp_pct > 0 else None
            sl = px * (1 - sl_pct / 100)
            actions[i] = SignalAction.BUY
            metas[i]   = {
                "suggested_tp": tp,
                "suggested_sl": sl,
                "metadata": {"rsi": round(float(rsi.iloc[i]), 2),
                              "triggered_level": float(buy_trig.iloc[i])}
            }

        for idx in sell_idx:
            i  = data.index.get_loc(idx)
            px = float(close.iloc[i])
            tp = px * (1 - tp_pct / 100) if tp_pct > 0 else None
            sl = px * (1 + sl_pct / 100)
            actions[i] = SignalAction.SELL
            metas[i]   = {
                "suggested_tp": tp,
                "suggested_sl": sl,
                "metadata": {"rsi": round(float(rsi.iloc[i]), 2),
                              "triggered_level": float(sell_trig.iloc[i])}
            }

        return actions, metas
