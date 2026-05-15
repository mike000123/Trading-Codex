"""
strategies/atr_rsi_strategy.py
───────────────────────────────
ATR-Adaptive RSI Strategy
━━━━━━━━━━━━━━━━━━━━━━━━━
Suited for: Both GC=F and UVXY — adapts stop/TP to current volatility

Logic:
  Same RSI threshold crossing as the base RSI strategy, but:
  - Stop-loss is 1.5× ATR (not fixed %) → adapts to volatility
  - Take-profit is 2× ATR (not fixed %) → adapts to volatility
  - ATR-based thresholds prevent stops being too tight in volatile
    UVXY sessions or too wide in calm GC=F sessions

Rationale (from research):
  ATR is explicitly recommended for gold (quantvps 2025): "place stop-loss
  1.5 to 2× ATR from entry". For UVXY, implied daily move ~8.8% makes fixed
  % stops unreliable — ATR adjusts dynamically to each volatility regime.
  This is the most robust single-indicator approach for both instruments.

Params:
  rsi_period    – RSI lookback (default 9)
  buy_levels    – RSI oversold levels, comma-separated (default "30")
  sell_levels   – RSI overbought levels, comma-separated (default "70")
  atr_period    – ATR lookback (default 14)
  atr_sl_mult   – SL distance = atr_sl_mult × ATR (default 1.5)
  atr_tp_mult   – TP distance = atr_tp_mult × ATR (default 2.5)
  tp_disabled   – If True, exit only on SL or counter-signal (default False)
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


def _calc_atr(data: pd.DataFrame, period: int) -> pd.Series:
    high  = data["high"].astype(float)
    low   = data["low"].astype(float)
    close = data["close"].astype(float)
    prev  = close.shift(1)
    tr    = pd.concat([high-low, (high-prev).abs(), (low-prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()


def _parse_levels(raw: Any) -> list[float]:
    if raw is None: return []
    if isinstance(raw, (int, float)): return [float(raw)]
    s = str(raw).strip().lower()
    if s in ("", "none", "off", "-"): return []
    parts = s.replace(";", ",").split(",")
    try:
        return sorted(float(p.strip()) for p in parts if p.strip())
    except ValueError:
        return []


@register_strategy
class ATRRSIStrategy(BaseStrategy):
    strategy_id = "atr_rsi"
    name        = "ATR-Adaptive RSI"
    description = (
        "RSI threshold crossings with ATR-based stop and take-profit distances. "
        "Adapts to volatility automatically — reliable for both GC=F and UVXY. "
        "ATR stops prevent being whipsawed in high-volatility regimes."
    )

    def default_params(self) -> dict[str, Any]:
        return {
            "rsi_period":   9,
            "buy_levels":   "30",
            "sell_levels":  "70",
            "atr_period":   14,
            "atr_sl_mult":  1.5,
            "atr_tp_mult":  2.5,
            "tp_disabled":  False,
        }

    def validate_params(self) -> list[str]:
        p           = {**self.default_params(), **self.params}
        buy_levels  = _parse_levels(p["buy_levels"])
        sell_levels = _parse_levels(p["sell_levels"])
        errors      = []
        if not buy_levels and not sell_levels:
            errors.append("Both buy_levels and sell_levels are empty — no trades will fire.")
        if float(p["atr_sl_mult"]) <= 0:
            errors.append("atr_sl_mult must be > 0.")
        return errors

    def min_warmup_bars(self, symbol=None, source=None, interval=None) -> int:
        p = {**self.default_params(), **self.params}
        return max(int(p["rsi_period"]), int(p["atr_period"])) + 10

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Signal:
        p           = {**self.default_params(), **self.params}
        rsi_period  = int(p["rsi_period"])
        buy_levels  = _parse_levels(p["buy_levels"])
        sell_levels = _parse_levels(p["sell_levels"])
        atr_period  = int(p["atr_period"])
        sl_mult     = float(p["atr_sl_mult"])
        tp_mult     = float(p["atr_tp_mult"])
        tp_disabled = bool(p["tp_disabled"])

        min_bars = max(rsi_period, atr_period) + 2
        if len(data) < min_bars:
            return Signal(strategy_id=self.strategy_id, symbol=symbol,
                          action=SignalAction.HOLD,
                          metadata={"reason": "insufficient_data"})

        rsi        = _calc_rsi(data["close"], rsi_period)
        atr        = _calc_atr(data, atr_period)
        curr_rsi   = float(rsi.iloc[-1])
        prev_rsi   = float(rsi.iloc[-2])
        curr_close = float(data["close"].iloc[-1])
        curr_atr   = float(atr.iloc[-1])

        action: SignalAction           = SignalAction.HOLD
        suggested_tp: Optional[float] = None
        suggested_sl: Optional[float] = None
        triggered_level: Optional[float] = None

        if buy_levels:
            for lvl in sorted(buy_levels, reverse=True):
                if prev_rsi >= lvl > curr_rsi:
                    action          = SignalAction.BUY
                    triggered_level = lvl
                    suggested_sl    = curr_close - sl_mult * curr_atr
                    suggested_tp    = None if tp_disabled else curr_close + tp_mult * curr_atr
                    break

        if action == SignalAction.HOLD and sell_levels:
            for lvl in sorted(sell_levels):
                if prev_rsi <= lvl < curr_rsi:
                    action          = SignalAction.SELL
                    triggered_level = lvl
                    suggested_sl    = curr_close + sl_mult * curr_atr
                    suggested_tp    = None if tp_disabled else curr_close - tp_mult * curr_atr
                    break

        return Signal(
            strategy_id  = self.strategy_id,
            symbol       = symbol,
            action       = action,
            confidence   = min(1.0, abs(curr_rsi - 50) / 50),
            suggested_tp = suggested_tp,
            suggested_sl = suggested_sl,
            metadata     = {
                "rsi":             round(curr_rsi, 2),
                "prev_rsi":        round(prev_rsi, 2),
                "atr":             round(curr_atr, 4),
                "close":           curr_close,
                "triggered_level": triggered_level,
                "sl_price":        round(suggested_sl, 4) if suggested_sl else None,
                "tp_price":        round(suggested_tp, 4) if suggested_tp else None,
                "sl_distance_atr": round(sl_mult, 2),
                "tp_distance_atr": round(tp_mult, 2),
                "implied_rr":      round(tp_mult / sl_mult, 2) if not tp_disabled else "TP off",
            },
        )
    def generate_signals_bulk(self, data: pd.DataFrame, symbol: str):
        """Vectorised bulk — computes ATR + RSI once over full dataset."""
        p           = {**self.default_params(), **self.params}
        rsi_period  = int(p["rsi_period"])
        buy_levels  = _parse_levels(p["buy_levels"])
        sell_levels = _parse_levels(p["sell_levels"])
        atr_period  = int(p["atr_period"])
        sl_mult     = float(p["atr_sl_mult"])
        tp_mult     = float(p["atr_tp_mult"])
        tp_disabled = bool(p["tp_disabled"])

        close    = data["close"].astype(float)
        rsi      = _calc_rsi(close, rsi_period)
        atr      = _calc_atr(data, atr_period)
        prev_rsi = rsi.shift(1)
        n        = len(data)
        actions  = [SignalAction.HOLD] * n
        metas    = [{"suggested_tp": None, "suggested_sl": None, "metadata": {}}] * n

        buy_mask  = pd.Series(False, index=data.index)
        sell_mask = pd.Series(False, index=data.index)

        for lvl in sorted(buy_levels, reverse=True):
            buy_mask  = buy_mask  | ((prev_rsi >= lvl) & (rsi < lvl))
        for lvl in sorted(sell_levels):
            sell_mask = sell_mask | ((prev_rsi <= lvl) & (rsi >= lvl))
        sell_mask = sell_mask & ~buy_mask

        for i in data.index[buy_mask & rsi.notna()]:
            pos = data.index.get_loc(i)
            px  = float(close.iloc[pos])
            a   = float(atr.iloc[pos])
            sl  = px - sl_mult * a
            tp  = None if tp_disabled else px + tp_mult * a
            actions[pos] = SignalAction.BUY
            metas[pos]   = {"suggested_tp": tp, "suggested_sl": sl,
                             "metadata": {"rsi": round(float(rsi.iloc[pos]), 2), "atr": round(a, 4)}}

        for i in data.index[sell_mask & rsi.notna()]:
            pos = data.index.get_loc(i)
            px  = float(close.iloc[pos])
            a   = float(atr.iloc[pos])
            sl  = px + sl_mult * a
            tp  = None if tp_disabled else px - tp_mult * a
            actions[pos] = SignalAction.SELL
            metas[pos]   = {"suggested_tp": tp, "suggested_sl": sl,
                             "metadata": {"rsi": round(float(rsi.iloc[pos]), 2), "atr": round(a, 4)}}

        return actions, metas
