"""XLF preset for the Bollinger + RSI spike-aware strategy.

XLF fit the calmer edge of the USO family better than the broad-index trend
families. The winning profile is effectively the USO preset with one small but
useful exit tweak:

  - keep the same mixed trend / rebound / fade stack
  - lower the RSI-flush take-profit from 3.0% to 2.0%
  - let the shock-rebound take-profit run to 5.5% on the second pass

That earlier take-profit matched financial-sector bounce behavior better on the
current sample and improved total return, drawdown, and Sharpe without adding
new logic.

Tuned on the local 1-minute Alpaca cache over
2024-04-04 -> 2026-04-02 with current-engine costs
(`spread 0.06% + slippage 0.02%`).
"""
from __future__ import annotations


from .uso import PRESET as USO_PRESET


PRESET: dict[str, object] = {
    **dict(USO_PRESET),
    "rsi_flush_tp_pct": 2.0,
    "shock_rebound_tp_pct": 5.5,
}
