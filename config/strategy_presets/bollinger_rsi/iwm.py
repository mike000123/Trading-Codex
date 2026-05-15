"""IWM preset for the Bollinger + RSI spike-aware strategy.

IWM did not respond well to the sparse trend-bias families. The best
current-engine fit was much simpler:

  - keep the baseline Bollinger long mean-reversion leg
  - disable the symmetric short leg
  - require a wider band touch before entering
  - use a slightly tighter stop band once the long fires

This keeps the strategy small and lets us reuse the generic framework instead
of inventing a bespoke small-cap module before the evidence is there.

Tuned on the local 1-minute Alpaca cache over
2024-04-04 -> 2026-04-02 with current-engine costs
(`spread 0.06% + slippage 0.02%`).
"""
from __future__ import annotations


PRESET: dict[str, object] = {
    "bb_std": 2.0,
    "sl_band_mult": 0.15,
    "normal_long_enabled": True,
    "normal_short_enabled": False,
}
