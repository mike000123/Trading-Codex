from __future__ import annotations


def rsi_flush_rebound_long_ready(
    *,
    current_bar_pct: float,
    current_rsi: float,
    min_drop_pct: float,
    rsi_trigger: float,
) -> bool:
    """
    Simple oversold flush-buy setup.

    This intentionally mirrors the user's research hypothesis rather than the
    stricter shock-rebound leg: buy the flush bar itself when the one-bar drop
    is already deep and RSI is extremely washed out.
    """
    return current_bar_pct <= -abs(min_drop_pct) and current_rsi <= rsi_trigger
