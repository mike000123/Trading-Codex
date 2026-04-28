from __future__ import annotations


def rsi_spike_fade_short_ready(
    *,
    current_bar_pct: float,
    current_rsi: float,
    min_rise_pct: float,
    rsi_trigger: float,
) -> bool:
    """
    Simple overbought spike-fade setup.

    This mirrors the flush-long hypothesis on the short side: sell the spike
    bar itself when the one-bar rise is already abrupt and RSI is extremely
    stretched, then look for a quick same-session fade.
    """
    return current_bar_pct >= abs(min_rise_pct) and current_rsi >= rsi_trigger
