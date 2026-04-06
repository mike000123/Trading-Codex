from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EventShortSetup:
    anchor_price: float = 0.0
    peak_price: float = 0.0
    rise_bars: int = 0
    peak_excess_pct: float = 0.0
    confirm_drop_pct: float = 0.0


def event_completion_target(spike_start: float, spike_peak: float, completion_pct: float) -> float:
    return spike_peak - (completion_pct / 100.0) * (spike_peak - spike_start)


def build_event_short_setup(
    close_arr: np.ndarray,
    spike_peak_pos: int,
    spike_peak: float,
    current_price: float,
    anchor_lookback: int,
    spike_start_price: float = 0.0,
    spike_start_pos: int = -1,
) -> EventShortSetup:
    if spike_peak_pos < 0 or spike_peak <= 0 or anchor_lookback < 1:
        return EventShortSetup()

    if spike_start_price > 0 and 0 <= spike_start_pos <= spike_peak_pos:
        anchor_price = float(spike_start_price)
        rise_bars = spike_peak_pos - spike_start_pos
    else:
        anchor_start = max(0, spike_peak_pos - anchor_lookback + 1)
        anchor_window = close_arr[anchor_start : spike_peak_pos + 1]
        if len(anchor_window) == 0:
            return EventShortSetup()

        anchor_rel = int(np.argmin(anchor_window))
        anchor_price = float(anchor_window[anchor_rel])
        rise_bars = spike_peak_pos - (anchor_start + anchor_rel)
    peak_excess_pct = ((spike_peak / max(anchor_price, 1e-9)) - 1.0) * 100 if anchor_price > 0 else 0.0
    confirm_drop_pct = ((spike_peak - current_price) / max(spike_peak, 1e-9)) * 100 if spike_peak > 0 else 0.0
    return EventShortSetup(
        anchor_price=anchor_price,
        peak_price=float(spike_peak),
        rise_bars=rise_bars,
        peak_excess_pct=peak_excess_pct,
        confirm_drop_pct=confirm_drop_pct,
    )


def event_short_ready(
    setup: EventShortSetup,
    *,
    min_peak_pct: float,
    max_rise_bars: int,
    confirm_drop_pct: float,
) -> bool:
    return (
        setup.anchor_price > 0
        and setup.peak_excess_pct >= min_peak_pct
        and setup.rise_bars <= max_rise_bars
        and setup.confirm_drop_pct >= confirm_drop_pct
    )
