"""
Reusable building blocks for composing strategy behavior.

These helpers keep spike-specific math and gating logic out of the
strategy policy layer so that future strategies can reuse them.
"""

from .spike_event import EventShortSetup, build_event_short_setup, event_completion_target, event_short_ready
from .spike_momentum import spike_breakout_long_ready, spike_momentum_long_ready
from .pullback_short import intraday_pullback_short_ready
from .trend_long import trend_bias_long_ready

__all__ = [
    "EventShortSetup",
    "build_event_short_setup",
    "event_completion_target",
    "event_short_ready",
    "spike_breakout_long_ready",
    "spike_momentum_long_ready",
    "intraday_pullback_short_ready",
    "trend_bias_long_ready",
]
