"""
Reusable building blocks for composing strategy behavior.

These helpers keep spike-specific math and gating logic out of the
strategy policy layer so that future strategies can reuse them.
"""

from .spike_event import EventShortSetup, build_event_short_setup, event_completion_target, event_short_ready
from .spike_momentum import spike_breakout_long_ready, spike_momentum_long_ready
from .pullback_short import intraday_pullback_short_ready
from .shock_reversal_short import shock_reversal_short_ready
from .cascade_breakdown_short import cascade_breakdown_short_ready
from .fair_gap_fade_short import fair_gap_fade_short_ready
from .macro_bear_continuation_short import macro_bear_continuation_short_ready
from .shock_rebound_long import shock_rebound_long_ready
from .rsi_flush_rebound_long import rsi_flush_rebound_long_ready
from .rsi_spike_fade_short import rsi_spike_fade_short_ready
from .trend_long import trend_bias_long_ready, trend_context_ready, weighted_trend_context_score
from .gold_macro_regime import (
    directional_trend_state,
    gold_macro_regime_state,
    weighted_gold_macro_regime_score,
)

__all__ = [
    "EventShortSetup",
    "build_event_short_setup",
    "event_completion_target",
    "event_short_ready",
    "spike_breakout_long_ready",
    "spike_momentum_long_ready",
    "intraday_pullback_short_ready",
    "shock_reversal_short_ready",
    "cascade_breakdown_short_ready",
    "fair_gap_fade_short_ready",
    "macro_bear_continuation_short_ready",
    "shock_rebound_long_ready",
    "rsi_flush_rebound_long_ready",
    "rsi_spike_fade_short_ready",
    "trend_bias_long_ready",
    "weighted_trend_context_score",
    "trend_context_ready",
    "directional_trend_state",
    "weighted_gold_macro_regime_score",
    "gold_macro_regime_state",
]
