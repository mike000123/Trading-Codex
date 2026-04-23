"""Optional GLD candidate preset sets from constrained sweep research."""
from __future__ import annotations


# These are intentionally not wired in as defaults. They are stored here so we
# can test them explicitly from research runs or surface them later in the UI
# as named alternatives.
CANDIDATES: dict[str, dict[str, object]] = {
    "gld_best_1x_sweep_20260419": {
        "intraday_pullback_rsi_trigger": 85.0,
        "shock_reversal_tp_pct": 1.4,
        "shock_reversal_trail_pct": 0.5,
        "cascade_breakdown_tp_pct": 1.0,
        "shock_rebound_tp_pct": 2.2,
        "shock_rebound_trail_pct": 1.3,
        "spike_momentum_max": 6,
    },
    "gld_best_leverage_5x_20260419": {
        "leverage": 5.0,
        "risk_max_loss_pct_of_capital": 50.0,
        "intraday_pullback_rsi_trigger": 85.0,
        "shock_reversal_tp_pct": 1.4,
        "shock_reversal_trail_pct": 0.5,
        "cascade_breakdown_tp_pct": 1.0,
        "shock_rebound_tp_pct": 2.2,
        "shock_rebound_trail_pct": 1.3,
        "spike_momentum_max": 6,
    },
    "gld_best_leverage_5x_tuned_20260419": {
        "leverage": 5.0,
        "risk_max_loss_pct_of_capital": 50.0,
        "intraday_pullback_rsi_trigger": 85.0,
        "shock_reversal_tp_pct": 1.6,
        "shock_reversal_trail_pct": 0.4,
        "cascade_breakdown_tp_pct": 0.8,
        "shock_rebound_tp_pct": 2.2,
        "shock_rebound_trail_pct": 1.3,
        "shock_rebound_sl_pct": 0.8,
        "spike_momentum_max": 6,
    },
}


def get_candidate(name: str) -> dict[str, object]:
    return dict(CANDIDATES.get(name, {}))
