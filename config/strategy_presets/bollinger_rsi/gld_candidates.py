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
    "gld_rsi_spike_fade_short_20260426": {
        "rsi_spike_fade_short_enabled": True,
        "rsi_spike_rise_pct": 0.8,
        "rsi_spike_rsi_trigger": 88.0,
        "rsi_spike_sl_pct": 1.0,
        "rsi_spike_tp_pct": 0.5,
        "rsi_spike_cooldown": 0,
        "rsi_spike_require_red_reversal_bar": False,
        "rsi_spike_reversal_confirm_bars": 3,
        "rsi_spike_trend_filter_bars": 0,
    },
    "gld_fair_gap_fade_short_20260426": {
        "fair_gap_fade_short_enabled": True,
        "fair_gap_fade_gap_pct": 3.0,
        "fair_gap_fade_daily_rsi_trigger": 80.0,
        "fair_gap_fade_bar_drop_pct": 0.1,
        "fair_gap_fade_sl_pct": 1.0,
        "fair_gap_fade_tp_pct": 0.6,
        "fair_gap_fade_trail_pct": 0.6,
        "fair_gap_fade_cooldown": 180,
    },
    "gld_weak_0800_shock_reversal_filter_20260426": {
        "gld_weak_0800_filter_enabled": True,
        "gld_weak_0800_block_shock_rebound_long": False,
        "gld_weak_0800_block_shock_reversal_short": True,
    },
}

def get_candidate(name: str) -> dict[str, object]:
    return dict(CANDIDATES.get(name, {}))
