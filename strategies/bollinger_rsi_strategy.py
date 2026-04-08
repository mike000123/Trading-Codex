"""
strategies/bollinger_rsi_strategy.py
──────────────────────────────────────
Bollinger + RSI — UVXY spike-aware mean reversion

Champion-base version with additive improvements:
  - Keep the stronger baseline behaviour
  - Remove the fragile first reversal short by default
  - Add generic decay bounce-failure continuation shorts
  - Classify spikes by structure inside the same strategy and suppress
    normal longs when slower persistent spikes begin rolling over
  - Optionally add an event-anchored short that targets a configured
    completion percentage of the spike unwind
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from config.strategy_presets.bollinger_rsi import (
    get_symbol_preset as get_bollinger_rsi_symbol_preset,
)
from core.models import Signal, SignalAction
from strategies.base import BaseStrategy, register_strategy
from strategies.components import (
    build_event_short_setup,
    event_completion_target,
    event_short_ready,
    intraday_pullback_short_ready,
    spike_breakout_long_ready,
    spike_momentum_long_ready,
)


def _calc_rsi(series: pd.Series, period: int) -> pd.Series:
    d = series.astype(float).diff()
    g = d.clip(lower=0.0)
    l = (-d).clip(lower=0.0)
    ag = g.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    al = l.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return 100.0 - (100.0 / (1.0 + ag / al.replace(0.0, float("nan"))))


def _calc_bollinger(series: pd.Series, period: int, std_dev: float):
    sma = series.rolling(period, min_periods=period).mean()
    std = series.rolling(period, min_periods=period).std(ddof=1)
    return sma + std_dev * std, sma, sma - std_dev * std, std


def _calc_atr(data: pd.DataFrame, period: int) -> pd.Series:
    hi = data["high"].astype(float)
    lo = data["low"].astype(float)
    cl = data["close"].astype(float)
    prev = cl.shift(1)
    tr = pd.concat([hi - lo, (hi - prev).abs(), (lo - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def bw_pct_ok(bw_pct: pd.Series, min_bw_pct: float) -> pd.Series:
    return bw_pct >= min_bw_pct


@register_strategy
class BollingerRSIStrategy(BaseStrategy):
    strategy_id = "bollinger_rsi"
    name = "Bollinger + RSI (Spike-Aware)"
    description = (
        "UVXY Bollinger/RSI strategy with explicit spike-to-decay handling. "
        "Dedicated spike longs, stricter reversal-confirmed shorts, and "
        "decay-only short re-entries."
    )

    def companion_contexts(
        self,
        symbol: str,
        source: str | None = None,
        interval: str | None = None,
    ) -> list[str]:
        if source not in {"alpaca", "yfinance", "forward_blend"}:
            return []
        params = self.resolve_params(symbol=symbol, source=source, interval=interval)
        contexts = ["equity_benchmark", "dollar_benchmark", "rates_benchmark"]
        if float(params.get("long_rates_weakness_weight", 0.0)) != 0.0:
            contexts.append("long_rates_benchmark")
        if bool(params.get("gold_peer_confirm_enabled", False)) or (
            bool(params.get("gold_context_assist_enabled", False))
            and float(params.get("gold_peer_strength_weight", 0.0)) != 0.0
        ):
            contexts.append("precious_metal_peer")
        if bool(params.get("gold_miners_confirm_enabled", False)) or (
            bool(params.get("gold_context_assist_enabled", False))
            and float(params.get("gold_miners_strength_weight", 0.0)) != 0.0
        ):
            contexts.append("miners_proxy")
        if bool(params.get("gold_riskoff_override_enabled", False)) or (
            bool(params.get("gold_context_assist_enabled", False))
            and float(params.get("gold_riskoff_strength_weight", 0.0)) != 0.0
        ):
            contexts.append("riskoff_proxy")
        return contexts

    def symbol_param_overrides(
        self,
        symbol: str,
        source: str | None = None,
        interval: str | None = None,
    ) -> dict[str, Any]:
        return get_bollinger_rsi_symbol_preset(symbol)

    def default_params(self) -> dict[str, Any]:
        return {
            "bb_period": 20,
            "bb_std": 1.8,
            "rsi_period": 9,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "sl_band_mult": 0.2,
            "require_cross": True,
            "min_band_width_pct": 2.0,
            "min_rr_ratio": 1.5,
            "cooldown_bars": 3,
            "min_atr_pct": 0.3,
            "normal_long_enabled": True,
            "normal_short_enabled": True,
            "intraday_pullback_short_enabled": False,
            "intraday_pullback_rsi_trigger": 80.0,
            "intraday_pullback_rsi_fade_pts": 8.0,
            "intraday_pullback_lookback_bars": 30,
            "intraday_pullback_drop_pct": 1.2,
            "intraday_pullback_min_atr_pct": 0.3,
            "intraday_pullback_sl_pct": 2.5,
            "intraday_pullback_tp_pct": 3.5,
            "intraday_pullback_trail_pct": 2.0,
            "intraday_pullback_cooldown": 30,
            "spike_gap_pct": 15.0,
            "grad_spike_lookback": 1560,
            "grad_spike_pct": 30.0,
            "grad_spike_rearm_bars": 3900,
            "rise_lookback": 1170,
            "rise_pct": 5.0,
            "spike_long_sl_pct": 20.0,
            "spike_long_trail_pct": 20.0,
            "spike_long_max": 1,
            "spike_long_cooldown": 2000,
            "spike_lockout_bars": 3900,
            "spike_long_min_rsi": 58.0,
            "spike_long_min_peak_pct": 18.0,
            "spike_long_min_atr_pct": 4.0,
            "spike_profile_shock_gap_pct": 18.0,
            "spike_profile_shock_peak_pct": 45.0,
            "spike_rollover_watch_bars": 195,
            "spike_rollover_watch_peak_pct": 20.0,
            "spike_rollover_fast_ema_tol": 1.01,
            "persistent_rollover_long_rsi_max": 24.0,
            "persistent_spread_block_pct": 2.5,
            "persistent_spread_deep_rsi_max": 24.0,
            "persistent_prepeak_block_peak_pct": 12.0,
            "persistent_prepeak_block_rsi_min": 55.0,
            "spike_atr_mult": 4.5,
            "spike_sl_pct": 8.0,
            "spike_atr_trail": 3.0,
            "spike_max_entries": 4,
            "spike_cooldown": 195,
            "spike_momentum_max": 2,
            "spike_momo_atr_mult": 1.2,
            "spike_momo_momentum_bars": 3,
            "spike_momo_momentum_pct": 0.5,
            "spike_momo_min_rsi": 45.0,
            "spike_momo_max_rsi": 98.0,
            "spike_momo_min_peak_pct": 4.0,
            "spike_momo_min_atr_pct": 0.6,
            "spike_momo_trail_pct": 8.0,
            "spike_momo_sl_pct": 3.0,
            "spike_momo_cooldown": 195,
            "psshort_drop_pct": 3.0,
            "psshort_sl_pct": 12.0,
            "psshort_trail_pct": 8.0,
            "psshort_max": 0,
            "event_target_short_enabled": True,
            "event_target_anchor_lookback": 3900,
            "event_target_max_rise_bars": 10000,
            "event_target_min_peak_pct": 100.0,
            "event_target_completion_pct": 90.0,
            "event_target_confirm_drop_pct": 15.0,
            "event_target_persistent_confirm_drop_pct": 25.0,
            "event_target_sl_pct": 12.0,
            "event_target_profit_giveback_frac": 0.50,
            "event_target_profit_giveback_min_pct": 5.0,
            "spy_selloff_assist_ret_30": -0.18,
            "spy_selloff_assist_ret_120": -0.35,
            "spy_rebound_block_ret_30": 0.16,
            "spy_rebound_block_ret_120": 0.05,
            "dollar_strength_block_ret_30": 999.0,
            "dollar_strength_block_ret_120": 999.0,
            "rates_weakness_block_ret_30": -999.0,
            "rates_weakness_block_ret_120": -999.0,
            "long_rates_weakness_block_ret_30": -999.0,
            "long_rates_weakness_block_ret_120": -999.0,
            "gold_macro_score_enabled": False,
            "gold_macro_block_score": 1.0,
            "dollar_strength_weight": 1.0,
            "rates_weakness_weight": 1.0,
            "long_rates_weakness_weight": 1.0,
            "gold_peer_confirm_enabled": False,
            "gold_peer_confirm_ret_30": 0.0,
            "gold_peer_confirm_ret_120": 0.0,
            "gold_miners_confirm_enabled": False,
            "gold_miners_confirm_ret_30": 0.0,
            "gold_miners_confirm_ret_120": 0.0,
            "gold_riskoff_override_enabled": False,
            "gold_riskoff_ret_30": 0.5,
            "gold_riskoff_ret_120": 0.0,
            "gold_context_assist_enabled": False,
            "gold_context_assist_min_score": 1.0,
            "gold_peer_strength_weight": 0.5,
            "gold_miners_strength_weight": 0.7,
            "gold_riskoff_strength_weight": 0.4,
            "psshort_cooldown": 390,
            "psshort_window": 4000,
            "spike_reversal_atr_frac": 0.85,
            "spike_reversal_ema_fast": 156,
            "spike_reversal_ema_slow": 780,
            "spike_reversal_min_bars": 390,
            "spike_reversal_min_peak_pct": 18.0,
            "decay_reentry_rsi": 52.0,
            "decay_bounce_min_pct": 2.0,
            "decay_bounce_fail_pct": 1.50,
            "decay_bounce_cooldown": 195,
            "decay_bounce_max": 4,
            "spike_high_window": 1170,
            "spike_ema_mult": 1.5,
            "spike_ema_span": 1950,
            "peak_drop_pct": 999.0,
            "reversion_tp_pct": 15.0,
            "reversion_sl_pct": 5.0,
            "reversion_rsi_min": 40,
            "decay_ema_period": 1950,
            "decay_slope_lb": 780,
            "decay_slope_min_pct": 999.0,
            "decay_atr_trail": 4.5,
            "decay_sl_pct": 12.0,
            "decay_cooldown": 780,
            "decay_max_entries": 12,
            "decay_floor": 44.0,
            "low_price_chop_price": 80.0,
            "low_price_chop_bandwidth_pct": 6.0,
        }

    def validate_params(self) -> list[str]:
        p = {**self.default_params(), **self.params}
        errors = []
        if float(p["bb_std"]) <= 0:
            errors.append("bb_std must be > 0.")
        if float(p["rsi_oversold"]) >= float(p["rsi_overbought"]):
            errors.append("rsi_oversold must be < rsi_overbought.")
        if float(p["intraday_pullback_rsi_trigger"]) <= float(p["rsi_overbought"]):
            errors.append("intraday_pullback_rsi_trigger must be > rsi_overbought.")
        if float(p["intraday_pullback_rsi_fade_pts"]) < 0:
            errors.append("intraday_pullback_rsi_fade_pts must be >= 0.")
        if int(p["intraday_pullback_lookback_bars"]) < 2:
            errors.append("intraday_pullback_lookback_bars must be >= 2.")
        if float(p["intraday_pullback_drop_pct"]) <= 0:
            errors.append("intraday_pullback_drop_pct must be > 0.")
        if float(p["intraday_pullback_min_atr_pct"]) <= 0:
            errors.append("intraday_pullback_min_atr_pct must be > 0.")
        if float(p["intraday_pullback_sl_pct"]) <= 0:
            errors.append("intraday_pullback_sl_pct must be > 0.")
        if float(p["intraday_pullback_tp_pct"]) <= 0:
            errors.append("intraday_pullback_tp_pct must be > 0.")
        if float(p["intraday_pullback_trail_pct"]) < 0:
            errors.append("intraday_pullback_trail_pct must be >= 0.")
        if int(p["intraday_pullback_cooldown"]) < 0:
            errors.append("intraday_pullback_cooldown must be >= 0.")
        if float(p["spike_reversal_atr_frac"]) <= 0 or float(p["spike_reversal_atr_frac"]) >= 1.5:
            errors.append("spike_reversal_atr_frac must be between 0 and 1.5.")
        if int(p["spike_reversal_min_bars"]) < 0:
            errors.append("spike_reversal_min_bars must be >= 0.")
        if int(p["spike_lockout_bars"]) < 0:
            errors.append("spike_lockout_bars must be >= 0.")
        if float(p["low_price_chop_price"]) <= 0:
            errors.append("low_price_chop_price must be > 0.")
        if int(p["spike_momentum_max"]) < 0:
            errors.append("spike_momentum_max must be >= 0.")
        if int(p["spike_momo_momentum_bars"]) < 1:
            errors.append("spike_momo_momentum_bars must be >= 1.")
        if float(p["spike_momo_min_atr_pct"]) <= 0:
            errors.append("spike_momo_min_atr_pct must be > 0.")
        if int(p["event_target_anchor_lookback"]) < 1:
            errors.append("event_target_anchor_lookback must be >= 1.")
        if int(p["event_target_max_rise_bars"]) < 1:
            errors.append("event_target_max_rise_bars must be >= 1.")
        if float(p["spike_momo_max_rsi"]) < float(p["spike_momo_min_rsi"]):
            errors.append("spike_momo_max_rsi must be >= spike_momo_min_rsi.")
        if float(p["event_target_confirm_drop_pct"]) <= 0:
            errors.append("event_target_confirm_drop_pct must be > 0.")
        if float(p["event_target_persistent_confirm_drop_pct"]) <= 0:
            errors.append("event_target_persistent_confirm_drop_pct must be > 0.")
        if float(p["event_target_completion_pct"]) <= 0 or float(p["event_target_completion_pct"]) >= 100:
            errors.append("event_target_completion_pct must be between 0 and 100.")
        if float(p["event_target_profit_giveback_frac"]) < 0 or float(p["event_target_profit_giveback_frac"]) >= 1:
            errors.append("event_target_profit_giveback_frac must be between 0 and 1.")
        if float(p["event_target_profit_giveback_min_pct"]) < 0:
            errors.append("event_target_profit_giveback_min_pct must be >= 0.")
        return errors

    def _compute_regimes_bulk(self, close: pd.Series, data: pd.DataFrame, p: dict):
        atr_s = _calc_atr(data, 14)
        atr_ma = atr_s.rolling(20, min_periods=5).mean()

        prev_cl = close.shift(1)
        gap_pct = float(p["spike_gap_pct"])
        bar_chg = (close - prev_cl) / prev_cl.replace(0, np.nan) * 100
        spike_gap_onset = bar_chg > gap_pct

        grad_lb = int(p["grad_spike_lookback"])
        grad_pct = float(p["grad_spike_pct"])
        grad_rearm_bars = int(p["grad_spike_rearm_bars"])
        grad_low = close.rolling(grad_lb, min_periods=1).min()
        grad_rise = (close / grad_low.replace(0, np.nan) - 1) * 100
        grad_rearmed = grad_rise.shift(1).rolling(grad_rearm_bars, min_periods=1).max() < grad_pct
        grad_spike_onset = (grad_rise >= grad_pct) & grad_rearmed
        any_spike_onset = spike_gap_onset | grad_spike_onset

        window = int(p["psshort_window"])
        spike_active = any_spike_onset.rolling(window, min_periods=1).sum() > 0
        in_spike_atr = (atr_s > float(p["spike_atr_mult"]) * atr_ma) | any_spike_onset

        rise_lb = int(p["rise_lookback"])
        rise_pct = float(p["rise_pct"])
        roll_min = close.rolling(rise_lb, min_periods=1).min()
        rising = (close / roll_min.replace(0, np.nan) - 1) * 100 > rise_pct
        suppress_shorts = in_spike_atr | rising | spike_active

        hw = int(p["spike_high_window"])
        ema_span = int(p["spike_ema_span"])
        ema_mult = float(p["spike_ema_mult"])
        drop_pct = float(p["peak_drop_pct"])
        high_nd = close.rolling(hw, min_periods=1).max()
        long_ema = close.ewm(span=ema_span, adjust=False).mean()
        spike_occ = high_nd > long_ema * ema_mult
        post_spike = spike_occ & (close < high_nd * (1 - drop_pct / 100))

        atr_pct_s = atr_s / close.replace(0, np.nan) * 100
        is_drift = atr_pct_s < float(p["min_atr_pct"])

        decay_ema = close.ewm(span=int(p["decay_ema_period"]), adjust=False).mean()
        ema_prev = decay_ema.shift(int(p["decay_slope_lb"]))
        ema_slope_pct = (decay_ema - ema_prev) / ema_prev.replace(0, np.nan) * 100
        in_decay = (ema_slope_pct < -float(p["decay_slope_min_pct"])) & (close < decay_ema)

        return in_spike_atr, suppress_shorts, post_spike, is_drift, in_decay, atr_s, any_spike_onset, spike_active

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Signal:
        p = self.resolve_params(symbol)
        min_bars = max(int(p["bb_period"]), int(p["rsi_period"]), int(p["spike_high_window"])) + 2
        if len(data) < min_bars:
            metadata: dict[str, Any] = {
                "reason": "insufficient_data",
                "verdict": SignalAction.HOLD.value,
                "verdict_reason": f"Insufficient warm-up bars: have {len(data)}, need {min_bars}",
            }
            gate_values: dict[str, Any] = {
                "bars_available": int(len(data)),
                "bars_required": int(min_bars),
            }
            if len(data) >= int(p["rsi_period"]) + 1 and "close" in data.columns:
                latest_rsi = _calc_rsi(data["close"], int(p["rsi_period"])).iloc[-1]
                if pd.notna(latest_rsi):
                    metadata["rsi"] = round(float(latest_rsi), 2)
                    gate_values["rsi"] = metadata["rsi"]
            if len(data) >= 2 and {"high", "low", "close"}.issubset(data.columns):
                latest_atr = _calc_atr(data, 14).iloc[-1]
                latest_close = float(data["close"].iloc[-1])
                if pd.notna(latest_atr) and latest_close > 0:
                    gate_values["atr_pct"] = round(float(latest_atr / latest_close * 100), 3)
                    gate_values["min_atr_pct"] = round(float(p["min_atr_pct"]), 3)
                    gate_values["atr_ok"] = bool(gate_values["atr_pct"] >= gate_values["min_atr_pct"])
            metadata["gate_values"] = gate_values
            metadata["gate_summary"] = ", ".join(f"{k}={v}" for k, v in gate_values.items())
            return Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                action=SignalAction.HOLD,
                metadata=metadata,
            )
        actions, metas = self.generate_signals_bulk(data, symbol, include_diagnostics=True)
        meta = metas[-1]
        metadata = dict(meta.get("metadata", {}) or {})
        if metadata.get("rsi") is None:
            latest_rsi = _calc_rsi(data["close"], int(p["rsi_period"])).iloc[-1]
            if pd.notna(latest_rsi):
                metadata["rsi"] = round(float(latest_rsi), 2)
        rsi_v = float(metadata.get("rsi", 50.0))
        return Signal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            action=actions[-1],
            confidence=min(1.0, abs(rsi_v - 50) / 50),
            suggested_tp=meta.get("suggested_tp"),
            suggested_sl=meta.get("suggested_sl"),
            metadata=metadata,
        )

    def generate_signals_bulk(self, data: pd.DataFrame, symbol: str, include_diagnostics: bool = False):  # noqa: C901
        p = self.resolve_params(symbol)

        bb_period = int(p["bb_period"])
        bb_std = float(p["bb_std"])
        rsi_period = int(p["rsi_period"])
        oversold = float(p["rsi_oversold"])
        overbought = float(p["rsi_overbought"])
        sl_band_mult = float(p["sl_band_mult"])
        require_cross = bool(p["require_cross"])
        min_bw_pct = float(p["min_band_width_pct"])
        min_rr = float(p["min_rr_ratio"])
        cooldown = int(p["cooldown_bars"])
        intraday_pullback_short_enabled = bool(p["intraday_pullback_short_enabled"])
        intraday_pullback_rsi_trigger = float(p["intraday_pullback_rsi_trigger"])
        intraday_pullback_rsi_fade_pts = float(p["intraday_pullback_rsi_fade_pts"])
        intraday_pullback_lookback_bars = int(p["intraday_pullback_lookback_bars"])
        intraday_pullback_drop_pct = float(p["intraday_pullback_drop_pct"])
        intraday_pullback_min_atr_pct = float(p["intraday_pullback_min_atr_pct"])
        intraday_pullback_sl_pct = float(p["intraday_pullback_sl_pct"])
        intraday_pullback_tp_pct = float(p["intraday_pullback_tp_pct"])
        intraday_pullback_trail_pct = float(p["intraday_pullback_trail_pct"])
        intraday_pullback_cooldown = int(p["intraday_pullback_cooldown"])

        rev_tp_pct = float(p["reversion_tp_pct"])
        rev_sl_pct = float(p["reversion_sl_pct"])
        rev_rsi_min = float(p["reversion_rsi_min"])

        spike_sl_pct = float(p["spike_sl_pct"])
        spike_atr_trail = float(p["spike_atr_trail"])
        spike_max_entries = int(p["spike_max_entries"])
        spike_cooldown = int(p["spike_cooldown"])

        splong_sl_pct = float(p["spike_long_sl_pct"])
        splong_trail_pct = float(p["spike_long_trail_pct"])
        splong_max = int(p["spike_long_max"])
        splong_cooldown = int(p["spike_long_cooldown"])
        spike_lockout_bars = int(p["spike_lockout_bars"])
        splong_min_rsi = float(p["spike_long_min_rsi"])
        splong_min_peak_pct = float(p["spike_long_min_peak_pct"])
        splong_min_atr_pct = float(p["spike_long_min_atr_pct"])
        shock_gap_pct = float(p["spike_profile_shock_gap_pct"])
        shock_peak_pct = float(p["spike_profile_shock_peak_pct"])
        rollover_watch_bars = int(p["spike_rollover_watch_bars"])
        rollover_watch_peak_pct = float(p["spike_rollover_watch_peak_pct"])
        rollover_fast_ema_tol = float(p["spike_rollover_fast_ema_tol"])
        persistent_rollover_long_rsi_max = float(p["persistent_rollover_long_rsi_max"])
        persistent_spread_block_pct = float(p["persistent_spread_block_pct"])
        persistent_spread_deep_rsi_max = float(p["persistent_spread_deep_rsi_max"])
        persistent_prepeak_block_peak_pct = float(p["persistent_prepeak_block_peak_pct"])
        persistent_prepeak_block_rsi_min = float(p["persistent_prepeak_block_rsi_min"])
        spike_momentum_max = int(p["spike_momentum_max"])
        spike_momo_atr_mult = float(p["spike_momo_atr_mult"])
        spike_momo_momentum_bars = int(p["spike_momo_momentum_bars"])
        spike_momo_momentum_pct = float(p["spike_momo_momentum_pct"])
        spike_momo_min_rsi = float(p["spike_momo_min_rsi"])
        spike_momo_max_rsi = float(p["spike_momo_max_rsi"])
        spike_momo_min_peak_pct = float(p["spike_momo_min_peak_pct"])
        spike_momo_min_atr_pct = float(p["spike_momo_min_atr_pct"])
        spike_momo_trail_pct = float(p["spike_momo_trail_pct"])
        spike_momo_sl_pct = float(p["spike_momo_sl_pct"])
        spike_momo_cooldown = int(p["spike_momo_cooldown"])

        psshort_drop_pct = float(p["psshort_drop_pct"])
        psshort_sl_pct = float(p["psshort_sl_pct"])
        psshort_trail_pct = float(p["psshort_trail_pct"])
        psshort_max = int(p["psshort_max"])
        event_target_short_enabled = bool(p["event_target_short_enabled"])
        event_target_anchor_lookback = int(p["event_target_anchor_lookback"])
        event_target_max_rise_bars = int(p["event_target_max_rise_bars"])
        event_target_min_peak_pct = float(p["event_target_min_peak_pct"])
        event_target_completion_pct = float(p["event_target_completion_pct"])
        event_target_confirm_drop_pct = float(p["event_target_confirm_drop_pct"])
        event_target_persistent_confirm_drop_pct = float(p["event_target_persistent_confirm_drop_pct"])
        event_target_sl_pct = float(p["event_target_sl_pct"])
        event_target_profit_giveback_frac = float(p["event_target_profit_giveback_frac"])
        event_target_profit_giveback_min_pct = float(p["event_target_profit_giveback_min_pct"])
        spy_selloff_assist_ret_30 = float(p["spy_selloff_assist_ret_30"])
        spy_selloff_assist_ret_120 = float(p["spy_selloff_assist_ret_120"])
        spy_rebound_block_ret_30 = float(p["spy_rebound_block_ret_30"])
        spy_rebound_block_ret_120 = float(p["spy_rebound_block_ret_120"])
        dollar_strength_block_ret_30 = float(p["dollar_strength_block_ret_30"])
        dollar_strength_block_ret_120 = float(p["dollar_strength_block_ret_120"])
        rates_weakness_block_ret_30 = float(p["rates_weakness_block_ret_30"])
        rates_weakness_block_ret_120 = float(p["rates_weakness_block_ret_120"])
        long_rates_weakness_block_ret_30 = float(p["long_rates_weakness_block_ret_30"])
        long_rates_weakness_block_ret_120 = float(p["long_rates_weakness_block_ret_120"])
        gold_macro_score_enabled = bool(p["gold_macro_score_enabled"])
        gold_macro_block_score = float(p["gold_macro_block_score"])
        dollar_strength_weight = float(p["dollar_strength_weight"])
        rates_weakness_weight = float(p["rates_weakness_weight"])
        long_rates_weakness_weight = float(p["long_rates_weakness_weight"])
        gold_peer_confirm_enabled = bool(p["gold_peer_confirm_enabled"])
        gold_peer_confirm_ret_30 = float(p["gold_peer_confirm_ret_30"])
        gold_peer_confirm_ret_120 = float(p["gold_peer_confirm_ret_120"])
        gold_miners_confirm_enabled = bool(p["gold_miners_confirm_enabled"])
        gold_miners_confirm_ret_30 = float(p["gold_miners_confirm_ret_30"])
        gold_miners_confirm_ret_120 = float(p["gold_miners_confirm_ret_120"])
        gold_riskoff_override_enabled = bool(p["gold_riskoff_override_enabled"])
        gold_riskoff_ret_30 = float(p["gold_riskoff_ret_30"])
        gold_riskoff_ret_120 = float(p["gold_riskoff_ret_120"])
        gold_context_assist_enabled = bool(p["gold_context_assist_enabled"])
        gold_context_assist_min_score = float(p["gold_context_assist_min_score"])
        gold_peer_strength_weight = float(p["gold_peer_strength_weight"])
        gold_miners_strength_weight = float(p["gold_miners_strength_weight"])
        gold_riskoff_strength_weight = float(p["gold_riskoff_strength_weight"])
        psshort_cooldown = int(p["psshort_cooldown"])
        reversal_atr_frac = float(p["spike_reversal_atr_frac"])
        reversal_min_bars = int(p["spike_reversal_min_bars"])
        reversal_min_peak_pct = float(p["spike_reversal_min_peak_pct"])
        decay_reentry_rsi = float(p["decay_reentry_rsi"])
        decay_bounce_min_pct = float(p["decay_bounce_min_pct"])
        decay_bounce_fail_pct = float(p["decay_bounce_fail_pct"])
        decay_bounce_cooldown = int(p["decay_bounce_cooldown"])
        decay_bounce_max = int(p["decay_bounce_max"])

        decay_atr_trail = float(p["decay_atr_trail"])
        decay_sl_pct = float(p["decay_sl_pct"])
        decay_cooldown = int(p["decay_cooldown"])
        decay_max_entries = int(p["decay_max_entries"])
        decay_floor = float(p["decay_floor"])
        low_price_chop_price = float(p["low_price_chop_price"])
        low_price_chop_bandwidth_pct = float(p["low_price_chop_bandwidth_pct"])

        close = data["close"].astype(float)
        upper, sma, lower, _ = _calc_bollinger(close, bb_period, bb_std)
        rsi = _calc_rsi(close, rsi_period)
        reversal_ema_fast = close.ewm(span=int(p["spike_reversal_ema_fast"]), adjust=False).mean()
        reversal_ema_slow = close.ewm(span=int(p["spike_reversal_ema_slow"]), adjust=False).mean()
        band_width = upper - lower
        bw_pct = band_width / close.replace(0, np.nan) * 100
        recent_max_rsi = rsi.rolling(intraday_pullback_lookback_bars, min_periods=1).max()
        recent_high = data["high"].astype(float).rolling(intraday_pullback_lookback_bars, min_periods=1).max()

        in_spike_atr, suppress_shorts, post_spike, is_drift, _in_decay_raw, atr_s, spike_onset, spike_active = (
            self._compute_regimes_bulk(close, data, p)
        )

        prev_close = close.shift(1)
        prev_upper = upper.shift(1)
        prev_lower = lower.shift(1)
        if require_cross:
            long_bb = (prev_close > prev_lower) & (close <= lower)
            short_bb = (prev_close < prev_upper) & (close >= upper)
        else:
            long_bb = close <= lower
            short_bb = close >= upper

        long_sl = lower - sl_band_mult * band_width
        long_sl_valid = long_sl < close
        long_tp_valid = sma > close
        long_tp_d = (sma - close).where(long_tp_valid, np.nan)
        long_sl_d = (close - long_sl).where(long_sl_valid, np.nan)
        long_rr_ok = ((long_tp_d / long_sl_d) >= min_rr).fillna(False)

        short_sl = upper + sl_band_mult * band_width
        short_sl_valid = short_sl > close
        short_tp_valid = sma < close
        short_tp_d = (close - sma).where(short_tp_valid, np.nan)
        short_sl_d = (short_sl - close).where(short_sl_valid, np.nan)
        short_rr_ok = ((short_tp_d / short_sl_d) >= min_rr).fillna(False)

        atr_ok = (atr_s / close.replace(0, np.nan) * 100) >= float(p["min_atr_pct"])
        long_data_ok = close > (lower * 0.5)

        spike_long_sig = (
            in_spike_atr & ~is_drift & ~post_spike
            & long_bb & (rsi <= overbought) & bw_pct_ok(bw_pct, min_bw_pct) & long_rr_ok & atr_ok & long_data_ok
        )
        normal_long_sig = (
            ~is_drift & ~post_spike
            & long_bb & (rsi <= overbought) & bw_pct_ok(bw_pct, min_bw_pct) & long_rr_ok & atr_ok & long_data_ok
        )
        normal_short_sig = (
            ~is_drift & ~post_spike & ~suppress_shorts & ~in_spike_atr
            & short_bb & (rsi >= oversold) & bw_pct_ok(bw_pct, min_bw_pct) & short_rr_ok & atr_ok
        )
        rev_sig = post_spike & ~is_drift & (rsi >= rev_rsi_min)

        n = len(data)
        actions = [SignalAction.HOLD] * n
        metas = [{"suggested_tp": None, "suggested_sl": None, "metadata": {}} for _ in range(n)]

        last_normal_bar = -cooldown - 1
        last_spike_bar = -spike_cooldown - 1
        last_splong_bar = -splong_cooldown - 1
        last_spike_momo_bar = -spike_momo_cooldown - 1
        last_intraday_pullback_bar = -intraday_pullback_cooldown - 1
        last_psshort_bar = -psshort_cooldown - 1
        last_event_target_bar = -psshort_cooldown - 1
        last_decay_bar = -decay_cooldown - 1
        last_decay_bounce_bar = -decay_bounce_cooldown - 1
        last_rev_bar = -cooldown * 10 - 1
        last_episode_end_bar = -spike_lockout_bars - 1

        spike_entries = 0
        splong_entries = 0
        spike_momo_entries = 0
        psshort_entries = 0
        event_target_entries = 0
        decay_entries = 0
        decay_bounce_entries = 0
        rev_entries = 0
        max_rev_entries = 2

        spike_peak = 0.0
        spike_peak_atr = 0.0
        spike_base = 0.0
        spike_start_price = 0.0
        spike_start_pos = -1
        event_base_price = 0.0
        event_base_pos = -1
        spike_peak_pos = -1
        episode_phase = "idle"
        episode_type = "neutral"
        was_spike_active = False
        prev_spike_onset = False
        decay_reentry_anchor = 0.0
        decay_bounce_high = 0.0
        decay_bounce_low = 0.0
        decay_bounce_armed = False

        close_arr = close.to_numpy()
        sma_arr = sma.to_numpy()
        sl_arr = long_sl.to_numpy()
        sh_sl_arr = short_sl.to_numpy()
        rsi_arr = rsi.to_numpy()
        atr_arr = atr_s.to_numpy()
        atr_ma_arr = atr_s.rolling(20, min_periods=5).mean().to_numpy()
        fast_arr = reversal_ema_fast.to_numpy()
        slow_arr = reversal_ema_slow.to_numpy()
        recent_max_rsi_arr = recent_max_rsi.to_numpy(dtype=float)
        recent_high_arr = recent_high.to_numpy(dtype=float)
        onset_arr = spike_onset.to_numpy()
        active_arr = spike_active.to_numpy()

        def _context_arrays(prefix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            close_col = f"{prefix}_close"
            if close_col not in data.columns:
                empty = np.full(n, np.nan)
                return empty, empty, empty, empty, empty
            context_close = pd.Series(data[close_col], dtype=float)
            context_fast = context_close.ewm(span=30, adjust=False).mean()
            context_slow = context_close.ewm(span=120, adjust=False).mean()
            context_ret_30 = context_close.pct_change(30, fill_method=None) * 100.0
            context_ret_120 = context_close.pct_change(120, fill_method=None) * 100.0
            return (
                context_close.to_numpy(dtype=float),
                context_fast.to_numpy(dtype=float),
                context_slow.to_numpy(dtype=float),
                context_ret_30.to_numpy(dtype=float),
                context_ret_120.to_numpy(dtype=float),
            )

        if "benchmark_close" not in data.columns:
            benchmark_close_arr = np.full(n, np.nan)
            benchmark_fast_arr = np.full(n, np.nan)
            benchmark_slow_arr = np.full(n, np.nan)
            benchmark_ret_30_arr = np.full(n, np.nan)
            benchmark_ret_120_arr = np.full(n, np.nan)
        else:
            benchmark_close = pd.Series(data["benchmark_close"], dtype=float)
            benchmark_fast = benchmark_close.ewm(span=30, adjust=False).mean()
            benchmark_slow = benchmark_close.ewm(span=120, adjust=False).mean()
            benchmark_ret_30 = benchmark_close.pct_change(30, fill_method=None) * 100.0
            benchmark_ret_120 = benchmark_close.pct_change(120, fill_method=None) * 100.0
            benchmark_close_arr = benchmark_close.to_numpy(dtype=float)
            benchmark_fast_arr = benchmark_fast.to_numpy(dtype=float)
            benchmark_slow_arr = benchmark_slow.to_numpy(dtype=float)
            benchmark_ret_30_arr = benchmark_ret_30.to_numpy(dtype=float)
            benchmark_ret_120_arr = benchmark_ret_120.to_numpy(dtype=float)
        dollar_close_arr, dollar_fast_arr, dollar_slow_arr, dollar_ret_30_arr, dollar_ret_120_arr = _context_arrays("dollar")
        rates_close_arr, rates_fast_arr, rates_slow_arr, rates_ret_30_arr, rates_ret_120_arr = _context_arrays("rates")
        long_rates_close_arr, long_rates_fast_arr, long_rates_slow_arr, long_rates_ret_30_arr, long_rates_ret_120_arr = _context_arrays("long_rates")
        metal_peer_close_arr, metal_peer_fast_arr, metal_peer_slow_arr, metal_peer_ret_30_arr, metal_peer_ret_120_arr = _context_arrays("metal_peer")
        miners_close_arr, miners_fast_arr, miners_slow_arr, miners_ret_30_arr, miners_ret_120_arr = _context_arrays("miners")
        riskoff_close_arr, riskoff_fast_arr, riskoff_slow_arr, riskoff_ret_30_arr, riskoff_ret_120_arr = _context_arrays("riskoff")

        def _uptrend_risk_score(
            close_value: float,
            fast_value: float,
            slow_value: float,
            ret_30_value: float,
            ret_120_value: float,
            ret_30_threshold: float,
            ret_120_threshold: float,
        ) -> float:
            if (
                np.isnan(close_value)
                or np.isnan(fast_value)
                or np.isnan(slow_value)
                or np.isnan(ret_30_value)
                or np.isnan(ret_120_value)
                or ret_30_threshold <= 0
                or ret_120_threshold <= 0
                or close_value < fast_value
                or fast_value < slow_value
                or ret_30_value < ret_30_threshold
                or ret_120_value < ret_120_threshold
            ):
                return 0.0
            ret_30_score = ret_30_value / max(ret_30_threshold, 1e-9)
            ret_120_score = ret_120_value / max(ret_120_threshold, 1e-9)
            return min(2.0, max(0.0, (ret_30_score + ret_120_score) / 2.0))

        def _downtrend_risk_score(
            close_value: float,
            fast_value: float,
            slow_value: float,
            ret_30_value: float,
            ret_120_value: float,
            ret_30_threshold: float,
            ret_120_threshold: float,
        ) -> float:
            if (
                np.isnan(close_value)
                or np.isnan(fast_value)
                or np.isnan(slow_value)
                or np.isnan(ret_30_value)
                or np.isnan(ret_120_value)
                or ret_30_threshold >= 0
                or ret_120_threshold >= 0
                or close_value > fast_value
                or fast_value > slow_value
                or ret_30_value > ret_30_threshold
                or ret_120_value > ret_120_threshold
            ):
                return 0.0
            ret_30_score = abs(ret_30_value) / max(abs(ret_30_threshold), 1e-9)
            ret_120_score = abs(ret_120_value) / max(abs(ret_120_threshold), 1e-9)
            return min(2.0, max(0.0, (ret_30_score + ret_120_score) / 2.0))

        def _uptrend_confirmation(
            close_value: float,
            fast_value: float,
            slow_value: float,
            ret_30_value: float,
            ret_120_value: float,
            ret_30_threshold: float,
            ret_120_threshold: float,
        ) -> bool:
            return (
                not np.isnan(close_value)
                and not np.isnan(fast_value)
                and not np.isnan(slow_value)
                and not np.isnan(ret_30_value)
                and not np.isnan(ret_120_value)
                and close_value >= fast_value
                and fast_value >= slow_value
                and ret_30_value >= ret_30_threshold
                and ret_120_value >= ret_120_threshold
            )

        for pos in range(n):
            px = close_arr[pos]
            prev_px = close_arr[pos - 1] if pos > 0 else px
            bar_pct = ((px / max(prev_px, 1e-9)) - 1.0) * 100 if pos > 0 else 0.0
            rsi_val = float(rsi_arr[pos]) if not np.isnan(rsi_arr[pos]) else 50.0
            atr_now = atr_arr[pos]
            atr_ma_now = atr_ma_arr[pos] if pos < len(atr_ma_arr) and not np.isnan(atr_ma_arr[pos]) else 0.0
            fast_now = fast_arr[pos]
            slow_now = slow_arr[pos]
            recent_max_rsi_now = recent_max_rsi_arr[pos] if not np.isnan(recent_max_rsi_arr[pos]) else rsi_val
            recent_high_now = recent_high_arr[pos] if not np.isnan(recent_high_arr[pos]) else px
            benchmark_close_now = benchmark_close_arr[pos]
            benchmark_fast_now = benchmark_fast_arr[pos]
            benchmark_slow_now = benchmark_slow_arr[pos]
            benchmark_ret_30_now = benchmark_ret_30_arr[pos]
            benchmark_ret_120_now = benchmark_ret_120_arr[pos]
            dollar_close_now = dollar_close_arr[pos]
            dollar_fast_now = dollar_fast_arr[pos]
            dollar_slow_now = dollar_slow_arr[pos]
            dollar_ret_30_now = dollar_ret_30_arr[pos]
            dollar_ret_120_now = dollar_ret_120_arr[pos]
            rates_close_now = rates_close_arr[pos]
            rates_fast_now = rates_fast_arr[pos]
            rates_slow_now = rates_slow_arr[pos]
            rates_ret_30_now = rates_ret_30_arr[pos]
            rates_ret_120_now = rates_ret_120_arr[pos]
            long_rates_close_now = long_rates_close_arr[pos]
            long_rates_fast_now = long_rates_fast_arr[pos]
            long_rates_slow_now = long_rates_slow_arr[pos]
            long_rates_ret_30_now = long_rates_ret_30_arr[pos]
            long_rates_ret_120_now = long_rates_ret_120_arr[pos]
            metal_peer_close_now = metal_peer_close_arr[pos]
            metal_peer_fast_now = metal_peer_fast_arr[pos]
            metal_peer_slow_now = metal_peer_slow_arr[pos]
            metal_peer_ret_30_now = metal_peer_ret_30_arr[pos]
            metal_peer_ret_120_now = metal_peer_ret_120_arr[pos]
            miners_close_now = miners_close_arr[pos]
            miners_fast_now = miners_fast_arr[pos]
            miners_slow_now = miners_slow_arr[pos]
            miners_ret_30_now = miners_ret_30_arr[pos]
            miners_ret_120_now = miners_ret_120_arr[pos]
            riskoff_close_now = riskoff_close_arr[pos]
            riskoff_fast_now = riskoff_fast_arr[pos]
            riskoff_slow_now = riskoff_slow_arr[pos]
            riskoff_ret_30_now = riskoff_ret_30_arr[pos]
            riskoff_ret_120_now = riskoff_ret_120_arr[pos]
            active_now = bool(active_arr[pos])
            onset_now = bool(onset_arr[pos])
            just_confirmed_decay = False
            atr_pct_now = (atr_now / max(px, 1e-9)) * 100 if px > 0 else 0.0
            drop_from_recent_high_pct = ((recent_high_now - px) / max(recent_high_now, 1e-9)) * 100 if recent_high_now > 0 else 0.0
            momentum_prev_pos = pos - spike_momo_momentum_bars
            momentum_prev_px = close_arr[momentum_prev_pos] if momentum_prev_pos >= 0 else px
            spike_momo_momentum_live = ((px / max(momentum_prev_px, 1e-9)) - 1.0) * 100 if momentum_prev_pos >= 0 else 0.0

            if onset_now and not prev_spike_onset:
                episode_phase = "spike"
                episode_type = "shock" if bar_pct >= shock_gap_pct else "persistent_trend"
                spike_peak = px
                spike_peak_atr = atr_now
                spike_base = px
                spike_start_price = px
                spike_start_pos = pos
                event_anchor_start = max(0, pos - event_target_anchor_lookback + 1)
                event_anchor_window = close_arr[event_anchor_start : pos + 1]
                if len(event_anchor_window) > 0:
                    event_anchor_rel = int(np.argmin(event_anchor_window))
                    event_base_pos = event_anchor_start + event_anchor_rel
                    event_base_price = float(event_anchor_window[event_anchor_rel])
                else:
                    event_base_pos = pos
                    event_base_price = px
                spike_peak_pos = pos
                splong_entries = 0
                spike_momo_entries = 0
                psshort_entries = 0
                decay_entries = 0
                decay_bounce_entries = 0
                decay_reentry_anchor = 0.0
                decay_bounce_high = 0.0
                decay_bounce_low = 0.0
                decay_bounce_armed = False
            prev_spike_onset = onset_now

            if not active_now and was_spike_active:
                episode_phase = "idle"
                episode_type = "neutral"
                spike_peak = 0.0
                spike_peak_atr = 0.0
                spike_base = 0.0
                spike_start_price = 0.0
                spike_start_pos = -1
                event_base_price = 0.0
                event_base_pos = -1
                spike_peak_pos = -1
                splong_entries = 0
                spike_momo_entries = 0
                psshort_entries = 0
                decay_entries = 0
                decay_bounce_entries = 0
                decay_reentry_anchor = 0.0
                decay_bounce_high = 0.0
                decay_bounce_low = 0.0
                decay_bounce_armed = False
                last_episode_end_bar = pos
            was_spike_active = active_now

            if active_now:
                prior_spike_peak = spike_peak
                if spike_base == 0.0:
                    spike_base = px
                if spike_start_price == 0.0:
                    spike_start_price = spike_base
                    spike_start_pos = pos
                if event_base_price == 0.0:
                    event_base_price = spike_start_price
                    event_base_pos = spike_start_pos
                peak_excess_live = ((spike_peak / max(spike_base, 1e-9)) - 1.0) * 100 if spike_base > 0 else 0.0
                if peak_excess_live >= shock_peak_pct:
                    episode_type = "shock"
                if px >= spike_peak:
                    spike_peak = px
                    spike_peak_atr = atr_now
                    spike_peak_pos = pos
                    if episode_phase == "decay":
                        episode_phase = "spike"
                        psshort_entries = 0
                        event_target_entries = 0
                        decay_entries = 0
                        decay_bounce_entries = 0
                        spike_momo_entries = 0
                        decay_reentry_anchor = 0.0
                        decay_bounce_high = 0.0
                        decay_bounce_low = 0.0
                        decay_bounce_armed = False
                made_new_peak = px > prior_spike_peak
            else:
                made_new_peak = False

            trend_bearish = (
                not np.isnan(fast_now)
                and not np.isnan(slow_now)
                and fast_now < slow_now
                and px < fast_now
                and px < slow_now
            )
            atr_cooled = spike_peak_atr > 0 and atr_now <= spike_peak_atr * reversal_atr_frac
            peak_age_ok = spike_peak_pos >= 0 and (pos - spike_peak_pos) >= reversal_min_bars
            peak_excess_pct = ((spike_peak / max(spike_base, 1e-9)) - 1.0) * 100 if spike_base > 0 else 0.0
            event_setup = build_event_short_setup(
                close_arr=close_arr,
                spike_peak_pos=spike_peak_pos,
                spike_peak=spike_peak,
                current_price=px,
                anchor_lookback=event_target_anchor_lookback,
                spike_start_price=event_base_price,
                spike_start_pos=event_base_pos,
            )
            peak_excess_ok = peak_excess_pct >= reversal_min_peak_pct
            in_spike_lockout = (pos - last_episode_end_bar) < spike_lockout_bars
            strong_spike_long = spike_momentum_long_ready(
                active_now=active_now,
                peak_excess_pct=peak_excess_pct,
                min_peak_pct=splong_min_peak_pct,
                rsi_value=rsi_val,
                min_rsi=splong_min_rsi,
                atr_value=atr_now,
                price=px,
                min_atr_pct=splong_min_atr_pct,
            )
            breakout_spike_long = spike_breakout_long_ready(
                active_now=active_now,
                episode_phase=episode_phase,
                in_spike_lockout=in_spike_lockout,
                atr_value=atr_now,
                atr_ma_value=atr_ma_now,
                atr_mult=spike_momo_atr_mult,
                momentum_pct=spike_momo_momentum_live,
                min_momentum_pct=spike_momo_momentum_pct,
                peak_excess_pct=peak_excess_pct,
                min_peak_pct=spike_momo_min_peak_pct,
                rsi_value=rsi_val,
                min_rsi=spike_momo_min_rsi,
                max_rsi=spike_momo_max_rsi,
                price=px,
                min_atr_pct=spike_momo_min_atr_pct,
            )
            spy_selloff_strong = (
                not np.isnan(benchmark_close_now)
                and not np.isnan(benchmark_fast_now)
                and not np.isnan(benchmark_slow_now)
                and not np.isnan(benchmark_ret_30_now)
                and not np.isnan(benchmark_ret_120_now)
                and benchmark_ret_30_now <= spy_selloff_assist_ret_30
                and benchmark_ret_120_now <= spy_selloff_assist_ret_120
                and benchmark_close_now <= benchmark_fast_now
                and benchmark_fast_now <= benchmark_slow_now
            )
            spy_rebound_risk = (
                not np.isnan(benchmark_close_now)
                and not np.isnan(benchmark_fast_now)
                and not np.isnan(benchmark_slow_now)
                and not np.isnan(benchmark_ret_30_now)
                and not np.isnan(benchmark_ret_120_now)
                and benchmark_ret_30_now >= spy_rebound_block_ret_30
                and benchmark_ret_120_now >= spy_rebound_block_ret_120
                and benchmark_close_now >= benchmark_fast_now
                and benchmark_fast_now >= benchmark_slow_now
            )
            dollar_strength_risk = (
                not np.isnan(dollar_close_now)
                and not np.isnan(dollar_fast_now)
                and not np.isnan(dollar_slow_now)
                and not np.isnan(dollar_ret_30_now)
                and not np.isnan(dollar_ret_120_now)
                and dollar_ret_30_now >= dollar_strength_block_ret_30
                and dollar_ret_120_now >= dollar_strength_block_ret_120
                and dollar_close_now >= dollar_fast_now
                and dollar_fast_now >= dollar_slow_now
            )
            rates_weakness_risk = (
                not np.isnan(rates_close_now)
                and not np.isnan(rates_fast_now)
                and not np.isnan(rates_slow_now)
                and not np.isnan(rates_ret_30_now)
                and not np.isnan(rates_ret_120_now)
                and rates_ret_30_now <= rates_weakness_block_ret_30
                and rates_ret_120_now <= rates_weakness_block_ret_120
                and rates_close_now <= rates_fast_now
                and rates_fast_now <= rates_slow_now
            )
            long_rates_weakness_risk = (
                not np.isnan(long_rates_close_now)
                and not np.isnan(long_rates_fast_now)
                and not np.isnan(long_rates_slow_now)
                and not np.isnan(long_rates_ret_30_now)
                and not np.isnan(long_rates_ret_120_now)
                and long_rates_ret_30_now <= long_rates_weakness_block_ret_30
                and long_rates_ret_120_now <= long_rates_weakness_block_ret_120
                and long_rates_close_now <= long_rates_fast_now
                and long_rates_fast_now <= long_rates_slow_now
            )
            dollar_strength_score = _uptrend_risk_score(
                dollar_close_now,
                dollar_fast_now,
                dollar_slow_now,
                dollar_ret_30_now,
                dollar_ret_120_now,
                dollar_strength_block_ret_30,
                dollar_strength_block_ret_120,
            )
            rates_weakness_score = _downtrend_risk_score(
                rates_close_now,
                rates_fast_now,
                rates_slow_now,
                rates_ret_30_now,
                rates_ret_120_now,
                rates_weakness_block_ret_30,
                rates_weakness_block_ret_120,
            )
            long_rates_weakness_score = _downtrend_risk_score(
                long_rates_close_now,
                long_rates_fast_now,
                long_rates_slow_now,
                long_rates_ret_30_now,
                long_rates_ret_120_now,
                long_rates_weakness_block_ret_30,
                long_rates_weakness_block_ret_120,
            )
            gold_macro_score = (
                dollar_strength_weight * dollar_strength_score
                + rates_weakness_weight * rates_weakness_score
                + long_rates_weakness_weight * long_rates_weakness_score
            )
            gold_macro_risk = (
                gold_macro_score >= gold_macro_block_score
                if gold_macro_score_enabled
                else dollar_strength_risk or rates_weakness_risk or long_rates_weakness_risk
            )
            metal_peer_confirm = _uptrend_confirmation(
                metal_peer_close_now,
                metal_peer_fast_now,
                metal_peer_slow_now,
                metal_peer_ret_30_now,
                metal_peer_ret_120_now,
                gold_peer_confirm_ret_30,
                gold_peer_confirm_ret_120,
            )
            miners_confirm = _uptrend_confirmation(
                miners_close_now,
                miners_fast_now,
                miners_slow_now,
                miners_ret_30_now,
                miners_ret_120_now,
                gold_miners_confirm_ret_30,
                gold_miners_confirm_ret_120,
            )
            riskoff_override = _uptrend_confirmation(
                riskoff_close_now,
                riskoff_fast_now,
                riskoff_slow_now,
                riskoff_ret_30_now,
                riskoff_ret_120_now,
                gold_riskoff_ret_30,
                gold_riskoff_ret_120,
            )
            if gold_riskoff_override_enabled and riskoff_override:
                gold_macro_risk = False
            gold_context_assist_score = (
                (gold_peer_strength_weight if metal_peer_confirm else 0.0)
                + (gold_miners_strength_weight if miners_confirm else 0.0)
                + (gold_riskoff_strength_weight if riskoff_override else 0.0)
            )
            spy_assisted_breakout = (
                active_now
                and episode_phase == "spike"
                and not in_spike_lockout
                and spy_selloff_strong
                and atr_ma_now > 0
                and atr_now >= atr_ma_now * max(1.0, spike_momo_atr_mult * 0.8)
                and spike_momo_momentum_live >= max(0.2, spike_momo_momentum_pct * 0.6)
                and peak_excess_pct >= max(2.0, spike_momo_min_peak_pct * 0.5)
                and spike_momo_min_rsi <= rsi_val <= spike_momo_max_rsi
                and ((atr_now / max(px, 1e-9)) * 100) >= max(0.4, spike_momo_min_atr_pct * 0.75)
            )
            gold_context_assisted_breakout = (
                gold_context_assist_enabled
                and active_now
                and episode_phase == "spike"
                and not in_spike_lockout
                and gold_context_assist_score >= gold_context_assist_min_score
                and not spy_rebound_risk
                and not gold_macro_risk
                and atr_ma_now > 0
                and atr_now >= atr_ma_now * max(0.6, spike_momo_atr_mult * 0.65)
                and spike_momo_momentum_live >= max(0.05, spike_momo_momentum_pct * 0.5)
                and peak_excess_pct >= max(0.2, spike_momo_min_peak_pct * 0.5)
                and spike_momo_min_rsi <= rsi_val <= spike_momo_max_rsi
                and ((atr_now / max(px, 1e-9)) * 100) >= max(0.01, spike_momo_min_atr_pct * 0.5)
            )
            if spy_rebound_risk or gold_macro_risk:
                breakout_spike_long = False
            if gold_peer_confirm_enabled and not metal_peer_confirm:
                breakout_spike_long = False
            if gold_miners_confirm_enabled and not miners_confirm:
                breakout_spike_long = False
            fast_slow_spread_pct = ((fast_now / slow_now) - 1.0) * 100 if (not np.isnan(fast_now) and not np.isnan(slow_now) and slow_now != 0) else 0.0
            drawdown_from_peak_pct = ((spike_peak / max(px, 1e-9)) - 1.0) * 100 if spike_peak > 0 else 0.0
            persistent_rollover = (
                active_now
                and episode_type != "shock"
                and peak_excess_pct >= rollover_watch_peak_pct
                and spike_peak_pos >= 0
                and (pos - spike_peak_pos) >= rollover_watch_bars
                and not np.isnan(fast_now)
                and px <= fast_now * rollover_fast_ema_tol
                and trend_bearish
            )
            if active_now and episode_type != "shock":
                episode_type = "persistent_rollover" if persistent_rollover else "persistent_trend"
            event_confirm_drop_req = (
                event_target_confirm_drop_pct
                if episode_type == "shock"
                else max(event_target_confirm_drop_pct, event_target_persistent_confirm_drop_pct)
            )
            persistent_rebound_trap = (
                active_now
                and episode_type == "persistent_trend"
                and episode_phase == "spike"
                and peak_excess_pct >= persistent_prepeak_block_peak_pct
                and drawdown_from_peak_pct >= 6.0
            )
            reversal_hit = (
                active_now
                and spike_peak > 0
                and px <= spike_peak * (1 - psshort_drop_pct / 100)
                and trend_bearish
                and atr_cooled
                and peak_age_ok
                and peak_excess_ok
            )
            if reversal_hit and episode_phase != "decay":
                episode_phase = "decay"
                psshort_entries = 0
                event_target_entries = 0
                decay_entries = 0
                decay_bounce_entries = 0
                decay_reentry_anchor = px
                decay_bounce_high = px
                decay_bounce_low = px
                decay_bounce_armed = False
                just_confirmed_decay = True

            event_drop_hit = (
                active_now
                and spike_peak > 0
                and spike_peak_pos >= 0
                and pos > spike_peak_pos
                and event_setup.confirm_drop_pct >= event_confirm_drop_req
                and trend_bearish
            )

            if (
                event_drop_hit
                and event_target_short_enabled
                and event_short_ready(
                    event_setup,
                    min_peak_pct=event_target_min_peak_pct,
                    max_rise_bars=event_target_max_rise_bars,
                    confirm_drop_pct=event_confirm_drop_req,
                )
                and event_target_entries < 1
                and (pos - last_event_target_bar) >= psshort_cooldown
                and event_setup.anchor_price > 0
            ):
                event_tp = event_completion_target(event_setup.anchor_price, event_setup.peak_price, event_target_completion_pct)
                event_sl = px * (1 + event_target_sl_pct / 100)
                if event_tp < px and event_sl > px:
                    actions[pos] = SignalAction.SELL
                    metas[pos] = {
                        "suggested_tp": event_tp,
                        "suggested_sl": event_sl,
                        "metadata": {
                            "rsi": round(rsi_val, 2),
                            "regime": "event_target_short",
                            "spike_type": episode_type,
                            "peak_excess_pct": round(event_setup.peak_excess_pct, 2),
                            "event_start_price": round(event_setup.anchor_price, 4),
                            "event_peak_price": round(event_setup.peak_price, 4),
                            "event_target_price": round(event_tp, 4),
                            "event_completion_pct": round(event_target_completion_pct, 2),
                            "event_confirm_drop_req": round(event_confirm_drop_req, 4),
                            "profit_giveback_frac": round(event_target_profit_giveback_frac, 4),
                            "profit_giveback_min_pct": round(event_target_profit_giveback_min_pct, 4),
                        },
                    }
                    event_target_entries += 1
                    last_event_target_bar = pos
                    continue

            if breakout_spike_long and spike_momentum_max > 0:
                if persistent_rebound_trap and not made_new_peak:
                    breakout_spike_long = False
            if (breakout_spike_long or spy_assisted_breakout or gold_context_assisted_breakout) and spike_momentum_max > 0:
                if spike_momo_entries < spike_momentum_max and (pos - last_spike_momo_bar) >= spike_momo_cooldown:
                    actions[pos] = SignalAction.BUY
                    metas[pos] = {
                        "suggested_tp": None,
                        "suggested_sl": px * (1 - spike_momo_sl_pct / 100),
                        "metadata": {
                            "rsi": round(rsi_val, 2),
                            "regime": "spike_momentum_long",
                            "spike_type": episode_type,
                            "pct_trail": spike_momo_trail_pct,
                            "peak_excess_pct": round(peak_excess_pct, 2),
                            "bar_pct": round(bar_pct, 2),
                            "benchmark_ret_30": round(float(benchmark_ret_30_now), 3) if not np.isnan(benchmark_ret_30_now) else None,
                            "benchmark_ret_120": round(float(benchmark_ret_120_now), 3) if not np.isnan(benchmark_ret_120_now) else None,
                            "dollar_ret_30": round(float(dollar_ret_30_now), 3) if not np.isnan(dollar_ret_30_now) else None,
                            "dollar_ret_120": round(float(dollar_ret_120_now), 3) if not np.isnan(dollar_ret_120_now) else None,
                            "rates_ret_30": round(float(rates_ret_30_now), 3) if not np.isnan(rates_ret_30_now) else None,
                            "rates_ret_120": round(float(rates_ret_120_now), 3) if not np.isnan(rates_ret_120_now) else None,
                            "long_rates_ret_30": round(float(long_rates_ret_30_now), 3) if not np.isnan(long_rates_ret_30_now) else None,
                            "long_rates_ret_120": round(float(long_rates_ret_120_now), 3) if not np.isnan(long_rates_ret_120_now) else None,
                            "metal_peer_ret_30": round(float(metal_peer_ret_30_now), 3) if not np.isnan(metal_peer_ret_30_now) else None,
                            "metal_peer_ret_120": round(float(metal_peer_ret_120_now), 3) if not np.isnan(metal_peer_ret_120_now) else None,
                            "miners_ret_30": round(float(miners_ret_30_now), 3) if not np.isnan(miners_ret_30_now) else None,
                            "miners_ret_120": round(float(miners_ret_120_now), 3) if not np.isnan(miners_ret_120_now) else None,
                            "riskoff_ret_30": round(float(riskoff_ret_30_now), 3) if not np.isnan(riskoff_ret_30_now) else None,
                            "riskoff_ret_120": round(float(riskoff_ret_120_now), 3) if not np.isnan(riskoff_ret_120_now) else None,
                            "gold_macro_score": round(float(gold_macro_score), 3),
                            "gold_context_assist_score": round(float(gold_context_assist_score), 3),
                            "metal_peer_confirm": bool(metal_peer_confirm),
                            "miners_confirm": bool(miners_confirm),
                            "riskoff_override": bool(riskoff_override),
                            "spy_assisted": bool(spy_assisted_breakout and not breakout_spike_long),
                            "gold_context_assisted": bool(gold_context_assisted_breakout and not breakout_spike_long),
                        },
                    }
                    spike_momo_entries += 1
                    last_spike_momo_bar = pos
                    continue

            if bool(rev_sig.iloc[pos]):
                if rev_entries < max_rev_entries and (pos - last_rev_bar) >= cooldown * 10:
                    tp = px * (1 - rev_tp_pct / 100)
                    sl = px * (1 + rev_sl_pct / 100)
                    actions[pos] = SignalAction.SELL
                    metas[pos] = {
                        "suggested_tp": tp,
                        "suggested_sl": sl,
                        "metadata": {"rsi": round(rsi_val, 2), "regime": "post_spike", "spike_type": episode_type},
                    }
                    rev_entries += 1
                    last_rev_bar = pos
                continue

            if onset_now and not in_spike_lockout:
                if splong_entries < splong_max and (pos - last_splong_bar) >= splong_cooldown:
                    if strong_spike_long:
                        actions[pos] = SignalAction.BUY
                        metas[pos] = {
                            "suggested_tp": None,
                            "suggested_sl": px * (1 - splong_sl_pct / 100),
                            "metadata": {
                                "rsi": round(rsi_val, 2),
                                "regime": "spike_long",
                                "spike_type": episode_type,
                                "pct_trail": splong_trail_pct,
                                "peak_excess_pct": round(peak_excess_pct, 2),
                            },
                        }
                        splong_entries += 1
                        last_splong_bar = pos
                        continue

            if just_confirmed_decay and not in_spike_lockout and (pos - last_splong_bar) >= 100:
                if psshort_entries < psshort_max and (pos - last_psshort_bar) >= psshort_cooldown:
                    actions[pos] = SignalAction.SELL
                    metas[pos] = {
                        "suggested_tp": None,
                        "suggested_sl": px * (1 + psshort_sl_pct / 100),
                        "metadata": {
                            "rsi": round(rsi_val, 2),
                            "regime": "post_spike_short",
                            "spike_type": episode_type,
                            "pct_trail": psshort_trail_pct,
                            "peak_excess_pct": round(peak_excess_pct, 2),
                        },
                    }
                    psshort_entries += 1
                    last_psshort_bar = pos
                    continue

            if episode_phase == "spike" and not in_spike_lockout and strong_spike_long and bool(spike_long_sig.iloc[pos]):
                if spike_entries < spike_max_entries and (pos - last_spike_bar) >= spike_cooldown:
                    tp = float(sma_arr[pos])
                    sl = px * (1 - spike_sl_pct / 100)
                    rr = (tp - px) / max(px - sl, 1e-9)
                    if rr >= min_rr:
                        actions[pos] = SignalAction.BUY
                        metas[pos] = {
                            "suggested_tp": tp,
                            "suggested_sl": sl,
                            "metadata": {
                                "rsi": round(rsi_val, 2),
                                "regime": "spike",
                                "spike_type": episode_type,
                                "trailing_atr_mult": spike_atr_trail,
                                "peak_excess_pct": round(peak_excess_pct, 2),
                            },
                        }
                        spike_entries += 1
                        last_spike_bar = pos
                        continue

            if episode_phase == "decay":
                if decay_reentry_anchor <= 0.0:
                    decay_reentry_anchor = px
                    decay_bounce_high = px
                    decay_bounce_low = px
                decay_reentry_anchor = min(decay_reentry_anchor, px)
                decay_bounce_low = min(decay_bounce_low, px) if decay_bounce_low > 0 else px

                rebound_pct = ((px / max(decay_bounce_low, 1e-9)) - 1.0) * 100
                trend_decay_ok = (
                    not np.isnan(fast_now)
                    and not np.isnan(slow_now)
                    and fast_now < slow_now
                    and px <= fast_now * 1.01
                )
                if rebound_pct >= decay_bounce_min_pct and rsi_val >= decay_reentry_rsi:
                    decay_bounce_armed = True
                    decay_bounce_high = max(decay_bounce_high, px)
                elif decay_bounce_armed:
                    decay_bounce_high = max(decay_bounce_high, px)

                bounce_failed = (
                    decay_bounce_armed
                    and decay_bounce_high > 0.0
                    and px <= decay_bounce_high * (1 - decay_bounce_fail_pct / 100)
                    and px < prev_px
                    and trend_decay_ok
                )
                if bounce_failed and decay_bounce_entries < decay_bounce_max and (pos - last_decay_bounce_bar) >= decay_bounce_cooldown:
                    sl = px * (1 + decay_sl_pct / 100)
                    if px > decay_floor and bool(atr_ok.iloc[pos]):
                        actions[pos] = SignalAction.SELL
                        metas[pos] = {
                            "suggested_tp": None,
                            "suggested_sl": sl,
                            "metadata": {
                                "rsi": round(rsi_val, 2),
                                "regime": "decay_bounce_short",
                                "spike_type": episode_type,
                                "trailing_atr_mult": decay_atr_trail,
                                "peak_excess_pct": round(peak_excess_pct, 2),
                                "rebound_pct": round(rebound_pct, 2),
                            },
                        }
                        decay_bounce_entries += 1
                        last_decay_bounce_bar = pos
                        decay_bounce_armed = False
                        decay_reentry_anchor = px
                        decay_bounce_high = px
                        decay_bounce_low = px
                        continue

            decay_reentry_sig = (
                episode_phase == "decay"
                and bool(short_bb.iloc[pos])
                and bool(atr_ok.iloc[pos])
                and px > decay_floor
                and rsi_val >= max(oversold, decay_reentry_rsi)
                and trend_bearish
            )
            if decay_reentry_sig:
                if decay_entries < decay_max_entries and (pos - last_decay_bar) >= decay_cooldown:
                    tp = float(sma_arr[pos])
                    sl = px * (1 + decay_sl_pct / 100)
                    rr = (px - tp) / max(sl - px, 1e-9)
                    if rr >= min_rr:
                        actions[pos] = SignalAction.SELL
                        metas[pos] = {
                            "suggested_tp": tp,
                            "suggested_sl": sl,
                            "metadata": {
                                "rsi": round(rsi_val, 2),
                                "regime": "decay",
                                "spike_type": episode_type,
                                "trailing_atr_mult": decay_atr_trail,
                                "peak_excess_pct": round(peak_excess_pct, 2),
                            },
                        }
                        decay_entries += 1
                        last_decay_bar = pos
                continue

            intraday_pullback_ready = intraday_pullback_short_ready(
                episode_phase=episode_phase,
                active_spike=active_now,
                recent_max_rsi=recent_max_rsi_now,
                current_rsi=rsi_val,
                rsi_trigger=intraday_pullback_rsi_trigger,
                min_rsi_fade_points=intraday_pullback_rsi_fade_pts,
                drop_from_recent_high_pct=drop_from_recent_high_pct,
                min_drop_pct=intraday_pullback_drop_pct,
                recent_upper_band_rejection=bool(short_bb.iloc[pos] or (pos > 0 and bool(short_bb.iloc[pos - 1]))),
                price_below_fast_ema=(not np.isnan(fast_now) and px < fast_now),
                down_bar=(px < prev_px),
                atr_pct=atr_pct_now,
                min_atr_pct=intraday_pullback_min_atr_pct,
            )
            if (
                intraday_pullback_short_enabled
                and intraday_pullback_ready
                and (pos - last_intraday_pullback_bar) >= intraday_pullback_cooldown
                and px > low_price_chop_price
            ):
                tp = px * (1 - intraday_pullback_tp_pct / 100)
                sl = px * (1 + intraday_pullback_sl_pct / 100)
                actions[pos] = SignalAction.SELL
                metas[pos] = {
                    "suggested_tp": tp,
                    "suggested_sl": sl,
                    "metadata": {
                        "rsi": round(rsi_val, 2),
                        "regime": "intraday_pullback_short",
                        "spike_type": episode_type,
                        "pct_trail": intraday_pullback_trail_pct,
                        "recent_max_rsi": round(float(recent_max_rsi_now), 2),
                        "rsi_fade_points": round(float(recent_max_rsi_now - rsi_val), 2),
                        "drop_from_recent_high_pct": round(float(drop_from_recent_high_pct), 2),
                    },
                }
                last_intraday_pullback_bar = pos
                continue

            if (pos - last_normal_bar) < cooldown:
                continue

            allow_normal_long = bool(normal_long_sig.iloc[pos])
            if not bool(p["normal_long_enabled"]):
                allow_normal_long = False
            if active_now and spy_rebound_risk:
                allow_normal_long = False
            if episode_type == "persistent_rollover":
                allow_normal_long = allow_normal_long and rsi_val <= persistent_rollover_long_rsi_max
            if active_now and episode_type != "shock" and fast_slow_spread_pct >= persistent_spread_block_pct:
                allow_normal_long = allow_normal_long and rsi_val <= persistent_spread_deep_rsi_max
            if (
                active_now
                and episode_type == "persistent_trend"
                and episode_phase == "spike"
                and peak_excess_pct >= persistent_prepeak_block_peak_pct
                and rsi_val >= persistent_prepeak_block_rsi_min
                and fast_slow_spread_pct >= persistent_spread_block_pct
            ):
                allow_normal_long = False
            if persistent_rebound_trap:
                allow_normal_long = allow_normal_long and rsi_val <= persistent_rollover_long_rsi_max
            if px <= low_price_chop_price and float(bw_pct.iloc[pos]) <= low_price_chop_bandwidth_pct:
                allow_normal_long = False

            latest_diagnostics: dict[str, Any] | None = None
            if include_diagnostics and pos == n - 1:
                normal_long_rr = None
                normal_short_rr = None
                if (
                    not np.isnan(long_tp_d.iloc[pos])
                    and not np.isnan(long_sl_d.iloc[pos])
                    and long_sl_d.iloc[pos] > 0
                ):
                    normal_long_rr = round(float(long_tp_d.iloc[pos] / max(long_sl_d.iloc[pos], 1e-9)), 3)
                if (
                    not np.isnan(short_tp_d.iloc[pos])
                    and not np.isnan(short_sl_d.iloc[pos])
                    and short_sl_d.iloc[pos] > 0
                ):
                    normal_short_rr = round(float(short_tp_d.iloc[pos] / max(short_sl_d.iloc[pos], 1e-9)), 3)
                gate_values = {
                    "active_spike": bool(active_now),
                    "episode_phase": episode_phase,
                    "episode_type": episode_type,
                    "rsi": round(rsi_val, 2),
                    "rsi_overbought": bool(rsi_val >= overbought),
                    "rsi_oversold": bool(rsi_val <= oversold),
                    "band_width_pct": round(float(bw_pct.iloc[pos]), 3) if not np.isnan(bw_pct.iloc[pos]) else None,
                    "atr_pct": round(float((atr_now / max(px, 1e-9)) * 100), 3),
                    "min_atr_pct": round(float(p["min_atr_pct"]), 3),
                    "atr_ok": bool(atr_ok.iloc[pos]),
                    "is_drift": bool(is_drift.iloc[pos]),
                    "post_spike": bool(post_spike.iloc[pos]),
                    "suppress_shorts": bool(suppress_shorts.iloc[pos]),
                    "in_spike_atr": bool(in_spike_atr.iloc[pos]),
                    "long_bb": bool(long_bb.iloc[pos]),
                    "short_bb": bool(short_bb.iloc[pos]),
                    "normal_long_sig": bool(normal_long_sig.iloc[pos]),
                    "normal_short_sig": bool(normal_short_sig.iloc[pos]),
                    "normal_long_allowed": bool(allow_normal_long),
                    "normal_long_rr": normal_long_rr,
                    "normal_short_rr": normal_short_rr,
                    "recent_max_rsi": round(float(recent_max_rsi_now), 2),
                    "rsi_fade_points": round(float(recent_max_rsi_now - rsi_val), 3),
                    "drop_from_recent_high_pct": round(float(drop_from_recent_high_pct), 3),
                    "intraday_pullback_ready": bool(intraday_pullback_ready),
                    "breakout_spike_long": bool(breakout_spike_long),
                    "spy_assisted_breakout": bool(spy_assisted_breakout),
                    "spy_rebound_risk": bool(spy_rebound_risk),
                    "event_drop_hit": bool(event_drop_hit),
                    "event_short_ready": bool(event_short_ready(
                        event_setup,
                        min_peak_pct=event_target_min_peak_pct,
                        max_rise_bars=event_target_max_rise_bars,
                        confirm_drop_pct=event_confirm_drop_req,
                    )),
                    "event_entries_used": int(event_target_entries),
                    "decay_reentry_sig": bool(decay_reentry_sig),
                    "decay_bounce_armed": bool(decay_bounce_armed),
                    "persistent_rollover": bool(persistent_rollover),
                    "persistent_rebound_trap": bool(persistent_rebound_trap),
                    "trend_bearish": bool(trend_bearish),
                    "peak_excess_pct": round(float(peak_excess_pct), 3),
                    "drawdown_from_peak_pct": round(float(drawdown_from_peak_pct), 3),
                    "benchmark_ret_30": round(float(benchmark_ret_30_now), 3) if not np.isnan(benchmark_ret_30_now) else None,
                    "benchmark_ret_120": round(float(benchmark_ret_120_now), 3) if not np.isnan(benchmark_ret_120_now) else None,
                }
                blocked_by = []
                if bool(rsi_val >= overbought):
                    blocked_by.append("RSI overbought alone is not a short trigger")
                if intraday_pullback_short_enabled and not intraday_pullback_ready:
                    blocked_by.append("intraday pullback short not confirmed")
                if not bool(normal_short_sig.iloc[pos]):
                    blocked_by.append("normal_short_sig=false")
                if episode_phase != "idle":
                    blocked_by.append(f"normal shorts require idle phase, got {episode_phase}")
                if bool(suppress_shorts.iloc[pos]):
                    blocked_by.append("shorts suppressed")
                if bool(in_spike_atr.iloc[pos]):
                    blocked_by.append("in spike ATR regime")
                if not bool(atr_ok.iloc[pos]):
                    blocked_by.append("ATR below minimum")
                if bool(active_now) and bool(spy_rebound_risk):
                    blocked_by.append("SPY rebound risk")
                latest_diagnostics = {
                    "verdict": actions[pos].value,
                    "verdict_reason": "Trade gate passed" if actions[pos] != SignalAction.HOLD else "; ".join(blocked_by[:5]) or "No entry gate passed",
                    "gate_values": gate_values,
                    "gate_summary": ", ".join(f"{k}={v}" for k, v in gate_values.items()),
                }

            if episode_phase != "decay" and allow_normal_long:
                tp = float(sma_arr[pos])
                sl = float(sl_arr[pos])
                rr = (tp - px) / max(px - sl, 1e-9)
                if rr >= min_rr:
                    actions[pos] = SignalAction.BUY
                    metas[pos] = {
                        "suggested_tp": tp,
                        "suggested_sl": sl,
                        "metadata": {"rsi": round(rsi_val, 2), "regime": "normal", "spike_type": episode_type},
                    }
                    last_normal_bar = pos
            elif (
                bool(p["normal_short_enabled"])
                and episode_phase == "idle"
                and px > low_price_chop_price
                and bool(normal_short_sig.iloc[pos])
            ):
                tp = float(sma_arr[pos])
                sl = float(sh_sl_arr[pos])
                rr = (px - tp) / max(sl - px, 1e-9)
                if rr >= min_rr:
                    actions[pos] = SignalAction.SELL
                    metas[pos] = {
                        "suggested_tp": tp,
                        "suggested_sl": sl,
                        "metadata": {"rsi": round(rsi_val, 2), "regime": "normal", "spike_type": episode_type},
                    }
                    last_normal_bar = pos

            if latest_diagnostics is not None:
                meta_payload = metas[pos].setdefault("metadata", {})
                meta_payload.update(latest_diagnostics)

        return actions, metas
