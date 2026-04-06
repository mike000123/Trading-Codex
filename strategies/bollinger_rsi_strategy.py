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

from core.models import Signal, SignalAction
from strategies.base import BaseStrategy, register_strategy
from strategies.components import (
    build_event_short_setup,
    event_completion_target,
    event_short_ready,
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
            "spike_momentum_max": 1,
            "spike_momo_atr_mult": 1.2,
            "spike_momo_momentum_bars": 3,
            "spike_momo_momentum_pct": 0.5,
            "spike_momo_min_rsi": 45.0,
            "spike_momo_max_rsi": 98.0,
            "spike_momo_min_peak_pct": 4.0,
            "spike_momo_min_atr_pct": 0.8,
            "spike_momo_trail_pct": 6.0,
            "spike_momo_sl_pct": 3.0,
            "spike_momo_cooldown": 390,
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
            "low_price_chop_price": 70.0,
            "low_price_chop_bandwidth_pct": 4.0,
        }

    def validate_params(self) -> list[str]:
        p = {**self.default_params(), **self.params}
        errors = []
        if float(p["bb_std"]) <= 0:
            errors.append("bb_std must be > 0.")
        if float(p["rsi_oversold"]) >= float(p["rsi_overbought"]):
            errors.append("rsi_oversold must be < rsi_overbought.")
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
        p = {**self.default_params(), **self.params}
        min_bars = max(int(p["bb_period"]), int(p["rsi_period"]), int(p["spike_high_window"])) + 2
        if len(data) < min_bars:
            return Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                action=SignalAction.HOLD,
                metadata={"reason": "insufficient_data"},
            )
        actions, metas = self.generate_signals_bulk(data, symbol)
        meta = metas[-1]
        rsi_v = float(meta.get("metadata", {}).get("rsi", 50.0))
        return Signal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            action=actions[-1],
            confidence=min(1.0, abs(rsi_v - 50) / 50),
            suggested_tp=meta.get("suggested_tp"),
            suggested_sl=meta.get("suggested_sl"),
            metadata=meta.get("metadata", {}),
        )

    def generate_signals_bulk(self, data: pd.DataFrame, symbol: str):  # noqa: C901
        p = {**self.default_params(), **self.params}

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
        onset_arr = spike_onset.to_numpy()
        active_arr = spike_active.to_numpy()

        for pos in range(n):
            px = close_arr[pos]
            prev_px = close_arr[pos - 1] if pos > 0 else px
            bar_pct = ((px / max(prev_px, 1e-9)) - 1.0) * 100 if pos > 0 else 0.0
            rsi_val = float(rsi_arr[pos]) if not np.isnan(rsi_arr[pos]) else 50.0
            atr_now = atr_arr[pos]
            atr_ma_now = atr_ma_arr[pos] if pos < len(atr_ma_arr) and not np.isnan(atr_ma_arr[pos]) else 0.0
            fast_now = fast_arr[pos]
            slow_now = slow_arr[pos]
            active_now = bool(active_arr[pos])
            onset_now = bool(onset_arr[pos])
            just_confirmed_decay = False
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
            if breakout_spike_long and spike_momentum_max > 0:
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

            if (pos - last_normal_bar) < cooldown:
                continue

            allow_normal_long = bool(normal_long_sig.iloc[pos])
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
            elif episode_phase == "idle" and px > low_price_chop_price and bool(normal_short_sig.iloc[pos]):
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

        return actions, metas
