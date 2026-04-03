"""
strategies/bollinger_rsi_strategy.py
──────────────────────────────────────
Bollinger + RSI — Five-Regime UVXY Strategy
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REGIMES (priority order, highest first):
  1. DRIFT       — ATR too low (quiet, no edge): ALL signals paused
  2. SPIKE       — ATR expansion confirmed (in_spike_atr): LONG entries OK,
                   all shorts suppressed
  3. POST-SPIKE  — 3-day high > 1.5× 5d-EMA AND dropped ≥8% from peak:
                   aggressive SHORTS only (fixed TP=15%, SL=5%)
  4. DECAY       — EMA slope declining + price below EMA: SHORT toward ~$40
                   floor using ATR trailing stop.  Normal Bollinger trades
                   still run concurrently (do NOT suppress them with
                   in_decay_raw — kills 7 months of good trading).
  5. NORMAL      — Bollinger mean reversion, both directions

CRITICAL ARCHITECTURAL DETAIL — two separate spike signals:
  in_spike_atr    = ATR expansion only
                    → triggers Spike LONG entries
  suppress_shorts = ATR expansion OR gradual price rise (close > 5% above
                    3-day low)
                    → blocks normal/decay shorts
  These MUST remain separate.  Merging them causes longs to fire during
  gradual declines and destroys performance.

LESSONS LEARNED (do not revert):
  * Do NOT suppress normal Bollinger trades with in_decay_raw
  * Do NOT add decay_confirm_bars requiring 5 days confirmed decay
  * Do NOT use trail_peak = entry_px (never-updated) in the engine
  * Do NOT merge in_spike_atr and suppress_shorts into one variable
  * EMA cross (5/13 bars) fires every ~26 bars on 5-min — too noisy as
    a post-spike exit trigger
"""
from __future__ import annotations
from typing import Any, Optional
import numpy as np
import pandas as pd
from core.models import Signal, SignalAction
from strategies.base import BaseStrategy, register_strategy


# ─── Indicator helpers ────────────────────────────────────────────────────────

def _calc_rsi(series: pd.Series, period: int) -> pd.Series:
    """Wilder's recursive EWM RSI — matches TradingView / Bloomberg."""
    d  = series.astype(float).diff()
    g  = d.clip(lower=0.0)
    l  = (-d).clip(lower=0.0)
    ag = g.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    al = l.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return 100.0 - (100.0 / (1.0 + ag / al.replace(0.0, float("nan"))))


def _calc_bollinger(series: pd.Series, period: int, std_dev: float):
    sma = series.rolling(period, min_periods=period).mean()
    std = series.rolling(period, min_periods=period).std(ddof=1)
    return sma + std_dev * std, sma, sma - std_dev * std, std


def _calc_atr(data: pd.DataFrame, period: int) -> pd.Series:
    hi   = data["high"].astype(float)
    lo   = data["low"].astype(float)
    cl   = data["close"].astype(float)
    prev = cl.shift(1)
    tr   = pd.concat([hi - lo, (hi - prev).abs(), (lo - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


# ─── Strategy ─────────────────────────────────────────────────────────────────

@register_strategy
class BollingerRSIStrategy(BaseStrategy):
    strategy_id = "bollinger_rsi"
    name        = "Bollinger + RSI (5-Regime)"
    description = (
        "Five-regime UVXY strategy: Drift (pause), Spike (ATR long only), "
        "Post-spike (aggressive shorts), Decay (EMA-slope shorts w/ ATR trail), "
        "Normal (Bollinger mean reversion both ways)."
    )

    def default_params(self) -> dict[str, Any]:
        return {
            # ── Bollinger / RSI ──────────────────────────────────────────────
            "bb_period":           20,
            "bb_std":              2.0,
            "rsi_period":          9,
            "rsi_oversold":        30,
            "rsi_overbought":      70,
            "sl_band_mult":        0.2,
            "require_cross":       True,
            "min_band_width_pct":  2.0,
            "min_rr_ratio":        1.5,
            "cooldown_bars":       5,
            "min_atr_pct":         0.3,
            # ── Spike detection (gap-based — fires on extreme single-bar moves) ──
            # Detects spikes via single-bar price change > spike_gap_pct.
            # This catches overnight gap-ups like Aug 5 2024 (+40%) and Apr 7 2025 (+16%).
            # Also used to suppress normal shorts during gradual rises.
            "spike_gap_pct":       15.0,   # single-bar % change to declare spike onset
            "grad_spike_lookback": 1560,   # bars (~8.75 days) for gradual-spike detection
            "grad_spike_pct":      30.0,   # % above N-bar low → gradual spike onset
            "rise_lookback":       1170,   # bars for gradual-rise suppress-shorts check
            "rise_pct":            5.0,    # % above N-bar low → suppress shorts
            # ── Spike LONG params (enter at OPEN of gap bar) ─────────────────────
            # Enters immediately on the spike bar's open — no BB cross required.
            # Uses a percentage-based trailing stop (not ATR) because ATR is
            # meaningless on the spike bar itself (range is enormous).
            "spike_long_sl_pct":   20.0,   # hard SL % below entry (survives spike-bar range)
            "spike_long_trail_pct": 20.0,  # pct trail from running high (grace: skip bar 0)  [OPTIMAL]
            "spike_long_max":       1,     # 1 long per spike episode
            "spike_long_cooldown":  2000,  # bars between episodes (~11 days)
            # ── Post-spike SHORT params (enter after peak confirms reversal) ────
            # Enters short when price drops X% below rolling spike peak.
            # This catches the multi-week decay after VIX events ($344→$108 in Aug 2024).
            "psshort_drop_pct":     3.0,   # % drop from rolling high to trigger entry
            "psshort_sl_pct":      25.0,   # hard SL % above entry (wider for multi-week hold)
            "psshort_trail_pct":   10.0,   # tight trail — locks in gains quickly on post-spike drops
            "psshort_max":          3,     # up to 3 entries while spike_active window is open
            "psshort_cooldown":    390,    # bars between post-spike shorts
            "psshort_window":      4000,   # bars after spike onset (~22 days covers full UVXY decay)
            # ── Old ATR-based spike params (kept for BB-spike-long regime) ──────
            "spike_atr_mult":      4.5,    # ATR mult for residual BB long filter
            "spike_sl_pct":        8.0,    # BB spike long hard SL %
            "spike_atr_trail":     3.0,    # BB spike long ATR trail
            "spike_max_entries":   4,
            "spike_cooldown":      195,
            "spike_momentum_sl_pct":    12.0,
            "spike_momentum_atr_trail":  3.0,
            "spike_momentum_max":        2,
            "spike_momentum_cooldown": 390,
            # ── Post-spike SHORT params ──────────────────────────────────────
            "spike_high_window":   1170,
            "spike_ema_mult":      1.5,
            "spike_ema_span":      1950,
            "peak_drop_pct":       999.0,  # 999 = post_spike disabled (net negative on UVXY ETH data)
            "reversion_tp_pct":    15.0,
            "reversion_sl_pct":    5.0,
            "reversion_rsi_min":   40,
            # ── Decay regime params ──────────────────────────────────────────
            # NOTE: decay_slope_min_pct=999 disables decay shorts (tuned off on
            # extended-hours data where EMA slope is structurally always negative).
            # peak_drop_pct is kept at 8 but post_spike needs reversion_rsi_min
            # tuning per dataset — set to 999 to disable.
            "decay_ema_period":    1950,   # EMA span for slope detection (~1 week on 5-min)
            "decay_slope_lb":      780,    # lookback bars for slope calc (~65 h)
            "decay_slope_min_pct": 999.0,  # 999 = disabled; set to 0.3+ to enable
            "decay_atr_trail":     4.5,    # ATR trailing mult for decay shorts
            "decay_sl_pct":        12.0,   # hard SL % above entry (catastrophic-loss guard)
            "decay_cooldown":      780,    # ~65 h between decay entries
            "decay_max_entries":   12,     # max entries per decay episode
            "decay_floor":         44.0,   # don't short if price is within ~10% of floor
        }

    def validate_params(self) -> list[str]:
        p = {**self.default_params(), **self.params}
        errors = []
        if float(p["bb_std"]) <= 0:
            errors.append("bb_std must be > 0.")
        if float(p["rsi_oversold"]) >= float(p["rsi_overbought"]):
            errors.append("rsi_oversold must be < rsi_overbought.")
        if float(p["decay_slope_min_pct"]) <= 0:
            errors.append("decay_slope_min_pct must be > 0.")
        return errors

    # ─── Regime computation ───────────────────────────────────────────────────

    def _compute_regimes_bulk(self, close: pd.Series, data: pd.DataFrame, p: dict):
        """
        Returns:
            in_spike_atr   – bool series: ATR expansion OR gap spike → spike LONGs
            suppress_shorts– bool series: spike OR gradual rise → blocks normal shorts
            post_spike     – bool series (original EMA-based)
            is_drift       – bool series
            in_decay       – bool series
            atr_s          – ATR series
            spike_gap_onset– bool series: single-bar gap > spike_gap_pct (new spike detector)
            spike_active   – bool series: within spike_long_window bars of a spike onset
        """
        # ── ATR ──────────────────────────────────────────────────────────────
        atr_s  = _calc_atr(data, 14)
        atr_ma = atr_s.rolling(20, min_periods=5).mean()

        # ── Gap-based spike onset: single 5-min bar moves > spike_gap_pct ────
        # This catches the UVXY overnight gap-up events that ATR expansion misses.
        gap_pct = float(p.get("spike_gap_pct", 10.0))
        prev_cl = close.shift(1)
        bar_chg = (close - prev_cl) / prev_cl.replace(0, np.nan) * 100
        spike_gap_onset = bar_chg > gap_pct   # fires on bars with extreme single-bar moves

        # Gradual spike: price risen grad_spike_pct% above its N-bar low.
        # Catches slow-burn spikes (Jan/Mar 2025) that the gap detector misses.
        grad_lb  = int(p.get("grad_spike_lookback", 1560))
        grad_pct = float(p.get("grad_spike_pct", 40.0))
        grad_low  = close.rolling(grad_lb, min_periods=1).min()
        grad_rise = (close / grad_low.replace(0, np.nan) - 1) * 100
        grad_spike_onset = (grad_rise >= grad_pct) & \
                           (grad_rise.shift(1).fillna(0) < grad_pct)

        # Combined onset: gap OR gradual
        any_spike_onset = spike_gap_onset | grad_spike_onset

        # Spike_active: True for psshort_window bars after any spike onset.
        window = int(p.get("psshort_window", 4000))
        spike_active = any_spike_onset.rolling(window, min_periods=1).sum() > 0

        # ── ATR-based spike (kept for BB mean-reversion spike longs) ─────────
        in_spike_atr = (atr_s > float(p["spike_atr_mult"]) * atr_ma) | any_spike_onset

        rise_lb    = int(p["rise_lookback"])
        rise_pct   = float(p["rise_pct"])
        roll_min   = close.rolling(rise_lb, min_periods=1).min()
        rising     = (close / roll_min.replace(0, np.nan) - 1) * 100 > rise_pct
        suppress_shorts = in_spike_atr | rising                       # ATR OR gradual

        # ── Post-spike ───────────────────────────────────────────────────────
        hw         = int(p["spike_high_window"])
        ema_span   = int(p["spike_ema_span"])
        ema_mult   = float(p["spike_ema_mult"])
        drop_pct   = float(p["peak_drop_pct"])
        high_nd    = close.rolling(hw, min_periods=1).max()
        long_ema   = close.ewm(span=ema_span, adjust=False).mean()
        spike_occ  = high_nd > long_ema * ema_mult
        post_spike = spike_occ & (close < high_nd * (1 - drop_pct / 100))

        # ── Drift ────────────────────────────────────────────────────────────
        atr_pct_s = atr_s / close.replace(0, np.nan) * 100
        is_drift  = atr_pct_s < float(p["min_atr_pct"])

        # ── Decay ────────────────────────────────────────────────────────────
        # Criteria: long-period EMA is declining AND price is below it.
        # "Declining" = EMA now vs decay_slope_lb bars ago, expressed as %.
        decay_ema_span = int(p["decay_ema_period"])
        slope_lb       = int(p["decay_slope_lb"])
        slope_min      = float(p["decay_slope_min_pct"])
        decay_ema      = close.ewm(span=decay_ema_span, adjust=False).mean()
        ema_prev       = decay_ema.shift(slope_lb)
        ema_slope_pct  = (decay_ema - ema_prev) / ema_prev.replace(0, np.nan) * 100
        in_decay       = (ema_slope_pct < -slope_min) & (close < decay_ema)

        return in_spike_atr, suppress_shorts, post_spike, is_drift, in_decay, atr_s,                any_spike_onset, spike_active

    # ─── Single-bar signal (used by forward test / live) ──────────────────────

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Signal:
        p = {**self.default_params(), **self.params}
        min_bars = max(int(p["bb_period"]), int(p["rsi_period"]),
                       int(p["spike_high_window"])) + 2
        if len(data) < min_bars:
            return Signal(strategy_id=self.strategy_id, symbol=symbol,
                          action=SignalAction.HOLD,
                          metadata={"reason": "insufficient_data"})

        close = data["close"].astype(float)
        upper, sma, lower, _ = _calc_bollinger(close, int(p["bb_period"]), float(p["bb_std"]))
        rsi   = _calc_rsi(close, int(p["rsi_period"]))
        in_spike_atr, suppress_shorts, post_spike, is_drift, in_decay, atr_s,                spike_gap_onset, spike_active =             self._compute_regimes_bulk(close, data, p)

        curr   = close.iloc[-1]
        rsi_v  = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
        bw     = float(upper.iloc[-1] - lower.iloc[-1])
        bw_pct = bw / max(curr, 1e-9) * 100

        # Priority order
        if bool(is_drift.iloc[-1]):
            regime = "drift"
        elif bool(post_spike.iloc[-1]):
            regime = "post_spike"
        elif bool(in_spike_atr.iloc[-1]):
            regime = "spike"
        elif bool(in_decay.iloc[-1]):
            regime = "decay"
        else:
            regime = "normal"

        action: SignalAction           = SignalAction.HOLD
        suggested_tp: Optional[float] = None
        suggested_sl: Optional[float] = None
        meta: dict                     = {"rsi": round(rsi_v, 2), "regime": regime,
                                          "bw_pct": round(bw_pct, 2)}

        if regime == "post_spike" and rsi_v >= int(p["reversion_rsi_min"]):
            tp = curr * (1 - float(p["reversion_tp_pct"]) / 100)
            sl = curr * (1 + float(p["reversion_sl_pct"]) / 100)
            action, suggested_tp, suggested_sl = SignalAction.SELL, tp, sl

        elif regime == "spike":
            # Only LONG entries during ATR spike; no short
            prev  = float(close.iloc[-2])
            lo_v  = float(lower.iloc[-1])
            lo_p  = float(lower.iloc[-2])
            req   = bool(p["require_cross"])
            lo_c  = (prev > lo_p and curr <= lo_v) if req else curr <= lo_v
            if lo_c and rsi_v <= float(p["rsi_overbought"]) and bw_pct >= float(p["min_band_width_pct"]):
                tp = float(sma.iloc[-1])
                sl = curr * (1 - float(p["spike_sl_pct"]) / 100)
                if (tp - curr) / max(curr - sl, 1e-9) >= float(p["min_rr_ratio"]):
                    action, suggested_tp, suggested_sl = SignalAction.BUY, tp, sl
                    meta["trailing_atr_mult"] = float(p["spike_atr_trail"])

        elif regime in ("decay", "normal"):
            req   = bool(p["require_cross"])
            prev  = float(close.iloc[-2])
            lo_v  = float(lower.iloc[-1]); hi_v = float(upper.iloc[-1])
            lo_p  = float(lower.iloc[-2]); hi_p = float(upper.iloc[-2])
            lo_c  = (prev > lo_p and curr <= lo_v) if req else curr <= lo_v
            hi_c  = (prev < hi_p and curr >= hi_v) if req else curr >= hi_v

            # Long (both regimes)
            if lo_c and rsi_v <= float(p["rsi_overbought"]) and bw_pct >= float(p["min_band_width_pct"]):
                tp = float(sma.iloc[-1])
                sl = lo_v - float(p["sl_band_mult"]) * bw
                if (tp - curr) / max(curr - sl, 1e-9) >= float(p["min_rr_ratio"]):
                    action, suggested_tp, suggested_sl = SignalAction.BUY, tp, sl

            # Short — normal allowed; decay allowed IF not suppress_shorts
            elif (hi_c
                  and not bool(suppress_shorts.iloc[-1])
                  and rsi_v >= float(p["rsi_oversold"])
                  and bw_pct >= float(p["min_band_width_pct"])):
                tp = float(sma.iloc[-1])
                sl = hi_v + float(p["sl_band_mult"]) * bw
                if (curr - tp) / max(sl - curr, 1e-9) >= float(p["min_rr_ratio"]):
                    action, suggested_tp, suggested_sl = SignalAction.SELL, tp, sl
                    if regime == "decay":
                        # Override: use decay hard SL + ATR trail instead of band SL
                        sl = curr * (1 + float(p["decay_sl_pct"]) / 100)
                        suggested_sl = sl
                        meta["trailing_atr_mult"] = float(p["decay_atr_trail"])

        return Signal(strategy_id=self.strategy_id, symbol=symbol, action=action,
                      confidence=min(1.0, abs(rsi_v - 50) / 50),
                      suggested_tp=suggested_tp, suggested_sl=suggested_sl,
                      metadata=meta)

    # ─── Bulk signal generation (used by backtest engine) ─────────────────────

    def generate_signals_bulk(self, data: pd.DataFrame, symbol: str):  # noqa: C901
        p = {**self.default_params(), **self.params}

        bb_period     = int(p["bb_period"]); bb_std = float(p["bb_std"])
        rsi_period    = int(p["rsi_period"])
        oversold      = float(p["rsi_oversold"]); overbought = float(p["rsi_overbought"])
        sl_band_mult  = float(p["sl_band_mult"])
        require_cross = bool(p["require_cross"])
        min_bw_pct    = float(p["min_band_width_pct"])
        min_rr        = float(p["min_rr_ratio"])
        cooldown      = int(p["cooldown_bars"])

        # Post-spike params
        rev_tp_pct  = float(p["reversion_tp_pct"])
        rev_sl_pct  = float(p["reversion_sl_pct"])
        rev_rsi_min = float(p["reversion_rsi_min"])

        # Spike long params (BB mean-reversion)
        spike_sl_pct      = float(p["spike_sl_pct"])
        spike_atr_trail   = float(p["spike_atr_trail"])
        spike_max_entries = int(p["spike_max_entries"])
        spike_cooldown    = int(p["spike_cooldown"])

        # Spike momentum params (ATR-based onset entry — legacy)
        smom_sl_pct   = float(p.get("spike_momentum_sl_pct",    12.0))
        smom_trail    = float(p.get("spike_momentum_atr_trail",  4.5))
        smom_max      = int(  p.get("spike_momentum_max",         2))
        smom_cooldown = int(  p.get("spike_momentum_cooldown",   390))

        # NEW gap-spike LONG params
        splong_sl_pct    = float(p.get("spike_long_sl_pct",    20.0))
        splong_trail_pct = float(p.get("spike_long_trail_pct", 20.0))
        splong_max       = int(  p.get("spike_long_max",         1))
        splong_cooldown  = int(  p.get("spike_long_cooldown",  2000))

        # Post-spike SHORT params
        psshort_drop_pct  = float(p.get("psshort_drop_pct",   5.0))
        psshort_sl_pct    = float(p.get("psshort_sl_pct",    25.0))
        psshort_trail_pct = float(p.get("psshort_trail_pct", 40.0))
        psshort_max       = int(  p.get("psshort_max",          3))
        psshort_cooldown  = int(  p.get("psshort_cooldown",   390))

        # Decay params
        decay_atr_trail   = float(p["decay_atr_trail"])
        decay_sl_pct      = float(p["decay_sl_pct"])
        decay_cooldown    = int(p["decay_cooldown"])
        decay_max_entries = int(p["decay_max_entries"])
        decay_floor       = float(p["decay_floor"])

        # ── Indicator series ─────────────────────────────────────────────────
        close              = data["close"].astype(float)
        upper, sma, lower, _ = _calc_bollinger(close, bb_period, bb_std)
        rsi                = _calc_rsi(close, rsi_period)
        band_width         = upper - lower
        bw_pct             = band_width / close.replace(0, np.nan) * 100

        in_spike_atr, suppress_shorts, post_spike, is_drift, in_decay, atr_s, \
               spike_gap_onset, spike_active = \
            self._compute_regimes_bulk(close, data, p)

        prev_close  = close.shift(1)
        prev_upper  = upper.shift(1)
        prev_lower  = lower.shift(1)

        if require_cross:
            long_bb  = (prev_close > prev_lower) & (close <= lower)
            short_bb = (prev_close < prev_upper) & (close >= upper)
        else:
            long_bb  = close <= lower
            short_bb = close >= upper

        # Normal / decay long SL and TP
        long_sl    = lower - sl_band_mult * band_width
        long_tp_d  = (sma - close).clip(lower=0)
        long_sl_d  = (close - long_sl).clip(lower=1e-9)
        long_rr_ok = (long_tp_d / long_sl_d) >= min_rr

        # Normal short SL and TP
        short_sl     = upper + sl_band_mult * band_width
        short_tp_d   = (close - sma).clip(lower=0)
        short_sl_d   = (short_sl - close).clip(lower=1e-9)
        short_rr_ok  = (short_tp_d / short_sl_d) >= min_rr

        atr_ok = (atr_s / close.replace(0, np.nan) * 100) >= float(p["min_atr_pct"])

        # ── Pre-compute signal masks ──────────────────────────────────────────
        # Spike momentum: first bar where in_spike_atr transitions F→T.
        # This is the ONSET of a new ATR expansion event — enter immediately
        # without waiting for a Bollinger lower-band touch.
        prev_spike_atr       = in_spike_atr.shift(1).fillna(False)
        spike_onset          = in_spike_atr & ~prev_spike_atr & ~is_drift & ~post_spike
        spike_momentum_sig   = spike_onset  # refined further in loop (cooldown/cap)

        # Sanity guard: close must be within 50% of the lower band.
        # This blocks data-corruption artifacts (e.g. reverse-split transaction bars
        # that show prices at 1/3 of the surrounding regime) without filtering
        # legitimate deep BB touches where close is a tick below long_sl.
        long_data_ok = close > (lower * 0.5)

        # Spike LONG: only when in_spike_atr (not the broader suppress_shorts)
        spike_long_sig = (
            in_spike_atr & ~is_drift & ~post_spike
            & long_bb & (rsi <= overbought) & bw_pct_ok(bw_pct, min_bw_pct) & long_rr_ok & atr_ok
            & long_data_ok
        )

        # Normal LONG: any regime except drift/post_spike; decay bars included
        normal_long_sig = (
            ~is_drift & ~post_spike
            & long_bb & (rsi <= overbought) & bw_pct_ok(bw_pct, min_bw_pct) & long_rr_ok & atr_ok
            & long_data_ok
        )

        # Normal SHORT: not in spike suppress zone
        normal_short_sig = (
            ~is_drift & ~post_spike & ~suppress_shorts & ~in_spike_atr
            & short_bb & (rsi >= oversold) & bw_pct_ok(bw_pct, min_bw_pct) & short_rr_ok & atr_ok
        )

        # Decay SHORT: in decay, not in suppress zone, price above floor buffer
        decay_short_sig = (
            in_decay & ~suppress_shorts & ~post_spike & ~is_drift
            & short_bb & (rsi >= oversold) & bw_pct_ok(bw_pct, min_bw_pct) & atr_ok
            & (close > decay_floor)
        )

        # Post-spike SHORT
        rev_sig = post_spike & ~is_drift & (rsi >= rev_rsi_min)

        # ── Serial loop with cooldowns and entry caps ─────────────────────────
        n       = len(data)
        actions = [SignalAction.HOLD] * n
        metas   = [{"suggested_tp": None, "suggested_sl": None, "metadata": {}}
                   for _ in range(n)]

        last_normal_bar   = -cooldown - 1
        last_spike_bar    = -spike_cooldown - 1
        last_smom_bar     = -smom_cooldown - 1
        last_splong_bar   = -splong_cooldown - 1
        last_psshort_bar  = -psshort_cooldown - 1
        last_decay_bar    = -decay_cooldown - 1
        last_rev_bar      = -cooldown * 10 - 1

        spike_entries     = 0
        smom_entries      = 0
        splong_entries    = 0   # gap-spike long entries per episode
        psshort_entries   = 0   # post-spike short entries per episode
        decay_entries     = 0
        rev_entries       = 0
        max_rev_entries   = 2

        was_post_spike    = False
        was_in_spike      = False
        was_in_decay      = False
        was_spike_active  = False

        # Rolling-high tracker for post-spike short (tracks peak since spike onset)
        spike_episode_high = 0.0

        close_arr     = close.to_numpy()
        sma_arr       = sma.to_numpy()
        upper_arr     = upper.to_numpy()
        lower_arr     = lower.to_numpy()
        bw_arr        = band_width.to_numpy()
        sl_arr        = long_sl.to_numpy()
        sh_sl_arr     = short_sl.to_numpy()
        rsi_arr       = rsi.to_numpy()
        gap_onset_arr = spike_gap_onset.to_numpy()
        spactive_arr  = spike_active.to_numpy()

        for pos in range(n):
            currently_post   = bool(post_spike.iloc[pos])
            currently_spike  = bool(in_spike_atr.iloc[pos])
            currently_decay  = bool(in_decay.iloc[pos])
            currently_active = bool(spactive_arr[pos])
            gap_onset_now    = bool(gap_onset_arr[pos])

            # Reset counters on regime transitions
            if not currently_post and was_post_spike:
                rev_entries = 0
            if not currently_spike and was_in_spike:
                spike_entries = 0
                smom_entries  = 0
            if not currently_decay and was_in_decay:
                decay_entries = 0
            if not currently_active and was_spike_active:
                splong_entries  = 0
                psshort_entries = 0
                spike_episode_high = 0.0

            # Track running high during spike episode (for post-spike short)
            if currently_active:
                spike_episode_high = max(spike_episode_high, float(close_arr[pos]))

            was_post_spike   = currently_post
            was_in_spike     = currently_spike
            was_in_decay     = currently_decay
            was_spike_active = currently_active

            px      = close_arr[pos]
            rsi_val = float(rsi_arr[pos]) if not np.isnan(rsi_arr[pos]) else 50.0

            # ── Priority 1: Post-spike SHORT ──────────────────────────────────
            if bool(rev_sig.iloc[pos]):
                if rev_entries < max_rev_entries and (pos - last_rev_bar) >= cooldown * 10:
                    tp = px * (1 - rev_tp_pct / 100)
                    sl = px * (1 + rev_sl_pct / 100)
                    actions[pos] = SignalAction.SELL
                    metas[pos]   = {"suggested_tp": tp, "suggested_sl": sl,
                                    "metadata": {"rsi": round(rsi_val, 2),
                                                 "regime": "post_spike",
                                                 "entry_n": rev_entries + 1}}
                    rev_entries    += 1
                    last_rev_bar    = pos
                continue

            # ── Priority 1b: Gap-spike LONG (NEW — enters on spike-bar open) ─
            # Fires when a single bar moves > spike_gap_pct% from the prior close.
            # Uses percentage-based trailing stop (not ATR) — ATR is meaningless
            # on the spike bar itself due to extreme intrabar range.
            # The engine will open this at the NEXT bar's open (1-bar delay),
            # so we flag it here and let the normal execution path handle it.
            if gap_onset_now:
                if splong_entries < splong_max and (pos - last_splong_bar) >= splong_cooldown:
                    sl = px * (1 - splong_sl_pct / 100)
                    actions[pos] = SignalAction.BUY
                    metas[pos]   = {"suggested_tp": None, "suggested_sl": sl,
                                    "metadata": {"rsi": round(rsi_val, 2),
                                                 "regime": "spike_long",
                                                 "pct_trail": splong_trail_pct,
                                                 "entry_n": splong_entries + 1}}
                    splong_entries += 1
                    last_splong_bar = pos
                    spike_episode_high = max(spike_episode_high, px)
                    continue

            # ── Priority 1c: Post-spike SHORT ────────────────────────────────
            # Enter short when price drops psshort_drop_pct% below spike peak.
            # Only valid within psshort_window bars of a spike onset.
            # suppress_shorts is intentionally OVERRIDDEN here — after a spike
            # the price is elevated and we WANT to short the decay.
            if (currently_active
                    and spike_episode_high > 0
                    and px <= spike_episode_high * (1 - psshort_drop_pct / 100)
                    and not currently_post
                    and (pos - last_splong_bar) >= 100):  # 100 bars after spike_long
                if psshort_entries < psshort_max and (pos - last_psshort_bar) >= psshort_cooldown:
                    sl = px * (1 + psshort_sl_pct / 100)
                    actions[pos] = SignalAction.SELL
                    metas[pos]   = {"suggested_tp": None, "suggested_sl": sl,
                                    "metadata": {"rsi": round(rsi_val, 2),
                                                 "regime": "post_spike_short",
                                                 "pct_trail": psshort_trail_pct,
                                                 "entry_n": psshort_entries + 1}}
                    psshort_entries += 1
                    last_psshort_bar = pos
                    continue

            # ── Priority 2a: Spike MOMENTUM LONG (ATR-based onset entry) ─────
            # Enter immediately when in_spike_atr fires for the first time.
            # No BB cross needed — this catches the initial explosive move.
            if bool(spike_momentum_sig.iloc[pos]):
                if smom_entries < smom_max and (pos - last_smom_bar) >= smom_cooldown:
                    sl = px * (1 - smom_sl_pct / 100)
                    # No fixed TP — let the ATR trail ride it out
                    actions[pos] = SignalAction.BUY
                    metas[pos]   = {"suggested_tp": None, "suggested_sl": sl,
                                    "metadata": {"rsi": round(rsi_val, 2),
                                                 "regime": "spike_momentum",
                                                 "trailing_atr_mult": smom_trail,
                                                 "entry_n": smom_entries + 1}}
                    smom_entries   += 1
                    last_smom_bar   = pos
                    continue   # momentum fired — skip BB spike and normal blocks

            # ── Priority 2b: Spike LONG (BB mean-reversion during elevated ATR) ──
            # Only `continue` if the spike long actually fired; otherwise fall
            # through so normal longs can fire on the same bar.
            if bool(spike_long_sig.iloc[pos]):
                if spike_entries < spike_max_entries and (pos - last_spike_bar) >= spike_cooldown:
                    tp = float(sma_arr[pos])
                    sl = px * (1 - spike_sl_pct / 100)
                    rr = (tp - px) / max(px - sl, 1e-9)
                    if rr >= min_rr:
                        actions[pos] = SignalAction.BUY
                        metas[pos]   = {"suggested_tp": tp, "suggested_sl": sl,
                                        "metadata": {"rsi": round(rsi_val, 2),
                                                     "regime": "spike",
                                                     "trailing_atr_mult": spike_atr_trail,
                                                     "entry_n": spike_entries + 1}}
                        spike_entries  += 1
                        last_spike_bar  = pos
                        continue   # spike long fired — don't also emit a normal long

            # ── Priority 3: Decay SHORT ───────────────────────────────────────
            # When decay is active, upgrade a normal short signal's exit parameters
            # (wider SL + ATR trail to ride the multi-week grind) instead of
            # blocking it. This way decay ENHANCES normal shorts rather than
            # replacing them and cutting off 80% of normal short opportunities.
            # Decay does not fire independently — it only modifies an existing
            # normal short signal on the same bar (handled in priority 4 below).
            # We still track decay_entries to cap how many enhanced entries fire.

            # ── Priority 4: Normal LONG / SHORT ──────────────────────────────
            if (pos - last_normal_bar) < cooldown:
                continue

            if bool(normal_long_sig.iloc[pos]) and not np.isnan(rsi_arr[pos]):
                tp = float(sma_arr[pos])
                sl = float(sl_arr[pos])
                rr = (tp - px) / max(px - sl, 1e-9)
                if rr >= min_rr:
                    actions[pos] = SignalAction.BUY
                    metas[pos]   = {"suggested_tp": tp, "suggested_sl": sl,
                                    "metadata": {"rsi": round(rsi_val, 2),
                                                 "regime": "spike" if currently_spike else "normal"}}
                    last_normal_bar = pos

            elif bool(normal_short_sig.iloc[pos]) and not np.isnan(rsi_arr[pos]):
                tp = float(sma_arr[pos])
                sl = float(sh_sl_arr[pos])
                rr = (px - tp) / max(sl - px, 1e-9)
                if rr >= min_rr:
                    # If decay is also active and we have budget, upgrade this
                    # short with decay's wider SL and ATR trailing stop.
                    if (currently_decay
                            and decay_entries < decay_max_entries
                            and (pos - last_decay_bar) >= decay_cooldown):
                        sl = px * (1 + decay_sl_pct / 100)
                        regime_tag = "decay"
                        trail_tag  = decay_atr_trail
                        decay_entries  += 1
                        last_decay_bar  = pos
                    else:
                        regime_tag = "normal"
                        trail_tag  = None
                    meta_inner = {"rsi": round(rsi_val, 2), "regime": regime_tag}
                    if trail_tag is not None:
                        meta_inner["trailing_atr_mult"] = trail_tag
                    actions[pos] = SignalAction.SELL
                    metas[pos]   = {"suggested_tp": tp, "suggested_sl": sl,
                                    "metadata": meta_inner}
                    last_normal_bar = pos

        return actions, metas


def bw_pct_ok(bw_pct: pd.Series, min_bw_pct: float) -> pd.Series:
    """Helper so the mask expressions stay readable."""
    return bw_pct >= min_bw_pct
