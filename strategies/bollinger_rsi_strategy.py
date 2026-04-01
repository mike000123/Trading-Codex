"""
strategies/bollinger_rsi_strategy.py
──────────────────────────────────────
Bollinger + RSI with Four-Regime Market Classification
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FOUR REGIMES for UVXY:

  1. NORMAL  — standard mean reversion
     Both long and short Bollinger signals active.

  2. SPIKE   — VIX explosion (ATR surging, price surging up)
     Shorts PAUSED (don't short the spike).
     Longs still active (buy intra-spike dips).

  3. POST-SPIKE REVERSION — peak confirmed, price falling hard
     This is UVXY's most profitable regime.
     Aggressive SHORTS opened on any strength.
     Uses a dedicated TP (% move target) not middle band.
     Longs paused (don't buy into a collapsing spike).

  4. DRIFT   — ATR so low that expected move < minimum profit threshold
     ALL signals paused. Direct profitability check, not slope-based.

REGIME DETECTION:
  Spike:       ATR > spike_atr_mult × ATR_MA20  OR  price surged > spike_price_pct in N bars
  Post-spike:  In spike AND price < N-bar-high × (1 - peak_drop_pct)
               i.e. price has pulled back X% from the spike peak
  Drift:       50-bar slope < -drift_slope_pct AND ATR < drift_atr_pct% of price
  Normal:      everything else
"""
from __future__ import annotations

from typing import Any, Optional
import numpy as np
import pandas as pd

from core.models import Signal, SignalAction
from strategies.base import BaseStrategy, register_strategy


def _calc_rsi(series: pd.Series, period: int) -> pd.Series:
    d  = series.astype(float).diff()
    g  = d.clip(lower=0.0)
    l  = (-d).clip(lower=0.0)
    ag = g.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    al = l.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    return 100.0 - (100.0 / (1.0 + ag / al.replace(0.0, float("nan"))))


def _calc_bollinger(series: pd.Series, period: int, std_dev: float):
    sma   = series.rolling(period, min_periods=period).mean()
    std   = series.rolling(period, min_periods=period).std(ddof=1)
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, sma, lower, std


def _calc_atr(data: pd.DataFrame, period: int) -> pd.Series:
    hi   = data["high"].astype(float)
    lo   = data["low"].astype(float)
    cl   = data["close"].astype(float)
    prev = cl.shift(1)
    tr   = pd.concat([hi-lo, (hi-prev).abs(), (lo-prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()


@register_strategy
class BollingerRSIStrategy(BaseStrategy):
    strategy_id = "bollinger_rsi"
    name        = "Bollinger + RSI (4-Regime)"
    description = (
        "Four-regime UVXY strategy: Normal mean-reversion, "
        "Spike (no shorts), Post-spike reversion (aggressive shorts), "
        "Drift (flat). Captures the full VIX spike cycle."
    )

    def default_params(self) -> dict[str, Any]:
        return {
            # ── Core Bollinger + RSI ──────────────────────────────────────
            "bb_period":           20,
            "bb_std":              2.0,
            "rsi_period":          9,
            "rsi_oversold":        30,
            "rsi_overbought":      70,
            "sl_band_mult":        0.2,
            "require_cross":       True,
            # ── Quality filters ───────────────────────────────────────────
            "min_band_width_pct":  2.0,
            "min_rr_ratio":        1.5,
            "cooldown_bars":       5,
            # ── Spike regime ──────────────────────────────────────────────
            "spike_atr_mult":      2.0,    # ATR > X × ATR_MA → fast spike
            "spike_price_pct":     5.0,    # price moved >X% in short lookback → fast spike
            "spike_lookback":      10,
            "trend_lookback":      390,    # bars for slow rise check (=1 trading day on 1-min)
            "trend_rise_pct":      8.0,    # price > X% above N bars ago → rising trend → suppress shorts
            # ── Post-spike reversion regime ───────────────────────────────
            "peak_window":         60,     # bars to look back for recent peak
            "peak_drop_pct":       8.0,    # price dropped >X% from N-bar high → post-spike
            "reversion_tp_pct":    15.0,   # TP = entry × (1 - X%) for post-spike shorts
            "reversion_sl_pct":    5.0,    # SL = entry × (1 + X%) for post-spike shorts
            # ── Drift / low-profit filter ─────────────────────────────────
            # Block trades when ATR is so low that the expected move
            # from band to middle band cannot cover trading costs.
            # Much more reliable than slope-based drift detection
            # (slope kills everything on a declining asset like UVXY).
            "min_atr_pct":         0.3,    # ATR must be >= X% of price to trade at all
            "min_tp_atr_ratio":    0.5,    # TP distance must be >= X × ATR (expected move filter)
        }

    def validate_params(self) -> list[str]:
        p = {**self.default_params(), **self.params}
        errors = []
        if float(p["bb_std"]) <= 0:
            errors.append("bb_std must be > 0.")
        if float(p["rsi_oversold"]) >= float(p["rsi_overbought"]):
            errors.append("rsi_oversold must be < rsi_overbought.")
        return errors

    def _classify_regime(self, close: pd.Series, atr_s: pd.Series,
                         atr_ma: pd.Series, p: dict, i: int = -1) -> str:
        """Classify current bar into one of four regimes."""
        spike_atr_m   = float(p["spike_atr_mult"])
        spike_px_pct  = float(p["spike_price_pct"])
        spike_lb      = int(p["spike_lookback"])
        peak_window   = int(p["peak_window"])
        peak_drop_pct = float(p["peak_drop_pct"])
        drift_slope   = float(p["drift_slope_pct"])
        drift_atr_pct = float(p["drift_atr_pct"])

        curr_close  = float(close.iloc[i])
        curr_atr    = float(atr_s.iloc[i])
        curr_atr_ma = float(atr_ma.iloc[i]) if not pd.isna(atr_ma.iloc[i]) else curr_atr

        # Spike detection — catches BOTH fast (ATR/short ROC) and gradual (multi-day ROC) rises
        atr_spike    = curr_atr > spike_atr_m * curr_atr_ma
        lb_idx       = max(0, len(close) + i - spike_lb) if i == -1 else max(0, i - spike_lb)
        px_lb_ago    = float(close.iloc[lb_idx])
        price_surge  = abs(curr_close - px_lb_ago) / max(px_lb_ago, 1e-9) * 100 > spike_px_pct
        trend_lb     = int(p.get("trend_lookback", 390))
        trend_pct_v  = float(p.get("trend_rise_pct", 8.0))
        tl_idx       = max(0, len(close) + i - trend_lb) if i == -1 else max(0, i - trend_lb)
        px_trend_ago = float(close.iloc[tl_idx])
        trend_rising = (curr_close - px_trend_ago) / max(px_trend_ago, 1e-9) * 100 > trend_pct_v
        in_spike     = atr_spike or price_surge or trend_rising

        if in_spike:
            # Post-spike: are we below the recent peak by enough?
            pw_idx       = max(0, len(close) + i - peak_window) if i == -1 else max(0, i - peak_window)
            recent_high  = float(close.iloc[pw_idx:].max() if i == -1
                                 else close.iloc[pw_idx:i+1].max())
            drop_from_peak = (recent_high - curr_close) / max(recent_high, 1e-9) * 100
            if drop_from_peak >= peak_drop_pct:
                return "post_spike"
            return "spike"

        # Drift / low-profit detection: ATR too low to make money
        min_atr_pct = float(p.get("min_atr_pct", 0.3))
        atr_pct     = curr_atr / max(curr_close, 1e-9) * 100
        if atr_pct < min_atr_pct:
            return "drift"

        return "normal"

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Signal:
        p = {**self.default_params(), **self.params}
        bb_period       = int(p["bb_period"])
        bb_std          = float(p["bb_std"])
        rsi_period      = int(p["rsi_period"])
        oversold        = float(p["rsi_oversold"])
        overbought      = float(p["rsi_overbought"])
        sl_band_mult    = float(p["sl_band_mult"])
        require_cross   = bool(p["require_cross"])
        min_bw_pct      = float(p["min_band_width_pct"])
        min_rr          = float(p["min_rr_ratio"])
        rev_tp_pct      = float(p["reversion_tp_pct"])
        rev_sl_pct      = float(p["reversion_sl_pct"])

        min_bars = max(bb_period, rsi_period, 60) + 2
        if len(data) < min_bars:
            return Signal(strategy_id=self.strategy_id, symbol=symbol,
                          action=SignalAction.HOLD,
                          metadata={"reason": "insufficient_data"})

        close              = data["close"].astype(float)
        upper, sma, lower, _ = _calc_bollinger(close, bb_period, bb_std)
        rsi                = _calc_rsi(close, rsi_period)
        atr_s              = _calc_atr(data, 14)
        atr_ma             = atr_s.rolling(20, min_periods=5).mean()

        regime = self._classify_regime(close, atr_s, atr_ma, p)

        curr_close   = float(close.iloc[-1])
        prev_close   = float(close.iloc[-2])
        curr_upper   = float(upper.iloc[-1])
        curr_lower   = float(lower.iloc[-1])
        curr_sma     = float(sma.iloc[-1])
        curr_rsi     = float(rsi.iloc[-1])
        band_width   = curr_upper - curr_lower
        bw_pct       = band_width / max(curr_close, 1e-9) * 100

        action: SignalAction           = SignalAction.HOLD
        suggested_tp: Optional[float] = None
        suggested_sl: Optional[float] = None

        if regime == "drift":
            pass  # flat — nothing to trade

        elif regime == "post_spike":
            # Aggressive short — ride the VIX mean reversion
            # Don't require Bollinger touch, just enter on any bar in this regime
            # Use RSI to avoid shorting when already very oversold
            if curr_rsi > 40:  # don't short if already oversold
                tp = curr_close * (1 - rev_tp_pct / 100)
                sl = curr_close * (1 + rev_sl_pct / 100)
                action, suggested_tp, suggested_sl = SignalAction.SELL, tp, sl

        elif regime == "spike":
            # Long only — buy dips within the spike
            if require_cross:
                lower_cond = float(close.iloc[-2]) > float(lower.iloc[-2]) and curr_close <= curr_lower
            else:
                lower_cond = curr_close <= curr_lower
            if lower_cond and curr_rsi <= overbought and bw_pct >= min_bw_pct:
                tp = curr_sma
                sl = curr_lower - sl_band_mult * band_width
                tp_dist = tp - curr_close
                sl_dist = max(curr_close - sl, 1e-9)
                if tp_dist / sl_dist >= min_rr:
                    action, suggested_tp, suggested_sl = SignalAction.BUY, tp, sl

        else:  # normal
            if require_cross:
                lower_cond = float(close.iloc[-2]) > float(lower.iloc[-2]) and curr_close <= curr_lower
                upper_cond = float(close.iloc[-2]) < float(upper.iloc[-2]) and curr_close >= curr_upper
            else:
                lower_cond = curr_close <= curr_lower
                upper_cond = curr_close >= curr_upper

            if lower_cond and curr_rsi <= overbought and bw_pct >= min_bw_pct:
                tp = curr_sma
                sl = curr_lower - sl_band_mult * band_width
                if (tp - curr_close) / max(curr_close - sl, 1e-9) >= min_rr:
                    action, suggested_tp, suggested_sl = SignalAction.BUY, tp, sl
            elif upper_cond and curr_rsi >= oversold and bw_pct >= min_bw_pct:
                tp = curr_sma
                sl = curr_upper + sl_band_mult * band_width
                if (curr_close - tp) / max(sl - curr_close, 1e-9) >= min_rr:
                    action, suggested_tp, suggested_sl = SignalAction.SELL, tp, sl

        return Signal(
            strategy_id  = self.strategy_id,
            symbol       = symbol,
            action       = action,
            confidence   = min(1.0, abs(curr_rsi - 50) / 50),
            suggested_tp = suggested_tp,
            suggested_sl = suggested_sl,
            metadata     = {
                "rsi":    round(curr_rsi, 2),
                "regime": regime,
                "bw_pct": round(bw_pct, 3),
            },
        )

    def generate_signals_bulk(self, data: pd.DataFrame, symbol: str):
        """Vectorised bulk with four-regime classification."""
        p = {**self.default_params(), **self.params}
        bb_period       = int(p["bb_period"])
        bb_std          = float(p["bb_std"])
        rsi_period      = int(p["rsi_period"])
        oversold        = float(p["rsi_oversold"])
        overbought      = float(p["rsi_overbought"])
        sl_band_mult    = float(p["sl_band_mult"])
        require_cross   = bool(p["require_cross"])
        min_bw_pct      = float(p["min_band_width_pct"])
        min_rr          = float(p["min_rr_ratio"])
        cooldown        = int(p["cooldown_bars"])
        spike_atr_m     = float(p["spike_atr_mult"])
        spike_px_pct    = float(p["spike_price_pct"])
        spike_lb        = int(p["spike_lookback"])
        peak_window     = int(p["peak_window"])
        peak_drop_pct   = float(p["peak_drop_pct"])
        rev_tp_pct      = float(p["reversion_tp_pct"])
        rev_sl_pct      = float(p["reversion_sl_pct"])
        # drift params now read inline via p.get()

        close              = data["close"].astype(float)
        upper, sma, lower, _ = _calc_bollinger(close, bb_period, bb_std)
        rsi                = _calc_rsi(close, rsi_period)
        atr_s              = _calc_atr(data, 14)
        atr_ma             = atr_s.rolling(20, min_periods=5).mean()
        prev_close         = close.shift(1)
        prev_upper         = upper.shift(1)
        prev_lower         = lower.shift(1)
        band_width         = upper - lower
        bw_pct             = band_width / close.replace(0, np.nan) * 100

        # ── Regime flags ──────────────────────────────────────────────────
        atr_spike    = atr_s > spike_atr_m * atr_ma
        px_ago       = close.shift(spike_lb)
        price_surge  = ((close - px_ago).abs() / px_ago.replace(0, np.nan) * 100) > spike_px_pct
        trend_lb     = int(p.get("trend_lookback", 390))
        trend_pct_v  = float(p.get("trend_rise_pct", 8.0))
        px_trend     = close.shift(trend_lb)
        trend_rising = ((close - px_trend) / px_trend.replace(0, np.nan) * 100) > trend_pct_v
        in_spike     = atr_spike | price_surge | trend_rising

        # Rolling peak over peak_window bars
        rolling_high     = close.rolling(peak_window, min_periods=1).max()
        drop_from_peak   = (rolling_high - close) / rolling_high.replace(0, np.nan) * 100
        is_post_spike    = in_spike & (drop_from_peak >= peak_drop_pct)
        is_spike_only    = in_spike & ~is_post_spike

        # Drift: ATR too low to cover costs — block all signals
        min_atr_pct_v    = float(p.get("min_atr_pct", 0.3))
        min_tp_atr       = float(p.get("min_tp_atr_ratio", 0.5))
        atr_pct_series   = atr_s / close.replace(0, np.nan) * 100
        is_drift         = (atr_pct_series < min_atr_pct_v) & ~in_spike

        # ── Signal conditions ─────────────────────────────────────────────
        bw_ok = bw_pct >= min_bw_pct

        # Normal + spike_only: Bollinger long
        if require_cross:
            long_bb_cond  = (prev_close > prev_lower) & (close <= lower)
            short_bb_cond = (prev_close < prev_upper) & (close >= upper)
        else:
            long_bb_cond  = close <= lower
            short_bb_cond = close >= upper

        long_sl     = lower - sl_band_mult * band_width
        long_tp_d   = (sma - close).clip(lower=0)
        long_sl_d   = (close - long_sl).clip(lower=1e-9)
        long_rr_ok  = (long_tp_d / long_sl_d) >= min_rr

        short_sl    = upper + sl_band_mult * band_width
        short_tp_d  = (close - sma).clip(lower=0)
        short_sl_d  = (short_sl - close).clip(lower=1e-9)
        short_rr_ok = (short_tp_d / short_sl_d) >= min_rr

        # Post-spike: aggressive short
        rev_tp_price   = close * (1 - rev_tp_pct / 100)
        rev_sl_price   = close * (1 + rev_sl_pct / 100)
        post_spike_rsi_ok = rsi > 40  # don't short if already oversold

        # Final signal masks
        # TP must be at least min_tp_atr_ratio × ATR away (profitability filter)
        tp_atr_ok_long  = long_tp_d >= (min_tp_atr * atr_s)
        tp_atr_ok_short = short_tp_d >= (min_tp_atr * atr_s)
        long_normal    = (~is_drift & ~is_post_spike &
                          long_bb_cond & (rsi <= overbought) & bw_ok & long_rr_ok & tp_atr_ok_long)
        short_normal   = (~is_drift & ~is_post_spike & ~in_spike &
                          short_bb_cond & (rsi >= oversold) & bw_ok & short_rr_ok & tp_atr_ok_short)
        short_post     = (is_post_spike & post_spike_rsi_ok)

        n       = len(data)
        actions = [SignalAction.HOLD] * n
        metas   = [{"suggested_tp": None, "suggested_sl": None, "metadata": {}}] * n

        last_signal_bar = -cooldown - 1

        for pos in range(n):
            is_long   = bool(long_normal.iloc[pos])
            is_short  = bool(short_normal.iloc[pos])
            is_rev    = bool(short_post.iloc[pos])

            if not (is_long or is_short or is_rev):
                continue

            # Post-spike gets priority and ignores cooldown
            # (we want to be short as soon as the regime flips)
            if is_rev:
                if pos > 0 and actions[pos-1] == SignalAction.SELL:
                    continue  # already short from yesterday — don't re-enter
                px   = float(close.iloc[pos])
                tp   = float(rev_tp_price.iloc[pos])
                sl   = float(rev_sl_price.iloc[pos])
                actions[pos] = SignalAction.SELL
                metas[pos]   = {"suggested_tp": tp, "suggested_sl": sl,
                                 "metadata": {"rsi": round(float(rsi.iloc[pos]), 2),
                                              "regime": "post_spike",
                                              "drop_from_peak": round(float(drop_from_peak.iloc[pos]), 1)}}
                last_signal_bar = pos
                continue

            if pos - last_signal_bar < cooldown:
                continue

            rsi_val = float(rsi.iloc[pos]) if not pd.isna(rsi.iloc[pos]) else 50.0

            if is_long and not pd.isna(rsi.iloc[pos]):
                tp = float(sma.iloc[pos])
                sl = float(long_sl.iloc[pos])
                actions[pos] = SignalAction.BUY
                metas[pos]   = {"suggested_tp": tp, "suggested_sl": sl,
                                 "metadata": {"rsi": round(rsi_val, 2), "regime": "normal"}}
                last_signal_bar = pos
            elif is_short and not pd.isna(rsi.iloc[pos]):
                tp = float(sma.iloc[pos])
                sl = float(short_sl.iloc[pos])
                actions[pos] = SignalAction.SELL
                metas[pos]   = {"suggested_tp": tp, "suggested_sl": sl,
                                 "metadata": {"rsi": round(rsi_val, 2), "regime": "normal"}}
                last_signal_bar = pos

        return actions, metas
