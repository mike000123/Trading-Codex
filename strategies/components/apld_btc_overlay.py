"""Helpers for the APLD BTC-driven opening overlay."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def _finite(*values: float) -> bool:
    return all(v is not None and math.isfinite(float(v)) for v in values)


def apld_btc_confirmation_short_ready(
    *,
    btc_overnight_pct: float,
    relative_gap_pct: float,
    peak_from_open_pct: float,
    pullback_from_peak_pct: float,
    confirm_close_from_open_pct: float,
    btc_threshold: float,
    gap_threshold: float,
    peak_threshold: float,
    pullback_threshold: float,
    confirm_close_max_pct: float,
) -> bool:
    if not _finite(
        btc_overnight_pct,
        relative_gap_pct,
        peak_from_open_pct,
        pullback_from_peak_pct,
        confirm_close_from_open_pct,
    ):
        return False
    return (
        float(btc_overnight_pct) >= float(btc_threshold)
        and float(relative_gap_pct) >= float(gap_threshold)
        and float(peak_from_open_pct) >= float(peak_threshold)
        and float(pullback_from_peak_pct) >= float(pullback_threshold)
        and float(confirm_close_from_open_pct) <= float(confirm_close_max_pct)
    )


def apld_btc_confirmation_long_ready(
    *,
    btc_overnight_pct: float,
    apld_gap_pct: float,
    trough_from_open_pct: float,
    rebound_from_trough_pct: float,
    confirm_close_from_open_pct: float,
    btc_threshold: float,
    gap_threshold: float,
    trough_threshold: float,
    rebound_threshold: float,
    confirm_close_min_pct: float,
) -> bool:
    if not _finite(
        btc_overnight_pct,
        apld_gap_pct,
        trough_from_open_pct,
        rebound_from_trough_pct,
        confirm_close_from_open_pct,
    ):
        return False
    return (
        float(btc_overnight_pct) <= -float(btc_threshold)
        and float(apld_gap_pct) <= -float(gap_threshold)
        and float(trough_from_open_pct) <= -float(trough_threshold)
        and float(rebound_from_trough_pct) >= float(rebound_threshold)
        and float(confirm_close_from_open_pct) >= float(confirm_close_min_pct)
    )


def _safe_pct(current: float, base: float) -> float:
    if not _finite(current, base):
        return float("nan")
    current_f = float(current)
    base_f = float(base)
    if base_f == 0.0:
        return float("nan")
    return (current_f / base_f - 1.0) * 100.0


def build_apld_btc_overlay_payloads(
    data: pd.DataFrame,
    *,
    short_enabled: bool,
    long_enabled: bool,
    short_btc_threshold: float,
    short_gap_threshold: float,
    short_entry_offset_min: int,
    short_peak_threshold: float,
    short_pullback_threshold: float,
    short_confirm_close_max_pct: float,
    short_stop_loss_pct: float,
    long_btc_threshold: float,
    long_gap_threshold: float,
    long_entry_offset_min: int,
    long_trough_threshold: float,
    long_rebound_threshold: float,
    long_confirm_close_min_pct: float,
    long_stop_loss_pct: float,
    long_qqq_weak_filter_enabled: bool,
    long_qqq_weak_close_max_pct: float,
    long_qqq_weak_rebound_min_pct: float,
    market_tz: str = "America/New_York",
) -> dict[int, dict[str, Any]]:
    """Return per-bar overlay payloads keyed by the original row index.

    The overlay only acts during the regular cash session and only on APLD rows
    whose prepared frame includes companion context from QQQ and BTC-USD.
    """
    if data.empty:
        return {}
    required = {
        "date",
        "open",
        "high",
        "low",
        "close",
        "benchmark_open",
        "benchmark_close",
        "crypto_close",
    }
    if not required.issubset(data.columns):
        return {}

    work = data.loc[:, sorted(required)].copy()
    work["orig_index"] = np.arange(len(work), dtype=int)
    work["date"] = pd.to_datetime(work["date"], errors="coerce", utc=True)
    work = work.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date").reset_index(drop=True)
    if work.empty:
        return {}

    work["date_ny"] = work["date"].dt.tz_convert(market_tz)
    work["minutes_ny"] = work["date_ny"].dt.hour * 60 + work["date_ny"].dt.minute
    work["session_date"] = work["date_ny"].dt.date
    work = work.loc[(work["minutes_ny"] >= 570) & (work["minutes_ny"] < 960)].reset_index(drop=True)
    if work.empty:
        return {}

    sessions = [
        grp.reset_index(drop=True)
        for _, grp in work.groupby("session_date", sort=True)
        if len(grp) >= 60
    ]
    payloads: dict[int, dict[str, Any]] = {}

    for prev_session, session in zip(sessions, sessions[1:]):
        prev_last = prev_session.iloc[-1]
        open_row = session.iloc[0]
        open_price = float(open_row["open"])
        benchmark_open = open_row.get("benchmark_open")
        btc_overnight_pct = _safe_pct(open_row.get("crypto_close"), prev_last.get("crypto_close"))
        apld_gap_pct = _safe_pct(open_price, prev_last["close"])
        qqq_gap_pct = _safe_pct(benchmark_open, prev_last.get("benchmark_close"))
        relative_gap_pct = (
            apld_gap_pct - qqq_gap_pct
            if math.isfinite(apld_gap_pct) and math.isfinite(qqq_gap_pct)
            else float("nan")
        )

        if short_enabled and len(session) >= max(short_entry_offset_min, 1):
            prefix = session.iloc[:short_entry_offset_min]
            confirm_row = session.iloc[short_entry_offset_min - 1]
            confirm_close = float(confirm_row["close"])
            peak_from_open_pct = _safe_pct(float(prefix["high"].max()), open_price)
            pullback_from_peak_pct = _safe_pct(float(prefix["high"].max()), confirm_close)
            confirm_close_from_open_pct = _safe_pct(confirm_close, open_price)
            if apld_btc_confirmation_short_ready(
                btc_overnight_pct=btc_overnight_pct,
                relative_gap_pct=relative_gap_pct,
                peak_from_open_pct=peak_from_open_pct,
                pullback_from_peak_pct=pullback_from_peak_pct,
                confirm_close_from_open_pct=confirm_close_from_open_pct,
                btc_threshold=short_btc_threshold,
                gap_threshold=short_gap_threshold,
                peak_threshold=short_peak_threshold,
                pullback_threshold=short_pullback_threshold,
                confirm_close_max_pct=short_confirm_close_max_pct,
            ):
                payloads[int(confirm_row["orig_index"])] = {
                    "action": "SELL",
                    "suggested_tp": None,
                    "suggested_sl": confirm_close * (1.0 + short_stop_loss_pct / 100.0),
                    "metadata": {
                        "regime": "apld_btc_confirm_short",
                        "session_exit": "eod",
                        "apld_btc_overlay": True,
                        "btc_overnight_pct": round(float(btc_overnight_pct), 3),
                        "apld_gap_pct": round(float(apld_gap_pct), 3),
                        "qqq_gap_pct": round(float(qqq_gap_pct), 3) if math.isfinite(float(qqq_gap_pct)) else None,
                        "relative_gap_pct": round(float(relative_gap_pct), 3) if math.isfinite(float(relative_gap_pct)) else None,
                        "peak_from_open_pct": round(float(peak_from_open_pct), 3),
                        "pullback_from_peak_pct": round(float(pullback_from_peak_pct), 3),
                        "confirm_close_from_open_pct": round(float(confirm_close_from_open_pct), 3),
                        "stop_loss_pct": float(short_stop_loss_pct),
                        "entry_offset_min": int(short_entry_offset_min),
                        "verdict_reason": (
                            f"APLD BTC overlay short: BTC overnight {btc_overnight_pct:.2f}% and "
                            f"relative gap {relative_gap_pct:.2f}% met the fade setup; enter after "
                            f"{short_entry_offset_min}m failure and hold to session close unless stopped."
                        ),
                    },
                }

        if long_enabled and len(session) >= max(long_entry_offset_min, 1):
            prefix = session.iloc[:long_entry_offset_min]
            confirm_row = session.iloc[long_entry_offset_min - 1]
            confirm_close = float(confirm_row["close"])
            trough_from_open_pct = _safe_pct(float(prefix["low"].min()), open_price)
            rebound_from_trough_pct = _safe_pct(confirm_close, float(prefix["low"].min()))
            confirm_close_from_open_pct = _safe_pct(confirm_close, open_price)
            qqq_confirm_close_from_open_pct = _safe_pct(confirm_row.get("benchmark_close"), benchmark_open)
            if apld_btc_confirmation_long_ready(
                btc_overnight_pct=btc_overnight_pct,
                apld_gap_pct=apld_gap_pct,
                trough_from_open_pct=trough_from_open_pct,
                rebound_from_trough_pct=rebound_from_trough_pct,
                confirm_close_from_open_pct=confirm_close_from_open_pct,
                btc_threshold=long_btc_threshold,
                gap_threshold=long_gap_threshold,
                trough_threshold=long_trough_threshold,
                rebound_threshold=long_rebound_threshold,
                confirm_close_min_pct=long_confirm_close_min_pct,
            ):
                qqq_weak_veto = (
                    long_qqq_weak_filter_enabled
                    and math.isfinite(float(qqq_confirm_close_from_open_pct))
                    and float(qqq_confirm_close_from_open_pct) < float(long_qqq_weak_close_max_pct)
                    and float(rebound_from_trough_pct) < float(long_qqq_weak_rebound_min_pct)
                )
                if qqq_weak_veto:
                    continue
                payloads[int(confirm_row["orig_index"])] = {
                    "action": "BUY",
                    "suggested_tp": None,
                    "suggested_sl": confirm_close * (1.0 - long_stop_loss_pct / 100.0),
                    "metadata": {
                        "regime": "apld_btc_confirm_long",
                        "session_exit": "eod",
                        "apld_btc_overlay": True,
                        "btc_overnight_pct": round(float(btc_overnight_pct), 3),
                        "apld_gap_pct": round(float(apld_gap_pct), 3),
                        "qqq_gap_pct": round(float(qqq_gap_pct), 3) if math.isfinite(float(qqq_gap_pct)) else None,
                        "relative_gap_pct": round(float(relative_gap_pct), 3) if math.isfinite(float(relative_gap_pct)) else None,
                        "qqq_confirm_close_from_open_pct": (
                            round(float(qqq_confirm_close_from_open_pct), 3)
                            if math.isfinite(float(qqq_confirm_close_from_open_pct))
                            else None
                        ),
                        "trough_from_open_pct": round(float(trough_from_open_pct), 3),
                        "rebound_from_trough_pct": round(float(rebound_from_trough_pct), 3),
                        "confirm_close_from_open_pct": round(float(confirm_close_from_open_pct), 3),
                        "stop_loss_pct": float(long_stop_loss_pct),
                        "entry_offset_min": int(long_entry_offset_min),
                        "verdict_reason": (
                            f"APLD BTC overlay long: BTC overnight {btc_overnight_pct:.2f}% and "
                            f"APLD gap {apld_gap_pct:.2f}% met the reclaim setup; enter after "
                            f"{long_entry_offset_min}m rebound and hold to session close unless stopped."
                            + (
                                f" Weak QQQ tape was allowed only because rebound strength reached "
                                f"{long_qqq_weak_rebound_min_pct:.2f}%."
                                if (
                                    long_qqq_weak_filter_enabled
                                    and math.isfinite(float(qqq_confirm_close_from_open_pct))
                                    and float(qqq_confirm_close_from_open_pct) < float(long_qqq_weak_close_max_pct)
                                )
                                else ""
                            )
                        ),
                    },
                }

    return payloads
