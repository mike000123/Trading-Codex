"""
Validate whether out-of-hours BTC moves add useful context for APLD.

This research pass intentionally reuses the existing companion-data framework:

1. A small probe strategy declares `equity_benchmark` + `crypto_benchmark`
   companion contexts.
2. `prepare_strategy_data(...)` resolves those contexts through
   `config.symbol_profiles` and merges them onto the APLD timestamps.
3. We then study whether BTC overnight moves predict:
   - the APLD opening gap
   - the first 30/60 minutes
   - the full open-to-close move
   - the market-adjusted versions of those moves vs QQQ

The initial default window uses Yahoo 5-minute bars over the last 60 calendar
days. That keeps the validation inside the current cache/data plumbing while
still capturing the overnight hypothesis we care about.
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _install_dummy_logger() -> None:
    if "core.logger" in sys.modules:
        return
    logger_mod = types.ModuleType("core.logger")

    class _Dummy:
        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

        def debug(self, *args, **kwargs):
            pass

    logger_mod.log = _Dummy()
    sys.modules["core.logger"] = logger_mod


_install_dummy_logger()

from config.settings import settings
from core.models import Signal, SignalAction
from data.ingestion import load_from_ticker, load_from_alpaca_history, prepare_strategy_data
from strategies.base import BaseStrategy


ARTIFACT_DIR = ROOT / "artifacts" / "optimization"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
_YF_TZ_CACHE = ROOT / "data_cache" / "_yfinance_tz_cache"
_YF_TZ_CACHE.mkdir(parents=True, exist_ok=True)
try:
    yf.set_tz_cache_location(str(_YF_TZ_CACHE))
except Exception:
    pass

MARKET_TZ = "America/New_York"
ALPACA_SAFE_BUFFER_DAYS = 3


class _APLDBTCProbeStrategy(BaseStrategy):
    strategy_id = "apld_btc_probe"
    name = "APLD BTC Context Probe"
    description = "Companion-data validation scaffold for APLD vs BTC-USD."

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Signal:
        return Signal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            action=SignalAction.HOLD,
            confidence=0.0,
            suggested_tp=None,
            suggested_sl=None,
            metadata={"probe": True},
        )

    def companion_contexts(
        self,
        symbol: str,
        source: str | None = None,
        interval: str | None = None,
    ) -> list[str]:
        return ["equity_benchmark", "crypto_benchmark"]


@dataclass
class SummaryStats:
    observations: int
    mean_pct: float
    median_pct: float
    win_rate_pct: float


def _safe_pct(current, base) -> float:
    if current is None or base is None:
        return np.nan
    try:
        current_f = float(current)
        base_f = float(base)
    except Exception:
        return np.nan
    if not np.isfinite(current_f) or not np.isfinite(base_f) or base_f == 0:
        return np.nan
    return (current_f / base_f - 1.0) * 100.0


def _series_stats(series: pd.Series) -> SummaryStats:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return SummaryStats(0, np.nan, np.nan, np.nan)
    return SummaryStats(
        observations=int(len(clean)),
        mean_pct=float(clean.mean()),
        median_pct=float(clean.median()),
        win_rate_pct=float((clean > 0).mean() * 100.0),
    )


def _corr(df: pd.DataFrame, left: str, right: str) -> float:
    valid = df[[left, right]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(valid) < 3:
        return np.nan
    return float(valid[left].corr(valid[right]))


def _sign_match_pct(df: pd.DataFrame, left: str, right: str) -> float:
    valid = df[[left, right]].apply(pd.to_numeric, errors="coerce").dropna()
    if valid.empty:
        return np.nan
    return float((np.sign(valid[left]) == np.sign(valid[right])).mean() * 100.0)


def _bucket_stats(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["btc_bucket"] = pd.cut(
        pd.to_numeric(work["btc_overnight_pct"], errors="coerce"),
        bins=[-np.inf, -3.0, -1.0, 1.0, 3.0, np.inf],
        labels=["<= -3%", "-3% to -1%", "-1% to 1%", "1% to 3%", ">= 3%"],
    )
    grouped = (
        work.dropna(subset=["btc_bucket"])
        .groupby("btc_bucket", observed=False)
        .agg(
            sessions=("session_date", "count"),
            mean_btc_overnight_pct=("btc_overnight_pct", "mean"),
            apld_gap_pct=("apld_gap_pct", "mean"),
            relative_gap_pct=("relative_gap_pct", "mean"),
            apld_open_to_30m_pct=("apld_open_to_30m_pct", "mean"),
            relative_open_to_30m_pct=("relative_open_to_30m_pct", "mean"),
            apld_open_to_60m_pct=("apld_open_to_60m_pct", "mean"),
            relative_open_to_60m_pct=("relative_open_to_60m_pct", "mean"),
            apld_open_to_close_pct=("apld_open_to_close_pct", "mean"),
            relative_open_to_close_pct=("relative_open_to_close_pct", "mean"),
        )
        .reset_index()
    )
    return grouped


def _load_apld_context(start: pd.Timestamp, end: pd.Timestamp, interval: str, source: str) -> pd.DataFrame:
    if source == "alpaca":
        if not settings.alpaca.has_paper_credentials():
            raise SystemExit("Alpaca paper credentials are required for the Alpaca-backed APLD validation pass.")
        primary = load_from_alpaca_history(
            "APLD",
            interval,
            start,
            end,
            settings.alpaca.paper_api_key,
            settings.alpaca.paper_secret_key,
            paper=True,
            use_cache=True,
        )
    else:
        primary = load_from_ticker("APLD", interval, start, end, use_cache=True)
    if primary is None or primary.empty:
        raise SystemExit("Could not load APLD price history for the validation window.")
    strategy = _APLDBTCProbeStrategy(params={})
    prepared = prepare_strategy_data(
        primary,
        strategy,
        primary_symbol="APLD",
        source=source,
        interval=interval,
        start=start,
        end=end,
    )
    return prepared


def _session_features(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce", utc=True)
    work = work.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    work["date_ny"] = work["date"].dt.tz_convert(MARKET_TZ)
    work["minutes_ny"] = work["date_ny"].dt.hour * 60 + work["date_ny"].dt.minute
    work["session_date"] = work["date_ny"].dt.date
    is_rth = (work["minutes_ny"] >= 570) & (work["minutes_ny"] < 960)
    work = work.loc[is_rth].reset_index(drop=True)
    if work.empty:
        raise SystemExit("No regular-session APLD bars were available after preprocessing.")

    sessions = [
        grp.reset_index(drop=True)
        for _, grp in work.groupby("session_date", sort=True)
        if len(grp) >= 12
    ]
    rows: list[dict[str, Any]] = []
    for prev_session, curr_session in zip(sessions, sessions[1:]):
        prev_last = prev_session.iloc[-1]
        open_row = curr_session.iloc[0]
        row_30m = curr_session.iloc[min(5, len(curr_session) - 1)]
        row_60m = curr_session.iloc[min(11, len(curr_session) - 1)]
        close_row = curr_session.iloc[-1]

        apld_gap_pct = _safe_pct(open_row["open"], prev_last["close"])
        qqq_gap_pct = _safe_pct(open_row.get("benchmark_open"), prev_last.get("benchmark_close"))
        btc_overnight_pct = _safe_pct(open_row.get("crypto_close"), prev_last.get("crypto_close"))

        apld_open_to_30m_pct = _safe_pct(row_30m["close"], open_row["open"])
        apld_open_to_60m_pct = _safe_pct(row_60m["close"], open_row["open"])
        apld_open_to_close_pct = _safe_pct(close_row["close"], open_row["open"])

        qqq_open_to_30m_pct = _safe_pct(row_30m.get("benchmark_close"), open_row.get("benchmark_open"))
        qqq_open_to_60m_pct = _safe_pct(row_60m.get("benchmark_close"), open_row.get("benchmark_open"))
        qqq_open_to_close_pct = _safe_pct(close_row.get("benchmark_close"), open_row.get("benchmark_open"))

        rows.append(
            {
                "session_date": pd.Timestamp(open_row["date_ny"]).date().isoformat(),
                "open_bar_utc": pd.Timestamp(open_row["date"]).isoformat(),
                "apld_gap_pct": apld_gap_pct,
                "btc_overnight_pct": btc_overnight_pct,
                "qqq_gap_pct": qqq_gap_pct,
                "relative_gap_pct": apld_gap_pct - qqq_gap_pct if np.isfinite(apld_gap_pct) and np.isfinite(qqq_gap_pct) else np.nan,
                "apld_open_to_30m_pct": apld_open_to_30m_pct,
                "apld_open_to_60m_pct": apld_open_to_60m_pct,
                "apld_open_to_close_pct": apld_open_to_close_pct,
                "qqq_open_to_30m_pct": qqq_open_to_30m_pct,
                "qqq_open_to_60m_pct": qqq_open_to_60m_pct,
                "qqq_open_to_close_pct": qqq_open_to_close_pct,
                "relative_open_to_30m_pct": apld_open_to_30m_pct - qqq_open_to_30m_pct if np.isfinite(apld_open_to_30m_pct) and np.isfinite(qqq_open_to_30m_pct) else np.nan,
                "relative_open_to_60m_pct": apld_open_to_60m_pct - qqq_open_to_60m_pct if np.isfinite(apld_open_to_60m_pct) and np.isfinite(qqq_open_to_60m_pct) else np.nan,
                "relative_open_to_close_pct": apld_open_to_close_pct - qqq_open_to_close_pct if np.isfinite(apld_open_to_close_pct) and np.isfinite(qqq_open_to_close_pct) else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        raise SystemExit("Not enough APLD sessions were available to build overnight features.")
    return out


def _summary_payload(features: pd.DataFrame, prepared: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, interval: str) -> dict[str, Any]:
    companion_cols = [
        c for c in prepared.columns
        if c.startswith("crypto_") or c.startswith("benchmark_")
    ]
    strong_up = features.loc[pd.to_numeric(features["btc_overnight_pct"], errors="coerce") >= 1.0]
    strong_down = features.loc[pd.to_numeric(features["btc_overnight_pct"], errors="coerce") <= -1.0]
    return {
        "symbol": "APLD",
        "context_symbols": {
            "equity_benchmark": "QQQ",
            "crypto_benchmark": "BTC-USD",
        },
        "source": "yfinance",
        "interval": interval,
        "window": {
            "start": pd.Timestamp(start).isoformat(),
            "end": pd.Timestamp(end).isoformat(),
        },
        "sessions": int(len(features)),
        "prepared_rows": int(len(prepared)),
        "companion_columns": companion_cols,
        "correlations": {
            "btc_vs_apld_gap": _corr(features, "btc_overnight_pct", "apld_gap_pct"),
            "btc_vs_relative_gap": _corr(features, "btc_overnight_pct", "relative_gap_pct"),
            "btc_vs_apld_30m": _corr(features, "btc_overnight_pct", "apld_open_to_30m_pct"),
            "btc_vs_relative_30m": _corr(features, "btc_overnight_pct", "relative_open_to_30m_pct"),
            "btc_vs_apld_60m": _corr(features, "btc_overnight_pct", "apld_open_to_60m_pct"),
            "btc_vs_relative_60m": _corr(features, "btc_overnight_pct", "relative_open_to_60m_pct"),
            "btc_vs_apld_close": _corr(features, "btc_overnight_pct", "apld_open_to_close_pct"),
            "btc_vs_relative_close": _corr(features, "btc_overnight_pct", "relative_open_to_close_pct"),
        },
        "directional_alignment_pct": {
            "btc_vs_apld_gap": _sign_match_pct(features, "btc_overnight_pct", "apld_gap_pct"),
            "btc_vs_relative_gap": _sign_match_pct(features, "btc_overnight_pct", "relative_gap_pct"),
            "btc_vs_apld_30m": _sign_match_pct(features, "btc_overnight_pct", "apld_open_to_30m_pct"),
            "btc_vs_apld_close": _sign_match_pct(features, "btc_overnight_pct", "apld_open_to_close_pct"),
        },
        "btc_overnight_>=1pct": {
            "sessions": int(len(strong_up)),
            "apld_gap": _series_stats(strong_up["apld_gap_pct"]).__dict__,
            "apld_open_to_30m": _series_stats(strong_up["apld_open_to_30m_pct"]).__dict__,
            "apld_open_to_close": _series_stats(strong_up["apld_open_to_close_pct"]).__dict__,
            "relative_open_to_close": _series_stats(strong_up["relative_open_to_close_pct"]).__dict__,
        },
        "btc_overnight_<=-1pct": {
            "sessions": int(len(strong_down)),
            "apld_gap": _series_stats(strong_down["apld_gap_pct"]).__dict__,
            "apld_open_to_30m": _series_stats(strong_down["apld_open_to_30m_pct"]).__dict__,
            "apld_open_to_close": _series_stats(strong_down["apld_open_to_close_pct"]).__dict__,
            "relative_open_to_close": _series_stats(strong_down["relative_open_to_close_pct"]).__dict__,
        },
        "notes": [
            "This first pass reuses the existing companion-data framework rather than introducing a dedicated crypto model.",
            "The default validation window uses Yahoo 5-minute data over the last 60 calendar days so BTC-USD and APLD can be aligned through the same cache-backed loader.",
            "If the BTC overnight relationship looks promising here, the next step should be strategy gating around the APLD open rather than immediate full-day always-on trading.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate BTC overnight context usefulness for APLD.")
    parser.add_argument("--source", default="yfinance", choices=["yfinance", "alpaca"], help="Primary/companion source to use (default: yfinance).")
    parser.add_argument("--interval", default="5m", help="Yahoo interval to use for the initial pass (default: 5m).")
    parser.add_argument("--lookback-days", type=int, default=60, help="Calendar lookback window for the initial pass (default: 60).")
    args = parser.parse_args()

    source = str(args.source).strip().lower()
    interval = str(args.interval).strip()
    interval_key = interval.lower()
    lookback_days = int(args.lookback_days)
    if source == "yfinance" and interval_key in {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}:
        # Yahoo intraday windows are strict. For 5m specifically the practical
        # safe range is slightly under the stated 60-day cap.
        lookback_days = min(lookback_days, 58 if interval_key == "5m" else lookback_days)

    end = pd.Timestamp.utcnow().tz_localize(None)
    if source == "alpaca":
        end = end - pd.Timedelta(days=ALPACA_SAFE_BUFFER_DAYS)
    start = end - pd.Timedelta(days=lookback_days)

    prepared = _load_apld_context(start, end, interval, source)
    features = _session_features(prepared)
    summary = _summary_payload(features, prepared, start, end, interval)
    summary["source"] = source
    buckets = _bucket_stats(features)

    summary_path = ARTIFACT_DIR / "apld_btc_validation_summary.json"
    features_path = ARTIFACT_DIR / "apld_btc_validation_sessions.csv"
    buckets_path = ARTIFACT_DIR / "apld_btc_validation_buckets.csv"

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    features.to_csv(features_path, index=False)
    buckets.to_csv(buckets_path, index=False)

    print(json.dumps(summary, indent=2))
    print()
    print(f"Saved session features -> {features_path}")
    print(f"Saved bucket summary   -> {buckets_path}")


if __name__ == "__main__":
    main()
