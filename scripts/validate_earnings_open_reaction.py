"""
Validate whether off-hours earnings events create tradable open-reaction curves.

Research goal:
1. Pull historical earnings timestamps + surprises for cached stock symbols.
2. Align each before-market-open (BMO) or after-market-close (AMC) event to the
   next regular trading session open in our local Alpaca 1-minute cache.
3. Measure whether the post-open move tends to fade or continue.
4. Test simple confirmation rules such as:
   - positive surprise + gap-up + early weakness -> fade short
   - negative surprise + gap-down + early strength -> fade long

This is intentionally a research pass. It helps us decide whether an
earnings-driven strategy is worth formal integration before we add realistic
execution/slippage assumptions for live trading.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import types
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


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

# Keep Yahoo/yfinance caches inside the writable repo so the validator works
# cleanly under the workspace sandbox.
os.environ["HOME"] = str(ROOT)
os.environ["USERPROFILE"] = str(ROOT)

import yfinance as yf


ARTIFACT_DIR = ROOT / "artifacts" / "optimization"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
YF_TZ_CACHE = ROOT / "data_cache" / "_yfinance_tz_cache"
YF_TZ_CACHE.mkdir(parents=True, exist_ok=True)
try:
    yf.set_tz_cache_location(str(YF_TZ_CACHE))
except Exception:
    pass

ALPACA_CACHE_DIR = ROOT / "data_cache" / "alpaca"
RTH_OPEN_MINUTE = 9 * 60 + 30
RTH_CLOSE_MINUTE = 16 * 60
KNOWN_NON_STOCKS = {
    "BTC-USD",
    "GLD",
    "SLV",
    "QQQ",
    "SPY",
    "IWM",
    "TLT",
    "TIP",
    "UUP",
    "USO",
    "UVXY",
    "VIXY",
    "VXX",
    "VXZ",
    "XLF",
    "IEF",
    "GDX",
}


@dataclass
class StrategyResult:
    variant: str
    trades: int
    win_rate_pct: float
    mean_return_pct: float
    median_return_pct: float
    compounded_return_pct: float
    long_trades: int
    short_trades: int


def _safe_pct(current, base) -> float:
    try:
        cur = float(current)
        ref = float(base)
    except Exception:
        return np.nan
    if not np.isfinite(cur) or not np.isfinite(ref) or ref == 0:
        return np.nan
    return (cur / ref - 1.0) * 100.0


def _compound_return_pct(returns_pct: pd.Series, starting_equity: float = 1000.0) -> float:
    clean = pd.to_numeric(returns_pct, errors="coerce").dropna()
    if clean.empty:
        return 0.0
    equity = float(starting_equity)
    for ret in clean:
        equity *= 1.0 + (float(ret) / 100.0)
    return (equity / float(starting_equity) - 1.0) * 100.0


def _load_cached_intraday(symbol: str) -> pd.DataFrame | None:
    path = ALPACA_CACHE_DIR / symbol / "1Min.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path, parse_dates=["date"])
    if frame.empty:
        return None
    frame = frame.sort_values("date").reset_index(drop=True)
    mins = frame["date"].dt.hour * 60 + frame["date"].dt.minute
    frame["minutes_et"] = mins
    frame["session_date"] = frame["date"].dt.date
    return frame


def _cached_symbols(limit: int = 0) -> list[str]:
    symbols: list[str] = []
    for entry in sorted(ALPACA_CACHE_DIR.iterdir()):
        if not entry.is_dir():
            continue
        sym = entry.name.upper().strip()
        if sym in KNOWN_NON_STOCKS:
            continue
        if not (entry / "1Min.csv").exists():
            continue
        symbols.append(sym)
    return symbols[:limit] if limit > 0 else symbols


def _earnings_dates(symbol: str, limit: int) -> pd.DataFrame:
    try:
        frame = yf.Ticker(symbol).get_earnings_dates(limit=limit)
    except Exception:
        return pd.DataFrame()
    if frame is None or len(frame) == 0:
        return pd.DataFrame()
    out = frame.reset_index().rename(columns={"Earnings Date": "earnings_dt"})
    if "earnings_dt" not in out.columns:
        return pd.DataFrame()
    out["earnings_dt"] = pd.to_datetime(out["earnings_dt"], errors="coerce", utc=True)
    out["earnings_dt_et"] = out["earnings_dt"].dt.tz_convert("America/New_York").dt.tz_localize(None)
    out["surprise_pct"] = pd.to_numeric(out.get("Surprise(%)"), errors="coerce")
    out["eps_estimate"] = pd.to_numeric(out.get("EPS Estimate"), errors="coerce")
    out["reported_eps"] = pd.to_numeric(out.get("Reported EPS"), errors="coerce")
    return out.dropna(subset=["earnings_dt_et"]).reset_index(drop=True)


def _session_dates(frame: pd.DataFrame) -> list[pd.Timestamp]:
    rth = frame[(frame["minutes_et"] >= RTH_OPEN_MINUTE) & (frame["minutes_et"] <= RTH_CLOSE_MINUTE)]
    dates = sorted(pd.to_datetime(pd.Series(rth["session_date"].unique())))
    return list(dates)


def _reaction_date(
    event_ts: pd.Timestamp,
    trading_dates: list[pd.Timestamp],
) -> tuple[pd.Timestamp | None, str | None]:
    if event_ts is None or not trading_dates:
        return None, None
    event_date = pd.Timestamp(event_ts.date())
    mins = event_ts.hour * 60 + event_ts.minute
    if mins < RTH_OPEN_MINUTE:
        for d in trading_dates:
            if d == event_date:
                return d, "bmo"
        for d in trading_dates:
            if d > event_date:
                return d, "bmo_next"
        return None, None
    if mins >= RTH_CLOSE_MINUTE:
        for d in trading_dates:
            if d > event_date:
                return d, "amc"
        return None, None
    return None, None


def _session_slice(frame: pd.DataFrame, session_date: pd.Timestamp) -> pd.DataFrame:
    day = pd.Timestamp(session_date).date()
    out = frame[(frame["session_date"] == day) & (frame["minutes_et"] >= RTH_OPEN_MINUTE) & (frame["minutes_et"] <= RTH_CLOSE_MINUTE)].copy()
    return out.sort_values("date").reset_index(drop=True)


def _prev_session_close(frame: pd.DataFrame, trading_dates: list[pd.Timestamp], session_date: pd.Timestamp) -> float | None:
    try:
        idx = trading_dates.index(pd.Timestamp(session_date))
    except ValueError:
        return None
    if idx <= 0:
        return None
    prev_day = pd.Timestamp(trading_dates[idx - 1]).date()
    prev_rows = frame[(frame["session_date"] == prev_day) & (frame["minutes_et"] <= RTH_CLOSE_MINUTE)]
    if prev_rows.empty:
        return None
    return float(prev_rows.iloc[-1]["close"])


def _afterhours_slice(frame: pd.DataFrame, event_ts: pd.Timestamp, reaction_date: pd.Timestamp) -> pd.DataFrame:
    reaction_open = pd.Timestamp(reaction_date) + pd.Timedelta(hours=9, minutes=30)
    out = frame[(frame["date"] >= event_ts) & (frame["date"] < reaction_open)].copy()
    return out.sort_values("date").reset_index(drop=True)


def _row_at_or_after(frame: pd.DataFrame, ts: pd.Timestamp) -> pd.Series | None:
    rows = frame[frame["date"] >= ts]
    if rows.empty:
        return None
    return rows.iloc[0]


def _build_events_for_symbol(
    symbol: str,
    cache: pd.DataFrame,
    earnings: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[dict]:
    rows: list[dict] = []
    trading_dates = _session_dates(cache)
    if not trading_dates:
        return rows

    for _, event in earnings.iterrows():
        event_ts = pd.Timestamp(event["earnings_dt_et"])
        if event_ts < start or event_ts > end + pd.Timedelta(days=1):
            continue
        reaction_date, timing = _reaction_date(event_ts, trading_dates)
        if reaction_date is None or timing is None:
            continue

        session = _session_slice(cache, reaction_date)
        if session.empty:
            continue
        prev_close = _prev_session_close(cache, trading_dates, reaction_date)
        if prev_close is None or prev_close <= 0:
            continue

        open_row = _row_at_or_after(session, pd.Timestamp(reaction_date) + pd.Timedelta(hours=9, minutes=30))
        close_row = session.iloc[-1] if not session.empty else None
        row_5m = _row_at_or_after(session, pd.Timestamp(reaction_date) + pd.Timedelta(hours=9, minutes=35))
        row_15m = _row_at_or_after(session, pd.Timestamp(reaction_date) + pd.Timedelta(hours=9, minutes=45))
        row_30m = _row_at_or_after(session, pd.Timestamp(reaction_date) + pd.Timedelta(hours=10, minutes=0))
        row_60m = _row_at_or_after(session, pd.Timestamp(reaction_date) + pd.Timedelta(hours=10, minutes=30))
        if open_row is None or close_row is None or row_5m is None:
            continue

        afterhours = _afterhours_slice(cache, event_ts, reaction_date)
        ah_high = float(afterhours["high"].max()) if not afterhours.empty else np.nan
        ah_low = float(afterhours["low"].min()) if not afterhours.empty else np.nan

        open_px = float(open_row["open"])
        close_px = float(close_row["close"])
        row = {
            "symbol": symbol,
            "event_time_et": event_ts.isoformat(),
            "reaction_date": pd.Timestamp(reaction_date).date().isoformat(),
            "timing": timing,
            "surprise_pct": float(event["surprise_pct"]) if pd.notna(event["surprise_pct"]) else np.nan,
            "eps_estimate": float(event["eps_estimate"]) if pd.notna(event["eps_estimate"]) else np.nan,
            "reported_eps": float(event["reported_eps"]) if pd.notna(event["reported_eps"]) else np.nan,
            "prev_close": prev_close,
            "open_px": open_px,
            "close_px": close_px,
            "gap_pct": _safe_pct(open_px, prev_close),
            "open_to_close_pct": _safe_pct(close_px, open_px),
            "confirm5_close_from_open_pct": _safe_pct(float(row_5m["close"]), open_px),
            "confirm15_close_from_open_pct": _safe_pct(float(row_15m["close"]), open_px) if row_15m is not None else np.nan,
            "confirm30_close_from_open_pct": _safe_pct(float(row_30m["close"]), open_px) if row_30m is not None else np.nan,
            "open_to_30m_pct": _safe_pct(float(row_30m["close"]), open_px) if row_30m is not None else np.nan,
            "open_to_60m_pct": _safe_pct(float(row_60m["close"]), open_px) if row_60m is not None else np.nan,
            "first30_high_from_open_pct": _safe_pct(float(session[session["date"] <= pd.Timestamp(reaction_date) + pd.Timedelta(hours=10)]["high"].max()), open_px),
            "first30_low_from_open_pct": _safe_pct(float(session[session["date"] <= pd.Timestamp(reaction_date) + pd.Timedelta(hours=10)]["low"].min()), open_px),
            "afterhours_high_from_prev_close_pct": _safe_pct(ah_high, prev_close),
            "afterhours_low_from_prev_close_pct": _safe_pct(ah_low, prev_close),
            "entry5_close": float(row_5m["close"]),
            "entry15_close": float(row_15m["close"]) if row_15m is not None else np.nan,
            "entry30_close": float(row_30m["close"]) if row_30m is not None else np.nan,
        }
        rows.append(row)
    return rows


def _trade_frame(
    events: pd.DataFrame,
    *,
    min_gap_pct: float,
    confirm_minutes: int,
    mode: str,
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    confirm_col = {
        5: "confirm5_close_from_open_pct",
        15: "confirm15_close_from_open_pct",
        30: "confirm30_close_from_open_pct",
    }[int(confirm_minutes)]
    entry_col = {
        5: "entry5_close",
        15: "entry15_close",
        30: "entry30_close",
    }[int(confirm_minutes)]

    work = events.copy()
    work["direction"] = None
    work["return_pct"] = np.nan

    positive = work[(pd.to_numeric(work["surprise_pct"], errors="coerce") > 0) & (pd.to_numeric(work["gap_pct"], errors="coerce") >= float(min_gap_pct))]
    negative = work[(pd.to_numeric(work["surprise_pct"], errors="coerce") < 0) & (pd.to_numeric(work["gap_pct"], errors="coerce") <= -float(min_gap_pct))]

    long_mask = pd.Series(False, index=work.index)
    short_mask = pd.Series(False, index=work.index)
    confirm = pd.to_numeric(work[confirm_col], errors="coerce")

    if mode == "fade":
        short_idx = positive[confirm.loc[positive.index] <= 0].index
        long_idx = negative[confirm.loc[negative.index] >= 0].index
    elif mode == "continuation":
        long_idx = positive[confirm.loc[positive.index] >= 0].index
        short_idx = negative[confirm.loc[negative.index] <= 0].index
    else:
        raise ValueError(f"Unknown mode: {mode}")

    long_mask = work.index.isin(long_idx)
    short_mask = work.index.isin(short_idx)
    work.loc[long_mask, "direction"] = "long"
    work.loc[short_mask, "direction"] = "short"
    entry_px = pd.to_numeric(work[entry_col], errors="coerce")
    close_px = pd.to_numeric(work["close_px"], errors="coerce")
    work.loc[long_mask, "return_pct"] = (close_px[long_mask] / entry_px[long_mask] - 1.0) * 100.0
    work.loc[short_mask, "return_pct"] = -(close_px[short_mask] / entry_px[short_mask] - 1.0) * 100.0

    out = work.dropna(subset=["direction", "return_pct"]).copy()
    out["variant"] = f"{mode}_{confirm_minutes}m_gap{int(min_gap_pct)}"
    return out.reset_index(drop=True)


def _summarize_variant(trades: pd.DataFrame, variant: str) -> StrategyResult:
    clean = trades.copy()
    if clean.empty:
        return StrategyResult(variant, 0, np.nan, np.nan, np.nan, 0.0, 0, 0)
    rets = pd.to_numeric(clean["return_pct"], errors="coerce").dropna()
    return StrategyResult(
        variant=variant,
        trades=int(len(rets)),
        win_rate_pct=float((rets > 0).mean() * 100.0),
        mean_return_pct=float(rets.mean()),
        median_return_pct=float(rets.median()),
        compounded_return_pct=float(_compound_return_pct(rets)),
        long_trades=int((clean["direction"] == "long").sum()),
        short_trades=int((clean["direction"] == "short").sum()),
    )


def _bucket_summary(events: pd.DataFrame) -> dict:
    if events.empty:
        return {}
    work = events.copy()
    work["pos_surprise_gap_up"] = (pd.to_numeric(work["surprise_pct"], errors="coerce") > 0) & (pd.to_numeric(work["gap_pct"], errors="coerce") > 0)
    work["neg_surprise_gap_down"] = (pd.to_numeric(work["surprise_pct"], errors="coerce") < 0) & (pd.to_numeric(work["gap_pct"], errors="coerce") < 0)

    def _stats(mask: pd.Series) -> dict:
        subset = work[mask].copy()
        if subset.empty:
            return {"events": 0}
        return {
            "events": int(len(subset)),
            "mean_gap_pct": float(pd.to_numeric(subset["gap_pct"], errors="coerce").mean()),
            "mean_open_to_close_pct": float(pd.to_numeric(subset["open_to_close_pct"], errors="coerce").mean()),
            "mean_confirm5_pct": float(pd.to_numeric(subset["confirm5_close_from_open_pct"], errors="coerce").mean()),
            "median_open_to_close_pct": float(pd.to_numeric(subset["open_to_close_pct"], errors="coerce").median()),
            "win_rate_open_to_close_pct": float((pd.to_numeric(subset["open_to_close_pct"], errors="coerce") > 0).mean() * 100.0),
        }

    return {
        "positive_surprise_gap_up": _stats(work["pos_surprise_gap_up"]),
        "negative_surprise_gap_down": _stats(work["neg_surprise_gap_down"]),
        "timing_breakdown": work["timing"].value_counts(dropna=False).to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate off-hours earnings open-reaction strategies on cached 1-minute stock data.")
    parser.add_argument("--start", default="2024-04-04", help="Start date for event study window (default: 2024-04-04).")
    parser.add_argument("--end", default="2026-05-01", help="End date for event study window (default: 2026-05-01).")
    parser.add_argument("--earnings-limit", type=int, default=12, help="How many historical earnings dates to request per symbol (default: 12).")
    parser.add_argument("--limit-symbols", type=int, default=0, help="Optional limit for quick testing; 0 means all cached stock symbols.")
    parser.add_argument("--pause-seconds", type=float, default=0.15, help="Small pause between Yahoo earnings requests (default: 0.15).")
    parser.add_argument("--artifact-stem", default="earnings_open_reaction", help="Artifact stem (default: earnings_open_reaction).")
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    symbols = _cached_symbols(limit=int(args.limit_symbols or 0))
    print(f"Studying {len(symbols)} cached stock symbols from {start.date()} to {end.date()}")

    all_events: list[dict] = []
    symbol_status: list[dict] = []

    for idx, symbol in enumerate(symbols, start=1):
        print(f"[{idx:03d}/{len(symbols):03d}] {symbol} ...", flush=True)
        cache = _load_cached_intraday(symbol)
        if cache is None or cache.empty:
            symbol_status.append({"symbol": symbol, "status": "no_cache", "events": 0})
            continue
        earnings = _earnings_dates(symbol, limit=int(args.earnings_limit))
        if earnings.empty:
            symbol_status.append({"symbol": symbol, "status": "no_earnings_data", "events": 0})
            time.sleep(float(args.pause_seconds))
            continue
        events = _build_events_for_symbol(symbol, cache, earnings, start, end)
        symbol_status.append({"symbol": symbol, "status": "ok", "events": len(events)})
        all_events.extend(events)
        time.sleep(float(args.pause_seconds))

    events_df = pd.DataFrame(all_events)
    events_path = ARTIFACT_DIR / f"{args.artifact_stem}_events.csv"
    status_path = ARTIFACT_DIR / f"{args.artifact_stem}_symbol_status.csv"
    summary_path = ARTIFACT_DIR / f"{args.artifact_stem}_summary.json"
    variants_path = ARTIFACT_DIR / f"{args.artifact_stem}_strategy_variants.csv"

    pd.DataFrame(symbol_status).to_csv(status_path, index=False)
    if events_df.empty:
        events_df.to_csv(events_path, index=False)
        summary_path.write_text(json.dumps({"events": 0, "message": "No valid off-hours earnings reaction events found."}, indent=2), encoding="utf-8")
        pd.DataFrame().to_csv(variants_path, index=False)
        print("No valid events found.")
        return

    events_df = events_df.sort_values(["reaction_date", "symbol"]).reset_index(drop=True)
    events_df.to_csv(events_path, index=False)

    variant_rows: list[StrategyResult] = []
    variant_trades: list[pd.DataFrame] = []
    for mode in ("fade", "continuation"):
        for confirm_minutes in (5, 15, 30):
            for min_gap in (2.0, 4.0, 6.0, 8.0):
                trades = _trade_frame(
                    events_df,
                    min_gap_pct=float(min_gap),
                    confirm_minutes=int(confirm_minutes),
                    mode=mode,
                )
                variant_trades.append(trades)
                variant_rows.append(_summarize_variant(trades, f"{mode}_{confirm_minutes}m_gap{int(min_gap)}"))

    variants_df = pd.DataFrame([asdict(row) for row in variant_rows]).sort_values(
        ["compounded_return_pct", "mean_return_pct", "trades"],
        ascending=[False, False, False],
    )
    variants_df.to_csv(variants_path, index=False)

    best_variant = variants_df.iloc[0].to_dict() if not variants_df.empty else {}
    top_fade = variants_df[variants_df["variant"].str.startswith("fade_")].head(5)
    top_cont = variants_df[variants_df["variant"].str.startswith("continuation_")].head(5)

    summary = {
        "window": {"start": start.date().isoformat(), "end": end.date().isoformat()},
        "symbols_studied": len(symbols),
        "symbols_with_events": int(sum(1 for row in symbol_status if row["status"] == "ok" and row["events"] > 0)),
        "events": int(len(events_df)),
        "bucket_summary": _bucket_summary(events_df),
        "best_variant": best_variant,
        "top_fade_variants": top_fade.to_dict(orient="records"),
        "top_continuation_variants": top_cont.to_dict(orient="records"),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print(f"Events studied: {len(events_df)}")
    print("Best variant:")
    if best_variant:
        print(json.dumps(best_variant, indent=2))
    print(f"Artifacts written:\n- {events_path}\n- {status_path}\n- {variants_path}\n- {summary_path}")


if __name__ == "__main__":
    main()
