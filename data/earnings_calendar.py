"""
Shared earnings-calendar cache and reaction-day helpers.

This module serves two related needs:

1. Historical research/backtest support via the labeled event artifact.
2. Forward/Paper live support via a small daily yfinance-based earnings cache.

The daily cache is intentionally narrow:
  - refresh once per day on app startup (or when explicitly forced)
  - scan the tracked cached-equity universe
  - keep events from yesterday through two days ahead so today's AMC/BMO
    reaction sessions are available alongside near-future schedules
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

import pandas as pd
import yfinance as yf
from config.equity_universes import NASDAQ100_TICKERS, SP500_TICKERS

try:
    import pandas_market_calendars as mcal

    _NYSE = mcal.get_calendar("NYSE")
except Exception:
    _NYSE = None


ROOT = Path(__file__).resolve().parents[1]
ALPACA_CACHE_DIR = ROOT / "data_cache" / "alpaca"
YFINANCE_CACHE_DIR = ROOT / "data_cache" / "yfinance"
DERIVED_CACHE_DIR = ROOT / "data_cache" / "derived"
DERIVED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
YF_TZ_CACHE_DIR = ROOT / "data_cache" / "_yfinance_tz_cache"
YF_TZ_CACHE_DIR.mkdir(parents=True, exist_ok=True)

LIVE_CACHE_CSV = DERIVED_CACHE_DIR / "earnings_calendar_daily.csv"
LIVE_CACHE_META = DERIVED_CACHE_DIR / "earnings_calendar_daily_meta.json"
UNIVERSE_CFG_JSON = DERIVED_CACHE_DIR / "earnings_calendar_universe.json"
HISTORICAL_EVENTS_PATH = ROOT / "artifacts" / "optimization" / "earnings_overshoot_dump_events_labeled.csv"

KNOWN_NON_EQUITY = {
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

_LIVE_COLUMNS = [
    "symbol",
    "earnings_date",
    "earnings_ts_et",
    "timing",
    "reaction_date",
    "surprise_pct",
    "eps_estimate",
    "reported_eps",
    "source",
]

_HISTORICAL_CACHE: pd.DataFrame | None = None
_LIVE_CACHE: pd.DataFrame | None = None

DEFAULT_EARNINGS_UNIVERSE = "tracked_plus_nasdaq100"


def available_earnings_universe_labels() -> dict[str, str]:
    labels = {
        "tracked_cached_equities": "Tracked Cached Equities",
        "nasdaq100": "Nasdaq-100",
        "tracked_plus_nasdaq100": "Tracked Cached Equities + Nasdaq-100",
    }
    if SP500_TICKERS:
        labels["sp500"] = "S&P 500"
        labels["nasdaq100_plus_sp500"] = "Nasdaq-100 + S&P 500"
    return labels


def normalize_earnings_universe_id(universe_id: str | None) -> str:
    value = str(universe_id or "").strip().lower()
    labels = available_earnings_universe_labels()
    if value in labels:
        return value
    return DEFAULT_EARNINGS_UNIVERSE


def load_earnings_universe_config() -> dict:
    if not UNIVERSE_CFG_JSON.exists():
        return {"universe_id": DEFAULT_EARNINGS_UNIVERSE, "saved_at": None}
    try:
        payload = json.loads(UNIVERSE_CFG_JSON.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    return {
        "universe_id": normalize_earnings_universe_id(payload.get("universe_id")),
        "saved_at": payload.get("saved_at"),
    }


def persist_earnings_universe_config(universe_id: str) -> str:
    resolved = normalize_earnings_universe_id(universe_id)
    payload = {
        "universe_id": resolved,
        "saved_at": _now_et().isoformat(),
    }
    UNIVERSE_CFG_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return resolved


def _ensure_yfinance_cache_setup() -> None:
    os.environ.setdefault("HOME", str(ROOT))
    os.environ.setdefault("USERPROFILE", str(ROOT))
    try:
        yf.set_tz_cache_location(str(YF_TZ_CACHE_DIR))
    except Exception:
        pass


def _now_et() -> pd.Timestamp:
    ts = pd.Timestamp.utcnow()
    if getattr(ts, "tzinfo", None) is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.tz_convert("America/New_York")


def _date_iso(value) -> str:
    try:
        return pd.Timestamp(value).date().isoformat()
    except Exception:
        return str(value)


def _empty_live_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_LIVE_COLUMNS)


def _is_candidate_equity(symbol: str) -> bool:
    sym = str(symbol or "").strip().upper()
    if not sym or sym in KNOWN_NON_EQUITY:
        return False
    if sym.startswith("^"):
        return False
    return True


def tracked_equity_symbols() -> list[str]:
    symbols: set[str] = set()

    for base in (ALPACA_CACHE_DIR, YFINANCE_CACHE_DIR):
        if not base.exists():
            continue
        for entry in base.iterdir():
            if not entry.is_dir():
                continue
            sym = str(entry.name or "").strip().upper()
            if _is_candidate_equity(sym):
                symbols.add(sym)

    hist = load_historical_events_table()
    if not hist.empty and "symbol" in hist.columns:
        for sym in hist["symbol"].dropna().astype(str).str.upper().tolist():
            if _is_candidate_equity(sym):
                symbols.add(sym)

    return sorted(symbols)


def earnings_universe_symbols(universe_id: str | None = None) -> list[str]:
    resolved = normalize_earnings_universe_id(
        universe_id or load_earnings_universe_config().get("universe_id")
    )
    tracked = set(tracked_equity_symbols())
    nasdaq100 = {sym for sym in NASDAQ100_TICKERS if _is_candidate_equity(sym)}
    sp500 = {sym for sym in SP500_TICKERS if _is_candidate_equity(sym)}

    if resolved == "nasdaq100":
        return sorted(nasdaq100)
    if resolved == "tracked_plus_nasdaq100":
        return sorted(tracked | nasdaq100)
    if resolved == "sp500":
        return sorted(sp500) if sp500 else sorted(tracked)
    if resolved == "nasdaq100_plus_sp500":
        combined = nasdaq100 | sp500
        return sorted(combined) if combined else sorted(nasdaq100)
    return sorted(tracked)


def _earnings_dates(symbol: str, limit: int = 8) -> pd.DataFrame:
    _ensure_yfinance_cache_setup()
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
    out = out.dropna(subset=["earnings_dt"]).copy()
    if out.empty:
        return pd.DataFrame()
    out["earnings_dt_et"] = out["earnings_dt"].dt.tz_convert("America/New_York").dt.tz_localize(None)
    out["surprise_pct"] = pd.to_numeric(out.get("Surprise(%)"), errors="coerce")
    out["eps_estimate"] = pd.to_numeric(out.get("EPS Estimate"), errors="coerce")
    out["reported_eps"] = pd.to_numeric(out.get("Reported EPS"), errors="coerce")
    return out.reset_index(drop=True)


def _timing_from_event_ts(event_ts_et: pd.Timestamp) -> str | None:
    if event_ts_et is None or pd.isna(event_ts_et):
        return None
    mins = int(event_ts_et.hour) * 60 + int(event_ts_et.minute)
    if mins < (9 * 60 + 30):
        return "bmo"
    if mins >= (16 * 60):
        return "amc"
    return None


def _is_nyse_trading_day(day) -> bool:
    if _NYSE is None:
        return pd.Timestamp(day).weekday() < 5
    try:
        date_str = pd.Timestamp(day).strftime("%Y-%m-%d")
    except Exception:
        return False
    try:
        sched = _NYSE.schedule(start_date=date_str, end_date=date_str)
    except Exception:
        return pd.Timestamp(day).weekday() < 5
    return not sched.empty


def _next_nyse_trading_day(day) -> pd.Timestamp:
    candidate = pd.Timestamp(day).normalize()
    for _ in range(10):
        candidate = candidate + pd.Timedelta(days=1)
        if _is_nyse_trading_day(candidate):
            return candidate
    return candidate


def _reaction_date_for_event(event_ts_et: pd.Timestamp, timing: str) -> str | None:
    event_day = pd.Timestamp(event_ts_et).normalize()
    if timing == "bmo":
        reaction_day = event_day if _is_nyse_trading_day(event_day) else _next_nyse_trading_day(event_day)
        return reaction_day.date().isoformat()
    if timing == "amc":
        return _next_nyse_trading_day(event_day).date().isoformat()
    return None


def _load_meta() -> dict:
    if not LIVE_CACHE_META.exists():
        return {}
    try:
        return json.loads(LIVE_CACHE_META.read_text(encoding="utf-8"))
    except Exception:
        return {}


def daily_cache_is_stale(target_date=None, universe_id: str | None = None) -> bool:
    if not LIVE_CACHE_CSV.exists() or not LIVE_CACHE_META.exists():
        return True
    target_iso = _date_iso(target_date or _now_et().date())
    meta = _load_meta()
    cached_universe = normalize_earnings_universe_id(meta.get("universe_id"))
    requested_universe = normalize_earnings_universe_id(
        universe_id or load_earnings_universe_config().get("universe_id")
    )
    return (
        str(meta.get("refreshed_for_date") or "") != target_iso
        or cached_universe != requested_universe
    )


def load_historical_events_table() -> pd.DataFrame:
    global _HISTORICAL_CACHE
    if _HISTORICAL_CACHE is not None:
        return _HISTORICAL_CACHE
    if not HISTORICAL_EVENTS_PATH.exists():
        _HISTORICAL_CACHE = pd.DataFrame()
        return _HISTORICAL_CACHE

    try:
        df = pd.read_csv(HISTORICAL_EVENTS_PATH)
    except Exception:
        _HISTORICAL_CACHE = pd.DataFrame()
        return _HISTORICAL_CACHE

    if df.empty:
        _HISTORICAL_CACHE = pd.DataFrame()
        return _HISTORICAL_CACHE

    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["reaction_date"] = pd.to_datetime(df["reaction_date"], errors="coerce").dt.date.astype(str)
    if "timing" in df.columns:
        df["timing"] = df["timing"].astype(str).str.lower()
    else:
        df["timing"] = ""
    if "source" not in df.columns:
        df["source"] = "historical_research"
    _HISTORICAL_CACHE = df.reset_index(drop=True)
    return _HISTORICAL_CACHE


def load_daily_earnings_cache() -> pd.DataFrame:
    global _LIVE_CACHE
    if _LIVE_CACHE is not None:
        return _LIVE_CACHE
    if not LIVE_CACHE_CSV.exists():
        _LIVE_CACHE = _empty_live_frame()
        return _LIVE_CACHE
    try:
        df = pd.read_csv(LIVE_CACHE_CSV)
    except Exception:
        _LIVE_CACHE = _empty_live_frame()
        return _LIVE_CACHE
    if df.empty:
        _LIVE_CACHE = _empty_live_frame()
        return _LIVE_CACHE
    for col in _LIVE_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["reaction_date"] = pd.to_datetime(df["reaction_date"], errors="coerce").dt.date.astype(str)
    df["earnings_date"] = pd.to_datetime(df["earnings_date"], errors="coerce").dt.date.astype(str)
    df["timing"] = df["timing"].astype(str).str.lower()
    df["source"] = df["source"].fillna("live_calendar").astype(str)
    _LIVE_CACHE = df[_LIVE_COLUMNS].reset_index(drop=True)
    return _LIVE_CACHE


def refresh_daily_earnings_cache(
    *,
    force: bool = False,
    horizon_days: int = 2,
    universe_id: str | None = None,
    status_cb: Callable[[str], None] | None = None,
) -> dict:
    global _LIVE_CACHE

    today_et = _now_et().date()
    target_iso = today_et.isoformat()
    resolved_universe = normalize_earnings_universe_id(
        universe_id or load_earnings_universe_config().get("universe_id")
    )
    universe_label = available_earnings_universe_labels().get(resolved_universe, resolved_universe)
    if not force and not daily_cache_is_stale(today_et, resolved_universe):
        current = load_daily_earnings_cache()
        return {
            "ok": True,
            "refreshed": False,
            "refreshed_for_date": target_iso,
            "universe_id": resolved_universe,
            "universe_label": universe_label,
            "symbols_scanned": 0,
            "event_rows": int(len(current)),
            "reaction_symbols_today": reaction_symbols_for_date(today_et),
        }

    universe = earnings_universe_symbols(resolved_universe)
    window_start = pd.Timestamp(today_et) - pd.Timedelta(days=1)
    window_end = pd.Timestamp(today_et) + pd.Timedelta(days=int(horizon_days))
    existing = load_daily_earnings_cache()

    rows: list[dict] = []
    processed = 0
    errors: list[str] = []

    for idx, symbol in enumerate(universe, start=1):
        if status_cb is not None and (idx == 1 or idx % 10 == 0 or idx == len(universe)):
            status_cb(
                f"Earnings calendar: scanning {symbol} ({idx}/{len(universe)}) for "
                f"{universe_label} reaction days through {window_end.date().isoformat()} ET"
            )
        try:
            frame = _earnings_dates(symbol, limit=8)
            processed += 1
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{symbol}: {exc}")
            continue
        if frame is None or frame.empty:
            continue
        for rec in frame.to_dict(orient="records"):
            event_ts_et = pd.Timestamp(rec.get("earnings_dt_et"))
            if pd.isna(event_ts_et):
                continue
            event_day = pd.Timestamp(event_ts_et.date())
            if event_day < window_start or event_day > window_end:
                continue
            timing = _timing_from_event_ts(event_ts_et)
            if timing not in {"bmo", "amc"}:
                continue
            reaction_date = _reaction_date_for_event(event_ts_et, timing)
            if not reaction_date:
                continue
            reaction_day = pd.Timestamp(reaction_date)
            if reaction_day < pd.Timestamp(today_et) or reaction_day > window_end:
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "earnings_date": event_day.date().isoformat(),
                    "earnings_ts_et": event_ts_et.isoformat(),
                    "timing": timing,
                    "reaction_date": reaction_date,
                    "surprise_pct": rec.get("surprise_pct"),
                    "eps_estimate": rec.get("eps_estimate"),
                    "reported_eps": rec.get("reported_eps"),
                    "source": "live_calendar",
                }
            )

    if processed == 0 and errors:
        return {
            "ok": False,
            "refreshed": False,
            "refreshed_for_date": target_iso,
            "universe_id": resolved_universe,
            "universe_label": universe_label,
            "symbols_scanned": 0,
            "event_rows": int(len(existing)),
            "errors": errors[:20],
            "used_existing_cache": bool(not existing.empty),
        }

    live_df = pd.DataFrame(rows, columns=_LIVE_COLUMNS) if rows else _empty_live_frame()
    if not live_df.empty:
        live_df["symbol"] = live_df["symbol"].astype(str).str.upper()
        live_df = (
            live_df.sort_values(["reaction_date", "symbol", "earnings_ts_et"])
            .drop_duplicates(subset=["symbol", "reaction_date"], keep="last")
            .reset_index(drop=True)
        )

    LIVE_CACHE_CSV.parent.mkdir(parents=True, exist_ok=True)
    live_df.to_csv(LIVE_CACHE_CSV, index=False)
    LIVE_CACHE_META.write_text(
        json.dumps(
            {
                "refreshed_for_date": target_iso,
                "refreshed_at": _now_et().isoformat(),
                "horizon_days": int(horizon_days),
                "universe_id": resolved_universe,
                "universe_label": universe_label,
                "symbols_scanned": int(len(universe)),
                "symbols_processed": int(processed),
                "event_rows": int(len(live_df)),
                "errors": errors[:50],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _LIVE_CACHE = None
    refreshed = load_daily_earnings_cache()
    return {
        "ok": True,
        "refreshed": True,
        "refreshed_for_date": target_iso,
        "universe_id": resolved_universe,
        "universe_label": universe_label,
        "symbols_scanned": int(len(universe)),
        "symbols_processed": int(processed),
        "event_rows": int(len(refreshed)),
        "reaction_symbols_today": reaction_symbols_for_date(today_et),
        "errors": errors[:20],
    }


def reaction_symbols_for_date(target_date=None) -> list[str]:
    target_iso = _date_iso(target_date or _now_et().date())
    live = load_daily_earnings_cache()
    if live.empty:
        return []
    rows = live[live["reaction_date"] == target_iso]
    if rows.empty:
        return []
    return sorted(rows["symbol"].dropna().astype(str).str.upper().unique().tolist())


def merged_events_for_symbol(symbol: str, sign: str | None = None) -> pd.DataFrame:
    symbol_u = str(symbol or "").strip().upper()
    hist = load_historical_events_table()
    live = load_daily_earnings_cache()

    parts: list[pd.DataFrame] = []
    if not hist.empty:
        hist_sym = hist[hist["symbol"] == symbol_u].copy()
        if sign == "positive":
            hist_sym = hist_sym[pd.to_numeric(hist_sym.get("surprise_pct"), errors="coerce") > 0].copy()
        elif sign == "negative":
            hist_sym = hist_sym[pd.to_numeric(hist_sym.get("surprise_pct"), errors="coerce") < 0].copy()
        if not hist_sym.empty:
            if "source" not in hist_sym.columns:
                hist_sym["source"] = "historical_research"
            parts.append(hist_sym)

    if not live.empty:
        live_sym = live[live["symbol"] == symbol_u].copy()
        if not live_sym.empty and sign in {"positive", "negative"}:
            surprise = pd.to_numeric(live_sym.get("surprise_pct"), errors="coerce")
            if sign == "positive":
                live_sym = live_sym[surprise.isna() | (surprise > 0)].copy()
            else:
                live_sym = live_sym[surprise.isna() | (surprise < 0)].copy()
        if not live_sym.empty:
            parts.append(live_sym)

    if not parts:
        return pd.DataFrame()

    merged = pd.concat(parts, ignore_index=True, sort=False)
    if "source" not in merged.columns:
        merged["source"] = ""
    merged["source_priority"] = merged["source"].map(
        {"historical_research": 0, "live_calendar": 1}
    ).fillna(99)
    merged = (
        merged.sort_values(["reaction_date", "source_priority"])
        .drop_duplicates(subset=["symbol", "reaction_date"], keep="first")
        .drop(columns=["source_priority"], errors="ignore")
        .reset_index(drop=True)
    )
    return merged


def is_symbol_reaction_day(symbol: str, target_date=None) -> bool:
    target_iso = _date_iso(target_date or _now_et().date())
    events = merged_events_for_symbol(symbol, sign=None)
    if events.empty or "reaction_date" not in events.columns:
        return False
    return bool((events["reaction_date"].astype(str) == target_iso).any())
