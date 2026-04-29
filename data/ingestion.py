"""
data/ingestion.py
─────────────────
Unified data loading layer with local caching.

All loaders return clean DataFrames: date, open, high, low, close, volume.

Cache behaviour:
  - On first fetch: downloads full range, saves to data_cache/<source>/<symbol>/<tf>.csv
  - On subsequent fetches: loads cache, checks for gap, fetches only new bars, appends
  - Cache is persistent on local PC; ephemeral per-session on Streamlit Cloud
  - Different sources kept in separate folders (different formats, different symbols)
"""
from __future__ import annotations

import io
from typing import Optional

import pandas as pd
import yfinance as yf

from config.symbol_profiles import context_label, context_prefix, resolve_context_symbol
from config.settings import settings
from core.logger import log
from data.cache import DataCache
from data.fair_value import prepare_gld_fair_value_context

_cache = DataCache()


# ─── Internal helpers ────────────────────────────────────────────────────────

def _canonicalize(name: str) -> str:
    return "".join(ch for ch in name.lower().strip() if ch.isalnum())


def _to_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _normalize_df(df: pd.DataFrame, col_map: dict[str, str]) -> pd.DataFrame:
    """Build standardized OHLCV frame from a raw df + column mapping."""
    volume_col = col_map.get("volume")
    normalized = pd.DataFrame(
        {
            "date":   pd.to_datetime(df[col_map["date"]], errors="coerce",
                                     utc=True).dt.tz_localize(None),
            "open":   _to_numeric(df[col_map["open"]]),
            "high":   _to_numeric(df[col_map["high"]]),
            "low":    _to_numeric(df[col_map["low"]]),
            "close":  _to_numeric(df[col_map["close"]]),
            "volume": (_to_numeric(df[volume_col]) if volume_col
                       else pd.Series(0.0, index=df.index)),
        }
    ).dropna(subset=["date", "open", "high", "low", "close"])

    return (normalized
            .sort_values("date")
            .drop_duplicates("date")
            .reset_index(drop=True))


# ─── CSV upload ──────────────────────────────────────────────────────────────

def load_from_csv(file_buffer) -> pd.DataFrame:
    """Parse an uploaded CSV file into a standard OHLCV DataFrame."""
    raw = file_buffer.read() if hasattr(file_buffer, "read") else file_buffer
    if not raw:
        raise ValueError("The uploaded file is empty.")

    df = pd.read_csv(io.BytesIO(raw) if isinstance(raw, bytes) else io.StringIO(raw))
    canon = {_canonicalize(c): c for c in df.columns}

    aliases: dict[str, list[str]] = {
        "date":   ["date", "datetime", "time", "timestamp"],
        "open":   ["open", "openingprice"],
        "high":   ["high", "max"],
        "low":    ["low", "min"],
        "close":  ["close", "closelast", "closeprice", "last"],
        "volume": ["volume", "vol"],
    }

    col_map: dict[str, Optional[str]] = {}
    missing = []
    for target, candidates in aliases.items():
        matched = next((canon[c] for c in candidates if c in canon), None)
        if matched is None and target != "volume":
            missing.append(target)
        col_map[target] = matched  # type: ignore[assignment]

    if missing:
        raise ValueError(
            f"CSV missing required columns: {', '.join(missing)}. "
            "Accepted names: Date/Datetime, Open, High/Max, Low/Min, Close/CloseLast."
        )

    result = _normalize_df(df, {k: v for k, v in col_map.items() if v})
    if result.empty:
        raise ValueError("No valid rows after parsing CSV.")

    log.info(f"CSV: {len(result)} rows "
             f"{result['date'].min().date()} → {result['date'].max().date()}")
    return result


# ─── Yahoo Finance ────────────────────────────────────────────────────────────

def load_from_ticker(
    ticker: str,
    interval: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch OHLCV from Yahoo Finance via yfinance.
    If use_cache=True, only fetches the missing date range and appends to cache.

    Note: yfinance 1-min data is limited to last 7 days.
    For longer history use Alpaca (load_from_alpaca_history).
    """
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("Ticker cannot be empty.")
    if end_date < start_date:
        raise ValueError("End date must be on or after start date.")

    source = "yfinance"

    # ── Cache check ───────────────────────────────────────────────────────────
    if use_cache:
        fetch_start, fetch_end = _cache.missing_range(
            source, ticker, interval, start_date, end_date)
        if fetch_start is None:
            # Full range already cached
            cached = _cache.load(source, ticker, interval)
            mask   = ((cached["date"] >= start_date) &
                      (cached["date"] <= end_date))
            log.info(f"yfinance CACHE HIT: {ticker}/{interval} — "
                     f"serving from cache, no download needed")
            return cached[mask].reset_index(drop=True)
    else:
        fetch_start, fetch_end = start_date, end_date

    log.info(f"yfinance FETCH: {ticker} {interval} "
             f"{fetch_start.date()} → {fetch_end.date()}")

    fetch_failed: Exception | None = None
    try:
        data = yf.download(
            ticker,
            start     = fetch_start.strftime("%Y-%m-%d"),
            end       = (fetch_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            interval  = interval,
            auto_adjust = False,
            progress  = False,
        )
    except Exception as exc:  # offline / DNS / 429 / yfinance internal
        fetch_failed = exc
        data = None

    # Cache fallback path: serve cached range when fresh fetch is unavailable
    # OR when it returned nothing. Without this branch, an offline backtest
    # would raise even with a fully populated data_cache.
    if fetch_failed is not None or data is None or data.empty:
        if use_cache:
            cached = _cache.load(source, ticker, interval)
            if cached is not None and not cached.empty:
                mask = ((cached["date"] >= start_date) &
                        (cached["date"] <= end_date))
                sliced = cached[mask].reset_index(drop=True)
                if not sliced.empty:
                    if fetch_failed is not None:
                        log.warning(f"yfinance fetch failed for {ticker}/{interval} "
                                    f"({fetch_failed.__class__.__name__}: {fetch_failed}). "
                                    f"Serving {len(sliced)} cached bars instead.")
                    else:
                        log.warning(f"yfinance returned no new data for {ticker}/{interval}. "
                                    f"Serving {len(sliced)} cached bars.")
                    return sliced
        if fetch_failed is not None:
            raise ValueError(
                f"yfinance fetch failed for {ticker}/{interval} and no cached "
                f"data is available for {start_date.date()} → {end_date.date()}: "
                f"{fetch_failed.__class__.__name__}: {fetch_failed}"
            )
        raise ValueError(f"No data returned for {ticker} / {interval}.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close"]
    for col in required:
        if col not in data.columns:
            raise ValueError(f"Fetched data missing column: {col}")

    vol_col = "Volume" if "Volume" in data.columns else None
    raw     = data.copy()
    raw["_date"] = (pd.to_datetime(data.index, errors="coerce", utc=True)
                    .tz_localize(None))
    col_map = {"date": "_date", "open": "Open",
               "high": "High",  "low": "Low", "close": "Close"}
    if vol_col:
        col_map["volume"] = vol_col

    new_data = _normalize_df(raw, col_map)
    if new_data.empty:
        raise ValueError("No valid rows after fetching ticker data.")

    # ── Cache update ──────────────────────────────────────────────────────────
    if use_cache:
        merged = _cache.append(source, ticker, interval, new_data)
        # Return only the requested range
        mask   = ((merged["date"] >= start_date) &
                  (merged["date"] <= end_date))
        return merged[mask].reset_index(drop=True)

    return new_data


# ─── Alpaca ───────────────────────────────────────────────────────────────────

def load_from_alpaca_history(
    symbol: str,
    timeframe: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    api_key: str,
    secret_key: str,
    paper: bool = True,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch historical bars from Alpaca with local caching.

    First call: downloads full range, saves to data_cache/alpaca/<symbol>/<tf>.csv
    Subsequent calls: loads cache, fetches only new bars since last cached date.

    Prefers SIP feed first, but automatically falls back to IEX when the
    account is not permitted to query recent SIP bars. That keeps forward /
    paper warm-up seeding working on hosted environments with lighter plans.
    """
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    except ImportError:
        raise ImportError("alpaca-py not installed. Run: pip install alpaca-py")

    tf_map = {
        "1Min":  TimeFrame(1,  TimeFrameUnit.Minute),
        "2Min":  TimeFrame(2,  TimeFrameUnit.Minute),
        "5Min":  TimeFrame(5,  TimeFrameUnit.Minute),
        "15Min": TimeFrame(15, TimeFrameUnit.Minute),
        "30Min": TimeFrame(30, TimeFrameUnit.Minute),
        "1Hour": TimeFrame(1,  TimeFrameUnit.Hour),
        "1Day":  TimeFrame(1,  TimeFrameUnit.Day),
    }
    tf     = tf_map.get(timeframe, TimeFrame(1, TimeFrameUnit.Day))
    source = "alpaca"

    # ── Cache check ───────────────────────────────────────────────────────────
    if use_cache:
        fetch_start, fetch_end = _cache.missing_range(
            source, symbol, timeframe, start, end)
        if fetch_start is None:
            cached = _cache.load(source, symbol, timeframe)
            mask   = ((cached["date"] >= start) & (cached["date"] <= end))
            log.info(f"Alpaca CACHE HIT: {symbol}/{timeframe} — "
                     f"serving {mask.sum()} bars from cache, no API call needed")
            return cached[mask].reset_index(drop=True)
    else:
        fetch_start, fetch_end = start, end

    log.info(f"Alpaca FETCH: {symbol} {timeframe} "
             f"{fetch_start.date()} → {fetch_end.date()}")

    # ── Alpaca API call ───────────────────────────────────────────────────────
    client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)

    fetch_failed: Exception | None = None
    bars = pd.DataFrame()
    used_feed = None
    for feed_name in ("sip", "iex"):
        req = StockBarsRequest(
            symbol_or_symbols = symbol,
            timeframe         = tf,
            start             = fetch_start.to_pydatetime(),
            end               = fetch_end.to_pydatetime(),
            feed              = feed_name,
            adjustment        = "all",
        )
        try:
            response = client.get_stock_bars(req)
            bars = response.df
            if bars is not None and not bars.empty:
                used_feed = feed_name
                break
        except Exception as exc:  # offline / DNS / 429 / auth / feed entitlement
            fetch_failed = exc
            bars = pd.DataFrame()
            continue

    # Cache fallback: serve cached range when fresh fetch is unavailable
    # OR when it returned nothing. Without this, an offline backtest would
    # raise even with a fully populated data_cache for the requested range.
    if fetch_failed is not None or bars.empty:
        if use_cache:
            cached = _cache.load(source, symbol, timeframe)
            if cached is not None and not cached.empty:
                mask = ((cached["date"] >= start) & (cached["date"] <= end))
                sliced = cached[mask].reset_index(drop=True)
                if not sliced.empty:
                    if fetch_failed is not None:
                        log.warning(f"Alpaca fetch failed for {symbol}/{timeframe} "
                                    f"({fetch_failed.__class__.__name__}: {fetch_failed}). "
                                    f"Serving {len(sliced)} cached bars instead.")
                    else:
                        log.warning(f"Alpaca returned no new bars for {symbol}/{timeframe}. "
                                    f"Returning {len(sliced)} cached bars.")
                    return sliced
        if fetch_failed is not None:
            raise ValueError(
                f"Alpaca fetch failed for {symbol}/{timeframe} and no cached "
                f"data is available for {start.date()} → {end.date()}: "
                f"{fetch_failed.__class__.__name__}: {fetch_failed}"
            )
        raise ValueError(
            f"No data returned for {symbol} ({timeframe}) "
            f"{fetch_start.date()} → {fetch_end.date()}.\n"
            f"Check: (1) symbol is a US equity/ETF, "
            f"(2) date range includes NYSE trading days, "
            f"(3) Alpaca API keys are valid."
        )
    if used_feed == "iex":
        log.warning(
            f"Alpaca fallback feed in use for {symbol}/{timeframe}: "
            "SIP unavailable, serving IEX history instead."
        )

    # ── Flatten multi-level index (symbol, timestamp) ─────────────────────────
    bars  = bars.reset_index()

    ts_col = None
    for candidate in ["timestamp", "t"]:
        if candidate in bars.columns:
            ts_col = candidate
            break
    if ts_col is None:
        for col in bars.columns:
            if pd.api.types.is_datetime64_any_dtype(bars[col]):
                ts_col = col
                break
    if ts_col is None:
        raise ValueError(f"Cannot find timestamp in Alpaca response. "
                         f"Columns: {list(bars.columns)}")

    bars["date"] = pd.to_datetime(bars[ts_col], utc=True).dt.tz_localize(None)

    new_data = pd.DataFrame({
        "date":   bars["date"],
        "open":   bars["open"].astype(float),
        "high":   bars["high"].astype(float),
        "low":    bars["low"].astype(float),
        "close":  bars["close"].astype(float),
        "volume": bars["volume"].astype(float),
    }).dropna().sort_values("date").reset_index(drop=True)

    log.info(f"Alpaca API: {symbol} {timeframe} → {len(new_data)} new bars "
             f"({new_data['date'].iloc[0].date()} → "
             f"{new_data['date'].iloc[-1].date()})")

    # ── Cache update ──────────────────────────────────────────────────────────
    if use_cache:
        merged = _cache.append(source, symbol, timeframe, new_data)
        mask   = ((merged["date"] >= start) & (merged["date"] <= end))
        return merged[mask].reset_index(drop=True)

    return new_data


def _alpaca_timeframe_from_yfinance_interval(interval: str | None) -> str | None:
    if not interval:
        return None
    mapping = {
        "1m": "1Min",
        "2m": "2Min",
        "5m": "5Min",
        "15m": "15Min",
        "30m": "30Min",
        "1h": "1Hour",
        "1d": "1Day",
    }
    return mapping.get(str(interval).lower())


def _forward_alpaca_credentials() -> tuple[str, str, bool] | None:
    """
    Pick whichever Alpaca credentials are available so forward-paper pages can
    self-seed a cold cache on hosted environments.
    """
    if settings.alpaca.has_paper_credentials():
        return settings.alpaca.paper_api_key, settings.alpaca.paper_secret_key, True
    if settings.alpaca.has_live_credentials():
        return settings.alpaca.live_api_key, settings.alpaca.live_secret_key, False
    return None


def _forward_yahoo_recent_window(interval: str) -> pd.Timedelta | None:
    """Return the maximum recent window we should request from Yahoo.

    For 1-minute bars, Yahoo is the preferred source for the most recent slice,
    while Alpaca is better used for the older warm-up segment. Requesting the
    full 10-day window from Yahoo for 1m often underdelivers because Yahoo only
    serves about 7 recent days at that granularity.
    """
    key = str(interval).lower()
    if key in {"1m", "1min"}:
        return pd.Timedelta(days=7)
    if key in {"2m", "2min", "5m", "5min", "15m", "15min", "30m", "30min", "1h", "1hour"}:
        return pd.Timedelta(days=60)
    return None


def load_forward_blended_data(
    symbol: str,
    interval: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    lookback: int | None = None,
) -> pd.DataFrame:
    """
    Forward-test helper: combine older Alpaca cache with recent Yahoo bars.

    Alpaca gives us the dense historical warm-up cache, while Yahoo can provide
    recent 1-minute bars when Alpaca SIP access is delayed. On a cold hosted
    session, if neither the local Alpaca cache nor Yahoo is enough, this
    function will self-seed the cache from Alpaca history when credentials are
    configured. Yahoo rows still win on overlapping timestamps.
    """
    symbol = symbol.strip().upper()
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    frames: list[pd.DataFrame] = []
    yahoo_recent_window = _forward_yahoo_recent_window(interval)
    yahoo_start_ts = start_ts
    if yahoo_recent_window is not None:
        yahoo_start_ts = max(start_ts, end_ts - yahoo_recent_window)

    alpaca_tf = _alpaca_timeframe_from_yfinance_interval(interval)
    if alpaca_tf:
        cached = _cache.load("alpaca", symbol, alpaca_tf)
        if cached is not None and not cached.empty:
            alpaca_cache_end = end_ts
            if yahoo_recent_window is not None:
                alpaca_cache_end = max(start_ts, end_ts - yahoo_recent_window) - pd.Timedelta(minutes=1)
            mask = (cached["date"] >= start_ts) & (cached["date"] <= alpaca_cache_end)
            frames.append(cached.loc[mask].copy())

    try:
        yahoo = load_from_ticker(symbol, interval, yahoo_start_ts, end_ts, use_cache=True)
        if yahoo is not None and not yahoo.empty:
            frames.append(yahoo.copy())
    except Exception as e:
        log.warning(f"Forward Yahoo refresh failed for {symbol}/{interval}: {e}")

    need_alpaca_seed = False
    creds = _forward_alpaca_credentials()
    if alpaca_tf and creds is not None:
        if not frames:
            need_alpaca_seed = True
        elif lookback is not None and lookback > 0:
            current_rows = sum(len(frame) for frame in frames)
            if current_rows < int(lookback):
                need_alpaca_seed = True

    if need_alpaca_seed:
        api_key, secret_key, paper = creds
        alpaca_seed_start = start_ts
        alpaca_seed_end = end_ts
        if yahoo_recent_window is not None:
            older_cutoff = max(start_ts, end_ts - yahoo_recent_window)
            alpaca_seed_end = older_cutoff - pd.Timedelta(minutes=1)
        if alpaca_seed_end <= alpaca_seed_start:
            alpaca_seed_end = pd.NaT
        try:
            if pd.notna(alpaca_seed_end):
                alpaca_seed = load_from_alpaca_history(
                    symbol,
                    alpaca_tf,
                    alpaca_seed_start,
                    alpaca_seed_end,
                    api_key=api_key,
                    secret_key=secret_key,
                    paper=paper,
                    use_cache=True,
                )
                if alpaca_seed is not None and not alpaca_seed.empty:
                    frames.append(alpaca_seed.copy())
        except Exception as e:
            log.warning(f"Forward Alpaca warm-up failed for {symbol}/{interval}: {e}")

    if not frames:
        raise ValueError(f"No forward data available for {symbol}/{interval}.")

    merged = (
        pd.concat(frames, ignore_index=True)
        .assign(date=lambda df: pd.to_datetime(df["date"], errors="coerce"))
        .dropna(subset=["date"])
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )
    if lookback is not None and lookback > 0:
        merged = merged.tail(int(lookback)).reset_index(drop=True)
    return merged


def _merge_tolerance(interval: str | None) -> pd.Timedelta:
    if not interval:
        return pd.Timedelta(minutes=10)
    key = interval.lower()
    mapping = {
        "1m": pd.Timedelta(minutes=10),
        "1min": pd.Timedelta(minutes=10),
        "5m": pd.Timedelta(minutes=20),
        "5min": pd.Timedelta(minutes=20),
        "15m": pd.Timedelta(minutes=45),
        "15min": pd.Timedelta(minutes=45),
        "30m": pd.Timedelta(minutes=90),
        "30min": pd.Timedelta(minutes=90),
        "1h": pd.Timedelta(hours=3),
        "1hour": pd.Timedelta(hours=3),
        "1d": pd.Timedelta(days=3),
        "1day": pd.Timedelta(days=3),
        "1wk": pd.Timedelta(days=10),
    }
    return mapping.get(key, pd.Timedelta(minutes=10))


def _symbol_prefix(symbol: str) -> str:
    return "".join(ch for ch in symbol.lower() if ch.isalnum())


def _load_companion_data(
    source: str,
    symbol: str,
    interval: str | None,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame | None:
    if source == "alpaca":
        creds = settings.alpaca
        key_ = creds.paper_api_key if not settings.is_live() else creds.live_api_key
        sec_ = creds.paper_secret_key if not settings.is_live() else creds.live_secret_key
        if not key_ or not sec_ or not interval:
            return None
        return load_from_alpaca_history(symbol, interval, start, end, key_, sec_, paper=not settings.is_live())
    if source == "yfinance":
        if not interval:
            return None
        return load_from_ticker(symbol, interval, start, end)
    if source == "forward_blend":
        if not interval:
            return None
        return load_forward_blended_data(symbol, interval, start, end)
    return None


def _strategy_companion_requests(strategy, primary_symbol: str, source: str | None, interval: str | None) -> list[tuple[str, str, str]]:
    requests: list[tuple[str, str, str]] = []
    if strategy is None:
        return requests

    if hasattr(strategy, "companion_contexts"):
        for context_key in strategy.companion_contexts(primary_symbol, source=source, interval=interval) or []:
            resolved = resolve_context_symbol(primary_symbol, context_key)
            if resolved and resolved.strip().upper() != primary_symbol.strip().upper():
                requests.append((context_key, resolved.strip().upper(), context_prefix(context_key)))

    elif hasattr(strategy, "companion_symbols"):
        for symbol in strategy.companion_symbols(primary_symbol, source=source, interval=interval) or []:
            if symbol and symbol.strip().upper() != primary_symbol.strip().upper():
                sym = symbol.strip().upper()
                requests.append((sym.lower(), sym, _symbol_prefix(sym)))

    return requests


def _strategy_derived_requests(strategy, primary_symbol: str, source: str | None, interval: str | None) -> list[str]:
    if strategy is None or not hasattr(strategy, "derived_contexts"):
        return []
    try:
        return list(strategy.derived_contexts(primary_symbol, source=source, interval=interval) or [])
    except Exception as e:
        log.warning(f"Derived context resolution failed for {primary_symbol}: {e}")
        return []


def prepare_strategy_data(
    data: pd.DataFrame,
    strategy,
    primary_symbol: str,
    source: str | None,
    interval: str | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Enrich a primary OHLCV frame with companion-symbol context declared
    by the strategy. Companion bars are fetched through the same cache-backed
    ingestion layer and merged onto the primary timestamps.
    """
    if data is None or data.empty:
        return data

    def _normalize_merge_dates(frame: pd.DataFrame) -> pd.DataFrame:
        if frame is None or frame.empty:
            return frame
        out = frame.copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce", utc=True).dt.tz_localize(None)
        out["date"] = out["date"].astype("datetime64[ns]")
        out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last")
        return out.reset_index(drop=True)

    def _merge_asof_on_date(left: pd.DataFrame, right: pd.DataFrame, tolerance) -> pd.DataFrame:
        left_m = left.copy()
        right_m = right.copy()
        left_m["__merge_ts"] = left_m["date"].astype("int64")
        right_m["__merge_ts"] = right_m["date"].astype("int64")
        tol_ns = pd.Timedelta(tolerance).value if tolerance is not None else None
        merged = pd.merge_asof(
            left_m.sort_values("__merge_ts"),
            right_m.sort_values("__merge_ts").drop(columns=["date"], errors="ignore"),
            on="__merge_ts",
            direction="backward",
            tolerance=tol_ns,
        )
        return merged.drop(columns=["__merge_ts"], errors="ignore")

    companion_requests = _strategy_companion_requests(strategy, primary_symbol, source, interval)
    if not companion_requests or source not in {"alpaca", "yfinance", "forward_blend"}:
        enriched = _normalize_merge_dates(data)
    else:
        enriched = _normalize_merge_dates(data)
        start_ts = pd.Timestamp(start) if start is not None else pd.Timestamp(enriched["date"].min())
        end_ts = pd.Timestamp(end) if end is not None else pd.Timestamp(enriched["date"].max()) + pd.Timedelta(minutes=1)
        tolerance = _merge_tolerance(interval)

        for _context_key, companion_symbol, prefix in companion_requests:
            try:
                companion = _load_companion_data(source, companion_symbol, interval, start_ts, end_ts)
            except Exception as e:
                log.warning(f"Companion load failed for {companion_symbol}/{interval}: {e}")
                continue
            if companion is None or companion.empty:
                continue

            comp = _normalize_merge_dates(companion)
            rename_map = {
                "open": f"{prefix}_open",
                "high": f"{prefix}_high",
                "low": f"{prefix}_low",
                "close": f"{prefix}_close",
                "volume": f"{prefix}_volume",
            }
            comp = comp.rename(columns=rename_map)
            keep_cols = ["date", *rename_map.values()]
            enriched = _merge_asof_on_date(enriched, comp[keep_cols], tolerance)

    derived_requests = _strategy_derived_requests(strategy, primary_symbol, source, interval)
    if "gold_fair_value" in derived_requests and primary_symbol.strip().upper() == "GLD":
        start_ts = pd.Timestamp(start) if start is not None else pd.Timestamp(enriched["date"].min())
        end_ts = pd.Timestamp(end) if end is not None else pd.Timestamp(enriched["date"].max())
        try:
            resolved = strategy.resolve_params(
                symbol=primary_symbol,
                source=source,
                interval=interval,
            )
            enriched = prepare_gld_fair_value_context(
                enriched,
                start=start_ts,
                end=end_ts,
                bullish_slope_min=float(resolved.get("gold_fair_value_bullish_slope_min", 1.0)),
                bearish_slope_max=float(resolved.get("gold_fair_value_bearish_slope_max", -1.0)),
                undervalued_gap_pct=float(resolved.get("gold_fair_value_undervalued_gap_pct", 2.5)),
                overvalued_gap_pct=float(resolved.get("gold_fair_value_overvalued_gap_pct", 2.5)),
            )
        except Exception as e:
            log.warning(f"Derived GLD fair-value context failed: {e}")

    out = enriched.reset_index(drop=True)
    try:
        out.attrs["_strategy_symbol"] = str(primary_symbol or "").strip().upper()
        out.attrs["_strategy_source"] = str(source or "").strip().lower()
        out.attrs["_strategy_interval"] = str(interval or "").strip().lower()
    except Exception:
        pass
    return out


def prefetch_strategy_companions(
    strategy,
    primary_symbol: str,
    source: str | None,
    interval: str | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> list[str]:
    """
    Warm companion-symbol caches for a strategy using the normal cache-aware
    loaders. Existing cached data is reused and only missing bars are appended.
    """
    if strategy is None:
        return []

    companion_requests = _strategy_companion_requests(strategy, primary_symbol, source, interval)
    if not companion_requests or source not in {"alpaca", "yfinance", "forward_blend"}:
        return []

    start_ts = pd.Timestamp(start) if start is not None else None
    end_ts = pd.Timestamp(end) if end is not None else None
    if start_ts is None or end_ts is None:
        return []

    loaded: list[str] = []
    for context_key, companion_symbol, _prefix in companion_requests:
        try:
            companion = _load_companion_data(source, companion_symbol, interval, start_ts, end_ts)
        except Exception as e:
            log.warning(f"Companion prefetch failed for {companion_symbol}/{interval}: {e}")
            continue
        if companion is not None and not companion.empty:
            loaded.append(f"{context_label(context_key)}: {companion_symbol}")
    return loaded
