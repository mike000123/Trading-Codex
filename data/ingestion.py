"""
data/ingestion.py
─────────────────
Unified data loading layer.
Returns clean pandas DataFrames with columns: date, open, high, low, close, volume.
"""
from __future__ import annotations

import io
from typing import Optional

import pandas as pd
import yfinance as yf

from core.logger import log


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
            "date": pd.to_datetime(df[col_map["date"]], errors="coerce", utc=True).dt.tz_localize(None),
            "open": _to_numeric(df[col_map["open"]]),
            "high": _to_numeric(df[col_map["high"]]),
            "low": _to_numeric(df[col_map["low"]]),
            "close": _to_numeric(df[col_map["close"]]),
            "volume": _to_numeric(df[volume_col]) if volume_col else pd.Series(0.0, index=df.index),
        }
    ).dropna(subset=["date", "open", "high", "low", "close"])

    return (
        normalized
        .sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )


# ─── Public API ──────────────────────────────────────────────────────────────

def load_from_csv(file_buffer) -> pd.DataFrame:
    """Parse an uploaded CSV file into a standard OHLCV DataFrame."""
    raw = file_buffer.read() if hasattr(file_buffer, "read") else file_buffer
    if not raw:
        raise ValueError("The uploaded file is empty.")

    df = pd.read_csv(io.BytesIO(raw) if isinstance(raw, bytes) else io.StringIO(raw))
    canon = {_canonicalize(c): c for c in df.columns}

    aliases: dict[str, list[str]] = {
        "date": ["date", "datetime", "time", "timestamp"],
        "open": ["open", "openingprice"],
        "high": ["high", "max"],
        "low": ["low", "min"],
        "close": ["close", "closelast", "closeprice", "last"],
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

    log.info(f"CSV loaded: {len(result)} rows {result['date'].min().date()} → {result['date'].max().date()}")
    return result


def load_from_ticker(
    ticker: str,
    interval: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Fetch OHLCV from Yahoo Finance via yfinance."""
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("Ticker cannot be empty.")
    if end_date < start_date:
        raise ValueError("End date must be on or after start date.")

    log.info(f"Fetching {ticker} {interval} {start_date.date()} → {end_date.date()}")

    data = yf.download(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=(end_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval=interval,
        auto_adjust=False,
        progress=False,
    )

    if data is None or data.empty:
        raise ValueError(f"No data returned for {ticker} / {interval}.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close"]
    for col in required:
        if col not in data.columns:
            raise ValueError(f"Fetched data missing column: {col}")

    vol_col = "Volume" if "Volume" in data.columns else None
    raw = data.copy()
    raw["_date"] = pd.to_datetime(data.index, errors="coerce", utc=True).tz_localize(None)
    col_map = {"date": "_date", "open": "Open", "high": "High", "low": "Low", "close": "Close"}
    if vol_col:
        col_map["volume"] = vol_col

    result = _normalize_df(raw, col_map)
    if result.empty:
        raise ValueError("No valid rows after fetching ticker data.")

    log.info(f"Fetched {ticker}: {len(result)} bars")
    return result


def load_from_alpaca_history(
    symbol: str,
    timeframe: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    api_key: str,
    secret_key: str,
    paper: bool = True,
) -> pd.DataFrame:
    """
    Fetch historical bars from Alpaca (paper or live endpoint).
    Requires alpaca-py.
    """
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    except ImportError:
        raise ImportError("alpaca-py not installed. Run: pip install alpaca-py")

    tf_map = {
        "1Min": TimeFrame(1, TimeFrameUnit.Minute),
        "5Min": TimeFrame(5, TimeFrameUnit.Minute),
        "15Min": TimeFrame(15, TimeFrameUnit.Minute),
        "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
        "1Day": TimeFrame(1, TimeFrameUnit.Day),
    }
    tf = tf_map.get(timeframe, TimeFrame(1, TimeFrameUnit.Day))

    client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=tf,
        start=start.to_pydatetime(),
        end=end.to_pydatetime(),
    )
    bars = client.get_stock_bars(req).df

    if bars.empty:
        raise ValueError(f"No Alpaca data returned for {symbol}.")

    bars = bars.reset_index()
    bars["date"] = pd.to_datetime(bars["timestamp"], utc=True).dt.tz_localize(None)

    result = pd.DataFrame(
        {
            "date": bars["date"],
            "open": bars["open"].astype(float),
            "high": bars["high"].astype(float),
            "low": bars["low"].astype(float),
            "close": bars["close"].astype(float),
            "volume": bars["volume"].astype(float),
        }
    ).dropna().sort_values("date").reset_index(drop=True)

    log.info(f"Alpaca historical data: {symbol} {len(result)} bars")
    return result
