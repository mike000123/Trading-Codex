"""
data/cache.py
─────────────
Local OHLCV data cache — avoids re-downloading data already fetched.

Structure on disk:
  data_cache/
    alpaca/
      UVXY/
        1Min.csv
        5Min.csv
      GLD/
        1Min.csv
    yfinance/
      UVXY/
        1m.csv
      GC=F/
        5m.csv
    csv_upload/
      (user-uploaded files stored as-is)

CSV format (all sources normalised):
  date, open, high, low, close, volume
  date is UTC-naive ISO timestamp: 2024-03-25 09:31:00

Usage:
  from data.cache import DataCache
  cache = DataCache()
  df = cache.load("alpaca", "UVXY", "1Min")       # None if not cached
  cache.save("alpaca", "UVXY", "1Min", df)
  gap_start, gap_end = cache.missing_range(
      "alpaca", "UVXY", "1Min",
      requested_start, requested_end)              # returns what still needs fetching
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from core.logger import log

# Root cache directory — relative to project root
_CACHE_ROOT = Path("data_cache")


def _cache_path(source: str, symbol: str, timeframe: str) -> Path:
    """
    Returns the path to the CSV file for this source/symbol/timeframe.
    source    : "alpaca" | "yfinance" | "csv_upload"
    symbol    : "UVXY", "GC=F", "GLD" etc.
    timeframe : "1Min", "5Min", "1m", "5m" etc.
    """
    # Sanitise symbol for filesystem (GC=F → GC_F)
    safe_symbol    = symbol.replace("=", "_").replace("/", "_").replace("\\", "_")
    safe_timeframe = timeframe.replace(" ", "_")
    return _CACHE_ROOT / source / safe_symbol / f"{safe_timeframe}.csv"


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


class DataCache:
    """
    Persistent local OHLCV cache.
    Thread-safe for single-process Streamlit apps.
    """

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = root or _CACHE_ROOT

    def path(self, source: str, symbol: str, timeframe: str) -> Path:
        safe_sym = symbol.replace("=","_").replace("/","_")
        safe_tf  = timeframe.replace(" ", "_")
        return self.root / source / safe_sym / f"{safe_tf}.csv"

    def exists(self, source: str, symbol: str, timeframe: str) -> bool:
        return self.path(source, symbol, timeframe).exists()

    def load(self, source: str, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Load cached data. Returns None if no cache exists."""
        p = self.path(source, symbol, timeframe)
        if not p.exists():
            return None
        try:
            df = pd.read_csv(p, parse_dates=["date"])
            df = df.sort_values("date").reset_index(drop=True)
            log.debug(f"Cache HIT: {source}/{symbol}/{timeframe} "
                      f"— {len(df)} bars "
                      f"({df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()})")
            return df
        except Exception as e:
            log.warning(f"Cache read failed for {p}: {e} — treating as miss")
            return None

    def save(self, source: str, symbol: str, timeframe: str,
             df: pd.DataFrame) -> None:
        """Save (or overwrite) cache for this source/symbol/timeframe."""
        p = self.path(source, symbol, timeframe)
        _ensure_dir(p)
        df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
        df.to_csv(p, index=False)
        log.info(f"Cache SAVED: {source}/{symbol}/{timeframe} "
                 f"— {len(df)} bars "
                 f"({df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()})")

    def append(self, source: str, symbol: str, timeframe: str,
               new_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge new_df with existing cache.
        Deduplicates on date, keeps all rows, sorted.
        Returns the merged DataFrame and also saves it.
        """
        existing = self.load(source, symbol, timeframe)
        if existing is None or existing.empty:
            merged = new_df
        else:
            merged = (pd.concat([existing, new_df], ignore_index=True)
                        .drop_duplicates(subset=["date"])
                        .sort_values("date")
                        .reset_index(drop=True))
        self.save(source, symbol, timeframe, merged)
        new_count = len(merged) - (len(existing) if existing is not None else 0)
        log.info(f"Cache APPEND: {source}/{symbol}/{timeframe} "
                 f"+{new_count} new bars → {len(merged)} total")
        return merged

    def missing_range(
        self,
        source:           str,
        symbol:           str,
        timeframe:        str,
        requested_start:  pd.Timestamp,
        requested_end:    pd.Timestamp,
    ) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """
        Given a requested date range, return the sub-range that is NOT yet cached.

        Returns (fetch_start, fetch_end) where:
          - fetch_start = max(requested_start, last_cached_date + 1 bar)
          - fetch_end   = requested_end

        Returns (None, None) if the full requested range is already cached.

        Note: we always re-fetch from the last cached timestamp onwards to catch
        any late-arriving bars from the previous session.
        """
        existing = self.load(source, symbol, timeframe)

        if existing is None or existing.empty:
            # Nothing cached at all — fetch everything
            return requested_start, requested_end

        first_cached = existing["date"].min()
        last_cached  = existing["date"].max()

        # If the requested range is entirely before what we have, nothing to do
        if requested_end <= first_cached:
            return None, None

        # Fetch from the last cached bar onward so we catch any new bars
        # (overlap by 1 period ensures no gaps at the boundary)
        fetch_start = last_cached

        if fetch_start >= requested_end:
            return None, None

        return fetch_start, requested_end

    def list_cached(self) -> list[dict]:
        """List all cached datasets with metadata."""
        results = []
        if not self.root.exists():
            return results
        for csv_path in sorted(self.root.rglob("*.csv")):
            parts = csv_path.relative_to(self.root).parts
            if len(parts) != 3:
                continue
            source, symbol, tf_file = parts
            timeframe = tf_file.replace(".csv", "")
            try:
                df  = pd.read_csv(csv_path, parse_dates=["date"])
                results.append({
                    "source":    source,
                    "symbol":    symbol.replace("_", "="),
                    "timeframe": timeframe,
                    "bars":      len(df),
                    "from":      str(df["date"].min().date()),
                    "to":        str(df["date"].max().date()),
                    "size_kb":   round(csv_path.stat().st_size / 1024, 1),
                    "path":      str(csv_path),
                })
            except Exception:
                pass
        return results

    def delete(self, source: str, symbol: str, timeframe: str) -> bool:
        """Delete a cached file. Returns True if deleted."""
        p = self.path(source, symbol, timeframe)
        if p.exists():
            p.unlink()
            log.info(f"Cache DELETED: {source}/{symbol}/{timeframe}")
            return True
        return False

    def clear_all(self) -> int:
        """Delete all cache files. Returns count deleted."""
        count = 0
        if self.root.exists():
            for f in self.root.rglob("*.csv"):
                f.unlink()
                count += 1
        return count
