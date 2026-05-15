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
from threading import Lock
import time

import pandas as pd

from core.logger import log

# Root cache directory — relative to project root
_CACHE_ROOT = Path("data_cache")
_MEMO_LOCK = Lock()
_FRAME_MEMO: dict[str, tuple[tuple[bool, int, int], pd.DataFrame]] = {}
_LOAD_LOCKS_GUARD = Lock()
_LOAD_LOCKS: dict[str, Lock] = {}
_RANGE_CHUNK_SIZE = 50_000


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


def _load_lock_for(key: str) -> Lock:
    with _LOAD_LOCKS_GUARD:
        lock = _LOAD_LOCKS.get(key)
        if lock is None:
            lock = Lock()
            _LOAD_LOCKS[key] = lock
        return lock


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

    def binary_path(self, source: str, symbol: str, timeframe: str) -> Path:
        safe_sym = symbol.replace("=","_").replace("/","_")
        safe_tf  = timeframe.replace(" ", "_")
        return self.root / source / safe_sym / f"{safe_tf}.pkl"

    @staticmethod
    def _file_signature(path: Path) -> tuple[bool, int, int]:
        if not path.exists():
            return (False, 0, 0)
        stat = path.stat()
        return (True, int(stat.st_mtime_ns), int(stat.st_size))

    @staticmethod
    def _memo_key(path: Path, start: Optional[pd.Timestamp] = None, end: Optional[pd.Timestamp] = None) -> str:
        try:
            base = str(path.resolve())
        except Exception:
            base = str(path)
        if start is None and end is None:
            return base
        start_txt = pd.Timestamp(start).isoformat() if start is not None else ""
        end_txt = pd.Timestamp(end).isoformat() if end is not None else ""
        return f"{base}|{start_txt}|{end_txt}"

    def _memo_get(
        self,
        path: Path,
        sig: tuple[bool, int, int],
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
    ) -> Optional[pd.DataFrame]:
        key = self._memo_key(path, start, end)
        with _MEMO_LOCK:
            payload = _FRAME_MEMO.get(key)
            if payload is None:
                return None
            cached_sig, cached_df = payload
            if cached_sig != sig:
                _FRAME_MEMO.pop(key, None)
                return None
            return cached_df.copy()

    def _memo_put(
        self,
        path: Path,
        sig: tuple[bool, int, int],
        df: pd.DataFrame,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        key = self._memo_key(path, start, end)
        frame = df.copy()
        with _MEMO_LOCK:
            _FRAME_MEMO[key] = (sig, frame)
        return frame.copy()

    def _memo_drop(self, *paths: Path) -> None:
        with _MEMO_LOCK:
            for path in paths:
                try:
                    base = str(path.resolve())
                except Exception:
                    base = str(path)
                to_drop = [key for key in _FRAME_MEMO if key == base or key.startswith(base + "|")]
                for key in to_drop:
                    _FRAME_MEMO.pop(key, None)

    def exists(self, source: str, symbol: str, timeframe: str) -> bool:
        return self.path(source, symbol, timeframe).exists() or self.binary_path(source, symbol, timeframe).exists()

    @staticmethod
    def _load_csv_window(
        csv_path: Path,
        *,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        start_ts = pd.Timestamp(start) if start is not None else None
        end_ts = pd.Timestamp(end) if end is not None else None
        chunks: list[pd.DataFrame] = []
        for chunk in pd.read_csv(csv_path, parse_dates=["date"], chunksize=_RANGE_CHUNK_SIZE):
            if "date" not in chunk.columns:
                continue
            chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
            chunk = chunk.dropna(subset=["date"])
            if chunk.empty:
                continue
            first_date = chunk["date"].iloc[0]
            last_date = chunk["date"].iloc[-1]
            if start_ts is not None and last_date < start_ts:
                continue
            if end_ts is not None and first_date > end_ts:
                break
            if start_ts is not None:
                chunk = chunk.loc[chunk["date"] >= start_ts]
            if end_ts is not None:
                chunk = chunk.loc[chunk["date"] <= end_ts]
            if not chunk.empty:
                chunks.append(chunk)
            if end_ts is not None and last_date >= end_ts:
                break
        if not chunks:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
        return pd.concat(chunks, ignore_index=True)

    def load(
        self,
        source: str,
        symbol: str,
        timeframe: str,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
    ) -> Optional[pd.DataFrame]:
        """Load cached data. Returns None if no cache exists."""
        csv_path = self.path(source, symbol, timeframe)
        pkl_path = self.binary_path(source, symbol, timeframe)
        if not csv_path.exists() and not pkl_path.exists():
            return None
        started_at = time.perf_counter()
        start_ts = pd.Timestamp(start) if start is not None else None
        end_ts = pd.Timestamp(end) if end is not None else None
        has_bounds = start_ts is not None or end_ts is not None
        try:
            csv_sig = self._file_signature(csv_path)
            pkl_sig = self._file_signature(pkl_path)
            preferred_path = csv_path
            if pkl_sig[0] and (not csv_sig[0] or pkl_sig[1] >= csv_sig[1]):
                preferred_path = pkl_path
            fallback_path = csv_path if preferred_path == pkl_path and csv_sig[0] else None
            preferred_sig = pkl_sig if preferred_path == pkl_path else csv_sig

            memo_df = self._memo_get(preferred_path, preferred_sig, start_ts, end_ts)
            if memo_df is not None:
                log.debug(f"Cache MEMO HIT: {source}/{symbol}/{timeframe} — {len(memo_df)} bars")
                return memo_df

            load_lock_key = self._memo_key(preferred_path, start_ts, end_ts)
            with _load_lock_for(load_lock_key):
                memo_df = self._memo_get(preferred_path, preferred_sig, start_ts, end_ts)
                if memo_df is not None:
                    log.debug(f"Cache MEMO HIT(after-lock): {source}/{symbol}/{timeframe} — {len(memo_df)} bars")
                    return memo_df

                try:
                    if has_bounds and preferred_path == csv_path:
                        df = self._load_csv_window(csv_path, start=start_ts, end=end_ts)
                        preferred_path = csv_path
                        preferred_sig = csv_sig
                    elif preferred_path == pkl_path:
                        df = pd.read_pickle(pkl_path)
                        if not isinstance(df, pd.DataFrame):
                            raise ValueError("Binary cache did not contain a DataFrame.")
                    else:
                        df = pd.read_csv(csv_path, parse_dates=["date"])
                except Exception:
                    if fallback_path is None:
                        raise
                    df = pd.read_csv(fallback_path, parse_dates=["date"])
                    preferred_path = fallback_path
                    preferred_sig = csv_sig
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
            if has_bounds:
                if start_ts is not None:
                    df = df.loc[df["date"] >= start_ts]
                if end_ts is not None:
                    df = df.loc[df["date"] <= end_ts]
            if self._is_intraday_cache_corrupt(df, timeframe):
                log.warning(
                    f"Cache CORRUPT: {source}/{symbol}/{timeframe} at {preferred_path} "
                    "appears to have lost intraday timestamps; treating as miss"
                )
                self._memo_drop(csv_path, pkl_path)
                return None
            df = df.sort_values("date").reset_index(drop=True)
            if not pkl_path.exists() and csv_path.exists():
                try:
                    _ensure_dir(pkl_path)
                    df.to_pickle(pkl_path)
                except Exception as pickle_exc:
                    log.debug(f"Binary sidecar write skipped for {csv_path}: {pickle_exc}")
            elapsed = time.perf_counter() - started_at
            if elapsed >= 1.0:
                loaded_mb = 0.0
                try:
                    loaded_mb = preferred_path.stat().st_size / (1024 * 1024)
                except Exception:
                    loaded_mb = 0.0
                bounds_txt = ""
                if has_bounds:
                    bounds_txt = (
                        f" for {start_ts.date() if start_ts is not None else '...'}"
                        f" → {end_ts.date() if end_ts is not None else '...'}"
                    )
                log.info(
                    f"Cache LOAD SLOW: {source}/{symbol}/{timeframe} from {preferred_path.name} "
                    f"— {len(df):,} bars{bounds_txt} in {elapsed:.2f}s ({loaded_mb:.2f} MB)"
                )
            if not df.empty:
                log.debug(f"Cache HIT: {source}/{symbol}/{timeframe} "
                          f"— {len(df)} bars "
                          f"({df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()})")
            return self._memo_put(preferred_path, preferred_sig, df, start_ts, end_ts)
        except Exception as e:
            self._memo_drop(csv_path, pkl_path)
            log.warning(f"Cache read failed for {preferred_path if 'preferred_path' in locals() else csv_path}: {e} — treating as miss")
            return None

    def save(self, source: str, symbol: str, timeframe: str,
             df: pd.DataFrame) -> None:
        """Save (or overwrite) cache for this source/symbol/timeframe."""
        csv_path = self.path(source, symbol, timeframe)
        pkl_path = self.binary_path(source, symbol, timeframe)
        _ensure_dir(csv_path)
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
        df.to_csv(csv_path, index=False, date_format="%Y-%m-%d %H:%M:%S")
        try:
            df.to_pickle(pkl_path)
        except Exception as pickle_exc:
            log.debug(f"Binary cache write failed for {pkl_path}: {pickle_exc}")
        csv_sig = self._file_signature(csv_path)
        pkl_sig = self._file_signature(pkl_path)
        self._memo_put(csv_path, csv_sig, df)
        if pkl_sig[0]:
            self._memo_put(pkl_path, pkl_sig, df)
        log.info(f"Cache SAVED: {source}/{symbol}/{timeframe} "
                 f"— {len(df)} bars "
                 f"({df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()})")

    @staticmethod
    def _is_intraday_cache_corrupt(df: pd.DataFrame, timeframe: str) -> bool:
        tf = str(timeframe).lower()
        intraday = any(token in tf for token in ("min", "hour", "h"))
        if not intraday or df.empty:
            return False
        dates = pd.to_datetime(df["date"], errors="coerce")
        if dates.isna().all():
            return True
        midnight_only = (
            (dates.dt.hour.fillna(0) == 0)
            & (dates.dt.minute.fillna(0) == 0)
            & (dates.dt.second.fillna(0) == 0)
        ).all()
        repeated_days = dates.dt.normalize().duplicated().any()
        return bool(midnight_only and repeated_days)

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

        # If the request starts before the earliest cached bar, we are missing
        # a front segment. Re-fetch the whole requested window and let append()
        # deduplicate. This keeps caches consistent across environments even
        # when one host already has only the later part of the desired range.
        if requested_start < first_cached:
            return requested_start, requested_end

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
        csv_path = self.path(source, symbol, timeframe)
        pkl_path = self.binary_path(source, symbol, timeframe)
        deleted = False
        if csv_path.exists():
            csv_path.unlink()
            deleted = True
        if pkl_path.exists():
            pkl_path.unlink()
            deleted = True
        if deleted:
            self._memo_drop(csv_path, pkl_path)
            log.info(f"Cache DELETED: {source}/{symbol}/{timeframe}")
        return deleted

    def clear_all(self) -> int:
        """Delete all cache files. Returns count deleted."""
        count = 0
        if self.root.exists():
            for pattern in ("*.csv", "*.pkl"):
                for f in self.root.rglob(pattern):
                    f.unlink()
                    count += 1
        with _MEMO_LOCK:
            _FRAME_MEMO.clear()
        return count

    def migrate_csv_sidecars(self) -> dict[str, int]:
        """
        Create or refresh .pkl sidecars for all cached CSV datasets.

        This is a one-time/bulk accelerator for existing caches. Normal app
        loads already do lazy sidecar creation, but this method lets us warm
        the whole cache tree in one pass.
        """
        created = 0
        refreshed = 0
        skipped = 0
        failed = 0

        if not self.root.exists():
            return {"created": 0, "refreshed": 0, "skipped": 0, "failed": 0}

        for csv_path in sorted(self.root.rglob("*.csv")):
            pkl_path = csv_path.with_suffix(".pkl")
            csv_sig = self._file_signature(csv_path)
            pkl_sig = self._file_signature(pkl_path)
            if pkl_sig[0] and pkl_sig[1] >= csv_sig[1]:
                skipped += 1
                continue
            try:
                had_pkl = pkl_path.exists()
                df = pd.read_csv(csv_path, parse_dates=["date"])
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = (
                    df.dropna(subset=["date"])
                    .sort_values("date")
                    .drop_duplicates(subset=["date"])
                    .reset_index(drop=True)
                )
                _ensure_dir(pkl_path)
                df.to_pickle(pkl_path)
                csv_sig = self._file_signature(csv_path)
                pkl_sig = self._file_signature(pkl_path)
                self._memo_put(csv_path, csv_sig, df)
                self._memo_put(pkl_path, pkl_sig, df)
                if had_pkl:
                    refreshed += 1
                else:
                    created += 1
            except Exception as exc:
                failed += 1
                log.warning(f"Cache sidecar migration failed for {csv_path}: {exc}")

        return {
            "created": created,
            "refreshed": refreshed,
            "skipped": skipped,
            "failed": failed,
        }
