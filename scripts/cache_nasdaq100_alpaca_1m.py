from __future__ import annotations

import argparse
import json
import sys
import time
import types
from dataclasses import asdict, dataclass
from pathlib import Path

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

from config.settings import settings
from config.equity_universes import NASDAQ100_TICKERS
from data.cache import DataCache
from data.ingestion import load_from_alpaca_history


ARTIFACT_DIR = ROOT / "artifacts" / "cache_jobs"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

SAFE_ALPACA_BUFFER_DAYS = 3
OFFICIAL_SOURCE_URL = "https://www.nasdaq.com/solutions/global-indexes/nasdaq-100/companies"
OFFICIAL_SOURCE_NOTE = (
    "Verified against Nasdaq's official Nasdaq-100 companies page on 2026-05-04. "
    "The page text exposed the full constituent list and showed a footer note "
    "reading 'Last updated 05/19/2025'."
)

@dataclass
class CacheResult:
    symbol: str
    status: str
    bars: int
    first_bar: str | None
    last_bar: str | None
    cache_csv: str
    cache_pkl: str
    message: str = ""


def _artifact_path(stem: str, suffix: str) -> Path:
    return ARTIFACT_DIR / f"{stem}{suffix}"


def _default_end() -> pd.Timestamp:
    return pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=SAFE_ALPACA_BUFFER_DAYS)


def _selected_tickers(raw_symbols: str) -> list[str]:
    if not str(raw_symbols).strip():
        return list(NASDAQ100_TICKERS)
    requested = [part.strip().upper() for part in str(raw_symbols).split(",") if part.strip()]
    known = set(NASDAQ100_TICKERS)
    invalid = [sym for sym in requested if sym not in known]
    if invalid:
        raise SystemExit(
            "Unknown Nasdaq-100 symbols requested: "
            + ", ".join(invalid)
            + ".\nUse a comma-separated subset of the verified current Nasdaq-100 list."
        )
    return requested


def _write_artifacts(stem: str, payload: dict, rows: list[CacheResult]) -> None:
    json_path = _artifact_path(stem, ".json")
    csv_path = _artifact_path(stem, ".csv")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame([asdict(row) for row in rows]).to_csv(csv_path, index=False)


def _status_for_range(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> tuple[int, str | None, str | None]:
    if df is None or df.empty:
        return 0, None, None
    dates = pd.to_datetime(df["date"], errors="coerce").dropna()
    if dates.empty:
        return 0, None, None
    mask = (dates >= start) & (dates <= end)
    sliced = dates[mask]
    if sliced.empty:
        return 0, None, None
    return int(len(sliced)), sliced.min().isoformat(), sliced.max().isoformat()


def _fetch_symbol(symbol: str, start: pd.Timestamp, end: pd.Timestamp, timeframe: str, cache: DataCache) -> CacheResult:
    csv_path = cache.path("alpaca", symbol, timeframe)
    pkl_path = cache.binary_path("alpaca", symbol, timeframe)
    try:
        df = load_from_alpaca_history(
            symbol,
            timeframe,
            start,
            end,
            settings.alpaca.paper_api_key,
            settings.alpaca.paper_secret_key,
            paper=True,
            use_cache=True,
        )
        bars, first_bar, last_bar = _status_for_range(df, start, end)
        status = "ok" if bars > 0 else "empty"
        message = ""
        if bars == 0:
            message = "No bars returned in the requested window."
        return CacheResult(
            symbol=symbol,
            status=status,
            bars=bars,
            first_bar=first_bar,
            last_bar=last_bar,
            cache_csv=str(csv_path),
            cache_pkl=str(pkl_path),
            message=message,
        )
    except Exception as exc:
        return CacheResult(
            symbol=symbol,
            status="error",
            bars=0,
            first_bar=None,
            last_bar=None,
            cache_csv=str(csv_path),
            cache_pkl=str(pkl_path),
            message=f"{exc.__class__.__name__}: {exc}",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download 1-minute Alpaca history for the verified current Nasdaq-100 "
            "constituents into data_cache/alpaca."
        )
    )
    parser.add_argument("--start", default="2024-01-01", help="Start date for the cache job (default: 2024-01-01).")
    parser.add_argument("--end", default="", help="Optional end date; defaults to now minus a safe Alpaca buffer.")
    parser.add_argument("--timeframe", default="1Min", help="Alpaca timeframe to cache (default: 1Min).")
    parser.add_argument("--symbols", default="", help="Optional comma-separated subset of the verified Nasdaq-100 symbols.")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit for testing; 0 means all selected symbols.")
    parser.add_argument("--pause-seconds", type=float, default=0.15, help="Small pause between symbols (default: 0.15s).")
    parser.add_argument("--artifact-stem", default="nasdaq100_alpaca_1m_cache", help="Stem for the progress artifacts.")
    args = parser.parse_args()

    if not settings.alpaca.has_paper_credentials():
        raise SystemExit("Alpaca paper credentials are required for this cache job.")

    start = pd.Timestamp(args.start).tz_localize(None)
    end = pd.Timestamp(args.end).tz_localize(None) if str(args.end).strip() else _default_end()
    timeframe = str(args.timeframe).strip() or "1Min"

    tickers = _selected_tickers(args.symbols)
    if args.limit and args.limit > 0:
        tickers = tickers[: int(args.limit)]

    cache = DataCache()
    rows: list[CacheResult] = []
    total = len(tickers)

    print(f"Caching {total} verified Nasdaq-100 symbols from Alpaca ({timeframe})")
    print(f"Window: {start.date()} -> {end.date()}")
    print(f"Source: {OFFICIAL_SOURCE_URL}")
    print(OFFICIAL_SOURCE_NOTE)
    print()

    for idx, symbol in enumerate(tickers, start=1):
        print(f"[{idx:03d}/{total:03d}] {symbol} ...", flush=True)
        result = _fetch_symbol(symbol, start, end, timeframe, cache)
        rows.append(result)

        ok_count = sum(1 for row in rows if row.status == "ok")
        empty_count = sum(1 for row in rows if row.status == "empty")
        error_count = sum(1 for row in rows if row.status == "error")

        payload = {
            "job": {
                "artifact_stem": args.artifact_stem,
                "verified_source_url": OFFICIAL_SOURCE_URL,
                "verified_source_note": OFFICIAL_SOURCE_NOTE,
                "timeframe": timeframe,
                "window": {
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                },
                "selected_symbols": tickers,
                "completed_symbols": idx,
                "total_symbols": total,
            },
            "summary": {
                "ok": ok_count,
                "empty": empty_count,
                "error": error_count,
            },
            "results": [asdict(row) for row in rows],
        }
        _write_artifacts(args.artifact_stem, payload, rows)

        if result.status == "ok":
            print(
                f"       ok  {result.bars} bars  "
                f"{result.first_bar or '?'} -> {result.last_bar or '?'}",
                flush=True,
            )
        elif result.status == "empty":
            print("       empty  no bars in requested window", flush=True)
        else:
            print(f"       error  {result.message}", flush=True)

        if args.pause_seconds > 0 and idx < total:
            time.sleep(args.pause_seconds)

    print()
    print("Finished.")
    print(f"JSON summary: {_artifact_path(args.artifact_stem, '.json')}")
    print(f"CSV summary : {_artifact_path(args.artifact_stem, '.csv')}")


if __name__ == "__main__":
    main()
