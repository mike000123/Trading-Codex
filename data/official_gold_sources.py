"""
Helpers for trusted slow gold-related data sources.

This module keeps official-source parsing separate from the trading logic so
the fair-value layer can consume cached monthly proxies regardless of where
they came from.

Current supported workflow:
- SPDR GLD historical archive XLSX -> cached monthly ETF holdings proxy

Prepared workflows:
- IMF / IFS style monthly central-bank reserve series from CSV/DataFrame
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd

from data.xlsx_xml import sheet_to_frame


DEFAULT_DBNOMICS_IMF_GOLD_RESERVE_AREAS = (
    "US",
    "DE",
    "IT",
    "FR",
    "RU",
    "CN",
    "CH",
    "IN",
    "JP",
    "TR",
    "NL",
    "PL",
    "KZ",
)


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )


def _write_proxy_ohlcv(frame: pd.DataFrame, path: Path, value_col: str) -> None:
    out = frame[["date", value_col]].copy()
    out["open"] = out[value_col]
    out["high"] = out[value_col]
    out["low"] = out[value_col]
    out["close"] = out[value_col]
    out["volume"] = 0.0
    out = out[["date", "open", "high", "low", "close", "volume"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, date_format="%Y-%m-%d %H:%M:%S")


def cache_monthly_proxy_series(
    frame: pd.DataFrame,
    *,
    date_col: str,
    value_col: str,
    cache_root: str | Path = "data_cache",
    symbol: str,
) -> tuple[Path, Path]:
    cache_root = Path(cache_root)
    work = frame.copy()
    work["date"] = pd.to_datetime(work[date_col], errors="coerce")
    work["proxy"] = _safe_numeric(work[value_col])
    work = work.dropna(subset=["date", "proxy"]).sort_values("date").drop_duplicates("date")
    if work.empty:
        raise ValueError("No valid rows to cache")

    daily_path = cache_root / "custom" / symbol / "1D.csv"
    monthly_path = cache_root / "custom" / symbol / "1M.csv"

    _write_proxy_ohlcv(work[["date", "proxy"]], daily_path, "proxy")
    monthly = (
        work.set_index("date")[["proxy"]]
        .resample("MS")
        .last()
        .dropna()
        .reset_index()
    )
    _write_proxy_ohlcv(monthly, monthly_path, "proxy")
    return daily_path, monthly_path


def cache_spdr_gld_holdings_from_xlsx(
    xlsx_path: str | Path,
    *,
    cache_root: str | Path = "data_cache",
    symbol: str = "GLD_ETF_PROXY",
) -> tuple[Path, Path]:
    """
    Parse the official SPDR GLD historical archive workbook and cache the
    'Tonnes of Gold' series as a slow proxy.
    """
    df = sheet_to_frame(xlsx_path, "US GLD Historical Archive", header_row=1)
    if df.empty:
        raise ValueError("SPDR archive sheet is empty")

    df = df.rename(columns=lambda c: str(c).strip())
    if "Date" not in df.columns or "Tonnes of Gold" not in df.columns:
        raise ValueError("SPDR archive missing expected columns")

    holidays = df["Tonnes of Gold"].astype(str).str.contains("Holiday", case=False, na=False)
    parsed = df.loc[~holidays, ["Date", "Tonnes of Gold"]].copy()
    parsed["Date"] = pd.to_datetime(parsed["Date"], format="%d-%b-%Y", errors="coerce")
    return cache_monthly_proxy_series(
        parsed,
        date_col="Date",
        value_col="Tonnes of Gold",
        cache_root=cache_root,
        symbol=symbol,
    )


def cache_imf_central_bank_proxy_from_csv(
    csv_path: str | Path,
    *,
    date_col: str,
    value_col: str,
    cache_root: str | Path = "data_cache",
    symbol: str = "GLD_CB_PROXY",
) -> tuple[Path, Path]:
    """
    Cache a monthly central-bank proxy from an IMF-style CSV export.

    The CSV only needs:
    - one date column
    - one numeric value column
    """
    frame = pd.read_csv(csv_path)
    return cache_monthly_proxy_series(
        frame,
        date_col=date_col,
        value_col=value_col,
        cache_root=cache_root,
        symbol=symbol,
    )


def cache_imf_central_bank_proxy_from_rows(
    rows: Iterable[dict],
    *,
    date_key: str,
    value_key: str,
    cache_root: str | Path = "data_cache",
    symbol: str = "GLD_CB_PROXY",
) -> tuple[Path, Path]:
    frame = pd.DataFrame(list(rows))
    return cache_monthly_proxy_series(
        frame,
        date_col=date_key,
        value_col=value_key,
        cache_root=cache_root,
        symbol=symbol,
    )


def _normalize_dbnomics_series_ids(
    series_ids: Iterable[str] | None = None,
    *,
    ref_areas: Iterable[str] | None = None,
) -> list[str]:
    ids = [str(s).strip() for s in (series_ids or []) if str(s).strip()]
    if ids:
        return ids
    areas = [str(a).strip().upper() for a in (ref_areas or DEFAULT_DBNOMICS_IMF_GOLD_RESERVE_AREAS)]
    return [f"IMF/IRFCL/M.{area}.RAFAGOLD_USD.S1X" for area in areas]


def fetch_dbnomics_series_rows(
    series_ids: Iterable[str],
    *,
    timeout: float = 30.0,
) -> list[dict]:
    """
    Fetch DBnomics series docs and return one row per observation.

    This is intentionally generic so we can reuse it for later IMF-derived
    official proxies without hardcoding them into the fair-value layer.
    """
    ids = [str(s).strip() for s in series_ids if str(s).strip()]
    if not ids:
        raise ValueError("No DBnomics series IDs provided")

    joined = ",".join(ids)
    url = (
        "https://api.db.nomics.world/v22/series?"
        f"series_ids={quote(joined, safe='/,')}&format=json&observations=1"
    )
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    docs = payload.get("series", {}).get("docs", [])
    if not docs:
        errors = payload.get("errors") or []
        raise ValueError(f"DBnomics returned no series docs. Errors: {errors}")

    rows: list[dict] = []
    for doc in docs:
        periods = doc.get("period_start_day") or doc.get("period") or []
        values = doc.get("value") or []
        dimensions = doc.get("dimensions") or {}
        for period, value in zip(periods, values):
            rows.append(
                {
                    "series_id": doc.get("series_code"),
                    "series_name": doc.get("series_name"),
                    "provider_code": doc.get("provider_code"),
                    "dataset_code": doc.get("dataset_code"),
                    "date": period,
                    "value": value,
                    "ref_area": dimensions.get("REF_AREA"),
                    "indicator": dimensions.get("INDICATOR"),
                    "frequency": doc.get("@frequency") or dimensions.get("FREQ"),
                }
            )
    if not rows:
        raise ValueError("DBnomics series docs contained no observations")
    return rows


def cache_dbnomics_central_bank_proxy(
    *,
    series_ids: Iterable[str] | None = None,
    ref_areas: Iterable[str] | None = None,
    cache_root: str | Path = "data_cache",
    symbol: str = "GLD_CB_PROXY",
    components_filename: str = "components_1M.csv",
    timeout: float = 30.0,
) -> tuple[Path, Path]:
    """
    Fetch a monthly central-bank gold reserve basket from DBnomics/IMF and
    cache an aggregate proxy for the fair-value model.

    The default basket uses a curated list of major reserve holders so we can
    avoid brittle website scraping while still keeping the input official.
    """
    ids = _normalize_dbnomics_series_ids(series_ids, ref_areas=ref_areas)
    rows = fetch_dbnomics_series_rows(ids, timeout=timeout)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No DBnomics central-bank rows fetched")

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["value"] = _safe_numeric(frame["value"])
    frame = frame.dropna(subset=["date", "value"]).sort_values(["date", "series_id"])
    frame = frame.loc[frame["frequency"].astype(str).str.lower().eq("monthly")].copy()
    if frame.empty:
        raise ValueError("DBnomics returned no monthly central-bank observations")

    aggregate = (
        frame.groupby("date", as_index=False)["value"]
        .sum(min_count=1)
        .rename(columns={"value": "proxy"})
        .sort_values("date")
    )
    daily_path, monthly_path = cache_monthly_proxy_series(
        aggregate,
        date_col="date",
        value_col="proxy",
        cache_root=cache_root,
        symbol=symbol,
    )

    components = (
        frame.assign(month=lambda d: d["date"].dt.to_period("M").dt.to_timestamp())
        .pivot_table(
            index="month",
            columns="ref_area",
            values="value",
            aggfunc="last",
        )
        .sort_index()
    )
    components_path = Path(cache_root) / "custom" / symbol / components_filename
    components_path.parent.mkdir(parents=True, exist_ok=True)
    components.reset_index().rename(columns={"month": "date"}).to_csv(
        components_path,
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
    )
    return daily_path, monthly_path
