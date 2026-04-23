"""
Slow fair-value diagnostics for macro-sensitive symbols.

The first supported use case is GLD, where we want a continuous
background fair-value proxy from slower macro and cross-asset inputs
before deciding whether it deserves to influence live trading logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.cache import DataCache


@dataclass(frozen=True)
class FairValueDiagnostics:
    symbol: str
    frame: pd.DataFrame
    stats: dict[str, Any]
    model: dict[str, Any]


_GLD_FAIR_VALUE_CACHE_VERSION = "gld_fair_value_v2"


def _rolling_zscore(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    min_obs = min_periods or max(3, min(window, window // 3 if window >= 3 else window))
    mu = s.rolling(window, min_periods=min_obs).mean()
    sigma = s.rolling(window, min_periods=min_obs).std(ddof=0).replace(0.0, np.nan)
    return ((s - mu) / sigma).clip(-6.0, 6.0)


def _load_close_series(
    cache: DataCache,
    source: str,
    symbol: str,
    timeframe: str,
) -> pd.Series | None:
    df = cache.load(source, symbol, timeframe)
    if df is None or df.empty or "close" not in df.columns:
        return None
    series = pd.Series(
        pd.to_numeric(df["close"], errors="coerce").values,
        index=pd.to_datetime(df["date"], errors="coerce"),
        name=symbol,
    ).dropna()
    if series.empty:
        return None
    return series[~series.index.duplicated(keep="last")].sort_index()


def _load_frame(
    cache: DataCache,
    source: str,
    symbol: str,
    timeframe: str,
) -> pd.DataFrame | None:
    df = cache.load(source, symbol, timeframe)
    if df is None or df.empty or "date" not in df.columns:
        return None
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])
    if out.empty:
        return None
    return out.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)


def _load_optional_proxy_series(
    cache: DataCache,
    candidates: list[tuple[str, str, str]],
) -> tuple[pd.Series | None, str | None]:
    for source, symbol, timeframe in candidates:
        series = _load_close_series(cache, source, symbol, timeframe)
        if series is not None and not series.empty:
            return series, f"{source}:{symbol}:{timeframe}"
    return None, None


def _derive_gld_etf_proxies(cache_root: Path, timeframe: str = "1M") -> dict[str, pd.Series]:
    cache = DataCache(root=cache_root)
    daily = _load_frame(cache, "derived", "GLD", "1D")
    if daily is None or daily.empty:
        return {}

    close = pd.to_numeric(daily["close"], errors="coerce")
    volume = pd.to_numeric(daily["volume"], errors="coerce")
    dates = pd.to_datetime(daily["date"], errors="coerce")
    df = pd.DataFrame(
        {
            "close": close.to_numpy(),
            "volume": volume.to_numpy(),
        },
        index=dates,
    ).dropna()
    if df.empty:
        return {}

    ret = df["close"].pct_change().fillna(0.0)
    direction = np.sign(ret)
    dollar_volume = (df["close"] * df["volume"]).clip(lower=0.0)

    # Market-derived ETF demand proxies. These are intentionally labeled as
    # proxies rather than official holdings/flow data.
    obv = (direction * df["volume"]).cumsum()
    signed_turnover = (np.tanh(ret / 0.01) * dollar_volume).cumsum()

    if timeframe.upper() == "1M":
        obv = obv.resample("MS").last()
        signed_turnover = signed_turnover.resample("MS").last()
    elif timeframe.upper() != "1D":
        raise ValueError(f"Unsupported ETF proxy timeframe: {timeframe}")

    proxies = {
        "GLD_ETF_OBV_PROXY": obv.rename("GLD_ETF_OBV_PROXY"),
        "GLD_ETF_TURNOVER_PROXY": signed_turnover.rename("GLD_ETF_TURNOVER_PROXY"),
    }
    return {name: series.dropna() for name, series in proxies.items() if not series.dropna().empty}


def _aligned_dataset(cache_root: Path, timeframe: str = "1M") -> pd.DataFrame:
    cache = DataCache(root=cache_root)
    required = {
        "GLD": ("derived", "GLD", timeframe),
        "UUP": ("derived", "UUP", timeframe),
        "IEF": ("derived", "IEF", timeframe),
        "SLV": ("derived", "SLV", timeframe),
        "GDX": ("derived", "GDX", timeframe),
        "VIXY": ("derived", "VIXY", timeframe),
        "CPIAUCSL": ("fred", "CPIAUCSL", timeframe),
        "DFII10": ("fred", "DFII10", timeframe),
        "WALCL": ("fred", "WALCL", timeframe),
        "M2SL": ("fred", "M2SL", timeframe),
        "VIXCLS": ("fred", "VIXCLS", timeframe),
    }

    series_map: dict[str, pd.Series] = {}
    for label, (source, symbol, timeframe) in required.items():
        series = _load_close_series(cache, source, symbol, timeframe)
        if series is None:
            raise FileNotFoundError(f"Missing cached slow dataset: {source}/{symbol}/{timeframe}")
        series_map[label] = series.rename(label)

    optional = {
        "ETF_OFFICIAL": [
            ("custom", "GLD_ETF_PROXY", timeframe),
            ("custom", "GLD_HOLDINGS_PROXY", timeframe),
            ("custom", "GLD_ETF_HOLDINGS", timeframe),
        ],
        "CB_OFFICIAL": [
            ("custom", "GLD_CB_PROXY", timeframe),
            ("custom", "GOLD_CB_PROXY", timeframe),
            ("custom", "GOLD_CENTRAL_BANK_PROXY", timeframe),
        ],
    }
    optional_sources: dict[str, str] = {}
    for label, candidates in optional.items():
        series, source_tag = _load_optional_proxy_series(cache, candidates)
        if series is not None:
            series_map[label] = series.rename(label)
            optional_sources[label] = source_tag or label

    for label, series in _derive_gld_etf_proxies(cache_root, timeframe=timeframe).items():
        series_map[label] = series.rename(label)
        optional_sources[label] = "derived:GLD:1D"

    base_index = series_map["GLD"].index
    df = pd.DataFrame(index=base_index)
    for label, series in series_map.items():
        df[label] = series.reindex(base_index).ffill()
    out = df.dropna(subset=["GLD"]).copy()
    out.attrs["optional_sources"] = optional_sources
    return out


def _build_feature_frame(raw: pd.DataFrame, z_window: int, *, is_monthly: bool) -> pd.DataFrame:
    df = raw.copy()
    gld = pd.to_numeric(df["GLD"], errors="coerce")
    uup = pd.to_numeric(df["UUP"], errors="coerce")
    ief = pd.to_numeric(df["IEF"], errors="coerce")
    slv = pd.to_numeric(df["SLV"], errors="coerce")
    gdx = pd.to_numeric(df["GDX"], errors="coerce")
    vixy = pd.to_numeric(df["VIXY"], errors="coerce")
    cpi = pd.to_numeric(df["CPIAUCSL"], errors="coerce")
    real_yield = pd.to_numeric(df["DFII10"], errors="coerce")
    walcl = pd.to_numeric(df["WALCL"], errors="coerce")
    m2 = pd.to_numeric(df["M2SL"], errors="coerce")
    vix = pd.to_numeric(df["VIXCLS"], errors="coerce")

    features = pd.DataFrame(index=df.index)
    features["target"] = np.log(gld.replace(0.0, np.nan))
    features["actual"] = gld

    yoy_lag = 12 if is_monthly else 252
    liquidity_lag = 3 if is_monthly else 63
    money_lag = 6 if is_monthly else 126
    momentum_lag = 3 if is_monthly else 63

    inflation_yoy = cpi.pct_change(yoy_lag)
    liquidity_flow = np.log(walcl.replace(0.0, np.nan)).diff(liquidity_lag)
    money_flow = np.log(m2.replace(0.0, np.nan)).diff(money_lag)
    gold_mom = gld.pct_change(momentum_lag)

    features["real_yield"] = _rolling_zscore(-real_yield, z_window)
    features["usd"] = _rolling_zscore(-np.log(uup.replace(0.0, np.nan)), z_window)
    features["inflation"] = _rolling_zscore(inflation_yoy, z_window)
    features["liquidity"] = _rolling_zscore(liquidity_flow, z_window)
    features["money"] = _rolling_zscore(money_flow, z_window)
    features["stress"] = _rolling_zscore(vix, z_window)
    features["peer"] = _rolling_zscore(np.log((slv / gld).replace(0.0, np.nan)), z_window)
    features["miners"] = _rolling_zscore(np.log((gdx / gld).replace(0.0, np.nan)), z_window)
    features["riskoff"] = _rolling_zscore(np.log(vixy.replace(0.0, np.nan)), z_window)
    features["rates_proxy"] = _rolling_zscore(np.log(ief.replace(0.0, np.nan)), z_window)
    features["momentum"] = _rolling_zscore(gold_mom, z_window)
    if "GLD_ETF_OBV_PROXY" in df.columns:
        features["etf_proxy"] = _rolling_zscore(pd.to_numeric(df["GLD_ETF_OBV_PROXY"], errors="coerce"), z_window)
    if "GLD_ETF_TURNOVER_PROXY" in df.columns:
        features["etf_turnover_proxy"] = _rolling_zscore(
            pd.to_numeric(df["GLD_ETF_TURNOVER_PROXY"], errors="coerce"),
            z_window,
        )
    if "ETF_OFFICIAL" in df.columns:
        features["etf_official"] = _rolling_zscore(pd.to_numeric(df["ETF_OFFICIAL"], errors="coerce"), z_window)
    if "CB_OFFICIAL" in df.columns:
        features["cb_official"] = _rolling_zscore(pd.to_numeric(df["CB_OFFICIAL"], errors="coerce"), z_window)
    trend_lag = 3 if is_monthly else 63
    accel_lag = 1 if is_monthly else 21
    for col in [
        "real_yield",
        "usd",
        "inflation",
        "liquidity",
        "money",
        "stress",
        "peer",
        "miners",
        "riskoff",
        "rates_proxy",
        "momentum",
        "etf_proxy",
        "etf_turnover_proxy",
        "etf_official",
        "cb_official",
    ]:
        if col not in features.columns:
            continue
        series = features[col]
        features[f"{col}_trend"] = series.diff(trend_lag)
        features[f"{col}_accel"] = series.diff(accel_lag)
    for col in ["real_yield", "usd", "inflation", "liquidity", "money", "stress"]:
        series = features[col]
        features[f"{col}_lag1"] = series.shift(1)
        features[f"{col}_lag2"] = series.shift(2)
    return features


def _ridge_fit(y: np.ndarray, x: np.ndarray, alpha: float, weights: np.ndarray | None = None) -> np.ndarray:
    n_rows, n_cols = x.shape
    x_design = np.column_stack([np.ones(n_rows), x])
    if weights is None:
        weights = np.ones(n_rows)
    sqrt_w = np.sqrt(weights)
    xw = x_design * sqrt_w[:, None]
    yw = y * sqrt_w
    penalty = np.diag([0.0] + [alpha] * n_cols)
    try:
        beta = np.linalg.solve(xw.T @ xw + penalty, xw.T @ yw)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(xw.T @ xw + penalty, xw.T @ yw, rcond=None)[0]
    return beta


def _rolling_fair_value(
    features: pd.DataFrame,
    feature_cols: list[str],
    fit_window: int,
    min_fit_obs: int,
    ridge_alpha: float,
    smooth_span: int,
) -> tuple[pd.Series, dict[str, float]]:
    fair = pd.Series(index=features.index, dtype=float)
    last_betas = {name: np.nan for name in feature_cols}

    for pos in range(len(features)):
        hist_start = max(0, pos - fit_window)
        window = features.iloc[hist_start:pos][["target"] + feature_cols].dropna()
        if len(window) < min_fit_obs:
            continue

        row_now = features.iloc[pos]
        if row_now[feature_cols].isna().any():
            continue

        y = window["target"].to_numpy(dtype=float)
        x = window[feature_cols].to_numpy(dtype=float)
        weights = np.exp(np.linspace(-2.0, 0.0, len(window)))
        beta = _ridge_fit(y, x, alpha=ridge_alpha, weights=weights)
        fair.iloc[pos] = float(np.exp(beta[0] + np.dot(row_now[feature_cols].to_numpy(dtype=float), beta[1:])))
        last_betas = {name: float(val) for name, val in zip(feature_cols, beta[1:])}

    if smooth_span > 1:
        fair = fair.ewm(span=smooth_span, adjust=False, min_periods=max(2, smooth_span // 2)).mean()
    return fair, last_betas


def _two_layer_fair_value(
    features: pd.DataFrame,
    *,
    structural_cols: list[str],
    market_cols: list[str],
    structural_fit_window: int,
    market_fit_window: int,
    min_struct_obs: int,
    min_market_obs: int,
    ridge_alpha: float,
    smooth_span: int,
) -> tuple[pd.Series, pd.Series, pd.Series, dict[str, float]]:
    fair = pd.Series(index=features.index, dtype=float)
    structural_only = pd.Series(index=features.index, dtype=float)
    market_adjustment = pd.Series(index=features.index, dtype=float)
    last_betas = {name: np.nan for name in [*structural_cols, *market_cols]}

    for pos in range(len(features)):
        row_now = features.iloc[pos]
        if row_now[structural_cols].isna().any():
            continue

        struct_start = max(0, pos - structural_fit_window)
        struct_window = features.iloc[struct_start:pos][["target"] + structural_cols].dropna()
        if len(struct_window) < min_struct_obs:
            continue

        y_struct = struct_window["target"].to_numpy(dtype=float)
        x_struct = struct_window[structural_cols].to_numpy(dtype=float)
        struct_weights = np.exp(np.linspace(-2.4, 0.0, len(struct_window)))
        beta_struct = _ridge_fit(y_struct, x_struct, alpha=ridge_alpha, weights=struct_weights)

        structural_log_now = float(
            beta_struct[0] + np.dot(row_now[structural_cols].to_numpy(dtype=float), beta_struct[1:])
        )
        structural_only.iloc[pos] = float(np.exp(structural_log_now))

        if not market_cols or row_now[market_cols].isna().any():
            fair.iloc[pos] = structural_only.iloc[pos]
            last_betas = {name: float(val) for name, val in zip(structural_cols, beta_struct[1:])}
            continue

        market_start = max(0, pos - market_fit_window)
        market_cols_all = list(dict.fromkeys(["target"] + structural_cols + market_cols))
        market_hist = features.iloc[market_start:pos][market_cols_all].dropna()
        if len(market_hist) < min_market_obs:
            fair.iloc[pos] = structural_only.iloc[pos]
            last_betas = {name: float(val) for name, val in zip(structural_cols, beta_struct[1:])}
            continue

        struct_hist_log = beta_struct[0] + market_hist[structural_cols].to_numpy(dtype=float) @ beta_struct[1:]
        residual_target = market_hist["target"].to_numpy(dtype=float) - struct_hist_log
        x_market = market_hist[market_cols].to_numpy(dtype=float)
        market_weights = np.exp(np.linspace(-1.7, 0.0, len(market_hist)))
        beta_market = _ridge_fit(residual_target, x_market, alpha=ridge_alpha, weights=market_weights)
        market_log_now = float(
            beta_market[0] + np.dot(row_now[market_cols].to_numpy(dtype=float), beta_market[1:])
        )
        market_adjustment.iloc[pos] = market_log_now
        fair.iloc[pos] = float(np.exp(structural_log_now + market_log_now))

        last_betas = {name: float(val) for name, val in zip(structural_cols, beta_struct[1:])}
        last_betas.update({name: float(val) for name, val in zip(market_cols, beta_market[1:])})

    if smooth_span > 1:
        fair = fair.ewm(span=smooth_span, adjust=False, min_periods=max(2, smooth_span // 2)).mean()
        structural_only = structural_only.ewm(
            span=smooth_span, adjust=False, min_periods=max(2, smooth_span // 2)
        ).mean()
        market_adjustment = market_adjustment.ewm(
            span=smooth_span, adjust=False, min_periods=max(2, smooth_span // 2)
        ).mean()
    return fair, structural_only, market_adjustment, last_betas


def _fit_stats(frame: pd.DataFrame) -> dict[str, float]:
    aligned = frame.dropna(subset=["actual", "fair_value"]).copy()
    if len(aligned) < 10:
        return {
            "corr": np.nan,
            "r2": np.nan,
            "mae_pct": np.nan,
            "rmse_pct": np.nan,
            "directional_hit": np.nan,
            "fit_score": -np.inf,
        }

    actual = aligned["actual"].astype(float)
    fair = aligned["fair_value"].astype(float)
    gap_pct = (fair / actual.replace(0.0, np.nan) - 1.0) * 100.0
    residual = fair - actual
    denom = float(((actual - actual.mean()) ** 2).sum())
    r2 = float(1.0 - (residual**2).sum() / denom) if denom > 0 else np.nan
    corr = float(actual.corr(fair))
    mae_pct = float(gap_pct.abs().mean())
    rmse_pct = float(np.sqrt(np.nanmean(gap_pct**2)))
    directional_hit = float(
        (np.sign(fair.diff().fillna(0.0)) == np.sign(actual.diff().fillna(0.0))).mean()
    )
    fit_score = (
        (0.38 * (0.0 if np.isnan(corr) else corr))
        + (0.27 * max(r2, -1.0))
        + (0.25 * directional_hit)
        - (0.10 * min(mae_pct / 25.0, 2.0))
    )
    return {
        "corr": corr,
        "r2": r2,
        "mae_pct": mae_pct,
        "rmse_pct": rmse_pct,
        "directional_hit": directional_hit,
        "fit_score": fit_score,
    }


def fair_value_cache_fingerprint(cache_root: str | Path = "data_cache") -> str:
    root = Path(cache_root)
    watched = [
        root / "custom" / "GLD_ETF_PROXY" / "1M.csv",
        root / "custom" / "GLD_CB_PROXY" / "1M.csv",
        root / "custom" / "GLD_HOLDINGS_PROXY" / "1M.csv",
        root / "custom" / "GOLD_CB_PROXY" / "1M.csv",
        root / "custom" / "GOLD_CENTRAL_BANK_PROXY" / "1M.csv",
    ]
    bits: list[str] = []
    for path in watched:
        if path.exists():
            stat = path.stat()
            bits.append(f"{path.name}:{stat.st_mtime_ns}:{stat.st_size}")
        else:
            bits.append(f"{path.name}:missing")
    return "|".join(bits)


def _fair_value_store_dir(cache_root: str | Path) -> Path:
    return Path(cache_root) / "custom" / "GLD_FAIR_VALUE_CACHE"


def _fair_value_store_paths(cache_root: str | Path, cache_fingerprint: str) -> tuple[Path, Path]:
    digest = hashlib.sha1(cache_fingerprint.encode("utf-8")).hexdigest()[:16]
    base_name = f"{_GLD_FAIR_VALUE_CACHE_VERSION}_{digest}"
    store_dir = _fair_value_store_dir(cache_root)
    return store_dir / f"{base_name}.pkl", store_dir / f"{base_name}.meta.json"


def _load_cached_gld_fair_value_base(
    cache_root: str | Path,
    cache_fingerprint: str,
) -> FairValueDiagnostics | None:
    payload_path, meta_path = _fair_value_store_paths(cache_root, cache_fingerprint)
    if not payload_path.exists() or not meta_path.exists():
        return None
    try:
        meta = pd.read_json(meta_path, typ="series")
        if str(meta.get("version", "")) != _GLD_FAIR_VALUE_CACHE_VERSION:
            return None
        if str(meta.get("fingerprint", "")) != cache_fingerprint:
            return None
        payload = pd.read_pickle(payload_path)
        frame = payload.get("frame")
        stats = payload.get("stats")
        model = payload.get("model")
        if not isinstance(frame, pd.DataFrame) or not isinstance(stats, dict) or not isinstance(model, dict):
            return None
        return FairValueDiagnostics(
            symbol="GLD",
            frame=frame,
            stats=stats,
            model=model,
        )
    except Exception:
        return None


def _store_cached_gld_fair_value_base(
    diagnostics: FairValueDiagnostics,
    cache_root: str | Path,
    cache_fingerprint: str,
) -> None:
    payload_path, meta_path = _fair_value_store_paths(cache_root, cache_fingerprint)
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(
        {
            "frame": diagnostics.frame,
            "stats": diagnostics.stats,
            "model": diagnostics.model,
        },
        payload_path,
    )
    pd.Series(
        {
            "version": _GLD_FAIR_VALUE_CACHE_VERSION,
            "fingerprint": cache_fingerprint,
        }
    ).to_json(meta_path)


def compute_gld_fair_value_diagnostics(
    *,
    cache_root: str | Path = "data_cache",
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> FairValueDiagnostics | None:
    best = _compute_gld_fair_value_base(str(Path(cache_root)), fair_value_cache_fingerprint(cache_root))
    if best is None:
        return None

    frame = best.frame.copy()
    if start is not None:
        frame = frame[frame["date"] >= pd.Timestamp(start)]
    if end is not None:
        frame = frame[frame["date"] <= pd.Timestamp(end)]
    if frame.empty:
        return None

    frame = frame.reset_index(drop=True)
    return FairValueDiagnostics(
        symbol=best.symbol,
        frame=frame,
        stats=_fit_stats(frame),
        model=best.model,
    )


@lru_cache(maxsize=4)
def _compute_gld_fair_value_base(cache_root: str, cache_fingerprint: str) -> FairValueDiagnostics | None:
    cached = _load_cached_gld_fair_value_base(cache_root, cache_fingerprint)
    if cached is not None:
        return cached

    raw = _aligned_dataset(Path(cache_root), timeframe="1M")
    is_monthly = True
    optional_sources = dict(raw.attrs.get("optional_sources", {}))
    one_layer_sets = {
        "macro_core": ["real_yield", "usd", "inflation", "liquidity", "stress"],
        "macro_core_lag1": ["real_yield_lag1", "usd_lag1", "inflation_lag1", "liquidity_lag1", "stress_lag1"],
        "macro_core_lag2": ["real_yield_lag2", "usd_lag2", "inflation_lag2", "liquidity_lag2", "stress_lag2"],
        "macro_plus_market": ["real_yield", "usd", "inflation", "liquidity", "stress", "peer", "miners"],
        "macro_plus_market_lag1": [
            "real_yield_lag1",
            "usd_lag1",
            "inflation_lag1",
            "liquidity_lag1",
            "stress_lag1",
            "peer",
            "miners",
        ],
        "macro_plus_market_lag2": [
            "real_yield_lag2",
            "usd_lag2",
            "inflation_lag2",
            "liquidity_lag2",
            "stress_lag2",
            "peer",
            "miners",
        ],
        "macro_hybrid_lag1": [
            "real_yield_lag1",
            "usd_lag1",
            "inflation_lag1",
            "liquidity_lag1",
            "stress",
            "peer",
            "miners",
        ],
        "macro_plus_market_etf": [
            "real_yield",
            "usd",
            "inflation",
            "liquidity",
            "stress",
            "peer",
            "miners",
            "etf_proxy",
        ],
        "macro_plus_market_turnover": [
            "real_yield",
            "usd",
            "inflation",
            "liquidity",
            "stress",
            "peer",
            "miners",
            "etf_turnover_proxy",
        ],
        "macro_plus_market_official": [
            "real_yield",
            "usd",
            "inflation",
            "liquidity",
            "stress",
            "peer",
            "miners",
            "etf_official",
            "cb_official",
        ],
        "full": [
            "real_yield",
            "usd",
            "inflation",
            "liquidity",
            "stress",
            "peer",
            "miners",
            "etf_proxy",
            "etf_turnover_proxy",
            "etf_official",
            "cb_official",
            "money",
            "riskoff",
            "rates_proxy",
            "momentum",
        ],
        "full_with_trends": [
            "real_yield",
            "real_yield_trend",
            "usd",
            "usd_trend",
            "inflation",
            "liquidity",
            "liquidity_trend",
            "stress",
            "peer",
            "peer_trend",
            "miners",
            "miners_trend",
            "etf_proxy",
            "etf_turnover_proxy",
            "etf_official",
            "cb_official",
            "momentum",
            "momentum_trend",
        ],
    }
    structural_sets = {
        "structural_core": ["real_yield", "usd", "inflation", "liquidity", "stress"],
        "structural_core_lag1": ["real_yield_lag1", "usd_lag1", "inflation_lag1", "liquidity_lag1", "stress_lag1"],
        "structural_core_lag2": ["real_yield_lag2", "usd_lag2", "inflation_lag2", "liquidity_lag2", "stress_lag2"],
        "structural_plus_money": [
            "real_yield",
            "real_yield_trend",
            "usd",
            "usd_trend",
            "inflation",
            "liquidity",
            "liquidity_trend",
            "money",
            "stress",
        ],
        "structural_plus_money_official": [
            "real_yield",
            "real_yield_trend",
            "usd",
            "usd_trend",
            "inflation",
            "liquidity",
            "liquidity_trend",
            "money",
            "stress",
            "etf_official",
            "cb_official",
        ],
        "structural_plus_money_lag1": [
            "real_yield_lag1",
            "real_yield_trend",
            "usd_lag1",
            "usd_trend",
            "inflation_lag1",
            "liquidity_lag1",
            "liquidity_trend",
            "money_lag1",
            "stress_lag1",
        ],
    }
    market_sets = {
        "market_peer": ["peer", "peer_trend", "miners", "miners_trend", "momentum"],
        "market_full": [
            "peer",
            "peer_trend",
            "miners",
            "miners_trend",
            "etf_proxy",
            "etf_turnover_proxy",
            "etf_official",
            "cb_official",
            "riskoff",
            "rates_proxy",
            "momentum",
            "momentum_trend",
        ],
    }
    z_windows = (12, 18, 24)
    structural_fit_windows = (12, 18, 24, 30)
    market_fit_windows = (12, 18)
    ridge_alphas = (0.0, 0.5, 1.5)
    smooth_spans = (2, 3, 4)

    best: FairValueDiagnostics | None = None
    best_score = -np.inf
    for z_window in z_windows:
        features = _build_feature_frame(raw, z_window=z_window, is_monthly=is_monthly)
        for set_name, cols in one_layer_sets.items():
            available_cols = [col for col in cols if col in features.columns]
            for fit_window in structural_fit_windows:
                min_fit_obs = max(8, fit_window // 2)
                for ridge_alpha in ridge_alphas:
                    for smooth_span in smooth_spans:
                        fair_value, betas = _rolling_fair_value(
                            features,
                            feature_cols=available_cols,
                            fit_window=fit_window,
                            min_fit_obs=min_fit_obs,
                            ridge_alpha=ridge_alpha,
                            smooth_span=smooth_span,
                        )
                        frame = pd.DataFrame(
                            {
                                "date": features.index,
                                "actual": features["actual"],
                                "fair_value": fair_value,
                            }
                        ).dropna(subset=["actual", "fair_value"])
                        if frame.empty:
                            continue
                        frame["fair_gap_pct"] = (
                            frame["fair_value"] / frame["actual"].replace(0.0, np.nan) - 1.0
                        ) * 100.0
                        stats = _fit_stats(frame)
                        if stats["fit_score"] <= best_score:
                            continue
                        best_score = stats["fit_score"]
                        best = FairValueDiagnostics(
                            symbol="GLD",
                            frame=frame.reset_index(drop=True),
                            stats=stats,
                            model={
                                "model_type": "one_layer",
                                "feature_set": set_name,
                                "features": tuple(available_cols),
                                "z_window": int(z_window),
                                "fit_window": int(fit_window),
                                "ridge_alpha": float(ridge_alpha),
                                "smooth_span": int(smooth_span),
                                "betas": betas,
                                "optional_sources": optional_sources,
                            },
                        )
        for struct_name, structural_cols in structural_sets.items():
            available_structural = [col for col in structural_cols if col in features.columns]
            for market_name, market_cols in market_sets.items():
                available_market = [col for col in market_cols if col in features.columns]
                for structural_fit_window in structural_fit_windows:
                    min_struct_obs = max(10, structural_fit_window // 2)
                    for market_fit_window in market_fit_windows:
                        min_market_obs = max(8, market_fit_window // 2)
                        for ridge_alpha in ridge_alphas:
                            for smooth_span in smooth_spans:
                                fair_value, structural_only, market_adjustment, betas = _two_layer_fair_value(
                                    features,
                                    structural_cols=available_structural,
                                    market_cols=available_market,
                                    structural_fit_window=structural_fit_window,
                                    market_fit_window=market_fit_window,
                                    min_struct_obs=min_struct_obs,
                                    min_market_obs=min_market_obs,
                                    ridge_alpha=ridge_alpha,
                                    smooth_span=smooth_span,
                                )
                                frame = pd.DataFrame(
                                    {
                                        "date": features.index,
                                        "actual": features["actual"],
                                        "fair_value": fair_value,
                                        "structural_fair_value": structural_only,
                                        "market_adjustment_log": market_adjustment,
                                    }
                                ).dropna(subset=["actual", "fair_value"])
                                if frame.empty:
                                    continue
                                frame["fair_gap_pct"] = (
                                    frame["fair_value"] / frame["actual"].replace(0.0, np.nan) - 1.0
                                ) * 100.0
                                stats = _fit_stats(frame)
                                if stats["fit_score"] <= best_score:
                                    continue
                                best_score = stats["fit_score"]
                                best = FairValueDiagnostics(
                                    symbol="GLD",
                                    frame=frame.reset_index(drop=True),
                                    stats=stats,
                                    model={
                                        "model_type": "two_layer",
                                        "structural_set": struct_name,
                                        "market_set": market_name,
                                        "structural_features": tuple(available_structural),
                                        "market_features": tuple(available_market),
                                        "z_window": int(z_window),
                                        "structural_fit_window": int(structural_fit_window),
                                        "market_fit_window": int(market_fit_window),
                                        "ridge_alpha": float(ridge_alpha),
                                        "smooth_span": int(smooth_span),
                                        "betas": betas,
                                        "optional_sources": optional_sources,
                                    },
                                )

    if best is not None:
        _store_cached_gld_fair_value_base(best, cache_root, cache_fingerprint)
    return best


def prepare_gld_fair_value_context(
    data: pd.DataFrame,
    *,
    cache_root: str | Path = "data_cache",
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    bullish_slope_min: float = 1.0,
    bearish_slope_max: float = -1.0,
    undervalued_gap_pct: float = 2.5,
    overvalued_gap_pct: float = 2.5,
) -> pd.DataFrame:
    diagnostics = compute_gld_fair_value_diagnostics(cache_root=cache_root, start=start, end=end)
    enriched = data.copy()
    if diagnostics is None or diagnostics.frame.empty or "date" not in enriched.columns:
        return enriched

    fair = diagnostics.frame.copy()
    fair["date"] = pd.to_datetime(fair["date"], errors="coerce")
    fair = fair.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    fair["fair_slope_pct"] = fair["fair_value"].pct_change().fillna(0.0) * 100.0
    fair["fair_confidence"] = float(diagnostics.stats.get("directional_hit", np.nan))

    fair["gold_fair_value_regime"] = "neutral"
    bullish = (fair["fair_slope_pct"] >= bullish_slope_min) & (fair["fair_gap_pct"] >= -undervalued_gap_pct)
    bearish = (fair["fair_slope_pct"] <= bearish_slope_max) & (fair["fair_gap_pct"] <= -overvalued_gap_pct)
    fair.loc[bullish, "gold_fair_value_regime"] = "bullish"
    fair.loc[bearish, "gold_fair_value_regime"] = "bearish"

    keep_cols = [
        "date",
        "fair_value",
        "fair_gap_pct",
        "fair_slope_pct",
        "fair_confidence",
        "gold_fair_value_regime",
    ]
    primary = enriched.copy()
    primary["date"] = pd.to_datetime(primary["date"], errors="coerce")
    merged = pd.merge_asof(
        primary.sort_values("date"),
        fair[keep_cols].sort_values("date"),
        on="date",
        direction="backward",
        tolerance=pd.Timedelta(days=45),
    )
    return merged.reset_index(drop=True)
