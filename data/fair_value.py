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


_GLD_FAIR_VALUE_CACHE_VERSION = "gld_fair_value_v5"


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


def _parse_numeric_text(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )


def _load_custom_gld_daily_target(cache_root: Path) -> tuple[pd.Series | None, str | None]:
    path = cache_root / "custom" / "GLD ETF Stock Price History.csv"
    if not path.exists():
        return None, None
    try:
        raw = pd.read_csv(path)
    except Exception:
        return None, None
    if "Date" not in raw.columns or "Price" not in raw.columns:
        return None, None
    dates = pd.to_datetime(raw["Date"], errors="coerce", format="%m/%d/%Y")
    close = _parse_numeric_text(raw["Price"])
    series = pd.Series(close.to_numpy(), index=dates, name="GLD").dropna()
    if series.empty:
        return None, None
    series = series[~series.index.duplicated(keep="last")].sort_index()
    return series, f"custom:{path.name}"


def _load_gld_target_series(cache_root: Path, timeframe: str) -> tuple[pd.Series | None, str | None]:
    cache = DataCache(root=cache_root)
    tf_u = timeframe.upper()
    if tf_u == "1D":
        custom, source_tag = _load_custom_gld_daily_target(cache_root)
        if custom is not None and not custom.empty:
            return custom, source_tag
    derived = _load_close_series(cache, "derived", "GLD", timeframe)
    if derived is not None and not derived.empty:
        return derived.rename("GLD"), f"derived:GLD:{timeframe}"
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
    tf_u = timeframe.upper()
    target_series, target_source = _load_gld_target_series(cache_root, tf_u)
    if target_series is None:
        raise FileNotFoundError(f"Missing cached GLD fair-value target series for timeframe {tf_u}")

    companion_specs = {
        "UUP": ("derived", "UUP", tf_u),
        "IEF": ("derived", "IEF", tf_u),
        "SLV": ("derived", "SLV", tf_u),
        "GDX": ("derived", "GDX", tf_u),
        "VIXY": ("derived", "VIXY", tf_u),
        "CPIAUCSL": ("fred", "CPIAUCSL", tf_u),
        "DFII10": ("fred", "DFII10", tf_u),
        "WALCL": ("fred", "WALCL", tf_u),
        "M2SL": ("fred", "M2SL", tf_u),
        "VIXCLS": ("fred", "VIXCLS", tf_u),
    }
    must_have = {"CPIAUCSL", "DFII10", "WALCL", "M2SL", "VIXCLS"}

    series_map: dict[str, pd.Series] = {"GLD": target_series.rename("GLD")}
    for label, (source, symbol, tf_req) in companion_specs.items():
        series = _load_close_series(cache, source, symbol, tf_req)
        if series is None:
            if label in must_have:
                raise FileNotFoundError(f"Missing cached slow dataset: {source}/{symbol}/{tf_req}")
            continue
        series_map[label] = series.rename(label)

    optional = {
        "ETF_OFFICIAL": [
            ("custom", "GLD_ETF_PROXY", tf_u),
            ("custom", "GLD_HOLDINGS_PROXY", tf_u),
            ("custom", "GLD_ETF_HOLDINGS", tf_u),
        ],
        "CB_OFFICIAL": [
            ("custom", "GLD_CB_PROXY", tf_u),
            ("custom", "GOLD_CB_PROXY", tf_u),
            ("custom", "GOLD_CENTRAL_BANK_PROXY", tf_u),
        ],
    }
    optional_sources: dict[str, str] = {}
    for label, candidates in optional.items():
        series, source_tag = _load_optional_proxy_series(cache, candidates)
        if series is not None:
            series_map[label] = series.rename(label)
            optional_sources[label] = source_tag or label

    for label, series in _derive_gld_etf_proxies(cache_root, timeframe=tf_u).items():
        series_map[label] = series.rename(label)
        optional_sources[label] = "derived:GLD:1D"

    base_index = series_map["GLD"].index
    df = pd.DataFrame(index=base_index)
    for label in [
        "GLD",
        *companion_specs.keys(),
        "ETF_OFFICIAL",
        "CB_OFFICIAL",
        "GLD_ETF_OBV_PROXY",
        "GLD_ETF_TURNOVER_PROXY",
    ]:
        series = series_map.get(label)
        if series is None:
            df[label] = np.nan
        else:
            df[label] = series.reindex(base_index).ffill()
    out = df.dropna(subset=["GLD"]).copy()
    out.attrs["optional_sources"] = optional_sources
    out.attrs["target_source"] = target_source or "unknown"
    out.attrs["data_timeframe"] = tf_u
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


def _calc_rsi(series: pd.Series, period: int = 9) -> pd.Series:
    delta = series.astype(float).diff()
    gains = delta.clip(lower=0.0)
    losses = (-delta).clip(lower=0.0)
    avg_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100.0 - (100.0 / (1.0 + rs))).clip(0.0, 100.0)


def _predictive_score_components(frame: pd.DataFrame, *, is_monthly: bool) -> dict[str, float]:
    aligned = frame.dropna(subset=["actual", "fair_value"]).copy()
    if aligned.empty:
        return {
            "gap_reversion_score": 0.0,
            "gap_reversion_focus_horizon": float("nan"),
        }

    if "fair_gap_pct" not in aligned.columns:
        actual = aligned["actual"].astype(float)
        fair = aligned["fair_value"].astype(float)
        aligned["fair_gap_pct"] = (fair / actual.replace(0.0, np.nan) - 1.0) * 100.0

    actual = aligned["actual"].astype(float)
    gap = aligned["fair_gap_pct"].astype(float)
    horizons = (1, 2, 3) if is_monthly else (3, 5, 10)
    focus_horizon = horizons[1] if len(horizons) > 1 else horizons[0]
    spread_divisor = 10.0 if is_monthly else 5.0

    out: dict[str, float] = {
        "gap_reversion_score": 0.0,
        "gap_reversion_focus_horizon": float(focus_horizon),
    }
    scores: list[float] = []

    for horizon in horizons:
        suffix = f"{horizon}{'m' if is_monthly else 'd'}"
        future_ret = (actual.shift(-horizon) / actual.replace(0.0, np.nan) - 1.0) * 100.0
        valid = pd.DataFrame({"gap": gap, "future_ret": future_ret}).dropna()
        if len(valid) < max(12, horizon * 4):
            out[f"gap_fwd_{suffix}_corr"] = np.nan
            out[f"gap_fwd_{suffix}_directional_hit"] = np.nan
            out[f"gap_fwd_{suffix}_quintile_spread"] = np.nan
            continue

        corr = float(valid["gap"].corr(valid["future_ret"]))
        directional_hit = float(
            (np.sign(valid["gap"].fillna(0.0)) == np.sign(valid["future_ret"].fillna(0.0))).mean()
        )
        lower_gap = float(valid["gap"].quantile(0.2))
        upper_gap = float(valid["gap"].quantile(0.8))
        overvalued_ret = valid.loc[valid["gap"] <= lower_gap, "future_ret"]
        undervalued_ret = valid.loc[valid["gap"] >= upper_gap, "future_ret"]
        if overvalued_ret.empty or undervalued_ret.empty:
            quintile_spread = np.nan
        else:
            quintile_spread = float(undervalued_ret.mean() - overvalued_ret.mean())

        centered_hit = (directional_hit - 0.5) * 2.0
        spread_score = 0.0 if np.isnan(quintile_spread) else float(np.clip(quintile_spread / spread_divisor, -2.0, 2.0))
        predictive_score = (
            0.50 * (0.0 if np.isnan(corr) else corr)
            + 0.35 * centered_hit
            + 0.15 * spread_score
        )
        scores.append(predictive_score)
        out[f"gap_fwd_{suffix}_corr"] = corr
        out[f"gap_fwd_{suffix}_directional_hit"] = directional_hit
        out[f"gap_fwd_{suffix}_quintile_spread"] = quintile_spread

    if scores:
        out["gap_reversion_score"] = float(np.nanmean(scores))
    return out


def _selection_score(
    frame: pd.DataFrame,
    stats: dict[str, float],
    *,
    raw_index: pd.Index,
    is_monthly: bool,
) -> dict[str, float]:
    if frame.empty or raw_index.empty:
        return {
            "coverage_ratio": 0.0,
            "history_span_ratio": 0.0,
            "gap_reversion_score": 0.0,
            "gap_reversion_focus_horizon": float("nan"),
            "selection_score": -np.inf,
        }

    coverage_ratio = float(len(frame) / max(len(raw_index), 1))
    raw_start = pd.to_datetime(raw_index.min(), errors="coerce")
    raw_end = pd.to_datetime(raw_index.max(), errors="coerce")
    fit_start = pd.to_datetime(frame["date"].min(), errors="coerce")
    fit_end = pd.to_datetime(frame["date"].max(), errors="coerce")
    raw_span_days = max(float((raw_end - raw_start).days), 1.0)
    fit_span_days = max(float((fit_end - fit_start).days), 0.0)
    history_span_ratio = min(1.0, fit_span_days / raw_span_days)

    predictive = _predictive_score_components(frame, is_monthly=is_monthly)

    if is_monthly:
        coverage_weight = 0.03
        span_weight = 0.02
        predictive_weight = 0.08
    else:
        coverage_weight = 0.08
        span_weight = 0.04
        predictive_weight = 0.18

    selection_score = float(
        stats["fit_score"]
        + coverage_weight * coverage_ratio
        + span_weight * history_span_ratio
        + predictive_weight * predictive.get("gap_reversion_score", 0.0)
    )
    return {
        "coverage_ratio": coverage_ratio,
        "history_span_ratio": history_span_ratio,
        **predictive,
        "selection_score": selection_score,
    }


def fair_value_cache_fingerprint(cache_root: str | Path = "data_cache") -> str:
    root = Path(cache_root)
    watched: list[Path] = [
        root / "custom" / "GLD ETF Stock Price History.csv",
        root / "custom" / "GLD_ETF_PROXY" / "1D.csv",
        root / "custom" / "GLD_ETF_PROXY" / "1M.csv",
        root / "custom" / "GLD_CB_PROXY" / "1D.csv",
        root / "custom" / "GLD_CB_PROXY" / "1M.csv",
        root / "custom" / "GLD_HOLDINGS_PROXY" / "1D.csv",
        root / "custom" / "GLD_HOLDINGS_PROXY" / "1M.csv",
        root / "custom" / "GOLD_CB_PROXY" / "1D.csv",
        root / "custom" / "GOLD_CB_PROXY" / "1M.csv",
        root / "custom" / "GOLD_CENTRAL_BANK_PROXY" / "1D.csv",
        root / "custom" / "GOLD_CENTRAL_BANK_PROXY" / "1M.csv",
    ]
    for symbol in ["GLD", "UUP", "IEF", "SLV", "GDX", "VIXY"]:
        watched.append(root / "derived" / symbol / "1D.csv")
        watched.append(root / "derived" / symbol / "1M.csv")
    for series in ["CPIAUCSL", "DFII10", "WALCL", "M2SL", "VIXCLS"]:
        watched.append(root / "fred" / series / "1D.csv")
        watched.append(root / "fred" / series / "1M.csv")
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


def _load_latest_cached_gld_fair_value_base(cache_root: str | Path) -> FairValueDiagnostics | None:
    store_dir = _fair_value_store_dir(cache_root)
    if not store_dir.exists():
        return None
    candidates = sorted(
        store_dir.glob(f"{_GLD_FAIR_VALUE_CACHE_VERSION}_*.meta.json"),
        key=lambda p: p.stat().st_mtime_ns,
        reverse=True,
    )
    for meta_path in candidates:
        payload_path = meta_path.with_suffix("").with_suffix(".pkl")
        if not payload_path.exists():
            continue
        try:
            payload = pd.read_pickle(payload_path)
            frame = payload.get("frame")
            stats = payload.get("stats")
            model = payload.get("model")
            if not isinstance(frame, pd.DataFrame) or not isinstance(stats, dict) or not isinstance(model, dict):
                continue
            model = dict(model)
            model["cache_mode"] = "stale_fallback"
            return FairValueDiagnostics(
                symbol="GLD",
                frame=frame,
                stats=stats,
                model=model,
            )
        except Exception:
            continue
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


def _search_space_for_timeframe(is_monthly: bool) -> dict[str, Any]:
    one_layer_sets = {
        "macro_core": ["real_yield", "usd", "inflation", "liquidity", "stress"],
        "macro_core_no_usd": ["real_yield", "inflation", "liquidity", "stress", "money"],
        "macro_plus_market": ["real_yield", "usd", "inflation", "liquidity", "stress", "peer", "miners"],
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
        "macro_structural_official_no_usd": [
            "real_yield",
            "inflation",
            "liquidity",
            "stress",
            "money",
            "etf_official",
            "cb_official",
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
        "structural_core_no_usd": ["real_yield", "inflation", "liquidity", "stress", "money"],
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
        "structural_plus_money_official_no_usd": [
            "real_yield",
            "real_yield_trend",
            "inflation",
            "liquidity",
            "liquidity_trend",
            "money",
            "stress",
            "etf_official",
            "cb_official",
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
    if is_monthly:
        one_layer_sets = {
            **one_layer_sets,
            "macro_core_lag1": ["real_yield_lag1", "usd_lag1", "inflation_lag1", "liquidity_lag1", "stress_lag1"],
            "macro_core_lag2": ["real_yield_lag2", "usd_lag2", "inflation_lag2", "liquidity_lag2", "stress_lag2"],
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
        }
        structural_sets = {
            **structural_sets,
            "structural_core_lag1": [
                "real_yield_lag1",
                "usd_lag1",
                "inflation_lag1",
                "liquidity_lag1",
                "stress_lag1",
            ],
            "structural_core_lag2": [
                "real_yield_lag2",
                "usd_lag2",
                "inflation_lag2",
                "liquidity_lag2",
                "stress_lag2",
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
        return {
            "one_layer_sets": one_layer_sets,
            "structural_sets": structural_sets,
            "market_sets": market_sets,
            "z_windows": (12, 18, 24),
            "structural_fit_windows": (12, 18, 24, 30),
            "market_fit_windows": (12, 18),
            "ridge_alphas": (0.0, 0.5, 1.5),
            "smooth_spans": (2, 3, 4),
        }
    return {
        "one_layer_sets": one_layer_sets,
        "structural_sets": structural_sets,
        "market_sets": market_sets,
        "z_windows": (126, 252),
        "structural_fit_windows": (126, 252, 504),
        "market_fit_windows": (63, 126, 252),
        "ridge_alphas": (0.5, 1.5),
        "smooth_spans": (2, 5, 10, 21),
    }


def _search_best_fair_value_model(
    raw: pd.DataFrame,
    *,
    is_monthly: bool,
    data_timeframe: str,
    optional_sources: dict[str, str],
    target_source: str,
) -> FairValueDiagnostics | None:
    search = _search_space_for_timeframe(is_monthly)
    best: FairValueDiagnostics | None = None
    best_selection_score = -np.inf
    raw_index = raw.index

    for z_window in search["z_windows"]:
        features = _build_feature_frame(raw, z_window=z_window, is_monthly=is_monthly)
        for set_name, cols in search["one_layer_sets"].items():
            available_cols = [col for col in cols if col in features.columns]
            if not available_cols:
                continue
            for fit_window in search["structural_fit_windows"]:
                min_fit_obs = max(8, fit_window // 2)
                for ridge_alpha in search["ridge_alphas"]:
                    for smooth_span in search["smooth_spans"]:
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
                        stats.update(
                            _selection_score(
                                frame,
                                stats,
                                raw_index=raw_index,
                                is_monthly=is_monthly,
                            )
                        )
                        if stats["selection_score"] <= best_selection_score:
                            continue
                        best_selection_score = stats["selection_score"]
                        best = FairValueDiagnostics(
                            symbol="GLD",
                            frame=frame.reset_index(drop=True),
                            stats=stats,
                            model={
                                "model_type": "one_layer",
                                "data_timeframe": data_timeframe,
                                "target_source": target_source,
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
        for struct_name, structural_cols in search["structural_sets"].items():
            available_structural = [col for col in structural_cols if col in features.columns]
            if not available_structural:
                continue
            for market_name, market_cols in search["market_sets"].items():
                available_market = [col for col in market_cols if col in features.columns]
                for structural_fit_window in search["structural_fit_windows"]:
                    min_struct_obs = max(10, structural_fit_window // 2)
                    for market_fit_window in search["market_fit_windows"]:
                        min_market_obs = max(8, market_fit_window // 2)
                        for ridge_alpha in search["ridge_alphas"]:
                            for smooth_span in search["smooth_spans"]:
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
                                stats.update(
                                    _selection_score(
                                        frame,
                                        stats,
                                        raw_index=raw_index,
                                        is_monthly=is_monthly,
                                    )
                                )
                                if stats["selection_score"] <= best_selection_score:
                                    continue
                                best_selection_score = stats["selection_score"]
                                best = FairValueDiagnostics(
                                    symbol="GLD",
                                    frame=frame.reset_index(drop=True),
                                    stats=stats,
                                    model={
                                        "model_type": "two_layer",
                                        "data_timeframe": data_timeframe,
                                        "target_source": target_source,
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
    return best


def compute_gld_fair_value_diagnostics(
    *,
    cache_root: str | Path = "data_cache",
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> FairValueDiagnostics | None:
    try:
        best = _compute_gld_fair_value_base(str(Path(cache_root)), fair_value_cache_fingerprint(cache_root))
    except FileNotFoundError:
        best = None
    if best is None:
        best = _load_latest_cached_gld_fair_value_base(cache_root)
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
    window_stats = _fit_stats(frame)
    window_stats.update(
        _selection_score(
            frame,
            window_stats,
            raw_index=pd.Index(best.frame["date"]),
            is_monthly=str(best.model.get("data_timeframe", "1M")).upper() == "1M",
        )
    )
    return FairValueDiagnostics(
        symbol=best.symbol,
        frame=frame,
        stats=window_stats,
        model=best.model,
    )


@lru_cache(maxsize=4)
def _compute_gld_fair_value_base(cache_root: str, cache_fingerprint: str) -> FairValueDiagnostics | None:
    cached = _load_cached_gld_fair_value_base(cache_root, cache_fingerprint)
    if cached is not None:
        return cached

    best: FairValueDiagnostics | None = None
    best_score = -np.inf
    for timeframe in ("1D", "1M"):
        try:
            raw = _aligned_dataset(Path(cache_root), timeframe=timeframe)
        except FileNotFoundError:
            continue
        candidate = _search_best_fair_value_model(
            raw,
            is_monthly=(timeframe == "1M"),
            data_timeframe=timeframe,
            optional_sources=dict(raw.attrs.get("optional_sources", {})),
            target_source=str(raw.attrs.get("target_source", "unknown")),
        )
        if candidate is None:
            continue
        candidate_score = float(candidate.stats.get("selection_score", candidate.stats.get("fit_score", -np.inf)))
        if candidate_score <= best_score:
            continue
        best_score = candidate_score
        best = candidate

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
    diagnostics = compute_gld_fair_value_diagnostics(cache_root=cache_root)
    enriched = data.copy()
    if diagnostics is None or diagnostics.frame.empty or "date" not in enriched.columns:
        return enriched

    fair = diagnostics.frame.copy()
    fair["date"] = pd.to_datetime(fair["date"], errors="coerce")
    fair = fair.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    data_timeframe = str(diagnostics.model.get("data_timeframe", "1M")).upper()
    is_daily = data_timeframe == "1D"
    slope_smooth_span = 63 if is_daily else 3
    slope_lookback = 21 if is_daily else 1
    confidence_window = 252 if is_daily else 12
    slope_norm_window = 252 if is_daily else 12
    tolerance = pd.Timedelta(days=10 if is_daily else 45)

    fair_smoothed = fair["fair_value"].ewm(
        span=slope_smooth_span,
        adjust=False,
        min_periods=max(2, slope_smooth_span // 4),
    ).mean()
    fair["fair_slope_pct"] = fair_smoothed.pct_change(slope_lookback).fillna(0.0) * 100.0
    fair_slope_mean = fair["fair_slope_pct"].rolling(
        slope_norm_window,
        min_periods=max(6, slope_norm_window // 4),
    ).mean()
    fair_slope_std = fair["fair_slope_pct"].rolling(
        slope_norm_window,
        min_periods=max(6, slope_norm_window // 4),
    ).std()
    fair["fair_slope_z"] = (
        (fair["fair_slope_pct"] - fair_slope_mean)
        / fair_slope_std.replace(0.0, np.nan)
    )

    actual_dir = np.sign(fair["actual"].pct_change(slope_lookback).fillna(0.0))
    fair_dir = np.sign(fair_smoothed.pct_change(slope_lookback).fillna(0.0))
    fair["fair_confidence"] = (
        (actual_dir == fair_dir)
        .astype(float)
        .rolling(confidence_window, min_periods=max(6, confidence_window // 4))
        .mean()
    )

    regime_confidence_min = max(0.35, min(0.85, float(np.nanmedian(fair["fair_confidence"])) if fair["fair_confidence"].notna().any() else 0.5))
    hold_confidence_min = max(0.30, regime_confidence_min - 0.08)
    bullish_hold_slope_min = max(0.25, bullish_slope_min * 0.3)
    bearish_hold_slope_max = min(-0.25, bearish_slope_max * 0.3)
    bullish_hold_gap_floor = -(undervalued_gap_pct * 0.5)
    bearish_hold_gap_cap = overvalued_gap_pct * 1.5
    bullish_entry_slope_z_min = 0.0
    bullish_hold_slope_z_min = -0.35
    bearish_entry_slope_z_max = -0.75 if is_daily else -0.5
    bearish_hold_slope_z_max = -0.25

    regime: list[str] = []
    current = "neutral"
    for _, row in fair.iterrows():
        slope = float(row["fair_slope_pct"]) if not pd.isna(row["fair_slope_pct"]) else np.nan
        slope_z = float(row["fair_slope_z"]) if not pd.isna(row["fair_slope_z"]) else np.nan
        gap = float(row["fair_gap_pct"]) if not pd.isna(row["fair_gap_pct"]) else np.nan
        confidence = float(row["fair_confidence"]) if not pd.isna(row["fair_confidence"]) else np.nan

        bullish_entry = (
            not np.isnan(confidence)
            and confidence >= regime_confidence_min
            and not np.isnan(slope)
            and slope >= bullish_slope_min
            and not np.isnan(slope_z)
            and slope_z >= bullish_entry_slope_z_min
            and (np.isnan(gap) or gap >= 0.0)
        )
        bearish_entry = (
            not np.isnan(confidence)
            and confidence >= regime_confidence_min
            and (
                (not np.isnan(slope) and slope <= bearish_slope_max)
                or (not np.isnan(slope_z) and slope_z <= bearish_entry_slope_z_max)
            )
            and (np.isnan(gap) or gap <= overvalued_gap_pct)
        )
        bullish_hold = (
            not np.isnan(confidence)
            and confidence >= hold_confidence_min
            and not np.isnan(slope)
            and slope >= bullish_hold_slope_min
            and not np.isnan(slope_z)
            and slope_z >= bullish_hold_slope_z_min
            and (np.isnan(gap) or gap >= bullish_hold_gap_floor)
        )
        bearish_hold = (
            not np.isnan(confidence)
            and confidence >= hold_confidence_min
            and (
                (not np.isnan(slope) and slope <= bearish_hold_slope_max)
                or (not np.isnan(slope_z) and slope_z <= bearish_hold_slope_z_max)
            )
            and (np.isnan(gap) or gap <= bearish_hold_gap_cap)
        )

        if current == "bullish":
            if bullish_hold:
                regime.append("bullish")
                continue
            current = "neutral"
        elif current == "bearish":
            if bearish_hold:
                regime.append("bearish")
                continue
            current = "neutral"

        if bullish_entry and not bearish_entry:
            current = "bullish"
        elif bearish_entry and not bullish_entry:
            current = "bearish"
        else:
            current = "neutral"
        regime.append(current)

    fair["gold_fair_value_regime"] = regime
    fair["fair_macro_bearish"] = fair["gold_fair_value_regime"].eq("bearish").astype(float)
    fair["fair_valuation_overvalued"] = fair["fair_gap_pct"].le(-overvalued_gap_pct).astype(float)
    fair["fair_short_permission"] = fair["fair_macro_bearish"].astype(float)
    fair["fair_short_aggressive"] = (
        fair["fair_short_permission"].ge(0.5)
        & fair["fair_valuation_overvalued"].ge(0.5)
        & fair["fair_confidence"].ge(regime_confidence_min)
        & fair["fair_slope_z"].le(bearish_hold_slope_z_max)
    ).astype(float)
    # Backward-compatible alias for older consumers and cached diagnostics.
    fair["fair_short_boost"] = fair["fair_short_aggressive"]
    fair["fair_daily_rsi9"] = _calc_rsi(fair["actual"], period=9)

    keep_cols = [
        "date",
        "fair_value",
        "fair_gap_pct",
        "fair_slope_pct",
        "fair_confidence",
        "fair_daily_rsi9",
        "gold_fair_value_regime",
        "fair_macro_bearish",
        "fair_valuation_overvalued",
        "fair_short_permission",
        "fair_short_aggressive",
        "fair_short_boost",
    ]
    primary = enriched.copy()
    primary["date"] = pd.to_datetime(primary["date"], errors="coerce")
    merged = pd.merge_asof(
        primary.sort_values("date"),
        fair[keep_cols].sort_values("date"),
        on="date",
        direction="backward",
        tolerance=tolerance,
    )
    # Propagate the cache freshness flag onto every row so downstream
    # consumers (e.g. paper-trading entry gating) can detect a stale
    # fallback without reaching back into the diagnostics object. Backtest
    # is unaffected because it just records the value alongside the rest
    # of the strategy frame.
    cache_mode_value = str((diagnostics.model or {}).get("cache_mode", "live"))
    merged["fair_value_cache_mode"] = cache_mode_value
    return merged.reset_index(drop=True)
