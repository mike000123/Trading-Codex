"""
Slow fair-value diagnostics for macro-sensitive symbols.

The first supported use case is GLD, where we want a continuous
background fair-value proxy from slower macro and cross-asset inputs
before deciding whether it deserves to influence live trading logic.
"""
from __future__ import annotations

from dataclasses import dataclass
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


def _rolling_zscore(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    min_obs = min_periods or max(20, window // 3)
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


def _aligned_daily_dataset(cache_root: Path) -> pd.DataFrame:
    cache = DataCache(root=cache_root)
    required = {
        "GLD": ("derived", "GLD", "1D"),
        "UUP": ("derived", "UUP", "1D"),
        "IEF": ("derived", "IEF", "1D"),
        "SLV": ("derived", "SLV", "1D"),
        "GDX": ("derived", "GDX", "1D"),
        "VIXY": ("derived", "VIXY", "1D"),
        "CPIAUCSL": ("fred", "CPIAUCSL", "1D"),
        "DFII10": ("fred", "DFII10", "1D"),
        "WALCL": ("fred", "WALCL", "1D"),
        "M2SL": ("fred", "M2SL", "1D"),
        "VIXCLS": ("fred", "VIXCLS", "1D"),
    }

    series_map: dict[str, pd.Series] = {}
    for label, (source, symbol, timeframe) in required.items():
        series = _load_close_series(cache, source, symbol, timeframe)
        if series is None:
            raise FileNotFoundError(f"Missing cached slow dataset: {source}/{symbol}/{timeframe}")
        series_map[label] = series.rename(label)

    base_index = series_map["GLD"].index
    df = pd.DataFrame(index=base_index)
    for label, series in series_map.items():
        df[label] = series.reindex(base_index).ffill()
    return df.dropna(subset=["GLD"]).copy()


def _build_feature_frame(raw: pd.DataFrame, z_window: int) -> pd.DataFrame:
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

    inflation_yoy = cpi.pct_change(252)
    liquidity_flow = np.log(walcl.replace(0.0, np.nan)).diff(63)
    money_flow = np.log(m2.replace(0.0, np.nan)).diff(126)
    gold_mom = gld.pct_change(63)

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


def compute_gld_fair_value_diagnostics(
    *,
    cache_root: str | Path = "data_cache",
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> FairValueDiagnostics | None:
    raw = _aligned_daily_dataset(Path(cache_root))
    feature_sets = {
        "macro_core": ["real_yield", "usd", "inflation", "liquidity", "stress"],
        "macro_plus_market": ["real_yield", "usd", "inflation", "liquidity", "stress", "peer", "miners"],
        "full": [
            "real_yield",
            "usd",
            "inflation",
            "liquidity",
            "stress",
            "peer",
            "miners",
            "money",
            "riskoff",
            "rates_proxy",
            "momentum",
        ],
    }
    z_windows = (63, 126, 252)
    fit_windows = (252, 378, 504)
    ridge_alphas = (0.0, 0.5, 2.0)
    smooth_spans = (5, 10, 21)

    best: FairValueDiagnostics | None = None
    best_score = -np.inf
    for z_window in z_windows:
        features = _build_feature_frame(raw, z_window=z_window)
        for set_name, cols in feature_sets.items():
            available_cols = [col for col in cols if col in features.columns]
            for fit_window in fit_windows:
                min_fit_obs = max(126, fit_window // 2)
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
                        frame["fair_gap_pct"] = (frame["fair_value"] / frame["actual"].replace(0.0, np.nan) - 1.0) * 100.0
                        stats = _fit_stats(frame)
                        if stats["fit_score"] <= best_score:
                            continue
                        best_score = stats["fit_score"]
                        best = FairValueDiagnostics(
                            symbol="GLD",
                            frame=frame.reset_index(drop=True),
                            stats=stats,
                            model={
                                "feature_set": set_name,
                                "features": tuple(available_cols),
                                "z_window": int(z_window),
                                "fit_window": int(fit_window),
                                "ridge_alpha": float(ridge_alpha),
                                "smooth_span": int(smooth_span),
                                "betas": betas,
                            },
                        )

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
