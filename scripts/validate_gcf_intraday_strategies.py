from __future__ import annotations

import json
import sys
import types
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARTIFACT_DIR = ROOT / "artifacts" / "optimization"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
YF_CACHE_DIR = ROOT / "artifacts" / "tmp_yf_cache"
YF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

SYMBOL = "GC=F"
WINDOWS = [
    {"label": "60m_730d", "interval": "60m", "lookback_days": 730},
    {"label": "5m_59d", "interval": "5m", "lookback_days": 59},
    {"label": "1m_7d", "interval": "1m", "lookback_days": 7},
]
STRATEGY_IDS = ["vwap_rsi", "ema_trend_rsi"]


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

from config.settings import RiskConfig
from data.ingestion import prepare_strategy_data
from reporting.backtest import BacktestEngine
from risk.manager import RiskManager
from strategies import get_strategy


def _fetch_prices(interval: str, lookback_days: int) -> pd.DataFrame:
    try:
        yf.set_tz_cache_location(str(YF_CACHE_DIR.resolve()))
    except Exception:
        pass

    end = pd.Timestamp.utcnow().tz_localize(None).floor("min")
    raw = None
    attempted_days: list[int] = []
    for backoff_days in (0, 1, 2, 5):
        actual_days = max(1, int(lookback_days) - backoff_days)
        if actual_days in attempted_days:
            continue
        attempted_days.append(actual_days)
        start = end - pd.Timedelta(days=actual_days)
        raw = yf.download(
            SYMBOL,
            start=start.strftime("%Y-%m-%d"),
            end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            interval=interval,
            progress=False,
            auto_adjust=False,
            prepost=False,
        )
        if raw is not None and not raw.empty:
            break
    if raw is None or raw.empty:
        raise RuntimeError(
            f"No data returned for {SYMBOL} {interval} after trying lookbacks {attempted_days} days."
        )
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    out = raw.reset_index().rename(columns={"Datetime": "date", "Date": "date"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce", utc=True).dt.tz_localize(None)
    out = out.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    keep = ["date", "open", "high", "low", "close", "volume"]
    out = out[keep].dropna(subset=["date", "open", "high", "low", "close"]).copy()
    if "volume" not in out.columns:
        out["volume"] = 0.0
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0.0)
    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    return out


def _build_engine(strategy, *, spread_pct: float = 0.06, slippage_pct: float = 0.02) -> BacktestEngine:
    return BacktestEngine(
        strategy,
        risk_manager=RiskManager(
            RiskConfig(
                max_capital_per_trade_pct=100.0,
                max_daily_loss_pct=100.0,
                max_open_positions=999,
                default_max_loss_pct_of_capital=50.0,
            )
        ),
        spread_pct=spread_pct,
        slippage_pct=slippage_pct,
        commission_per_trade=0.0,
    )


def _trade_rows(result, strategy_id: str, window_label: str, interval: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trade in result.trades:
        row = asdict(trade)
        row["direction"] = str(getattr(trade.direction, "value", trade.direction))
        row["outcome"] = str(getattr(trade.outcome, "value", trade.outcome)) if trade.outcome is not None else None
        row["strategy_id"] = strategy_id
        row["window_label"] = window_label
        row["interval"] = interval
        row["entry_day"] = pd.Timestamp(trade.entry_time).date().isoformat() if trade.entry_time is not None else None
        rows.append(row)
    return rows


def _session_features(prices: pd.DataFrame, window_label: str, interval: str) -> pd.DataFrame:
    tmp = prices.copy()
    tmp["session_day"] = pd.to_datetime(tmp["date"]).dt.date
    grouped = (
        tmp.groupby("session_day")
        .agg(
            bars=("close", "size"),
            session_open=("open", "first"),
            session_close=("close", "last"),
            session_high=("high", "max"),
            session_low=("low", "min"),
        )
        .reset_index()
    )
    grouped["trend_pct"] = (grouped["session_close"] - grouped["session_open"]) / grouped["session_open"] * 100.0
    grouped["abs_trend_pct"] = grouped["trend_pct"].abs()
    grouped["range_pct"] = (grouped["session_high"] - grouped["session_low"]) / grouped["session_open"] * 100.0
    grouped["window_label"] = window_label
    grouped["interval"] = interval
    grouped["session_day"] = grouped["session_day"].astype(str)
    return grouped


def _run_strategy(prices: pd.DataFrame, strategy_id: str, interval: str):
    cls = get_strategy(strategy_id)
    base_strategy = cls(params={})
    params = base_strategy.effective_default_params(symbol=SYMBOL, source="yfinance", interval=interval)
    strategy = cls(params=params)
    prepared = prepare_strategy_data(
        prices.copy(),
        strategy,
        primary_symbol=SYMBOL,
        source="yfinance",
        interval=interval,
        start=prices["date"].min(),
        end=prices["date"].max(),
    )
    engine = _build_engine(strategy)
    result = engine.run(
        prepared,
        SYMBOL,
        leverage=1.0,
        capital_per_trade=1000.0,
        starting_equity=1000.0,
    )
    return params, result


def _compare_by_regime(trades_df: pd.DataFrame, session_df: pd.DataFrame) -> dict[str, Any]:
    if trades_df.empty or session_df.empty:
        return {}
    merged = trades_df.merge(
        session_df,
        left_on=["window_label", "interval", "entry_day"],
        right_on=["window_label", "interval", "session_day"],
        how="left",
    )
    merged["leveraged_return_pct"] = pd.to_numeric(merged["leveraged_return_pct"], errors="coerce").fillna(0.0)
    merged["pnl"] = pd.to_numeric(merged["pnl"], errors="coerce").fillna(0.0)

    summaries: dict[str, Any] = {}
    for (window_label, strategy_id), frame in merged.groupby(["window_label", "strategy_id"]):
        valid = frame.dropna(subset=["abs_trend_pct", "range_pct"]).copy()
        if valid.empty:
            continue
        trend_median = float(valid["abs_trend_pct"].median())
        low_trend = valid[valid["abs_trend_pct"] <= trend_median]
        high_trend = valid[valid["abs_trend_pct"] > trend_median]
        summaries[f"{window_label}:{strategy_id}"] = {
            "median_abs_trend_pct": round(trend_median, 4),
            "avg_abs_trend_pct_on_trades": round(float(valid["abs_trend_pct"].mean()), 4),
            "avg_range_pct_on_trades": round(float(valid["range_pct"].mean()), 4),
            "low_trend_trade_return_pct": round(float(low_trend["leveraged_return_pct"].mean()), 4) if not low_trend.empty else None,
            "high_trend_trade_return_pct": round(float(high_trend["leveraged_return_pct"].mean()), 4) if not high_trend.empty else None,
            "profitable_trade_abs_trend_pct": round(float(valid.loc[valid["pnl"] > 0, "abs_trend_pct"].mean()), 4) if (valid["pnl"] > 0).any() else None,
            "losing_trade_abs_trend_pct": round(float(valid.loc[valid["pnl"] <= 0, "abs_trend_pct"].mean()), 4) if (~(valid["pnl"] > 0)).any() else None,
        }
    return summaries


def main() -> None:
    all_summary: list[dict[str, Any]] = []
    all_trades: list[dict[str, Any]] = []
    all_sessions: list[pd.DataFrame] = []

    for window in WINDOWS:
        label = str(window["label"])
        interval = str(window["interval"])
        lookback_days = int(window["lookback_days"])
        prices = _fetch_prices(interval, lookback_days)
        prices.to_csv(ARTIFACT_DIR / f"gcf_intraday_prices_{label}.csv", index=False)

        session_df = _session_features(prices, label, interval)
        all_sessions.append(session_df)

        for strategy_id in STRATEGY_IDS:
            params, result = _run_strategy(prices, strategy_id, interval)
            all_summary.append(
                {
                    "window_label": label,
                    "interval": interval,
                    "lookback_days": lookback_days,
                    "bars": int(len(prices)),
                    "strategy_id": strategy_id,
                    "strategy_name": get_strategy(strategy_id).name,
                    "window_start": str(pd.Timestamp(prices["date"].min())),
                    "window_end": str(pd.Timestamp(prices["date"].max())),
                    "total_return_pct": float(result.total_return_pct),
                    "win_rate_pct": float(result.win_rate_pct),
                    "max_drawdown_pct": float(result.max_drawdown_pct),
                    "sharpe_ratio": float(result.sharpe_ratio),
                    "total_trades": int(result.total_trades),
                    "winning_trades": int(result.winning_trades),
                    "losing_trades": int(result.losing_trades),
                    "avg_win_pct": float(result.avg_win_pct),
                    "avg_loss_pct": float(result.avg_loss_pct),
                    "params": json.dumps(params, sort_keys=True),
                }
            )
            all_trades.extend(_trade_rows(result, strategy_id, label, interval))

    summary_df = pd.DataFrame(all_summary)
    trades_df = pd.DataFrame(all_trades)
    sessions_df = pd.concat(all_sessions, ignore_index=True) if all_sessions else pd.DataFrame()
    regime_summary = _compare_by_regime(trades_df, sessions_df)

    summary_df.to_csv(ARTIFACT_DIR / "gcf_intraday_strategy_compare_summary.csv", index=False)
    trades_df.to_csv(ARTIFACT_DIR / "gcf_intraday_strategy_compare_trades.csv", index=False)
    sessions_df.to_csv(ARTIFACT_DIR / "gcf_intraday_strategy_compare_sessions.csv", index=False)

    payload = {
        "symbol": SYMBOL,
        "windows": WINDOWS,
        "strategy_results": all_summary,
        "regime_summary": regime_summary,
        "artifacts": {
            "summary": str(ARTIFACT_DIR / "gcf_intraday_strategy_compare_summary.csv"),
            "trades": str(ARTIFACT_DIR / "gcf_intraday_strategy_compare_trades.csv"),
            "sessions": str(ARTIFACT_DIR / "gcf_intraday_strategy_compare_sessions.csv"),
        },
    }
    (ARTIFACT_DIR / "gcf_intraday_strategy_compare_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
