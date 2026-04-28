from __future__ import annotations

import argparse
import json
import re
import sys
import types
from pathlib import Path
from typing import Any

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

from config.settings import RiskConfig
from config.strategy_presets.bollinger_rsi.gld_candidates import get_candidate
from data.ingestion import prepare_strategy_data
from reporting.backtest import BacktestEngine
from risk.manager import RiskManager
from strategies import get_strategy


ARTIFACT_DIR = ROOT / "artifacts" / "optimization"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

GLD_CACHE = ROOT / "data_cache" / "alpaca" / "GLD" / "1Min.csv"
DEFAULT_START = pd.Timestamp("2024-04-04")
DEFAULT_END = pd.Timestamp("2026-04-03 23:59:59")
REGIME_RE = re.compile(r"regime=([^|]+)")


def _load_gld_prices(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    prices = pd.read_csv(GLD_CACHE)
    prices["date"] = pd.to_datetime(prices["date"])
    return prices[(prices["date"] >= start) & (prices["date"] <= end)].reset_index(drop=True)


def _resolved_params(candidate_name: str | None) -> dict[str, Any]:
    cls = get_strategy("bollinger_rsi")
    strategy = cls(params={})
    params = dict(strategy.effective_default_params(symbol="GLD"))
    if candidate_name:
        params.update(get_candidate(candidate_name))
    return params


def _prepare_and_run(params: dict[str, Any], start: pd.Timestamp, end: pd.Timestamp):
    prices = _load_gld_prices(start, end)
    cls = get_strategy("bollinger_rsi")
    strategy = cls(params=dict(params))
    prepared = prepare_strategy_data(
        prices,
        strategy,
        primary_symbol="GLD",
        source="alpaca",
        interval="1Min",
        start=prices["date"].min(),
        end=prices["date"].max(),
    )
    engine = BacktestEngine(
        strategy,
        risk_manager=RiskManager(
            RiskConfig(
                max_capital_per_trade_pct=100.0,
                max_daily_loss_pct=100.0,
                max_open_positions=999,
                default_max_loss_pct_of_capital=50.0,
            )
        ),
        spread_pct=0.06,
        slippage_pct=0.02,
        commission_per_trade=0.0,
    )
    return engine.run(
        prepared,
        "GLD",
        leverage=1.0,
        capital_per_trade=1000.0,
        starting_equity=1000.0,
    )


def _trade_frame(result) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trade in result.trades:
        entry_ts = pd.to_datetime(trade.entry_time, utc=True)
        exit_ts = pd.to_datetime(trade.exit_time, utc=True) if trade.exit_time is not None else pd.NaT
        entry_et = entry_ts.tz_convert("America/New_York")
        exit_et = exit_ts.tz_convert("America/New_York") if pd.notna(exit_ts) else pd.NaT
        match = REGIME_RE.search(trade.notes or "")
        regime = match.group(1).strip() if match else "unknown"
        minute_bucket = int((entry_et.minute // 30) * 30)
        bucket = f"{entry_et.hour:02d}:{minute_bucket:02d}"
        hold_minutes = None
        if pd.notna(exit_et):
            hold_minutes = float((exit_et - entry_et).total_seconds() / 60.0)
        rows.append(
            {
                "symbol": trade.symbol,
                "direction": str(trade.direction.value),
                "regime": regime,
                "entry_time_et": entry_et,
                "exit_time_et": exit_et,
                "entry_date_et": entry_et.date().isoformat(),
                "entry_weekday": entry_et.day_name(),
                "entry_bucket_30m": bucket,
                "entry_hour_et": int(entry_et.hour),
                "leveraged_return_pct": float(trade.leveraged_return_pct or 0.0),
                "pnl": float(trade.pnl or 0.0),
                "hold_minutes": hold_minutes,
                "outcome": str(trade.outcome.value if trade.outcome is not None else "Open"),
                "win": float((trade.pnl or 0.0) > 0.0),
            }
        )
    return pd.DataFrame(rows)


def _aggregate(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    grouped = (
        frame.groupby(group_cols, dropna=False)
        .agg(
            trades=("pnl", "size"),
            total_pnl=("pnl", "sum"),
            avg_pnl=("pnl", "mean"),
            median_pnl=("pnl", "median"),
            avg_return_pct=("leveraged_return_pct", "mean"),
            median_return_pct=("leveraged_return_pct", "median"),
            win_rate_pct=("win", lambda s: float(s.mean() * 100.0) if len(s) else 0.0),
            avg_hold_minutes=("hold_minutes", "mean"),
        )
        .reset_index()
    )
    grouped["expectancy_score"] = grouped["avg_return_pct"] * (grouped["win_rate_pct"] / 100.0)
    return grouped.sort_values(["expectancy_score", "avg_return_pct", "trades"], ascending=[False, False, False])


def main() -> None:
    parser = argparse.ArgumentParser(description="Report GLD time-of-day expectancy for the current engine.")
    parser.add_argument("--candidate", default="", help="Optional GLD candidate overlay name.")
    parser.add_argument("--start", default=str(DEFAULT_START.date()), help="Backtest start date.")
    parser.add_argument("--end", default=str(DEFAULT_END.date()), help="Backtest end date.")
    parser.add_argument("--max-hold-minutes", type=float, default=None, help="Optional max holding time filter.")
    parser.add_argument(
        "--exclude-regime",
        action="append",
        default=[],
        help="Optional regime(s) to exclude from the expectancy report. Can be passed multiple times.",
    )
    parser.add_argument("--min-trades", type=int, default=1, help="Minimum trades for top summary rows.")
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    params = _resolved_params(args.candidate or None)
    result = _prepare_and_run(params, start, end)
    trades = _trade_frame(result)
    if args.max_hold_minutes is not None:
        trades = trades[trades["hold_minutes"].fillna(float("inf")) <= float(args.max_hold_minutes)].copy()
    if args.exclude_regime:
        trades = trades[~trades["regime"].isin(args.exclude_regime)].copy()

    bucket_summary = _aggregate(trades, ["entry_bucket_30m"])
    regime_bucket_summary = _aggregate(trades, ["regime", "entry_bucket_30m"])
    regime_summary = _aggregate(trades, ["regime"])
    weekday_summary = _aggregate(trades, ["entry_weekday"])

    label_parts = [args.candidate or "current_default"]
    if args.max_hold_minutes is not None:
        label_parts.append(f"maxhold_{int(args.max_hold_minutes)}m")
    if args.exclude_regime:
        label_parts.append("ex_" + "_".join(sorted(args.exclude_regime)))
    label = "__".join(label_parts)
    trades.to_csv(ARTIFACT_DIR / f"gld_time_of_day_trades_{label}.csv", index=False)
    bucket_summary.to_csv(ARTIFACT_DIR / f"gld_time_of_day_buckets_{label}.csv", index=False)
    regime_bucket_summary.to_csv(ARTIFACT_DIR / f"gld_time_of_day_regime_buckets_{label}.csv", index=False)
    regime_summary.to_csv(ARTIFACT_DIR / f"gld_time_of_day_regimes_{label}.csv", index=False)
    weekday_summary.to_csv(ARTIFACT_DIR / f"gld_time_of_day_weekdays_{label}.csv", index=False)

    payload = {
        "candidate": label,
        "window_start": str(start),
        "window_end": str(end),
        "filters": {
            "max_hold_minutes": args.max_hold_minutes,
            "exclude_regime": list(args.exclude_regime),
            "min_trades": int(args.min_trades),
        },
        "total_trades": int(result.total_trades),
        "filtered_trades": int(len(trades)),
        "total_return_pct": float(result.total_return_pct),
        "max_drawdown_pct": float(result.max_drawdown_pct),
        "sharpe_ratio": float(result.sharpe_ratio),
        "top_30m_buckets": bucket_summary[bucket_summary["trades"] >= int(args.min_trades)].head(10).to_dict(orient="records"),
        "top_regimes": regime_summary[regime_summary["trades"] >= int(args.min_trades)].head(10).to_dict(orient="records"),
        "top_regime_bucket_rows": regime_bucket_summary[regime_bucket_summary["trades"] >= int(args.min_trades)].head(15).to_dict(orient="records"),
    }
    (ARTIFACT_DIR / f"gld_time_of_day_summary_{label}.json").write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )

    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
