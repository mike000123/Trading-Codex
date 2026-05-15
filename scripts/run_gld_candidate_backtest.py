from __future__ import annotations

import argparse
import json
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


GLD_CACHE = ROOT / "data_cache" / "alpaca" / "GLD" / "1Min.csv"
DEFAULT_START = pd.Timestamp("2024-04-04")
DEFAULT_END = pd.Timestamp("2026-04-03 23:59:59")


def _load_gld_prices(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    prices = pd.read_csv(GLD_CACHE)
    prices["date"] = pd.to_datetime(prices["date"])
    return prices[(prices["date"] >= start) & (prices["date"] <= end)].reset_index(drop=True)


def _prepare_gld_data(prices: pd.DataFrame) -> pd.DataFrame:
    cls = get_strategy("bollinger_rsi")
    strategy = cls(params={})
    return prepare_strategy_data(
        prices,
        strategy,
        primary_symbol="GLD",
        source="alpaca",
        interval="1Min",
        start=prices["date"].min(),
        end=prices["date"].max(),
    )


def _build_engine(max_loss_pct_of_capital: float) -> BacktestEngine:
    return BacktestEngine(
        get_strategy("bollinger_rsi")(params={}),
        risk_manager=RiskManager(
            RiskConfig(
                max_capital_per_trade_pct=100.0,
                max_daily_loss_pct=100.0,
                max_open_positions=999,
                default_max_loss_pct_of_capital=float(max_loss_pct_of_capital),
            )
        ),
        spread_pct=0.06,
        slippage_pct=0.02,
        commission_per_trade=0.0,
    )


def _run_candidate(candidate_name: str, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    candidate = get_candidate(candidate_name)
    if not candidate:
        raise SystemExit(f"Unknown candidate: {candidate_name}")

    leverage = float(candidate.pop("leverage", 1.0))
    risk_cap = float(candidate.pop("risk_max_loss_pct_of_capital", 50.0))

    prices = _load_gld_prices(start, end)
    prepared = _prepare_gld_data(prices)

    cls = get_strategy("bollinger_rsi")
    strategy = cls(params=dict(candidate))
    engine = BacktestEngine(
        strategy,
        risk_manager=RiskManager(
            RiskConfig(
                max_capital_per_trade_pct=100.0,
                max_daily_loss_pct=100.0,
                max_open_positions=999,
                default_max_loss_pct_of_capital=risk_cap,
            )
        ),
        spread_pct=0.06,
        slippage_pct=0.02,
        commission_per_trade=0.0,
    )
    result = engine.run(
        prepared,
        "GLD",
        leverage=leverage,
        capital_per_trade=1000.0,
        starting_equity=1000.0,
    )

    return {
        "candidate": candidate_name,
        "window_start": str(start),
        "window_end": str(end),
        "leverage": leverage,
        "risk_max_loss_pct_of_capital": risk_cap,
        "total_return_pct": float(result.total_return_pct),
        "max_drawdown_pct": float(result.max_drawdown_pct),
        "sharpe_ratio": float(result.sharpe_ratio),
        "total_trades": int(result.total_trades),
        "params": candidate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stored GLD optimizer candidates.")
    parser.add_argument(
        "--candidate",
        required=True,
        choices=[
            "gld_best_1x_sweep_20260419",
            "gld_best_leverage_5x_20260419",
            "gld_best_leverage_5x_tuned_20260419",
            "gld_rsi_spike_fade_short_20260426",
            "gld_fair_gap_fade_short_20260426",
            "gld_weak_0800_shock_reversal_filter_20260426",
        ],
        help="Stored GLD candidate preset to run.",
    )
    parser.add_argument("--start", default=str(DEFAULT_START.date()), help="Backtest start date.")
    parser.add_argument("--end", default=str(DEFAULT_END.date()), help="Backtest end date.")
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    payload = _run_candidate(args.candidate, start, end)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
