from __future__ import annotations

import json
import math
import sys
import types
from dataclasses import asdict, dataclass
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

    dummy = _Dummy()
    logger_mod.log = dummy
    logger_mod.logger = dummy
    sys.modules["core.logger"] = logger_mod


_install_dummy_logger()

from config.settings import RiskConfig
from data.ingestion import prepare_strategy_data
from reporting.backtest import BacktestEngine
from risk.manager import RiskManager
from strategies import get_strategy


ARTIFACT_DIR = ROOT / "artifacts" / "optimization"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

WINDOWS = {
    "QQQ": (pd.Timestamp("2024-04-04"), pd.Timestamp("2026-04-02 23:59:00")),
    "IWM": (pd.Timestamp("2024-04-04"), pd.Timestamp("2026-04-02 23:59:00")),
    "XLF": (pd.Timestamp("2024-04-04"), pd.Timestamp("2026-04-02 23:59:00")),
}


@dataclass
class EvalResult:
    ticker: str
    label: str
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    win_rate_pct: float
    total_trades: int
    score: float
    params: dict[str, Any]


def _safe_sharpe(value: float) -> float:
    return 0.0 if math.isnan(float(value)) else float(value)


def _score_result(result) -> float:
    trades_bonus = min(float(result.total_trades), 80.0) * 0.02
    sharpe_bonus = _safe_sharpe(float(result.sharpe_ratio)) * 10.0
    dd_penalty = abs(float(result.max_drawdown_pct)) * 2.5
    return float(result.total_return_pct) + sharpe_bonus + trades_bonus - dd_penalty


def _load_prices(symbol: str) -> pd.DataFrame:
    cache = ROOT / "data_cache" / "alpaca" / symbol / "1Min.csv"
    prices = pd.read_csv(cache)
    prices["date"] = pd.to_datetime(prices["date"])
    start, end = WINDOWS[symbol]
    return prices[(prices["date"] >= start) & (prices["date"] <= end)].reset_index(drop=True)


def _prepare_data(symbol: str, prices: pd.DataFrame) -> pd.DataFrame:
    cls = get_strategy("bollinger_rsi")
    strategy = cls(params={})
    return prepare_strategy_data(
        prices,
        strategy,
        primary_symbol=symbol,
        source="alpaca",
        interval="1Min",
        start=prices["date"].min(),
        end=prices["date"].max(),
    )


def _defaults_for(symbol: str) -> dict[str, Any]:
    cls = get_strategy("bollinger_rsi")
    strategy = cls(params={})
    return dict(strategy.effective_default_params(symbol=symbol))


def _evaluate(symbol: str, prepared_prices: pd.DataFrame, overrides: dict[str, Any], label: str) -> EvalResult:
    cls = get_strategy("bollinger_rsi")
    strategy = cls(params=dict(overrides))
    result = BacktestEngine(
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
    ).run(
        prepared_prices,
        symbol,
        leverage=1.0,
        capital_per_trade=1000.0,
        starting_equity=1000.0,
    )
    return EvalResult(
        ticker=symbol,
        label=label,
        total_return_pct=float(result.total_return_pct),
        max_drawdown_pct=float(result.max_drawdown_pct),
        sharpe_ratio=_safe_sharpe(float(result.sharpe_ratio)),
        win_rate_pct=float(result.win_rate_pct),
        total_trades=int(result.total_trades),
        score=_score_result(result),
        params=dict(overrides),
    )


def _search_space_for(symbol: str) -> dict[str, list[Any]]:
    if symbol == "QQQ":
        return {
            "trend_bias_trail_pct": [3.6, 4.0, 4.4, 4.8],
            "trend_bias_sl_pct": [1.1, 1.3, 1.5, 1.7],
            "trend_bias_cooldown": [90, 120, 180],
            "trend_bias_min_rsi": [47.0, 49.0, 51.0],
            "trend_bias_min_retrace_pct": [0.4, 0.5, 0.6],
            "trend_bias_min_momentum_120": [0.5, 0.7, 0.9],
        }
    if symbol == "IWM":
        return {
            "bb_std": [1.9, 2.0, 2.1],
            "sl_band_mult": [0.1, 0.15, 0.18],
            "min_rr_ratio": [1.3, 1.5, 1.7, 2.0],
            "cooldown_bars": [0, 3, 5],
            "min_band_width_pct": [1.5, 2.0, 2.5],
            "min_atr_pct": [0.2, 0.3, 0.4],
            "require_cross": [True, False],
        }
    if symbol == "XLF":
        return {
            "trend_bias_trail_pct": [4.0, 5.0, 6.0],
            "trend_bias_sl_pct": [2.5, 3.0, 3.5],
            "intraday_pullback_tp_pct": [2.5, 3.0, 3.5],
            "shock_rebound_tp_pct": [3.5, 4.5, 5.5],
            "rsi_flush_tp_pct": [1.5, 2.0, 2.5],
            "trend_bias_cooldown": [1950, 3900, 7800],
        }
    raise ValueError(symbol)


def _coordinate_sweep(symbol: str, prepared_prices: pd.DataFrame) -> tuple[EvalResult, list[EvalResult]]:
    tuned = _defaults_for(symbol)
    history: list[EvalResult] = []

    best = _evaluate(symbol, prepared_prices, tuned, "baseline")
    history.append(best)

    search_space = _search_space_for(symbol)
    improved = True
    while improved:
        improved = False
        for key, values in search_space.items():
            local_best = best
            for value in values:
                if tuned.get(key) == value:
                    continue
                candidate_params = dict(tuned)
                candidate_params[key] = value
                candidate = _evaluate(symbol, prepared_prices, candidate_params, f"{key}={value}")
                history.append(candidate)
                if candidate.score > local_best.score:
                    local_best = candidate
            if local_best.score > best.score:
                tuned = dict(local_best.params)
                best = local_best
                improved = True
    return best, history


def _annualized_return(total_return_pct: float, years: float = 2.0) -> float:
    total_mult = 1.0 + (float(total_return_pct) / 100.0)
    if total_mult <= 0:
        return -100.0
    return (total_mult ** (1.0 / years) - 1.0) * 100.0


def main() -> None:
    payload: dict[str, Any] = {"tickers": {}}

    for symbol in ("QQQ", "IWM", "XLF"):
        prices = _load_prices(symbol)
        prepared = _prepare_data(symbol, prices)
        best, history = _coordinate_sweep(symbol, prepared)
        baseline = next(r for r in history if r.label == "baseline")
        top = sorted(history, key=lambda r: r.score, reverse=True)[:20]
        payload["tickers"][symbol] = {
            "window_start": str(WINDOWS[symbol][0]),
            "window_end": str(WINDOWS[symbol][1]),
            "baseline": asdict(baseline),
            "best_second_pass": asdict(best),
            "top_search": [asdict(r) for r in top],
            "estimated_annualized_return_pct": round(_annualized_return(best.total_return_pct), 3),
            "estimated_profit_per_1000_window": round(best.total_return_pct / 100.0 * 1000.0, 2),
        }
        print(symbol)
        print(json.dumps(payload["tickers"][symbol], indent=2))
        print()

    (ARTIFACT_DIR / "new_equity_etfs_second_pass_results.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote results to {ARTIFACT_DIR / 'new_equity_etfs_second_pass_results.json'}")


if __name__ == "__main__":
    main()
