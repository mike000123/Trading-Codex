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

SYMBOL = "QQQ"
WINDOW_START = pd.Timestamp("2024-04-04")
WINDOW_END = pd.Timestamp("2026-04-02 23:59:00")
FOLDS = [
    ("2024-04-04", "2024-09-30 23:59:00"),
    ("2024-10-01", "2025-03-31 23:59:00"),
    ("2025-04-01", "2025-09-30 23:59:00"),
    ("2025-10-01", "2026-04-02 23:59:00"),
]


@dataclass
class FoldResult:
    fold_start: str
    fold_end: str
    bars: int
    prepared_bars: int
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    total_trades: int
    win_rate_pct: float


def _safe_sharpe(value: float) -> float:
    return 0.0 if math.isnan(float(value)) else float(value)


def _load_prices() -> pd.DataFrame:
    cache = ROOT / "data_cache" / "alpaca" / SYMBOL / "1Min.csv"
    prices = pd.read_csv(cache)
    prices["date"] = pd.to_datetime(prices["date"])
    return prices[(prices["date"] >= WINDOW_START) & (prices["date"] <= WINDOW_END)].reset_index(drop=True)


def _base_params() -> dict[str, Any]:
    cls = get_strategy("bollinger_rsi")
    strategy = cls(params={})
    return dict(strategy.effective_default_params(symbol=SYMBOL))


def _candidate_variants() -> dict[str, dict[str, Any]]:
    base = _base_params()
    return {
        "current_default": dict(base),
        "trail_4.4": {**dict(base), "trend_bias_trail_pct": 4.4},
        "trail_5.2": {**dict(base), "trend_bias_trail_pct": 5.2},
        "min_rsi_51": {**dict(base), "trend_bias_min_rsi": 51.0},
        "legacy_trail_4.0": {**dict(base), "trend_bias_trail_pct": 4.0},
    }


def _evaluate_fold(prices: pd.DataFrame, params: dict[str, Any], start: str, end: str) -> FoldResult:
    window = prices[(prices["date"] >= pd.Timestamp(start)) & (prices["date"] <= pd.Timestamp(end))].reset_index(drop=True)
    cls = get_strategy("bollinger_rsi")
    strategy_for_prep = cls(params={})
    prepared = prepare_strategy_data(
        window,
        strategy_for_prep,
        primary_symbol=SYMBOL,
        source="alpaca",
        interval="1Min",
        start=window["date"].min(),
        end=window["date"].max(),
    )
    strategy = cls(params=dict(params))
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
        prepared,
        SYMBOL,
        leverage=1.0,
        capital_per_trade=1000.0,
        starting_equity=1000.0,
    )
    return FoldResult(
        fold_start=str(pd.Timestamp(start)),
        fold_end=str(pd.Timestamp(end)),
        bars=int(len(window)),
        prepared_bars=int(len(prepared)),
        total_return_pct=float(result.total_return_pct),
        max_drawdown_pct=float(result.max_drawdown_pct),
        sharpe_ratio=_safe_sharpe(float(result.sharpe_ratio)),
        total_trades=int(result.total_trades),
        win_rate_pct=float(result.win_rate_pct),
    )


def _summarize_walkforward(folds: list[FoldResult]) -> dict[str, Any]:
    compounded = 1.0
    positive_folds = 0
    total_trades = 0
    worst_fold = None
    best_fold = None
    returns: list[float] = []
    sharpes: list[float] = []
    for fold in folds:
        r = float(fold.total_return_pct)
        compounded *= 1.0 + (r / 100.0)
        if r > 0:
            positive_folds += 1
        total_trades += int(fold.total_trades)
        returns.append(r)
        sharpes.append(float(fold.sharpe_ratio))
        if worst_fold is None or r < worst_fold.total_return_pct:
            worst_fold = fold
        if best_fold is None or r > best_fold.total_return_pct:
            best_fold = fold
    return {
        "compounded_return_pct": round((compounded - 1.0) * 100.0, 3),
        "average_fold_return_pct": round(sum(returns) / len(returns), 3),
        "median_fold_return_pct": round(pd.Series(returns).median(), 3),
        "positive_folds": positive_folds,
        "total_folds": len(folds),
        "total_trades": total_trades,
        "average_sharpe": round(sum(sharpes) / len(sharpes), 4),
        "worst_fold_return_pct": round(float(worst_fold.total_return_pct), 3) if worst_fold else 0.0,
        "best_fold_return_pct": round(float(best_fold.total_return_pct), 3) if best_fold else 0.0,
    }


def main() -> None:
    prices = _load_prices()
    variants = _candidate_variants()
    payload: dict[str, Any] = {
        "symbol": SYMBOL,
        "window_start": str(WINDOW_START),
        "window_end": str(WINDOW_END),
        "costs": {"spread_pct": 0.06, "slippage_pct": 0.02, "commission": 0.0},
        "folds": [{"start": s, "end": e} for s, e in FOLDS],
        "variants": {},
    }

    for name, params in variants.items():
        fold_results = [_evaluate_fold(prices, params, start, end) for start, end in FOLDS]
        payload["variants"][name] = {
            "params": params,
            "fold_results": [asdict(f) for f in fold_results],
            "summary": _summarize_walkforward(fold_results),
        }
        print(name)
        print(json.dumps(payload["variants"][name]["summary"], indent=2))
        print()

    out = ARTIFACT_DIR / "qqq_walkforward_validation.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote results to {out}")


if __name__ == "__main__":
    main()
