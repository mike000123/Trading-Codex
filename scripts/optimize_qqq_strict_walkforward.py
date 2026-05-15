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
class EvalResult:
    label: str
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    win_rate_pct: float
    total_trades: int
    score: float
    params: dict[str, Any]


@dataclass
class FoldChoice:
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    chosen_label: str
    chosen_params: dict[str, Any]
    train_result: dict[str, Any]
    test_result: dict[str, Any]


def _safe_sharpe(value: float) -> float:
    return 0.0 if math.isnan(float(value)) else float(value)


def _score_result(result) -> float:
    trades_bonus = min(float(result.total_trades), 80.0) * 0.02
    sharpe_bonus = _safe_sharpe(float(result.sharpe_ratio)) * 10.0
    dd_penalty = abs(float(result.max_drawdown_pct)) * 2.5
    return float(result.total_return_pct) + sharpe_bonus + trades_bonus - dd_penalty


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
    variants: dict[str, dict[str, Any]] = {}
    for trail in (4.0, 4.4, 4.8, 5.2):
        variants[f"trail_{trail:.1f}"] = {**dict(base), "trend_bias_trail_pct": trail}
    for min_rsi in (47.0, 49.0, 51.0):
        variants[f"min_rsi_{int(min_rsi)}"] = {**dict(base), "trend_bias_min_rsi": min_rsi}
    for sl in (1.1, 1.3, 1.5):
        variants[f"sl_{sl:.1f}"] = {**dict(base), "trend_bias_sl_pct": sl}
    for momentum in (0.5, 0.7, 0.9):
        variants[f"mom120_{momentum:.1f}"] = {**dict(base), "trend_bias_min_momentum_120": momentum}
    variants["current_default"] = dict(base)
    return variants


def _evaluate_window(prices: pd.DataFrame, params: dict[str, Any], start: str, end: str, label: str) -> EvalResult:
    window = prices[(prices["date"] >= pd.Timestamp(start)) & (prices["date"] <= pd.Timestamp(end))].reset_index(drop=True)
    cls = get_strategy("bollinger_rsi")
    prep_strategy = cls(params={})
    prepared = prepare_strategy_data(
        window,
        prep_strategy,
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
    return EvalResult(
        label=label,
        total_return_pct=float(result.total_return_pct),
        max_drawdown_pct=float(result.max_drawdown_pct),
        sharpe_ratio=_safe_sharpe(float(result.sharpe_ratio)),
        win_rate_pct=float(result.win_rate_pct),
        total_trades=int(result.total_trades),
        score=_score_result(result),
        params=dict(params),
    )


def _summarize_oos(choices: list[FoldChoice]) -> dict[str, Any]:
    compounded = 1.0
    returns: list[float] = []
    sharpes: list[float] = []
    total_trades = 0
    positive_folds = 0
    labels: dict[str, int] = {}
    for choice in choices:
        test = choice.test_result
        r = float(test["total_return_pct"])
        compounded *= 1.0 + (r / 100.0)
        returns.append(r)
        sharpes.append(float(test["sharpe_ratio"]))
        total_trades += int(test["total_trades"])
        if r > 0:
            positive_folds += 1
        labels[choice.chosen_label] = labels.get(choice.chosen_label, 0) + 1
    return {
        "compounded_oos_return_pct": round((compounded - 1.0) * 100.0, 3),
        "average_oos_fold_return_pct": round(sum(returns) / len(returns), 3),
        "median_oos_fold_return_pct": round(pd.Series(returns).median(), 3),
        "positive_oos_folds": positive_folds,
        "total_oos_folds": len(choices),
        "total_oos_trades": total_trades,
        "average_oos_sharpe": round(sum(sharpes) / len(sharpes), 4),
        "selection_counts": labels,
    }


def _summarize_eval_results(results: list[dict[str, Any]], label: str) -> dict[str, Any]:
    compounded = 1.0
    returns: list[float] = []
    sharpes: list[float] = []
    total_trades = 0
    positive_folds = 0
    for item in results:
        r = float(item["total_return_pct"])
        compounded *= 1.0 + (r / 100.0)
        returns.append(r)
        sharpes.append(float(item["sharpe_ratio"]))
        total_trades += int(item["total_trades"])
        if r > 0:
            positive_folds += 1
    return {
        "compounded_oos_return_pct": round((compounded - 1.0) * 100.0, 3),
        "average_oos_fold_return_pct": round(sum(returns) / len(returns), 3),
        "median_oos_fold_return_pct": round(pd.Series(returns).median(), 3),
        "positive_oos_folds": positive_folds,
        "total_oos_folds": len(results),
        "total_oos_trades": total_trades,
        "average_oos_sharpe": round(sum(sharpes) / len(sharpes), 4),
        "selection_counts": {label: len(results)},
    }


def main() -> None:
    prices = _load_prices()
    variants = _candidate_variants()
    fold_choices: list[FoldChoice] = []

    for i in range(1, len(FOLDS)):
        train_start = FOLDS[0][0]
        train_end = FOLDS[i - 1][1]
        test_start, test_end = FOLDS[i]

        ranked: list[EvalResult] = []
        for label, params in variants.items():
            ranked.append(_evaluate_window(prices, params, train_start, train_end, label))
        ranked.sort(key=lambda r: r.score, reverse=True)
        chosen = ranked[0]
        test_eval = _evaluate_window(prices, chosen.params, test_start, test_end, chosen.label)

        fold_choices.append(
            FoldChoice(
                train_start=str(pd.Timestamp(train_start)),
                train_end=str(pd.Timestamp(train_end)),
                test_start=str(pd.Timestamp(test_start)),
                test_end=str(pd.Timestamp(test_end)),
                chosen_label=chosen.label,
                chosen_params=dict(chosen.params),
                train_result=asdict(chosen),
                test_result=asdict(test_eval),
            )
        )

    current_default = variants["current_default"]
    current_default_oos = []
    for test_start, test_end in FOLDS[1:]:
        item = asdict(_evaluate_window(prices, current_default, test_start, test_end, "current_default"))
        item["fold_start"] = str(pd.Timestamp(test_start))
        item["fold_end"] = str(pd.Timestamp(test_end))
        current_default_oos.append(item)

    payload = {
        "symbol": SYMBOL,
        "window_start": str(WINDOW_START),
        "window_end": str(WINDOW_END),
        "costs": {"spread_pct": 0.06, "slippage_pct": 0.02, "commission": 0.0},
        "train_test_folds": [
            {
                "train_start": choice.train_start,
                "train_end": choice.train_end,
                "test_start": choice.test_start,
                "test_end": choice.test_end,
            }
            for choice in fold_choices
        ],
        "candidate_labels": list(variants.keys()),
        "walkforward_choices": [asdict(choice) for choice in fold_choices],
        "walkforward_summary": _summarize_oos(fold_choices),
        "current_default_oos": current_default_oos,
        "current_default_oos_summary": _summarize_eval_results(current_default_oos, "current_default"),
    }

    out = ARTIFACT_DIR / "qqq_strict_walkforward_optimization.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["walkforward_summary"], indent=2))
    print(json.dumps(payload["current_default_oos_summary"], indent=2))
    print(f"Wrote results to {out}")


if __name__ == "__main__":
    main()
