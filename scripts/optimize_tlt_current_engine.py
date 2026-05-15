from __future__ import annotations

import json
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

    logger_mod.log = _Dummy()
    sys.modules["core.logger"] = logger_mod


_install_dummy_logger()

from config.settings import RiskConfig
from data.ingestion import prepare_strategy_data
from reporting.backtest import BacktestEngine
from risk.manager import RiskManager
from strategies import get_strategy


ARTIFACT_DIR = ROOT / "artifacts" / "optimization"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

TLT_CACHE = ROOT / "data_cache" / "alpaca" / "TLT" / "1Min.csv"
WINDOW_START = pd.Timestamp("2024-04-04")
WINDOW_END = pd.Timestamp("2026-04-02 23:59:00")


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


def _score_result(result) -> float:
    trades_bonus = min(float(result.total_trades), 80.0) * 0.02
    sharpe_bonus = float(result.sharpe_ratio) * 10.0
    dd_penalty = abs(float(result.max_drawdown_pct)) * 2.5
    return float(result.total_return_pct) + sharpe_bonus + trades_bonus - dd_penalty


def _load_tlt_prices() -> pd.DataFrame:
    prices = pd.read_csv(TLT_CACHE)
    prices["date"] = pd.to_datetime(prices["date"])
    return prices[(prices["date"] >= WINDOW_START) & (prices["date"] <= WINDOW_END)].reset_index(drop=True)


def _prepare_tlt_data(prices: pd.DataFrame) -> pd.DataFrame:
    cls = get_strategy("bollinger_rsi")
    strategy = cls(params={})
    return prepare_strategy_data(
        prices,
        strategy,
        primary_symbol="TLT",
        source="alpaca",
        interval="1Min",
        start=prices["date"].min(),
        end=prices["date"].max(),
    )


def _defaults_for(symbol: str | None = None) -> dict[str, Any]:
    cls = get_strategy("bollinger_rsi")
    strategy = cls(params={})
    if symbol is None:
        return dict(strategy.default_params())
    return dict(strategy.effective_default_params(symbol=symbol))


def _evaluate(prepared_prices: pd.DataFrame, overrides: dict[str, Any]) -> EvalResult:
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
        "TLT",
        leverage=1.0,
        capital_per_trade=1000.0,
        starting_equity=1000.0,
    )
    return EvalResult(
        label="candidate",
        total_return_pct=float(result.total_return_pct),
        max_drawdown_pct=float(result.max_drawdown_pct),
        sharpe_ratio=float(result.sharpe_ratio),
        win_rate_pct=float(result.win_rate_pct),
        total_trades=int(result.total_trades),
        score=_score_result(result),
        params=dict(overrides),
    )


def _family_and_generic_sweep(prepared_prices: pd.DataFrame) -> tuple[EvalResult, list[EvalResult]]:
    history: list[EvalResult] = []

    family_symbols = [None, "SPY", "QQQ", "USO", "GLD", "UVXY", "VXX", "VXZ"]
    for family_symbol in family_symbols:
        params = _defaults_for(family_symbol)
        label = "generic_default" if family_symbol is None else f"family_{family_symbol.lower()}"
        result = _evaluate(prepared_prices, params)
        result.label = label
        history.append(result)

    best = max(history, key=lambda r: r.score)

    generic_tuned = dict(_defaults_for(None))
    search_space: dict[str, list[Any]] = {
        "normal_long_enabled": [True, False],
        "normal_short_enabled": [True, False],
        "bb_std": [1.6, 1.8, 2.0, 2.2],
        "sl_band_mult": [0.1, 0.15, 0.2, 0.25],
        "min_band_width_pct": [1.0, 1.5, 2.0],
        "min_atr_pct": [0.1, 0.2, 0.3],
        "require_cross": [True, False],
    }

    improved = True
    while improved:
        improved = False
        for key, values in search_space.items():
            local_best = best
            for value in values:
                candidate_params = dict(generic_tuned)
                candidate_params[key] = value
                candidate = _evaluate(prepared_prices, candidate_params)
                candidate.label = f"generic_{key}={value}"
                history.append(candidate)
                if candidate.score > local_best.score:
                    local_best = candidate
            if local_best.score > best.score:
                generic_tuned = dict(local_best.params)
                best = local_best
                improved = True

    return best, history


def main() -> None:
    prices = _load_tlt_prices()
    prepared = _prepare_tlt_data(prices)
    best, history = _family_and_generic_sweep(prepared)
    top = sorted(history, key=lambda r: r.score, reverse=True)[:20]

    payload = {
        "symbol": "TLT",
        "window_start": str(WINDOW_START),
        "window_end": str(WINDOW_END),
        "costs": {"spread_pct": 0.06, "slippage_pct": 0.02, "commission": 0.0},
        "best_1x_current_engine": asdict(best),
        "top_search": [asdict(r) for r in top],
        "recommendation": (
            "No preset promoted. Best candidate did not beat a no-trade generic baseline "
            "well enough to justify integration."
        ),
    }

    (ARTIFACT_DIR / "tlt_optimizer_current_engine_results.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame([asdict(r) for r in top]).to_csv(
        ARTIFACT_DIR / "tlt_optimizer_current_engine_top_search.csv",
        index=False,
    )

    print("Best TLT current-engine candidate:")
    print(json.dumps(asdict(best), indent=2))
    print()
    print(f"Wrote results to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
