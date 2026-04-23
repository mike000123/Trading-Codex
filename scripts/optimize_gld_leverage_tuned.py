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
from config.strategy_presets.bollinger_rsi.gld_candidates import get_candidate
from data.ingestion import prepare_strategy_data
from reporting.backtest import BacktestEngine
from risk.manager import RiskManager
from strategies import get_strategy


ARTIFACT_DIR = ROOT / "artifacts" / "optimization"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

GLD_CACHE = ROOT / "data_cache" / "alpaca" / "GLD" / "1Min.csv"
WINDOW_START = pd.Timestamp("2024-04-04")
WINDOW_END = pd.Timestamp("2026-04-03 23:59:59")
TARGET_LEVERAGE = 5.0
TARGET_RISK_CAP = 50.0


@dataclass
class EvalResult:
    label: str
    leverage: float
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    total_trades: int
    score: float
    params: dict[str, Any]


def _score_result(result) -> float:
    trades_bonus = min(float(result.total_trades), 80.0) * 0.02
    sharpe_bonus = float(result.sharpe_ratio) * 14.0
    dd_penalty = abs(float(result.max_drawdown_pct)) * 2.4
    return float(result.total_return_pct) + sharpe_bonus + trades_bonus - dd_penalty


def _load_gld_prices() -> pd.DataFrame:
    prices = pd.read_csv(GLD_CACHE)
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices[(prices["date"] >= WINDOW_START) & (prices["date"] <= WINDOW_END)].reset_index(drop=True)
    return prices


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
                default_max_loss_pct_of_capital=TARGET_RISK_CAP,
            )
        ),
        spread_pct=0.06,
        slippage_pct=0.02,
        commission_per_trade=0.0,
    ).run(
        prepared_prices,
        "GLD",
        leverage=TARGET_LEVERAGE,
        capital_per_trade=1000.0,
        starting_equity=1000.0,
    )
    return EvalResult(
        label="candidate",
        leverage=TARGET_LEVERAGE,
        total_return_pct=float(result.total_return_pct),
        max_drawdown_pct=float(result.max_drawdown_pct),
        sharpe_ratio=float(result.sharpe_ratio),
        total_trades=int(result.total_trades),
        score=_score_result(result),
        params=dict(overrides),
    )


def _coordinate_sweep(prepared_prices: pd.DataFrame) -> tuple[EvalResult, list[EvalResult]]:
    tuned = get_candidate("gld_best_1x_sweep_20260419")
    tuned.pop("leverage", None)
    tuned.pop("risk_max_loss_pct_of_capital", None)

    if not tuned:
        raise RuntimeError("Base GLD 1x candidate not found.")

    search_space: dict[str, list[Any]] = {
        "spike_momo_sl_pct": [0.9, 1.0, 1.1, 1.2],
        "spike_momo_trail_pct": [2.6, 3.0, 3.4],
        "intraday_pullback_sl_pct": [0.6, 0.8, 1.0],
        "intraday_pullback_tp_pct": [1.0, 1.2, 1.4],
        "intraday_pullback_trail_pct": [0.4, 0.6, 0.8],
        "shock_reversal_sl_pct": [0.6, 0.8, 1.0],
        "shock_reversal_tp_pct": [1.2, 1.4, 1.6],
        "shock_reversal_trail_pct": [0.4, 0.5, 0.6],
        "cascade_breakdown_sl_pct": [0.6, 0.8, 1.0],
        "cascade_breakdown_tp_pct": [0.8, 1.0, 1.2],
        "cascade_breakdown_trail_pct": [0.4, 0.6, 0.8],
        "shock_rebound_sl_pct": [0.8, 1.0, 1.2],
        "shock_rebound_tp_pct": [2.0, 2.2, 2.4, 2.6],
        "shock_rebound_trail_pct": [0.9, 1.1, 1.3, 1.5],
        "spike_momentum_max": [5, 6, 7],
    }

    history: list[EvalResult] = []
    best = _evaluate(prepared_prices, tuned)
    best.label = "baseline_scaled_5x"
    history.append(best)

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
                candidate = _evaluate(prepared_prices, candidate_params)
                candidate.label = f"{key}={value}"
                history.append(candidate)
                if candidate.score > local_best.score:
                    local_best = candidate
            if local_best.score > best.score:
                tuned = dict(local_best.params)
                best = local_best
                improved = True

    return best, history


def main() -> None:
    prices = _load_gld_prices()
    prepared = _prepare_gld_data(prices)
    best, history = _coordinate_sweep(prepared)
    top_search = sorted(history, key=lambda r: r.score, reverse=True)[:20]

    payload = {
        "symbol": "GLD",
        "window_start": str(WINDOW_START),
        "window_end": str(WINDOW_END),
        "costs": {"spread_pct": 0.06, "slippage_pct": 0.02, "commission": 0.0},
        "sizing_mode": "min(capital_per_trade, current_equity)",
        "risk_cap_note": "Leverage-tuned search uses 5x leverage with max 50% capital loss cap via RiskManager.",
        "base_candidate": "gld_best_1x_sweep_20260419",
        "target_leverage": TARGET_LEVERAGE,
        "best_5x_tuned": asdict(best),
        "top_search": [asdict(r) for r in top_search],
    }

    (ARTIFACT_DIR / "gld_optimizer_5x_tuned_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame([asdict(r) for r in top_search]).to_csv(ARTIFACT_DIR / "gld_optimizer_5x_tuned_top_search.csv", index=False)

    print("Best leverage-tuned 5x candidate:")
    print(json.dumps(asdict(best), indent=2))
    print()
    print(f"Wrote results to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
