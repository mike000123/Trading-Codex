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

from config.settings import RiskConfig  # noqa: E402
from data.ingestion import prepare_strategy_data  # noqa: E402
from reporting.backtest import BacktestEngine  # noqa: E402
from risk.manager import RiskManager  # noqa: E402
from strategies import get_strategy  # noqa: E402


ARTIFACT_DIR = ROOT / "artifacts" / "optimization"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

UUP_CACHE = ROOT / "data_cache" / "alpaca" / "UUP" / "1Min.csv"
WINDOW_START = pd.Timestamp("2024-04-04")
WINDOW_END = pd.Timestamp("2026-04-23 23:59:00")


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
    trades_bonus = min(float(result.total_trades), 60.0) * 0.03
    sharpe_bonus = float(result.sharpe_ratio) * 10.0
    dd_penalty = abs(float(result.max_drawdown_pct)) * 2.0
    return float(result.total_return_pct) + sharpe_bonus + trades_bonus - dd_penalty


def _load_prices() -> pd.DataFrame:
    prices = pd.read_csv(UUP_CACHE)
    prices["date"] = pd.to_datetime(prices["date"])
    return prices[(prices["date"] >= WINDOW_START) & (prices["date"] <= WINDOW_END)].reset_index(drop=True)


def _defaults_for(strategy_id: str, symbol: str | None = None) -> dict[str, Any]:
    cls = get_strategy(strategy_id)
    strategy = cls(params={})
    if strategy_id == "bollinger_rsi" and symbol is not None:
        return dict(strategy.effective_default_params(symbol=symbol))
    return dict(strategy.default_params())


def _prepare_data(prices: pd.DataFrame, strategy_id: str, params: dict[str, Any]) -> pd.DataFrame:
    cls = get_strategy(strategy_id)
    strategy = cls(params=dict(params))
    return prepare_strategy_data(
        prices,
        strategy,
        primary_symbol="UUP",
        source="alpaca",
        interval="1Min",
        start=prices["date"].min(),
        end=prices["date"].max(),
    )


def _evaluate(prepared_prices: pd.DataFrame, strategy_id: str, overrides: dict[str, Any]) -> EvalResult:
    cls = get_strategy(strategy_id)
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
        enforce_rth=True,
        extended_hours=False,
        enforce_pdt=True,
        enforce_ssr=True,
        enforce_fractional=True,
        fill_diagnostic=True,
        enforce_monday_open_delay=False,
    ).run(
        prepared_prices,
        "UUP",
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


def _coordinate_sweep(prices: pd.DataFrame) -> tuple[EvalResult, list[EvalResult], dict[str, EvalResult]]:
    history: list[EvalResult] = []
    baselines: dict[str, EvalResult] = {}

    baseline_specs = [
        ("atr_rsi", None, "baseline_atr_rsi"),
        ("rsi_threshold", None, "baseline_rsi_threshold"),
        ("bollinger_rsi", None, "baseline_bollinger_generic"),
        ("bollinger_rsi", "GLD", "baseline_bollinger_gld_family"),
    ]
    prepared_cache: dict[tuple[str, str | None], pd.DataFrame] = {}

    for strategy_id, family_symbol, label in baseline_specs:
        params = _defaults_for(strategy_id, family_symbol)
        prepared = prepared_cache.get((strategy_id, family_symbol))
        if prepared is None:
            prepared = _prepare_data(prices, strategy_id, params)
            prepared_cache[(strategy_id, family_symbol)] = prepared
        baseline = _evaluate(prepared, strategy_id, params)
        baseline.label = label
        baselines[label] = baseline
        history.append(baseline)

    best = max(history, key=lambda r: r.score)
    tuned = dict(best.params)
    if best.label == "baseline_bollinger_gld_family":
        strategy_id = "bollinger_rsi"
    elif best.label == "baseline_bollinger_generic":
        strategy_id = "bollinger_rsi"
    elif best.label == "baseline_atr_rsi":
        strategy_id = "atr_rsi"
    else:
        strategy_id = "rsi_threshold"

    prepared = prepared_cache.get((strategy_id, "GLD" if best.label == "baseline_bollinger_gld_family" else None))
    if prepared is None:
        prepared = _prepare_data(prices, strategy_id, tuned)

    if strategy_id == "bollinger_rsi":
        search_space: dict[str, list[Any]] = {
            "normal_long_enabled": [False, True],
            "normal_short_enabled": [False, True],
            "bb_std": [1.6, 1.8, 2.0],
            "min_band_width_pct": [1.0, 1.5, 2.0],
            "min_atr_pct": [0.04, 0.08, 0.12],
            "trend_bias_long_enabled": [False, True],
            "trend_bias_min_retrace_pct": [0.4, 0.6, 0.8],
            "trend_bias_min_momentum_120": [0.0, 0.2, 0.4],
            "trend_bias_trail_pct": [2.0, 2.5, 3.0, 3.5],
            "trend_bias_sl_pct": [0.8, 1.0, 1.2, 1.4],
            "intraday_pullback_short_enabled": [False, True],
            "shock_rebound_long_enabled": [False, True],
            "rsi_flush_rebound_long_enabled": [False, True],
            "spike_momentum_max": [0, 1, 2],
            "spike_long_max": [0, 1, 2],
        }
    elif strategy_id == "atr_rsi":
        search_space = {
            "rsi_period": [7, 9, 14],
            "buy_levels": ["25", "30", "35"],
            "sell_levels": ["65", "70", "75"],
            "atr_period": [10, 14, 20],
            "atr_tp_mult": [1.5, 2.0, 2.5],
            "atr_sl_mult": [1.0, 1.5, 2.0],
            "tp_disabled": [False, True],
        }
    else:
        search_space = {
            "rsi_period": [7, 9, 14],
            "buy_levels": ["25", "30", "35"],
            "sell_levels": ["65", "70", "75"],
            "tp_pct": [0.6, 0.8, 1.0],
            "sl_pct": [0.4, 0.6, 0.8],
        }

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
                candidate = _evaluate(prepared, strategy_id, candidate_params)
                candidate.label = f"{strategy_id}_{key}={value}"
                history.append(candidate)
                if candidate.score > local_best.score:
                    local_best = candidate
            if local_best.score > best.score:
                tuned = dict(local_best.params)
                best = local_best
                improved = True

    return best, history, baselines


def main() -> None:
    prices = _load_prices()
    best, history, baselines = _coordinate_sweep(prices)
    top = sorted(history, key=lambda r: r.score, reverse=True)[:20]

    payload = {
        "symbol": "UUP",
        "window_start": str(WINDOW_START),
        "window_end": str(WINDOW_END),
        "costs": {"spread_pct": 0.06, "slippage_pct": 0.02, "commission": 0.0},
        "baseline_atr_rsi": asdict(baselines["baseline_atr_rsi"]),
        "baseline_rsi_threshold": asdict(baselines["baseline_rsi_threshold"]),
        "baseline_bollinger_generic": asdict(baselines["baseline_bollinger_generic"]),
        "baseline_bollinger_gld_family": asdict(baselines["baseline_bollinger_gld_family"]),
        "best_current_engine": asdict(best),
        "top_search": [asdict(r) for r in top],
    }

    (ARTIFACT_DIR / "uup_optimizer_current_engine_results.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame([asdict(r) for r in top]).to_csv(
        ARTIFACT_DIR / "uup_optimizer_current_engine_top_search.csv",
        index=False,
    )

    print("Best UUP current-engine candidate:")
    print(json.dumps(asdict(best), indent=2))
    print()
    print(f"Wrote results to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
