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

XLE_CACHE = ROOT / "data_cache" / "alpaca" / "XLE" / "1Min.csv"
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
    trades = float(result.total_trades)
    raw_sharpe = float(result.sharpe_ratio)
    sharpe = 0.0 if math.isnan(raw_sharpe) else raw_sharpe
    sharpe_weight = min(trades / 10.0, 1.0)
    sharpe_bonus = sharpe * 10.0 * sharpe_weight
    trades_bonus = min(trades, 80.0) * 0.10
    dd_penalty = abs(float(result.max_drawdown_pct)) * 2.5
    low_trade_penalty = max(0.0, 8.0 - trades) * 3.0
    return float(result.total_return_pct) + sharpe_bonus + trades_bonus - dd_penalty - low_trade_penalty


def _load_prices() -> pd.DataFrame:
    prices = pd.read_csv(XLE_CACHE)
    prices["date"] = pd.to_datetime(prices["date"])
    return prices[(prices["date"] >= WINDOW_START) & (prices["date"] <= WINDOW_END)].reset_index(drop=True)


def _defaults_for(symbol: str | None = None) -> dict[str, Any]:
    cls = get_strategy("bollinger_rsi")
    strategy = cls(params={})
    if symbol is None:
        return dict(strategy.default_params())
    return dict(strategy.effective_default_params(symbol=symbol))


def _prepare_data(prices: pd.DataFrame) -> pd.DataFrame:
    cls = get_strategy("bollinger_rsi")
    strategy = cls(params={})
    return prepare_strategy_data(
        prices,
        strategy,
        primary_symbol="XLE",
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
        "XLE",
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


def _coordinate_sweep(prepared_prices: pd.DataFrame) -> tuple[EvalResult, list[EvalResult], dict[str, EvalResult]]:
    baselines: dict[str, EvalResult] = {}
    for family_symbol in (None, "QQQ", "SPY", "VXZ", "USO", "XLF"):
        label = "generic_default" if family_symbol is None else f"baseline_{family_symbol.lower()}_family"
        baseline = _evaluate(prepared_prices, _defaults_for(family_symbol))
        baseline.label = label
        baselines[label] = baseline

    history: list[EvalResult] = list(baselines.values())
    best = max(history, key=lambda r: r.score)
    tuned = dict(best.params)

    search_space: dict[str, list[Any]] = {
        "normal_long_enabled": [False, True],
        "normal_short_enabled": [False, True],
        "min_atr_pct": [0.2, 0.25, 0.3, 0.35],
        "trend_bias_long_enabled": [True, False],
        "trend_bias_fast_ema": [120, 156, 195],
        "trend_bias_slow_ema": [780, 975, 1170],
        "trend_bias_lookback_bars": [60, 90, 120],
        "trend_bias_min_retrace_pct": [0.4, 0.5, 0.6, 0.8],
        "trend_bias_min_momentum_120": [0.4, 0.7, 1.0, 1.3],
        "trend_bias_min_atr_pct": [0.02, 0.03, 0.04, 0.05],
        "trend_bias_min_rsi": [47.0, 49.0, 51.0, 53.0],
        "trend_bias_max_rsi": [70.0, 72.0, 74.0],
        "trend_bias_trail_pct": [3.2, 3.6, 4.0, 4.4, 4.8],
        "trend_bias_sl_pct": [1.1, 1.3, 1.5, 1.7, 2.0],
        "trend_bias_cooldown": [90, 120, 180],
        "trend_bias_no_higher_reentry": [False, True],
        "shock_rebound_long_enabled": [False, True],
        "rsi_flush_rebound_long_enabled": [False, True],
        "spike_momentum_max": [0, 1, 2],
        "spike_long_max": [0, 1],
        "spike_max_entries": [0, 1, 2],
        "event_target_short_enabled": [False, True],
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
                candidate = _evaluate(prepared_prices, candidate_params)
                candidate.label = f"{key}={value}"
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
    prepared = _prepare_data(prices)
    best, history, baselines = _coordinate_sweep(prepared)
    top = sorted(history, key=lambda r: r.score, reverse=True)[:20]

    payload = {
        "symbol": "XLE",
        "window_start": str(WINDOW_START),
        "window_end": str(WINDOW_END),
        "costs": {"spread_pct": 0.06, "slippage_pct": 0.02, "commission": 0.0},
        "generic_default": asdict(baselines["generic_default"]),
        "baseline_qqq_family": asdict(baselines["baseline_qqq_family"]),
        "baseline_spy_family": asdict(baselines["baseline_spy_family"]),
        "baseline_vxz_family": asdict(baselines["baseline_vxz_family"]),
        "baseline_uso_family": asdict(baselines["baseline_uso_family"]),
        "baseline_xlf_family": asdict(baselines["baseline_xlf_family"]),
        "best_current_engine": asdict(best),
        "top_search": [asdict(r) for r in top],
    }

    (ARTIFACT_DIR / "xle_optimizer_current_engine_results.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame([asdict(r) for r in top]).to_csv(
        ARTIFACT_DIR / "xle_optimizer_current_engine_top_search.csv",
        index=False,
    )

    print("Best XLE current-engine candidate:")
    print(json.dumps(asdict(best), indent=2))
    print()
    print(f"Wrote results to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
