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

GLD_CACHE = ROOT / "data_cache" / "alpaca" / "GLD" / "1Min.csv"
WINDOW_START = pd.Timestamp("2024-04-04")
WINDOW_END = pd.Timestamp("2026-04-03 23:59:59")


@dataclass
class EvalResult:
    label: str
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    win_rate_pct: float
    total_trades: int
    score: float
    regime_loss_guard_rules: dict[str, dict[str, int]]


def _score_result(result) -> float:
    trades_bonus = min(float(result.total_trades), 80.0) * 0.025
    sharpe_bonus = float(result.sharpe_ratio) * 14.0
    dd_penalty = abs(float(result.max_drawdown_pct)) * 2.0
    return float(result.total_return_pct) + sharpe_bonus + trades_bonus - dd_penalty


def _load_gld_prices() -> pd.DataFrame:
    prices = pd.read_csv(GLD_CACHE)
    prices["date"] = pd.to_datetime(prices["date"])
    return prices[(prices["date"] >= WINDOW_START) & (prices["date"] <= WINDOW_END)].reset_index(drop=True)


def _effective_gld_defaults() -> dict[str, Any]:
    cls = get_strategy("bollinger_rsi")
    strategy = cls(params={})
    return dict(strategy.effective_default_params(symbol="GLD"))


def _prepare_gld_data(prices: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    cls = get_strategy("bollinger_rsi")
    strategy = cls(params=dict(params))
    return prepare_strategy_data(
        prices,
        strategy,
        primary_symbol="GLD",
        source="alpaca",
        interval="1Min",
        start=prices["date"].min(),
        end=prices["date"].max(),
    )


def _evaluate(prepared_prices: pd.DataFrame, params: dict[str, Any], label: str, rules: dict[str, dict[str, int]]) -> EvalResult:
    cls = get_strategy("bollinger_rsi")
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
        regime_loss_guard_rules=rules,
    ).run(
        prepared_prices,
        "GLD",
        leverage=1.0,
        capital_per_trade=1000.0,
        starting_equity=1000.0,
    )
    return EvalResult(
        label=label,
        total_return_pct=float(result.total_return_pct),
        max_drawdown_pct=float(result.max_drawdown_pct),
        sharpe_ratio=float(result.sharpe_ratio),
        win_rate_pct=float(result.win_rate_pct),
        total_trades=int(result.total_trades),
        score=_score_result(result),
        regime_loss_guard_rules={k: dict(v) for k, v in rules.items()},
    )


def main() -> None:
    prices = _load_gld_prices()
    base = _effective_gld_defaults()
    prepared = _prepare_gld_data(prices, base)

    candidates: list[tuple[str, dict[str, dict[str, int]]]] = [
        ("baseline_current_default", {}),
        (
            "shock_rebound_after_2_losses_cooldown_195",
            {"shock_rebound_long": {"trigger_losses": 2, "cooldown_bars": 195}},
        ),
        (
            "shock_rebound_after_2_losses_cooldown_390",
            {"shock_rebound_long": {"trigger_losses": 2, "cooldown_bars": 390}},
        ),
        (
            "shock_rebound_after_2_losses_cooldown_780",
            {"shock_rebound_long": {"trigger_losses": 2, "cooldown_bars": 780}},
        ),
        (
            "spike_momentum_after_3_losses_cooldown_390",
            {"spike_momentum_long": {"trigger_losses": 3, "cooldown_bars": 390}},
        ),
        (
            "spike_momentum_after_3_losses_cooldown_780",
            {"spike_momentum_long": {"trigger_losses": 3, "cooldown_bars": 780}},
        ),
        (
            "shock_rebound_2loss_390_plus_spike_momentum_3loss_390",
            {
                "shock_rebound_long": {"trigger_losses": 2, "cooldown_bars": 390},
                "spike_momentum_long": {"trigger_losses": 3, "cooldown_bars": 390},
            },
        ),
        (
            "shock_rebound_2loss_390_plus_spike_momentum_3loss_780",
            {
                "shock_rebound_long": {"trigger_losses": 2, "cooldown_bars": 390},
                "spike_momentum_long": {"trigger_losses": 3, "cooldown_bars": 780},
            },
        ),
        (
            "global_after_2_losses_cooldown_195",
            {"__all__": {"trigger_losses": 2, "cooldown_bars": 195}},
        ),
        (
            "global_after_2_losses_cooldown_390",
            {"__all__": {"trigger_losses": 2, "cooldown_bars": 390}},
        ),
        (
            "global_after_3_losses_cooldown_390",
            {"__all__": {"trigger_losses": 3, "cooldown_bars": 390}},
        ),
    ]

    history = [_evaluate(prepared, base, label, rules) for label, rules in candidates]
    best = max(history, key=lambda item: item.score)

    result_payload = {
        "baseline": asdict(history[0]),
        "best": asdict(best),
        "search_count": len(history),
        "history": [asdict(item) for item in history],
    }

    (ARTIFACT_DIR / "gld_optimizer_loss_cluster_guard_results.json").write_text(
        json.dumps(result_payload, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame([asdict(item) for item in history]).sort_values(
        ["score", "total_return_pct", "sharpe_ratio"], ascending=[False, False, False]
    ).to_csv(
        ARTIFACT_DIR / "gld_optimizer_loss_cluster_guard_top_search.csv",
        index=False,
    )

    print(json.dumps(result_payload, indent=2))


if __name__ == "__main__":
    main()
