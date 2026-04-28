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
    params: dict[str, Any]


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


def _evaluate(prepared_prices: pd.DataFrame, overrides: dict[str, Any], label: str) -> EvalResult:
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
        params=dict(overrides),
    )


def main() -> None:
    prices = _load_gld_prices()
    base = _effective_gld_defaults()
    prepared = _prepare_gld_data(prices, base)

    threshold_sets = [
        {
            "name": "q30_q70",
            "gld_volatility_calm_atr_pct_max": 0.029,
            "gld_volatility_calm_bw_pct_max": 0.123,
            "gld_volatility_expansion_atr_pct_min": 0.054,
            "gld_volatility_expansion_bw_pct_min": 0.259,
        },
        {
            "name": "q40_q80",
            "gld_volatility_calm_atr_pct_max": 0.0345,
            "gld_volatility_calm_bw_pct_max": 0.1486,
            "gld_volatility_expansion_atr_pct_min": 0.0658,
            "gld_volatility_expansion_bw_pct_min": 0.3268,
        },
        {
            "name": "mid_manual",
            "gld_volatility_calm_atr_pct_max": 0.03,
            "gld_volatility_calm_bw_pct_max": 0.15,
            "gld_volatility_expansion_atr_pct_min": 0.055,
            "gld_volatility_expansion_bw_pct_min": 0.26,
        },
    ]

    overlays = [
        {
            "name": "expansion_blocks_rebound_longs",
            "gld_volatility_block_shock_rebound_long_in_expansion": True,
            "gld_volatility_block_rsi_flush_long_in_expansion": True,
            "gld_volatility_block_spike_momentum_long_in_calm": False,
            "gld_volatility_block_shock_reversal_short_in_calm": False,
            "gld_volatility_block_cascade_breakdown_short_in_calm": False,
            "gld_volatility_block_intraday_pullback_short_in_calm": False,
            "gld_volatility_block_event_target_short_in_calm": False,
        },
        {
            "name": "calm_blocks_continuation",
            "gld_volatility_block_shock_rebound_long_in_expansion": False,
            "gld_volatility_block_rsi_flush_long_in_expansion": False,
            "gld_volatility_block_spike_momentum_long_in_calm": True,
            "gld_volatility_block_shock_reversal_short_in_calm": False,
            "gld_volatility_block_cascade_breakdown_short_in_calm": True,
            "gld_volatility_block_intraday_pullback_short_in_calm": False,
            "gld_volatility_block_event_target_short_in_calm": True,
        },
        {
            "name": "calm_blocks_continuation_plus_shock_reversal",
            "gld_volatility_block_shock_rebound_long_in_expansion": False,
            "gld_volatility_block_rsi_flush_long_in_expansion": False,
            "gld_volatility_block_spike_momentum_long_in_calm": True,
            "gld_volatility_block_shock_reversal_short_in_calm": True,
            "gld_volatility_block_cascade_breakdown_short_in_calm": True,
            "gld_volatility_block_intraday_pullback_short_in_calm": False,
            "gld_volatility_block_event_target_short_in_calm": True,
        },
        {
            "name": "both_core",
            "gld_volatility_block_shock_rebound_long_in_expansion": True,
            "gld_volatility_block_rsi_flush_long_in_expansion": True,
            "gld_volatility_block_spike_momentum_long_in_calm": True,
            "gld_volatility_block_shock_reversal_short_in_calm": False,
            "gld_volatility_block_cascade_breakdown_short_in_calm": True,
            "gld_volatility_block_intraday_pullback_short_in_calm": False,
            "gld_volatility_block_event_target_short_in_calm": True,
        },
        {
            "name": "both_plus_shock_reversal",
            "gld_volatility_block_shock_rebound_long_in_expansion": True,
            "gld_volatility_block_rsi_flush_long_in_expansion": True,
            "gld_volatility_block_spike_momentum_long_in_calm": True,
            "gld_volatility_block_shock_reversal_short_in_calm": True,
            "gld_volatility_block_cascade_breakdown_short_in_calm": True,
            "gld_volatility_block_intraday_pullback_short_in_calm": False,
            "gld_volatility_block_event_target_short_in_calm": True,
        },
        {
            "name": "both_plus_intraday_pullback",
            "gld_volatility_block_shock_rebound_long_in_expansion": True,
            "gld_volatility_block_rsi_flush_long_in_expansion": True,
            "gld_volatility_block_spike_momentum_long_in_calm": True,
            "gld_volatility_block_shock_reversal_short_in_calm": False,
            "gld_volatility_block_cascade_breakdown_short_in_calm": True,
            "gld_volatility_block_intraday_pullback_short_in_calm": True,
            "gld_volatility_block_event_target_short_in_calm": True,
        },
        {
            "name": "shock_rebound_only_in_expansion",
            "gld_volatility_block_shock_rebound_long_in_expansion": True,
            "gld_volatility_block_rsi_flush_long_in_expansion": False,
            "gld_volatility_block_spike_momentum_long_in_calm": False,
            "gld_volatility_block_shock_reversal_short_in_calm": False,
            "gld_volatility_block_cascade_breakdown_short_in_calm": False,
            "gld_volatility_block_intraday_pullback_short_in_calm": False,
            "gld_volatility_block_event_target_short_in_calm": False,
        },
    ]

    history: list[EvalResult] = []
    baseline = _evaluate(prepared, base, "baseline_current_default")
    history.append(baseline)

    best = baseline
    for thresh in threshold_sets:
        for overlay in overlays:
            params = dict(base)
            params.update(
                {
                    "gld_volatility_router_enabled": True,
                    "gld_volatility_calm_atr_pct_max": thresh["gld_volatility_calm_atr_pct_max"],
                    "gld_volatility_calm_bw_pct_max": thresh["gld_volatility_calm_bw_pct_max"],
                    "gld_volatility_expansion_atr_pct_min": thresh["gld_volatility_expansion_atr_pct_min"],
                    "gld_volatility_expansion_bw_pct_min": thresh["gld_volatility_expansion_bw_pct_min"],
                }
            )
            params.update(overlay)
            label = f"{thresh['name']}__{overlay['name']}"
            result = _evaluate(prepared, params, label)
            history.append(result)
            if result.score > best.score:
                best = result

    top = sorted(history, key=lambda r: r.score, reverse=True)[:20]
    payload = {
        "symbol": "GLD",
        "window_start": str(WINDOW_START),
        "window_end": str(WINDOW_END),
        "costs": {"spread_pct": 0.06, "slippage_pct": 0.02, "commission": 0.0},
        "sizing_mode": "min(capital_per_trade, current_equity)",
        "baseline": asdict(baseline),
        "best_volatility_router_1x": asdict(best),
        "top_search": [asdict(r) for r in top],
    }

    (ARTIFACT_DIR / "gld_optimizer_volatility_router_results.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame([asdict(r) for r in top]).to_csv(
        ARTIFACT_DIR / "gld_optimizer_volatility_router_top_search.csv",
        index=False,
    )

    print("Baseline current GLD default:")
    print(json.dumps(asdict(baseline), indent=2))
    print()
    print("Best GLD volatility-router 1x candidate:")
    print(json.dumps(asdict(best), indent=2))
    print()
    print(f"Wrote results to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
