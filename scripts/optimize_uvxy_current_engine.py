from __future__ import annotations

import argparse
import json
import sys
import time
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

UVXY_CACHE = ROOT / "data_cache" / "alpaca" / "UVXY" / "1Min.csv"
WINDOW_START = pd.Timestamp("2024-04-04")
WINDOW_END = pd.Timestamp("2026-04-02 23:59:00")
AUG_START = pd.Timestamp("2024-08-01")
AUG_END = pd.Timestamp("2024-08-12 23:59:00")
APR_START = pd.Timestamp("2025-04-15")
APR_END = pd.Timestamp("2025-05-05 23:59:00")


@dataclass
class EvalResult:
    label: str
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    win_rate_pct: float
    total_trades: int
    aug_return_pct: float
    aug_trades: int
    apr_return_pct: float
    apr_trades: int
    score: float
    elapsed_sec: float
    params: dict[str, Any]


def _score_result(full_result, aug_result, apr_result) -> float:
    sharpe = 0.0 if pd.isna(full_result.sharpe_ratio) else float(full_result.sharpe_ratio)
    trade_bonus = min(float(full_result.total_trades), 220.0) * 0.03
    dd_penalty = abs(float(full_result.max_drawdown_pct)) * 2.5
    return (
        float(full_result.total_return_pct)
        + float(aug_result.total_return_pct) * 0.45
        + float(apr_result.total_return_pct) * 0.30
        + sharpe * 8.0
        + trade_bonus
        - dd_penalty
    )


def _engine(strategy) -> BacktestEngine:
    return BacktestEngine(
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
    )


def _evaluate_frame(frame: pd.DataFrame, overrides: dict[str, Any]):
    cls = get_strategy("bollinger_rsi")
    strategy = cls(params=dict(overrides))
    return _engine(strategy).run(
        frame,
        "UVXY",
        leverage=1.0,
        capital_per_trade=1000.0,
        starting_equity=1000.0,
    )


def _slice_frame(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    out = frame[(frame["date"] >= start) & (frame["date"] <= end)].reset_index(drop=True)
    if out.empty:
        raise RuntimeError(f"Slice {start} -> {end} returned no bars.")
    return out


def _load_prepared_uvxy() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prices = pd.read_csv(UVXY_CACHE)
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices[(prices["date"] >= WINDOW_START) & (prices["date"] <= WINDOW_END)].reset_index(drop=True)

    cls = get_strategy("bollinger_rsi")
    strategy = cls(params={})
    prepared = prepare_strategy_data(
        prices,
        strategy,
        primary_symbol="UVXY",
        source="alpaca",
        interval="1Min",
        start=prices["date"].min(),
        end=prices["date"].max(),
    )
    return (
        prepared,
        _slice_frame(prepared, AUG_START, AUG_END),
        _slice_frame(prepared, APR_START, APR_END),
    )


def _effective_uvxy_defaults() -> dict[str, Any]:
    cls = get_strategy("bollinger_rsi")
    strategy = cls(params={})
    return dict(strategy.effective_default_params(symbol="UVXY"))


def _build_candidates() -> list[tuple[str, dict[str, Any]]]:
    base = _effective_uvxy_defaults()
    return [
        ("baseline_uvxy_default", dict(base)),
        (
            "pullback_active_spike",
            {
                **base,
                "intraday_pullback_allow_active_spike": True,
                "intraday_pullback_spike_drawdown_pct": 1.0,
                "intraday_pullback_trail_pct": 1.2,
                "intraday_pullback_cooldown": 30,
            },
        ),
        (
            "shock_reversal_on",
            {
                **base,
                "shock_reversal_short_enabled": True,
                "shock_reversal_rsi_trigger": 82.0,
                "shock_reversal_max_current_rsi": 74.0,
                "shock_reversal_bar_drop_pct": 0.5,
                "shock_reversal_drop_pct": 0.6,
                "shock_reversal_sl_pct": 1.2,
                "shock_reversal_tp_pct": 1.8,
                "shock_reversal_trail_pct": 0.8,
                "shock_reversal_cooldown": 90,
            },
        ),
        (
            "rsi_flush_rebound_on",
            {
                **base,
                "rsi_flush_rebound_long_enabled": True,
                "rsi_flush_drop_pct": 0.8,
                "rsi_flush_rsi_trigger": 22.0,
                "rsi_flush_sl_pct": 1.1,
                "rsi_flush_tp_pct": 0.8,
                "rsi_flush_trail_pct": 1.8,
                "rsi_flush_cooldown": 30,
                "rsi_flush_require_green_rebound_bar": True,
                "rsi_flush_rebound_confirm_bars": 2,
                "rsi_flush_trend_filter_bars": 30,
            },
        ),
        (
            "spike_momentum_more",
            {
                **base,
                "spike_momentum_max": 3,
                "spike_momo_trail_pct": 10.0,
                "spike_momo_sl_pct": 2.5,
                "spike_momo_min_peak_pct": 3.0,
                "spike_momo_min_atr_pct": 0.5,
                "spike_momo_cooldown": 120,
            },
        ),
        (
            "event_target_earlier",
            {
                **base,
                "event_target_min_peak_pct": 70.0,
                "event_target_completion_pct": 85.0,
                "event_target_confirm_drop_pct": 10.0,
                "event_target_persistent_confirm_drop_pct": 18.0,
                "event_target_sl_pct": 10.0,
                "event_target_profit_giveback_frac": 0.45,
                "event_target_profit_giveback_min_pct": 4.0,
            },
        ),
        (
            "decay_bounce_faster",
            {
                **base,
                "decay_bounce_min_pct": 1.25,
                "decay_bounce_fail_pct": 0.9,
                "decay_bounce_cooldown": 120,
                "decay_bounce_max": 6,
            },
        ),
        (
            "combined_spike_decay",
            {
                **base,
                "intraday_pullback_allow_active_spike": True,
                "intraday_pullback_spike_drawdown_pct": 1.0,
                "intraday_pullback_trail_pct": 1.2,
                "intraday_pullback_cooldown": 30,
                "shock_reversal_short_enabled": True,
                "shock_reversal_rsi_trigger": 82.0,
                "shock_reversal_max_current_rsi": 74.0,
                "shock_reversal_bar_drop_pct": 0.5,
                "shock_reversal_drop_pct": 0.6,
                "shock_reversal_sl_pct": 1.2,
                "shock_reversal_tp_pct": 1.8,
                "shock_reversal_trail_pct": 0.8,
                "shock_reversal_cooldown": 90,
                "event_target_min_peak_pct": 70.0,
                "event_target_completion_pct": 85.0,
                "event_target_confirm_drop_pct": 10.0,
                "event_target_persistent_confirm_drop_pct": 18.0,
                "event_target_sl_pct": 10.0,
                "event_target_profit_giveback_frac": 0.45,
                "event_target_profit_giveback_min_pct": 4.0,
                "decay_bounce_min_pct": 1.25,
                "decay_bounce_fail_pct": 0.9,
                "decay_bounce_cooldown": 120,
                "decay_bounce_max": 6,
                "spike_momentum_max": 3,
                "spike_momo_trail_pct": 10.0,
                "spike_momo_sl_pct": 2.5,
                "spike_momo_min_peak_pct": 3.0,
                "spike_momo_min_atr_pct": 0.5,
                "spike_momo_cooldown": 120,
            },
        ),
        (
            "combined_all",
            {
                **base,
                "intraday_pullback_allow_active_spike": True,
                "intraday_pullback_spike_drawdown_pct": 1.0,
                "intraday_pullback_trail_pct": 1.2,
                "intraday_pullback_cooldown": 30,
                "shock_reversal_short_enabled": True,
                "shock_reversal_rsi_trigger": 82.0,
                "shock_reversal_max_current_rsi": 74.0,
                "shock_reversal_bar_drop_pct": 0.5,
                "shock_reversal_drop_pct": 0.6,
                "shock_reversal_sl_pct": 1.2,
                "shock_reversal_tp_pct": 1.8,
                "shock_reversal_trail_pct": 0.8,
                "shock_reversal_cooldown": 90,
                "rsi_flush_rebound_long_enabled": True,
                "rsi_flush_drop_pct": 0.8,
                "rsi_flush_rsi_trigger": 22.0,
                "rsi_flush_sl_pct": 1.1,
                "rsi_flush_tp_pct": 0.8,
                "rsi_flush_trail_pct": 1.8,
                "rsi_flush_cooldown": 30,
                "rsi_flush_require_green_rebound_bar": True,
                "rsi_flush_rebound_confirm_bars": 2,
                "rsi_flush_trend_filter_bars": 30,
                "event_target_min_peak_pct": 70.0,
                "event_target_completion_pct": 85.0,
                "event_target_confirm_drop_pct": 10.0,
                "event_target_persistent_confirm_drop_pct": 18.0,
                "event_target_sl_pct": 10.0,
                "event_target_profit_giveback_frac": 0.45,
                "event_target_profit_giveback_min_pct": 4.0,
                "decay_bounce_min_pct": 1.25,
                "decay_bounce_fail_pct": 0.9,
                "decay_bounce_cooldown": 120,
                "decay_bounce_max": 6,
                "spike_momentum_max": 3,
                "spike_momo_trail_pct": 10.0,
                "spike_momo_sl_pct": 2.5,
                "spike_momo_min_peak_pct": 3.0,
                "spike_momo_min_atr_pct": 0.5,
                "spike_momo_cooldown": 120,
            },
        ),
    ]


def _evaluate_suite(
    label: str,
    prepared_full: pd.DataFrame,
    prepared_aug: pd.DataFrame,
    prepared_apr: pd.DataFrame,
    overrides: dict[str, Any],
) -> EvalResult:
    t0 = time.perf_counter()
    full_result = _evaluate_frame(prepared_full, overrides)
    aug_result = _evaluate_frame(prepared_aug, overrides)
    apr_result = _evaluate_frame(prepared_apr, overrides)
    elapsed = time.perf_counter() - t0
    return EvalResult(
        label=label,
        total_return_pct=float(full_result.total_return_pct),
        max_drawdown_pct=float(full_result.max_drawdown_pct),
        sharpe_ratio=float(full_result.sharpe_ratio),
        win_rate_pct=float(full_result.win_rate_pct),
        total_trades=int(full_result.total_trades),
        aug_return_pct=float(aug_result.total_return_pct),
        aug_trades=int(aug_result.total_trades),
        apr_return_pct=float(apr_result.total_return_pct),
        apr_trades=int(apr_result.total_trades),
        score=_score_result(full_result, aug_result, apr_result),
        elapsed_sec=float(elapsed),
        params=dict(overrides),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize UVXY current-engine Bollinger preset.")
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional candidate labels to run. Default: run all bundled candidates.",
    )
    args = parser.parse_args()

    prepared_full, prepared_aug, prepared_apr = _load_prepared_uvxy()
    candidates = _build_candidates()
    if args.labels:
        wanted = set(args.labels)
        candidates = [(label, params) for label, params in candidates if label in wanted]
        missing = wanted.difference({label for label, _ in candidates})
        if missing:
            raise SystemExit(f"Unknown candidate labels: {sorted(missing)}")

    history: list[EvalResult] = []
    for idx, (label, params) in enumerate(candidates, start=1):
        print(f"[{idx}/{len(candidates)}] Evaluating {label} ...", flush=True)
        result = _evaluate_suite(label, prepared_full, prepared_aug, prepared_apr, params)
        history.append(result)
        print(
            json.dumps(
                {
                    "label": result.label,
                    "full_return_pct": round(result.total_return_pct, 3),
                    "max_drawdown_pct": round(result.max_drawdown_pct, 3),
                    "aug_return_pct": round(result.aug_return_pct, 3),
                    "apr_return_pct": round(result.apr_return_pct, 3),
                    "trades": result.total_trades,
                    "score": round(result.score, 3),
                    "elapsed_sec": round(result.elapsed_sec, 2),
                },
                indent=2,
            ),
            flush=True,
        )

    top = sorted(history, key=lambda r: r.score, reverse=True)
    payload = {
        "symbol": "UVXY",
        "window_start": str(WINDOW_START),
        "window_end": str(WINDOW_END),
        "costs": {"spread_pct": 0.06, "slippage_pct": 0.02, "commission": 0.0},
        "engine_mode": "alpaca_realistic_current",
        "slice_windows": {
            "aug_2024": [str(AUG_START), str(AUG_END)],
            "apr_2025": [str(APR_START), str(APR_END)],
        },
        "candidates": [asdict(r) for r in top],
        "best_1x_current_engine": asdict(top[0]) if top else None,
    }

    (ARTIFACT_DIR / "uvxy_optimizer_current_engine_results.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame([asdict(r) for r in top]).to_csv(
        ARTIFACT_DIR / "uvxy_optimizer_current_engine_top_search.csv",
        index=False,
    )
    print(f"Wrote results to {ARTIFACT_DIR}", flush=True)


if __name__ == "__main__":
    main()
