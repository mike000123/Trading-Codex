from __future__ import annotations

import argparse
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

from config.settings import RiskConfig, settings
from data.ingestion import load_from_alpaca_history, prepare_strategy_data
from reporting.backtest import BacktestEngine
from risk.manager import RiskManager
from strategies import get_strategy


ARTIFACT_DIR = ROOT / "artifacts" / "optimization"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

ALPACA_SAFE_BUFFER_DAYS = 3


@dataclass
class EvalResult:
    label: str
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    win_rate_pct: float
    total_trades: int
    avg_win_pct: float
    avg_loss_pct: float
    score: float
    params: dict[str, Any]
    yearly_breakdown: dict[str, dict[str, float | int | None]]


def _score_result(result) -> float:
    trades_bonus = min(float(result.total_trades), 100.0) * 0.03
    sharpe_bonus = float(result.sharpe_ratio) * 8.0
    dd_penalty = abs(float(result.max_drawdown_pct)) * 2.25
    return float(result.total_return_pct) + sharpe_bonus + trades_bonus - dd_penalty


def _load_apld_prices(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if not settings.alpaca.has_paper_credentials():
        raise SystemExit("Alpaca paper credentials are required for the APLD research overlay validation.")
    prices = load_from_alpaca_history(
        "APLD",
        "1Min",
        start,
        end,
        settings.alpaca.paper_api_key,
        settings.alpaca.paper_secret_key,
        paper=True,
        use_cache=True,
    )
    if prices is None or prices.empty:
        raise SystemExit("Could not load APLD Alpaca history for the APLD research overlay validation.")
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices = prices.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date").reset_index(drop=True)
    return prices


def _prepare_apld_data(prices: pd.DataFrame) -> pd.DataFrame:
    cls = get_strategy("bollinger_rsi")
    strategy = cls(params={})
    return prepare_strategy_data(
        prices,
        strategy,
        primary_symbol="APLD",
        source="alpaca",
        interval="1Min",
        start=prices["date"].min(),
        end=prices["date"].max(),
    )


def _yearly_breakdown(trades: list) -> dict[str, dict[str, float | int | None]]:
    closed = [
        t for t in trades
        if t.leveraged_return_pct is not None and t.entry_time is not None
    ]

    def _stats(subset: list) -> dict[str, float | int | None]:
        if not subset:
            return {
                "signals": 0,
                "win_rate_pct": None,
                "mean_return_pct": None,
                "compounded_return_pct": None,
            }
        returns = pd.Series([float(t.leveraged_return_pct or 0.0) for t in subset], dtype=float)
        compounded = float((1.0 + returns / 100.0).prod() - 1.0) * 100.0
        return {
            "signals": int(len(returns)),
            "win_rate_pct": float((returns > 0).mean() * 100.0),
            "mean_return_pct": float(returns.mean()),
            "compounded_return_pct": compounded,
        }

    by_year: dict[str, dict[str, float | int | None]] = {}
    for year in [2024, 2025, 2026]:
        subset = [t for t in closed if pd.Timestamp(t.entry_time).year == year]
        by_year[str(year)] = _stats(subset)
    by_year["first_half"] = _stats([t for t in closed if pd.Timestamp(t.entry_time) < pd.Timestamp("2025-04-01")])
    by_year["second_half"] = _stats([t for t in closed if pd.Timestamp(t.entry_time) >= pd.Timestamp("2025-04-01")])
    return by_year


def _evaluate(prepared_prices: pd.DataFrame, label: str, overrides: dict[str, Any]) -> EvalResult:
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
        "APLD",
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
        avg_win_pct=float(result.avg_win_pct),
        avg_loss_pct=float(result.avg_loss_pct),
        score=_score_result(result),
        params=dict(overrides),
        yearly_breakdown=_yearly_breakdown(result.trades),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the APLD BTC research overlay inside the main Bollinger/RSI backtest engine.")
    parser.add_argument("--start", default="2024-04-01", help="Start date for the validation window (default: 2024-04-01).")
    parser.add_argument("--end", default="", help="Optional end date; defaults to now - safe Alpaca buffer.")
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end) if str(args.end).strip() else pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=ALPACA_SAFE_BUFFER_DAYS)

    prices = _load_apld_prices(start, end)
    prepared = _prepare_apld_data(prices)

    cls = get_strategy("bollinger_rsi")
    strategy = cls(params={})
    generic_params = dict(strategy.default_params())
    overlay_params = dict(strategy.effective_default_params(symbol="APLD"))
    long_only_params = dict(overlay_params)
    long_only_params["apld_btc_short_enabled"] = False
    short_only_params = dict(overlay_params)
    short_only_params["apld_btc_long_enabled"] = False

    generic = _evaluate(prepared, "generic_defaults", generic_params)
    long_only = _evaluate(prepared, "apld_overlay_long_only", long_only_params)
    short_only = _evaluate(prepared, "apld_overlay_short_only", short_only_params)
    combined = _evaluate(prepared, "apld_overlay_combined", overlay_params)

    ranked = sorted([generic, long_only, short_only, combined], key=lambda r: r.score, reverse=True)
    payload = {
        "symbol": "APLD",
        "source": "alpaca",
        "interval": "1Min",
        "window": {
            "start": pd.Timestamp(start).isoformat(),
            "end": pd.Timestamp(end).isoformat(),
        },
        "prepared_rows": int(len(prepared)),
        "variants": [asdict(item) for item in ranked],
        "notes": [
            "This pass validates the gated APLD BTC overlay inside the normal Bollinger/RSI backtest engine rather than in the standalone research scripts.",
            "The generic default acts as a control, while the long-only, short-only, and combined overlay variants show whether the research edge survives once it uses the app's standard entry/exit/trade accounting path.",
            "The APLD preset remains research-oriented even if the combined overlay wins here; the point is to validate compatibility and realism before any live promotion.",
        ],
    }

    out_json = ARTIFACT_DIR / "apld_research_overlay_main_engine.json"
    out_csv = ARTIFACT_DIR / "apld_research_overlay_main_engine_variants.csv"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame([asdict(item) for item in ranked]).to_csv(out_csv, index=False)

    print(f"Prepared rows: {len(prepared)}")
    for item in ranked:
        print(
            f"{item.label}: return={item.total_return_pct:.3f}% "
            f"dd={item.max_drawdown_pct:.3f}% sharpe={item.sharpe_ratio:.4f} "
            f"trades={item.total_trades} win={item.win_rate_pct:.1f}%"
        )
    print(f"Wrote results to {out_json}")


if __name__ == "__main__":
    main()
