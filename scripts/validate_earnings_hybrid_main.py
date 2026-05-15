from __future__ import annotations

import argparse
import json
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
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
from reporting.backtest import BacktestEngine
from risk.manager import RiskManager
from strategies.earnings_overshoot_hybrid_strategy import EarningsOvershootHybridStrategy
from scripts.validate_earnings_overshoot_first_dump import _load_labeled_events
from scripts.validate_earnings_wave_sequence_main import _load_session_frame


ARTIFACT_DIR = ROOT / "artifacts" / "optimization"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def _compound_return_pct(returns_pct: pd.Series) -> float:
    vals = pd.to_numeric(returns_pct, errors="coerce").dropna()
    if vals.empty:
        return 0.0
    return float(((1.0 + vals / 100.0).prod() - 1.0) * 100.0)


@dataclass
class ValidationSummary:
    label: str
    eligible_events: int
    validated_events: int
    triggered_events: int
    coverage_pct: float
    total_trades: int
    win_rate_pct: float
    mean_event_return_pct: float
    median_event_return_pct: float
    compounded_event_return_pct: float
    max_drawdown_pct: float
    avg_trade_return_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    branch_counts: dict[str, int]
    params: dict[str, Any]


def _risk_manager() -> RiskManager:
    return RiskManager(
        RiskConfig(
            max_capital_per_trade_pct=100.0,
            max_daily_loss_pct=100.0,
            max_open_positions=999,
            default_max_loss_pct_of_capital=50.0,
        )
    )


def _selector_branch(event: dict[str, Any], params: dict[str, Any]) -> str:
    if EarningsOvershootHybridStrategy._continuation_selector_match(event, params):
        return "continuation"

    peak_threshold = float(params.get("selector_peak_vs_prev_close_threshold", 21.0))
    weak_confirm15_max = float(params.get("selector_weak_confirm15_max", -0.5))
    weak_confirm30_max = float(params.get("selector_weak_confirm30_max", -1.0))
    strong_confirm15_min = float(params.get("selector_strong_confirm15_min", 1.0))
    strong_confirm30_min = float(params.get("selector_strong_confirm30_min", 1.0))

    peak = float(event.get("peak_vs_prev_close_pct", np.nan))
    confirm15 = float(event.get("confirm15_close_from_open_pct", np.nan))
    confirm30 = float(event.get("confirm30_close_from_open_pct", np.nan))

    if np.isfinite(peak) and peak >= peak_threshold:
        weak_open = (
            np.isfinite(confirm15)
            and np.isfinite(confirm30)
            and confirm15 <= weak_confirm15_max
            and confirm30 <= weak_confirm30_max
        )
        strong_open = (
            np.isfinite(confirm15)
            and np.isfinite(confirm30)
            and confirm15 >= strong_confirm15_min
            and confirm30 >= strong_confirm30_min
        )
        if weak_open:
            return "short_only"
        if strong_open:
            return "wave_robust"
        return "wave_best"
    return "failed_reclaim"


def _event_row(event: dict[str, Any], branch: str, result) -> dict[str, Any]:
    trades = result.trades
    directions = [t.direction.value for t in trades]
    returns = [float(t.leveraged_return_pct) if t.leveraged_return_pct is not None else np.nan for t in trades]
    notes = [t.notes for t in trades]
    return {
        "symbol": str(event["symbol"]).upper(),
        "reaction_date": pd.Timestamp(event["reaction_date"]).date().isoformat(),
        "timing": event.get("timing"),
        "surprise_pct": float(event.get("surprise_pct", np.nan)),
        "peak_vs_prev_close_pct": float(event.get("peak_vs_prev_close_pct", np.nan)),
        "confirm15_close_from_open_pct": float(event.get("confirm15_close_from_open_pct", np.nan)),
        "confirm30_close_from_open_pct": float(event.get("confirm30_close_from_open_pct", np.nan)),
        "selected_branch": branch,
        "total_return_pct": float(result.total_return_pct),
        "max_drawdown_pct": float(result.max_drawdown_pct),
        "sharpe_ratio": float(result.sharpe_ratio),
        "total_trades": int(result.total_trades),
        "directions": "|".join(directions),
        "trade_returns_pct": "|".join("" if not np.isfinite(v) else f"{v:.6f}" for v in returns),
        "trade_notes": " || ".join(n or "" for n in notes),
    }


def _trade_rows(event: dict[str, Any], branch: str, result) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, trade in enumerate(result.trades, start=1):
        rows.append(
            {
                "symbol": str(event["symbol"]).upper(),
                "reaction_date": pd.Timestamp(event["reaction_date"]).date().isoformat(),
                "selected_branch": branch,
                "trade_num": idx,
                "direction": trade.direction.value,
                "entry_time": pd.Timestamp(trade.entry_time).isoformat() if trade.entry_time is not None else None,
                "exit_time": pd.Timestamp(trade.exit_time).isoformat() if trade.exit_time is not None else None,
                "entry_price": float(trade.entry_price),
                "exit_price": float(trade.exit_price) if trade.exit_price is not None else np.nan,
                "return_pct": float(trade.leveraged_return_pct) if trade.leveraged_return_pct is not None else np.nan,
                "pnl": float(trade.pnl) if trade.pnl is not None else np.nan,
                "capital_allocated": float(trade.capital_allocated),
                "outcome": str(trade.outcome.value) if trade.outcome is not None else None,
                "notes": trade.notes,
            }
        )
    return rows


def evaluate_hybrid_strategy(
    events: pd.DataFrame,
    *,
    label: str = "integrated_current",
    params: dict[str, Any] | None = None,
) -> tuple[ValidationSummary, pd.DataFrame, pd.DataFrame]:
    params = dict(params or {})
    strategy = EarningsOvershootHybridStrategy(params=params)
    event_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []

    for event in events.to_dict(orient="records"):
        session = _load_session_frame(str(event["symbol"]).upper(), pd.Timestamp(event["reaction_date"]))
        if session is None or session.empty:
            continue
        branch = _selector_branch(event, strategy.resolve_params(symbol=str(event["symbol"]).upper()))
        engine = BacktestEngine(
            strategy,
            risk_manager=_risk_manager(),
            counter_signal_exit=True,
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
        result = engine.run(
            session,
            str(event["symbol"]).upper(),
            leverage=1.0,
            capital_per_trade=1000.0,
            starting_equity=1000.0,
        )
        if result.total_trades <= 0:
            continue
        event_rows.append(_event_row(event, branch, result))
        trade_rows.extend(_trade_rows(event, branch, result))

    events_df = pd.DataFrame(event_rows)
    trades_df = pd.DataFrame(trade_rows)

    if events_df.empty:
        summary = ValidationSummary(
            label=label,
            eligible_events=int(len(events)),
            validated_events=0,
            triggered_events=0,
            coverage_pct=0.0,
            total_trades=0,
            win_rate_pct=np.nan,
            mean_event_return_pct=np.nan,
            median_event_return_pct=np.nan,
            compounded_event_return_pct=0.0,
            max_drawdown_pct=np.nan,
            avg_trade_return_pct=np.nan,
            avg_win_pct=np.nan,
            avg_loss_pct=np.nan,
            branch_counts={},
            params=params,
        )
        return summary, events_df, trades_df

    event_returns = pd.to_numeric(events_df["total_return_pct"], errors="coerce").dropna()
    trade_returns = pd.to_numeric(trades_df["return_pct"], errors="coerce").dropna()
    equity_curve = (1.0 + event_returns / 100.0).cumprod()
    max_drawdown_pct = (
        float(((equity_curve / equity_curve.cummax()) - 1.0).min() * 100.0)
        if not equity_curve.empty
        else np.nan
    )
    summary = ValidationSummary(
        label=label,
        eligible_events=int(len(events)),
        validated_events=int(len(events_df)),
        triggered_events=int(len(events_df)),
        coverage_pct=float(len(events_df) / len(events) * 100.0) if len(events) > 0 else 0.0,
        total_trades=int(len(trades_df)),
        win_rate_pct=float((event_returns > 0).mean() * 100.0),
        mean_event_return_pct=float(event_returns.mean()),
        median_event_return_pct=float(event_returns.median()),
        compounded_event_return_pct=float(_compound_return_pct(event_returns)),
        max_drawdown_pct=max_drawdown_pct,
        avg_trade_return_pct=float(trade_returns.mean()) if not trade_returns.empty else np.nan,
        avg_win_pct=float(trade_returns[trade_returns > 0].mean()) if (trade_returns > 0).any() else np.nan,
        avg_loss_pct=float(trade_returns[trade_returns < 0].mean()) if (trade_returns < 0).any() else np.nan,
        branch_counts={
            str(k): int(v)
            for k, v in events_df["selected_branch"].value_counts(dropna=False).to_dict().items()
        },
        params=params,
    )
    return summary, events_df, trades_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the integrated earnings overshoot hybrid strategy across eligible event-days."
    )
    parser.add_argument("--start", default="2024-04-04", help="Start date for event selection (default: 2024-04-04).")
    parser.add_argument("--end", default="2026-05-01", help="End date for event selection (default: 2026-05-01).")
    parser.add_argument(
        "--artifact-stem",
        default="earnings_hybrid_main_engine",
        help="Artifact stem (default: earnings_hybrid_main_engine).",
    )
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    events = _load_labeled_events(start, end)
    summary, events_df, trades_df = evaluate_hybrid_strategy(events, label="integrated_current", params={})

    out_json = ARTIFACT_DIR / f"{args.artifact_stem}.json"
    out_events = ARTIFACT_DIR / f"{args.artifact_stem}_events.csv"
    out_trades = ARTIFACT_DIR / f"{args.artifact_stem}_trades.csv"

    payload = {
        "window": {"start": start.date().isoformat(), "end": end.date().isoformat()},
        "summary": asdict(summary),
        "top_events": events_df.sort_values("total_return_pct", ascending=False).head(20).to_dict(orient="records")
        if not events_df.empty
        else [],
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    events_df.to_csv(out_events, index=False)
    trades_df.to_csv(out_trades, index=False)

    print(
        f"{summary.label}: return={summary.compounded_event_return_pct:.3f}% "
        f"events={summary.triggered_events} trades={summary.total_trades}"
    )
    print(f"Wrote artifacts:\n- {out_json}\n- {out_events}\n- {out_trades}")


if __name__ == "__main__":
    main()
