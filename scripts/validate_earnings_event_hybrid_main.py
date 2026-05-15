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
from strategies.earnings_event_hybrid_strategy import EarningsEventHybridStrategy
from scripts.validate_earnings_negative_rebound_family import _load_negative_events
from scripts.validate_earnings_overshoot_first_dump import _load_labeled_events
from scripts.validate_earnings_wave_sequence_main import _load_session_frame


ARTIFACT_DIR = ROOT / "artifacts" / "optimization"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


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
    family_counts: dict[str, int]
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


def _compound_return_pct(returns_pct: pd.Series) -> float:
    vals = pd.to_numeric(returns_pct, errors="coerce").dropna()
    if vals.empty:
        return 0.0
    return float(((1.0 + vals / 100.0).prod() - 1.0) * 100.0)


def _load_unified_events(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    pos = _load_labeled_events(start, end).copy()
    neg = _load_negative_events(start, end).copy()
    for df, family in ((pos, "positive"), (neg, "negative")):
        if not df.empty:
            df["event_family"] = family
            df["symbol"] = df["symbol"].astype(str).str.upper()
            df["reaction_date"] = pd.to_datetime(df["reaction_date"], errors="coerce")
    merged = pd.concat([pos, neg], ignore_index=True, sort=False)
    if merged.empty:
        return merged
    merged["key"] = merged["symbol"].astype(str).str.upper() + "|" + merged["reaction_date"].dt.date.astype(str)
    merged = merged.drop_duplicates(subset=["key"]).drop(columns=["key"]).reset_index(drop=True)
    return merged


def _event_row(event: dict[str, Any], result) -> dict[str, Any]:
    trades = result.trades
    directions = [t.direction.value for t in trades]
    returns = [float(t.leveraged_return_pct) if t.leveraged_return_pct is not None else np.nan for t in trades]
    notes = [t.notes for t in trades]
    first_meta = trades[0].entry_signal.metadata if trades and getattr(trades[0], "entry_signal", None) is not None else {}
    first_meta = dict(first_meta or {})
    selected_branch = str(first_meta.get("earnings_branch") or "")
    if not selected_branch and notes:
        note = notes[0] or ""
        if "regime=earnings_continuation_long" in note:
            selected_branch = "continuation"
        elif "regime=earnings_failed_reclaim_short" in note:
            selected_branch = "failed_reclaim"
        elif "regime=earnings_short_only_short" in note:
            selected_branch = "short_only"
        elif "regime=earnings_negative_continuation_short" in note:
            selected_branch = "continuation_short"
        elif "regime=earnings_wave_robust_short" in note or "regime=earnings_wave_robust_long" in note:
            selected_branch = "wave_robust"
        elif "regime=earnings_wave_short" in note or "regime=earnings_wave_long" in note:
            selected_branch = "wave_best"
        elif "regime=earnings_negative_rebound_long" in note:
            selected_branch = "negative_rebound"
        elif "regime=earnings_negative_second_short" in note:
            selected_branch = "negative_two_leg"
    return {
        "symbol": str(event["symbol"]).upper(),
        "reaction_date": pd.Timestamp(event["reaction_date"]).date().isoformat(),
        "event_family": str(event.get("event_family") or ""),
        "timing": event.get("timing"),
        "surprise_pct": float(event.get("surprise_pct", np.nan)),
        "peak_vs_prev_close_pct": float(event.get("peak_vs_prev_close_pct", np.nan)),
        "trough_vs_prev_close_pct": float(event.get("trough_vs_prev_close_pct", np.nan)),
        "confirm15_close_from_open_pct": float(event.get("confirm15_close_from_open_pct", np.nan)),
        "confirm30_close_from_open_pct": float(event.get("confirm30_close_from_open_pct", np.nan)),
        "selected_family": str(first_meta.get("earnings_event_hybrid_family") or event.get("event_family") or ""),
        "selected_branch": selected_branch,
        "total_return_pct": float(result.total_return_pct),
        "max_drawdown_pct": float(result.max_drawdown_pct),
        "sharpe_ratio": float(result.sharpe_ratio),
        "total_trades": int(result.total_trades),
        "directions": "|".join(directions),
        "trade_returns_pct": "|".join("" if not np.isfinite(v) else f"{v:.6f}" for v in returns),
        "trade_notes": " || ".join(n or "" for n in notes),
    }


def _trade_rows(event: dict[str, Any], result) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, trade in enumerate(result.trades, start=1):
        entry_meta = dict((trade.entry_signal.metadata or {}) if getattr(trade, "entry_signal", None) is not None else {})
        selected_branch = str(entry_meta.get("earnings_branch") or "")
        note = trade.notes or ""
        if not selected_branch:
            if "regime=earnings_continuation_long" in note:
                selected_branch = "continuation"
            elif "regime=earnings_failed_reclaim_short" in note:
                selected_branch = "failed_reclaim"
            elif "regime=earnings_short_only_short" in note:
                selected_branch = "short_only"
            elif "regime=earnings_negative_continuation_short" in note:
                selected_branch = "continuation_short"
            elif "regime=earnings_wave_robust_short" in note or "regime=earnings_wave_robust_long" in note:
                selected_branch = "wave_robust"
            elif "regime=earnings_wave_short" in note or "regime=earnings_wave_long" in note:
                selected_branch = "wave_best"
            elif "regime=earnings_negative_rebound_long" in note:
                selected_branch = "negative_rebound"
            elif "regime=earnings_negative_second_short" in note:
                selected_branch = "negative_two_leg"
        rows.append(
            {
                "symbol": str(event["symbol"]).upper(),
                "reaction_date": pd.Timestamp(event["reaction_date"]).date().isoformat(),
                "event_family": str(event.get("event_family") or ""),
                "selected_family": str(entry_meta.get("earnings_event_hybrid_family") or event.get("event_family") or ""),
                "selected_branch": selected_branch,
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


def evaluate_unified_strategy(
    events: pd.DataFrame,
    *,
    label: str = "earnings_event_hybrid_main",
    params: dict[str, Any] | None = None,
) -> tuple[ValidationSummary, pd.DataFrame, pd.DataFrame]:
    params = dict(params or {})
    strategy = EarningsEventHybridStrategy(params=params)
    event_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []

    for event in events.to_dict(orient="records"):
        session = _load_session_frame(str(event["symbol"]).upper(), pd.Timestamp(event["reaction_date"]))
        if session is None or session.empty:
            continue
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
        event_rows.append(_event_row(event, result))
        trade_rows.extend(_trade_rows(event, result))

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
            family_counts={},
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

    family_counts = (
        events_df["selected_family"].fillna("").replace("", "unknown").value_counts().sort_index().to_dict()
        if "selected_family" in events_df.columns
        else {}
    )
    branch_counts = (
        events_df["selected_branch"].fillna("").replace("", "unknown").value_counts().sort_index().to_dict()
        if "selected_branch" in events_df.columns
        else {}
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
        avg_win_pct=float(trade_returns[trade_returns > 0].mean()) if not trade_returns[trade_returns > 0].empty else np.nan,
        avg_loss_pct=float(trade_returns[trade_returns <= 0].mean()) if not trade_returns[trade_returns <= 0].empty else np.nan,
        family_counts=family_counts,
        branch_counts=branch_counts,
        params=params,
    )
    return summary, events_df, trades_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the unified Earnings Event Hybrid (Research) strategy on event days only."
    )
    parser.add_argument("--start", default="2024-04-04", help="Event-study start date (default: 2024-04-04).")
    parser.add_argument("--end", default="2026-05-01", help="Event-study end date (default: 2026-05-01).")
    parser.add_argument(
        "--artifact-stem",
        default="earnings_event_hybrid_main_engine",
        help="Artifact stem (default: earnings_event_hybrid_main_engine).",
    )
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    events = _load_unified_events(start, end)
    summary, events_df, trades_df = evaluate_unified_strategy(events)

    out_json = ARTIFACT_DIR / f"{args.artifact_stem}.json"
    out_events = ARTIFACT_DIR / f"{args.artifact_stem}_events.csv"
    out_trades = ARTIFACT_DIR / f"{args.artifact_stem}_trades.csv"

    payload = {
        "window": {"start": start.date().isoformat(), "end": end.date().isoformat()},
        "summary": asdict(summary),
        "top_events": (
            events_df.sort_values(["total_return_pct", "symbol"], ascending=[False, True]).head(25).to_dict(orient="records")
            if not events_df.empty
            else []
        ),
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    events_df.to_csv(out_events, index=False)
    trades_df.to_csv(out_trades, index=False)

    print(
        f"{summary.label}: return={summary.compounded_event_return_pct:.3f}% "
        f"events={summary.triggered_events} trades={summary.total_trades}"
    )
    print("Wrote artifacts:")
    print(f"- {out_json}")
    print(f"- {out_events}")
    print(f"- {out_trades}")


if __name__ == "__main__":
    main()
