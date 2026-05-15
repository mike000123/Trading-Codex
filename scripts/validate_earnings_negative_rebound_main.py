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
from core.models import Signal, SignalAction
from reporting.backtest import BacktestEngine
from risk.manager import RiskManager
from strategies.base import BaseStrategy
from scripts.validate_earnings_negative_rebound_family import _find_long_rebound_entry_idx, _load_negative_events
from scripts.validate_earnings_overshoot_first_dump import _safe_pct
from scripts.validate_earnings_wave_sequence_main import _load_session_frame


ARTIFACT_DIR = ROOT / "artifacts" / "optimization"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


BEST_NEGATIVE_CFG = {
    "downside_prev_close_min": 12.5,
    "start_minute": 12 * 60,
    "max_minute": 15 * 60,
    "rebound_from_trough_min": 4.0,
    "vwap_reclaim_min": -0.5,
    "rebound_impulse_min": 0.5,
    "exit_vwap_touch_buffer": 1.0,
}


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
    params: dict[str, Any]


def _build_session_arrays(session: pd.DataFrame) -> dict[str, np.ndarray]:
    work = session.copy()
    work["minutes_et"] = work["date"].dt.hour * 60 + work["date"].dt.minute
    typical = (work["high"] + work["low"] + work["close"]) / 3.0
    vol = pd.to_numeric(work["volume"], errors="coerce").fillna(0.0)
    cum_vol = vol.cumsum().replace(0.0, np.nan)
    anchored_vwap = (typical * vol).cumsum() / cum_vol
    ema_fast = pd.Series(work["close"]).ewm(span=5, adjust=False).mean()

    arrays = {
        "date": work["date"].to_numpy(copy=True),
        "minutes_et": pd.to_numeric(work["minutes_et"], errors="coerce").to_numpy(dtype=np.int32, copy=True),
        "open": pd.to_numeric(work["open"], errors="coerce").to_numpy(dtype=np.float64, copy=True),
        "high": pd.to_numeric(work["high"], errors="coerce").to_numpy(dtype=np.float64, copy=True),
        "low": pd.to_numeric(work["low"], errors="coerce").to_numpy(dtype=np.float64, copy=True),
        "close": pd.to_numeric(work["close"], errors="coerce").to_numpy(dtype=np.float64, copy=True),
        "anchored_vwap": pd.to_numeric(anchored_vwap, errors="coerce").to_numpy(dtype=np.float64, copy=True),
        "ema_fast": ema_fast.to_numpy(dtype=np.float64, copy=True),
    }
    arrays["close_from_vwap_pct"] = (arrays["close"] / arrays["anchored_vwap"] - 1.0) * 100.0
    arrays["ret_5m_pct"] = pd.Series(arrays["close"]).pct_change(5).to_numpy(dtype=np.float64, copy=True) * 100.0
    arrays["running_trough_low"] = np.minimum.accumulate(arrays["low"])
    return arrays


class EarningsNegativeReboundStrategy(BaseStrategy):
    strategy_id = "earnings_negative_rebound"
    name = "Earnings Negative Rebound (Research)"
    description = "Research-only long rebound after large negative off-hours earnings overshoots."

    def __init__(self, *, event: dict[str, Any], config: dict[str, Any]) -> None:
        super().__init__(params={})
        self.event = dict(event)
        self.config = dict(config)

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Signal:
        return Signal(strategy_id=self.strategy_id, symbol=symbol, action=SignalAction.HOLD)

    @staticmethod
    def _empty_meta(n: int) -> tuple[list[SignalAction], list[dict[str, Any]]]:
        return [SignalAction.HOLD] * n, [
            {"suggested_tp": None, "suggested_sl": None, "metadata": {}}
            for _ in range(n)
        ]

    @staticmethod
    def _long_stop(arrays: dict[str, np.ndarray], entry_idx: int) -> float:
        entry_px = float(arrays["close"][entry_idx])
        local_low = float(np.nanmin(arrays["low"][: entry_idx + 1]))
        return min(local_low * 0.994, entry_px * 0.985)

    def generate_signals_bulk(self, data: pd.DataFrame, symbol: str) -> tuple[list, list]:
        n = len(data)
        actions, meta = self._empty_meta(n)
        if data.empty:
            return actions, meta

        arrays = _build_session_arrays(data)
        cfg = self.config
        entry_idx = _find_long_rebound_entry_idx(
            self.event,
            arrays,
            downside_prev_close_min=float(cfg["downside_prev_close_min"]),
            start_minute=int(cfg["start_minute"]),
            max_minute=int(cfg["max_minute"]),
            rebound_from_trough_min=float(cfg["rebound_from_trough_min"]),
            vwap_reclaim_min=float(cfg["vwap_reclaim_min"]),
            rebound_impulse_min=float(cfg["rebound_impulse_min"]),
        )
        if entry_idx is None:
            return actions, meta

        close_from_vwap = arrays["close_from_vwap_pct"]
        exit_idx = None
        for idx in range(entry_idx + 1, len(close_from_vwap)):
            if np.isfinite(close_from_vwap[idx]) and close_from_vwap[idx] >= float(cfg["exit_vwap_touch_buffer"]):
                exit_idx = int(idx)
                break
        if exit_idx is None:
            exit_idx = len(close_from_vwap) - 1

        actions[entry_idx] = SignalAction.BUY
        meta[entry_idx] = {
            "suggested_tp": None,
            "suggested_sl": self._long_stop(arrays, entry_idx),
            "metadata": {
                "regime": "earnings_negative_rebound_long",
                "session_exit": "eod",
                "earnings_negative_rebound": True,
                "event_reaction_date": str(self.event.get("reaction_date")),
                "verdict_reason": (
                    "Negative off-hours earnings overshot deeply below the prior close, "
                    "then printed a confirmed rebound back through the intraday fair zone."
                ),
            },
        }
        if exit_idx > entry_idx:
            actions[exit_idx] = SignalAction.SELL
            meta[exit_idx] = {
                "suggested_tp": None,
                "suggested_sl": None,
                "metadata": {
                    "regime": "earnings_negative_rebound_exit",
                    "cover_only": True,
                    "earnings_negative_rebound": True,
                    "exit_reason": "vwap_touch",
                },
            }
        return actions, meta


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


def evaluate_negative_rebound(events: pd.DataFrame) -> tuple[ValidationSummary, pd.DataFrame, pd.DataFrame]:
    event_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []

    for event in events.to_dict(orient="records"):
        session = _load_session_frame(str(event["symbol"]).upper(), pd.Timestamp(event["reaction_date"]))
        if session is None or session.empty:
            continue
        strategy = EarningsNegativeReboundStrategy(event=event, config=BEST_NEGATIVE_CFG)
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
        trades = result.trades
        event_rows.append(
            {
                "symbol": str(event["symbol"]).upper(),
                "reaction_date": pd.Timestamp(event["reaction_date"]).date().isoformat(),
                "timing": event.get("timing"),
                "surprise_pct": float(event.get("surprise_pct", np.nan)),
                "trough_vs_prev_close_pct": float(event.get("trough_vs_prev_close_pct", np.nan)),
                "confirm15_close_from_open_pct": float(event.get("confirm15_close_from_open_pct", np.nan)),
                "confirm30_close_from_open_pct": float(event.get("confirm30_close_from_open_pct", np.nan)),
                "total_return_pct": float(result.total_return_pct),
                "max_drawdown_pct": float(result.max_drawdown_pct),
                "sharpe_ratio": float(result.sharpe_ratio),
                "total_trades": int(result.total_trades),
                "trade_returns_pct": "|".join(
                    f"{float(t.leveraged_return_pct):.6f}" if t.leveraged_return_pct is not None else ""
                    for t in trades
                ),
                "trade_notes": " || ".join(t.notes or "" for t in trades),
            }
        )
        for idx, trade in enumerate(trades, start=1):
            trade_rows.append(
                {
                    "symbol": str(event["symbol"]).upper(),
                    "reaction_date": pd.Timestamp(event["reaction_date"]).date().isoformat(),
                    "trade_num": idx,
                    "direction": trade.direction.value,
                    "entry_time": pd.Timestamp(trade.entry_time).isoformat() if trade.entry_time is not None else None,
                    "exit_time": pd.Timestamp(trade.exit_time).isoformat() if trade.exit_time is not None else None,
                    "entry_price": float(trade.entry_price),
                    "exit_price": float(trade.exit_price) if trade.exit_price is not None else np.nan,
                    "return_pct": float(trade.leveraged_return_pct) if trade.leveraged_return_pct is not None else np.nan,
                    "pnl": float(trade.pnl) if trade.pnl is not None else np.nan,
                    "notes": trade.notes,
                }
            )

    events_df = pd.DataFrame(event_rows)
    trades_df = pd.DataFrame(trade_rows)
    if events_df.empty:
        summary = ValidationSummary(
            label="negative_rebound_main_engine",
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
            params=BEST_NEGATIVE_CFG,
        )
        return summary, events_df, trades_df

    event_rets = pd.to_numeric(events_df["total_return_pct"], errors="coerce").dropna()
    trade_rets = pd.to_numeric(trades_df["return_pct"], errors="coerce").dropna()
    equity_curve = (1.0 + event_rets / 100.0).cumprod()
    summary = ValidationSummary(
        label="negative_rebound_main_engine",
        eligible_events=int(len(events)),
        validated_events=int(len(events_df)),
        triggered_events=int(len(events_df)),
        coverage_pct=float(len(events_df) / len(events) * 100.0) if len(events) > 0 else 0.0,
        total_trades=int(len(trades_df)),
        win_rate_pct=float((event_rets > 0).mean() * 100.0),
        mean_event_return_pct=float(event_rets.mean()),
        median_event_return_pct=float(event_rets.median()),
        compounded_event_return_pct=float(_compound_return_pct(event_rets)),
        max_drawdown_pct=float(((equity_curve / equity_curve.cummax()) - 1.0).min() * 100.0),
        avg_trade_return_pct=float(trade_rets.mean()) if not trade_rets.empty else np.nan,
        avg_win_pct=float(trade_rets[trade_rets > 0].mean()) if (trade_rets > 0).any() else np.nan,
        avg_loss_pct=float(trade_rets[trade_rets < 0].mean()) if (trade_rets < 0).any() else np.nan,
        params=BEST_NEGATIVE_CFG,
    )
    return summary, events_df, trades_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the best mirrored negative-earnings rebound branch inside the normal backtest engine."
    )
    parser.add_argument("--start", default="2024-04-04", help="Event-study start date (default: 2024-04-04).")
    parser.add_argument("--end", default="2026-05-01", help="Event-study end date (default: 2026-05-01).")
    parser.add_argument(
        "--artifact-stem",
        default="earnings_negative_rebound_main_engine",
        help="Artifact stem (default: earnings_negative_rebound_main_engine).",
    )
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    all_events = _load_negative_events(start, end)
    eligible = all_events[
        pd.to_numeric(all_events["trough_vs_prev_close_pct"], errors="coerce") <= -BEST_NEGATIVE_CFG["downside_prev_close_min"]
    ].copy()

    summary, events_df, trades_df = evaluate_negative_rebound(eligible)

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
