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
from scripts.validate_earnings_overshoot_first_dump import (  # noqa: E402
    ALPACA_CACHE_DIR,
    RTH_CLOSE,
    RTH_OPEN,
    _compound_return_pct,
    _find_entry_idx,
    _load_labeled_events,
)
from scripts.validate_earnings_overshoot_wave_sequence import (  # noqa: E402
    _exit_long,
    _exit_short,
    _find_long_entry_idx,
)


ARTIFACT_DIR = ROOT / "artifacts" / "optimization"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

_SESSION_FRAME_CACHE: dict[tuple[str, str], pd.DataFrame | None] = {}


@dataclass(frozen=True)
class VariantConfig:
    label: str
    overshoot_prev_close_min: float
    start_minute: int
    max_minute: int
    peak_pullback_min: float
    vwap_break_min: float
    breakdown_impulse_min: float
    short_exit_mode: str
    short_rebound_exit_pct: float
    long_entry_mode: str
    long_momentum_min: float
    long_exit_mode: str
    long_vwap_touch_buffer: float
    long_ema_roll_gain_min: float


@dataclass
class VariantEval:
    label: str
    eligible_events: int
    validated_events: int
    triggered_events: int
    coverage_pct: float
    total_trades: int
    long_trades: int
    short_trades: int
    win_rate_pct: float
    mean_event_return_pct: float
    median_event_return_pct: float
    compounded_event_return_pct: float
    mean_trade_return_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    max_drawdown_pct: float
    params: dict[str, Any]


BEST_RETURN_VARIANT = VariantConfig(
    label="wave_sequence_best_return",
    overshoot_prev_close_min=20.0,
    start_minute=13 * 60,
    max_minute=14 * 60,
    peak_pullback_min=3.0,
    vwap_break_min=1.0,
    breakdown_impulse_min=0.5,
    short_exit_mode="time_10",
    short_rebound_exit_pct=0.5,
    long_entry_mode="ema_turn",
    long_momentum_min=0.0,
    long_exit_mode="ema_roll",
    long_vwap_touch_buffer=-0.5,
    long_ema_roll_gain_min=1.0,
)

ROBUST_VARIANT = VariantConfig(
    label="wave_sequence_robust",
    overshoot_prev_close_min=20.0,
    start_minute=13 * 60,
    max_minute=14 * 60 + 30,
    peak_pullback_min=3.0,
    vwap_break_min=0.5,
    breakdown_impulse_min=0.5,
    short_exit_mode="rebound",
    short_rebound_exit_pct=0.5,
    long_entry_mode="next_bar",
    long_momentum_min=0.0,
    long_exit_mode="time_20",
    long_vwap_touch_buffer=-0.5,
    long_ema_roll_gain_min=0.5,
)


def _session_key(symbol: str, reaction_date: pd.Timestamp) -> tuple[str, str]:
    return str(symbol).upper(), pd.Timestamp(reaction_date).date().isoformat()


def _load_session_frame(symbol: str, reaction_date: pd.Timestamp) -> pd.DataFrame | None:
    key = _session_key(symbol, reaction_date)
    if key in _SESSION_FRAME_CACHE:
        cached = _SESSION_FRAME_CACHE[key]
        return None if cached is None else cached.copy()

    path = ALPACA_CACHE_DIR / str(symbol).upper() / "1Min.csv"
    if not path.exists():
        _SESSION_FRAME_CACHE[key] = None
        return None

    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df["minutes_et"] = df["date"].dt.hour * 60 + df["date"].dt.minute
    day = pd.Timestamp(reaction_date).date()
    session = df[
        (df["date"].dt.date == day)
        & (df["minutes_et"] >= RTH_OPEN)
        & (df["minutes_et"] <= RTH_CLOSE)
    ].copy()
    if session.empty:
        _SESSION_FRAME_CACHE[key] = None
        return None

    session = session.sort_values("date").reset_index(drop=True)
    cols = ["date", "open", "high", "low", "close", "volume"]
    session = session.loc[:, cols].copy()
    _SESSION_FRAME_CACHE[key] = session
    return session.copy()


def _build_session_arrays(session: pd.DataFrame) -> dict[str, np.ndarray]:
    work = session.copy()
    work["minutes_et"] = work["date"].dt.hour * 60 + work["date"].dt.minute
    typical = (work["high"] + work["low"] + work["close"]) / 3.0
    vol = pd.to_numeric(work["volume"], errors="coerce").fillna(0.0)
    cum_vol = vol.cumsum().replace(0.0, np.nan)
    anchored_vwap = (typical * vol).cumsum() / cum_vol

    arrays = {
        "date": work["date"].to_numpy(copy=True),
        "minutes_et": pd.to_numeric(work["minutes_et"], errors="coerce").to_numpy(dtype=np.int32, copy=True),
        "open": pd.to_numeric(work["open"], errors="coerce").to_numpy(dtype=np.float64, copy=True),
        "high": pd.to_numeric(work["high"], errors="coerce").to_numpy(dtype=np.float64, copy=True),
        "low": pd.to_numeric(work["low"], errors="coerce").to_numpy(dtype=np.float64, copy=True),
        "close": pd.to_numeric(work["close"], errors="coerce").to_numpy(dtype=np.float64, copy=True),
        "anchored_vwap": pd.to_numeric(anchored_vwap, errors="coerce").to_numpy(dtype=np.float64, copy=True),
    }
    arrays["close_from_vwap_pct"] = (arrays["close"] / arrays["anchored_vwap"] - 1.0) * 100.0
    arrays["ret_5m_pct"] = pd.Series(arrays["close"]).pct_change(5).to_numpy(dtype=np.float64, copy=True) * 100.0
    arrays["running_peak_high"] = np.maximum.accumulate(arrays["high"])
    ema_fast = pd.Series(arrays["close"]).ewm(span=5, adjust=False).mean()
    ema_slow = pd.Series(arrays["close"]).ewm(span=13, adjust=False).mean()
    arrays["ema_fast"] = ema_fast.to_numpy(dtype=np.float64, copy=True)
    arrays["ema_slow"] = ema_slow.to_numpy(dtype=np.float64, copy=True)
    arrays["ema_fast_slope"] = ema_fast.diff().to_numpy(dtype=np.float64, copy=True)
    return arrays


class EarningsOvershootWaveSequenceStrategy(BaseStrategy):
    strategy_id = "earnings_overshoot_wave_sequence"
    name = "Earnings Overshoot Wave Sequence"
    description = (
        "Research-only positive-earnings overshoot sequence: short the first dump leg, "
        "then optionally reverse long into the rebound."
    )

    def __init__(self, *, event: dict[str, Any], config: VariantConfig) -> None:
        super().__init__(params={})
        self.event = dict(event)
        self.config = config

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Signal:
        return Signal(strategy_id=self.strategy_id, symbol=symbol, action=SignalAction.HOLD)

    @staticmethod
    def _short_stop(arrays: dict[str, np.ndarray], entry_idx: int) -> float:
        entry_px = float(arrays["close"][entry_idx])
        peak = float(arrays["running_peak_high"][entry_idx])
        return max(peak * 1.006, entry_px * 1.015)

    @staticmethod
    def _long_stop(arrays: dict[str, np.ndarray], entry_idx: int, from_idx: int) -> float:
        entry_px = float(arrays["close"][entry_idx])
        local_low = float(np.nanmin(arrays["low"][from_idx:entry_idx + 1]))
        return min(local_low * 0.994, entry_px * 0.985)

    @staticmethod
    def _empty_meta(n: int) -> tuple[list[SignalAction], list[dict[str, Any]]]:
        return [SignalAction.HOLD] * n, [
            {"suggested_tp": None, "suggested_sl": None, "metadata": {}}
            for _ in range(n)
        ]

    def generate_signals_bulk(self, data: pd.DataFrame, symbol: str) -> tuple[list, list]:
        n = len(data)
        actions, meta = self._empty_meta(n)
        if data.empty:
            return actions, meta

        arrays = _build_session_arrays(data)
        cfg = self.config
        short_entry_idx = _find_entry_idx(
            self.event,
            arrays,
            overshoot_prev_close_min=cfg.overshoot_prev_close_min,
            start_minute=cfg.start_minute,
            max_minute=cfg.max_minute,
            peak_pullback_min=cfg.peak_pullback_min,
            vwap_break_min=cfg.vwap_break_min,
            breakdown_impulse_min=cfg.breakdown_impulse_min,
        )
        if short_entry_idx is None:
            return actions, meta

        short_exit_idx, short_exit_reason = _exit_short(
            arrays,
            entry_idx=short_entry_idx,
            exit_mode=cfg.short_exit_mode,
            rebound_exit_pct=cfg.short_rebound_exit_pct,
            vwap_reclaim_buffer=0.0,
            max_hold_bars=20,
        )

        long_entry_idx, long_entry_reason = _find_long_entry_idx(
            arrays,
            start_idx=short_exit_idx,
            entry_mode=cfg.long_entry_mode,
            long_momentum_min=cfg.long_momentum_min,
            max_entry_bars=10,
        )

        long_exit_idx = None
        long_exit_reason = None
        if long_entry_idx is not None:
            long_exit_idx, long_exit_reason = _exit_long(
                arrays,
                entry_idx=long_entry_idx,
                exit_mode=cfg.long_exit_mode,
                vwap_touch_buffer=cfg.long_vwap_touch_buffer,
                ema_roll_gain_min=cfg.long_ema_roll_gain_min,
                max_hold_bars=20,
            )

        actions[short_entry_idx] = SignalAction.SELL
        meta[short_entry_idx] = {
            "suggested_tp": None,
            "suggested_sl": self._short_stop(arrays, short_entry_idx),
            "metadata": {
                "regime": "earnings_wave_short",
                "session_exit": "eod",
                "earnings_wave_sequence": True,
                "event_variant": cfg.label,
                "overshoot_prev_close_min": cfg.overshoot_prev_close_min,
                "verdict_reason": (
                    "Positive off-hours earnings overshoot confirmed a first dump leg; "
                    "enter short on the initial breakdown and manage with the wave sequence."
                ),
            },
        }

        if long_entry_idx is None:
            actions[short_exit_idx] = SignalAction.BUY
            meta[short_exit_idx] = {
                "suggested_tp": None,
                "suggested_sl": None,
                "metadata": {
                    "regime": "earnings_wave_short_cover",
                    "cover_only": True,
                    "short_exit_reason": short_exit_reason,
                },
            }
            return actions, meta

        long_stop = self._long_stop(arrays, long_entry_idx, short_exit_idx)

        if short_exit_idx == long_entry_idx:
            actions[short_exit_idx] = SignalAction.BUY
            meta[short_exit_idx] = {
                "suggested_tp": None,
                "suggested_sl": long_stop,
                "metadata": {
                    "regime": "earnings_wave_long",
                    "session_exit": "eod",
                    "earnings_wave_sequence": True,
                    "event_variant": cfg.label,
                    "short_exit_reason": short_exit_reason,
                    "long_entry_reason": long_entry_reason,
                    "verdict_reason": (
                        "Cover the first dump-leg short and reverse long once the rebound "
                        "setup confirms."
                    ),
                },
            }
        else:
            actions[short_exit_idx] = SignalAction.BUY
            meta[short_exit_idx] = {
                "suggested_tp": None,
                "suggested_sl": None,
                "metadata": {
                    "regime": "earnings_wave_short_cover",
                    "cover_only": True,
                    "short_exit_reason": short_exit_reason,
                },
            }
            actions[long_entry_idx] = SignalAction.BUY
            meta[long_entry_idx] = {
                "suggested_tp": None,
                "suggested_sl": long_stop,
                "metadata": {
                    "regime": "earnings_wave_long",
                    "session_exit": "eod",
                    "earnings_wave_sequence": True,
                    "event_variant": cfg.label,
                    "long_entry_reason": long_entry_reason,
                    "verdict_reason": (
                        "After the first dump leg exhausted, enter the rebound leg and hold "
                        "until the long exit condition or session close."
                    ),
                },
            }

        if long_exit_idx is not None:
            actions[long_exit_idx] = SignalAction.SELL
            meta[long_exit_idx] = {
                "suggested_tp": None,
                "suggested_sl": None,
                "metadata": {
                    "regime": "earnings_wave_long_exit",
                    "cover_only": True,
                    "long_exit_reason": long_exit_reason,
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


def _event_row(
    *,
    event: dict[str, Any],
    result,
    variant: VariantConfig,
) -> dict[str, Any]:
    trades = result.trades
    short_trade = next((t for t in trades if t.direction.value == "Short"), None)
    long_trade = next((t for t in trades if t.direction.value == "Long"), None)
    return {
        "variant": variant.label,
        "symbol": event["symbol"],
        "reaction_date": pd.Timestamp(event["reaction_date"]).date().isoformat(),
        "timing": event.get("timing"),
        "surprise_pct": float(event.get("surprise_pct", np.nan)),
        "peak_vs_prev_close_pct": float(event.get("peak_vs_prev_close_pct", np.nan)),
        "total_return_pct": float(result.total_return_pct),
        "max_drawdown_pct": float(result.max_drawdown_pct),
        "sharpe_ratio": float(result.sharpe_ratio),
        "total_trades": int(result.total_trades),
        "short_return_pct": float(short_trade.leveraged_return_pct) if short_trade and short_trade.leveraged_return_pct is not None else np.nan,
        "long_return_pct": float(long_trade.leveraged_return_pct) if long_trade and long_trade.leveraged_return_pct is not None else np.nan,
        "short_entry_time": pd.Timestamp(short_trade.entry_time).isoformat() if short_trade and short_trade.entry_time is not None else None,
        "short_exit_time": pd.Timestamp(short_trade.exit_time).isoformat() if short_trade and short_trade.exit_time is not None else None,
        "long_entry_time": pd.Timestamp(long_trade.entry_time).isoformat() if long_trade and long_trade.entry_time is not None else None,
        "long_exit_time": pd.Timestamp(long_trade.exit_time).isoformat() if long_trade and long_trade.exit_time is not None else None,
        "short_outcome": str(short_trade.outcome.value) if short_trade and short_trade.outcome is not None else None,
        "long_outcome": str(long_trade.outcome.value) if long_trade and long_trade.outcome is not None else None,
    }


def _trade_rows(*, event: dict[str, Any], result, variant: VariantConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, trade in enumerate(result.trades, start=1):
        rows.append(
            {
                "variant": variant.label,
                "symbol": event["symbol"],
                "reaction_date": pd.Timestamp(event["reaction_date"]).date().isoformat(),
                "trade_num": idx,
                "direction": trade.direction.value,
                "entry_time": pd.Timestamp(trade.entry_time).isoformat() if trade.entry_time is not None else None,
                "exit_time": pd.Timestamp(trade.exit_time).isoformat() if trade.exit_time is not None else None,
                "entry_price": float(trade.entry_price),
                "exit_price": float(trade.exit_price) if trade.exit_price is not None else np.nan,
                "return_pct": float(trade.leveraged_return_pct) if trade.leveraged_return_pct is not None else np.nan,
                "pnl": float(trade.pnl) if trade.pnl is not None else np.nan,
                "outcome": str(trade.outcome.value) if trade.outcome is not None else None,
                "notes": trade.notes,
            }
        )
    return rows


def _evaluate_variant(events: pd.DataFrame, variant: VariantConfig) -> tuple[VariantEval, pd.DataFrame, pd.DataFrame]:
    eligible = events[pd.to_numeric(events["peak_vs_prev_close_pct"], errors="coerce") >= variant.overshoot_prev_close_min].copy()
    event_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []

    for event in eligible.to_dict(orient="records"):
        session = _load_session_frame(str(event["symbol"]).upper(), pd.Timestamp(event["reaction_date"]))
        if session is None or session.empty:
            continue

        strategy = EarningsOvershootWaveSequenceStrategy(event=event, config=variant)
        engine = BacktestEngine(
            strategy,
            risk_manager=_risk_manager(),
            counter_signal_exit=True,
            spread_pct=0.06,
            slippage_pct=0.02,
            commission_per_trade=0.0,
            enforce_rth=True,
            extended_hours=False,
            enforce_pdt=False,
            enforce_ssr=False,
            enforce_fractional=False,
            fill_diagnostic=False,
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

        event_rows.append(_event_row(event=event, result=result, variant=variant))
        trade_rows.extend(_trade_rows(event=event, result=result, variant=variant))

    events_df = pd.DataFrame(event_rows)
    trades_df = pd.DataFrame(trade_rows)

    if events_df.empty:
        return (
            VariantEval(
                label=variant.label,
                eligible_events=int(len(eligible)),
                validated_events=0,
                triggered_events=0,
                coverage_pct=0.0,
                total_trades=0,
                long_trades=0,
                short_trades=0,
                win_rate_pct=np.nan,
                mean_event_return_pct=np.nan,
                median_event_return_pct=np.nan,
                compounded_event_return_pct=0.0,
                mean_trade_return_pct=np.nan,
                avg_win_pct=np.nan,
                avg_loss_pct=np.nan,
                max_drawdown_pct=np.nan,
                params=asdict(variant),
            ),
            events_df,
            trades_df,
        )

    event_returns = pd.to_numeric(events_df["total_return_pct"], errors="coerce").dropna()
    trade_returns = pd.to_numeric(trades_df["return_pct"], errors="coerce").dropna()
    compounded = float(_compound_return_pct(event_returns))
    equity_curve = (1.0 + event_returns / 100.0).cumprod()
    max_drawdown = float(((equity_curve / equity_curve.cummax()) - 1.0).min() * 100.0) if not equity_curve.empty else np.nan
    eval_row = VariantEval(
        label=variant.label,
        eligible_events=int(len(eligible)),
        validated_events=int(len(events_df)),
        triggered_events=int(len(events_df)),
        coverage_pct=float(len(events_df) / len(eligible) * 100.0) if len(eligible) > 0 else 0.0,
        total_trades=int(len(trades_df)),
        long_trades=int((trades_df["direction"] == "Long").sum()) if not trades_df.empty else 0,
        short_trades=int((trades_df["direction"] == "Short").sum()) if not trades_df.empty else 0,
        win_rate_pct=float((event_returns > 0).mean() * 100.0),
        mean_event_return_pct=float(event_returns.mean()),
        median_event_return_pct=float(event_returns.median()),
        compounded_event_return_pct=compounded,
        mean_trade_return_pct=float(trade_returns.mean()) if not trade_returns.empty else np.nan,
        avg_win_pct=float(trade_returns[trade_returns > 0].mean()) if (trade_returns > 0).any() else np.nan,
        avg_loss_pct=float(trade_returns[trade_returns < 0].mean()) if (trade_returns < 0).any() else np.nan,
        max_drawdown_pct=max_drawdown,
        params=asdict(variant),
    )
    return eval_row, events_df, trades_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the earnings overshoot wave sequence inside the normal backtest engine on event days only."
    )
    parser.add_argument("--start", default="2024-04-04", help="Start date for event selection (default: 2024-04-04).")
    parser.add_argument("--end", default="2026-05-01", help="End date for event selection (default: 2026-05-01).")
    parser.add_argument(
        "--artifact-stem",
        default="earnings_wave_sequence_main_engine",
        help="Artifact stem (default: earnings_wave_sequence_main_engine).",
    )
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    events = _load_labeled_events(start, end)

    variants = [BEST_RETURN_VARIANT, ROBUST_VARIANT]
    eval_rows: list[VariantEval] = []
    all_event_rows: list[pd.DataFrame] = []
    all_trade_rows: list[pd.DataFrame] = []

    for variant in variants:
        eval_row, events_df, trades_df = _evaluate_variant(events, variant)
        eval_rows.append(eval_row)
        if not events_df.empty:
            all_event_rows.append(events_df)
        if not trades_df.empty:
            all_trade_rows.append(trades_df)

    ranked = sorted(eval_rows, key=lambda row: (row.compounded_event_return_pct, row.mean_event_return_pct), reverse=True)
    eval_df = pd.DataFrame([asdict(row) for row in ranked])
    events_df = pd.concat(all_event_rows, ignore_index=True) if all_event_rows else pd.DataFrame()
    trades_df = pd.concat(all_trade_rows, ignore_index=True) if all_trade_rows else pd.DataFrame()

    out_json = ARTIFACT_DIR / f"{args.artifact_stem}.json"
    out_variants = ARTIFACT_DIR / f"{args.artifact_stem}_variants.csv"
    out_events = ARTIFACT_DIR / f"{args.artifact_stem}_events.csv"
    out_trades = ARTIFACT_DIR / f"{args.artifact_stem}_trades.csv"

    payload = {
        "window": {"start": start.date().isoformat(), "end": end.date().isoformat()},
        "notes": [
            "This validation runs the earnings overshoot wave sequence inside the normal backtest engine with its standard trade accounting and counter-signal behavior.",
            "To keep the test scoped to the strategy's design intent, it only evaluates the reaction session that follows each off-hours earnings event rather than stitching together the full multi-year timeline.",
            "Because the standalone research branch did not optimize stop-loss architecture yet, the engine variant uses structural emergency stops around the session peak/trough while relying mainly on reversal signals and session-close fallback for exits.",
        ],
        "variants": [asdict(row) for row in ranked],
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    eval_df.to_csv(out_variants, index=False)
    events_df.to_csv(out_events, index=False)
    trades_df.to_csv(out_trades, index=False)

    for row in ranked:
        print(
            f"{row.label}: eligible={row.eligible_events} triggered={row.triggered_events} "
            f"trades={row.total_trades} coverage={row.coverage_pct:.1f}% "
            f"return={row.compounded_event_return_pct:.3f}% "
            f"win={row.win_rate_pct:.1f}% dd={row.max_drawdown_pct:.3f}%"
        )
    print(f"Wrote results to {out_json}")


if __name__ == "__main__":
    main()
