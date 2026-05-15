from __future__ import annotations

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
from strategies.earnings_negative_hybrid_strategy import (
    EarningsNegativeHybridStrategy,
    RTH_CLOSE,
    RTH_OPEN,
    _build_session_arrays,
    _exit_long,
    _exit_short,
    _find_short_after_rebound_idx,
    _load_events_for_symbol,
)
from scripts.validate_earnings_negative_rebound_family import _load_negative_events
from scripts.validate_earnings_wave_sequence_main import _load_session_frame


ARTIFACT_DIR = ROOT / "artifacts" / "optimization"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def _safe_pct(current: float, base: float) -> float:
    try:
        cur = float(current)
        ref = float(base)
    except Exception:
        return float("nan")
    if not np.isfinite(cur) or not np.isfinite(ref) or ref == 0.0:
        return float("nan")
    return (cur / ref - 1.0) * 100.0


def _compound_return_pct(returns_pct: pd.Series) -> float:
    vals = pd.to_numeric(returns_pct, errors="coerce").dropna()
    if vals.empty:
        return 0.0
    return float(((1.0 + vals / 100.0).prod() - 1.0) * 100.0)


def _risk_manager() -> RiskManager:
    return RiskManager(
        RiskConfig(
            max_capital_per_trade_pct=100.0,
            max_daily_loss_pct=100.0,
            max_open_positions=999,
            default_max_loss_pct_of_capital=50.0,
        )
    )


def _max_drawdown_pct(returns_pct: pd.Series) -> float:
    vals = pd.to_numeric(returns_pct, errors="coerce").dropna()
    if vals.empty:
        return 0.0
    equity = (1.0 + vals / 100.0).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min() * 100.0)


def _find_long_rebound_entry_idx_enhanced(
    event: dict[str, Any],
    arrays: dict[str, np.ndarray],
    *,
    downside_prev_close_min: float,
    start_minute: int,
    max_minute: int,
    rebound_from_trough_min: float,
    vwap_reclaim_min: float,
    rebound_impulse_min: float,
    entry_min_close_from_open_pct: float | None,
    entry_require_close_above_ema_fast: bool,
    entry_require_ema_stack: bool,
    entry_require_ema_slope_pos: bool,
) -> int | None:
    prev_close = float(event.get("prev_close", np.nan))
    if not np.isfinite(prev_close) or prev_close <= 0.0:
        return None

    minutes = arrays["minutes_et"]
    close = arrays["close"]
    low = arrays["low"]
    open_px = arrays["open"]
    close_from_vwap = arrays["close_from_vwap_pct"]
    ret_5m = arrays["ret_5m_pct"]
    ema_fast = arrays["ema_fast"]
    ema_slow = arrays["ema_slow"]
    ema_fast_slope = arrays["ema_fast_slope"]

    running_trough = np.minimum.accumulate(low)
    running_trough_vs_prev = (running_trough / prev_close - 1.0) * 100.0
    rebound_from_trough = (close / running_trough - 1.0) * 100.0
    close_from_open_pct = (close / open_px - 1.0) * 100.0

    cond = (
        (minutes >= start_minute)
        & (minutes <= max_minute)
        & np.isfinite(close)
        & np.isfinite(close_from_vwap)
        & np.isfinite(ret_5m)
        & np.isfinite(running_trough_vs_prev)
        & np.isfinite(rebound_from_trough)
        & np.isfinite(close_from_open_pct)
        & (running_trough_vs_prev <= -downside_prev_close_min)
        & (rebound_from_trough >= rebound_from_trough_min)
        & (close_from_vwap >= vwap_reclaim_min)
        & (ret_5m >= rebound_impulse_min)
    )

    if entry_min_close_from_open_pct is not None:
        cond &= close_from_open_pct >= float(entry_min_close_from_open_pct)
    if entry_require_close_above_ema_fast:
        cond &= np.isfinite(ema_fast) & (close >= ema_fast)
    if entry_require_ema_stack:
        cond &= np.isfinite(ema_fast) & np.isfinite(ema_slow) & (ema_fast >= ema_slow)
    if entry_require_ema_slope_pos:
        cond &= np.isfinite(ema_fast_slope) & (ema_fast_slope > 0)

    hits = np.flatnonzero(cond)
    if hits.size == 0:
        return None
    return int(hits[0])


class EntryQualityNegativeHybridStrategy(BaseStrategy):
    strategy_id = "earnings_negative_hybrid_entry_quality"
    name = "Earnings Negative Hybrid Entry Quality"
    description = "Research-only validator for negative earnings rebound entry quality."

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params=params or {})

    def default_params(self) -> dict[str, Any]:
        base = EarningsNegativeHybridStrategy().default_params()
        base.update(
            {
                "entry_min_close_from_open_pct": None,
                "entry_require_close_above_ema_fast": False,
                "entry_require_ema_stack": False,
                "entry_require_ema_slope_pos": False,
            }
        )
        return base

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

    @staticmethod
    def _short_stop(arrays: dict[str, np.ndarray], entry_idx: int, long_start_idx: int) -> float:
        entry_px = float(arrays["close"][entry_idx])
        rebound_peak = float(np.nanmax(arrays["high"][long_start_idx: entry_idx + 1]))
        return max(rebound_peak * 1.006, entry_px * 1.015)

    def generate_signals_bulk(self, data: pd.DataFrame, symbol: str) -> tuple[list, list]:
        n = len(data)
        actions, meta = self._empty_meta(n)
        if data.empty or "date" not in data.columns:
            return actions, meta

        params = self.resolve_params(symbol=symbol)
        symbol_events = _load_events_for_symbol(symbol)
        if symbol_events.empty:
            return actions, meta

        work = data.copy().reset_index().rename(columns={"index": "orig_index"})
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
        work = work.dropna(subset=["date"]).copy()
        if work.empty:
            return actions, meta

        work["session_date"] = work["date"].dt.date.astype(str)
        work["minutes_raw"] = work["date"].dt.hour * 60 + work["date"].dt.minute
        available_dates = set(work["session_date"].astype(str).unique().tolist())
        event_rows = symbol_events[symbol_events["reaction_date"].isin(available_dates)].copy()
        if event_rows.empty:
            return actions, meta

        for event in event_rows.to_dict(orient="records"):
            reaction_date = str(event.get("reaction_date"))
            session = work[
                (work["session_date"] == reaction_date)
                & (work["minutes_raw"] >= RTH_OPEN)
                & (work["minutes_raw"] <= RTH_CLOSE)
            ].copy()
            if session.empty:
                continue
            session = session.sort_values("date").reset_index(drop=True)
            arrays = _build_session_arrays(session)

            long_entry_idx = _find_long_rebound_entry_idx_enhanced(
                event,
                arrays,
                downside_prev_close_min=float(params.get("downside_prev_close_min", 12.5)),
                start_minute=int(params.get("start_minute", 12 * 60)),
                max_minute=int(params.get("max_minute", 15 * 60)),
                rebound_from_trough_min=float(params.get("rebound_from_trough_min", 4.0)),
                vwap_reclaim_min=float(params.get("vwap_reclaim_min", -0.5)),
                rebound_impulse_min=float(params.get("rebound_impulse_min", 0.5)),
                entry_min_close_from_open_pct=params.get("entry_min_close_from_open_pct"),
                entry_require_close_above_ema_fast=bool(params.get("entry_require_close_above_ema_fast", False)),
                entry_require_ema_stack=bool(params.get("entry_require_ema_stack", False)),
                entry_require_ema_slope_pos=bool(params.get("entry_require_ema_slope_pos", False)),
            )
            if long_entry_idx is None:
                continue

            trough = float(event.get("trough_vs_prev_close_pct", np.nan))
            confirm15 = float(event.get("confirm15_close_from_open_pct", np.nan))
            confirm30 = float(event.get("confirm30_close_from_open_pct", np.nan))
            use_two_leg = (
                np.isfinite(trough)
                and np.isfinite(confirm15)
                and np.isfinite(confirm30)
                and trough <= -float(params.get("selector_two_leg_trough_threshold", 20.0))
                and confirm15 >= float(params.get("selector_two_leg_confirm15_min", 0.0))
                and confirm30 >= float(params.get("selector_two_leg_confirm30_min", 0.0))
            )

            g_long_entry = int(arrays["orig_index"][long_entry_idx])
            actions[g_long_entry] = SignalAction.BUY
            meta[g_long_entry] = {
                "suggested_tp": None,
                "suggested_sl": self._long_stop(arrays, long_entry_idx),
                "metadata": {
                    "regime": "earnings_negative_rebound_long",
                    "session_exit": "eod",
                    "earnings_negative_hybrid": True,
                    "earnings_branch": "rebound_long",
                    "event_reaction_date": reaction_date,
                },
            }

            if not use_two_leg:
                close_from_vwap = arrays["close_from_vwap_pct"]
                long_exit_idx = None
                for idx in range(long_entry_idx + 1, len(close_from_vwap)):
                    if np.isfinite(close_from_vwap[idx]) and close_from_vwap[idx] >= float(
                        params.get("exit_vwap_touch_buffer", 1.0)
                    ):
                        long_exit_idx = int(idx)
                        break
                if long_exit_idx is None:
                    long_exit_idx = len(close_from_vwap) - 1
                g_long_exit = int(arrays["orig_index"][long_exit_idx])
                if g_long_exit > g_long_entry:
                    actions[g_long_exit] = SignalAction.SELL
                    meta[g_long_exit] = {
                        "suggested_tp": None,
                        "suggested_sl": None,
                        "metadata": {
                            "regime": "earnings_negative_rebound_exit",
                            "cover_only": True,
                            "earnings_negative_hybrid": True,
                            "earnings_branch": "rebound_long",
                            "exit_reason": "vwap_touch",
                        },
                    }
                continue

            long_exit_idx, long_exit_reason = _exit_long(
                arrays,
                entry_idx=long_entry_idx,
                exit_mode=str(params.get("two_leg_long_exit_mode", "vwap_touch")),
                vwap_touch_buffer=float(params.get("two_leg_long_vwap_touch_buffer", 1.0)),
                ema_roll_gain_min=float(params.get("two_leg_long_ema_roll_gain_min", 1.0)),
                max_hold_bars=int(params.get("two_leg_long_max_hold_bars", 20)),
            )
            g_long_exit = int(arrays["orig_index"][long_exit_idx])

            short_entry_idx, short_entry_reason = _find_short_after_rebound_idx(
                arrays,
                start_idx=long_exit_idx,
                max_entry_bars=int(params.get("two_leg_short_entry_window_bars", 8)),
                pullback_from_peak_min=float(params.get("two_leg_short_pullback_from_peak_min", 1.0)),
                vwap_break_max=float(params.get("two_leg_short_vwap_break_max", -0.5)),
                downside_impulse_min=float(params.get("two_leg_short_downside_impulse_min", 1.0)),
            )

            if short_entry_idx is None:
                if g_long_exit > g_long_entry:
                    actions[g_long_exit] = SignalAction.SELL
                    meta[g_long_exit] = {
                        "suggested_tp": None,
                        "suggested_sl": None,
                        "metadata": {
                            "regime": "earnings_negative_rebound_exit",
                            "cover_only": True,
                            "earnings_negative_hybrid": True,
                            "earnings_branch": "deep_rebound_long",
                            "exit_reason": long_exit_reason,
                        },
                    }
                continue

            short_exit_idx, short_exit_reason = _exit_short(
                arrays,
                entry_idx=short_entry_idx,
                exit_mode=str(params.get("two_leg_short_exit_mode", "time_10")),
                rebound_exit_pct=float(params.get("two_leg_short_rebound_exit_pct", 0.5)),
                vwap_reclaim_buffer=float(params.get("two_leg_short_vwap_reclaim_buffer", 0.0)),
                max_hold_bars=int(params.get("two_leg_short_max_hold_bars", 15)),
            )
            g_short_entry = int(arrays["orig_index"][short_entry_idx])
            g_short_exit = int(arrays["orig_index"][short_exit_idx])

            if g_long_exit == g_short_entry:
                actions[g_long_exit] = SignalAction.SELL
                meta[g_long_exit] = {
                    "suggested_tp": None,
                    "suggested_sl": self._short_stop(arrays, short_entry_idx, long_entry_idx),
                    "metadata": {
                        "regime": "earnings_negative_short_after_rebound",
                        "earnings_negative_hybrid": True,
                        "earnings_branch": "rebound_then_fail",
                        "long_exit_reason": long_exit_reason,
                        "short_entry_reason": short_entry_reason,
                    },
                }
            else:
                actions[g_long_exit] = SignalAction.SELL
                meta[g_long_exit] = {
                    "suggested_tp": None,
                    "suggested_sl": None,
                    "metadata": {
                        "regime": "earnings_negative_rebound_exit",
                        "cover_only": True,
                        "earnings_negative_hybrid": True,
                        "earnings_branch": "deep_rebound_long",
                        "exit_reason": long_exit_reason,
                    },
                }
                actions[g_short_entry] = SignalAction.SELL
                meta[g_short_entry] = {
                    "suggested_tp": None,
                    "suggested_sl": self._short_stop(arrays, short_entry_idx, long_entry_idx),
                    "metadata": {
                        "regime": "earnings_negative_short_after_rebound",
                        "earnings_negative_hybrid": True,
                        "earnings_branch": "rebound_then_fail",
                        "short_entry_reason": short_entry_reason,
                    },
                }

            if g_short_exit > g_short_entry:
                actions[g_short_exit] = SignalAction.BUY
                meta[g_short_exit] = {
                    "suggested_tp": None,
                    "suggested_sl": None,
                    "metadata": {
                        "regime": "earnings_negative_short_cover",
                        "cover_only": True,
                        "earnings_negative_hybrid": True,
                        "exit_reason": short_exit_reason,
                    },
                }

        return actions, meta


@dataclass
class VariantEval:
    label: str
    events: int
    trades: int
    win_rate_pct: float
    mean_return_pct: float
    median_return_pct: float
    compounded_return_pct: float
    max_drawdown_pct: float
    params: dict[str, Any]


def _evaluate_variant(events_df: pd.DataFrame, params: dict[str, Any]) -> tuple[VariantEval, pd.DataFrame]:
    open_gate = params.get("entry_min_close_from_open_pct")
    close_above_ema = bool(params.get("entry_require_close_above_ema_fast", False))
    ema_stack = bool(params.get("entry_require_ema_stack", False))
    ema_slope = bool(params.get("entry_require_ema_slope_pos", False))

    rows: list[dict[str, Any]] = []
    for event in events_df.to_dict(orient="records"):
        symbol = str(event["symbol"]).upper()
        reaction_ts = pd.Timestamp(event["reaction_date"])
        session_frame = _load_session_frame(symbol, reaction_ts)
        if session_frame is None or session_frame.empty:
            continue

        strat = EntryQualityNegativeHybridStrategy(params=params)
        engine = BacktestEngine(
            strat,
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
            session_frame,
            symbol,
            starting_equity=1000.0,
            capital_per_trade=1000.0,
            leverage=1.0,
        )
        if not result.trades:
            continue
        rows.append(
            {
                "symbol": symbol,
                "reaction_date": reaction_ts.date().isoformat(),
                "timing": event.get("timing"),
                "surprise_pct": float(event.get("surprise_pct", np.nan)),
                "trough_vs_prev_close_pct": float(event.get("trough_vs_prev_close_pct", np.nan)),
                "confirm15_close_from_open_pct": float(event.get("confirm15_close_from_open_pct", np.nan)),
                "confirm30_close_from_open_pct": float(event.get("confirm30_close_from_open_pct", np.nan)),
                "return_pct": float(result.total_return_pct),
                "trades": int(len(result.trades)),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        label = (
            f"rq{params['rebound_from_trough_min']}_vwap{params['vwap_reclaim_min']}_"
            f"imp{params['rebound_impulse_min']}_open{open_gate}_"
            f"cef{int(close_above_ema)}_"
            f"est{int(ema_stack)}_"
            f"esp{int(ema_slope)}"
        )
        return (
            VariantEval(
                label=label,
                events=0,
                trades=0,
                win_rate_pct=0.0,
                mean_return_pct=0.0,
                median_return_pct=0.0,
                compounded_return_pct=0.0,
                max_drawdown_pct=0.0,
                params=dict(params),
            ),
            out,
        )

    label = (
        f"rq{params['rebound_from_trough_min']}_vwap{params['vwap_reclaim_min']}_"
        f"imp{params['rebound_impulse_min']}_open{open_gate}_"
        f"cef{int(close_above_ema)}_"
        f"est{int(ema_stack)}_"
        f"esp{int(ema_slope)}"
    )
    eval_row = VariantEval(
        label=label,
        events=int(len(out)),
        trades=int(out["trades"].sum()),
        win_rate_pct=float((out["return_pct"] > 0).mean() * 100.0),
        mean_return_pct=float(out["return_pct"].mean()),
        median_return_pct=float(out["return_pct"].median()),
        compounded_return_pct=_compound_return_pct(out["return_pct"]),
        max_drawdown_pct=_max_drawdown_pct(out["return_pct"]),
        params=dict(params),
    )
    return eval_row, out


def main() -> None:
    start = pd.Timestamp("2024-04-04")
    end = pd.Timestamp("2026-05-01")
    events_df = _load_negative_events(start, end)

    baseline_params = EntryQualityNegativeHybridStrategy().default_params()
    baseline_eval, baseline_events = _evaluate_variant(events_df, baseline_params)

    variants: list[VariantEval] = []
    best_any: tuple[VariantEval, pd.DataFrame] | None = None
    best_min10: tuple[VariantEval, pd.DataFrame] | None = None
    best_full12: tuple[VariantEval, pd.DataFrame] | None = None

    rebound_opts = [4.0, 5.0, 6.0]
    vwap_opts = [-0.5, 0.0, 0.5]
    impulse_opts = [0.5, 1.0]
    open_opts: list[float | None] = [None, 0.0, 0.25]
    bool_opts = [False, True]

    for rebound_min in rebound_opts:
        for vwap_min in vwap_opts:
            for impulse_min in impulse_opts:
                for open_min in open_opts:
                    for close_above_ema in bool_opts:
                        for ema_stack in bool_opts:
                            for ema_slope in bool_opts:
                                params = dict(baseline_params)
                                params.update(
                                    {
                                        "rebound_from_trough_min": rebound_min,
                                        "vwap_reclaim_min": vwap_min,
                                        "rebound_impulse_min": impulse_min,
                                        "entry_min_close_from_open_pct": open_min,
                                        "entry_require_close_above_ema_fast": close_above_ema,
                                        "entry_require_ema_stack": ema_stack,
                                        "entry_require_ema_slope_pos": ema_slope,
                                    }
                                )
                                eval_row, detail_df = _evaluate_variant(events_df, params)
                                variants.append(eval_row)
                                if best_any is None or eval_row.compounded_return_pct > best_any[0].compounded_return_pct:
                                    best_any = (eval_row, detail_df)
                                if eval_row.events >= 10 and (
                                    best_min10 is None or eval_row.compounded_return_pct > best_min10[0].compounded_return_pct
                                ):
                                    best_min10 = (eval_row, detail_df)
                                if eval_row.events >= 12 and (
                                    best_full12 is None or eval_row.compounded_return_pct > best_full12[0].compounded_return_pct
                                ):
                                    best_full12 = (eval_row, detail_df)

    variants_df = pd.DataFrame([asdict(v) for v in variants]).sort_values(
        ["compounded_return_pct", "events", "win_rate_pct"], ascending=[False, False, False]
    )
    variants_df.to_csv(ARTIFACT_DIR / "earnings_negative_entry_quality_variants.csv", index=False)
    baseline_events.to_csv(ARTIFACT_DIR / "earnings_negative_entry_quality_baseline_events.csv", index=False)
    if best_any is not None:
        best_any[1].to_csv(ARTIFACT_DIR / "earnings_negative_entry_quality_best_any_events.csv", index=False)
    if best_min10 is not None:
        best_min10[1].to_csv(ARTIFACT_DIR / "earnings_negative_entry_quality_best_min10_events.csv", index=False)
    if best_full12 is not None:
        best_full12[1].to_csv(ARTIFACT_DIR / "earnings_negative_entry_quality_best_full12_events.csv", index=False)

    summary = {
        "window": {"start": start.date().isoformat(), "end": end.date().isoformat()},
        "baseline": asdict(baseline_eval),
        "best_any": asdict(best_any[0]) if best_any is not None else None,
        "best_min10": asdict(best_min10[0]) if best_min10 is not None else None,
        "best_full12": asdict(best_full12[0]) if best_full12 is not None else None,
    }
    (ARTIFACT_DIR / "earnings_negative_entry_quality_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
