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
from strategies.earnings_negative_rebound_strategy import EarningsNegativeReboundStrategy
from scripts.validate_earnings_negative_rebound_family import (
    _find_long_rebound_entry_idx,
    _find_short_after_rebound_idx,
    _load_negative_events,
)
from scripts.validate_earnings_overshoot_wave_sequence import _exit_long, _exit_short
from scripts.validate_earnings_wave_sequence_main import _load_session_frame


ARTIFACT_DIR = ROOT / "artifacts" / "optimization"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

LONG_ONLY_CFG = {
    "downside_prev_close_min": 12.5,
    "start_minute": 12 * 60,
    "max_minute": 15 * 60,
    "rebound_from_trough_min": 4.0,
    "vwap_reclaim_min": -0.5,
    "rebound_impulse_min": 0.5,
    "exit_vwap_touch_buffer": 1.0,
}

TWO_LEG_CFG = {
    "downside_prev_close_min": 12.5,
    "start_minute": 12 * 60,
    "max_minute": 15 * 60,
    "rebound_from_trough_min": 4.0,
    "vwap_reclaim_min": -0.5,
    "rebound_impulse_min": 0.5,
    "long_exit_mode": "vwap_touch",
    "long_vwap_touch_buffer": 1.0,
    "long_ema_roll_gain_min": 1.0,
    "long_max_hold_bars": 20,
    "short_entry_window_bars": 8,
    "short_pullback_from_peak_min": 1.0,
    "short_vwap_break_max": -0.5,
    "short_downside_impulse_min": 1.0,
    "short_exit_mode": "time_10",
    "short_rebound_exit_pct": 0.5,
    "short_vwap_reclaim_buffer": 0.0,
    "short_max_hold_bars": 15,
}


@dataclass
class SelectorEval:
    label: str
    events: int
    selector_two_leg_events: int
    win_rate_pct: float
    mean_return_pct: float
    median_return_pct: float
    compounded_return_pct: float
    max_drawdown_pct: float
    details: dict[str, Any]


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


def _safe_pct(current: float, base: float) -> float:
    try:
        cur = float(current)
        ref = float(base)
    except Exception:
        return float("nan")
    if not np.isfinite(cur) or not np.isfinite(ref) or ref == 0.0:
        return float("nan")
    return (cur / ref - 1.0) * 100.0


def _build_session_arrays(session: pd.DataFrame) -> dict[str, np.ndarray]:
    work = session.copy()
    work["minutes_raw"] = work["date"].dt.hour * 60 + work["date"].dt.minute
    typical = (work["high"] + work["low"] + work["close"]) / 3.0
    vol = pd.to_numeric(work["volume"], errors="coerce").fillna(0.0)
    cum_vol = vol.cumsum().replace(0.0, np.nan)
    anchored_vwap = (typical * vol).cumsum() / cum_vol
    ema_fast = pd.Series(work["close"]).ewm(span=5, adjust=False).mean()
    ema_slow = pd.Series(work["close"]).ewm(span=13, adjust=False).mean()

    arrays = {
        "date": work["date"].to_numpy(copy=True),
        "orig_index": pd.to_numeric(work["orig_index"], errors="coerce").to_numpy(dtype=np.int64, copy=True),
        "minutes_et": pd.to_numeric(work["minutes_raw"], errors="coerce").to_numpy(dtype=np.int32, copy=True),
        "open": pd.to_numeric(work["open"], errors="coerce").to_numpy(dtype=np.float64, copy=True),
        "high": pd.to_numeric(work["high"], errors="coerce").to_numpy(dtype=np.float64, copy=True),
        "low": pd.to_numeric(work["low"], errors="coerce").to_numpy(dtype=np.float64, copy=True),
        "close": pd.to_numeric(work["close"], errors="coerce").to_numpy(dtype=np.float64, copy=True),
        "anchored_vwap": pd.to_numeric(anchored_vwap, errors="coerce").to_numpy(dtype=np.float64, copy=True),
        "ema_fast": ema_fast.to_numpy(dtype=np.float64, copy=True),
        "ema_slow": ema_slow.to_numpy(dtype=np.float64, copy=True),
        "ema_fast_slope": ema_fast.diff().to_numpy(dtype=np.float64, copy=True),
    }
    arrays["close_from_vwap_pct"] = (arrays["close"] / arrays["anchored_vwap"] - 1.0) * 100.0
    arrays["ret_5m_pct"] = pd.Series(arrays["close"]).pct_change(5).to_numpy(dtype=np.float64, copy=True) * 100.0
    arrays["running_trough_low"] = np.minimum.accumulate(arrays["low"])
    return arrays


class EarningsNegativeTwoLegStrategy(BaseStrategy):
    strategy_id = "earnings_negative_two_leg_selector"
    name = "Earnings Negative Two-Leg Selector"
    description = "Research-only negative earnings rebound plus optional failure short."

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

        work = data.copy().reset_index().rename(columns={"index": "orig_index"})
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
        work = work.dropna(subset=["date"]).copy()
        if work.empty:
            return actions, meta

        arrays = _build_session_arrays(work)
        cfg = self.config

        long_entry_idx = _find_long_rebound_entry_idx(
            self.event,
            arrays,
            downside_prev_close_min=float(cfg["downside_prev_close_min"]),
            start_minute=int(cfg["start_minute"]),
            max_minute=int(cfg["max_minute"]),
            rebound_from_trough_min=float(cfg["rebound_from_trough_min"]),
            vwap_reclaim_min=float(cfg["vwap_reclaim_min"]),
            rebound_impulse_min=float(cfg["rebound_impulse_min"]),
        )
        if long_entry_idx is None:
            return actions, meta

        long_exit_idx, long_exit_reason = _exit_long(
            arrays,
            entry_idx=long_entry_idx,
            exit_mode=str(cfg["long_exit_mode"]),
            vwap_touch_buffer=float(cfg["long_vwap_touch_buffer"]),
            ema_roll_gain_min=float(cfg["long_ema_roll_gain_min"]),
            max_hold_bars=int(cfg["long_max_hold_bars"]),
        )

        short_entry_idx, short_entry_reason = _find_short_after_rebound_idx(
            arrays,
            start_idx=long_exit_idx,
            max_entry_bars=int(cfg["short_entry_window_bars"]),
            pullback_from_peak_min=float(cfg["short_pullback_from_peak_min"]),
            vwap_break_max=float(cfg["short_vwap_break_max"]),
            downside_impulse_min=float(cfg["short_downside_impulse_min"]),
        )

        g_long_entry = int(arrays["orig_index"][long_entry_idx])
        g_long_exit = int(arrays["orig_index"][long_exit_idx])

        actions[g_long_entry] = SignalAction.BUY
        meta[g_long_entry] = {
            "suggested_tp": None,
            "suggested_sl": self._long_stop(arrays, long_entry_idx),
            "metadata": {
                "regime": "earnings_negative_rebound_long",
                "session_exit": "eod",
                "earnings_negative_rebound": True,
                "event_reaction_date": str(self.event.get("reaction_date")),
                "verdict_reason": (
                    "Negative off-hours earnings overshot deeply below the prior close, "
                    "then confirmed an intraday rebound back toward the fair zone."
                ),
            },
        }

        if short_entry_idx is None:
            if g_long_exit > g_long_entry:
                actions[g_long_exit] = SignalAction.SELL
                meta[g_long_exit] = {
                    "suggested_tp": None,
                    "suggested_sl": None,
                    "metadata": {
                        "regime": "earnings_negative_rebound_exit",
                        "cover_only": True,
                        "earnings_negative_rebound": True,
                        "exit_reason": long_exit_reason,
                    },
                }
            return actions, meta

        short_exit_idx, short_exit_reason = _exit_short(
            arrays,
            entry_idx=short_entry_idx,
            exit_mode=str(cfg["short_exit_mode"]),
            rebound_exit_pct=float(cfg["short_rebound_exit_pct"]),
            vwap_reclaim_buffer=float(cfg["short_vwap_reclaim_buffer"]),
            max_hold_bars=int(cfg["short_max_hold_bars"]),
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
                    "earnings_negative_rebound": True,
                    "long_exit_reason": long_exit_reason,
                    "short_entry_reason": short_entry_reason,
                    "verdict_reason": (
                        "The rebound exhausted and broke down again, so the strategy flipped "
                        "from rebound long into a second-leg failure short."
                    ),
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
                    "earnings_negative_rebound": True,
                    "exit_reason": long_exit_reason,
                },
            }
            actions[g_short_entry] = SignalAction.SELL
            meta[g_short_entry] = {
                "suggested_tp": None,
                "suggested_sl": self._short_stop(arrays, short_entry_idx, long_entry_idx),
                "metadata": {
                    "regime": "earnings_negative_short_after_rebound",
                    "earnings_negative_rebound": True,
                    "short_entry_reason": short_entry_reason,
                    "verdict_reason": (
                        "After the rebound failed, the strategy entered a second-leg short on the renewed breakdown."
                    ),
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
                    "earnings_negative_rebound": True,
                    "exit_reason": short_exit_reason,
                },
            }

        return actions, meta


def _event_row(event: dict[str, Any], result, branch_type: str) -> dict[str, Any]:
    trades = result.trades
    long_trade = next((t for t in trades if t.direction.value == "Long"), None)
    short_trade = next((t for t in trades if t.direction.value == "Short"), None)
    return {
        "symbol": str(event["symbol"]).upper(),
        "reaction_date": pd.Timestamp(event["reaction_date"]).date().isoformat(),
        "timing": event.get("timing"),
        "surprise_pct": float(event.get("surprise_pct", np.nan)),
        "trough_vs_prev_close_pct": float(event.get("trough_vs_prev_close_pct", np.nan)),
        "confirm15_close_from_open_pct": float(event.get("confirm15_close_from_open_pct", np.nan)),
        "confirm30_close_from_open_pct": float(event.get("confirm30_close_from_open_pct", np.nan)),
        "branch_type": branch_type,
        "total_return_pct": float(result.total_return_pct),
        "max_drawdown_pct": float(result.max_drawdown_pct),
        "total_trades": int(result.total_trades),
        "long_return_pct": float(long_trade.leveraged_return_pct) if long_trade and long_trade.leveraged_return_pct is not None else np.nan,
        "short_return_pct": float(short_trade.leveraged_return_pct) if short_trade and short_trade.leveraged_return_pct is not None else np.nan,
    }


def _evaluate_branch_events(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_rows: list[dict[str, Any]] = []
    two_rows: list[dict[str, Any]] = []
    rm = _risk_manager()

    for event in events.to_dict(orient="records"):
        symbol = str(event["symbol"]).upper()
        session = _load_session_frame(symbol, pd.Timestamp(event["reaction_date"]))
        if session is None or session.empty:
            continue

        long_strategy = EarningsNegativeReboundStrategy(params={})
        long_engine = BacktestEngine(
            long_strategy,
            risk_manager=rm,
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
        long_result = long_engine.run(session, symbol, leverage=1.0, capital_per_trade=1000.0, starting_equity=1000.0)
        if long_result.total_trades > 0:
            long_rows.append(_event_row(event, long_result, "long_only"))

        two_strategy = EarningsNegativeTwoLegStrategy(event=event, config=TWO_LEG_CFG)
        two_engine = BacktestEngine(
            two_strategy,
            risk_manager=rm,
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
        two_result = two_engine.run(session, symbol, leverage=1.0, capital_per_trade=1000.0, starting_equity=1000.0)
        if two_result.total_trades > 0:
            two_rows.append(_event_row(event, two_result, "two_leg"))

    return pd.DataFrame(long_rows), pd.DataFrame(two_rows)


def _evaluate_selector(merged: pd.DataFrame, *, trough_threshold: float, confirm15_min: float, confirm30_min: float, bmo_only: bool) -> SelectorEval:
    use_two = (
        (pd.to_numeric(merged["trough_vs_prev_close_pct"], errors="coerce") <= -trough_threshold)
        & (pd.to_numeric(merged["confirm15_close_from_open_pct"], errors="coerce") >= confirm15_min)
        & (pd.to_numeric(merged["confirm30_close_from_open_pct"], errors="coerce") >= confirm30_min)
    )
    if bmo_only:
        use_two &= merged["timing"].astype(str).str.lower().eq("bmo")

    chosen = pd.to_numeric(merged["long_return_pct"], errors="coerce").copy()
    chosen_branch = pd.Series("long_only", index=merged.index, dtype="object")
    mask = use_two & pd.to_numeric(merged["two_leg_return_pct"], errors="coerce").notna()
    chosen.loc[mask] = pd.to_numeric(merged.loc[mask, "two_leg_return_pct"], errors="coerce")
    chosen_branch.loc[mask] = "two_leg"

    equity = (1.0 + chosen / 100.0).cumprod()
    return SelectorEval(
        label=f"neg_selector_t{trough_threshold:g}_c15{confirm15_min:g}_c30{confirm30_min:g}_{'bmo' if bmo_only else 'all'}",
        events=int(len(chosen)),
        selector_two_leg_events=int(mask.sum()),
        win_rate_pct=float((chosen > 0).mean() * 100.0),
        mean_return_pct=float(chosen.mean()),
        median_return_pct=float(chosen.median()),
        compounded_return_pct=float(_compound_return_pct(chosen)),
        max_drawdown_pct=float(((equity / equity.cummax()) - 1.0).min() * 100.0),
        details={
            "trough_threshold": trough_threshold,
            "confirm15_min": confirm15_min,
            "confirm30_min": confirm30_min,
            "bmo_only": bmo_only,
            "selected_two_leg_events": merged.loc[mask, ["symbol", "reaction_date"]].to_dict(orient="records"),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a negative earnings hybrid selector.")
    parser.add_argument("--start", default="2024-04-04")
    parser.add_argument("--end", default="2026-05-01")
    parser.add_argument("--artifact-stem", default="earnings_negative_hybrid_selector")
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    all_events = _load_negative_events(start, end)
    eligible = all_events[
        pd.to_numeric(all_events["trough_vs_prev_close_pct"], errors="coerce") <= -LONG_ONLY_CFG["downside_prev_close_min"]
    ].copy()

    long_df, two_df = _evaluate_branch_events(eligible)
    merged = long_df[
        [
            "symbol",
            "reaction_date",
            "timing",
            "surprise_pct",
            "trough_vs_prev_close_pct",
            "confirm15_close_from_open_pct",
            "confirm30_close_from_open_pct",
            "total_return_pct",
            "total_trades",
        ]
    ].rename(columns={"total_return_pct": "long_return_pct", "total_trades": "long_trade_count"}).merge(
        two_df[["symbol", "reaction_date", "total_return_pct", "total_trades", "short_return_pct"]].rename(
            columns={"total_return_pct": "two_leg_return_pct", "total_trades": "two_leg_trade_count"}
        ),
        on=["symbol", "reaction_date"],
        how="left",
    )

    baseline_eval = SelectorEval(
        label="negative_long_only_baseline",
        events=int(len(merged)),
        selector_two_leg_events=0,
        win_rate_pct=float((pd.to_numeric(merged["long_return_pct"], errors="coerce") > 0).mean() * 100.0),
        mean_return_pct=float(pd.to_numeric(merged["long_return_pct"], errors="coerce").mean()),
        median_return_pct=float(pd.to_numeric(merged["long_return_pct"], errors="coerce").median()),
        compounded_return_pct=float(_compound_return_pct(pd.to_numeric(merged["long_return_pct"], errors="coerce"))),
        max_drawdown_pct=float(
            (((1.0 + pd.to_numeric(merged["long_return_pct"], errors="coerce") / 100.0).cumprod()) /
             ((1.0 + pd.to_numeric(merged["long_return_pct"], errors="coerce") / 100.0).cumprod().cummax()) - 1.0).min() * 100.0
        ),
        details={},
    )

    selector_results: list[SelectorEval] = []
    for trough in (15.0, 16.0, 17.0, 17.5, 18.0, 19.0, 19.5, 20.0):
        for c15 in (-0.25, 0.0, 0.1, 0.2):
            for c30 in (-0.25, -0.1, 0.0, 0.1, 0.15, 0.2):
                for bmo_only in (False, True):
                    selector_results.append(
                        _evaluate_selector(
                            merged,
                            trough_threshold=trough,
                            confirm15_min=c15,
                            confirm30_min=c30,
                            bmo_only=bmo_only,
                        )
                    )

    best_any = max(selector_results, key=lambda x: x.compounded_return_pct)
    best_min2 = max((r for r in selector_results if r.selector_two_leg_events >= 2), key=lambda x: x.compounded_return_pct, default=None)

    variants_df = pd.DataFrame([asdict(r) for r in selector_results]).sort_values("compounded_return_pct", ascending=False)
    merged_out = merged.copy()

    out_json = ARTIFACT_DIR / f"{args.artifact_stem}.json"
    out_variants = ARTIFACT_DIR / f"{args.artifact_stem}_variants.csv"
    out_events = ARTIFACT_DIR / f"{args.artifact_stem}_merged_events.csv"

    payload = {
        "window": {"start": start.date().isoformat(), "end": end.date().isoformat()},
        "baseline": asdict(baseline_eval),
        "best_any": asdict(best_any),
        "best_min2": asdict(best_min2) if best_min2 is not None else None,
        "long_only_cfg": LONG_ONLY_CFG,
        "two_leg_cfg": TWO_LEG_CFG,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    variants_df.to_csv(out_variants, index=False)
    merged_out.to_csv(out_events, index=False)

    print(
        f"negative_hybrid_selector: baseline={baseline_eval.compounded_return_pct:.3f}% "
        f"best_any={best_any.compounded_return_pct:.3f}% "
        f"best_min2={(best_min2.compounded_return_pct if best_min2 else float('nan')):.3f}%"
    )
    print(f"Wrote artifacts:\n- {out_json}\n- {out_variants}\n- {out_events}")


if __name__ == "__main__":
    main()
