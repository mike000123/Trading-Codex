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
from scripts.validate_earnings_overshoot_first_dump import _compound_return_pct, _load_labeled_events
from scripts.validate_earnings_wave_sequence_main import _build_session_arrays, _load_session_frame


ARTIFACT_DIR = ROOT / "artifacts" / "optimization"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

WAVE_EVENTS_PATH = ARTIFACT_DIR / "earnings_wave_sequence_main_engine_events.csv"
LEG_SPLIT_JSON_PATH = ARTIFACT_DIR / "earnings_wave_leg_split_engine.json"
LEG_SPLIT_EVENTS_PATH = ARTIFACT_DIR / "earnings_wave_leg_split_engine_events.csv"


@dataclass(frozen=True)
class FailedReclaimConfig:
    label: str
    overshoot_prev_close_min: float
    start_minute: int
    max_minute: int
    vwap_break_min: float
    rebound_min: float
    reclaim_window_bars: int
    reclaim_vwap_buffer: float
    lower_high_min: float
    breakdown_impulse_min: float


@dataclass
class FailedReclaimEval:
    label: str
    eligible_events: int
    triggered_events: int
    coverage_pct: float
    total_trades: int
    win_rate_pct: float
    mean_event_return_pct: float
    median_event_return_pct: float
    compounded_event_return_pct: float
    mean_trade_return_pct: float
    max_drawdown_pct: float
    params: dict[str, Any]


@dataclass
class SelectorEval:
    label: str
    trades: int
    win_rate_pct: float
    mean_return_pct: float
    median_return_pct: float
    compounded_return_pct: float
    coverage_pct: float
    selector_details: dict[str, Any]


def _risk_manager() -> RiskManager:
    return RiskManager(
        RiskConfig(
            max_capital_per_trade_pct=100.0,
            max_daily_loss_pct=100.0,
            max_open_positions=999,
            default_max_loss_pct_of_capital=50.0,
        )
    )


def _safe_pct(current, base) -> float:
    try:
        cur = float(current)
        ref = float(base)
    except Exception:
        return np.nan
    if not np.isfinite(cur) or not np.isfinite(ref) or ref == 0:
        return np.nan
    return (cur / ref - 1.0) * 100.0


def _find_failed_reclaim_trigger(
    event: dict[str, Any],
    arrays: dict[str, np.ndarray],
    *,
    cfg: FailedReclaimConfig,
) -> int | None:
    prev_close = float(event.get("prev_close", np.nan))
    if not np.isfinite(prev_close) or prev_close <= 0:
        return None

    minutes = arrays["minutes_et"]
    close = arrays["close"]
    high = arrays["high"]
    low = arrays["low"]
    close_from_vwap = arrays["close_from_vwap_pct"]
    high_from_vwap = (arrays["high"] / arrays["anchored_vwap"] - 1.0) * 100.0
    ret_5m = arrays["ret_5m_pct"]
    running_peak = arrays["running_peak_high"]
    running_peak_vs_prev = (running_peak / prev_close - 1.0) * 100.0

    candidate_mask = (
        (minutes >= cfg.start_minute)
        & (minutes <= cfg.max_minute)
        & np.isfinite(close)
        & np.isfinite(low)
        & np.isfinite(high)
        & np.isfinite(close_from_vwap)
        & np.isfinite(high_from_vwap)
        & np.isfinite(ret_5m)
        & np.isfinite(running_peak_vs_prev)
        & (running_peak_vs_prev >= cfg.overshoot_prev_close_min)
        & (close_from_vwap <= -cfg.vwap_break_min)
    )
    candidate_idxs = np.flatnonzero(candidate_mask)
    if candidate_idxs.size == 0:
        return None

    for break_idx in candidate_idxs:
        break_low = float(low[break_idx])
        break_close = float(close[break_idx])
        peak_ref = float(running_peak[break_idx])
        rebound_end = min(break_idx + cfg.reclaim_window_bars, len(close) - 1)
        if rebound_end <= break_idx:
            continue

        rebound_slice = slice(break_idx + 1, rebound_end + 1)
        rebound_high = high[rebound_slice]
        rebound_high_from_vwap = high_from_vwap[rebound_slice]
        if rebound_high.size == 0:
            continue

        max_rebound_high = float(np.nanmax(rebound_high))
        max_rebound_high_from_vwap = float(np.nanmax(rebound_high_from_vwap))
        rebound_size = _safe_pct(max_rebound_high, break_close)
        lower_high_from_peak = -_safe_pct(max_rebound_high, peak_ref)

        if not np.isfinite(rebound_size) or rebound_size < cfg.rebound_min:
            continue
        if not np.isfinite(max_rebound_high_from_vwap) or max_rebound_high_from_vwap > cfg.reclaim_vwap_buffer:
            continue
        if not np.isfinite(lower_high_from_peak) or lower_high_from_peak < cfg.lower_high_min:
            continue

        for confirm_idx in range(rebound_end + 1, len(close)):
            minute = int(minutes[confirm_idx])
            if minute > cfg.max_minute:
                break
            if not np.isfinite(close[confirm_idx]) or not np.isfinite(ret_5m[confirm_idx]):
                continue
            if close[confirm_idx] < break_low and ret_5m[confirm_idx] <= -cfg.breakdown_impulse_min:
                return int(confirm_idx)
    return None


class FailedReclaimEngineStrategy(BaseStrategy):
    strategy_id = "earnings_failed_reclaim_engine"
    name = "Earnings Failed Reclaim Engine"
    description = "Research-only failed-reclaim short after positive earnings overshoot."

    def __init__(self, *, event: dict[str, Any], config: FailedReclaimConfig) -> None:
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
        entry_idx = _find_failed_reclaim_trigger(self.event, arrays, cfg=self.config)
        if entry_idx is None:
            return actions, meta

        actions[entry_idx] = SignalAction.SELL
        meta[entry_idx] = {
            "suggested_tp": None,
            "suggested_sl": self._short_stop(arrays, entry_idx),
            "metadata": {
                "regime": "earnings_failed_reclaim_short",
                "session_exit": "eod",
                "earnings_failed_reclaim": True,
                "event_variant": self.config.label,
                "verdict_reason": (
                    "Positive earnings overshoot broke below anchored VWAP, rebounded weakly, "
                    "failed to reclaim, and broke down again."
                ),
            },
        }
        return actions, meta


def _evaluate_failed_reclaim_variant(events: pd.DataFrame, cfg: FailedReclaimConfig) -> tuple[FailedReclaimEval, pd.DataFrame]:
    eligible = events[pd.to_numeric(events["peak_vs_prev_close_pct"], errors="coerce") >= cfg.overshoot_prev_close_min].copy()
    rows: list[dict[str, Any]] = []
    for event in eligible.to_dict(orient="records"):
        session = _load_session_frame(str(event["symbol"]).upper(), pd.Timestamp(event["reaction_date"]))
        if session is None or session.empty:
            continue

        strategy = FailedReclaimEngineStrategy(event=event, config=cfg)
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
        trade = result.trades[0]
        rows.append(
            {
                "variant": cfg.label,
                "symbol": event["symbol"],
                "reaction_date": pd.Timestamp(event["reaction_date"]).date().isoformat(),
                "timing": event.get("timing"),
                "surprise_pct": float(event.get("surprise_pct", np.nan)),
                "gap_pct": float(event.get("gap_pct", np.nan)),
                "peak_vs_prev_close_pct": float(event.get("peak_vs_prev_close_pct", np.nan)),
                "peak_vs_equilibrium_pct": float(event.get("peak_vs_equilibrium_pct", np.nan)),
                "close_vs_peak_pct": float(event.get("close_vs_peak_pct", np.nan)),
                "total_return_pct": float(result.total_return_pct),
                "max_drawdown_pct": float(result.max_drawdown_pct),
                "sharpe_ratio": float(result.sharpe_ratio),
                "total_trades": int(result.total_trades),
                "short_return_pct": float(trade.leveraged_return_pct) if trade.leveraged_return_pct is not None else np.nan,
                "entry_time": pd.Timestamp(trade.entry_time).isoformat() if trade.entry_time is not None else None,
                "exit_time": pd.Timestamp(trade.exit_time).isoformat() if trade.exit_time is not None else None,
                "outcome": str(trade.outcome.value) if trade.outcome is not None else None,
            }
        )

    trades_df = pd.DataFrame(rows)
    if trades_df.empty:
        return (
            FailedReclaimEval(
                label=cfg.label,
                eligible_events=int(len(eligible)),
                triggered_events=0,
                coverage_pct=0.0,
                total_trades=0,
                win_rate_pct=np.nan,
                mean_event_return_pct=np.nan,
                median_event_return_pct=np.nan,
                compounded_event_return_pct=0.0,
                mean_trade_return_pct=np.nan,
                max_drawdown_pct=np.nan,
                params=asdict(cfg),
            ),
            trades_df,
        )

    returns = pd.to_numeric(trades_df["total_return_pct"], errors="coerce").dropna()
    equity_curve = (1.0 + returns / 100.0).cumprod()
    max_drawdown = float(((equity_curve / equity_curve.cummax()) - 1.0).min() * 100.0) if not equity_curve.empty else np.nan
    eval_row = FailedReclaimEval(
        label=cfg.label,
        eligible_events=int(len(eligible)),
        triggered_events=int(len(trades_df)),
        coverage_pct=float(len(trades_df) / len(eligible) * 100.0) if len(eligible) > 0 else 0.0,
        total_trades=int(len(trades_df)),
        win_rate_pct=float((returns > 0).mean() * 100.0),
        mean_event_return_pct=float(returns.mean()),
        median_event_return_pct=float(returns.median()),
        compounded_event_return_pct=float(_compound_return_pct(returns)),
        mean_trade_return_pct=float(returns.mean()),
        max_drawdown_pct=max_drawdown,
        params=asdict(cfg),
    )
    return eval_row, trades_df


def _failed_reclaim_grid() -> list[FailedReclaimConfig]:
    configs: list[FailedReclaimConfig] = []
    for overshoot_prev_close_min in (15.0, 20.0):
        for start_minute in (12 * 60, 13 * 60):
            for max_minute in (14 * 60,):
                for vwap_break_min in (0.0, 0.5):
                    for rebound_min in (0.25, 0.5):
                        for reclaim_window_bars in (10,):
                            for reclaim_vwap_buffer in (0.0,):
                                for lower_high_min in (2.0, 3.0):
                                    for breakdown_impulse_min in (0.5,):
                                        label = (
                                            f"failed_reclaim_pc{int(overshoot_prev_close_min)}"
                                            f"_start{int(start_minute)}"
                                            f"_break{vwap_break_min:.2f}"
                                            f"_reb{rebound_min:.2f}"
                                            f"_lh{lower_high_min:.1f}"
                                        )
                                        configs.append(
                                            FailedReclaimConfig(
                                                label=label,
                                                overshoot_prev_close_min=overshoot_prev_close_min,
                                                start_minute=start_minute,
                                                max_minute=max_minute,
                                                vwap_break_min=vwap_break_min,
                                                rebound_min=rebound_min,
                                                reclaim_window_bars=reclaim_window_bars,
                                                reclaim_vwap_buffer=reclaim_vwap_buffer,
                                                lower_high_min=lower_high_min,
                                                breakdown_impulse_min=breakdown_impulse_min,
                                            )
                                        )
    return configs


def _load_wave_best_events() -> pd.DataFrame:
    df = pd.read_csv(WAVE_EVENTS_PATH)
    df = df[df["variant"] == "wave_sequence_best_return"].copy()
    df["reaction_date"] = pd.to_datetime(df["reaction_date"], errors="coerce").dt.date.astype(str)
    df = df.rename(columns={"total_return_pct": "wave_return_pct"})
    return df


def _load_conditional_best_events() -> pd.DataFrame:
    if not LEG_SPLIT_JSON_PATH.exists() or not LEG_SPLIT_EVENTS_PATH.exists():
        return pd.DataFrame()
    payload = json.loads(LEG_SPLIT_JSON_PATH.read_text(encoding="utf-8"))
    label = payload.get("best_conditional_long", {}).get("label")
    if not label:
        return pd.DataFrame()
    df = pd.read_csv(LEG_SPLIT_EVENTS_PATH)
    df = df[df["variant"] == label].copy()
    df["reaction_date"] = pd.to_datetime(df["reaction_date"], errors="coerce").dt.date.astype(str)
    df = df.rename(columns={"total_return_pct": "conditional_return_pct"})
    return df


def _merge_branch_data(base_events: pd.DataFrame, failed_reclaim_events: pd.DataFrame) -> pd.DataFrame:
    base = base_events.copy()
    base["reaction_date"] = pd.to_datetime(base["reaction_date"], errors="coerce").dt.date.astype(str)

    wave_df = _load_wave_best_events().loc[:, ["symbol", "reaction_date", "wave_return_pct"]]
    cond_df = _load_conditional_best_events()
    if not cond_df.empty:
        cond_df = cond_df.loc[:, ["symbol", "reaction_date", "conditional_return_pct"]]

    fr_df = failed_reclaim_events.copy()
    if fr_df.empty:
        fr_df = pd.DataFrame(columns=["symbol", "reaction_date", "failed_reclaim_return_pct"])
    else:
        fr_df = fr_df.rename(columns={"total_return_pct": "failed_reclaim_return_pct"}).loc[
            :, ["symbol", "reaction_date", "failed_reclaim_return_pct"]
        ]

    merged = base.merge(wave_df, on=["symbol", "reaction_date"], how="left")
    merged = merged.merge(fr_df, on=["symbol", "reaction_date"], how="left")
    if not cond_df.empty:
        merged = merged.merge(cond_df, on=["symbol", "reaction_date"], how="left")
    else:
        merged["conditional_return_pct"] = np.nan
    return merged


def _evaluate_selector(df: pd.DataFrame, selection: pd.Series, label: str, details: dict[str, Any]) -> SelectorEval:
    chosen = df.copy()
    chosen["selected_branch"] = selection
    chosen["selected_return_pct"] = np.nan

    chosen.loc[chosen["selected_branch"] == "wave", "selected_return_pct"] = chosen["wave_return_pct"]
    chosen.loc[chosen["selected_branch"] == "failed_reclaim", "selected_return_pct"] = chosen["failed_reclaim_return_pct"]
    chosen.loc[chosen["selected_branch"] == "conditional", "selected_return_pct"] = chosen["conditional_return_pct"]

    returns = pd.to_numeric(chosen["selected_return_pct"], errors="coerce").dropna()
    trades = int(len(returns))
    coverage_pct = float(trades / len(chosen) * 100.0) if len(chosen) > 0 else 0.0
    if returns.empty:
        return SelectorEval(label, 0, np.nan, np.nan, np.nan, 0.0, 0.0, details)

    return SelectorEval(
        label=label,
        trades=trades,
        win_rate_pct=float((returns > 0).mean() * 100.0),
        mean_return_pct=float(returns.mean()),
        median_return_pct=float(returns.median()),
        compounded_return_pct=float(_compound_return_pct(returns)),
        coverage_pct=coverage_pct,
        selector_details=details,
    )


def _search_simple_selectors(df: pd.DataFrame) -> list[SelectorEval]:
    results: list[SelectorEval] = []

    # Baselines.
    results.append(
        _evaluate_selector(
            df,
            pd.Series(["wave"] * len(df), index=df.index),
            "baseline_wave",
            {"mode": "constant", "branch": "wave"},
        )
    )
    results.append(
        _evaluate_selector(
            df,
            pd.Series(["failed_reclaim"] * len(df), index=df.index),
            "baseline_failed_reclaim",
            {"mode": "constant", "branch": "failed_reclaim"},
        )
    )
    results.append(
        _evaluate_selector(
            df,
            pd.Series(["conditional"] * len(df), index=df.index),
            "baseline_conditional",
            {"mode": "constant", "branch": "conditional"},
        )
    )

    # Oracle upper bounds.
    oracle_two = []
    oracle_three = []
    for _, row in df.iterrows():
        options_two = {
            "wave": row.get("wave_return_pct"),
            "failed_reclaim": row.get("failed_reclaim_return_pct"),
        }
        finite_two = {k: float(v) for k, v in options_two.items() if pd.notna(v)}
        oracle_two.append(max(finite_two, key=finite_two.get) if finite_two else "wave")

        options_three = {
            "wave": row.get("wave_return_pct"),
            "failed_reclaim": row.get("failed_reclaim_return_pct"),
            "conditional": row.get("conditional_return_pct"),
        }
        finite_three = {k: float(v) for k, v in options_three.items() if pd.notna(v)}
        oracle_three.append(max(finite_three, key=finite_three.get) if finite_three else "wave")

    results.append(
        _evaluate_selector(
            df,
            pd.Series(oracle_two, index=df.index),
            "oracle_wave_vs_failed_reclaim",
            {"mode": "oracle", "branches": ["wave", "failed_reclaim"]},
        )
    )
    results.append(
        _evaluate_selector(
            df,
            pd.Series(oracle_three, index=df.index),
            "oracle_wave_failed_reclaim_conditional",
            {"mode": "oracle", "branches": ["wave", "failed_reclaim", "conditional"]},
        )
    )

    numeric_features = ["peak_vs_prev_close_pct", "surprise_pct", "gap_pct"]
    for feature in numeric_features:
        series = pd.to_numeric(df[feature], errors="coerce").dropna()
        if series.empty:
            continue
        thresholds = sorted(
            {
                rounded
                for v in series.unique()
                for rounded in (
                    round(float(v), 3),
                    round(float(v), 1),
                    round(float(v), 0),
                )
            }
        )
        for threshold in thresholds:
            choice = np.where(pd.to_numeric(df[feature], errors="coerce") >= threshold, "wave", "failed_reclaim")
            results.append(
                _evaluate_selector(
                    df,
                    pd.Series(choice, index=df.index),
                    f"selector_{feature}_ge_{threshold}",
                    {"mode": "threshold", "feature": feature, "op": ">=", "threshold": threshold, "high_branch": "wave", "low_branch": "failed_reclaim"},
                )
            )
            choice = np.where(pd.to_numeric(df[feature], errors="coerce") >= threshold, "failed_reclaim", "wave")
            results.append(
                _evaluate_selector(
                    df,
                    pd.Series(choice, index=df.index),
                    f"selector_{feature}_ge_{threshold}_flip",
                    {"mode": "threshold", "feature": feature, "op": ">=", "threshold": threshold, "high_branch": "failed_reclaim", "low_branch": "wave"},
                )
            )

    timing = df["timing"].astype(str)
    results.append(
        _evaluate_selector(
            df,
            pd.Series(np.where(timing == "bmo", "wave", "failed_reclaim"), index=df.index),
            "selector_timing_bmo_wave",
            {"mode": "timing", "bmo_branch": "wave", "amc_branch": "failed_reclaim"},
        )
    )
    results.append(
        _evaluate_selector(
            df,
            pd.Series(np.where(timing == "bmo", "failed_reclaim", "wave"), index=df.index),
            "selector_timing_bmo_failed_reclaim",
            {"mode": "timing", "bmo_branch": "failed_reclaim", "amc_branch": "wave"},
        )
    )

    # Two-stage selector with conditional branch in the middle.
    peak_series = pd.to_numeric(df["peak_vs_prev_close_pct"], errors="coerce")
    surprise_series = pd.to_numeric(df["surprise_pct"], errors="coerce")
    for peak_th in (22.0, 24.0, 26.0):
        for surprise_th in (10.0, 15.0, 25.0):
            choices = []
            for _, row in df.iterrows():
                peak = pd.to_numeric(pd.Series([row.get("peak_vs_prev_close_pct")]), errors="coerce").iloc[0]
                surprise = pd.to_numeric(pd.Series([row.get("surprise_pct")]), errors="coerce").iloc[0]
                if np.isfinite(peak) and peak >= peak_th:
                    choices.append("wave")
                elif np.isfinite(surprise) and surprise >= surprise_th:
                    choices.append("conditional")
                else:
                    choices.append("failed_reclaim")
            results.append(
                _evaluate_selector(
                    df,
                    pd.Series(choices, index=df.index),
                    f"selector_peak{peak_th}_surprise{surprise_th}",
                    {"mode": "two_stage", "wave_if_peak_ge": peak_th, "conditional_if_surprise_ge": surprise_th, "else": "failed_reclaim"},
                )
            )

    return sorted(results, key=lambda item: (item.compounded_return_pct, item.mean_return_pct, item.coverage_pct), reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a hybrid selector between earnings overshoot branches on event days only."
    )
    parser.add_argument("--start", default="2024-04-04", help="Event-study start date (default: 2024-04-04).")
    parser.add_argument("--end", default="2026-05-01", help="Event-study end date (default: 2026-05-01).")
    parser.add_argument(
        "--artifact-stem",
        default="earnings_hybrid_selector",
        help="Artifact stem (default: earnings_hybrid_selector).",
    )
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    base_events = _load_labeled_events(start, end)

    failed_rows: list[FailedReclaimEval] = []
    failed_event_frames: list[pd.DataFrame] = []
    for cfg in _failed_reclaim_grid():
        eval_row, events_df = _evaluate_failed_reclaim_variant(base_events, cfg)
        failed_rows.append(eval_row)
        if not events_df.empty:
            failed_event_frames.append(events_df)

    failed_df = pd.DataFrame([asdict(row) for row in failed_rows]).sort_values(
        ["compounded_event_return_pct", "coverage_pct", "triggered_events"],
        ascending=[False, False, False],
    )
    best_failed = {}
    if not failed_df.empty:
        candidates = failed_df[failed_df["triggered_events"] >= 5]
        best_failed = (candidates if not candidates.empty else failed_df).iloc[0].to_dict()

    failed_events_all = pd.concat(failed_event_frames, ignore_index=True) if failed_event_frames else pd.DataFrame()
    best_failed_events = pd.DataFrame()
    if not failed_events_all.empty and best_failed:
        best_failed_events = failed_events_all[failed_events_all["variant"] == best_failed["label"]].copy()

    merged = _merge_branch_data(base_events, best_failed_events)
    selector_results = _search_simple_selectors(merged)
    selector_df = pd.DataFrame([asdict(item) for item in selector_results])

    out_json = ARTIFACT_DIR / f"{args.artifact_stem}.json"
    out_failed = ARTIFACT_DIR / f"{args.artifact_stem}_failed_reclaim_variants.csv"
    out_selectors = ARTIFACT_DIR / f"{args.artifact_stem}_selector_variants.csv"
    out_merged = ARTIFACT_DIR / f"{args.artifact_stem}_merged_events.csv"

    payload = {
        "window": {"start": start.date().isoformat(), "end": end.date().isoformat()},
        "notes": [
            "This pass first engine-validates a compact failed-reclaim branch grid on event days.",
            "It then merges the best failed-reclaim engine branch with the earlier wave-sequence engine branch and the conditional wave branch.",
            "The selector search is intentionally simple: constant baselines, oracle upper bounds, single-feature thresholds, timing splits, and a small two-stage shape rule.",
        ],
        "best_failed_reclaim_branch": best_failed,
        "top_selectors": selector_df.head(20).to_dict(orient="records"),
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    failed_df.to_csv(out_failed, index=False)
    selector_df.to_csv(out_selectors, index=False)
    merged.to_csv(out_merged, index=False)

    if best_failed:
        print(
            f"best_failed_reclaim: {best_failed['label']} "
            f"return={float(best_failed['compounded_event_return_pct']):.3f}% "
            f"coverage={float(best_failed['coverage_pct']):.1f}% "
            f"events={int(best_failed['triggered_events'])}"
        )
    if selector_results:
        best_selector = selector_results[0]
        print(
            f"best_selector: {best_selector.label} "
            f"return={best_selector.compounded_return_pct:.3f}% "
            f"coverage={best_selector.coverage_pct:.1f}% "
            f"trades={best_selector.trades}"
        )
    print(f"Wrote results to {out_json}")


if __name__ == "__main__":
    main()
