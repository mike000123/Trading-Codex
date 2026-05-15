from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_earnings_overshoot_first_dump import _build_event_contexts, _compound_return_pct, _safe_pct
from scripts.validate_earnings_overshoot_wave_sequence import _exit_long, _exit_short


ARTIFACT_DIR = ROOT / "artifacts" / "optimization"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
LABELED_EVENTS_PATH = ARTIFACT_DIR / "earnings_overshoot_dump_events_labeled.csv"


@dataclass
class LongOnlyResult:
    variant: str
    events: int
    win_rate_pct: float
    mean_return_pct: float
    median_return_pct: float
    compounded_return_pct: float
    labeled_match_pct: float
    params: dict


@dataclass
class TwoLegResult:
    variant: str
    events: int
    total_trades: int
    short_leg_events: int
    win_rate_pct: float
    mean_event_return_pct: float
    median_event_return_pct: float
    compounded_event_return_pct: float
    mean_trade_return_pct: float
    labeled_match_pct: float
    params: dict


def _load_negative_events(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if not LABELED_EVENTS_PATH.exists():
        raise SystemExit(f"Missing labeled events at {LABELED_EVENTS_PATH}.")
    df = pd.read_csv(LABELED_EVENTS_PATH)
    if df.empty:
        raise SystemExit("The labeled earnings event table is empty.")
    df["reaction_date"] = pd.to_datetime(df["reaction_date"], errors="coerce")
    df["timing"] = df["timing"].astype(str).str.lower()
    df = df[(df["reaction_date"] >= start) & (df["reaction_date"] <= end)].copy()
    df = df[pd.to_numeric(df["surprise_pct"], errors="coerce") < 0].copy()
    df = df[df["timing"].isin({"bmo", "amc"})].copy()
    return df.reset_index(drop=True)


def _find_long_rebound_entry_idx(
    event: dict,
    arrays: dict,
    *,
    downside_prev_close_min: float,
    start_minute: int,
    max_minute: int,
    rebound_from_trough_min: float,
    vwap_reclaim_min: float,
    rebound_impulse_min: float,
) -> int | None:
    prev_close = float(event.get("prev_close", np.nan))
    if not np.isfinite(prev_close) or prev_close <= 0:
        return None

    minutes = arrays["minutes_et"]
    close = arrays["close"]
    close_from_vwap = arrays["close_from_vwap_pct"]
    ret_5m = arrays["ret_5m_pct"]
    low = arrays["low"]
    running_trough = np.minimum.accumulate(low)
    running_trough_vs_prev = (running_trough / prev_close - 1.0) * 100.0
    rebound_from_trough = (close / running_trough - 1.0) * 100.0

    cond = (
        (minutes >= start_minute)
        & (minutes <= max_minute)
        & np.isfinite(close)
        & np.isfinite(close_from_vwap)
        & np.isfinite(ret_5m)
        & np.isfinite(running_trough_vs_prev)
        & np.isfinite(rebound_from_trough)
        & (running_trough_vs_prev <= -downside_prev_close_min)
        & (rebound_from_trough >= rebound_from_trough_min)
        & (close_from_vwap >= vwap_reclaim_min)
        & (ret_5m >= rebound_impulse_min)
    )
    hits = np.flatnonzero(cond)
    if hits.size == 0:
        return None
    return int(hits[0])


def _find_short_after_rebound_idx(
    arrays: dict,
    *,
    start_idx: int,
    max_entry_bars: int,
    pullback_from_peak_min: float,
    vwap_break_max: float,
    downside_impulse_min: float,
) -> tuple[int | None, str | None]:
    close = arrays["close"]
    high = arrays["high"]
    close_from_vwap = arrays["close_from_vwap_pct"]
    ret_5m = arrays["ret_5m_pct"]
    n = len(close)
    if start_idx >= n - 1:
        return None, None
    end_idx = min(start_idx + max_entry_bars, n - 1)
    rebound_peak = float(np.nanmax(high[start_idx:end_idx + 1]))
    if not np.isfinite(rebound_peak):
        return None, None
    pullback_from_peak = (close / rebound_peak - 1.0) * 100.0
    for idx in range(start_idx + 1, end_idx + 1):
        if not np.isfinite(close[idx]) or not np.isfinite(close_from_vwap[idx]) or not np.isfinite(ret_5m[idx]):
            continue
        if (
            np.isfinite(pullback_from_peak[idx])
            and pullback_from_peak[idx] <= -pullback_from_peak_min
            and close_from_vwap[idx] <= vwap_break_max
            and ret_5m[idx] <= -downside_impulse_min
        ):
            return idx, "rebound_failure"
    return None, None


def _trade_long_only(
    event: dict,
    arrays: dict,
    *,
    downside_prev_close_min: float,
    start_minute: int,
    max_minute: int,
    rebound_from_trough_min: float,
    vwap_reclaim_min: float,
    rebound_impulse_min: float,
    long_exit_mode: str,
    long_vwap_touch_buffer: float,
    long_ema_roll_gain_min: float,
    long_max_hold_bars: int,
) -> dict | None:
    entry_idx = _find_long_rebound_entry_idx(
        event,
        arrays,
        downside_prev_close_min=downside_prev_close_min,
        start_minute=start_minute,
        max_minute=max_minute,
        rebound_from_trough_min=rebound_from_trough_min,
        vwap_reclaim_min=vwap_reclaim_min,
        rebound_impulse_min=rebound_impulse_min,
    )
    if entry_idx is None:
        return None

    exit_idx, exit_reason = _exit_long(
        arrays,
        entry_idx=entry_idx,
        exit_mode=long_exit_mode,
        vwap_touch_buffer=long_vwap_touch_buffer,
        ema_roll_gain_min=long_ema_roll_gain_min,
        max_hold_bars=long_max_hold_bars,
    )

    date = arrays["date"]
    close = arrays["close"]
    entry_px = float(close[entry_idx])
    exit_px = float(close[exit_idx])
    ret = _safe_pct(exit_px, entry_px)
    return {
        "symbol": event["symbol"],
        "reaction_date": pd.Timestamp(event["reaction_date"]).date().isoformat(),
        "timing": event.get("timing"),
        "surprise_pct": float(event.get("surprise_pct", np.nan)),
        "trough_vs_prev_close_pct": float(event.get("trough_vs_prev_close_pct", np.nan)),
        "entry_time": pd.Timestamp(date[entry_idx]).isoformat(),
        "entry_px": entry_px,
        "exit_time": pd.Timestamp(date[exit_idx]).isoformat(),
        "exit_px": exit_px,
        "return_pct": ret,
        "exit_reason": exit_reason,
        "label_match": bool(event.get("overshoot_rebound_label", False)),
    }


def _trade_two_leg(
    event: dict,
    arrays: dict,
    *,
    downside_prev_close_min: float,
    start_minute: int,
    max_minute: int,
    rebound_from_trough_min: float,
    vwap_reclaim_min: float,
    rebound_impulse_min: float,
    long_exit_mode: str,
    long_vwap_touch_buffer: float,
    long_ema_roll_gain_min: float,
    long_max_hold_bars: int,
    short_entry_window_bars: int,
    short_pullback_from_peak_min: float,
    short_vwap_break_max: float,
    short_downside_impulse_min: float,
    short_exit_mode: str,
    short_rebound_exit_pct: float,
    short_vwap_reclaim_buffer: float,
    short_max_hold_bars: int,
) -> dict | None:
    long_trade = _trade_long_only(
        event,
        arrays,
        downside_prev_close_min=downside_prev_close_min,
        start_minute=start_minute,
        max_minute=max_minute,
        rebound_from_trough_min=rebound_from_trough_min,
        vwap_reclaim_min=vwap_reclaim_min,
        rebound_impulse_min=rebound_impulse_min,
        long_exit_mode=long_exit_mode,
        long_vwap_touch_buffer=long_vwap_touch_buffer,
        long_ema_roll_gain_min=long_ema_roll_gain_min,
        long_max_hold_bars=long_max_hold_bars,
    )
    if long_trade is None:
        return None

    date = arrays["date"]
    close = arrays["close"]
    long_entry_idx = int(np.where(date == pd.Timestamp(long_trade["entry_time"]))[0][0])
    long_exit_idx = int(np.where(date == pd.Timestamp(long_trade["exit_time"]))[0][0])
    short_entry_idx, short_entry_reason = _find_short_after_rebound_idx(
        arrays,
        start_idx=long_exit_idx,
        max_entry_bars=short_entry_window_bars,
        pullback_from_peak_min=short_pullback_from_peak_min,
        vwap_break_max=short_vwap_break_max,
        downside_impulse_min=short_downside_impulse_min,
    )

    short_exit_idx = None
    short_exit_reason = None
    short_ret = 0.0
    short_entry_px = np.nan
    short_exit_px = np.nan

    if short_entry_idx is not None:
        short_exit_idx, short_exit_reason = _exit_short(
            arrays,
            entry_idx=short_entry_idx,
            exit_mode=short_exit_mode,
            rebound_exit_pct=short_rebound_exit_pct,
            vwap_reclaim_buffer=short_vwap_reclaim_buffer,
            max_hold_bars=short_max_hold_bars,
        )
        short_entry_px = float(close[short_entry_idx])
        short_exit_px = float(close[short_exit_idx])
        short_ret = -_safe_pct(short_exit_px, short_entry_px)

    long_ret = float(long_trade["return_pct"])
    event_ret = (1.0 + long_ret / 100.0) * (1.0 + short_ret / 100.0) - 1.0
    return {
        "symbol": event["symbol"],
        "reaction_date": pd.Timestamp(event["reaction_date"]).date().isoformat(),
        "timing": event.get("timing"),
        "surprise_pct": float(event.get("surprise_pct", np.nan)),
        "trough_vs_prev_close_pct": float(event.get("trough_vs_prev_close_pct", np.nan)),
        "long_entry_time": long_trade["entry_time"],
        "long_entry_px": long_trade["entry_px"],
        "long_exit_time": long_trade["exit_time"],
        "long_exit_px": long_trade["exit_px"],
        "long_exit_reason": long_trade["exit_reason"],
        "long_return_pct": long_ret,
        "short_entry_time": pd.Timestamp(date[short_entry_idx]).isoformat() if short_entry_idx is not None else None,
        "short_entry_px": short_entry_px,
        "short_entry_reason": short_entry_reason,
        "short_exit_time": pd.Timestamp(date[short_exit_idx]).isoformat() if short_exit_idx is not None else None,
        "short_exit_px": short_exit_px,
        "short_exit_reason": short_exit_reason,
        "short_return_pct": short_ret if short_entry_idx is not None else np.nan,
        "event_return_pct": event_ret * 100.0,
        "total_trades": 1 + int(short_entry_idx is not None),
        "label_match": bool(event.get("overshoot_rebound_label", False)),
    }


def _run_long_only_variant(
    contexts: list[dict],
    *,
    downside_prev_close_min: float,
    start_minute: int,
    max_minute: int,
    rebound_from_trough_min: float,
    vwap_reclaim_min: float,
    rebound_impulse_min: float,
    long_exit_mode: str,
    long_vwap_touch_buffer: float,
    long_ema_roll_gain_min: float,
    long_max_hold_bars: int,
) -> tuple[LongOnlyResult, pd.DataFrame]:
    rows: list[dict] = []
    for context in contexts:
        row = _trade_long_only(
            context["event"],
            context["session_arrays"],
            downside_prev_close_min=downside_prev_close_min,
            start_minute=start_minute,
            max_minute=max_minute,
            rebound_from_trough_min=rebound_from_trough_min,
            vwap_reclaim_min=vwap_reclaim_min,
            rebound_impulse_min=rebound_impulse_min,
            long_exit_mode=long_exit_mode,
            long_vwap_touch_buffer=long_vwap_touch_buffer,
            long_ema_roll_gain_min=long_ema_roll_gain_min,
            long_max_hold_bars=long_max_hold_bars,
        )
        if row is not None:
            rows.append(row)
    trades_df = pd.DataFrame(rows)
    variant = (
        f"neg_long_t{downside_prev_close_min:g}"
        f"_start{int(start_minute)}"
        f"_end{int(max_minute)}"
        f"_reb{rebound_from_trough_min:g}"
        f"_vwap{vwap_reclaim_min:g}"
        f"_imp{rebound_impulse_min:g}"
        f"_exit{long_exit_mode}"
        f"_lvwap{long_vwap_touch_buffer:g}"
        f"_lgain{long_ema_roll_gain_min:g}"
    )
    cfg = {
        "downside_prev_close_min": downside_prev_close_min,
        "start_minute": start_minute,
        "max_minute": max_minute,
        "rebound_from_trough_min": rebound_from_trough_min,
        "vwap_reclaim_min": vwap_reclaim_min,
        "rebound_impulse_min": rebound_impulse_min,
        "long_exit_mode": long_exit_mode,
        "long_vwap_touch_buffer": long_vwap_touch_buffer,
        "long_ema_roll_gain_min": long_ema_roll_gain_min,
        "long_max_hold_bars": long_max_hold_bars,
    }
    if trades_df.empty:
        return LongOnlyResult(variant, 0, np.nan, np.nan, np.nan, 0.0, np.nan, cfg), trades_df
    rets = pd.to_numeric(trades_df["return_pct"], errors="coerce").dropna()
    result = LongOnlyResult(
        variant=variant,
        events=int(len(trades_df)),
        win_rate_pct=float((rets > 0).mean() * 100.0),
        mean_return_pct=float(rets.mean()),
        median_return_pct=float(rets.median()),
        compounded_return_pct=float(_compound_return_pct(rets)),
        labeled_match_pct=float(pd.to_numeric(trades_df["label_match"], errors="coerce").mean() * 100.0),
        params=cfg,
    )
    trades_df["variant"] = variant
    return result, trades_df


def _run_two_leg_variant(
    contexts: list[dict],
    *,
    base_cfg: dict,
    short_entry_window_bars: int,
    short_pullback_from_peak_min: float,
    short_vwap_break_max: float,
    short_downside_impulse_min: float,
    short_exit_mode: str,
    short_rebound_exit_pct: float,
    short_vwap_reclaim_buffer: float,
    short_max_hold_bars: int,
) -> tuple[TwoLegResult, pd.DataFrame]:
    rows: list[dict] = []
    for context in contexts:
        row = _trade_two_leg(
            context["event"],
            context["session_arrays"],
            downside_prev_close_min=base_cfg["downside_prev_close_min"],
            start_minute=base_cfg["start_minute"],
            max_minute=base_cfg["max_minute"],
            rebound_from_trough_min=base_cfg["rebound_from_trough_min"],
            vwap_reclaim_min=base_cfg["vwap_reclaim_min"],
            rebound_impulse_min=base_cfg["rebound_impulse_min"],
            long_exit_mode=base_cfg["long_exit_mode"],
            long_vwap_touch_buffer=base_cfg["long_vwap_touch_buffer"],
            long_ema_roll_gain_min=base_cfg["long_ema_roll_gain_min"],
            long_max_hold_bars=base_cfg["long_max_hold_bars"],
            short_entry_window_bars=short_entry_window_bars,
            short_pullback_from_peak_min=short_pullback_from_peak_min,
            short_vwap_break_max=short_vwap_break_max,
            short_downside_impulse_min=short_downside_impulse_min,
            short_exit_mode=short_exit_mode,
            short_rebound_exit_pct=short_rebound_exit_pct,
            short_vwap_reclaim_buffer=short_vwap_reclaim_buffer,
            short_max_hold_bars=short_max_hold_bars,
        )
        if row is not None:
            rows.append(row)
    events_df = pd.DataFrame(rows)
    variant = (
        f"{base_cfg['variant']}"
        f"_shortwin{short_entry_window_bars}"
        f"_spull{short_pullback_from_peak_min:g}"
        f"_svwap{short_vwap_break_max:g}"
        f"_simp{short_downside_impulse_min:g}"
        f"_sexit{short_exit_mode}"
        f"_sreb{short_rebound_exit_pct:g}"
    )
    cfg = {
        **base_cfg,
        "short_entry_window_bars": short_entry_window_bars,
        "short_pullback_from_peak_min": short_pullback_from_peak_min,
        "short_vwap_break_max": short_vwap_break_max,
        "short_downside_impulse_min": short_downside_impulse_min,
        "short_exit_mode": short_exit_mode,
        "short_rebound_exit_pct": short_rebound_exit_pct,
        "short_vwap_reclaim_buffer": short_vwap_reclaim_buffer,
        "short_max_hold_bars": short_max_hold_bars,
    }
    if events_df.empty:
        return TwoLegResult(variant, 0, 0, 0, np.nan, np.nan, np.nan, 0.0, np.nan, np.nan, cfg), events_df
    event_rets = pd.to_numeric(events_df["event_return_pct"], errors="coerce").dropna()
    all_trade_rets = pd.concat(
        [
            pd.to_numeric(events_df["long_return_pct"], errors="coerce"),
            pd.to_numeric(events_df["short_return_pct"], errors="coerce"),
        ],
        ignore_index=True,
    ).dropna()
    result = TwoLegResult(
        variant=variant,
        events=int(len(events_df)),
        total_trades=int(pd.to_numeric(events_df["total_trades"], errors="coerce").sum()),
        short_leg_events=int(events_df["short_entry_time"].notna().sum()),
        win_rate_pct=float((event_rets > 0).mean() * 100.0),
        mean_event_return_pct=float(event_rets.mean()),
        median_event_return_pct=float(event_rets.median()),
        compounded_event_return_pct=float(_compound_return_pct(event_rets)),
        mean_trade_return_pct=float(all_trade_rets.mean()) if not all_trade_rets.empty else np.nan,
        labeled_match_pct=float(pd.to_numeric(events_df["label_match"], errors="coerce").mean() * 100.0),
        params=cfg,
    )
    events_df["variant"] = variant
    return result, events_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a mirrored negative-earnings overshoot / rebound family."
    )
    parser.add_argument("--start", default="2024-04-04", help="Event-study start date (default: 2024-04-04).")
    parser.add_argument("--end", default="2026-05-01", help="Event-study end date (default: 2026-05-01).")
    parser.add_argument(
        "--artifact-stem",
        default="earnings_negative_rebound_family",
        help="Artifact stem (default: earnings_negative_rebound_family).",
    )
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    events = _load_negative_events(start, end)
    contexts = _build_event_contexts(events)

    long_results: list[LongOnlyResult] = []
    long_rows: list[pd.DataFrame] = []
    for downside_prev_close_min in (7.5, 10.0, 12.5, 15.0):
        for start_minute in (11 * 60 + 30, 12 * 60, 13 * 60):
            for max_minute in (14 * 60, 14 * 60 + 30, 15 * 60):
                if max_minute <= start_minute:
                    continue
                for rebound_from_trough_min in (2.0, 3.0, 4.0):
                    for vwap_reclaim_min in (-0.5, 0.0, 0.5):
                        for rebound_impulse_min in (0.5, 1.0):
                            for long_exit_mode in ("time_10", "time_20", "ema_roll", "vwap_touch"):
                                for long_vwap_touch_buffer in (0.5, 1.0):
                                    for long_ema_roll_gain_min in (1.0, 2.0):
                                        result, rows = _run_long_only_variant(
                                            contexts,
                                            downside_prev_close_min=downside_prev_close_min,
                                            start_minute=start_minute,
                                            max_minute=max_minute,
                                            rebound_from_trough_min=rebound_from_trough_min,
                                            vwap_reclaim_min=vwap_reclaim_min,
                                            rebound_impulse_min=rebound_impulse_min,
                                            long_exit_mode=long_exit_mode,
                                            long_vwap_touch_buffer=long_vwap_touch_buffer,
                                            long_ema_roll_gain_min=long_ema_roll_gain_min,
                                            long_max_hold_bars=20,
                                        )
                                        long_results.append(result)
                                        if not rows.empty:
                                            long_rows.append(rows)

    long_df = pd.DataFrame([asdict(r) for r in long_results]).sort_values(
        ["compounded_return_pct", "win_rate_pct", "events"],
        ascending=[False, False, False],
    )
    best_long = long_df[long_df["events"] >= 10].head(1)
    best_long_label = str(best_long.iloc[0]["variant"]) if not best_long.empty else None
    long_events_df = pd.concat(long_rows, ignore_index=True) if long_rows else pd.DataFrame()
    best_long_events = (
        long_events_df[long_events_df["variant"] == best_long_label].copy()
        if best_long_label is not None and not long_events_df.empty
        else pd.DataFrame()
    )

    two_leg_results: list[TwoLegResult] = []
    two_leg_rows: list[pd.DataFrame] = []
    if best_long_label is not None:
        chosen_params = next((dict(r.params) | {"variant": r.variant} for r in long_results if r.variant == best_long_label), None)
        if chosen_params is not None:
            for short_entry_window_bars in (8, 10, 12):
                for short_pullback_from_peak_min in (1.0, 1.5, 2.0):
                    for short_vwap_break_max in (0.0, -0.5):
                        for short_downside_impulse_min in (0.5, 1.0):
                            for short_exit_mode in ("time_10", "rebound", "ema_turn"):
                                for short_rebound_exit_pct in (0.5, 1.0):
                                    result, rows = _run_two_leg_variant(
                                        contexts,
                                        base_cfg=chosen_params,
                                        short_entry_window_bars=short_entry_window_bars,
                                        short_pullback_from_peak_min=short_pullback_from_peak_min,
                                        short_vwap_break_max=short_vwap_break_max,
                                        short_downside_impulse_min=short_downside_impulse_min,
                                        short_exit_mode=short_exit_mode,
                                        short_rebound_exit_pct=short_rebound_exit_pct,
                                        short_vwap_reclaim_buffer=0.0,
                                        short_max_hold_bars=15,
                                    )
                                    two_leg_results.append(result)
                                    if not rows.empty:
                                        two_leg_rows.append(rows)

    two_leg_df = (
        pd.DataFrame([asdict(r) for r in two_leg_results]).sort_values(
            ["compounded_event_return_pct", "win_rate_pct", "events"],
            ascending=[False, False, False],
        )
        if two_leg_results
        else pd.DataFrame()
    )

    summary = {
        "window": {"start": start.date().isoformat(), "end": end.date().isoformat()},
        "negative_events": int(len(events)),
        "labeled_rebound_events": int((events["overshoot_rebound_label"] == True).sum()),
        "best_long_only_min10": long_df[long_df["events"] >= 10].head(10).to_dict(orient="records")
        if not long_df.empty
        else [],
        "best_two_leg_min8": two_leg_df[two_leg_df["events"] >= 8].head(10).to_dict(orient="records")
        if not two_leg_df.empty
        else [],
    }

    out_summary = ARTIFACT_DIR / f"{args.artifact_stem}_summary.json"
    out_long = ARTIFACT_DIR / f"{args.artifact_stem}_long_variants.csv"
    out_long_events = ARTIFACT_DIR / f"{args.artifact_stem}_best_long_events.csv"
    out_two_leg = ARTIFACT_DIR / f"{args.artifact_stem}_two_leg_variants.csv"
    out_two_leg_events = ARTIFACT_DIR / f"{args.artifact_stem}_best_two_leg_events.csv"

    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    long_df.to_csv(out_long, index=False)
    best_long_events.to_csv(out_long_events, index=False)
    two_leg_df.to_csv(out_two_leg, index=False)
    if two_leg_rows:
        all_two_leg = pd.concat(two_leg_rows, ignore_index=True)
        best_two_leg_label = None
        if not two_leg_df.empty:
            filt = two_leg_df[two_leg_df["events"] >= 8]
            if not filt.empty:
                best_two_leg_label = str(filt.iloc[0]["variant"])
        best_two_leg_events = (
            all_two_leg[all_two_leg["variant"] == best_two_leg_label].copy()
            if best_two_leg_label is not None
            else pd.DataFrame()
        )
        best_two_leg_events.to_csv(out_two_leg_events, index=False)
    else:
        pd.DataFrame().to_csv(out_two_leg_events, index=False)

    if not long_df.empty:
        top_long = long_df[long_df["events"] >= 10].head(1)
        if not top_long.empty:
            r = top_long.iloc[0]
            print(
                f"best_long_only: {r['variant']} "
                f"events={int(r['events'])} compounded={float(r['compounded_return_pct']):.3f}%"
            )
    if not two_leg_df.empty:
        top_two = two_leg_df[two_leg_df["events"] >= 8].head(1)
        if not top_two.empty:
            r = top_two.iloc[0]
            print(
                f"best_two_leg: {r['variant']} "
                f"events={int(r['events'])} compounded={float(r['compounded_event_return_pct']):.3f}%"
            )
    print(
        "Artifacts written:\n"
        f"- {out_summary}\n"
        f"- {out_long}\n"
        f"- {out_long_events}\n"
        f"- {out_two_leg}\n"
        f"- {out_two_leg_events}"
    )


if __name__ == "__main__":
    main()
