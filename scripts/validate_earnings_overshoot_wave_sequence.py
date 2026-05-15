"""
Validate a two-leg wave strategy after positive off-hours earnings overshoots.

Sequence:
1. Short the first dump leg after a large overshoot above the prior close.
2. Cover the short on early exhaustion / rebound.
3. Optionally reverse long into the rebound.

This is the next research branch after the single-leg first-dump strategy.
"""
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

from scripts.validate_earnings_overshoot_first_dump import (  # noqa: E402
    _build_event_contexts,
    _compound_return_pct,
    _find_entry_idx,
    _load_labeled_events,
    _safe_pct,
)


ARTIFACT_DIR = ROOT / "artifacts" / "optimization"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class VariantResult:
    variant: str
    events: int
    total_trades: int
    second_leg_events: int
    win_rate_pct: float
    mean_event_return_pct: float
    median_event_return_pct: float
    compounded_event_return_pct: float
    mean_trade_return_pct: float
    labeled_overshoot_match_pct: float


def _exit_short(
    arrays: dict,
    *,
    entry_idx: int,
    exit_mode: str,
    rebound_exit_pct: float,
    vwap_reclaim_buffer: float,
    max_hold_bars: int,
) -> tuple[int, str]:
    close = arrays["close"]
    close_from_vwap = arrays["close_from_vwap_pct"]
    ema_fast = arrays["ema_fast"]
    ema_slow = arrays["ema_slow"]
    ema_fast_slope = arrays["ema_fast_slope"]
    n = len(close)

    if exit_mode.startswith("time_"):
        hold_bars = int(exit_mode.split("_", 1)[1])
        exit_idx = min(entry_idx + hold_bars, n - 1)
        return exit_idx, f"time_{hold_bars}"

    if exit_mode == "rebound":
        post_low = float(close[entry_idx])
        for idx in range(entry_idx + 1, min(entry_idx + max_hold_bars + 1, n)):
            if np.isfinite(close[idx]):
                post_low = min(post_low, float(close[idx]))
                rebound = _safe_pct(float(close[idx]), post_low)
                if np.isfinite(rebound) and rebound >= rebound_exit_pct:
                    return idx, "rebound"
        return min(entry_idx + max_hold_bars, n - 1), "rebound_timeout"

    if exit_mode == "ema_turn":
        post_low = float(close[entry_idx])
        for idx in range(entry_idx + 1, min(entry_idx + max_hold_bars + 1, n)):
            if not np.isfinite(close[idx]):
                continue
            post_low = min(post_low, float(close[idx]))
            rebound = _safe_pct(float(close[idx]), post_low)
            if (
                np.isfinite(rebound)
                and rebound >= rebound_exit_pct
                and np.isfinite(ema_fast[idx])
                and np.isfinite(ema_fast_slope[idx])
                and close[idx] >= ema_fast[idx]
                and ema_fast_slope[idx] > 0
            ):
                return idx, "ema_turn"
        return min(entry_idx + max_hold_bars, n - 1), "ema_turn_timeout"

    if exit_mode == "vwap_reclaim":
        for idx in range(entry_idx + 1, min(entry_idx + max_hold_bars + 1, n)):
            if np.isfinite(close_from_vwap[idx]) and close_from_vwap[idx] >= vwap_reclaim_buffer:
                return idx, "vwap_reclaim"
        return min(entry_idx + max_hold_bars, n - 1), "vwap_timeout"

    if exit_mode == "ema_cross":
        post_low = float(close[entry_idx])
        for idx in range(entry_idx + 1, min(entry_idx + max_hold_bars + 1, n)):
            if not np.isfinite(close[idx]):
                continue
            post_low = min(post_low, float(close[idx]))
            rebound = _safe_pct(float(close[idx]), post_low)
            if (
                np.isfinite(rebound)
                and rebound >= rebound_exit_pct
                and np.isfinite(ema_fast[idx])
                and np.isfinite(ema_slow[idx])
                and np.isfinite(ema_fast_slope[idx])
                and ema_fast[idx] >= ema_slow[idx]
                and ema_fast_slope[idx] > 0
            ):
                return idx, "ema_cross"
        return min(entry_idx + max_hold_bars, n - 1), "ema_cross_timeout"

    raise ValueError(f"Unknown short exit mode: {exit_mode}")


def _find_long_entry_idx(
    arrays: dict,
    *,
    start_idx: int,
    entry_mode: str,
    long_momentum_min: float,
    max_entry_bars: int,
) -> tuple[int | None, str | None]:
    close = arrays["close"]
    high = arrays["high"]
    ema_fast = arrays["ema_fast"]
    ema_fast_slope = arrays["ema_fast_slope"]
    ret_5m = arrays["ret_5m_pct"]
    n = len(close)
    end_idx = min(start_idx + max_entry_bars, n - 1)

    if start_idx >= n - 1:
        return None, None

    if entry_mode == "next_bar":
        idx = start_idx + 1
        if idx <= end_idx and np.isfinite(close[idx]):
            return idx, "next_bar"
        return None, None

    for idx in range(start_idx + 1, end_idx + 1):
        if not np.isfinite(close[idx]):
            continue
        momentum_ok = (not np.isfinite(ret_5m[idx])) or (ret_5m[idx] >= long_momentum_min)
        if not momentum_ok:
            continue

        if entry_mode == "ema_turn":
            if np.isfinite(ema_fast[idx]) and np.isfinite(ema_fast_slope[idx]) and close[idx] >= ema_fast[idx] and ema_fast_slope[idx] > 0:
                return idx, "ema_turn"
        elif entry_mode == "break_prev_high":
            lookback_start = max(start_idx + 1, idx - 3)
            prev_high = float(np.nanmax(high[lookback_start:idx])) if idx > lookback_start else np.nan
            if np.isfinite(prev_high) and close[idx] > prev_high:
                return idx, "break_prev_high"
        else:
            raise ValueError(f"Unknown long entry mode: {entry_mode}")

    return None, None


def _exit_long(
    arrays: dict,
    *,
    entry_idx: int,
    exit_mode: str,
    vwap_touch_buffer: float,
    ema_roll_gain_min: float,
    max_hold_bars: int,
) -> tuple[int, str]:
    close = arrays["close"]
    close_from_vwap = arrays["close_from_vwap_pct"]
    ema_fast = arrays["ema_fast"]
    ema_fast_slope = arrays["ema_fast_slope"]
    n = len(close)

    if exit_mode.startswith("time_"):
        hold_bars = int(exit_mode.split("_", 1)[1])
        exit_idx = min(entry_idx + hold_bars, n - 1)
        return exit_idx, f"time_{hold_bars}"

    if exit_mode == "vwap_touch":
        for idx in range(entry_idx + 1, min(entry_idx + max_hold_bars + 1, n)):
            if np.isfinite(close_from_vwap[idx]) and close_from_vwap[idx] >= vwap_touch_buffer:
                return idx, "vwap_touch"
        return min(entry_idx + max_hold_bars, n - 1), "vwap_touch_timeout"

    if exit_mode == "ema_roll":
        peak = float(close[entry_idx])
        for idx in range(entry_idx + 1, min(entry_idx + max_hold_bars + 1, n)):
            if not np.isfinite(close[idx]):
                continue
            peak = max(peak, float(close[idx]))
            gain = _safe_pct(float(close[idx]), float(close[entry_idx]))
            if (
                np.isfinite(gain)
                and gain >= ema_roll_gain_min
                and np.isfinite(ema_fast[idx])
                and np.isfinite(ema_fast_slope[idx])
                and close[idx] <= ema_fast[idx]
                and ema_fast_slope[idx] < 0
            ):
                return idx, "ema_roll"
        return min(entry_idx + max_hold_bars, n - 1), "ema_roll_timeout"

    raise ValueError(f"Unknown long exit mode: {exit_mode}")


def _trade_event(
    event: dict,
    arrays: dict,
    *,
    overshoot_prev_close_min: float,
    start_minute: int,
    max_minute: int,
    peak_pullback_min: float,
    vwap_break_min: float,
    breakdown_impulse_min: float,
    short_exit_mode: str,
    short_rebound_exit_pct: float,
    short_vwap_reclaim_buffer: float,
    short_max_hold_bars: int,
    long_entry_mode: str,
    long_momentum_min: float,
    long_entry_window_bars: int,
    long_exit_mode: str,
    long_vwap_touch_buffer: float,
    long_ema_roll_gain_min: float,
    long_max_hold_bars: int,
) -> dict | None:
    short_entry_idx = _find_entry_idx(
        event,
        arrays,
        overshoot_prev_close_min=overshoot_prev_close_min,
        start_minute=start_minute,
        max_minute=max_minute,
        peak_pullback_min=peak_pullback_min,
        vwap_break_min=vwap_break_min,
        breakdown_impulse_min=breakdown_impulse_min,
    )
    if short_entry_idx is None:
        return None

    short_exit_idx, short_exit_reason = _exit_short(
        arrays,
        entry_idx=short_entry_idx,
        exit_mode=short_exit_mode,
        rebound_exit_pct=short_rebound_exit_pct,
        vwap_reclaim_buffer=short_vwap_reclaim_buffer,
        max_hold_bars=short_max_hold_bars,
    )

    date = arrays["date"]
    close = arrays["close"]
    short_entry_px = float(close[short_entry_idx])
    short_exit_px = float(close[short_exit_idx])
    short_ret = -_safe_pct(short_exit_px, short_entry_px)

    long_entry_idx, long_entry_reason = _find_long_entry_idx(
        arrays,
        start_idx=short_exit_idx,
        entry_mode=long_entry_mode,
        long_momentum_min=long_momentum_min,
        max_entry_bars=long_entry_window_bars,
    )

    long_exit_idx = None
    long_exit_reason = None
    long_entry_px = np.nan
    long_exit_px = np.nan
    long_ret = 0.0

    if long_entry_idx is not None:
        long_exit_idx, long_exit_reason = _exit_long(
            arrays,
            entry_idx=long_entry_idx,
            exit_mode=long_exit_mode,
            vwap_touch_buffer=long_vwap_touch_buffer,
            ema_roll_gain_min=long_ema_roll_gain_min,
            max_hold_bars=long_max_hold_bars,
        )
        long_entry_px = float(close[long_entry_idx])
        long_exit_px = float(close[long_exit_idx])
        long_ret = _safe_pct(long_exit_px, long_entry_px)

    event_ret = (1.0 + short_ret / 100.0) * (1.0 + long_ret / 100.0) - 1.0
    total_trades = 1 + int(long_entry_idx is not None)

    return {
        "symbol": event["symbol"],
        "reaction_date": pd.Timestamp(event["reaction_date"]).date().isoformat(),
        "timing": event.get("timing"),
        "surprise_pct": float(event.get("surprise_pct", np.nan)),
        "gap_pct": float(event.get("gap_pct", np.nan)),
        "peak_vs_prev_close_pct": float(event.get("peak_vs_prev_close_pct", np.nan)),
        "short_entry_time": pd.Timestamp(date[short_entry_idx]).isoformat(),
        "short_entry_px": short_entry_px,
        "short_exit_time": pd.Timestamp(date[short_exit_idx]).isoformat(),
        "short_exit_px": short_exit_px,
        "short_exit_reason": short_exit_reason,
        "short_return_pct": short_ret,
        "long_entry_time": pd.Timestamp(date[long_entry_idx]).isoformat() if long_entry_idx is not None else None,
        "long_entry_px": long_entry_px,
        "long_entry_reason": long_entry_reason,
        "long_exit_time": pd.Timestamp(date[long_exit_idx]).isoformat() if long_exit_idx is not None else None,
        "long_exit_px": long_exit_px,
        "long_exit_reason": long_exit_reason,
        "long_return_pct": long_ret if long_entry_idx is not None else np.nan,
        "event_return_pct": event_ret * 100.0,
        "total_trades": total_trades,
        "label_match": bool(event.get("overshoot_dump_label", False)),
    }


def _run_variant(
    contexts: list[dict],
    *,
    overshoot_prev_close_min: float,
    start_minute: int,
    max_minute: int,
    peak_pullback_min: float,
    vwap_break_min: float,
    breakdown_impulse_min: float,
    short_exit_mode: str,
    short_rebound_exit_pct: float,
    short_vwap_reclaim_buffer: float,
    short_max_hold_bars: int,
    long_entry_mode: str,
    long_momentum_min: float,
    long_entry_window_bars: int,
    long_exit_mode: str,
    long_vwap_touch_buffer: float,
    long_ema_roll_gain_min: float,
    long_max_hold_bars: int,
) -> tuple[VariantResult, pd.DataFrame]:
    event_rows: list[dict] = []
    for context in contexts:
        event = context["event"]
        peak_vs_prev = float(event.get("peak_vs_prev_close_pct", np.nan))
        if not np.isfinite(peak_vs_prev) or peak_vs_prev < overshoot_prev_close_min:
            continue
        row = _trade_event(
            event,
            context["session_arrays"],
            overshoot_prev_close_min=overshoot_prev_close_min,
            start_minute=start_minute,
            max_minute=max_minute,
            peak_pullback_min=peak_pullback_min,
            vwap_break_min=vwap_break_min,
            breakdown_impulse_min=breakdown_impulse_min,
            short_exit_mode=short_exit_mode,
            short_rebound_exit_pct=short_rebound_exit_pct,
            short_vwap_reclaim_buffer=short_vwap_reclaim_buffer,
            short_max_hold_bars=short_max_hold_bars,
            long_entry_mode=long_entry_mode,
            long_momentum_min=long_momentum_min,
            long_entry_window_bars=long_entry_window_bars,
            long_exit_mode=long_exit_mode,
            long_vwap_touch_buffer=long_vwap_touch_buffer,
            long_ema_roll_gain_min=long_ema_roll_gain_min,
            long_max_hold_bars=long_max_hold_bars,
        )
        if row is not None:
            event_rows.append(row)

    events_df = pd.DataFrame(event_rows)
    variant = (
        f"wave_pc{int(overshoot_prev_close_min)}"
        f"_start{int(start_minute)}"
        f"_end{int(max_minute)}"
        f"_pull{peak_pullback_min:.1f}"
        f"_vbreak{vwap_break_min:.1f}"
        f"_bd{breakdown_impulse_min:.1f}"
        f"_sexit{short_exit_mode}"
        f"_sreb{short_rebound_exit_pct:.1f}"
        f"_lentry{long_entry_mode}"
        f"_lmom{long_momentum_min:.2f}"
        f"_lexit{long_exit_mode}"
        f"_lvwap{long_vwap_touch_buffer:.1f}"
        f"_lgain{long_ema_roll_gain_min:.1f}"
    )

    if events_df.empty:
        return VariantResult(variant, 0, 0, 0, np.nan, np.nan, np.nan, 0.0, np.nan, np.nan), events_df

    event_rets = pd.to_numeric(events_df["event_return_pct"], errors="coerce").dropna()
    short_rets = pd.to_numeric(events_df["short_return_pct"], errors="coerce").dropna()
    long_rets = pd.to_numeric(events_df["long_return_pct"], errors="coerce").dropna()
    all_trade_rets = pd.concat([short_rets, long_rets], ignore_index=True)
    match_pct = float(pd.to_numeric(events_df["label_match"], errors="coerce").mean() * 100.0)
    result = VariantResult(
        variant=variant,
        events=int(len(events_df)),
        total_trades=int(pd.to_numeric(events_df["total_trades"], errors="coerce").sum()),
        second_leg_events=int(events_df["long_entry_time"].notna().sum()),
        win_rate_pct=float((event_rets > 0).mean() * 100.0),
        mean_event_return_pct=float(event_rets.mean()),
        median_event_return_pct=float(event_rets.median()),
        compounded_event_return_pct=float(_compound_return_pct(event_rets)),
        mean_trade_return_pct=float(all_trade_rets.mean()) if not all_trade_rets.empty else np.nan,
        labeled_overshoot_match_pct=match_pct,
    )
    events_df["variant"] = variant
    return result, events_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a two-leg earnings overshoot wave strategy.")
    parser.add_argument("--start", default="2024-04-04", help="Event-study start date (default: 2024-04-04).")
    parser.add_argument("--end", default="2026-05-01", help="Event-study end date (default: 2026-05-01).")
    parser.add_argument(
        "--artifact-stem",
        default="earnings_overshoot_wave_sequence",
        help="Artifact stem (default: earnings_overshoot_wave_sequence).",
    )
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    events = _load_labeled_events(start, end)
    contexts = _build_event_contexts(events)

    variants_path = ARTIFACT_DIR / f"{args.artifact_stem}_variants.csv"
    events_path = ARTIFACT_DIR / f"{args.artifact_stem}_top_events.csv"
    summary_path = ARTIFACT_DIR / f"{args.artifact_stem}_summary.json"

    variant_results: list[VariantResult] = []
    variant_event_rows: list[pd.DataFrame] = []
    for overshoot_prev_close in (15.0, 20.0):
        for start_minute in (12 * 60, 13 * 60):
            for max_minute in (14 * 60, 14 * 60 + 30):
                if max_minute <= start_minute:
                    continue
                for peak_pullback_min in (3.0, 4.0):
                    for vwap_break_min in (0.5, 1.0):
                        for breakdown_impulse_min in (0.5, 1.0):
                            for short_exit_mode in ("rebound", "ema_turn", "time_10"):
                                for short_rebound_exit_pct in (0.5, 1.0):
                                    for long_entry_mode in ("next_bar", "ema_turn", "break_prev_high"):
                                        for long_momentum_min in (0.0, 0.25):
                                            for long_exit_mode in ("time_10", "time_20", "vwap_touch", "ema_roll"):
                                                for long_vwap_touch_buffer in (-0.5, 0.0):
                                                    for long_ema_roll_gain_min in (0.5, 1.0):
                                                        result, event_rows = _run_variant(
                                                            contexts,
                                                            overshoot_prev_close_min=overshoot_prev_close,
                                                            start_minute=start_minute,
                                                            max_minute=max_minute,
                                                            peak_pullback_min=peak_pullback_min,
                                                            vwap_break_min=vwap_break_min,
                                                            breakdown_impulse_min=breakdown_impulse_min,
                                                            short_exit_mode=short_exit_mode,
                                                            short_rebound_exit_pct=short_rebound_exit_pct,
                                                            short_vwap_reclaim_buffer=0.0,
                                                            short_max_hold_bars=20,
                                                            long_entry_mode=long_entry_mode,
                                                            long_momentum_min=long_momentum_min,
                                                            long_entry_window_bars=10,
                                                            long_exit_mode=long_exit_mode,
                                                            long_vwap_touch_buffer=long_vwap_touch_buffer,
                                                            long_ema_roll_gain_min=long_ema_roll_gain_min,
                                                            long_max_hold_bars=20,
                                                        )
                                                        variant_results.append(result)
                                                        if not event_rows.empty:
                                                            variant_event_rows.append(event_rows)

    variants_df = pd.DataFrame([asdict(row) for row in variant_results]).sort_values(
        ["compounded_event_return_pct", "mean_event_return_pct", "events"],
        ascending=[False, False, False],
    )
    variants_df.to_csv(variants_path, index=False)

    top_variant = variants_df.iloc[0].to_dict() if not variants_df.empty else {}
    top_events = pd.DataFrame()
    if variant_event_rows and top_variant:
        all_events = pd.concat(variant_event_rows, ignore_index=True)
        top_events = all_events[all_events["variant"] == top_variant["variant"]].copy()
    top_events.to_csv(events_path, index=False)

    summary = {
        "window": {"start": start.date().isoformat(), "end": end.date().isoformat()},
        "positive_events": int(len(events)),
        "positive_peak_vs_prev_close_ge_15": int((pd.to_numeric(events["peak_vs_prev_close_pct"], errors="coerce") >= 15.0).sum()),
        "positive_peak_vs_prev_close_ge_20": int((pd.to_numeric(events["peak_vs_prev_close_pct"], errors="coerce") >= 20.0).sum()),
        "best_variant": top_variant,
        "top_variants": variants_df.head(10).to_dict(orient="records"),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Positive events: {len(events)}")
    if top_variant:
        print("Best variant:")
        print(json.dumps(top_variant, indent=2))
    print(f"Artifacts written:\n- {variants_path}\n- {events_path}\n- {summary_path}")


if __name__ == "__main__":
    main()
