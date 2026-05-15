"""
Validate a narrower earnings-overshoot short strategy:

- positive off-hours earnings surprise
- intraday overshoot reaches >= X% above the prior regular-session close
- price breaks below anchored VWAP
- a rebound attempt fails to reclaim anchored VWAP / make a fresh high
- price breaks back down and is shorted into the close

This is a more surgical follow-up to the broader earnings overshoot study.
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

ARTIFACT_DIR = ROOT / "artifacts" / "optimization"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
LABELED_EVENTS_PATH = ARTIFACT_DIR / "earnings_overshoot_dump_events_labeled.csv"
ALPACA_CACHE_DIR = ROOT / "data_cache" / "alpaca"
RTH_OPEN = 9 * 60 + 30
RTH_CLOSE = 16 * 60
_SESSION_ARRAY_CACHE: dict[tuple[str, str], dict | None] = {}


@dataclass
class VariantResult:
    variant: str
    trades: int
    win_rate_pct: float
    mean_return_pct: float
    median_return_pct: float
    compounded_return_pct: float
    labeled_overshoot_match_pct: float


def _safe_pct(current, base) -> float:
    try:
        cur = float(current)
        ref = float(base)
    except Exception:
        return np.nan
    if not np.isfinite(cur) or not np.isfinite(ref) or ref == 0:
        return np.nan
    return (cur / ref - 1.0) * 100.0


def _compound_return_pct(returns_pct: pd.Series, starting_equity: float = 1000.0) -> float:
    clean = pd.to_numeric(returns_pct, errors="coerce").dropna()
    if clean.empty:
        return 0.0
    equity = float(starting_equity)
    for ret in clean:
        equity *= 1.0 + float(ret) / 100.0
    return (equity / float(starting_equity) - 1.0) * 100.0


def _load_labeled_events(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if not LABELED_EVENTS_PATH.exists():
        raise SystemExit(
            f"Missing labeled overshoot events at {LABELED_EVENTS_PATH}. "
            "Run validate_earnings_overshoot_dump.py first."
        )
    df = pd.read_csv(LABELED_EVENTS_PATH)
    if df.empty:
        raise SystemExit("The labeled overshoot event table is empty.")
    df["reaction_date"] = pd.to_datetime(df["reaction_date"], errors="coerce")
    df = df[(df["reaction_date"] >= start) & (df["reaction_date"] <= end)].copy()
    df = df[pd.to_numeric(df["surprise_pct"], errors="coerce") > 0].copy()
    return df.reset_index(drop=True)


def _session_key(symbol: str, reaction_date: pd.Timestamp) -> tuple[str, str]:
    return str(symbol).upper(), pd.Timestamp(reaction_date).date().isoformat()


def _load_session_arrays(symbol: str, reaction_date: pd.Timestamp) -> dict | None:
    key = _session_key(symbol, reaction_date)
    if key in _SESSION_ARRAY_CACHE:
        return _SESSION_ARRAY_CACHE[key]

    path = ALPACA_CACHE_DIR / str(symbol).upper() / "1Min.csv"
    if not path.exists():
        _SESSION_ARRAY_CACHE[key] = None
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
        _SESSION_ARRAY_CACHE[key] = None
        return None

    session = session.sort_values("date").reset_index(drop=True)
    typical = (session["high"] + session["low"] + session["close"]) / 3.0
    vol = pd.to_numeric(session["volume"], errors="coerce").fillna(0.0)
    cum_vol = vol.cumsum().replace(0.0, np.nan)
    anchored_vwap = (typical * vol).cumsum() / cum_vol

    arrays = {
        "date": session["date"].to_numpy(copy=True),
        "minutes_et": pd.to_numeric(session["minutes_et"], errors="coerce").to_numpy(dtype=np.int32, copy=True),
        "close": pd.to_numeric(session["close"], errors="coerce").to_numpy(dtype=np.float64, copy=True),
        "high": pd.to_numeric(session["high"], errors="coerce").to_numpy(dtype=np.float64, copy=True),
        "low": pd.to_numeric(session["low"], errors="coerce").to_numpy(dtype=np.float64, copy=True),
        "anchored_vwap": pd.to_numeric(anchored_vwap, errors="coerce").to_numpy(dtype=np.float64, copy=True),
    }
    arrays["ret_5m_pct"] = pd.Series(arrays["close"]).pct_change(5).to_numpy(dtype=np.float64, copy=True) * 100.0
    arrays["close_from_vwap_pct"] = (arrays["close"] / arrays["anchored_vwap"] - 1.0) * 100.0
    arrays["high_from_vwap_pct"] = (arrays["high"] / arrays["anchored_vwap"] - 1.0) * 100.0
    arrays["running_peak_high"] = np.maximum.accumulate(arrays["high"])
    _SESSION_ARRAY_CACHE[key] = arrays
    return arrays


def _build_event_contexts(events: pd.DataFrame) -> list[dict]:
    contexts: list[dict] = []
    for event in events.to_dict(orient="records"):
        arrays = _load_session_arrays(str(event["symbol"]).upper(), pd.Timestamp(event["reaction_date"]))
        if arrays is None:
            continue
        contexts.append({"event": event, "session_arrays": arrays})
    return contexts


def _find_failed_reclaim_trigger(
    event: dict,
    arrays: dict,
    *,
    overshoot_prev_close_min: float,
    start_minute: int,
    max_minute: int,
    vwap_break_min: float,
    rebound_min: float,
    reclaim_window_bars: int,
    reclaim_vwap_buffer: float,
    lower_high_min: float,
    breakdown_impulse_min: float,
) -> dict | None:
    prev_close = float(event.get("prev_close", np.nan))
    if not np.isfinite(prev_close) or prev_close <= 0:
        return None

    minutes = arrays["minutes_et"]
    date = arrays["date"]
    close = arrays["close"]
    high = arrays["high"]
    low = arrays["low"]
    close_from_vwap = arrays["close_from_vwap_pct"]
    high_from_vwap = arrays["high_from_vwap_pct"]
    ret_5m = arrays["ret_5m_pct"]
    running_peak = arrays["running_peak_high"]
    running_peak_vs_prev = (running_peak / prev_close - 1.0) * 100.0

    candidate_mask = (
        (minutes >= start_minute)
        & (minutes <= max_minute)
        & np.isfinite(close)
        & np.isfinite(low)
        & np.isfinite(high)
        & np.isfinite(close_from_vwap)
        & np.isfinite(high_from_vwap)
        & np.isfinite(ret_5m)
        & np.isfinite(running_peak_vs_prev)
        & (running_peak_vs_prev >= overshoot_prev_close_min)
        & (close_from_vwap <= -vwap_break_min)
    )
    candidate_idxs = np.flatnonzero(candidate_mask)
    if candidate_idxs.size == 0:
        return None

    for break_idx in candidate_idxs:
        break_low = float(low[break_idx])
        break_close = float(close[break_idx])
        peak_ref = float(running_peak[break_idx])
        rebound_end = min(break_idx + reclaim_window_bars, len(close) - 1)
        if rebound_end <= break_idx:
            continue

        rebound_slice = slice(break_idx + 1, rebound_end + 1)
        rebound_high = high[rebound_slice]
        rebound_close = close[rebound_slice]
        rebound_high_from_vwap = high_from_vwap[rebound_slice]
        rebound_minutes = minutes[rebound_slice]

        if rebound_high.size == 0:
            continue

        max_rebound_high = float(np.nanmax(rebound_high))
        max_rebound_high_from_vwap = float(np.nanmax(rebound_high_from_vwap))
        rebound_size = _safe_pct(max_rebound_high, break_close)
        lower_high_from_peak = -_safe_pct(max_rebound_high, peak_ref)

        if not np.isfinite(rebound_size) or rebound_size < rebound_min:
            continue
        if not np.isfinite(max_rebound_high_from_vwap) or max_rebound_high_from_vwap > reclaim_vwap_buffer:
            continue
        if not np.isfinite(lower_high_from_peak) or lower_high_from_peak < lower_high_min:
            continue

        for confirm_idx in range(rebound_end + 1, len(close)):
            minute = int(minutes[confirm_idx])
            if minute > max_minute:
                break
            if not np.isfinite(close[confirm_idx]) or not np.isfinite(ret_5m[confirm_idx]):
                continue
            if close[confirm_idx] < break_low and ret_5m[confirm_idx] <= -breakdown_impulse_min:
                exit_px = float(close[-1])
                entry_px = float(close[confirm_idx])
                ret = -_safe_pct(exit_px, entry_px)
                return {
                    "symbol": event["symbol"],
                    "reaction_date": pd.Timestamp(event["reaction_date"]).date().isoformat(),
                    "timing": event.get("timing"),
                    "direction": "short",
                    "surprise_pct": float(event.get("surprise_pct", np.nan)),
                    "gap_pct": float(event.get("gap_pct", np.nan)),
                    "peak_vs_prev_close_pct": float(event.get("peak_vs_prev_close_pct", np.nan)),
                    "entry_time": pd.Timestamp(date[confirm_idx]).isoformat(),
                    "entry_px": entry_px,
                    "exit_px": exit_px,
                    "return_pct": ret,
                    "break_time": pd.Timestamp(date[break_idx]).isoformat(),
                    "break_close_from_vwap_pct": float(close_from_vwap[break_idx]),
                    "rebound_size_pct": rebound_size,
                    "rebound_max_high_from_vwap_pct": max_rebound_high_from_vwap,
                    "rebound_lower_high_from_peak_pct": lower_high_from_peak,
                    "label_match": bool(event.get("overshoot_dump_label", False)),
                }
    return None


def _run_variant(
    contexts: list[dict],
    *,
    overshoot_prev_close_min: float,
    start_minute: int,
    max_minute: int,
    vwap_break_min: float,
    rebound_min: float,
    reclaim_window_bars: int,
    reclaim_vwap_buffer: float,
    lower_high_min: float,
    breakdown_impulse_min: float,
) -> tuple[VariantResult, pd.DataFrame]:
    trades: list[dict] = []
    for context in contexts:
        event = context["event"]
        peak_vs_prev_close = float(event.get("peak_vs_prev_close_pct", np.nan))
        if not np.isfinite(peak_vs_prev_close) or peak_vs_prev_close < overshoot_prev_close_min:
            continue

        trade = _find_failed_reclaim_trigger(
            event,
            context["session_arrays"],
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
        if trade is not None:
            trades.append(trade)

    trades_df = pd.DataFrame(trades)
    variant = (
        f"vwapfail_pc{int(overshoot_prev_close_min)}"
        f"_start{int(start_minute)}"
        f"_end{int(max_minute)}"
        f"_break{vwap_break_min:.2f}"
        f"_reb{rebound_min:.2f}"
        f"_win{int(reclaim_window_bars)}"
        f"_rvwap{reclaim_vwap_buffer:.2f}"
        f"_lh{lower_high_min:.1f}"
        f"_bd{breakdown_impulse_min:.1f}"
    )

    if trades_df.empty:
        return VariantResult(variant, 0, np.nan, np.nan, np.nan, 0.0, np.nan), trades_df

    rets = pd.to_numeric(trades_df["return_pct"], errors="coerce").dropna()
    match_pct = float(pd.to_numeric(trades_df["label_match"], errors="coerce").mean() * 100.0)
    result = VariantResult(
        variant=variant,
        trades=int(len(trades_df)),
        win_rate_pct=float((rets > 0).mean() * 100.0),
        mean_return_pct=float(rets.mean()),
        median_return_pct=float(rets.median()),
        compounded_return_pct=float(_compound_return_pct(rets)),
        labeled_overshoot_match_pct=match_pct,
    )
    trades_df["variant"] = variant
    return result, trades_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a failed-reclaim dump strategy after positive off-hours earnings overshoots.")
    parser.add_argument("--start", default="2024-04-04", help="Event-study start date (default: 2024-04-04).")
    parser.add_argument("--end", default="2026-05-01", help="Event-study end date (default: 2026-05-01).")
    parser.add_argument(
        "--artifact-stem",
        default="earnings_overshoot_vwap_fail",
        help="Artifact stem (default: earnings_overshoot_vwap_fail).",
    )
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    events = _load_labeled_events(start, end)
    contexts = _build_event_contexts(events)

    variants_path = ARTIFACT_DIR / f"{args.artifact_stem}_variants.csv"
    trades_path = ARTIFACT_DIR / f"{args.artifact_stem}_top_trades.csv"
    summary_path = ARTIFACT_DIR / f"{args.artifact_stem}_summary.json"

    variant_results: list[VariantResult] = []
    variant_trades: list[pd.DataFrame] = []
    for overshoot_prev_close in (15.0, 20.0):
        for start_minute in (11 * 60, 12 * 60, 13 * 60):
            for max_minute in (14 * 60, 14 * 60 + 30, 15 * 60):
                if max_minute <= start_minute:
                    continue
                for vwap_break_min in (0.0, 0.25, 0.5):
                    for rebound_min in (0.25, 0.5, 1.0):
                        for reclaim_window_bars in (5, 10, 15):
                            for reclaim_vwap_buffer in (0.0, 0.25, 0.5):
                                for lower_high_min in (1.0, 2.0, 3.0):
                                    for breakdown_impulse_min in (0.5, 1.0, 1.5):
                                        result, trades = _run_variant(
                                            contexts,
                                            overshoot_prev_close_min=overshoot_prev_close,
                                            start_minute=start_minute,
                                            max_minute=max_minute,
                                            vwap_break_min=vwap_break_min,
                                            rebound_min=rebound_min,
                                            reclaim_window_bars=reclaim_window_bars,
                                            reclaim_vwap_buffer=reclaim_vwap_buffer,
                                            lower_high_min=lower_high_min,
                                            breakdown_impulse_min=breakdown_impulse_min,
                                        )
                                        variant_results.append(result)
                                        if not trades.empty:
                                            variant_trades.append(trades)

    variants_df = pd.DataFrame([asdict(row) for row in variant_results]).sort_values(
        ["compounded_return_pct", "mean_return_pct", "trades"],
        ascending=[False, False, False],
    )
    variants_df.to_csv(variants_path, index=False)

    top_variant = variants_df.iloc[0].to_dict() if not variants_df.empty else {}
    top_trades = pd.DataFrame()
    if variant_trades and top_variant:
        all_trades = pd.concat(variant_trades, ignore_index=True)
        top_trades = all_trades[all_trades["variant"] == top_variant["variant"]].copy()
    top_trades.to_csv(trades_path, index=False)

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
    print(f"Artifacts written:\n- {variants_path}\n- {trades_path}\n- {summary_path}")


if __name__ == "__main__":
    main()
