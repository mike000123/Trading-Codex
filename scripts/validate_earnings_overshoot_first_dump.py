"""
Validate an earlier "first dump leg" short strategy after positive off-hours
earnings overshoots.

Hypothesis:
- the stock reprices strongly above the prior close after positive earnings
- once the overshoot is established, the first decisive downside impulse can
  begin a tradable dump wave
- that first dump leg may be better captured with early short entries and
  wave-aware exits than with a late failed-reclaim short-to-close rule
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
        "open": pd.to_numeric(session["open"], errors="coerce").to_numpy(dtype=np.float64, copy=True),
        "high": pd.to_numeric(session["high"], errors="coerce").to_numpy(dtype=np.float64, copy=True),
        "low": pd.to_numeric(session["low"], errors="coerce").to_numpy(dtype=np.float64, copy=True),
        "close": pd.to_numeric(session["close"], errors="coerce").to_numpy(dtype=np.float64, copy=True),
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


def _find_entry_idx(
    event: dict,
    arrays: dict,
    *,
    overshoot_prev_close_min: float,
    start_minute: int,
    max_minute: int,
    peak_pullback_min: float,
    vwap_break_min: float,
    breakdown_impulse_min: float,
) -> int | None:
    prev_close = float(event.get("prev_close", np.nan))
    if not np.isfinite(prev_close) or prev_close <= 0:
        return None

    minutes = arrays["minutes_et"]
    close = arrays["close"]
    close_from_vwap = arrays["close_from_vwap_pct"]
    ret_5m = arrays["ret_5m_pct"]
    running_peak = arrays["running_peak_high"]
    running_peak_vs_prev = (running_peak / prev_close - 1.0) * 100.0
    pullback_from_peak = (close / running_peak - 1.0) * 100.0

    cond = (
        (minutes >= start_minute)
        & (minutes <= max_minute)
        & np.isfinite(close)
        & np.isfinite(close_from_vwap)
        & np.isfinite(ret_5m)
        & np.isfinite(running_peak_vs_prev)
        & np.isfinite(pullback_from_peak)
        & (running_peak_vs_prev >= overshoot_prev_close_min)
        & (pullback_from_peak <= -peak_pullback_min)
        & (close_from_vwap <= -vwap_break_min)
        & (ret_5m <= -breakdown_impulse_min)
    )
    hits = np.flatnonzero(cond)
    if hits.size == 0:
        return None
    return int(hits[0])


def _exit_trade(
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

    if exit_mode == "vwap_reclaim":
        for idx in range(entry_idx + 1, min(entry_idx + max_hold_bars + 1, n)):
            if np.isfinite(close_from_vwap[idx]) and close_from_vwap[idx] >= vwap_reclaim_buffer:
                return idx, "vwap_reclaim"
        return min(entry_idx + max_hold_bars, n - 1), "vwap_timeout"

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

    raise ValueError(f"Unknown exit mode: {exit_mode}")


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
    exit_mode: str,
    rebound_exit_pct: float,
    vwap_reclaim_buffer: float,
    max_hold_bars: int,
) -> dict | None:
    entry_idx = _find_entry_idx(
        event,
        arrays,
        overshoot_prev_close_min=overshoot_prev_close_min,
        start_minute=start_minute,
        max_minute=max_minute,
        peak_pullback_min=peak_pullback_min,
        vwap_break_min=vwap_break_min,
        breakdown_impulse_min=breakdown_impulse_min,
    )
    if entry_idx is None:
        return None

    exit_idx, exit_reason = _exit_trade(
        arrays,
        entry_idx=entry_idx,
        exit_mode=exit_mode,
        rebound_exit_pct=rebound_exit_pct,
        vwap_reclaim_buffer=vwap_reclaim_buffer,
        max_hold_bars=max_hold_bars,
    )

    date = arrays["date"]
    close = arrays["close"]
    entry_px = float(close[entry_idx])
    exit_px = float(close[exit_idx])
    ret = -_safe_pct(exit_px, entry_px)

    return {
        "symbol": event["symbol"],
        "reaction_date": pd.Timestamp(event["reaction_date"]).date().isoformat(),
        "timing": event.get("timing"),
        "surprise_pct": float(event.get("surprise_pct", np.nan)),
        "gap_pct": float(event.get("gap_pct", np.nan)),
        "peak_vs_prev_close_pct": float(event.get("peak_vs_prev_close_pct", np.nan)),
        "entry_time": pd.Timestamp(date[entry_idx]).isoformat(),
        "entry_px": entry_px,
        "exit_time": pd.Timestamp(date[exit_idx]).isoformat(),
        "exit_px": exit_px,
        "return_pct": ret,
        "exit_reason": exit_reason,
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
    exit_mode: str,
    rebound_exit_pct: float,
    vwap_reclaim_buffer: float,
    max_hold_bars: int,
) -> tuple[VariantResult, pd.DataFrame]:
    trades: list[dict] = []
    for context in contexts:
        event = context["event"]
        peak_vs_prev = float(event.get("peak_vs_prev_close_pct", np.nan))
        if not np.isfinite(peak_vs_prev) or peak_vs_prev < overshoot_prev_close_min:
            continue
        trade = _trade_event(
            event,
            context["session_arrays"],
            overshoot_prev_close_min=overshoot_prev_close_min,
            start_minute=start_minute,
            max_minute=max_minute,
            peak_pullback_min=peak_pullback_min,
            vwap_break_min=vwap_break_min,
            breakdown_impulse_min=breakdown_impulse_min,
            exit_mode=exit_mode,
            rebound_exit_pct=rebound_exit_pct,
            vwap_reclaim_buffer=vwap_reclaim_buffer,
            max_hold_bars=max_hold_bars,
        )
        if trade is not None:
            trades.append(trade)

    trades_df = pd.DataFrame(trades)
    variant = (
        f"firstdump_pc{int(overshoot_prev_close_min)}"
        f"_start{int(start_minute)}"
        f"_end{int(max_minute)}"
        f"_pull{peak_pullback_min:.1f}"
        f"_vbreak{vwap_break_min:.1f}"
        f"_bd{breakdown_impulse_min:.1f}"
        f"_exit{exit_mode}"
        f"_reb{rebound_exit_pct:.1f}"
        f"_vr{vwap_reclaim_buffer:.1f}"
        f"_hold{int(max_hold_bars)}"
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
    parser = argparse.ArgumentParser(description="Validate an earlier first-dump short strategy after positive earnings overshoots.")
    parser.add_argument("--start", default="2024-04-04", help="Event-study start date (default: 2024-04-04).")
    parser.add_argument("--end", default="2026-05-01", help="Event-study end date (default: 2026-05-01).")
    parser.add_argument(
        "--artifact-stem",
        default="earnings_overshoot_first_dump",
        help="Artifact stem (default: earnings_overshoot_first_dump).",
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
                for peak_pullback_min in (2.0, 3.0, 4.0):
                    for vwap_break_min in (0.0, 0.5, 1.0):
                        for breakdown_impulse_min in (0.5, 1.0, 1.5):
                            for exit_mode in ("time_10", "time_20", "time_30", "rebound", "vwap_reclaim", "ema_turn", "ema_cross"):
                                for rebound_exit_pct in (0.5, 1.0):
                                    for vwap_reclaim_buffer in (0.0, 0.5):
                                        max_hold_bars = 30
                                        result, trades = _run_variant(
                                            contexts,
                                            overshoot_prev_close_min=overshoot_prev_close,
                                            start_minute=start_minute,
                                            max_minute=max_minute,
                                            peak_pullback_min=peak_pullback_min,
                                            vwap_break_min=vwap_break_min,
                                            breakdown_impulse_min=breakdown_impulse_min,
                                            exit_mode=exit_mode,
                                            rebound_exit_pct=rebound_exit_pct,
                                            vwap_reclaim_buffer=vwap_reclaim_buffer,
                                            max_hold_bars=max_hold_bars,
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
