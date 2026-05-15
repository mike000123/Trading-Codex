"""
Validate an intraday overshoot-then-dump / overshoot-then-rebound pattern
after off-hours earnings events using cached 1-minute stock data.

This is the narrower strategy hypothesis that emerged after the broader
earnings-gap study:

- The raw opening gap is not enough by itself.
- Some events establish a new higher/lower post-earnings fair zone.
- Price can then overshoot above/below that new zone intraday.
- A reversal back toward the stabilisation band may be tradable.

We approximate that with a live-style proxy:
- off-hours earnings event with surprise sign
- intraday overshoot relative to the prior regular-session close
- pullback / rebound away from the local extreme
- anchored VWAP break / reclaim
- short-term reversal confirmation
- exit into the close
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
EVENTS_PATH = ARTIFACT_DIR / "earnings_open_reaction_events.csv"
ALPACA_CACHE_DIR = ROOT / "data_cache" / "alpaca"
RTH_OPEN = 9 * 60 + 30
RTH_CLOSE = 16 * 60
_SESSION_CACHE: dict[tuple[str, str], pd.DataFrame | None] = {}
_SESSION_ARRAY_CACHE: dict[tuple[str, str], dict | None] = {}


@dataclass
class VariantResult:
    variant: str
    trades: int
    win_rate_pct: float
    mean_return_pct: float
    median_return_pct: float
    compounded_return_pct: float
    long_trades: int
    short_trades: int
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


def _load_events() -> pd.DataFrame:
    if not EVENTS_PATH.exists():
        raise SystemExit(
            f"Missing prerequisite event table: {EVENTS_PATH}\n"
            "Run validate_earnings_open_reaction.py first."
        )
    df = pd.read_csv(EVENTS_PATH)
    if df.empty:
        raise SystemExit("The earnings event table is empty.")
    df["reaction_date"] = pd.to_datetime(df["reaction_date"], errors="coerce")
    return df.dropna(subset=["symbol", "reaction_date"]).reset_index(drop=True)


def _session_key(symbol: str, reaction_date: pd.Timestamp) -> tuple[str, str]:
    return str(symbol).upper(), pd.Timestamp(reaction_date).date().isoformat()


def _load_session(symbol: str, reaction_date: pd.Timestamp) -> pd.DataFrame | None:
    key = _session_key(symbol, reaction_date)
    cached = _SESSION_CACHE.get(key, None)
    if key in _SESSION_CACHE:
        return cached

    path = ALPACA_CACHE_DIR / str(symbol).upper() / "1Min.csv"
    if not path.exists():
        _SESSION_CACHE[key] = None
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
        _SESSION_CACHE[key] = None
        return None
    session = session.sort_values("date").reset_index(drop=True)
    typical = (session["high"] + session["low"] + session["close"]) / 3.0
    vol = pd.to_numeric(session["volume"], errors="coerce").fillna(0.0)
    cum_vol = vol.cumsum().replace(0.0, np.nan)
    session["anchored_vwap"] = (typical * vol).cumsum() / cum_vol
    session["close_from_open_pct"] = (session["close"] / float(session.iloc[0]["open"]) - 1.0) * 100.0
    session["close_from_vwap_pct"] = (session["close"] / session["anchored_vwap"] - 1.0) * 100.0
    session["ret_5m_pct"] = session["close"].pct_change(5) * 100.0
    _SESSION_CACHE[key] = session
    return session


def _load_session_arrays(symbol: str, reaction_date: pd.Timestamp) -> dict | None:
    key = _session_key(symbol, reaction_date)
    cached = _SESSION_ARRAY_CACHE.get(key, None)
    if key in _SESSION_ARRAY_CACHE:
        return cached

    session = _load_session(symbol, reaction_date)
    if session is None or session.empty:
        _SESSION_ARRAY_CACHE[key] = None
        return None

    arrays = {
        "date": session["date"].to_numpy(copy=True),
        "minutes_et": pd.to_numeric(session["minutes_et"], errors="coerce").to_numpy(dtype=np.int32, copy=True),
        "high": pd.to_numeric(session["high"], errors="coerce").to_numpy(dtype=np.float64, copy=True),
        "low": pd.to_numeric(session["low"], errors="coerce").to_numpy(dtype=np.float64, copy=True),
        "close": pd.to_numeric(session["close"], errors="coerce").to_numpy(dtype=np.float64, copy=True),
        "close_from_open_pct": pd.to_numeric(session["close_from_open_pct"], errors="coerce").to_numpy(dtype=np.float64, copy=True),
        "close_from_vwap_pct": pd.to_numeric(session["close_from_vwap_pct"], errors="coerce").to_numpy(dtype=np.float64, copy=True),
        "ret_5m_pct": pd.to_numeric(session["ret_5m_pct"], errors="coerce").to_numpy(dtype=np.float64, copy=True),
    }
    arrays["running_peak_high"] = np.maximum.accumulate(arrays["high"])
    arrays["running_trough_low"] = np.minimum.accumulate(arrays["low"])
    _SESSION_ARRAY_CACHE[key] = arrays
    return arrays


def _label_overshoot_shape(event_row: pd.Series, session: pd.DataFrame) -> dict:
    late = session[session["minutes_et"] >= (14 * 60 + 30)].copy()
    if late.empty:
        late = session.tail(min(60, len(session))).copy()
    equilibrium = float(late["close"].median()) if not late.empty else float(session.iloc[-1]["close"])

    surprise = float(event_row.get("surprise_pct", np.nan))
    if not np.isfinite(surprise):
        surprise = 0.0
    prev_close = float(event_row.get("prev_close", np.nan))

    peak = float(session["high"].max())
    trough = float(session["low"].min())
    close_px = float(session.iloc[-1]["close"])
    peak_time = pd.Timestamp(session.loc[session["high"].idxmax(), "date"])
    trough_time = pd.Timestamp(session.loc[session["low"].idxmin(), "date"])

    pos_shape = (
        surprise > 0
        and peak_time.hour * 60 + peak_time.minute >= (10 * 60)
        and _safe_pct(peak, equilibrium) >= 3.0
        and _safe_pct(close_px, peak) <= -3.0
    )
    neg_shape = (
        surprise < 0
        and trough_time.hour * 60 + trough_time.minute >= (10 * 60)
        and _safe_pct(trough, equilibrium) <= -3.0
        and _safe_pct(close_px, trough) >= 3.0
    )

    return {
        "late_equilibrium_px": equilibrium,
        "session_peak_px": peak,
        "session_peak_time": peak_time.isoformat(),
        "session_trough_px": trough,
        "session_trough_time": trough_time.isoformat(),
        "peak_vs_equilibrium_pct": _safe_pct(peak, equilibrium),
        "trough_vs_equilibrium_pct": _safe_pct(trough, equilibrium),
        "peak_vs_prev_close_pct": _safe_pct(peak, prev_close),
        "trough_vs_prev_close_pct": _safe_pct(trough, prev_close),
        "close_vs_prev_close_pct": _safe_pct(close_px, prev_close),
        "close_vs_peak_pct": _safe_pct(close_px, peak),
        "close_vs_trough_pct": _safe_pct(close_px, trough),
        "overshoot_dump_label": bool(pos_shape),
        "overshoot_rebound_label": bool(neg_shape),
    }


def _build_labeled_events(base_events: pd.DataFrame) -> pd.DataFrame:
    labeled_rows: list[dict] = []
    for _, event in base_events.iterrows():
        session = _load_session(str(event["symbol"]).upper(), pd.Timestamp(event["reaction_date"]))
        if session is None or session.empty:
            continue
        row = dict(event)
        row.update(_label_overshoot_shape(event, session))
        row["session_close_px"] = float(session.iloc[-1]["close"])
        labeled_rows.append(row)
    return pd.DataFrame(labeled_rows)


def _build_event_contexts(labeled_events: pd.DataFrame) -> list[dict]:
    contexts: list[dict] = []
    for event in labeled_events.to_dict(orient="records"):
        arrays = _load_session_arrays(str(event["symbol"]).upper(), pd.Timestamp(event["reaction_date"]))
        if arrays is None:
            continue
        contexts.append({"event": event, "session_arrays": arrays})
    return contexts


def _first_trigger(
    session_arrays: dict,
    *,
    direction: str,
    prev_close: float,
    start_minute: int,
    max_minute: int,
    overshoot_prev_close_min: float,
    pullback_from_extreme_min: float,
    vwap_confirm_min: float,
    reversal_5m_min: float,
) -> int | None:
    minutes = session_arrays["minutes_et"]
    close = session_arrays["close"]
    close_from_open = session_arrays["close_from_open_pct"]
    close_from_vwap = session_arrays["close_from_vwap_pct"]
    ret_5m = session_arrays["ret_5m_pct"]
    running_peak = session_arrays["running_peak_high"]
    running_trough = session_arrays["running_trough_low"]

    in_window = (minutes >= start_minute) & (minutes <= max_minute)
    if not np.any(in_window):
        return None

    if not np.isfinite(prev_close) or prev_close <= 0:
        return None

    running_peak_vs_prev = (running_peak / prev_close - 1.0) * 100.0
    running_trough_vs_prev = (running_trough / prev_close - 1.0) * 100.0
    pullback_from_peak = (close / running_peak - 1.0) * 100.0
    rebound_from_trough = (close / running_trough - 1.0) * 100.0

    if direction == "short":
        cond = (
            in_window
            & np.isfinite(close)
            & np.isfinite(running_peak_vs_prev)
            & np.isfinite(pullback_from_peak)
            & np.isfinite(close_from_vwap)
            & np.isfinite(ret_5m)
            & (running_peak_vs_prev >= overshoot_prev_close_min)
            & (pullback_from_peak <= -pullback_from_extreme_min)
            & (close_from_vwap <= -vwap_confirm_min)
            & (ret_5m <= -reversal_5m_min)
        )
    elif direction == "long":
        cond = (
            in_window
            & np.isfinite(close)
            & np.isfinite(running_trough_vs_prev)
            & np.isfinite(rebound_from_trough)
            & np.isfinite(close_from_vwap)
            & np.isfinite(ret_5m)
            & (running_trough_vs_prev <= -overshoot_prev_close_min)
            & (rebound_from_trough >= pullback_from_extreme_min)
            & (close_from_vwap >= vwap_confirm_min)
            & (ret_5m >= reversal_5m_min)
        )
    else:
        raise ValueError(f"Unknown direction: {direction}")

    hits = np.flatnonzero(cond)
    if hits.size == 0:
        return None
    return int(hits[0])


def _trade_event(
    event_row: dict,
    session_arrays: dict,
    *,
    start_minute: int,
    max_minute: int,
    overshoot_prev_close_min: float,
    pullback_from_extreme_min: float,
    vwap_confirm_min: float,
    reversal_5m_min: float,
) -> dict | None:
    surprise = float(event_row.get("surprise_pct", np.nan))
    prev_close = float(event_row.get("prev_close", np.nan))
    if not np.isfinite(surprise) or not np.isfinite(prev_close) or prev_close <= 0:
        return None

    trigger = None
    direction = None
    label_match = False

    if surprise > 0:
        trigger = _first_trigger(
            session_arrays,
            direction="short",
            prev_close=prev_close,
            start_minute=start_minute,
            max_minute=max_minute,
            overshoot_prev_close_min=overshoot_prev_close_min,
            pullback_from_extreme_min=pullback_from_extreme_min,
            vwap_confirm_min=vwap_confirm_min,
            reversal_5m_min=reversal_5m_min,
        )
        direction = "short" if trigger is not None else None
        label_match = bool(event_row.get("overshoot_dump_label", False))
    elif surprise < 0:
        trigger = _first_trigger(
            session_arrays,
            direction="long",
            prev_close=prev_close,
            start_minute=start_minute,
            max_minute=max_minute,
            overshoot_prev_close_min=overshoot_prev_close_min,
            pullback_from_extreme_min=pullback_from_extreme_min,
            vwap_confirm_min=vwap_confirm_min,
            reversal_5m_min=reversal_5m_min,
        )
        direction = "long" if trigger is not None else None
        label_match = bool(event_row.get("overshoot_rebound_label", False))

    if trigger is None or direction is None:
        return None

    close = session_arrays["close"]
    date = session_arrays["date"]
    entry_px = float(close[trigger])
    exit_px = float(close[-1])
    if direction == "short":
        ret = -_safe_pct(exit_px, entry_px)
    else:
        ret = _safe_pct(exit_px, entry_px)

    return {
        "symbol": event_row["symbol"],
        "reaction_date": pd.Timestamp(event_row["reaction_date"]).date().isoformat(),
        "timing": event_row.get("timing"),
        "direction": direction,
        "surprise_pct": surprise,
        "gap_pct": float(event_row.get("gap_pct", np.nan)),
        "peak_vs_prev_close_pct": float(event_row.get("peak_vs_prev_close_pct", np.nan)),
        "trough_vs_prev_close_pct": float(event_row.get("trough_vs_prev_close_pct", np.nan)),
        "entry_time": pd.Timestamp(date[trigger]).isoformat(),
        "entry_px": entry_px,
        "exit_px": exit_px,
        "return_pct": ret,
        "label_match": bool(label_match),
    }


def _run_variant(
    event_contexts: list[dict],
    *,
    start_minute: int,
    max_minute: int,
    overshoot_prev_close_min: float,
    pullback_from_extreme_min: float,
    vwap_confirm_min: float,
    reversal_5m_min: float,
) -> tuple[VariantResult, pd.DataFrame]:
    trades: list[dict] = []
    for context in event_contexts:
        event = context["event"]
        trade = _trade_event(
            event,
            context["session_arrays"],
            start_minute=start_minute,
            max_minute=max_minute,
            overshoot_prev_close_min=overshoot_prev_close_min,
            pullback_from_extreme_min=pullback_from_extreme_min,
            vwap_confirm_min=vwap_confirm_min,
            reversal_5m_min=reversal_5m_min,
        )
        if trade is not None:
            trades.append(trade)

    trades_df = pd.DataFrame(trades)
    variant = (
        f"overshoot_pc{int(overshoot_prev_close_min)}"
        f"_start{int(start_minute)}"
        f"_end{int(max_minute)}"
        f"_pull{pullback_from_extreme_min:.1f}"
        f"_vwap{vwap_confirm_min:.1f}"
        f"_rev{reversal_5m_min:.1f}"
    )
    if trades_df.empty:
        return VariantResult(variant, 0, np.nan, np.nan, np.nan, 0.0, 0, 0, np.nan), trades_df

    rets = pd.to_numeric(trades_df["return_pct"], errors="coerce").dropna()
    match_pct = float(pd.to_numeric(trades_df["label_match"], errors="coerce").mean() * 100.0)
    result = VariantResult(
        variant=variant,
        trades=int(len(trades_df)),
        win_rate_pct=float((rets > 0).mean() * 100.0),
        mean_return_pct=float(rets.mean()),
        median_return_pct=float(rets.median()),
        compounded_return_pct=float(_compound_return_pct(rets)),
        long_trades=int((trades_df["direction"] == "long").sum()),
        short_trades=int((trades_df["direction"] == "short").sum()),
        labeled_overshoot_match_pct=match_pct,
    )
    trades_df["variant"] = variant
    return result, trades_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate overshoot-dump / rebound rules after off-hours earnings events.")
    parser.add_argument("--start", default="2024-04-04", help="Event-study start date (default: 2024-04-04).")
    parser.add_argument("--end", default="2026-05-01", help="Event-study end date (default: 2026-05-01).")
    parser.add_argument("--artifact-stem", default="earnings_overshoot_dump", help="Artifact stem (default: earnings_overshoot_dump).")
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    events = _load_events()
    events = events[(events["reaction_date"] >= start) & (events["reaction_date"] <= end)].reset_index(drop=True)
    labeled = _build_labeled_events(events)

    labeled_path = ARTIFACT_DIR / f"{args.artifact_stem}_events_labeled.csv"
    variants_path = ARTIFACT_DIR / f"{args.artifact_stem}_variants.csv"
    trades_path = ARTIFACT_DIR / f"{args.artifact_stem}_top_trades.csv"
    summary_path = ARTIFACT_DIR / f"{args.artifact_stem}_summary.json"

    if labeled_path.exists():
        existing_labeled = pd.read_csv(labeled_path)
        if not existing_labeled.empty:
            existing_labeled["reaction_date"] = pd.to_datetime(existing_labeled["reaction_date"], errors="coerce")
            existing_labeled = existing_labeled[
                (existing_labeled["reaction_date"] >= start)
                & (existing_labeled["reaction_date"] <= end)
            ].reset_index(drop=True)
            required_cols = {
                "peak_vs_equilibrium_pct",
                "peak_vs_prev_close_pct",
                "trough_vs_prev_close_pct",
                "close_vs_prev_close_pct",
            }
            if len(existing_labeled) == len(events) and required_cols.issubset(existing_labeled.columns):
                labeled = existing_labeled
            else:
                labeled = _build_labeled_events(events)
    else:
        labeled = _build_labeled_events(events)

    labeled.to_csv(labeled_path, index=False)
    if labeled.empty:
        summary_path.write_text(json.dumps({"events": 0, "message": "No labeled overshoot events available."}, indent=2), encoding="utf-8")
        pd.DataFrame().to_csv(variants_path, index=False)
        pd.DataFrame().to_csv(trades_path, index=False)
        print("No labeled events available.")
        return

    event_contexts = _build_event_contexts(labeled)
    variant_results: list[VariantResult] = []
    variant_trades: list[pd.DataFrame] = []
    for overshoot_prev_close in (15.0, 20.0):
        for start_minute in (10 * 60, 11 * 60, 12 * 60):
            for max_minute in (13 * 60, 14 * 60, 14 * 60 + 30):
                if max_minute <= start_minute:
                    continue
                for pullback_from_extreme in (1.5, 3.0, 5.0):
                    for vwap_confirm in (0.0, 0.5, 1.0):
                        for reversal_5m in (0.5, 1.0, 1.5):
                            result, trades = _run_variant(
                                event_contexts,
                                start_minute=start_minute,
                                max_minute=max_minute,
                                overshoot_prev_close_min=overshoot_prev_close,
                                pullback_from_extreme_min=pullback_from_extreme,
                                vwap_confirm_min=vwap_confirm,
                                reversal_5m_min=reversal_5m,
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
        "labeled_events": int(len(labeled)),
        "positive_overshoot_dump_labels": int(pd.to_numeric(labeled["overshoot_dump_label"], errors="coerce").fillna(0).sum()),
        "negative_overshoot_rebound_labels": int(pd.to_numeric(labeled["overshoot_rebound_label"], errors="coerce").fillna(0).sum()),
        "positive_peak_vs_prev_close_ge_15": int(
            (
                (pd.to_numeric(labeled["surprise_pct"], errors="coerce") > 0)
                & (pd.to_numeric(labeled["peak_vs_prev_close_pct"], errors="coerce") >= 15.0)
            ).sum()
        ),
        "positive_peak_vs_prev_close_ge_20": int(
            (
                (pd.to_numeric(labeled["surprise_pct"], errors="coerce") > 0)
                & (pd.to_numeric(labeled["peak_vs_prev_close_pct"], errors="coerce") >= 20.0)
            ).sum()
        ),
        "best_variant": top_variant,
        "top_variants": variants_df.head(10).to_dict(orient="records"),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Labeled events: {len(labeled)}")
    if top_variant:
        print("Best variant:")
        print(json.dumps(top_variant, indent=2))
    print(f"Artifacts written:\n- {labeled_path}\n- {variants_path}\n- {trades_path}\n- {summary_path}")


if __name__ == "__main__":
    main()
