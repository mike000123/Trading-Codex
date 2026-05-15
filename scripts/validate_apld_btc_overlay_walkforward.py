from __future__ import annotations

import json
import sys
import types
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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

from scripts.validate_apld_btc_overlay import (
    ALPACA_SAFE_BUFFER_DAYS,
    ARTIFACT_DIR,
    BEST_LONG_RULE,
    BEST_SHORT_RULE,
    OverlayResult,
    OverlayRule,
    _compounded_return_pct,
    _evaluate_overlay,
    _load_prepared_context,
    _prepare_sessions,
)


FOLDS = [
    ("2024-04-01", "2024-09-30"),
    ("2024-10-01", "2025-03-31"),
    ("2025-04-01", "2025-09-30"),
    ("2025-10-01", "2026-05-01"),
]
STOP_CANDIDATES = [0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
SHORT_FIXED_STOP = 2.5
LONG_FIXED_STOP = 3.0
MIN_TRAIN_SIGNALS = 4


@dataclass
class EvalSnapshot:
    stop_loss_pct: float
    signals: int
    win_rate_pct: float
    mean_return_pct: float
    compounded_return_pct: float
    score: float


@dataclass
class FoldChoice:
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    chosen_stop_pct: float
    train_result: EvalSnapshot
    test_result: EvalSnapshot


def _subset_sessions(session_blobs: list[dict[str, Any]], start: str, end: str) -> list[dict[str, Any]]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    out: list[dict[str, Any]] = []
    for blob in session_blobs:
        session_date = pd.Timestamp(blob["feature"]["session_date"])
        if start_ts <= session_date <= end_ts:
            out.append(blob)
    return out


def _snapshot_from_result(result: OverlayResult) -> EvalSnapshot:
    return EvalSnapshot(
        stop_loss_pct=float(result.stop_loss_pct or 0.0),
        signals=int(result.signals),
        win_rate_pct=float(result.win_rate_pct),
        mean_return_pct=float(result.mean_return_pct),
        compounded_return_pct=float(result.compounded_return_pct),
        score=float(result.score),
    )


def _evaluate_stop(
    session_blobs: list[dict[str, Any]],
    *,
    rule: OverlayRule,
    stop_loss_pct: float,
) -> tuple[OverlayResult | None, pd.DataFrame]:
    return _evaluate_overlay(
        session_blobs,
        rule=rule,
        family="stop_eod",
        stop_loss_pct=stop_loss_pct,
        trail_pct=None,
        trail_activation_pct=None,
    )


def _pick_best_train_stop(session_blobs: list[dict[str, Any]], rule: OverlayRule) -> tuple[float, OverlayResult]:
    ranked: list[OverlayResult] = []
    for stop in STOP_CANDIDATES:
        result, _ = _evaluate_stop(session_blobs, rule=rule, stop_loss_pct=stop)
        if result is None or int(result.signals) < MIN_TRAIN_SIGNALS:
            continue
        ranked.append(result)
    if not ranked:
        raise RuntimeError(f"No valid train candidates found for {rule.name}")
    ranked.sort(key=lambda r: (r.score, r.compounded_return_pct, r.win_rate_pct), reverse=True)
    best = ranked[0]
    return float(best.stop_loss_pct), best


def _summarize_fold_choices(choices: list[FoldChoice]) -> dict[str, Any]:
    if not choices:
        return {}
    compounded = 1.0
    returns: list[float] = []
    wins: list[float] = []
    total_trades = 0
    positive = 0
    stops = Counter()
    for choice in choices:
        r = float(choice.test_result.compounded_return_pct)
        compounded *= 1.0 + (r / 100.0)
        returns.append(r)
        wins.append(float(choice.test_result.win_rate_pct))
        total_trades += int(choice.test_result.signals)
        if r > 0:
            positive += 1
        stops[str(choice.chosen_stop_pct)] += 1
    return {
        "compounded_oos_return_pct": round((compounded - 1.0) * 100.0, 3),
        "average_oos_fold_return_pct": round(sum(returns) / len(returns), 3),
        "median_oos_fold_return_pct": round(pd.Series(returns).median(), 3),
        "average_oos_win_rate_pct": round(sum(wins) / len(wins), 3),
        "positive_oos_folds": int(positive),
        "total_oos_folds": int(len(choices)),
        "total_oos_signals": int(total_trades),
        "selection_counts": dict(stops),
    }


def _summarize_fixed(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {}
    compounded = 1.0
    returns = []
    total_signals = 0
    positive = 0
    win_rates = []
    for item in results:
        r = float(item["compounded_return_pct"])
        compounded *= 1.0 + (r / 100.0)
        returns.append(r)
        total_signals += int(item["signals"])
        win_rates.append(float(item["win_rate_pct"]))
        if r > 0:
            positive += 1
    return {
        "compounded_oos_return_pct": round((compounded - 1.0) * 100.0, 3),
        "average_oos_fold_return_pct": round(sum(returns) / len(returns), 3),
        "median_oos_fold_return_pct": round(pd.Series(returns).median(), 3),
        "average_oos_win_rate_pct": round(sum(win_rates) / len(win_rates), 3),
        "positive_oos_folds": int(positive),
        "total_oos_folds": int(len(results)),
        "total_oos_signals": int(total_signals),
    }


def _summarize_combined_test_trades(trades: list[pd.DataFrame]) -> dict[str, Any]:
    parts = [t for t in trades if t is not None and not t.empty]
    if not parts:
        return {}
    combined = pd.concat(parts, ignore_index=True).sort_values("session_date")
    vals = pd.to_numeric(combined["ret_pct"], errors="coerce").dropna()
    return {
        "signals": int(len(vals)),
        "win_rate_pct": float((vals > 0).mean() * 100.0) if len(vals) else None,
        "mean_return_pct": float(vals.mean()) if len(vals) else None,
        "compounded_return_pct": _compounded_return_pct(vals) if len(vals) else None,
    }


def _run_walkforward_for_rule(
    session_blobs: list[dict[str, Any]],
    *,
    rule: OverlayRule,
    fixed_stop: float,
) -> tuple[list[FoldChoice], list[dict[str, Any]], list[pd.DataFrame], list[pd.DataFrame]]:
    choices: list[FoldChoice] = []
    fixed_fold_results: list[dict[str, Any]] = []
    chosen_test_trades: list[pd.DataFrame] = []
    fixed_test_trades: list[pd.DataFrame] = []

    for i in range(1, len(FOLDS)):
        train_start = FOLDS[0][0]
        train_end = FOLDS[i - 1][1]
        test_start, test_end = FOLDS[i]

        train_subset = _subset_sessions(session_blobs, train_start, train_end)
        test_subset = _subset_sessions(session_blobs, test_start, test_end)

        chosen_stop, train_result = _pick_best_train_stop(train_subset, rule)
        test_result, chosen_trades = _evaluate_stop(test_subset, rule=rule, stop_loss_pct=chosen_stop)
        if test_result is None:
            raise RuntimeError(f"No test result for {rule.name} with chosen stop {chosen_stop} on {test_start} -> {test_end}")
        choices.append(
            FoldChoice(
                train_start=str(pd.Timestamp(train_start)),
                train_end=str(pd.Timestamp(train_end)),
                test_start=str(pd.Timestamp(test_start)),
                test_end=str(pd.Timestamp(test_end)),
                chosen_stop_pct=float(chosen_stop),
                train_result=_snapshot_from_result(train_result),
                test_result=_snapshot_from_result(test_result),
            )
        )
        chosen_test_trades.append(chosen_trades)

        fixed_result, fixed_trades = _evaluate_stop(test_subset, rule=rule, stop_loss_pct=fixed_stop)
        if fixed_result is None:
            raise RuntimeError(f"No fixed-stop test result for {rule.name} on {test_start} -> {test_end}")
        fixed_fold_results.append(
            {
                "fold_start": str(pd.Timestamp(test_start)),
                "fold_end": str(pd.Timestamp(test_end)),
                "stop_loss_pct": float(fixed_stop),
                "signals": int(fixed_result.signals),
                "win_rate_pct": float(fixed_result.win_rate_pct),
                "mean_return_pct": float(fixed_result.mean_return_pct),
                "compounded_return_pct": float(fixed_result.compounded_return_pct),
                "score": float(fixed_result.score),
            }
        )
        fixed_test_trades.append(fixed_trades)

    return choices, fixed_fold_results, chosen_test_trades, fixed_test_trades


def main() -> None:
    start = pd.Timestamp(FOLDS[0][0])
    end = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=ALPACA_SAFE_BUFFER_DAYS)

    prepared = _load_prepared_context(start, end)
    session_blobs = _prepare_sessions(prepared)

    short_choices, short_fixed, short_chosen_trades, short_fixed_trades = _run_walkforward_for_rule(
        session_blobs,
        rule=BEST_SHORT_RULE,
        fixed_stop=SHORT_FIXED_STOP,
    )
    long_choices, long_fixed, long_chosen_trades, long_fixed_trades = _run_walkforward_for_rule(
        session_blobs,
        rule=BEST_LONG_RULE,
        fixed_stop=LONG_FIXED_STOP,
    )

    payload = {
        "symbol": "APLD",
        "source": "alpaca",
        "interval": "1Min",
        "window": {
            "start": str(start),
            "end": str(end),
        },
        "folds": [{"start": s, "end": e} for s, e in FOLDS],
        "stop_candidates": STOP_CANDIDATES,
        "min_train_signals": MIN_TRAIN_SIGNALS,
        "short_rule": asdict(BEST_SHORT_RULE),
        "long_rule": asdict(BEST_LONG_RULE),
        "short_walkforward_choices": [asdict(c) for c in short_choices],
        "short_walkforward_summary": _summarize_fold_choices(short_choices),
        "short_fixed_stop_pct": SHORT_FIXED_STOP,
        "short_fixed_oos": short_fixed,
        "short_fixed_summary": _summarize_fixed(short_fixed),
        "long_walkforward_choices": [asdict(c) for c in long_choices],
        "long_walkforward_summary": _summarize_fold_choices(long_choices),
        "long_fixed_stop_pct": LONG_FIXED_STOP,
        "long_fixed_oos": long_fixed,
        "long_fixed_summary": _summarize_fixed(long_fixed),
        "combined_walkforward_summary": _summarize_combined_test_trades(short_chosen_trades + long_chosen_trades),
        "combined_fixed_summary": _summarize_combined_test_trades(short_fixed_trades + long_fixed_trades),
        "notes": [
            "This pass applies a stricter chronological walk-forward to the stop_eod long and short overlays only.",
            "For each out-of-sample fold, the stop percentage is chosen on prior data only, then frozen on the next fold.",
            "The fixed baseline compares against the full-sample stop choices from the prior overlay pass.",
        ],
    }

    out = ARTIFACT_DIR / "apld_btc_overlay_walkforward.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["short_walkforward_summary"], indent=2))
    print(json.dumps(payload["long_walkforward_summary"], indent=2))
    print(json.dumps(payload["combined_walkforward_summary"], indent=2))
    print(f"Wrote results to {out}")


if __name__ == "__main__":
    main()
