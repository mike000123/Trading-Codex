from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_earnings_event_hybrid_main import (
    ARTIFACT_DIR,
    _load_unified_events,
    evaluate_unified_strategy,
)


def _frame_from_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(records).sort_values(
        by=["overall_compounded_return_pct", "overall_triggered_events", "overall_max_drawdown_pct"],
        ascending=[False, False, False],
        kind="stable",
    ).reset_index(drop=True)


def _overall_metrics(summary) -> dict[str, Any]:
    return {
        "overall_compounded_return_pct": float(summary.compounded_event_return_pct),
        "overall_triggered_events": int(summary.triggered_events),
        "overall_total_trades": int(summary.total_trades),
        "overall_win_rate_pct": float(summary.win_rate_pct),
        "overall_max_drawdown_pct": float(summary.max_drawdown_pct),
    }


def _branch_metrics(events_df: pd.DataFrame, branch: str) -> dict[str, Any]:
    subset = events_df[events_df["selected_branch"] == branch].copy()
    if subset.empty:
        return {
            "branch": branch,
            "branch_events": 0,
            "branch_compounded_return_pct": 0.0,
            "branch_mean_return_pct": 0.0,
            "branch_win_rate_pct": 0.0,
        }
    returns = pd.to_numeric(subset["total_return_pct"], errors="coerce").dropna()
    compounded = float(((1.0 + returns / 100.0).prod() - 1.0) * 100.0) if not returns.empty else 0.0
    return {
        "branch": branch,
        "branch_events": int(len(subset)),
        "branch_compounded_return_pct": compounded,
        "branch_mean_return_pct": float(returns.mean()) if not returns.empty else 0.0,
        "branch_win_rate_pct": float((returns > 0).mean() * 100.0) if not returns.empty else 0.0,
    }


def _negative_candidates() -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    for entry_min_close, vwap_reclaim, rebound_impulse, rebound_from_trough in itertools.product(
        [0.0, 0.25, 0.5],
        [-0.5, 0.0, 0.25],
        [0.5, 0.75, 1.0],
        [4.0, 5.0],
    ):
        variants.append(
            {
                "negative": {
                    "entry_min_close_from_open_pct": entry_min_close,
                    "vwap_reclaim_min": vwap_reclaim,
                    "rebound_impulse_min": rebound_impulse,
                    "rebound_from_trough_min": rebound_from_trough,
                }
            }
        )
    return variants


def _wave_candidates() -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    for entry_mode, entry_min_close, momentum_min in itertools.product(
        ["ema_turn", "break_prev_high"],
        [None, 0.0, 0.25, 0.5],
        [0.0, 0.25, 0.5],
    ):
        variants.append(
            {
                "positive": {
                    "wave_best_long_entry_mode": entry_mode,
                    "wave_best_long_entry_min_close_from_open_pct": entry_min_close,
                    "wave_best_long_momentum_min": momentum_min,
                }
            }
        )
    return variants


def _label_from_params(prefix: str, params: dict[str, Any]) -> str:
    flat: dict[str, Any] = params.get(prefix, {})
    parts = [prefix]
    for key in sorted(flat):
        val = flat[key]
        if val is None:
            sval = "none"
        else:
            sval = str(val).replace(".", "p")
        parts.append(f"{key}={sval}")
    return "|".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize targeted earnings-event hybrid refinements.")
    parser.add_argument("--start", default="2024-04-04")
    parser.add_argument("--end", default="2026-05-01")
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    events = _load_unified_events(start, end)

    baseline_summary, baseline_events, _ = evaluate_unified_strategy(events, label="baseline")
    baseline = {
        "summary": _overall_metrics(baseline_summary),
        "negative_rebound": _branch_metrics(baseline_events, "negative_rebound"),
        "wave_best": _branch_metrics(baseline_events, "wave_best"),
    }

    negative_rows: list[dict[str, Any]] = []
    for params in _negative_candidates():
        summary, events_df, _ = evaluate_unified_strategy(events, label="negative_refine", params=params)
        row = {
            "variant": _label_from_params("negative", params),
            "params_json": json.dumps(params, sort_keys=True),
        }
        row.update(_overall_metrics(summary))
        row.update(_branch_metrics(events_df, "negative_rebound"))
        negative_rows.append(row)
    negative_df = _frame_from_records(negative_rows)

    wave_rows: list[dict[str, Any]] = []
    for params in _wave_candidates():
        summary, events_df, _ = evaluate_unified_strategy(events, label="wave_refine", params=params)
        row = {
            "variant": _label_from_params("positive", params),
            "params_json": json.dumps(params, sort_keys=True),
        }
        row.update(_overall_metrics(summary))
        row.update(_branch_metrics(events_df, "wave_best"))
        wave_rows.append(row)
    wave_df = _frame_from_records(wave_rows)

    top_negative_params = json.loads(str(negative_df.iloc[0]["params_json"])) if not negative_df.empty else {}
    top_wave_params = json.loads(str(wave_df.iloc[0]["params_json"])) if not wave_df.empty else {}

    combined_rows: list[dict[str, Any]] = []
    combo_candidates = [({}, {}), (top_negative_params, {}), ({}, top_wave_params), (top_negative_params, top_wave_params)]
    for neg_params, wave_params in combo_candidates:
        merged: dict[str, Any] = {}
        merged.update(neg_params)
        for key, value in wave_params.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                new_value = dict(merged[key])
                new_value.update(value)
                merged[key] = new_value
            else:
                merged[key] = value
        summary, events_df, _ = evaluate_unified_strategy(events, label="combined_refine", params=merged)
        row = {
            "variant": "baseline"
            if not merged
            else (
                "negative_only"
                if "negative" in merged and "positive" not in merged
                else "wave_only"
                if "positive" in merged and "negative" not in merged
                else "negative_plus_wave"
            ),
            "params_json": json.dumps(merged, sort_keys=True),
        }
        row.update(_overall_metrics(summary))
        row.update(_branch_metrics(events_df, "negative_rebound"))
        row.update(_branch_metrics(events_df, "wave_best"))
        combined_rows.append(row)
    combined_df = _frame_from_records(combined_rows)

    summary_payload = {
        "window": {"start": start.date().isoformat(), "end": end.date().isoformat()},
        "baseline": baseline,
        "best_negative_variant": negative_df.iloc[0].to_dict() if not negative_df.empty else {},
        "best_wave_variant": wave_df.iloc[0].to_dict() if not wave_df.empty else {},
        "best_combined_variant": combined_df.iloc[0].to_dict() if not combined_df.empty else {},
    }

    (ARTIFACT_DIR / "earnings_event_hybrid_refinement_summary.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )
    negative_df.to_csv(ARTIFACT_DIR / "earnings_event_hybrid_negative_refinements.csv", index=False)
    wave_df.to_csv(ARTIFACT_DIR / "earnings_event_hybrid_wave_refinements.csv", index=False)
    combined_df.to_csv(ARTIFACT_DIR / "earnings_event_hybrid_combined_refinements.csv", index=False)
    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
