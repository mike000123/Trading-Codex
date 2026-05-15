from __future__ import annotations

import argparse
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


def _load_best_refinement_params() -> tuple[dict[str, Any], dict[str, Any]]:
    summary_path = ARTIFACT_DIR / "earnings_event_hybrid_refinement_summary.json"
    if not summary_path.exists():
        return {}, {}
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    best_negative = payload.get("best_negative_variant", {})
    best_wave = payload.get("best_wave_variant", {})
    try:
        neg_params = json.loads(best_negative.get("params_json", "{}"))
    except Exception:
        neg_params = {}
    try:
        wave_params = json.loads(best_wave.get("params_json", "{}"))
    except Exception:
        wave_params = {}
    return neg_params, wave_params


def _merge_params(*params_list: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for params in params_list:
        for key, value in params.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                updated = dict(merged[key])
                updated.update(value)
                merged[key] = updated
            else:
                merged[key] = value
    return merged


def _compound(events_df: pd.DataFrame) -> float:
    if events_df.empty:
        return 0.0
    vals = pd.to_numeric(events_df["total_return_pct"], errors="coerce").dropna()
    if vals.empty:
        return 0.0
    return float(((1.0 + vals / 100.0).prod() - 1.0) * 100.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward validation for Earnings Event Hybrid (Research).")
    parser.add_argument("--start", default="2024-04-04")
    parser.add_argument("--end", default="2026-05-01")
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    events = _load_unified_events(start, end).sort_values("reaction_date").reset_index(drop=True)

    neg_params, wave_params = _load_best_refinement_params()
    candidates = [
        ("baseline", {}),
        ("negative_only", neg_params),
        ("wave_only", wave_params),
        ("negative_plus_wave", _merge_params(neg_params, wave_params)),
    ]

    fold_size = max(1, len(events) // 4)
    train_test_slices: list[tuple[pd.DataFrame, pd.DataFrame, str]] = []
    for fold_idx in range(1, 4):
        train_end = fold_idx * fold_size
        test_end = min((fold_idx + 1) * fold_size, len(events))
        train = events.iloc[:train_end].copy()
        test = events.iloc[train_end:test_end].copy()
        if train.empty or test.empty:
            continue
        label = f"fold_{fold_idx}"
        train_test_slices.append((train, test, label))

    rows: list[dict[str, Any]] = []
    chosen_test_events: list[pd.DataFrame] = []
    fixed_baseline_events: list[pd.DataFrame] = []
    fixed_improved_events: list[pd.DataFrame] = []
    improved_params = _merge_params(neg_params, wave_params)

    for train, test, label in train_test_slices:
        train_scores: list[tuple[str, dict[str, Any], float]] = []
        for candidate_name, candidate_params in candidates:
            summary, _, _ = evaluate_unified_strategy(train, label=f"{label}_train_{candidate_name}", params=candidate_params)
            train_scores.append((candidate_name, candidate_params, float(summary.compounded_event_return_pct)))
        best_name, best_params, best_train_score = max(train_scores, key=lambda item: item[2])

        test_summary, test_events, _ = evaluate_unified_strategy(test, label=f"{label}_test_{best_name}", params=best_params)
        base_summary, base_events, _ = evaluate_unified_strategy(test, label=f"{label}_baseline", params={})
        improved_summary, improved_events, _ = evaluate_unified_strategy(
            test,
            label=f"{label}_improved",
            params=improved_params,
        )

        if not test_events.empty:
            chosen_test_events.append(test_events)
        if not base_events.empty:
            fixed_baseline_events.append(base_events)
        if not improved_events.empty:
            fixed_improved_events.append(improved_events)

        rows.append(
            {
                "fold": label,
                "train_events": int(len(train)),
                "test_events": int(len(test)),
                "chosen_candidate": best_name,
                "chosen_train_compounded_return_pct": best_train_score,
                "oos_compounded_return_pct": float(test_summary.compounded_event_return_pct),
                "oos_triggered_events": int(test_summary.triggered_events),
                "baseline_oos_compounded_return_pct": float(base_summary.compounded_event_return_pct),
                "improved_oos_compounded_return_pct": float(improved_summary.compounded_event_return_pct),
                "params_json": json.dumps(best_params, sort_keys=True),
            }
        )

    walkforward_df = pd.DataFrame(rows)
    chosen_all = pd.concat(chosen_test_events, ignore_index=True) if chosen_test_events else pd.DataFrame()
    baseline_all = pd.concat(fixed_baseline_events, ignore_index=True) if fixed_baseline_events else pd.DataFrame()
    improved_all = pd.concat(fixed_improved_events, ignore_index=True) if fixed_improved_events else pd.DataFrame()

    summary_payload = {
        "window": {"start": start.date().isoformat(), "end": end.date().isoformat()},
        "candidate_names": [name for name, _ in candidates],
        "selected_oos_compounded_return_pct": _compound(chosen_all),
        "selected_oos_triggered_events": int(len(chosen_all)),
        "fixed_baseline_oos_compounded_return_pct": _compound(baseline_all),
        "fixed_baseline_oos_triggered_events": int(len(baseline_all)),
        "fixed_improved_oos_compounded_return_pct": _compound(improved_all),
        "fixed_improved_oos_triggered_events": int(len(improved_all)),
        "folds": rows,
    }

    (ARTIFACT_DIR / "earnings_event_hybrid_walkforward.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )
    walkforward_df.to_csv(ARTIFACT_DIR / "earnings_event_hybrid_walkforward_folds.csv", index=False)
    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
