from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_earnings_hybrid_main import evaluate_hybrid_strategy
from scripts.validate_earnings_overshoot_first_dump import _load_labeled_events


ARTIFACT_DIR = ROOT / "artifacts" / "optimization"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def _continuation_event_count(events_df: pd.DataFrame) -> int:
    if events_df.empty or "trade_notes" not in events_df.columns:
        return 0
    return int(events_df["trade_notes"].astype(str).str.contains("regime=earnings_continuation_long", na=False).sum())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate whether a positive earnings continuation fallback improves the current positive hybrid."
    )
    parser.add_argument("--start", default="2024-04-04", help="Start date for event selection.")
    parser.add_argument("--end", default="2026-05-01", help="End date for event selection.")
    parser.add_argument(
        "--artifact-stem",
        default="earnings_positive_continuation",
        help="Artifact stem (default: earnings_positive_continuation).",
    )
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    events = _load_labeled_events(start, end)

    baseline_summary, baseline_events, _ = evaluate_hybrid_strategy(
        events,
        label="baseline_no_continuation",
        params={"continuation_enabled": False},
    )

    rows: list[dict] = []
    baseline_row = {
        "label": baseline_summary.label,
        "continuation_enabled": False,
        "continuation_gap_pct_min": None,
        "continuation_confirm5_min": None,
        "continuation_confirm15_min": None,
        "continuation_confirm30_min": None,
        "triggered_events": int(baseline_summary.triggered_events),
        "total_trades": int(baseline_summary.total_trades),
        "compounded_event_return_pct": float(baseline_summary.compounded_event_return_pct),
        "win_rate_pct": float(baseline_summary.win_rate_pct),
        "max_drawdown_pct": float(baseline_summary.max_drawdown_pct),
        "continuation_event_count": int(_continuation_event_count(baseline_events)),
    }
    rows.append(baseline_row)

    for gap_min in (4.0, 6.0):
        for confirm5_min in (0.0, 0.1, 0.25):
            for confirm15_min in (None, 0.0):
                for confirm30_min in (None, 0.0):
                    params = {
                        "continuation_enabled": True,
                        "continuation_gap_pct_min": gap_min,
                        "continuation_confirm5_min": confirm5_min,
                        "continuation_confirm15_min": confirm15_min,
                        "continuation_confirm30_min": confirm30_min,
                    }
                    summary, events_df, _ = evaluate_hybrid_strategy(
                        events,
                        label=(
                            f"cont_gap{gap_min:g}_c5{confirm5_min:g}_"
                            f"c15{('none' if confirm15_min is None else f'{float(confirm15_min):g}')}_"
                            f"c30{('none' if confirm30_min is None else f'{float(confirm30_min):g}')}"
                        ),
                        params=params,
                    )
                    rows.append(
                        {
                            "label": summary.label,
                            "continuation_enabled": True,
                            "continuation_gap_pct_min": gap_min,
                            "continuation_confirm5_min": confirm5_min,
                            "continuation_confirm15_min": confirm15_min,
                            "continuation_confirm30_min": confirm30_min,
                            "triggered_events": int(summary.triggered_events),
                            "total_trades": int(summary.total_trades),
                            "compounded_event_return_pct": float(summary.compounded_event_return_pct),
                            "win_rate_pct": float(summary.win_rate_pct),
                            "max_drawdown_pct": float(summary.max_drawdown_pct),
                            "continuation_event_count": int(_continuation_event_count(events_df)),
                        }
                    )

    results = pd.DataFrame(rows)
    best_any = (
        results[results["continuation_enabled"] == True]
        .sort_values(
            ["compounded_event_return_pct", "triggered_events", "continuation_event_count"],
            ascending=[False, False, False],
        )
        .head(1)
    )
    best_additive = (
        results[
            (results["continuation_enabled"] == True)
            & (results["triggered_events"] >= baseline_summary.triggered_events)
            & (results["continuation_event_count"] >= 5)
        ]
        .sort_values(
            ["compounded_event_return_pct", "triggered_events", "continuation_event_count"],
            ascending=[False, False, False],
        )
        .head(1)
    )

    payload = {
        "window": {"start": start.date().isoformat(), "end": end.date().isoformat()},
        "baseline": baseline_row,
        "best_any": best_any.iloc[0].to_dict() if not best_any.empty else None,
        "best_additive": best_additive.iloc[0].to_dict() if not best_additive.empty else None,
    }

    out_json = ARTIFACT_DIR / f"{args.artifact_stem}_summary.json"
    out_csv = ARTIFACT_DIR / f"{args.artifact_stem}_variants.csv"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    results.to_csv(out_csv, index=False)

    if payload["best_additive"] is not None:
        best = payload["best_additive"]
        print(
            f"best_additive: return={best['compounded_event_return_pct']:.3f}% "
            f"events={int(best['triggered_events'])} continuation_events={int(best['continuation_event_count'])}"
        )
    else:
        print("No additive continuation variant met the minimum selection criteria.")
    print(f"Wrote artifacts:\n- {out_json}\n- {out_csv}")


if __name__ == "__main__":
    main()
