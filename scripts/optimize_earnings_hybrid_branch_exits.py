from __future__ import annotations

import argparse
import json
import sys
import types
from dataclasses import asdict, dataclass
from itertools import product
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

from strategies.earnings_overshoot_hybrid_strategy import EarningsOvershootHybridStrategy
from scripts.validate_earnings_hybrid_main import _selector_branch, evaluate_hybrid_strategy
from scripts.validate_earnings_overshoot_first_dump import _load_labeled_events


ARTIFACT_DIR = ROOT / "artifacts" / "optimization"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class BranchVariant:
    branch: str
    label: str
    events: int
    trades: int
    win_rate_pct: float
    mean_event_return_pct: float
    compounded_event_return_pct: float
    params: dict[str, Any]


def _base_params() -> dict[str, Any]:
    return dict(EarningsOvershootHybridStrategy(params={}).default_params())


def _branch_subset(events: pd.DataFrame, branch: str, params: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event in events.to_dict(orient="records"):
        if _selector_branch(event, params) == branch:
            rows.append(event)
    return pd.DataFrame(rows)


def _evaluate_branch(
    branch_events: pd.DataFrame,
    *,
    branch: str,
    label: str,
    params: dict[str, Any],
) -> BranchVariant:
    summary, _, _ = evaluate_hybrid_strategy(branch_events, label=label, params=params)
    return BranchVariant(
        branch=branch,
        label=label,
        events=summary.triggered_events,
        trades=summary.total_trades,
        win_rate_pct=summary.win_rate_pct,
        mean_event_return_pct=summary.mean_event_return_pct,
        compounded_event_return_pct=summary.compounded_event_return_pct,
        params=dict(params),
    )


def _optimize_wave_best(branch_events: pd.DataFrame, base_params: dict[str, Any]) -> list[BranchVariant]:
    rows: list[BranchVariant] = []
    for short_exit_mode, short_rebound_exit_pct, long_entry_mode, long_momentum_min, long_exit_mode in product(
        ("time_10", "rebound"),
        (0.5, 1.0),
        ("ema_turn", "next_bar", "break_prev_high"),
        (0.0, 0.5, 1.0),
        ("ema_roll", "time_10", "time_20", "vwap_touch"),
    ):
        if long_entry_mode == "next_bar" and long_momentum_min > 0.0:
            continue
        params = dict(base_params)
        params["wave_best_short_exit_mode"] = short_exit_mode
        params["wave_best_short_rebound_exit_pct"] = short_rebound_exit_pct
        params["wave_best_long_entry_mode"] = long_entry_mode
        params["wave_best_long_momentum_min"] = long_momentum_min
        params["wave_best_long_exit_mode"] = long_exit_mode
        if long_exit_mode == "ema_roll":
            gain_values = (0.5, 1.0, 2.0)
            vwap_values = (-0.5,)
        elif long_exit_mode == "vwap_touch":
            gain_values = (1.0,)
            vwap_values = (-0.5, 0.0, 0.5)
        else:
            gain_values = (1.0,)
            vwap_values = (-0.5,)
        for ema_roll_gain_min in gain_values:
            for vwap_touch_buffer in vwap_values:
                variant_params = dict(params)
                variant_params["wave_best_long_ema_roll_gain_min"] = ema_roll_gain_min
                variant_params["wave_best_long_vwap_touch_buffer"] = vwap_touch_buffer
                label = (
                    f"wave_best_sexit{short_exit_mode}"
                    f"_sreb{short_rebound_exit_pct:g}"
                    f"_lentry{long_entry_mode}"
                    f"_lmom{long_momentum_min:.2f}"
                    f"_lexit{long_exit_mode}"
                    f"_lgain{ema_roll_gain_min:g}"
                    f"_lvwap{vwap_touch_buffer:g}"
                )
                rows.append(
                    _evaluate_branch(
                        branch_events,
                        branch="wave_best",
                        label=label,
                        params=variant_params,
                    )
                )
    return sorted(
        rows,
        key=lambda item: (item.compounded_event_return_pct, item.win_rate_pct, item.mean_event_return_pct),
        reverse=True,
    )


def _optimize_failed_reclaim(branch_events: pd.DataFrame, base_params: dict[str, Any]) -> list[BranchVariant]:
    rows: list[BranchVariant] = []
    for exit_mode in ("close_only", "time_10", "time_20", "time_30", "rebound", "ema_turn", "vwap_reclaim"):
        params = dict(base_params)
        params["failed_reclaim_short_exit_mode"] = exit_mode
        rebound_values = (0.5, 1.0) if exit_mode in {"rebound", "ema_turn"} else (0.5,)
        vwap_values = (0.0, 0.5) if exit_mode == "vwap_reclaim" else (0.0,)
        hold_values = (10, 20, 30) if exit_mode not in {"close_only"} else (20,)
        for rebound_exit_pct in rebound_values:
            for vwap_reclaim_buffer in vwap_values:
                for max_hold_bars in hold_values:
                    variant_params = dict(params)
                    variant_params["failed_reclaim_short_rebound_exit_pct"] = rebound_exit_pct
                    variant_params["failed_reclaim_short_vwap_reclaim_buffer"] = vwap_reclaim_buffer
                    variant_params["failed_reclaim_short_max_hold_bars"] = max_hold_bars
                    label = (
                        f"failed_reclaim_exit{exit_mode}"
                        f"_reb{rebound_exit_pct:g}"
                        f"_vwap{vwap_reclaim_buffer:g}"
                        f"_hold{max_hold_bars}"
                    )
                    rows.append(
                        _evaluate_branch(
                            branch_events,
                            branch="failed_reclaim",
                            label=label,
                            params=variant_params,
                        )
                    )
    return sorted(
        rows,
        key=lambda item: (item.compounded_event_return_pct, item.win_rate_pct, item.mean_event_return_pct),
        reverse=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize branch-specific exits for the integrated earnings overshoot hybrid strategy."
    )
    parser.add_argument("--start", default="2024-04-04", help="Start date for event selection (default: 2024-04-04).")
    parser.add_argument("--end", default="2026-05-01", help="End date for event selection (default: 2026-05-01).")
    parser.add_argument(
        "--artifact-stem",
        default="earnings_hybrid_branch_exit_optimization",
        help="Artifact stem (default: earnings_hybrid_branch_exit_optimization).",
    )
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    all_events = _load_labeled_events(start, end)
    base_params = _base_params()

    baseline_summary, baseline_events_df, _ = evaluate_hybrid_strategy(
        all_events,
        label="baseline_integrated_current",
        params=base_params,
    )
    wave_best_events = _branch_subset(all_events, "wave_best", base_params)
    failed_reclaim_events = _branch_subset(all_events, "failed_reclaim", base_params)

    wave_best_results = _optimize_wave_best(wave_best_events, base_params) if len(wave_best_events) >= 3 else []
    failed_reclaim_results = (
        _optimize_failed_reclaim(failed_reclaim_events, base_params) if len(failed_reclaim_events) >= 3 else []
    )

    best_wave_best = wave_best_results[0] if wave_best_results else None
    best_failed_reclaim = failed_reclaim_results[0] if failed_reclaim_results else None

    combined_params = dict(base_params)
    if best_wave_best is not None:
        combined_params.update(best_wave_best.params)
    if best_failed_reclaim is not None:
        combined_params.update(best_failed_reclaim.params)

    optimized_summary, optimized_events_df, optimized_trades_df = evaluate_hybrid_strategy(
        all_events,
        label="optimized_integrated_candidate",
        params=combined_params,
    )

    out_json = ARTIFACT_DIR / f"{args.artifact_stem}.json"
    out_wave_best = ARTIFACT_DIR / f"{args.artifact_stem}_wave_best_variants.csv"
    out_failed_reclaim = ARTIFACT_DIR / f"{args.artifact_stem}_failed_reclaim_variants.csv"
    out_events = ARTIFACT_DIR / f"{args.artifact_stem}_optimized_events.csv"
    out_trades = ARTIFACT_DIR / f"{args.artifact_stem}_optimized_trades.csv"

    pd.DataFrame([asdict(item) for item in wave_best_results]).to_csv(out_wave_best, index=False)
    pd.DataFrame([asdict(item) for item in failed_reclaim_results]).to_csv(out_failed_reclaim, index=False)
    optimized_events_df.to_csv(out_events, index=False)
    optimized_trades_df.to_csv(out_trades, index=False)

    payload = {
        "window": {"start": start.date().isoformat(), "end": end.date().isoformat()},
        "notes": [
            "This pass keeps the selector fixed and only optimizes exit/leg handling inside existing branches.",
            "To limit overfitting, only branches with at least three triggered event-days are optimized.",
            "Wave_robust and short_only remain unchanged because their current sample is too thin to justify retuning.",
        ],
        "baseline_summary": asdict(baseline_summary),
        "optimized_summary": asdict(optimized_summary),
        "baseline_branch_counts": {
            str(k): int(v)
            for k, v in baseline_events_df["selected_branch"].value_counts(dropna=False).to_dict().items()
        }
        if not baseline_events_df.empty
        else {},
        "best_wave_best": asdict(best_wave_best) if best_wave_best is not None else None,
        "best_failed_reclaim": asdict(best_failed_reclaim) if best_failed_reclaim is not None else None,
        "optimized_params": combined_params,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        f"baseline={baseline_summary.compounded_event_return_pct:.3f}% "
        f"optimized={optimized_summary.compounded_event_return_pct:.3f}%"
    )
    print(f"Wrote artifacts:\n- {out_json}\n- {out_wave_best}\n- {out_failed_reclaim}\n- {out_events}\n- {out_trades}")


if __name__ == "__main__":
    main()
