from __future__ import annotations

import json
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
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

from scripts.validate_earnings_hybrid_main import evaluate_hybrid_strategy


ARTIFACT_DIR = ROOT / "artifacts" / "optimization"
BASE_EVENTS_PATH = ARTIFACT_DIR / "earnings_hybrid_main_engine_events.csv"


@dataclass
class VariantEval:
    label: str
    events: int
    total_trades: int
    win_rate_pct: float
    mean_event_return_pct: float
    median_event_return_pct: float
    compounded_event_return_pct: float
    max_drawdown_pct: float
    branch_counts: dict[str, int]
    params: dict[str, Any]


def _build_variant_label(params: dict[str, Any]) -> str:
    return (
        f"wb_open{params.get('wave_best_long_entry_min_close_from_open_pct')}_"
        f"wb_mom{params.get('wave_best_long_momentum_min', 0.0)}_"
        f"wr_mode{params.get('wave_robust_long_entry_mode', 'next_bar')}_"
        f"wr_open{params.get('wave_robust_long_entry_min_close_from_open_pct')}_"
        f"wr_mom{params.get('wave_robust_long_momentum_min', 0.0)}"
    )


def _load_baseline_triggered_events() -> pd.DataFrame:
    if not BASE_EVENTS_PATH.exists():
        raise SystemExit(f"Missing baseline events file: {BASE_EVENTS_PATH}")
    df = pd.read_csv(BASE_EVENTS_PATH)
    if df.empty:
        raise SystemExit("Baseline positive hybrid events file is empty.")
    df["reaction_date"] = pd.to_datetime(df["reaction_date"], errors="coerce")
    return df[
        [
            "symbol",
            "reaction_date",
            "timing",
            "surprise_pct",
            "peak_vs_prev_close_pct",
            "confirm15_close_from_open_pct",
            "confirm30_close_from_open_pct",
        ]
    ].copy()


def _summary_to_variant(summary, params: dict[str, Any]) -> VariantEval:
    return VariantEval(
        label=_build_variant_label(params),
        events=int(summary.triggered_events),
        total_trades=int(summary.total_trades),
        win_rate_pct=float(summary.win_rate_pct),
        mean_event_return_pct=float(summary.mean_event_return_pct),
        median_event_return_pct=float(summary.median_event_return_pct),
        compounded_event_return_pct=float(summary.compounded_event_return_pct),
        max_drawdown_pct=float(summary.max_drawdown_pct),
        branch_counts=dict(summary.branch_counts),
        params=dict(params),
    )


def main() -> None:
    events = _load_baseline_triggered_events()

    baseline_params: dict[str, Any] = {}
    baseline_summary, baseline_events, baseline_trades = evaluate_hybrid_strategy(
        events,
        label="positive_entry_quality_baseline",
        params=baseline_params,
    )
    baseline_eval = _summary_to_variant(baseline_summary, baseline_params)

    wave_best_open_opts: list[float | None] = [None, 0.0, 0.25]
    wave_best_mom_opts = [0.0, 0.25, 0.5]
    wave_robust_mode_opts = ["next_bar", "ema_turn"]
    wave_robust_open_opts: list[float | None] = [None, 0.0, 0.25]
    wave_robust_mom_opts = [0.0, 0.25, 0.5]

    variants: list[VariantEval] = []
    best_any: tuple[VariantEval, pd.DataFrame, pd.DataFrame] | None = None
    best_full13: tuple[VariantEval, pd.DataFrame, pd.DataFrame] | None = None

    for wb_open in wave_best_open_opts:
        for wb_mom in wave_best_mom_opts:
            for wr_mode in wave_robust_mode_opts:
                for wr_open in wave_robust_open_opts:
                    for wr_mom in wave_robust_mom_opts:
                        params = {
                            "wave_best_long_entry_min_close_from_open_pct": wb_open,
                            "wave_best_long_momentum_min": wb_mom,
                            "wave_robust_long_entry_mode": wr_mode,
                            "wave_robust_long_entry_min_close_from_open_pct": wr_open,
                            "wave_robust_long_momentum_min": wr_mom,
                        }
                        summary, ev_df, tr_df = evaluate_hybrid_strategy(
                            events,
                            label=_build_variant_label(params),
                            params=params,
                        )
                        eval_row = _summary_to_variant(summary, params)
                        variants.append(eval_row)
                        if best_any is None or eval_row.compounded_event_return_pct > best_any[0].compounded_event_return_pct:
                            best_any = (eval_row, ev_df, tr_df)
                        if eval_row.events >= 13 and (
                            best_full13 is None or eval_row.compounded_event_return_pct > best_full13[0].compounded_event_return_pct
                        ):
                            best_full13 = (eval_row, ev_df, tr_df)

    variants_df = pd.DataFrame([asdict(v) for v in variants]).sort_values(
        ["compounded_event_return_pct", "events", "win_rate_pct", "total_trades"],
        ascending=[False, False, False, False],
    )
    variants_df.to_csv(ARTIFACT_DIR / "earnings_positive_entry_quality_variants.csv", index=False)
    baseline_events.to_csv(ARTIFACT_DIR / "earnings_positive_entry_quality_baseline_events.csv", index=False)
    baseline_trades.to_csv(ARTIFACT_DIR / "earnings_positive_entry_quality_baseline_trades.csv", index=False)

    if best_any is not None:
        best_any[1].to_csv(ARTIFACT_DIR / "earnings_positive_entry_quality_best_any_events.csv", index=False)
        best_any[2].to_csv(ARTIFACT_DIR / "earnings_positive_entry_quality_best_any_trades.csv", index=False)
    if best_full13 is not None:
        best_full13[1].to_csv(ARTIFACT_DIR / "earnings_positive_entry_quality_best_full13_events.csv", index=False)
        best_full13[2].to_csv(ARTIFACT_DIR / "earnings_positive_entry_quality_best_full13_trades.csv", index=False)

    summary_payload = {
        "baseline": asdict(baseline_eval),
        "best_any": asdict(best_any[0]) if best_any is not None else None,
        "best_full13": asdict(best_full13[0]) if best_full13 is not None else None,
    }
    (ARTIFACT_DIR / "earnings_positive_entry_quality_summary.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
