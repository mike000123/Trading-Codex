from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "optimization"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

EVENTS_PATH = ARTIFACT_DIR / "earnings_overshoot_dump_events_labeled.csv"
HYBRID_V1_JSON_PATH = ARTIFACT_DIR / "earnings_hybrid_selector.json"
HYBRID_V1_MERGED_EVENTS_PATH = ARTIFACT_DIR / "earnings_hybrid_selector_merged_events.csv"
WAVE_EVENTS_PATH = ARTIFACT_DIR / "earnings_wave_sequence_main_engine_events.csv"
LEG_SPLIT_JSON_PATH = ARTIFACT_DIR / "earnings_wave_leg_split_engine.json"
LEG_SPLIT_EVENTS_PATH = ARTIFACT_DIR / "earnings_wave_leg_split_engine_events.csv"


def _compound_return_pct(returns_pct: pd.Series) -> float:
    vals = pd.to_numeric(returns_pct, errors="coerce").dropna()
    if vals.empty:
        return 0.0
    return float(((1.0 + vals / 100.0).prod() - 1.0) * 100.0)


@dataclass
class SelectorEval:
    label: str
    trades: int
    win_rate_pct: float
    mean_return_pct: float
    median_return_pct: float
    compounded_return_pct: float
    coverage_pct: float
    selector_details: dict[str, Any]


def _load_base_events(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    df = pd.read_csv(EVENTS_PATH)
    if df.empty:
        return df
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["reaction_date"] = pd.to_datetime(df["reaction_date"], errors="coerce").dt.date.astype(str)
    df["timing"] = df["timing"].astype(str).str.lower()
    df = df[pd.to_numeric(df["surprise_pct"], errors="coerce") > 0].copy()
    df = df[df["timing"].isin({"bmo", "amc"})].copy()
    reaction_ts = pd.to_datetime(df["reaction_date"], errors="coerce")
    df = df[(reaction_ts >= start) & (reaction_ts <= end)].copy()
    return df.reset_index(drop=True)


def _load_failed_reclaim_best() -> pd.DataFrame:
    payload = json.loads(HYBRID_V1_JSON_PATH.read_text(encoding="utf-8"))
    label = payload.get("best_failed_reclaim_branch", {}).get("label")
    df = pd.read_csv(HYBRID_V1_MERGED_EVENTS_PATH)
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["reaction_date"] = pd.to_datetime(df["reaction_date"], errors="coerce").dt.date.astype(str)
    out = df.loc[:, ["symbol", "reaction_date", "failed_reclaim_return_pct"]].copy()
    out.attrs["label"] = label
    return out


def _load_wave_variants() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(WAVE_EVENTS_PATH)
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["reaction_date"] = pd.to_datetime(df["reaction_date"], errors="coerce").dt.date.astype(str)
    wave_best = df[df["variant"] == "wave_sequence_best_return"].loc[
        :, ["symbol", "reaction_date", "total_return_pct"]
    ].rename(columns={"total_return_pct": "wave_best_return_pct"})
    wave_robust = df[df["variant"] == "wave_sequence_robust"].loc[
        :, ["symbol", "reaction_date", "total_return_pct"]
    ].rename(columns={"total_return_pct": "wave_robust_return_pct"})
    return wave_best, wave_robust


def _load_short_only_best() -> pd.DataFrame:
    payload = json.loads(LEG_SPLIT_JSON_PATH.read_text(encoding="utf-8"))
    label = payload.get("best_short_only", {}).get("label")
    df = pd.read_csv(LEG_SPLIT_EVENTS_PATH)
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["reaction_date"] = pd.to_datetime(df["reaction_date"], errors="coerce").dt.date.astype(str)
    out = df[df["variant"] == label].loc[:, ["symbol", "reaction_date", "total_return_pct"]].rename(
        columns={"total_return_pct": "short_only_return_pct"}
    )
    out.attrs["label"] = label
    return out


def _merge_branch_returns(base_events: pd.DataFrame) -> pd.DataFrame:
    merged = base_events.copy()
    fr = _load_failed_reclaim_best()
    wave_best, wave_robust = _load_wave_variants()
    short_only = _load_short_only_best()
    merged = merged.merge(fr, on=["symbol", "reaction_date"], how="left")
    merged = merged.merge(wave_best, on=["symbol", "reaction_date"], how="left")
    merged = merged.merge(wave_robust, on=["symbol", "reaction_date"], how="left")
    merged = merged.merge(short_only, on=["symbol", "reaction_date"], how="left")
    for col in (
        "failed_reclaim_return_pct",
        "wave_best_return_pct",
        "wave_robust_return_pct",
        "short_only_return_pct",
    ):
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    return merged


def _evaluate_selection(
    df: pd.DataFrame,
    branch_choices: pd.Series,
    label: str,
    details: dict[str, Any],
) -> tuple[SelectorEval, pd.DataFrame]:
    chosen = df.copy()
    chosen["selected_branch"] = branch_choices
    chosen["selected_return_pct"] = np.nan
    branch_to_col = {
        "failed_reclaim": "failed_reclaim_return_pct",
        "wave_best": "wave_best_return_pct",
        "wave_robust": "wave_robust_return_pct",
        "short_only": "short_only_return_pct",
    }
    for branch, col in branch_to_col.items():
        chosen.loc[chosen["selected_branch"] == branch, "selected_return_pct"] = chosen[col]

    returns = pd.to_numeric(chosen["selected_return_pct"], errors="coerce").dropna()
    trades = int(len(returns))
    coverage_pct = float(trades / len(chosen) * 100.0) if len(chosen) > 0 else 0.0
    if returns.empty:
        eval_row = SelectorEval(label, 0, np.nan, np.nan, np.nan, 0.0, 0.0, details)
        return eval_row, chosen

    eval_row = SelectorEval(
        label=label,
        trades=trades,
        win_rate_pct=float((returns > 0).mean() * 100.0),
        mean_return_pct=float(returns.mean()),
        median_return_pct=float(returns.median()),
        compounded_return_pct=float(_compound_return_pct(returns)),
        coverage_pct=coverage_pct,
        selector_details=details,
    )
    return eval_row, chosen


def _build_current_selector(df: pd.DataFrame, peak_threshold: float = 21.0) -> pd.Series:
    peak = pd.to_numeric(df["peak_vs_prev_close_pct"], errors="coerce")
    return pd.Series(
        np.where(peak >= peak_threshold, "wave_best", "failed_reclaim"),
        index=df.index,
    )


def _search_three_way_selectors(df: pd.DataFrame) -> tuple[list[SelectorEval], dict[str, pd.DataFrame]]:
    results: list[SelectorEval] = []
    frames: dict[str, pd.DataFrame] = {}

    baseline_choices = {
        "baseline_failed_reclaim": pd.Series(["failed_reclaim"] * len(df), index=df.index),
        "baseline_wave_best": pd.Series(["wave_best"] * len(df), index=df.index),
        "baseline_wave_robust": pd.Series(["wave_robust"] * len(df), index=df.index),
        "baseline_short_only": pd.Series(["short_only"] * len(df), index=df.index),
        "baseline_current_hybrid_v1": _build_current_selector(df, 21.0),
    }
    for label, choice in baseline_choices.items():
        row, frame = _evaluate_selection(
            df,
            choice,
            label,
            {"mode": "baseline", "label": label},
        )
        results.append(row)
        frames[label] = frame

    for peak_th in (20.0, 20.5, 21.0, 22.0, 24.0):
        for weak15 in (-1.0, -0.5, 0.0, 0.5):
            for weak30 in (-2.0, -1.0, -0.5, 0.0, 0.5):
                for strong15 in (0.5, 1.0, 1.5, 2.0):
                    for strong30 in (0.5, 1.0, 1.5, 2.0):
                        for mid_branch in ("wave_best", "failed_reclaim"):
                            peak = pd.to_numeric(df["peak_vs_prev_close_pct"], errors="coerce")
                            c15 = pd.to_numeric(df["confirm15_close_from_open_pct"], errors="coerce")
                            c30 = pd.to_numeric(df["confirm30_close_from_open_pct"], errors="coerce")

                            choices: list[str] = []
                            for idx in df.index:
                                peak_v = peak.loc[idx]
                                c15_v = c15.loc[idx]
                                c30_v = c30.loc[idx]
                                if np.isfinite(peak_v) and peak_v >= peak_th:
                                    if np.isfinite(c15_v) and np.isfinite(c30_v) and c15_v <= weak15 and c30_v <= weak30:
                                        choices.append("short_only")
                                    elif np.isfinite(c15_v) and np.isfinite(c30_v) and c15_v >= strong15 and c30_v >= strong30:
                                        choices.append("wave_robust")
                                    else:
                                        choices.append(mid_branch)
                                else:
                                    choices.append("failed_reclaim")

                            label = (
                                f"selector3_peak{peak_th:g}"
                                f"_weak15{weak15:g}"
                                f"_weak30{weak30:g}"
                                f"_strong15{strong15:g}"
                                f"_strong30{strong30:g}"
                                f"_mid{mid_branch}"
                            )
                            row, frame = _evaluate_selection(
                                df,
                                pd.Series(choices, index=df.index),
                                label,
                                {
                                    "mode": "three_way_shape",
                                    "peak_vs_prev_close_ge": peak_th,
                                    "weak_confirm15_le": weak15,
                                    "weak_confirm30_le": weak30,
                                    "strong_confirm15_ge": strong15,
                                    "strong_confirm30_ge": strong30,
                                    "weak_branch": "short_only",
                                    "strong_branch": "wave_robust",
                                    "mid_branch": mid_branch,
                                    "fallback_branch": "failed_reclaim",
                                },
                            )
                            if row.trades >= 10:
                                results.append(row)
                                frames[label] = frame

    results = sorted(
        results,
        key=lambda item: (item.compounded_return_pct, item.win_rate_pct, item.coverage_pct),
        reverse=True,
    )
    return results, frames


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a broader three-shape earnings overshoot selector using early-tape features."
    )
    parser.add_argument("--start", default="2024-04-04", help="Event-study start date (default: 2024-04-04).")
    parser.add_argument("--end", default="2026-05-01", help="Event-study end date (default: 2026-05-01).")
    parser.add_argument(
        "--artifact-stem",
        default="earnings_hybrid_selector_v2",
        help="Artifact stem (default: earnings_hybrid_selector_v2).",
    )
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    base_events = _load_base_events(start, end)
    merged = _merge_branch_returns(base_events)
    selector_results, selector_frames = _search_three_way_selectors(merged)
    selector_df = pd.DataFrame([asdict(item) for item in selector_results])
    best_selector = selector_results[0] if selector_results else None
    best_frame = selector_frames.get(best_selector.label).copy() if best_selector is not None else pd.DataFrame()

    out_json = ARTIFACT_DIR / f"{args.artifact_stem}.json"
    out_selectors = ARTIFACT_DIR / f"{args.artifact_stem}_selector_variants.csv"
    out_events = ARTIFACT_DIR / f"{args.artifact_stem}_best_events.csv"
    out_merged = ARTIFACT_DIR / f"{args.artifact_stem}_merged_events.csv"

    payload = {
        "window": {"start": start.date().isoformat(), "end": end.date().isoformat()},
        "notes": [
            "This pass keeps the existing earnings-event gate and existing validated branch returns.",
            "It tests a simple three-shape selector rather than ticker-specific exceptions.",
            "Large overshoots route to short-only if the early tape is weak, to robust wave if the early tape is strong, and otherwise to the original wave branch.",
            "Smaller overshoots still route to the failed-reclaim branch.",
            "To limit overfitting, the search only varies a compact set of thresholds over peak-vs-prior-close and confirm15/confirm30 strength.",
            "Selectors with fewer than 10 triggered event-days are excluded from the ranking.",
        ],
        "top_selectors": selector_df.head(20).to_dict(orient="records"),
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    selector_df.to_csv(out_selectors, index=False)
    merged.to_csv(out_merged, index=False)
    if not best_frame.empty:
        best_frame.to_csv(out_events, index=False)

    if best_selector is not None:
        print(
            f"best_selector: {best_selector.label} "
            f"return={best_selector.compounded_return_pct:.3f}% "
            f"coverage={best_selector.coverage_pct:.1f}% "
            f"trades={best_selector.trades}"
        )
    print(f"Wrote results to {out_json}")


if __name__ == "__main__":
    main()
