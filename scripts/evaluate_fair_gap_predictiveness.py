"""
Broad validation harness for GLD fair-value gap usefulness.

This script answers two different questions:

1. Does the fair-value gap itself predict future GLD reversion broadly?
2. Do simple gap + daily RSI event rules show repeatable edge?

It intentionally works on daily fair-value output so it can use the longer
GLD history sourced from the custom daily target file when available.
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from dataclasses import dataclass
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
        def info(self, *args, **kwargs): pass
        def warning(self, *args, **kwargs): pass
        def error(self, *args, **kwargs): pass
        def debug(self, *args, **kwargs): pass

    logger_mod.log = _Dummy()
    sys.modules["core.logger"] = logger_mod


_install_dummy_logger()

from data.fair_value import compute_gld_fair_value_diagnostics


OUT_DIR = ROOT / "artifacts" / "optimization"


@dataclass
class EventStats:
    side: str
    gap_threshold: float
    rsi_threshold: float
    signals: int
    mean_return_pct: float
    median_return_pct: float
    win_rate_pct: float
    avg_hold_days: float
    fair_touch_rate_pct: float


def _rsi(series: pd.Series, period: int = 9) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    avg_up = up.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_down = down.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_up / avg_down.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.clip(0.0, 100.0)


def _prepare_daily_frame(cache_root: str | Path) -> tuple[dict[str, Any], pd.DataFrame]:
    diagnostics = compute_gld_fair_value_diagnostics(cache_root=cache_root)
    if diagnostics is None or diagnostics.frame.empty:
        raise SystemExit("Could not build GLD fair-value diagnostics from cache.")

    frame = diagnostics.frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["actual"] = pd.to_numeric(frame["actual"], errors="coerce")
    frame["fair_value"] = pd.to_numeric(frame["fair_value"], errors="coerce")
    frame = frame.dropna(subset=["date", "actual", "fair_value"]).sort_values("date").reset_index(drop=True)
    if "fair_gap_pct" not in frame.columns:
        frame["fair_gap_pct"] = (
            frame["fair_value"] / frame["actual"].replace(0.0, np.nan) - 1.0
        ) * 100.0
    frame["rsi9"] = _rsi(frame["actual"], period=9)
    frame["future_1d"] = (frame["actual"].shift(-1) / frame["actual"] - 1.0) * 100.0
    frame["future_3d"] = (frame["actual"].shift(-3) / frame["actual"] - 1.0) * 100.0
    frame["future_5d"] = (frame["actual"].shift(-5) / frame["actual"] - 1.0) * 100.0
    frame["future_10d"] = (frame["actual"].shift(-10) / frame["actual"] - 1.0) * 100.0
    return diagnostics.model, frame


def _horizon_metrics(frame: pd.DataFrame, horizons: tuple[int, ...] = (1, 3, 5, 10)) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    gap = frame["fair_gap_pct"].astype(float)
    for horizon in horizons:
        fwd = (frame["actual"].shift(-horizon) / frame["actual"] - 1.0) * 100.0
        valid = pd.DataFrame({"gap": gap, "fwd": fwd}).dropna()
        if valid.empty:
            continue
        low = float(valid["gap"].quantile(0.2))
        high = float(valid["gap"].quantile(0.8))
        over = valid.loc[valid["gap"] <= low, "fwd"]
        under = valid.loc[valid["gap"] >= high, "fwd"]
        rows.append(
            {
                "horizon_days": horizon,
                "observations": int(len(valid)),
                "gap_future_corr": float(valid["gap"].corr(valid["fwd"])),
                "directional_hit_pct": float(
                    (np.sign(valid["gap"].fillna(0.0)) == np.sign(valid["fwd"].fillna(0.0))).mean() * 100.0
                ),
                "undervalued_quintile_mean_fwd_pct": float(under.mean()) if not under.empty else np.nan,
                "overvalued_quintile_mean_fwd_pct": float(over.mean()) if not over.empty else np.nan,
                "quintile_spread_pct": (
                    float(under.mean() - over.mean())
                    if not under.empty and not over.empty
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _rolling_gap_metrics(
    frame: pd.DataFrame,
    *,
    window_days: int = 365,
    step_days: int = 90,
    horizon_days: int = 5,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    start = frame["date"].min().normalize()
    end = frame["date"].max().normalize()
    cursor = start
    while cursor + pd.Timedelta(days=window_days - 1) <= end:
        window_end = cursor + pd.Timedelta(days=window_days - 1)
        window = frame[(frame["date"] >= cursor) & (frame["date"] <= window_end)].copy()
        valid = window[["fair_gap_pct", "actual"]].copy()
        valid["future"] = (window["actual"].shift(-horizon_days) / window["actual"] - 1.0) * 100.0
        valid = valid.dropna()
        if len(valid) >= 40:
            low = float(valid["fair_gap_pct"].quantile(0.2))
            high = float(valid["fair_gap_pct"].quantile(0.8))
            under = valid.loc[valid["fair_gap_pct"] >= high, "future"]
            over = valid.loc[valid["fair_gap_pct"] <= low, "future"]
            rows.append(
                {
                    "window_start": cursor.date().isoformat(),
                    "window_end": window_end.date().isoformat(),
                    "observations": int(len(valid)),
                    "gap_future_corr": float(valid["fair_gap_pct"].corr(valid["future"])),
                    "directional_hit_pct": float(
                        (np.sign(valid["fair_gap_pct"].fillna(0.0)) == np.sign(valid["future"].fillna(0.0))).mean() * 100.0
                    ),
                    "quintile_spread_pct": (
                        float(under.mean() - over.mean())
                        if not under.empty and not over.empty
                        else np.nan
                    ),
                }
            )
        cursor = cursor + pd.Timedelta(days=step_days)
    return pd.DataFrame(rows)


def _fair_touch_event_study(
    frame: pd.DataFrame,
    *,
    side: str,
    gap_thresholds: tuple[float, ...],
    rsi_thresholds: tuple[float, ...],
    max_hold_days: int = 5,
) -> pd.DataFrame:
    rows: list[EventStats] = []
    for gap_threshold in gap_thresholds:
        for rsi_threshold in rsi_thresholds:
            returns: list[float] = []
            hold_days: list[int] = []
            touched: list[int] = []
            idx = 0
            last_entry_idx = len(frame) - max_hold_days - 1
            while idx <= last_entry_idx:
                row = frame.iloc[idx]
                gap = float(row["fair_gap_pct"])
                rsi = float(row["rsi9"]) if not pd.isna(row["rsi9"]) else np.nan
                actual = float(row["actual"])
                fair = float(row["fair_value"])

                if side == "short":
                    qualifies = gap <= -gap_threshold and not np.isnan(rsi) and rsi >= rsi_threshold
                else:
                    qualifies = gap >= gap_threshold and not np.isnan(rsi) and rsi <= rsi_threshold
                if not qualifies:
                    idx += 1
                    continue

                exit_idx = min(idx + max_hold_days, len(frame) - 1)
                fair_touched = False
                for look_ahead in range(idx + 1, exit_idx + 1):
                    look = frame.iloc[look_ahead]
                    look_actual = float(look["actual"])
                    look_fair = float(look["fair_value"])
                    if side == "short" and look_actual <= look_fair:
                        exit_idx = look_ahead
                        fair_touched = True
                        break
                    if side == "long" and look_actual >= look_fair:
                        exit_idx = look_ahead
                        fair_touched = True
                        break

                exit_price = float(frame.iloc[exit_idx]["actual"])
                if side == "short":
                    ret = (actual / exit_price - 1.0) * 100.0
                else:
                    ret = (exit_price / actual - 1.0) * 100.0

                returns.append(ret)
                hold_days.append(exit_idx - idx)
                touched.append(1 if fair_touched else 0)
                idx = exit_idx + 1

            if not returns:
                rows.append(
                    EventStats(
                        side=side,
                        gap_threshold=float(gap_threshold),
                        rsi_threshold=float(rsi_threshold),
                        signals=0,
                        mean_return_pct=np.nan,
                        median_return_pct=np.nan,
                        win_rate_pct=np.nan,
                        avg_hold_days=np.nan,
                        fair_touch_rate_pct=np.nan,
                    )
                )
                continue

            rows.append(
                EventStats(
                    side=side,
                    gap_threshold=float(gap_threshold),
                    rsi_threshold=float(rsi_threshold),
                    signals=len(returns),
                    mean_return_pct=float(np.mean(returns)),
                    median_return_pct=float(np.median(returns)),
                    win_rate_pct=float(np.mean(np.array(returns) > 0.0) * 100.0),
                    avg_hold_days=float(np.mean(hold_days)),
                    fair_touch_rate_pct=float(np.mean(touched) * 100.0),
                )
            )
    return pd.DataFrame([row.__dict__ for row in rows]).sort_values(
        ["mean_return_pct", "win_rate_pct", "signals"],
        ascending=[False, False, False],
        na_position="last",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate broad GLD fair-gap predictiveness.")
    parser.add_argument("--cache-root", default="data_cache")
    parser.add_argument("--max-hold-days", type=int, default=5)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model, frame = _prepare_daily_frame(args.cache_root)
    horizon_df = _horizon_metrics(frame)
    rolling_df = _rolling_gap_metrics(frame)
    short_df = _fair_touch_event_study(
        frame,
        side="short",
        gap_thresholds=(2.0, 3.0, 4.0, 5.0),
        rsi_thresholds=(75.0, 80.0, 85.0),
        max_hold_days=args.max_hold_days,
    )
    long_df = _fair_touch_event_study(
        frame,
        side="long",
        gap_thresholds=(2.0, 3.0, 4.0),
        rsi_thresholds=(35.0, 30.0, 25.0),
        max_hold_days=args.max_hold_days,
    )

    meta = {
        "date_start": str(frame["date"].min()),
        "date_end": str(frame["date"].max()),
        "rows": int(len(frame)),
        "model": model,
    }

    (OUT_DIR / "fair_gap_predictiveness_meta.json").write_text(
        json.dumps(meta, indent=2, default=str),
        encoding="utf-8",
    )
    horizon_df.to_csv(OUT_DIR / "fair_gap_predictiveness_horizons.csv", index=False)
    rolling_df.to_csv(OUT_DIR / "fair_gap_predictiveness_rolling_12m.csv", index=False)
    short_df.to_csv(OUT_DIR / "fair_gap_predictiveness_short_grid.csv", index=False)
    long_df.to_csv(OUT_DIR / "fair_gap_predictiveness_long_grid.csv", index=False)

    print("Model:")
    print(json.dumps(meta, indent=2, default=str))
    print("\nHorizon metrics:")
    print(horizon_df.to_string(index=False))
    print("\nTop short event-study rows:")
    print(short_df.head(8).to_string(index=False))
    print("\nTop long event-study rows:")
    print(long_df.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
