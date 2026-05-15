from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.equity_universes import NASDAQ100_TICKERS


ARTIFACT_DIR = ROOT / "artifacts" / "optimization"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
ALPACA_DIR = ROOT / "data_cache" / "alpaca"

QQQ_SYMBOL = "QQQ"
WINDOW_START = pd.Timestamp("2024-04-01")
WINDOW_END = pd.Timestamp("2026-04-30 23:59:00")
PREMARKET_START_MINUTE = 8 * 60
RTH_OPEN_MINUTE = 9 * 60 + 30
RTH_CLOSE_MINUTE = 16 * 60
MEGA_CAP_PROXY = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "GOOG",
    "GOOGL",
    "AVGO",
    "TSLA",
    "COST",
]
EXIT_MINUTE_LABELS = {
    9 * 60 + 45: "0945",
    10 * 60: "1000",
    11 * 60: "1100",
    16 * 60: "1600",
}
ENTRY_COST_PCT = {
    "09:29": 0.12,  # slightly more conservative for a premarket fill
    "09:30": 0.08,  # aligned with the project's usual round-trip assumption
}
FOLDS = [
    ("2024-04-01", "2024-09-30"),
    ("2024-10-01", "2025-03-31"),
    ("2025-04-01", "2025-09-30"),
    ("2025-10-01", "2026-04-30"),
]


@dataclass
class StrategyResult:
    variant: str
    trades: int
    long_trades: int
    short_trades: int
    win_rate_pct: float
    mean_return_pct: float
    median_return_pct: float
    compounded_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float


def _safe_pct(current, base) -> float:
    try:
        cur = float(current)
        ref = float(base)
    except Exception:
        return np.nan
    if not np.isfinite(cur) or not np.isfinite(ref) or ref == 0:
        return np.nan
    return (cur / ref - 1.0) * 100.0


def _load_intraday(symbol: str) -> pd.DataFrame | None:
    path = ALPACA_DIR / symbol / "1Min.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path, usecols=["date", "open", "close"])
    if frame.empty:
        return None
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    frame = frame[(frame["date"] >= WINDOW_START) & (frame["date"] <= WINDOW_END)].copy()
    if frame.empty:
        return None
    frame["session_date"] = frame["date"].dt.normalize()
    frame["minute"] = frame["date"].dt.hour * 60 + frame["date"].dt.minute
    return frame


def _session_metric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _build_symbol_daily(symbol: str) -> pd.DataFrame | None:
    frame = _load_intraday(symbol)
    if frame is None or frame.empty:
        return None

    pre = frame[(frame["minute"] >= PREMARKET_START_MINUTE) & (frame["minute"] < RTH_OPEN_MINUTE)]
    rth = frame[(frame["minute"] >= RTH_OPEN_MINUTE) & (frame["minute"] <= RTH_CLOSE_MINUTE)]

    pre_929 = pre.groupby("session_date")["close"].last().rename("pre_929_close")
    rth_close = rth.groupby("session_date")["close"].last().rename("rth_close")
    open_930 = frame[frame["minute"] == RTH_OPEN_MINUTE].groupby("session_date")["open"].first().rename("open_930")

    out = pd.concat([pre_929, rth_close, open_930], axis=1).sort_index()
    for exit_minute, label in EXIT_MINUTE_LABELS.items():
        out[f"close_{label}"] = (
            frame[frame["minute"] == exit_minute]
            .groupby("session_date")["close"]
            .last()
        )

    out["prev_rth_close"] = out["rth_close"].shift(1)
    out["pm_gap_pct"] = (_session_metric(out["pre_929_close"]) / _session_metric(out["prev_rth_close"]) - 1.0) * 100.0
    out["open_gap_pct"] = (_session_metric(out["open_930"]) / _session_metric(out["prev_rth_close"]) - 1.0) * 100.0
    out = out.reset_index().rename(columns={"session_date": "date"})
    out["symbol"] = symbol
    return out


def _build_daily_feature_table() -> pd.DataFrame:
    qqq = _build_symbol_daily(QQQ_SYMBOL)
    if qqq is None or qqq.empty:
        raise FileNotFoundError("QQQ 1Min Alpaca cache is required for this validator.")

    aggregate: dict[pd.Timestamp, dict[str, list[float]]] = {}
    available_symbols: list[str] = []
    mega_set = set(MEGA_CAP_PROXY)

    for symbol in NASDAQ100_TICKERS:
        daily = _build_symbol_daily(symbol)
        if daily is None or daily.empty:
            continue
        available_symbols.append(symbol)
        is_mega = symbol in mega_set
        for row in daily.itertuples(index=False):
            gap = getattr(row, "pm_gap_pct", np.nan)
            if not np.isfinite(gap):
                continue
            rec = aggregate.setdefault(
                pd.Timestamp(row.date),
                {
                    "all_gaps": [],
                    "mega_gaps": [],
                },
            )
            rec["all_gaps"].append(float(gap))
            if is_mega:
                rec["mega_gaps"].append(float(gap))

    rows: list[dict[str, Any]] = []
    qqq_map = {
        pd.Timestamp(row.date): row
        for row in qqq.itertuples(index=False)
    }
    for date, bucket in sorted(aggregate.items()):
        q = qqq_map.get(pd.Timestamp(date))
        if q is None:
            continue
        all_gaps = bucket["all_gaps"]
        mega_gaps = bucket["mega_gaps"]
        if len(all_gaps) < 25:
            continue

        row = {
            "date": pd.Timestamp(date),
            "all_count": int(len(all_gaps)),
            "mega_count": int(len(mega_gaps)),
            "ew_gap_pct": float(np.mean(all_gaps)),
            "median_gap_pct": float(np.median(all_gaps)),
            "breadth_up": float(np.mean([g > 0 for g in all_gaps])),
            "breadth_down": float(np.mean([g < 0 for g in all_gaps])),
            "mega_gap_pct": float(np.mean(mega_gaps)) if mega_gaps else np.nan,
            "qqq_prev_close": float(q.prev_rth_close) if np.isfinite(q.prev_rth_close) else np.nan,
            "qqq_pre_929_close": float(q.pre_929_close) if np.isfinite(q.pre_929_close) else np.nan,
            "qqq_open_930": float(q.open_930) if np.isfinite(q.open_930) else np.nan,
            "qqq_pm_gap_pct": float(q.pm_gap_pct) if np.isfinite(q.pm_gap_pct) else np.nan,
            "qqq_open_gap_pct": float(q.open_gap_pct) if np.isfinite(q.open_gap_pct) else np.nan,
        }
        for label in EXIT_MINUTE_LABELS.values():
            price = getattr(q, f"close_{label}", np.nan)
            row[f"qqq_close_{label}"] = float(price) if np.isfinite(price) else np.nan
        rows.append(row)

    daily = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    if daily.empty:
        raise RuntimeError("No overlapping Nasdaq100 + QQQ premarket sessions were found.")

    for source in ("ew_gap_pct", "mega_gap_pct", "median_gap_pct"):
        daily[f"{source}_vs_qqq_pm_pct"] = daily[source] - daily["qqq_pm_gap_pct"]
        daily[f"{source}_vs_qqq_open_pct"] = daily[source] - daily["qqq_open_gap_pct"]
    return daily


def _compound_return_pct(returns_pct: pd.Series, starting_equity: float = 1000.0) -> float:
    clean = pd.to_numeric(returns_pct, errors="coerce").dropna()
    if clean.empty:
        return 0.0
    equity = float(starting_equity)
    for ret in clean:
        equity *= 1.0 + (float(ret) / 100.0)
    return (equity / float(starting_equity) - 1.0) * 100.0


def _max_drawdown_pct(returns_pct: pd.Series, starting_equity: float = 1000.0) -> float:
    clean = pd.to_numeric(returns_pct, errors="coerce").dropna()
    if clean.empty:
        return 0.0
    equity = float(starting_equity)
    curve: list[float] = [equity]
    for ret in clean:
        equity *= 1.0 + (float(ret) / 100.0)
        curve.append(equity)
    ser = pd.Series(curve)
    peak = ser.cummax()
    dd = ser / peak - 1.0
    return float(dd.min() * 100.0)


def _sharpe_ratio(returns_pct: pd.Series) -> float:
    clean = pd.to_numeric(returns_pct, errors="coerce").dropna()
    if len(clean) < 2:
        return 0.0
    std = float(clean.std(ddof=1))
    if std == 0 or not np.isfinite(std):
        return 0.0
    return float((clean.mean() / std) * math.sqrt(len(clean)))


def _apply_variant(
    daily: pd.DataFrame,
    *,
    basket: str,
    entry_time: str,
    exit_time: str,
    threshold_pct: float,
    breadth_min: float | None,
) -> pd.DataFrame:
    feature_col = {
        "ew": "ew_gap_pct_vs_qqq_pm_pct",
        "mega": "mega_gap_pct_vs_qqq_pm_pct",
        "median": "median_gap_pct_vs_qqq_pm_pct",
    }[basket]
    source_col = {
        "ew": "ew_gap_pct",
        "mega": "mega_gap_pct",
        "median": "median_gap_pct",
    }[basket]
    frame = daily.copy()
    frame = frame[np.isfinite(frame[feature_col])].copy()
    if basket == "mega":
        frame = frame[frame["mega_count"] >= 6].copy()
    else:
        frame = frame[frame["all_count"] >= 70].copy()
    frame = frame[np.isfinite(frame[f"qqq_close_{exit_time}"])].copy()

    dislocation = frame[feature_col].astype(float)
    long_mask = dislocation >= float(threshold_pct)
    short_mask = dislocation <= -float(threshold_pct)
    if breadth_min is not None:
        long_mask &= frame["breadth_up"] >= float(breadth_min)
        short_mask &= frame["breadth_down"] >= float(breadth_min)
    signal = pd.Series(0, index=frame.index, dtype=int)
    signal.loc[long_mask] = 1
    signal.loc[short_mask] = -1
    frame = frame[signal != 0].copy()
    if frame.empty:
        frame["direction"] = pd.Series(dtype=int)
        frame["return_pct"] = pd.Series(dtype=float)
        return frame

    entry_col = "qqq_pre_929_close" if entry_time == "09:29" else "qqq_open_930"
    cost_pct = ENTRY_COST_PCT[entry_time]
    frame["direction"] = signal.loc[frame.index]
    frame["entry_price"] = frame[entry_col].astype(float)
    frame["exit_price"] = frame[f"qqq_close_{exit_time}"].astype(float)
    frame = frame[(frame["entry_price"] > 0) & np.isfinite(frame["exit_price"])].copy()
    frame["gross_return_pct"] = (frame["exit_price"] / frame["entry_price"] - 1.0) * 100.0 * frame["direction"]
    frame["return_pct"] = frame["gross_return_pct"] - cost_pct
    frame["basket_gap_pct"] = frame[source_col].astype(float)
    frame["qqq_signal_gap_pct"] = frame["qqq_pm_gap_pct"].astype(float)
    frame["dislocation_pct"] = frame[feature_col].astype(float)
    return frame


def _summarize_variant(name: str, trades: pd.DataFrame) -> StrategyResult:
    returns = pd.to_numeric(trades.get("return_pct"), errors="coerce").dropna()
    if returns.empty:
        return StrategyResult(
            variant=name,
            trades=0,
            long_trades=0,
            short_trades=0,
            win_rate_pct=0.0,
            mean_return_pct=0.0,
            median_return_pct=0.0,
            compounded_return_pct=0.0,
            max_drawdown_pct=0.0,
            sharpe_ratio=0.0,
        )
    return StrategyResult(
        variant=name,
        trades=int(len(trades)),
        long_trades=int((trades["direction"] > 0).sum()),
        short_trades=int((trades["direction"] < 0).sum()),
        win_rate_pct=float((returns > 0).mean() * 100.0),
        mean_return_pct=float(returns.mean()),
        median_return_pct=float(returns.median()),
        compounded_return_pct=float(_compound_return_pct(returns)),
        max_drawdown_pct=float(_max_drawdown_pct(returns)),
        sharpe_ratio=float(_sharpe_ratio(returns)),
    )


def _variant_grid() -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    for basket in ("ew", "mega", "median"):
        for entry_time in ("09:29", "09:30"):
            for exit_time in ("0945", "1000", "1100", "1600"):
                for threshold_pct in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
                    for breadth_min in (None, 0.55, 0.60):
                        variants.append(
                            {
                                "basket": basket,
                                "entry_time": entry_time,
                                "exit_time": exit_time,
                                "threshold_pct": threshold_pct,
                                "breadth_min": breadth_min,
                            }
                        )
    return variants


def _variant_name(spec: dict[str, Any]) -> str:
    breadth = "none" if spec["breadth_min"] is None else f"b{int(float(spec['breadth_min']) * 100)}"
    return (
        f"{spec['basket']}_entry{spec['entry_time'].replace(':','')}"
        f"_exit{spec['exit_time']}"
        f"_thr{int(round(float(spec['threshold_pct']) * 100))}"
        f"_{breadth}"
    )


def _run_variant_sweep(daily: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    trade_maps: dict[str, pd.DataFrame] = {}
    for spec in _variant_grid():
        name = _variant_name(spec)
        trades = _apply_variant(daily, **spec)
        trade_maps[name] = trades
        summary = _summarize_variant(name, trades)
        row = asdict(summary)
        row.update(spec)
        rows.append(row)
    variants = pd.DataFrame(rows).sort_values(
        ["compounded_return_pct", "sharpe_ratio", "trades"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return variants, trade_maps


def _choose_best_variant(variants: pd.DataFrame) -> str:
    eligible = variants[variants["trades"] >= 12].copy()
    if eligible.empty:
        eligible = variants[variants["trades"] >= 6].copy()
    if eligible.empty:
        eligible = variants.copy()
    eligible = eligible.sort_values(
        ["compounded_return_pct", "sharpe_ratio", "trades"],
        ascending=[False, False, False],
    )
    return str(eligible.iloc[0]["variant"])


def _walkforward(daily: pd.DataFrame, variants: pd.DataFrame, trade_maps: dict[str, pd.DataFrame]) -> dict[str, Any]:
    folds_payload: list[dict[str, Any]] = []
    test_returns: list[float] = []
    all_test_trades: list[pd.DataFrame] = []

    for idx in range(1, len(FOLDS)):
        train_start = pd.Timestamp(FOLDS[0][0])
        train_end = pd.Timestamp(FOLDS[idx - 1][1])
        test_start = pd.Timestamp(FOLDS[idx][0])
        test_end = pd.Timestamp(FOLDS[idx][1])

        train_rows: list[dict[str, Any]] = []
        for _, row in variants.iterrows():
            name = str(row["variant"])
            trades = trade_maps[name]
            train_trades = trades[(trades["date"] >= train_start) & (trades["date"] <= train_end)].copy()
            summary = _summarize_variant(name, train_trades)
            payload = asdict(summary)
            payload.update(
                {
                    "basket": row["basket"],
                    "entry_time": row["entry_time"],
                    "exit_time": row["exit_time"],
                    "threshold_pct": row["threshold_pct"],
                    "breadth_min": row["breadth_min"],
                }
            )
            train_rows.append(payload)
        train_df = pd.DataFrame(train_rows)
        chosen_name = _choose_best_variant(train_df)
        test_trades = trade_maps[chosen_name][
            (trade_maps[chosen_name]["date"] >= test_start) & (trade_maps[chosen_name]["date"] <= test_end)
        ].copy()
        test_summary = _summarize_variant(chosen_name, test_trades)
        test_returns.extend(pd.to_numeric(test_trades.get("return_pct"), errors="coerce").dropna().tolist())
        if not test_trades.empty:
            all_test_trades.append(test_trades.assign(selected_variant=chosen_name))
        folds_payload.append(
            {
                "train_start": str(train_start.date()),
                "train_end": str(train_end.date()),
                "test_start": str(test_start.date()),
                "test_end": str(test_end.date()),
                "selected_variant": chosen_name,
                "test_summary": asdict(test_summary),
            }
        )

    returns_series = pd.Series(test_returns, dtype=float)
    return {
        "folds": folds_payload,
        "summary": {
            "compounded_return_pct": round(_compound_return_pct(returns_series), 3),
            "max_drawdown_pct": round(_max_drawdown_pct(returns_series), 3),
            "sharpe_ratio": round(_sharpe_ratio(returns_series), 4),
            "trades": int(len(returns_series)),
            "win_rate_pct": round(float((returns_series > 0).mean() * 100.0) if not returns_series.empty else 0.0, 3),
        },
        "trades": pd.concat(all_test_trades, ignore_index=True) if all_test_trades else pd.DataFrame(),
    }


def main() -> None:
    daily = _build_daily_feature_table()
    variants, trade_maps = _run_variant_sweep(daily)
    walkforward = _walkforward(daily, variants, trade_maps)

    out_daily = ARTIFACT_DIR / "nasdaq_premarket_dislocation_daily.csv"
    out_variants = ARTIFACT_DIR / "nasdaq_premarket_dislocation_variants.csv"
    out_top = ARTIFACT_DIR / "nasdaq_premarket_dislocation_top_trades.csv"
    out_walk_trades = ARTIFACT_DIR / "nasdaq_premarket_dislocation_walkforward_trades.csv"
    out_summary = ARTIFACT_DIR / "nasdaq_premarket_dislocation_summary.json"

    daily.to_csv(out_daily, index=False)
    variants.to_csv(out_variants, index=False)

    best_variant = str(variants.iloc[0]["variant"])
    top_trades = trade_maps[best_variant].copy().sort_values("date")
    top_trades.to_csv(out_top, index=False)
    if isinstance(walkforward["trades"], pd.DataFrame) and not walkforward["trades"].empty:
        walkforward["trades"].to_csv(out_walk_trades, index=False)

    summary = {
        "window_start": str(WINDOW_START),
        "window_end": str(WINDOW_END),
        "nasdaq100_cached_symbols": len([s for s in NASDAQ100_TICKERS if (ALPACA_DIR / s / "1Min.csv").exists()]),
        "mega_cap_proxy": MEGA_CAP_PROXY,
        "days": int(len(daily)),
        "correlations": {
            "ew_gap_vs_qqq_pm_gap": round(float(daily["ew_gap_pct"].corr(daily["qqq_pm_gap_pct"])), 4),
            "mega_gap_vs_qqq_pm_gap": round(float(daily["mega_gap_pct"].corr(daily["qqq_pm_gap_pct"])), 4),
            "ew_dislocation_vs_qqq_open_to_1000": round(
                float(
                    daily["ew_gap_pct_vs_qqq_pm_pct"].corr(
                        (daily["qqq_close_1000"] / daily["qqq_open_930"] - 1.0) * 100.0
                    )
                ),
                4,
            ),
            "mega_dislocation_vs_qqq_open_to_1000": round(
                float(
                    daily["mega_gap_pct_vs_qqq_pm_pct"].corr(
                        (daily["qqq_close_1000"] / daily["qqq_open_930"] - 1.0) * 100.0
                    )
                ),
                4,
            ),
        },
        "best_variant": variants.iloc[0].to_dict(),
        "best_min12_variant": _choose_best_variant(variants[variants["trades"] >= 12] if (variants["trades"] >= 12).any() else variants),
        "walkforward": {
            "folds": walkforward["folds"],
            "summary": walkforward["summary"],
        },
        "artifacts": {
            "daily": str(out_daily),
            "variants": str(out_variants),
            "top_trades": str(out_top),
            "walkforward_trades": str(out_walk_trades),
        },
    }
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary["correlations"], indent=2))
    print()
    print("Best variant:")
    print(json.dumps(variants.iloc[0].to_dict(), indent=2, default=str))
    print()
    print("Walk-forward summary:")
    print(json.dumps(walkforward["summary"], indent=2))
    print()
    print(f"Wrote {out_summary}")


if __name__ == "__main__":
    main()
