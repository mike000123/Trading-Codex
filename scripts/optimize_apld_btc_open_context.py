from __future__ import annotations

import json
import argparse
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf


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

from config.settings import settings
from core.models import Signal, SignalAction
from data.ingestion import load_from_alpaca_history, prepare_strategy_data
from strategies.base import BaseStrategy


ARTIFACT_DIR = ROOT / "artifacts" / "optimization"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
_YF_TZ_CACHE = ROOT / "data_cache" / "_yfinance_tz_cache"
_YF_TZ_CACHE.mkdir(parents=True, exist_ok=True)
try:
    yf.set_tz_cache_location(str(_YF_TZ_CACHE))
except Exception:
    pass

MARKET_TZ = "America/New_York"
ALPACA_SAFE_BUFFER_DAYS = 3

class _APLDBTCProbeStrategy(BaseStrategy):
    strategy_id = "apld_btc_probe"
    name = "APLD BTC Context Probe"
    description = "Companion-data optimization scaffold for APLD vs BTC-USD."

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Signal:
        return Signal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            action=SignalAction.HOLD,
            confidence=0.0,
            suggested_tp=None,
            suggested_sl=None,
            metadata={"probe": True},
        )

    def companion_contexts(
        self,
        symbol: str,
        source: str | None = None,
        interval: str | None = None,
    ) -> list[str]:
        return ["equity_benchmark", "crypto_benchmark"]


@dataclass
class RuleResult:
    label: str
    side: str
    gap_metric: str
    btc_threshold: float
    gap_threshold: float
    entry_offset_min: int
    stretch_threshold: float
    exit_mode: str
    signals: int
    win_rate_pct: float
    mean_return_pct: float
    median_return_pct: float
    compounded_return_pct: float
    score: float


def _load_primary_from_alpaca(start: pd.Timestamp, end: pd.Timestamp, timeframe: str) -> pd.DataFrame:
    if not settings.alpaca.has_paper_credentials():
        raise SystemExit("Alpaca paper credentials are required for the Alpaca-backed APLD optimization pass.")
    primary = load_from_alpaca_history(
        "APLD",
        timeframe,
        start,
        end,
        settings.alpaca.paper_api_key,
        settings.alpaca.paper_secret_key,
        paper=True,
        use_cache=True,
    )
    primary["date"] = pd.to_datetime(primary["date"], errors="coerce")
    primary = primary.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date").reset_index(drop=True)
    return primary


def _prepare_cached_context(start: pd.Timestamp, end: pd.Timestamp, timeframe: str) -> pd.DataFrame:
    primary = _load_primary_from_alpaca(start, end, timeframe)
    strategy = _APLDBTCProbeStrategy(params={})
    return prepare_strategy_data(
        primary,
        strategy,
        primary_symbol="APLD",
        source="alpaca",
        interval=timeframe,
        start=primary["date"].min(),
        end=primary["date"].max(),
    )


def _safe_pct(current, base) -> float:
    try:
        current_f = float(current)
        base_f = float(base)
    except Exception:
        return np.nan
    if not np.isfinite(current_f) or not np.isfinite(base_f) or base_f == 0:
        return np.nan
    return (current_f / base_f - 1.0) * 100.0


def _close_at_offset(session: pd.DataFrame, offset_bars: int | None):
    if session.empty:
        return None
    if offset_bars is None:
        return session.iloc[-1]
    idx = min(int(offset_bars), len(session) - 1)
    return session.iloc[idx]


def _build_session_rows(prepared: pd.DataFrame) -> pd.DataFrame:
    work = prepared.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce", utc=True)
    work = work.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    work["date_ny"] = work["date"].dt.tz_convert(MARKET_TZ)
    work["minutes_ny"] = work["date_ny"].dt.hour * 60 + work["date_ny"].dt.minute
    work["session_date"] = work["date_ny"].dt.date
    work = work.loc[(work["minutes_ny"] >= 570) & (work["minutes_ny"] < 960)].reset_index(drop=True)

    sessions = [
        grp.reset_index(drop=True)
        for _, grp in work.groupby("session_date", sort=True)
        if len(grp) >= 12
    ]
    rows: list[dict[str, Any]] = []
    entry_offsets = {
        "entry_0m": 0,
        "entry_5m": 1,
        "entry_10m": 2,
        "entry_15m": 3,
        "entry_30m": 6,
        "entry_60m": 12,
    }
    exit_offsets = {
        "exit_30m": 6,
        "exit_60m": 12,
        "exit_120m": 24,
        "exit_close": None,
    }

    for prev_session, curr_session in zip(sessions, sessions[1:]):
        prev_last = prev_session.iloc[-1]
        open_row = curr_session.iloc[0]
        base = {
            "session_date": pd.Timestamp(open_row["date_ny"]).date().isoformat(),
            "open_bar_utc": pd.Timestamp(open_row["date"]).isoformat(),
            "btc_overnight_pct": _safe_pct(open_row.get("crypto_close"), prev_last.get("crypto_close")),
            "apld_gap_pct": _safe_pct(open_row["open"], prev_last["close"]),
            "qqq_gap_pct": _safe_pct(open_row.get("benchmark_open"), prev_last.get("benchmark_close")),
        }
        base["relative_gap_pct"] = (
            base["apld_gap_pct"] - base["qqq_gap_pct"]
            if np.isfinite(base["apld_gap_pct"]) and np.isfinite(base["qqq_gap_pct"])
            else np.nan
        )

        for entry_name, entry_idx in entry_offsets.items():
            if entry_idx >= len(curr_session):
                continue
            entry_row = curr_session.iloc[entry_idx]
            entry_price = float(entry_row["close"])
            base[entry_name] = entry_price
            base[f"{entry_name}_from_open_pct"] = _safe_pct(entry_price, open_row["open"])
            base[f"{entry_name}_benchmark_from_open_pct"] = _safe_pct(entry_row.get("benchmark_close"), open_row.get("benchmark_open"))
            for exit_name, exit_idx in exit_offsets.items():
                exit_row = _close_at_offset(curr_session, exit_idx)
                if exit_row is None:
                    continue
                apld_ret = _safe_pct(exit_row["close"], entry_price)
                qqq_ret = _safe_pct(exit_row.get("benchmark_close"), entry_row.get("benchmark_close"))
                base[f"{entry_name}_to_{exit_name}_long_pct"] = apld_ret
                base[f"{entry_name}_to_{exit_name}_short_pct"] = -apld_ret if np.isfinite(apld_ret) else np.nan
                base[f"{entry_name}_to_{exit_name}_relative_long_pct"] = (
                    apld_ret - qqq_ret if np.isfinite(apld_ret) and np.isfinite(qqq_ret) else np.nan
                )
                base[f"{entry_name}_to_{exit_name}_relative_short_pct"] = (
                    -(apld_ret - qqq_ret) if np.isfinite(apld_ret) and np.isfinite(qqq_ret) else np.nan
                )
        rows.append(base)
    out = pd.DataFrame(rows)
    if out.empty:
        raise SystemExit("No session rows built for APLD second-pass optimization.")
    return out


def _compounded_return_pct(returns_pct: pd.Series) -> float:
    clean = pd.to_numeric(returns_pct, errors="coerce").dropna()
    if clean.empty:
        return np.nan
    return float((np.prod(1.0 + clean / 100.0) - 1.0) * 100.0)


def _score_result(compounded_return_pct: float, mean_return_pct: float, win_rate_pct: float, signals: int) -> float:
    trade_bonus = min(float(signals), 20.0) * 0.2
    return float(compounded_return_pct) + float(mean_return_pct) * 2.0 + float(win_rate_pct) * 0.05 + trade_bonus


def _evaluate_rule(
    sessions: pd.DataFrame,
    *,
    side: str,
    gap_metric: str,
    btc_threshold: float,
    gap_threshold: float,
    entry_label: str,
    stretch_threshold: float,
    exit_label: str,
) -> RuleResult | None:
    signal_mask = pd.Series(False, index=sessions.index)
    if side == "short":
        signal_mask = (
            (pd.to_numeric(sessions["btc_overnight_pct"], errors="coerce") >= btc_threshold)
            & (pd.to_numeric(sessions[gap_metric], errors="coerce") >= gap_threshold)
            & (pd.to_numeric(sessions[f"{entry_label}_from_open_pct"], errors="coerce") >= stretch_threshold)
        )
        ret_col = f"{entry_label}_to_{exit_label}_short_pct"
    else:
        signal_mask = (
            (pd.to_numeric(sessions["btc_overnight_pct"], errors="coerce") <= -btc_threshold)
            & (pd.to_numeric(sessions[gap_metric], errors="coerce") <= -gap_threshold)
            & (pd.to_numeric(sessions[f"{entry_label}_from_open_pct"], errors="coerce") <= -stretch_threshold)
        )
        ret_col = f"{entry_label}_to_{exit_label}_long_pct"

    subset = sessions.loc[signal_mask].copy()
    if subset.empty:
        return None
    returns = pd.to_numeric(subset[ret_col], errors="coerce").dropna()
    if len(returns) < 3:
        return None
    compounded = _compounded_return_pct(returns)
    mean_ret = float(returns.mean())
    median_ret = float(returns.median())
    win_rate = float((returns > 0).mean() * 100.0)
    score = _score_result(compounded, mean_ret, win_rate, int(len(returns)))
    return RuleResult(
        label=f"{side}_{gap_metric}_{entry_label}_{exit_label}",
        side=side,
        gap_metric=gap_metric,
        btc_threshold=float(btc_threshold),
        gap_threshold=float(gap_threshold),
        entry_offset_min=int(entry_label.split("_")[1].replace("m", "")),
        stretch_threshold=float(stretch_threshold),
        exit_mode=exit_label.replace("exit_", ""),
        signals=int(len(returns)),
        win_rate_pct=win_rate,
        mean_return_pct=mean_ret,
        median_return_pct=median_ret,
        compounded_return_pct=float(compounded),
        score=float(score),
    )


def _grid_search(sessions: pd.DataFrame, side: str) -> list[RuleResult]:
    results: list[RuleResult] = []
    gap_metrics = ["apld_gap_pct", "relative_gap_pct"]
    btc_thresholds = [1.0, 1.5, 2.0, 3.0]
    gap_thresholds = [0.5, 1.0, 2.0, 3.0]
    entries = ["entry_0m", "entry_5m", "entry_10m", "entry_15m", "entry_30m", "entry_60m"]
    stretches = [0.0, 0.5, 1.0, 2.0, 3.0]
    exits = ["exit_30m", "exit_60m", "exit_120m", "exit_close"]
    for gap_metric in gap_metrics:
        for btc_th in btc_thresholds:
            for gap_th in gap_thresholds:
                for entry_label in entries:
                    for stretch_th in stretches:
                        for exit_label in exits:
                            res = _evaluate_rule(
                                sessions,
                                side=side,
                                gap_metric=gap_metric,
                                btc_threshold=btc_th,
                                gap_threshold=gap_th,
                                entry_label=entry_label,
                                stretch_threshold=stretch_th,
                                exit_label=exit_label,
                            )
                            if res is not None:
                                results.append(res)
    results.sort(key=lambda r: (r.score, r.compounded_return_pct, r.signals), reverse=True)
    return results


def _combined_result(sessions: pd.DataFrame, short_rule: RuleResult | None, long_rule: RuleResult | None) -> dict[str, Any]:
    if short_rule is None and long_rule is None:
        return {}

    def _select(rule: RuleResult) -> pd.DataFrame:
        side = rule.side
        entry_label = f"entry_{rule.entry_offset_min}m"
        exit_label = f"exit_{rule.exit_mode}"
        if side == "short":
            mask = (
                (pd.to_numeric(sessions["btc_overnight_pct"], errors="coerce") >= rule.btc_threshold)
                & (pd.to_numeric(sessions[rule.gap_metric], errors="coerce") >= rule.gap_threshold)
                & (pd.to_numeric(sessions[f"{entry_label}_from_open_pct"], errors="coerce") >= rule.stretch_threshold)
            )
            ret_col = f"{entry_label}_to_{exit_label}_short_pct"
        else:
            mask = (
                (pd.to_numeric(sessions["btc_overnight_pct"], errors="coerce") <= -rule.btc_threshold)
                & (pd.to_numeric(sessions[rule.gap_metric], errors="coerce") <= -rule.gap_threshold)
                & (pd.to_numeric(sessions[f"{entry_label}_from_open_pct"], errors="coerce") <= -rule.stretch_threshold)
            )
            ret_col = f"{entry_label}_to_{exit_label}_long_pct"
        out = sessions.loc[mask, ["session_date", ret_col]].copy()
        out = out.rename(columns={ret_col: "ret_pct"})
        out["side"] = side
        return out.dropna(subset=["ret_pct"])

    parts = []
    if short_rule is not None:
        parts.append(_select(short_rule))
    if long_rule is not None:
        parts.append(_select(long_rule))
    if not parts:
        return {}
    combined = pd.concat(parts, ignore_index=True).sort_values("session_date").drop_duplicates(subset=["session_date"], keep="first")
    if combined.empty:
        return {}
    returns = pd.to_numeric(combined["ret_pct"], errors="coerce").dropna()
    return {
        "signals": int(len(returns)),
        "win_rate_pct": float((returns > 0).mean() * 100.0) if len(returns) else np.nan,
        "mean_return_pct": float(returns.mean()) if len(returns) else np.nan,
        "median_return_pct": float(returns.median()) if len(returns) else np.nan,
        "compounded_return_pct": _compounded_return_pct(returns),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize APLD open-context rules using Alpaca + BTC companion data.")
    parser.add_argument("--start", default="2024-04-01", help="Start date for the validation window (default: 2024-04-01).")
    parser.add_argument("--end", default="", help="Optional end date; defaults to now.")
    parser.add_argument("--timeframe", default="5Min", help="Alpaca timeframe for the pass (default: 5Min).")
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end) if str(args.end).strip() else pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=ALPACA_SAFE_BUFFER_DAYS)
    timeframe = str(args.timeframe).strip()

    prepared = _prepare_cached_context(start, end, timeframe)
    sessions = _build_session_rows(prepared)
    short_results = _grid_search(sessions, "short")
    long_results = _grid_search(sessions, "long")

    best_short = short_results[0] if short_results else None
    best_long = long_results[0] if long_results else None
    combined = _combined_result(sessions, best_short, best_long)

    payload = {
        "symbol": "APLD",
        "source": "alpaca",
        "interval": timeframe,
        "window": {
            "start": pd.Timestamp(start).isoformat(),
            "end": pd.Timestamp(end).isoformat(),
        },
        "sessions": int(len(sessions)),
        "prepared_rows": int(len(prepared)),
        "best_short": asdict(best_short) if best_short else None,
        "best_long": asdict(best_long) if best_long else None,
        "combined_best_pair": combined,
        "notes": [
            "This second pass searches practical open-context event rules rather than an always-on strategy.",
            "Rules are built from the existing companion-data framework using QQQ and BTC-USD merged onto cached APLD bars.",
            "The strongest candidate should be treated as research until it survives a larger sample and a cleaner walk-forward split.",
        ],
    }

    (ARTIFACT_DIR / "apld_btc_open_context_results.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame([asdict(r) for r in short_results[:40]]).to_csv(
        ARTIFACT_DIR / "apld_btc_open_context_top_shorts.csv",
        index=False,
    )
    pd.DataFrame([asdict(r) for r in long_results[:40]]).to_csv(
        ARTIFACT_DIR / "apld_btc_open_context_top_longs.csv",
        index=False,
    )
    sessions.to_csv(
        ARTIFACT_DIR / "apld_btc_open_context_sessions.csv",
        index=False,
    )

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
